# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Engine-agnostic workload machinery shared by the drafter benchmarks.

``bench_sglang`` and ``bench_vllm`` drive the same OpenAI-style
chat-completions workload and report the same throughput/speedup numbers; they
differ only in how each engine exposes its acceptance statistics (SGLang's
``/server_info`` vs vLLM's Prometheus ``/metrics``). This module holds the
shared layer: the prompt loader, the timed HTTP workload runner, and the
throughput / speedup / CLI-bounds helpers.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

# ``GenerationConfig`` (model + sampling params for a chat completion) is shared
# with the regeneration tool -- same four fields, same meaning -- so reuse it
# rather than redefining an identical dataclass here.
from nemo_automodel.components.speculative.regenerate import (
    GenerationConfig,
    _extract_prompt_messages,
    _import_aiohttp,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkloadResult:
    """Aggregate timing for one workload pass against a server."""

    wall_clock_s: float
    output_tokens: int
    completed: int
    failed: int


def _speedup(spec_throughput: float | None, baseline_throughput: float | None) -> float | None:
    """Return ``spec / baseline`` output throughput, or ``None`` if not computable."""
    if not spec_throughput or not baseline_throughput or baseline_throughput <= 0:
        return None
    return spec_throughput / baseline_throughput


def _output_throughput(result: WorkloadResult) -> float | None:
    """Output tokens per wall-clock second, or ``None`` if nothing was timed."""
    if result.wall_clock_s <= 0 or result.output_tokens <= 0:
        return None
    return result.output_tokens / result.wall_clock_s


def _normalize_server_url(url: str) -> str:
    """Return the server root URL without a trailing slash or ``/v1`` suffix.

    Chat completions live at ``<root>/v1/chat/completions`` and the engine's
    stats endpoint at its own root-relative path; accept either
    ``http://host:port`` or the OpenAI-style ``http://host:port/v1`` so the
    flag is forgiving.
    """
    root = url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


async def _chat_completion(
    session,
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    max_retries: int,
) -> int:
    """POST one chat completion and return its ``completion_tokens`` (0 on no usage)."""
    aiohttp = _import_aiohttp()
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                if resp.status >= 500 or resp.status == 429:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status} from {url}: {text[:200]}")
                resp.raise_for_status()
                data = await resp.json()
                usage = data.get("usage") if isinstance(data, dict) else None
                if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                    return usage["completion_tokens"]
                return 0
        except aiohttp.ClientResponseError:
            # 5xx and 429 are turned into RuntimeError above, so the only status
            # that reaches raise_for_status() is a non-429 4xx (400/401/403/404,
            # ...) -- a client error that will not succeed on retry. Surface it
            # immediately instead of burning the whole retry budget on it.
            raise
        except Exception as exc:  # noqa: BLE001 -- retry transport / 5xx / 429 errors
            last_err = exc
            if attempt == max_retries:
                raise
            await asyncio.sleep(min(2.0**attempt, 30.0))
    raise RuntimeError(f"Unreachable: retries exhausted without raising. Last error: {last_err}")


async def _run_workload(
    server: str,
    prompts: list[list[dict[str, Any]]],
    gen_cfg: GenerationConfig,
    *,
    concurrency: int,
    timeout_s: float,
    max_retries: int,
) -> WorkloadResult:
    """Send every prompt through ``<server>/v1/chat/completions`` and time the pass."""
    aiohttp = _import_aiohttp()
    url = _normalize_server_url(server) + "/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)

    async def _worker(slot: int, prompt: list[dict[str, Any]]) -> int | None:
        """Return the request's completion-token count, or ``None`` on failure."""
        payload = {
            "model": gen_cfg.model,
            "messages": prompt,
            "max_tokens": gen_cfg.max_new_tokens,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
        }
        async with semaphore:
            try:
                return await _chat_completion(session, url, payload, timeout_s=timeout_s, max_retries=max_retries)
            except Exception as exc:  # noqa: BLE001 -- one bad request must not abort the run
                logger.warning("Request %d failed: %s", slot, exc)
                return None

    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        token_counts = await asyncio.gather(*[_worker(i, prompt) for i, prompt in enumerate(prompts)])
        wall_clock_s = time.perf_counter() - start

    completed = [c for c in token_counts if c is not None]
    return WorkloadResult(
        wall_clock_s=wall_clock_s,
        output_tokens=sum(completed),
        completed=len(completed),
        failed=len(token_counts) - len(completed),
    )


def _load_prompts(args: argparse.Namespace) -> list[list[dict[str, Any]]]:
    """Load up to ``--num-prompts`` chat prompts (trailing assistant turn dropped)."""
    from nemo_automodel.components.datasets.llm.chat_dataset import _load_openai_messages

    dataset = _load_openai_messages(
        args.input_data,
        split=args.split,
        name=args.dataset_name,
        shuffle_seed=args.shuffle_seed,
    )
    prompts: list[list[dict[str, Any]]] = []
    for row in dataset:
        prompt = _extract_prompt_messages(row[args.messages_column])
        if prompt is not None:
            prompts.append(prompt)
        if len(prompts) >= args.num_prompts:
            break
    return prompts


def _validate_workload_args(args: argparse.Namespace) -> None:
    """Reject invalid values of the CLI flags every benchmark shares."""
    if args.num_prompts < 1:
        raise ValueError(f"--num-prompts must be >= 1, got {args.num_prompts}")
    if args.concurrency < 1:
        raise ValueError(f"--concurrency must be >= 1, got {args.concurrency}")
    if args.max_new_tokens < 1:
        raise ValueError(f"--max-new-tokens must be >= 1, got {args.max_new_tokens}")
    if args.max_retries < 0:
        raise ValueError(f"--max-retries must be >= 0, got {args.max_retries}")
    if args.timeout_s <= 0:
        raise ValueError(f"--timeout-s must be > 0, got {args.timeout_s}")
