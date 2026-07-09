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

"""Offline acceptance / speedup benchmark for a trained EAGLE drafter on SGLang.

Training reports draft loss and top-1 token accuracy, but the metric that
actually matters for deployment is the *speculative-decoding acceptance length*:
how many draft tokens the target accepts per verification step. This script
drives a workload against a running SGLang server that hosts the drafter and
reports:

* ``accept_length`` -- SGLang's ``avg_spec_accept_length`` (mean tokens emitted
  per target verify step, including the one guaranteed bonus token). This is the
  "tokens accepted" headline number.
* ``acceptance_rate`` -- the fraction of the proposed draft chain that is
  accepted, derived as ``(accept_length - 1) / speculative_num_steps``.
* ``output_throughput_tok_s`` -- measured decode throughput (output tokens per
  wall-clock second).
* ``speedup`` -- optional: ``output_throughput`` divided by the same workload's
  throughput against a ``--baseline-server`` running *without* speculation.

The acceptance length is read exactly the way SGLang's own ``bench_serving``
reads it -- ``GET /server_info`` -> ``internal_states[0].avg_spec_accept_length``
(unwrapping a ``decode`` stage for PD-disaggregated servers). Because that value
is a server-cumulative running average, point this benchmark at a *freshly
started* server dedicated to the run for an accurate number.

Typical usage (after ``serve_sglang`` launches the drafter on port 30000):

    python -m nemo_automodel.components.speculative.bench_sglang \\
        --server http://localhost:30000 \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --input-data Aeala/ShareGPT_Vicuna_unfiltered \\
        --num-prompts 64 --concurrency 16 --max-new-tokens 256

Add ``--baseline-server http://localhost:30001`` (a second server started
without ``--speculative-algorithm``) to also report the end-to-end speedup.

SGLang is intentionally NOT a dependency of this script -- it talks to the
server over HTTP, so only ``aiohttp`` is required (already pulled in by the
project). The server itself must be running separately; see ``serve_sglang``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

# The chat-completions workload machinery (prompt loading, the timed HTTP pass,
# throughput / speedup math, shared CLI bounds) is engine-agnostic and shared
# with ``bench_vllm``; only the acceptance-statistics source below is SGLang's.
from nemo_automodel.components.speculative.bench_common import (
    WorkloadResult,
    _load_prompts,
    _normalize_server_url,
    _output_throughput,
    _run_workload,
    _speedup,
    _validate_workload_args,
)
from nemo_automodel.components.speculative.regenerate import (
    GenerationConfig,
    _import_aiohttp,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server-info parsing (mirrors sglang/bench_serving.py exactly)
# ---------------------------------------------------------------------------


def _unwrap_server_info(server_info_json: Any) -> dict[str, Any] | None:
    """Return the dict that holds ``internal_states``.

    PD-disaggregated servers nest the decode engine's state under a ``decode``
    list; ``bench_serving`` unwraps ``server_info_json["decode"][0]`` before
    reading ``internal_states``. Mirror that so both server topologies work.
    """
    if not isinstance(server_info_json, dict):
        return None
    if "decode" in server_info_json:
        decode = server_info_json["decode"]
        if isinstance(decode, list) and decode and isinstance(decode[0], dict):
            return decode[0]
        return None
    return server_info_json


def _internal_state(server_info_json: Any) -> dict[str, Any] | None:
    """Return ``internal_states[0]`` from a ``/server_info`` payload, or ``None``."""
    unwrapped = _unwrap_server_info(server_info_json)
    if unwrapped is None:
        return None
    states = unwrapped.get("internal_states")
    if isinstance(states, list) and states and isinstance(states[0], dict):
        return states[0]
    return None


def _extract_accept_length(server_info_json: Any) -> float | None:
    """Read ``avg_spec_accept_length`` the way SGLang's bench_serving does."""
    state = _internal_state(server_info_json)
    if state is None:
        return None
    value = state.get("avg_spec_accept_length")
    return float(value) if isinstance(value, (int, float)) else None


def _extract_num_steps(server_info_json: Any) -> int | None:
    """Read ``speculative_num_steps`` from ``/server_info`` if the server reports it."""
    state = _internal_state(server_info_json)
    if state is None:
        return None
    value = state.get("speculative_num_steps")
    return int(value) if isinstance(value, int) and value > 0 else None


def _acceptance_rate(accept_length: float | None, num_steps: int | None) -> float | None:
    """Fraction of the proposed draft chain accepted: ``(accept_length - 1) / num_steps``.

    ``accept_length`` counts the one guaranteed bonus token from the target, so
    ``accept_length - 1`` is the mean number of *draft* tokens accepted per step,
    and dividing by the proposed depth ``num_steps`` gives a [0, 1] rate. This is
    exact for a linear draft chain (topk=1) and approximate for tree drafting.
    Returns ``None`` when either input is unavailable.
    """
    if accept_length is None or num_steps is None or num_steps <= 0:
        return None
    return max(0.0, (accept_length - 1.0) / num_steps)


async def _fetch_server_info(server: str, *, timeout_s: float) -> dict[str, Any] | None:
    """GET ``<server>/server_info``; return the parsed JSON or ``None`` on failure."""
    aiohttp = _import_aiohttp()
    url = _normalize_server_url(server) + "/server_info"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                if resp.status != 200:
                    logger.warning("GET %s returned HTTP %d; accept length unavailable.", url, resp.status)
                    return None
                return await resp.json()
    except Exception as exc:  # noqa: BLE001 -- server-info is best-effort
        logger.warning("Failed to query %s (%s); accept length unavailable.", url, exc)
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _summarize(
    *,
    gen_cfg: GenerationConfig,
    spec_result: WorkloadResult,
    server_info: dict[str, Any] | None,
    num_steps_arg: int | None,
    baseline_result: WorkloadResult | None,
) -> dict[str, Any]:
    """Assemble the metrics dict reported to stdout / ``--output-json``."""
    accept_length = _extract_accept_length(server_info)
    num_steps = _extract_num_steps(server_info)
    if num_steps is None:
        num_steps = num_steps_arg
    spec_throughput = _output_throughput(spec_result)

    summary: dict[str, Any] = {
        "model": gen_cfg.model,
        "num_prompts": spec_result.completed + spec_result.failed,
        "completed": spec_result.completed,
        "failed": spec_result.failed,
        "output_tokens": spec_result.output_tokens,
        "wall_clock_s": round(spec_result.wall_clock_s, 4),
        "output_throughput_tok_s": round(spec_throughput, 4) if spec_throughput is not None else None,
        "accept_length": accept_length,
        "speculative_num_steps": num_steps,
        "acceptance_rate": _acceptance_rate(accept_length, num_steps),
    }
    if baseline_result is not None:
        baseline_throughput = _output_throughput(baseline_result)
        summary["baseline_throughput_tok_s"] = (
            round(baseline_throughput, 4) if baseline_throughput is not None else None
        )
        summary["speedup"] = _speedup(spec_throughput, baseline_throughput)
    return summary


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid CLI values before any network work starts."""
    _validate_workload_args(args)
    if args.num_steps is not None and args.num_steps < 1:
        raise ValueError(f"--num-steps must be >= 1 when set, got {args.num_steps}")


async def _run(args: argparse.Namespace) -> int:
    """Async driver: load prompts, run the workload(s), report metrics. Returns an exit code."""
    _validate_args(args)
    gen_cfg = GenerationConfig(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    prompts = _load_prompts(args)
    if not prompts:
        logger.error("No usable prompts loaded from %s; nothing to benchmark.", args.input_data)
        return 1
    logger.info("Benchmarking %d prompts against %s", len(prompts), args.server)

    spec_result = await _run_workload(
        args.server,
        prompts,
        gen_cfg,
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )
    server_info = await _fetch_server_info(args.server, timeout_s=args.timeout_s)

    baseline_result = None
    if args.baseline_server:
        logger.info("Running baseline workload against %s", args.baseline_server)
        baseline_result = await _run_workload(
            args.baseline_server,
            prompts,
            gen_cfg,
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
        )

    summary = _summarize(
        gen_cfg=gen_cfg,
        spec_result=spec_result,
        server_info=server_info,
        num_steps_arg=args.num_steps,
        baseline_result=baseline_result,
    )
    if summary["accept_length"] is None:
        logger.warning(
            "Server did not report avg_spec_accept_length. Is speculative decoding enabled, "
            "and is %s a fresh SGLang server? Throughput is still reported.",
            args.server,
        )

    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(rendered + "\n")
        logger.info("Wrote metrics to %s", args.output_json)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a trained EAGLE drafter served by SGLang: acceptance length, rate, and speedup.",
    )
    parser.add_argument(
        "--server",
        required=True,
        help="Root URL of the running SGLang server hosting the drafter, e.g. http://localhost:30000.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Served model name to send in the chat payload (the SGLang --served-model-name).",
    )
    parser.add_argument(
        "--input-data",
        required=True,
        help="HF dataset id or local path (parquet/dir/json/jsonl) with a chat messages column.",
    )
    parser.add_argument(
        "--baseline-server",
        default=None,
        help="Optional second SGLang server running WITHOUT speculation; enables the speedup metric.",
    )
    parser.add_argument("--num-prompts", type=int, default=64, help="Number of prompts to send.")
    parser.add_argument("--concurrency", type=int, default=16, help="Maximum in-flight requests.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="max_tokens per request.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Default 0.0 = greedy.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="speculative_num_steps fallback for acceptance_rate when /server_info omits it.",
    )
    parser.add_argument("--messages-column", default="messages", help="Column holding the OpenAI messages list.")
    parser.add_argument("--split", default="train", help="HF dataset split (supports slice syntax).")
    parser.add_argument("--dataset-name", default=None, help="HF dataset configuration name, if any.")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Optional shuffle seed before slicing.")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries on 5xx / 429 / transport errors.")
    parser.add_argument("--output-json", default=None, help="Optional path to also write the metrics JSON to.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Parses ``argv`` and returns the process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
