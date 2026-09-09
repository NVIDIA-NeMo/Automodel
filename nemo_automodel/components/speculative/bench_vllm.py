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

"""Offline acceptance / speedup benchmark for a trained drafter served by vLLM.

This is the vLLM companion to ``bench_sglang``: the same chat-completions
workload machinery (which it imports), with the acceptance statistics read from
vLLM's Prometheus ``/metrics`` endpoint instead of SGLang's ``/server_info``.
It covers every drafter ``serve_vllm`` can launch -- EAGLE-3, P-EAGLE, and the
DFlash family -- and reports:

* ``accept_length`` -- mean tokens emitted per target verification step,
  including the one guaranteed bonus token:
  ``1 + num_accepted_tokens / num_drafts``.
* ``acceptance_rate`` -- fraction of proposed draft tokens accepted:
  ``num_accepted_tokens / num_draft_tokens``.
* ``output_throughput_tok_s`` -- measured decode throughput (output tokens per
  wall-clock second).
* ``speedup`` -- optional: throughput divided by the same workload's throughput
  against a ``--baseline-server`` running *without* speculation.

The spec-decode counters (``vllm:spec_decode_num_drafts_total``,
``vllm:spec_decode_num_draft_tokens_total``,
``vllm:spec_decode_num_accepted_tokens_total``) are snapshotted before and after
the workload and differenced, so -- unlike SGLang's server-cumulative
``avg_spec_accept_length`` -- the numbers cover exactly this benchmark's
requests even on a server that has already handled other traffic.

Typical usage (after ``serve_vllm`` launches the drafter on port 8000):

    python -m nemo_automodel.components.speculative.bench_vllm \\
        --server http://localhost:8000 \\
        --model Qwen/Qwen3-8B \\
        --input-data Aeala/ShareGPT_Vicuna_unfiltered \\
        --num-prompts 64 --concurrency 16 --max-new-tokens 256

Add ``--baseline-server http://localhost:8001`` (a second server started
without ``--speculative-config``) to also report the end-to-end speedup.

vLLM is intentionally NOT a dependency of this script -- it talks to the server
over HTTP, so only ``aiohttp`` is required (already pulled in by the project).
The server itself must be running separately; see ``serve_vllm``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Any

from nemo_automodel.components.speculative.bench_common import (
    WorkloadResult,
    _load_prompts,
    _normalize_server_url,
    _output_throughput,
    _report_summary,
    _run_workload,
    _speedup,
    _validate_workload_args,
)
from nemo_automodel.components.speculative.regenerate import (
    GenerationConfig,
    _import_aiohttp,
)

logger = logging.getLogger(__name__)

# Prometheus sample names of vLLM's spec-decode counters (the ``_total`` suffix
# is appended by the Prometheus client to every Counter).
_METRIC_NUM_DRAFTS = "vllm:spec_decode_num_drafts_total"
_METRIC_NUM_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
_METRIC_NUM_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens_total"
_SPEC_METRIC_NAMES = (_METRIC_NUM_DRAFTS, _METRIC_NUM_DRAFT_TOKENS, _METRIC_NUM_ACCEPTED_TOKENS)


@dataclass(frozen=True)
class SpecMetrics:
    """One snapshot of vLLM's cumulative spec-decode counters."""

    num_drafts: float
    num_draft_tokens: float
    num_accepted_tokens: float

    def delta(self, before: SpecMetrics) -> SpecMetrics:
        """Return the counter increase since ``before`` (clamped at zero per counter)."""
        return SpecMetrics(
            num_drafts=max(0.0, self.num_drafts - before.num_drafts),
            num_draft_tokens=max(0.0, self.num_draft_tokens - before.num_draft_tokens),
            num_accepted_tokens=max(0.0, self.num_accepted_tokens - before.num_accepted_tokens),
        )


def _parse_spec_metrics(metrics_text: str) -> SpecMetrics | None:
    """Parse the spec-decode counters out of a Prometheus exposition payload.

    Values are summed across label sets (multiple engines / model labels), and
    ``None`` is returned when the drafts counter is absent -- i.e. the server is
    not running with speculative decoding.
    """
    totals = dict.fromkeys(_SPEC_METRIC_NAMES, 0.0)
    drafts_seen = False
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in totals:
            continue
        try:
            totals[name] += float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        drafts_seen = drafts_seen or name == _METRIC_NUM_DRAFTS
    if not drafts_seen:
        return None
    return SpecMetrics(
        num_drafts=totals[_METRIC_NUM_DRAFTS],
        num_draft_tokens=totals[_METRIC_NUM_DRAFT_TOKENS],
        num_accepted_tokens=totals[_METRIC_NUM_ACCEPTED_TOKENS],
    )


async def _fetch_spec_metrics(server: str, *, timeout_s: float) -> SpecMetrics | None:
    """GET ``<server>/metrics`` and parse the spec-decode counters; ``None`` on failure."""
    aiohttp = _import_aiohttp()
    url = _normalize_server_url(server) + "/metrics"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                if resp.status != 200:
                    logger.warning("GET %s returned HTTP %d; acceptance metrics unavailable.", url, resp.status)
                    return None
                return _parse_spec_metrics(await resp.text())
    except Exception as exc:  # noqa: BLE001 -- metrics are best-effort
        logger.warning("Failed to query %s (%s); acceptance metrics unavailable.", url, exc)
        return None


def _accept_length(delta: SpecMetrics | None) -> float | None:
    """Mean tokens per verify step incl. the bonus token: ``1 + accepted / drafts``."""
    if delta is None or delta.num_drafts <= 0:
        return None
    return 1.0 + delta.num_accepted_tokens / delta.num_drafts


def _acceptance_rate(delta: SpecMetrics | None) -> float | None:
    """Fraction of proposed draft tokens accepted: ``accepted / draft_tokens``."""
    if delta is None or delta.num_draft_tokens <= 0:
        return None
    return delta.num_accepted_tokens / delta.num_draft_tokens


def _summarize(
    *,
    gen_cfg: GenerationConfig,
    spec_result: WorkloadResult,
    metrics_delta: SpecMetrics | None,
    baseline_result: WorkloadResult | None,
) -> dict[str, Any]:
    """Assemble the metrics dict reported to stdout / ``--output-json``."""
    spec_throughput = _output_throughput(spec_result)
    accept_length = _accept_length(metrics_delta)

    summary: dict[str, Any] = {
        "model": gen_cfg.model,
        "num_prompts": spec_result.completed + spec_result.failed,
        "completed": spec_result.completed,
        "failed": spec_result.failed,
        "output_tokens": spec_result.output_tokens,
        "wall_clock_s": round(spec_result.wall_clock_s, 4),
        "output_throughput_tok_s": round(spec_throughput, 4) if spec_throughput is not None else None,
        "accept_length": round(accept_length, 4) if accept_length is not None else None,
        "acceptance_rate": _acceptance_rate(metrics_delta),
        "num_drafts": int(metrics_delta.num_drafts) if metrics_delta is not None else None,
        "num_draft_tokens": int(metrics_delta.num_draft_tokens) if metrics_delta is not None else None,
        "num_accepted_tokens": int(metrics_delta.num_accepted_tokens) if metrics_delta is not None else None,
    }
    if baseline_result is not None:
        baseline_throughput = _output_throughput(baseline_result)
        summary["baseline_throughput_tok_s"] = (
            round(baseline_throughput, 4) if baseline_throughput is not None else None
        )
        summary["speedup"] = _speedup(spec_throughput, baseline_throughput)
    return summary


async def _run_summary(args: argparse.Namespace) -> dict[str, Any] | None:
    """Validate args, run the workload(s), and return the metrics dict.

    Returns ``None`` when no usable prompts were loaded (the caller's cue to
    report a failure without raising -- a bad ``--num-prompts``/etc. value is a
    real programming error and still raises via ``_validate_workload_args``).
    Split out of ``_run`` so ``bench_sweep`` can drive one dataset at a time
    without the printing / ``--output-json`` side effects below.
    """
    _validate_workload_args(args)
    gen_cfg = GenerationConfig(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    prompts = _load_prompts(args)
    if not prompts:
        logger.error("No usable prompts loaded from %s; nothing to benchmark.", args.input_data)
        return None
    logger.info("Benchmarking %d prompts against %s", len(prompts), args.server)

    metrics_before = await _fetch_spec_metrics(args.server, timeout_s=args.timeout_s)
    spec_result = await _run_workload(
        args.server,
        prompts,
        gen_cfg,
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )
    metrics_after = await _fetch_spec_metrics(args.server, timeout_s=args.timeout_s)
    metrics_delta = None
    if metrics_after is not None:
        # A missing "before" snapshot (e.g. the server came up mid-setup) falls
        # back to the cumulative counters, which is exact for a fresh server.
        metrics_delta = metrics_after.delta(metrics_before) if metrics_before is not None else metrics_after

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
        metrics_delta=metrics_delta,
        baseline_result=baseline_result,
    )
    if summary["accept_length"] is None:
        logger.warning(
            "Server did not report spec-decode counters on /metrics. Is speculative decoding "
            "enabled on %s? Throughput is still reported.",
            args.server,
        )
    return summary


async def _run(args: argparse.Namespace) -> int:
    """Async driver: compute the benchmark summary and report it. Returns an exit code."""
    summary = await _run_summary(args)
    return _report_summary(summary, args.output_json)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a trained drafter served by vLLM (EAGLE-3 / P-EAGLE / DFlash): "
            "acceptance length, rate, and speedup."
        ),
    )
    parser.add_argument(
        "--server",
        required=True,
        help="Root URL of the running vLLM server hosting the drafter, e.g. http://localhost:8000.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Served model name to send in the chat payload (the vLLM --served-model-name / --model).",
    )
    parser.add_argument(
        "--input-data",
        required=True,
        help="HF dataset id or local path (parquet/dir/json/jsonl) with a chat messages column.",
    )
    parser.add_argument(
        "--baseline-server",
        default=None,
        help="Optional second vLLM server running WITHOUT speculation; enables the speedup metric.",
    )
    parser.add_argument("--num-prompts", type=int, default=64, help="Number of prompts to send.")
    parser.add_argument("--concurrency", type=int, default=16, help="Maximum in-flight requests.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="max_tokens per request.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Default 0.0 = greedy.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--messages-column", default="messages", help="Column holding the OpenAI messages list.")
    parser.add_argument(
        "--prompt-column",
        default=None,
        help="Column holding a raw prompt string (or list of turns, first used) instead of --messages-column, "
        "for datasets that are not already chat-messages-shaped.",
    )
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
