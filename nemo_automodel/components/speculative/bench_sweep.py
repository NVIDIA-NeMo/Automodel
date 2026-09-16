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

"""Sweep a trained drafter's acceptance-length benchmark across multiple datasets.

``bench_sglang`` / ``bench_vllm`` measure ONE workload per invocation. The same
draft can behave very differently across task types -- conversational data
(ShareGPT / MT-Bench) tends to have far higher acceptance than math (GSM8K) or
code (HumanEval), whose token distributions diverge sharply from the training
mix. This script drives the same server through several named datasets in one
pass and reports a per-dataset table plus a completed-weighted aggregate,
instead of the user invoking ``bench_sglang``/``bench_vllm`` once per dataset
and collating the output by hand.

Default dataset suite -- the four benchmarks the EAGLE / EAGLE-2 papers report
acceptance / speedup numbers on: MT-Bench (first turn), HumanEval (code),
GSM8K (math), and Alpaca (single-turn instruction-following). None of these
ship a chat-messages column the way ``bench_sglang``/``bench_vllm``'s
``--messages-column`` expects, so each reads a raw text field instead
(``bench_common``'s ``--prompt-column`` path: the field is wrapped into a
fresh single-turn user message; a list value, e.g. MT-Bench's two-turn
``prompt`` column, uses its first entry).

Override the default suite with ``--datasets-config <path.yaml>``: a YAML list
of entries, each ``{name, input_data, split, dataset_name, messages_column |
prompt_column, max_new_tokens}`` (``dataset_name``/``max_new_tokens`` optional;
exactly one of ``messages_column``/``prompt_column`` required). ``--datasets``
further narrows either suite to a name subset.

Typical usage (after ``serve_sglang`` launches the drafter on port 30000)::

    python -m nemo_automodel.components.speculative.bench_sweep \\
        --engine sglang --server http://localhost:30000 \\
        --model meta-llama/Llama-3.1-8B-Instruct

Add ``--baseline-server`` for the speedup column, ``--engine vllm`` for a vLLM
server, and ``--datasets-config`` to point at a custom dataset list. One
dataset failing to load or benchmark (bad HF id, unreachable server for that
request, ...) does not abort the sweep -- it is reported as an error row and
excluded from the aggregate.

CAVEAT (``--engine sglang`` with more than one dataset): SGLang's
``avg_spec_accept_length`` is a server-cumulative running average with no
reset/delta API (see ``bench_sglang``'s module docstring), so sweeping N>1
datasets against ONE live SGLang server means every dataset after the first
reports a blend with prior datasets' traffic, not an independent number. A
warning is logged when this applies. Restart the server between datasets for
independent numbers, or use ``--engine vllm``, which snapshots and diffs its
Prometheus counters per dataset and has no such caveat.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any

from nemo_automodel.components.speculative import bench_sglang, bench_vllm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset to sweep: how to load it and where its prompt text lives.

    Exactly one of ``messages_column`` (an existing OpenAI-messages list) or
    ``prompt_column`` (a raw text field, wrapped into a single-turn user
    message) must be set -- see ``bench_common._load_prompts``.

    ``prompt_context_column`` is an optional second raw-text field appended to
    ``prompt_column`` (separated by a blank line) when it is non-empty, for
    datasets whose task context lives in a separate column (e.g. Alpaca's
    ``input``). It is only valid alongside ``prompt_column``.
    """

    name: str
    input_data: str
    split: str = "train"
    dataset_name: str | None = None
    messages_column: str | None = None
    prompt_column: str | None = None
    prompt_context_column: str | None = None
    max_new_tokens: int | None = None

    def __post_init__(self):
        if bool(self.messages_column) == bool(self.prompt_column):
            raise ValueError(
                f"dataset {self.name!r}: exactly one of messages_column / prompt_column must be set "
                f"(got messages_column={self.messages_column!r}, prompt_column={self.prompt_column!r})"
            )
        if self.prompt_context_column and not self.prompt_column:
            raise ValueError(
                f"dataset {self.name!r}: prompt_context_column={self.prompt_context_column!r} requires "
                "prompt_column to be set (it is appended to the prompt_column text)."
            )


# The classic EAGLE / EAGLE-2 paper suite: chat (MT-Bench, first turn), code
# (HumanEval), math (GSM8K), and instruction-following (Alpaca).
DEFAULT_DATASET_PRESETS: tuple[DatasetSpec, ...] = (
    # HuggingFaceH4/mt_bench_prompts stores the two-turn sequence under ``prompt``
    # (a list); the first turn is used (see bench_common._extract_prompt_text).
    DatasetSpec(name="mt_bench", input_data="HuggingFaceH4/mt_bench_prompts", split="train", prompt_column="prompt"),
    DatasetSpec(
        name="humaneval",
        input_data="openai/openai_humaneval",
        split="test",
        prompt_column="prompt",
        max_new_tokens=512,
    ),
    DatasetSpec(name="gsm8k", input_data="openai/gsm8k", dataset_name="main", split="test", prompt_column="question"),
    # Alpaca's task context lives in a separate ``input`` field (non-empty for
    # ~40% of rows); append it so those prompts are not truncated to the bare
    # instruction. The reference ``output`` is never included.
    DatasetSpec(
        name="alpaca",
        input_data="tatsu-lab/alpaca",
        split="train",
        prompt_column="instruction",
        prompt_context_column="input",
    ),
)

_ENGINE_MODULES = {"sglang": bench_sglang, "vllm": bench_vllm}

# Validates the workload flags shared by every dataset (num_prompts, concurrency,
# max_new_tokens, max_retries, timeout_s, and sglang's num_steps). A bad value here
# is a single misconfiguration, not a per-dataset failure -- see _run's upfront call.
_ENGINE_VALIDATORS = {"sglang": bench_sglang._validate_args, "vllm": bench_vllm._validate_workload_args}


def _load_dataset_specs(config_path: str | None) -> list[DatasetSpec]:
    """Return the sweep's dataset list: the built-in suite, or a ``--datasets-config`` YAML override.

    The config's top level must be a mapping with a ``datasets:`` list (not a
    bare top-level list) so the file passes the repo's example-YAML linter,
    which requires every ``examples/`` YAML to parse to a mapping.
    """
    if config_path is None:
        return list(DEFAULT_DATASET_PRESETS)
    import yaml

    with open(config_path) as f:
        document = yaml.safe_load(f)
    entries = document.get("datasets") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{config_path} must contain a non-empty top-level `datasets:` list.")
    specs: list[DatasetSpec] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{config_path}: each dataset entry must be a mapping, got {entry!r}.")
        try:
            spec = DatasetSpec(**entry)
        except TypeError as exc:
            raise ValueError(f"{config_path}: invalid dataset entry {entry!r}: {exc}") from exc
        if spec.name in seen:
            raise ValueError(f"{config_path}: duplicate dataset name {spec.name!r}.")
        seen.add(spec.name)
        specs.append(spec)
    return specs


def _select_datasets(specs: list[DatasetSpec], names: list[str] | None) -> list[DatasetSpec]:
    """Narrow ``specs`` to ``--datasets``, preserving the sweep's declared order."""
    if names is None:
        return specs
    available = {s.name for s in specs}
    missing = set(names) - available
    if missing:
        raise ValueError(f"--datasets names not in the sweep list: {sorted(missing)} (available: {sorted(available)})")
    wanted = set(names)
    return [s for s in specs if s.name in wanted]


def _dataset_args(base_args: argparse.Namespace, spec: DatasetSpec) -> argparse.Namespace:
    """Clone the shared server/model/workload args, overridden with one dataset's spec."""
    args = copy.copy(base_args)
    args.input_data = spec.input_data
    args.split = spec.split
    args.dataset_name = spec.dataset_name
    args.messages_column = spec.messages_column
    args.prompt_column = spec.prompt_column
    args.prompt_context_column = spec.prompt_context_column
    if spec.max_new_tokens is not None:
        args.max_new_tokens = spec.max_new_tokens
    return args


def _warn_if_sglang_stats_will_be_cumulative(args: argparse.Namespace, num_datasets: int) -> None:
    """SGLang's ``avg_spec_accept_length`` is a server-cumulative running average (see bench_sglang.py's
    module docstring): it has no reset/delta API, so sweeping N>1 datasets against ONE live SGLang
    server means every dataset after the first reports a blend with prior datasets' traffic, not an
    independent number. vLLM's Prometheus counters are snapshotted before/after each dataset instead
    (see bench_vllm._run_summary), so this caveat is sglang-only.
    """
    if args.engine == "sglang" and num_datasets > 1:
        logger.warning(
            "Sweeping %d datasets against one SGLang server: accept_length/acceptance_rate for every "
            "dataset after the first will be contaminated by earlier datasets' traffic (SGLang's "
            "avg_spec_accept_length is a server-cumulative running average with no reset). Restart the "
            "server between datasets for independent per-dataset numbers, or use --engine vllm, which "
            "diffs its counters per dataset.",
            num_datasets,
        )


async def _run_sweep(args: argparse.Namespace, specs: list[DatasetSpec]) -> list[dict[str, Any]]:
    """Run one benchmark per dataset spec; one dataset's failure does not abort the sweep."""
    engine = _ENGINE_MODULES[args.engine]
    _warn_if_sglang_stats_will_be_cumulative(args, len(specs))
    results: list[dict[str, Any]] = []
    for spec in specs:
        dataset_args = _dataset_args(args, spec)
        logger.info("[%s] benchmarking %s (engine=%s)", spec.name, spec.input_data, args.engine)
        try:
            summary = await engine._run_summary(dataset_args)
        except Exception as exc:  # noqa: BLE001 -- one bad dataset must not abort the sweep
            logger.error("[%s] failed: %s", spec.name, exc)
            results.append({"dataset": spec.name, "error": str(exc)})
            continue
        if summary is None:
            results.append({"dataset": spec.name, "error": "no usable prompts loaded"})
            continue
        # _run_workload swallows per-request HTTP errors, so an engine whose every
        # request failed still returns a summary with completed=0 rather than raising.
        # Treat that as an error row so it is excluded from num_datasets_ok and the
        # sweep's exit status reflects that no benchmark actually ran.
        if summary.get("completed", 0) < 1:
            results.append({"dataset": spec.name, "error": "all requests failed (0 completed)"})
            continue
        results.append({"dataset": spec.name, **summary})
    return results


def _weighted_mean(ok_results: list[dict[str, Any]], key: str) -> float | None:
    """Completed-weighted mean of ``key`` over already-filtered (error-free) results."""
    pairs = [(r[key], r.get("completed", 0)) for r in ok_results if r.get(key) is not None]
    weight = sum(w for _, w in pairs)
    if not pairs or weight <= 0:
        return None
    return sum(v * w for v, w in pairs) / weight


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Completed-weighted rollup of accept_length / acceptance_rate / throughput / speedup."""
    ok = [r for r in results if "error" not in r]
    return {
        "num_datasets": len(results),
        "num_datasets_ok": len(ok),
        "total_completed": sum(r.get("completed", 0) for r in ok),
        "accept_length": _weighted_mean(ok, "accept_length"),
        "acceptance_rate": _weighted_mean(ok, "acceptance_rate"),
        "output_throughput_tok_s": _weighted_mean(ok, "output_throughput_tok_s"),
        "speedup": _weighted_mean(ok, "speedup"),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _row_line(name: str, row: dict[str, Any]) -> str:
    """Format one fixed-width row (a dataset result or the aggregate) sharing one column layout."""
    if "error" in row:
        return f"{name:<12} {'ERROR':>9} {row['error']}"
    return (
        f"{name:<12} {row.get('completed', row.get('total_completed', 0)):>9} "
        f"{_fmt(row.get('accept_length')):>10} {_fmt(row.get('acceptance_rate')):>11} "
        f"{_fmt(row.get('output_throughput_tok_s'), 1):>10} {_fmt(row.get('speedup')):>8}"
    )


def _print_table(results: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    """Hand-rolled fixed-width table: one row per dataset, an error row for failures, then the aggregate."""
    header = f"{'dataset':<12} {'completed':>9} {'accept_len':>10} {'accept_rate':>11} {'tok/s':>10} {'speedup':>8}"
    rule = "-" * len(header)
    print(header)
    print(rule)
    for r in results:
        print(_row_line(r["dataset"], r))
    print(rule)
    print(_row_line("aggregate", aggregate))


async def _run(args: argparse.Namespace) -> int:
    """Async driver: sweep every selected dataset, print the table, optionally write JSON."""
    # A bad SHARED flag (e.g. --num-prompts 0) applies identically to every dataset;
    # validate once here so it raises immediately with a clear message, instead of
    # being caught by _run_sweep's per-dataset try/except and reported as N copies
    # of the same "dataset failure".
    _ENGINE_VALIDATORS[args.engine](args)
    specs = _select_datasets(_load_dataset_specs(args.datasets_config), args.datasets)

    results = await _run_sweep(args, specs)
    aggregate = _aggregate(results)
    _print_table(results, aggregate)

    if args.output_json:
        payload = {"results": results, "aggregate": aggregate}
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        logger.info("Wrote sweep results to %s", args.output_json)

    return 0 if aggregate["num_datasets_ok"] > 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep a trained drafter's acceptance-length benchmark across multiple datasets.",
    )
    parser.add_argument("--engine", choices=sorted(_ENGINE_MODULES), required=True, help="Which server to drive.")
    parser.add_argument("--server", required=True, help="Root URL of the running server hosting the drafter.")
    parser.add_argument("--model", required=True, help="Served model name to send in the chat payload.")
    parser.add_argument(
        "--baseline-server", default=None, help="Optional second server running WITHOUT speculation; enables speedup."
    )
    parser.add_argument(
        "--datasets-config", default=None, help="YAML list of dataset entries overriding the default 4-dataset suite."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, help="Subset of dataset names to run (default: the full suite)."
    )
    parser.add_argument("--num-prompts", type=int, default=64, help="Number of prompts to send per dataset.")
    parser.add_argument("--concurrency", type=int, default=16, help="Maximum in-flight requests.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Default max_tokens per request (per-dataset overridable)."
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Default 0.0 = greedy.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--num-steps", type=int, default=None, help="sglang engine only: speculative_num_steps fallback."
    )
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Optional shuffle seed before slicing.")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries on 5xx / 429 / transport errors.")
    parser.add_argument("--output-json", default=None, help="Optional path to write the full sweep results JSON to.")
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
