#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Prepare and length-prefilter the CoderForge agent-trajectory dataset for SFT.

CoderForge (``togethercomputer/CoderForge-Preview``) ships OpenHands agent
trajectories already in OpenAI chat format, but with two gotchas this script
handles so the output drops straight into ``ChatDataset``:

  1. ``messages`` and ``tools`` are JSON *strings* (not parsed structures).
  2. Every message carries the union of all message keys, so plain assistant
     turns have ``tool_calls: null``. ``ChatDataset._normalize_messages`` raises
     on that, so we strip null ``tool_calls`` / legacy ``function_call`` / null
     fields, emitting minimal clean OpenAI messages.

The output is written as **JSONL** (not Parquet) on purpose: Arrow unifies the
per-message struct keys and would re-add ``tool_calls: null``, reintroducing the
crash. ``ChatDataset`` parses local JSONL manually, preserving per-message keys.

Pipeline (two passes; the expensive pass runs once):

  * Pass 1 (analyze): parse + clean every trajectory, tokenize it once with the
    Gemma4 chat template (tools-aware, untruncated) to get an exact ``n_tokens``,
    and write an *analyzed* JSONL cache holding the clean messages/tools plus
    ``n_tokens``. Prints a coverage curve (retention vs seq_length).
  * Pass 2 (filter): keep trajectories with ``n_tokens <= seq_length`` and write
    a per-seq_length JSONL ready for training. This re-reads the analyzed cache,
    so producing a larger-seq_length variant later costs a cheap re-filter with
    no re-tokenization.

``n_tokens`` is the exact untruncated chat-template length, so filtering on it
guarantees no training sample is truncated (truncation drops the terminal
``<turn|>`` and causes inference death-looping; see the tulu3 README).

Usage:
    # Analysis-first: tokenize once, print the coverage curve, cache analyzed JSONL
    python examples/convergence/coderforge/data/prefilter_dataset.py \
        --model /path/to/hf_gemma4_31b_it \
        --cache_dir ./cached

    # Produce a training-ready cache at a chosen seq_length (cheap if analyzed)
    python examples/convergence/coderforge/data/prefilter_dataset.py \
        --model /path/to/hf_gemma4_31b_it \
        --cache_dir ./cached \
        --seq_length 32768

    # Quick smoke test on a handful of trajectories
    python examples/convergence/coderforge/data/prefilter_dataset.py \
        --model /path/to/hf_gemma4_31b_it --max_samples 20 --seq_length 16384

    # Use the cached JSONL in a training config:
    #   dataset:
    #     _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
    #     path_or_dataset_id: ./cached/coderforge_filtered_reward1_seq32768/data.jsonl
    #     seq_length: 32768
"""

import argparse
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Candidate sequence lengths for the coverage curve. Each is a multiple of 2048,
# so it is divisible by 2*cp_size for any cp_size that is a power of two up to
# 1024 -- the context-parallel batch sharder pads to 2*cp_size anyway.
DEFAULT_CANDIDATE_SEQ_LENS = [8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for CoderForge prefiltering."""
    p = argparse.ArgumentParser(description="Parse, clean, length-analyze and prefilter CoderForge for SFT.")
    p.add_argument("--dataset", type=str, default="togethercomputer/CoderForge-Preview", help="HF dataset ID")
    p.add_argument("--name", type=str, default="trajectories", help="HF dataset config/subset name")
    p.add_argument("--split", type=str, default="filtered_reward1", help="Dataset split")
    p.add_argument("--model", type=str, required=True, help="Tokenizer path or HF id (Gemma4 checkpoint dir)")
    p.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Override chat template (path to .json/.jinja or inline Jinja). Default: tokenizer's own template.",
    )
    p.add_argument(
        "--seq_length",
        type=int,
        default=None,
        help="If set, also write a training-ready cache keeping trajectories with n_tokens <= seq_length.",
    )
    p.add_argument(
        "--candidate_seq_lens",
        type=int,
        nargs="+",
        default=DEFAULT_CANDIDATE_SEQ_LENS,
        help="Sequence lengths to report retention for in the coverage curve.",
    )
    p.add_argument("--reward_threshold", type=float, default=None, help="Optional: keep only reward >= this value.")
    p.add_argument(
        "--max_samples", type=int, default=0, help="Limit number of trajectories (0 = all). For smoke tests."
    )
    p.add_argument("--num_proc", type=int, default=None, help="Parallel workers for tokenization (default: auto).")
    p.add_argument("--cache_dir", type=str, default=None, help="Directory for analyzed + filtered JSONL caches.")
    p.add_argument("--hf_home", type=str, default=None, help="Override HF_HOME.")
    p.add_argument("--force_analyze", action="store_true", help="Recompute the analyzed cache even if it exists.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Message cleaning (CoderForge union schema -> minimal clean OpenAI messages)
# ---------------------------------------------------------------------------


def clean_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a CoderForge message to a minimal, ChatDataset-safe OpenAI message.

    Drops the union-schema fields that would otherwise break normalization:
    null ``tool_calls`` on non-tool-calling turns, legacy ``function_call``, and
    null ``name`` / ``tool_call_id`` where they do not apply.

    Args:
        msg: A single raw CoderForge message dict.

    Returns:
        A cleaned message dict with only the keys the chat template needs.
    """
    role = msg.get("role")
    out: Dict[str, Any] = {"role": role, "content": msg.get("content") or ""}

    if role == "assistant":
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            cleaned: List[Dict[str, Any]] = []
            for i, call in enumerate(tool_calls):
                fn = call.get("function") or {}
                args = fn.get("arguments")
                if args is None:
                    args = ""
                elif not isinstance(args, str):
                    args = json.dumps(args)
                cleaned.append(
                    {
                        "id": call.get("id") or f"call_{i}",
                        "type": call.get("type") or "function",
                        "function": {"name": fn.get("name") or "", "arguments": args},
                    }
                )
            out["tool_calls"] = cleaned
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            out["reasoning_content"] = reasoning
    elif role == "tool":
        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            out["tool_call_id"] = tool_call_id
        name = msg.get("name")
        if name:
            out["name"] = name

    return out


def parse_and_clean(messages_str: str, tools_str: Optional[str]) -> tuple[List[Dict[str, Any]], Optional[List[Dict]]]:
    """Parse the JSON-string ``messages``/``tools`` columns and clean messages.

    Args:
        messages_str: JSON-encoded list of messages.
        tools_str: JSON-encoded list of tool schemas, or ``None``.

    Returns:
        A ``(messages, tools)`` tuple of parsed Python objects; ``tools`` is
        ``None`` when the trajectory ships no tools.
    """
    messages = [clean_message(m) for m in json.loads(messages_str)]
    tools = None
    if tools_str:
        parsed = json.loads(tools_str)
        if isinstance(parsed, list) and parsed:
            tools = parsed
    return messages, tools


# ---------------------------------------------------------------------------
# Loading + tokenization
# ---------------------------------------------------------------------------


def _auto_num_proc(dataset_len: int, requested: Optional[int] = None) -> int:
    """Pick a worker count: explicit value, else ~80% of CPUs, clamped sanely."""
    if requested is not None:
        return max(1, requested)
    cpu_count = os.cpu_count() or 4
    return max(1, min(int(cpu_count * 0.8), max(dataset_len, 1)))


def load_raw_dataset(dataset_id: str, name: str, split: str, max_samples: int):
    """Load only the target split's parquet shards (messages/tools stay as JSON strings).

    CoderForge's ``trajectories`` config defines several splits in one config, and
    ``load_dataset(..., split=split)`` would still download every split's shards
    before selecting. Pointing ``data_files`` at just this split's shards (e.g.
    ``trajectories/filtered_reward1-*.parquet``) downloads only what we need.
    """
    from datasets import load_dataset

    data_files = f"{name}/{split}-*.parquet" if name else f"{split}-*.parquet"
    split_arg = f"train[:{max_samples}]" if max_samples and max_samples > 0 else "train"
    return load_dataset(dataset_id, data_files=data_files, split=split_arg)


def add_token_lengths(dataset, tokenizer, num_proc: Optional[int] = None):
    """Add an exact ``n_tokens`` column (tools-aware, untruncated) and a ``valid`` flag.

    Keeps ``messages``/``tools`` as their original string columns to avoid Arrow
    struct unification; only parses them transiently to measure length.
    """
    from nemo_automodel.components.datasets.llm.formatting_utils import _tokenized_chat_length

    np = _auto_num_proc(len(dataset), num_proc)

    def _measure(row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            messages, tools = parse_and_clean(row["messages"], row.get("tools"))
            n = _tokenized_chat_length(tokenizer, messages, tools=tools)
            return {"n_tokens": n, "valid": True}
        except Exception:
            return {"n_tokens": -1, "valid": False}

    return dataset.map(_measure, num_proc=np, desc=f"Tokenizing (np={np})")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_length_distribution(lengths: List[int]) -> None:
    """Print percentile, head/tail and bucket-histogram stats for token lengths."""
    if not lengths:
        logger.info("No valid samples to analyze.")
        return

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def _pct(p: int) -> int:
        return lengths_sorted[min(int(n * p / 100), n - 1)]

    logger.info("\n" + "=" * 70)
    logger.info("TOKEN LENGTH DISTRIBUTION (%d valid trajectories)", n)
    logger.info("=" * 70)
    for label, val in [
        ("Min", lengths_sorted[0]),
        ("p10", _pct(10)),
        ("p25", _pct(25)),
        ("Median", _pct(50)),
        ("p75", _pct(75)),
        ("p90", _pct(90)),
        ("p95", _pct(95)),
        ("p99", _pct(99)),
        ("Max", lengths_sorted[-1]),
    ]:
        logger.info("  %-8s %9d tokens", label, val)
    logger.info("  %-8s %9.0f tokens", "Mean", sum(lengths) / n)


def print_coverage_curve(lengths: List[int], candidate_seq_lens: List[int]) -> None:
    """Print retention (% kept) at each candidate seq_length -- the analysis-first view."""
    if not lengths:
        return
    n = len(lengths)
    logger.info("\n" + "=" * 70)
    logger.info("COVERAGE CURVE (retention vs seq_length)")
    logger.info("=" * 70)
    logger.info("  %-12s %10s %9s   %s", "seq_length", "kept", "kept %", "bar")
    logger.info("  " + "-" * 60)
    for s in sorted(candidate_seq_lens):
        kept = sum(1 for length in lengths if length <= s)
        pct = 100 * kept / n
        bar = "#" * int(pct / 2)
        logger.info("  %-12d %10d %8.1f%%   %s", s, kept, pct, bar)


# ---------------------------------------------------------------------------
# JSONL caching
# ---------------------------------------------------------------------------


def _cache_root(cache_dir: Optional[str]) -> str:
    if cache_dir:
        return cache_dir
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(hf_home, "coderforge_cached")


def analyzed_cache_path(cache_dir: Optional[str], dataset_id: str, split: str, model: str, chat_template) -> str:
    """Deterministic path for the analyzed (parsed + clean + n_tokens) JSONL cache."""
    ds_name = dataset_id.replace("/", "_")
    model_short = model.rstrip("/").split("/")[-1]
    config_hash = hashlib.md5(f"{dataset_id}|{split}|{model}|{chat_template}".encode()).hexdigest()[:8]
    fname = f"{ds_name}_{split}_{model_short}_{config_hash}_analyzed.jsonl"
    return os.path.join(_cache_root(cache_dir), "analyzed", fname)


def filtered_cache_path(cache_dir: Optional[str], dataset_id: str, split: str, seq_length: int) -> str:
    """Deterministic directory for a training-ready, length-filtered JSONL."""
    ds_name = dataset_id.replace("/", "_")
    dirname = f"{ds_name}_{split}_seq{seq_length}"
    return os.path.join(_cache_root(cache_dir), dirname)


def write_analyzed_jsonl(dataset, path: str, reward_threshold: Optional[float]) -> List[int]:
    """Write the analyzed JSONL (clean messages/tools + n_tokens) and return kept lengths."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lengths: List[int] = []
    skipped_invalid = 0
    skipped_reward = 0
    t0 = time.perf_counter()
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            if not row.get("valid", False) or row.get("n_tokens", -1) < 0:
                skipped_invalid += 1
                continue
            if reward_threshold is not None and float(row.get("reward", 0.0)) < reward_threshold:
                skipped_reward += 1
                continue
            messages, tools = parse_and_clean(row["messages"], row.get("tools"))
            record = {
                "trajectory_id": row.get("trajectory_id"),
                "reward": row.get("reward"),
                "n_tokens": row["n_tokens"],
                "messages": messages,
                "tools": tools,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            lengths.append(row["n_tokens"])
    logger.info(
        "Analyzed cache written: %d trajectories in %.1fs (skipped %d invalid, %d below reward) -> %s",
        len(lengths),
        time.perf_counter() - t0,
        skipped_invalid,
        skipped_reward,
        path,
    )
    return lengths


def read_analyzed_jsonl(path: str) -> List[int]:
    """Read an analyzed JSONL and return its ``n_tokens`` values (for re-filtering)."""
    lengths: List[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lengths.append(json.loads(line)["n_tokens"])
    return lengths


def write_filtered_jsonl(analyzed_path: str, out_dir: str, seq_length: int) -> int:
    """Stream the analyzed JSONL and write trajectories with n_tokens <= seq_length.

    Emits only ``messages`` and ``tools`` (the fields ChatDataset consumes) plus
    lightweight metadata. Returns the number of trajectories kept.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.jsonl")
    kept = 0
    total = 0
    with open(analyzed_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            if record["n_tokens"] > seq_length:
                continue
            fout.write(
                json.dumps(
                    {
                        "trajectory_id": record.get("trajectory_id"),
                        "messages": record["messages"],
                        "tools": record["tools"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
    logger.info(
        "Filtered cache (seq_length=%d): kept %d / %d (%.1f%%) -> %s",
        seq_length,
        kept,
        total,
        100 * kept / max(total, 1),
        out_path,
    )
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the CoderForge analyze + (optional) filter pipeline."""
    args = parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    analyzed_path = analyzed_cache_path(args.cache_dir, args.dataset, args.split, args.model, args.chat_template)

    if os.path.exists(analyzed_path) and not args.force_analyze:
        logger.info("Reusing analyzed cache: %s (use --force_analyze to rebuild)", analyzed_path)
        lengths = read_analyzed_jsonl(analyzed_path)
    else:
        from transformers import AutoTokenizer

        logger.info("Loading tokenizer: %s", args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if args.chat_template:
            from nemo_automodel.components.datasets.llm.formatting_utils import _resolve_chat_template

            tokenizer.chat_template = _resolve_chat_template(args.chat_template)
            logger.info("Using custom chat template from: %s", args.chat_template)

        logger.info("Loading %s (name=%s, split=%s)", args.dataset, args.name, args.split)
        dataset = load_raw_dataset(args.dataset, args.name, args.split, args.max_samples)
        logger.info("Loaded %d trajectories", len(dataset))

        dataset = add_token_lengths(dataset, tokenizer, num_proc=args.num_proc)
        lengths = write_analyzed_jsonl(dataset, analyzed_path, args.reward_threshold)

    print_length_distribution(lengths)
    print_coverage_curve(lengths, args.candidate_seq_lens)

    if args.seq_length is not None:
        out_dir = filtered_cache_path(args.cache_dir, args.dataset, args.split, args.seq_length)
        write_filtered_jsonl(analyzed_path, out_dir, args.seq_length)
        logger.info("\nTo use in a training config:")
        logger.info("  dataset:")
        logger.info("    _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset")
        logger.info("    path_or_dataset_id: %s", os.path.join(out_dir, "data.jsonl"))
        logger.info("    seq_length: %d", args.seq_length)


if __name__ == "__main__":
    main()
