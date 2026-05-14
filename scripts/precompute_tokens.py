# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#!/usr/bin/env python3
"""Pre-compute accurate text token counts for all samples in a meta dataset.

Adds ``_text_tokens`` to each sample in JSONL files so that
LengthGroupedSampler can use exact values instead of the chars//3 heuristic.

Datasets are processed **sequentially**, but within each dataset all N
workers tokenize lines in parallel — no long-tail stalls.

Usage:
    python scripts/precompute_tokens.py \
        --meta /path/to/sft_v15.json \
        --processor /path/to/Qwen3-VL-8B-Instruct \
        --output-dir /path/to/output \
        --workers 32
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import tempfile
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

_MEDIA_TAG_RE = re.compile(r"<image>|<video>", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------
_worker_tokenizer = None  # loaded once per worker


def _worker_init(tokenizer_path):
    """Called once per worker process to load the tokenizer."""
    global _worker_tokenizer
    from transformers import AutoTokenizer

    _worker_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )


def _process_chunk(args):
    """Process a chunk of lines. Returns list of (index, modified_json_line)."""
    start, chunk_lines, columns, tags = args
    tokenizer = _worker_tokenizer

    messages_col = columns.get("messages", "messages")
    role_tag = tags.get("role_tag", "role")
    content_tag = tags.get("content_tag", "content")
    user_tag = tags.get("user_tag", "user")
    assistant_tag = tags.get("assistant_tag", "assistant")

    results = []
    for i, line in enumerate(chunk_lines, start=start):
        sample = json.loads(line)

        # Extract text
        texts = []
        for msg in sample.get(messages_col, []):
            role = msg.get(role_tag, "")
            if role not in (user_tag, assistant_tag):
                continue
            content = msg.get(content_tag, "")
            text = _MEDIA_TAG_RE.sub("", content).strip()
            if text:
                texts.append(text)

        if texts:
            text_tokens = len(tokenizer.encode("\n".join(texts), add_special_tokens=False))
        else:
            text_tokens = 0

        sample["_text_tokens"] = text_tokens
        results.append((i, json.dumps(sample, ensure_ascii=False) + "\n"))

    return results


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def _process_one_dataset(ds_name, file_path, output_path, columns, tags, pool, chunk_size=2000):
    """Process one JSONL file using a pre-created worker pool."""
    t0 = time.monotonic()

    logger.info("[%s] Reading %s ...", ds_name, file_path)
    with open(file_path) as f:
        all_lines = [line for line in f if line.strip()]
    n_lines = len(all_lines)
    t_read = time.monotonic() - t0
    logger.info("[%s] %d lines read in %.1fs", ds_name, n_lines, t_read)

    # Build chunks — lines are passed via IPC to each worker
    chunk_args = []
    for start in range(0, n_lines, chunk_size):
        end = min(start + chunk_size, n_lines)
        chunk_args.append((start, all_lines[start:end], columns, tags))

    # Tokenize + serialize in parallel (pool already has tokenizer loaded)
    t_tok_start = time.monotonic()
    output_lines = [None] * n_lines
    done = 0

    for result_batch in pool.imap_unordered(_process_chunk, chunk_args):
        for idx, modified_line in result_batch:
            output_lines[idx] = modified_line
        done += len(result_batch)
        if done % 100_000 < chunk_size or done == n_lines:
            elapsed = time.monotonic() - t_tok_start
            logger.info(
                "[%s] %d/%d (%.1fs, %.0f lines/s)",
                ds_name,
                done,
                n_lines,
                elapsed,
                done / max(elapsed, 1e-6),
            )

    t_tok = time.monotonic() - t_tok_start
    del all_lines  # free memory

    # Write output — lines already serialized, just join and write
    t_write_start = time.monotonic()
    tmpfd, tmppath = tempfile.mkstemp(suffix=".jsonl", dir=os.path.dirname(output_path))
    try:
        with os.fdopen(tmpfd, "w") as fout:
            fout.writelines(output_lines)
        os.replace(tmppath, output_path)
    except BaseException:
        try:
            os.unlink(tmppath)
        except OSError:
            pass
        raise
    t_write = time.monotonic() - t_write_start

    elapsed = time.monotonic() - t0
    return {
        "name": ds_name,
        "n_lines": n_lines,
        "t_read": t_read,
        "t_tokenize": t_tok,
        "t_write": t_write,
        "t_total": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Precompute text token counts for datasets listed in a meta file."""

    mp.set_start_method("fork", force=True)  # COW shared memory

    parser = argparse.ArgumentParser(description="Pre-compute text token counts for meta dataset")
    parser.add_argument("--meta", required=True, help="Path to meta JSON file")
    parser.add_argument("--processor", required=True, help="Processor/tokenizer path")
    parser.add_argument("--output-dir", default=None, help="Output directory for modified JSONL files")
    parser.add_argument("--inplace", action="store_true", help="Modify JSONL files in place")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers per dataset")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Lines per worker chunk")
    parser.add_argument("--datasets", nargs="*", default=None, help="Only process these datasets")
    args = parser.parse_args()

    if not args.inplace and not args.output_dir:
        parser.error("Must specify either --output-dir or --inplace")

    with open(args.meta) as f:
        meta = json.load(f)
    meta_dir = os.path.dirname(os.path.abspath(args.meta))

    if args.datasets:
        missing = set(args.datasets) - set(meta.keys())
        if missing:
            parser.error(f"Datasets not found in meta: {missing}")
        selected = {k: meta[k] for k in args.datasets}
    else:
        selected = meta

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Build task list
    tasks = []
    for ds_name in sorted(selected.keys()):
        cfg = selected[ds_name]
        file_name = cfg.get("file_name", "")
        if not os.path.isabs(file_name):
            file_name = os.path.join(meta_dir, file_name)
        if not os.path.exists(file_name):
            logger.warning("Skipping %s: file not found (%s)", ds_name, file_name)
            continue

        columns = cfg.get("columns", {})
        tags = cfg.get("tags", {})

        if args.inplace:
            output_path = file_name
        else:
            output_path = os.path.join(args.output_dir, os.path.basename(file_name))

        tasks.append((ds_name, file_name, output_path, columns, tags))

    logger.info("Processing %d datasets sequentially, %d workers each", len(tasks), args.workers)
    logger.info("Initializing worker pool (loading tokenizer in each worker) ...")

    t_start = time.monotonic()
    results = []

    # Create pool ONCE — workers load tokenizer in _worker_init, then reuse
    with mp.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.processor,),
    ) as pool:
        t_init = time.monotonic() - t_start
        logger.info("Worker pool ready in %.1fs", t_init)

        for i, (ds_name, fpath, opath, cols, tgs) in enumerate(tasks):
            logger.info("━━━ [%d/%d] %s ━━━", i + 1, len(tasks), ds_name)
            result = _process_one_dataset(
                ds_name,
                fpath,
                opath,
                cols,
                tgs,
                pool=pool,
                chunk_size=args.chunk_size,
            )
            results.append(result)

    # Write updated meta JSON
    if args.output_dir:
        new_meta = dict(meta)
        for ds_name, file_name, output_path, _, _ in tasks:
            new_meta[ds_name] = dict(meta[ds_name])
            new_meta[ds_name]["file_name"] = output_path
        new_meta_path = os.path.join(args.output_dir, os.path.basename(args.meta))
        with open(new_meta_path, "w") as f:
            json.dump(new_meta, f, indent=2, ensure_ascii=False)
        logger.info("Wrote updated meta JSON: %s", new_meta_path)

    # Summary
    t_total = time.monotonic() - t_start
    total_lines = sum(r["n_lines"] for r in results)

    results_sorted = sorted(results, key=lambda r: r["t_total"], reverse=True)
    BAR_W = 30
    max_time = results_sorted[0]["t_total"] if results_sorted else 1.0
    max_name = max((len(r["name"]) for r in results), default=10)

    lines = [
        "",
        f"  ┌{'─' * (max_name + BAR_W + 42)}┐",
        f"  │{'PRECOMPUTE TOKEN SUMMARY':^{max_name + BAR_W + 42}}│",
        f"  ├{'─' * (max_name + BAR_W + 42)}┤",
        f"  │  {'Dataset':<{max_name}}  {'Timeline':<{BAR_W}}  {'Read':>6}  {'Token':>6}  {'Write':>6}  {'Lines':>9}  │",
        f"  ├{'─' * (max_name + BAR_W + 42)}┤",
    ]
    for r in results_sorted[:20]:
        ratio = r["t_total"] / max_time
        bar = ("█" * int(ratio * BAR_W)).ljust(BAR_W)
        lines.append(
            f"  │  {r['name']:<{max_name}}  {bar}  {r['t_read']:>5.1f}s  {r['t_tokenize']:>5.1f}s  {r['t_write']:>5.1f}s  {r['n_lines']:>9,}  │"
        )
    if len(results_sorted) > 20:
        lines.append(f"  │  {'... and ' + str(len(results_sorted) - 20) + ' more':<{max_name + BAR_W + 36}}  │")

    lines.extend(
        [
            f"  ├{'─' * (max_name + BAR_W + 42)}┤",
            f"  │  {'Total':<{max_name}}  {'':>{BAR_W}}  {'':>6}  {'':>6}  {'':>6}  {total_lines:>9,}  │",
            f"  │  {'Wall clock':<{max_name}}  {'':>{BAR_W}}  {'':>6}  {t_total:>5.1f}s  {'':>6}  {'':>9}  │",
            f"  └{'─' * (max_name + BAR_W + 42)}┘",
            "",
        ]
    )
    logger.info("\n".join(lines))


if __name__ == "__main__":
    main()
