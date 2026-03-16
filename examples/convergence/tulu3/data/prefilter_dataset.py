#!/usr/bin/env python3
"""Benchmark and cache filtered datasets for ChatDataset.

Compares three filtering strategies and saves the filtered dataset to disk
so it can be loaded directly during training (skipping the expensive
per-sample tokenization filter).

Strategies benchmarked:
  1. baseline     — current _filter_long_samples: tokenize every sample via
                    apply_chat_template(tokenize=True)
  2. two_pass     — char-length pre-filter to skip Jinja render on obviously-too-long
                    samples, then exact tokenization on the rest
  3. text_only    — apply_chat_template(tokenize=False) + tokenizer.encode()
                    (skips return_dict/assistant_mask overhead)

Usage:
    # Benchmark all strategies on 5000 samples, then cache the full filtered dataset
    python scripts/prefilter_dataset.py \
        --dataset allenai/tulu-3-sft-mixture \
        --model Qwen/Qwen3-30B-A3B \
        --seq_length 1024 \
        --split train \
        --benchmark --num_benchmark_samples 5000

    # Just cache (no benchmark), save to custom dir
    python scripts/prefilter_dataset.py \
        --dataset allenai/tulu-3-sft-mixture \
        --model Qwen/Qwen3-30B-A3B \
        --seq_length 1024 \
        --cache_dir /path/to/cache

    # Use a custom chat template (file path or inline Jinja string)
    python scripts/prefilter_dataset.py \
        --dataset allenai/tulu-3-sft-mixture \
        --model Qwen/Qwen3-30B-A3B \
        --seq_length 1024 \
        --chat_template /path/to/template.json

    # Load a previously cached dataset in training config:
    #   dataset:
    #     _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
    #     path_or_dataset_id: /path/to/cache/tulu-3-sft-mixture_train_seq1024_filtered
    #     filter_long: false   # already filtered
"""

import argparse
import hashlib
import logging
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark filter_long strategies and cache filtered datasets.")
    p.add_argument("--dataset", type=str, required=True, help="HF dataset ID or local path")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B", help="Tokenizer model name")
    p.add_argument("--seq_length", type=int, required=True, help="Max sequence length")
    p.add_argument("--split", type=str, default="train", help="Dataset split")
    p.add_argument("--name", type=str, default=None, help="HF dataset config/subset name")
    p.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Chat template: path to a .json/.jinja file, or an inline Jinja string",
    )
    p.add_argument("--shuffle_seed", type=int, default=None, help="Shuffle seed (applied before slicing)")
    p.add_argument("--num_proc", type=int, default=None, help="Number of parallel workers (default: auto)")
    p.add_argument("--hf_home", type=str, default=None, help="Override HF_HOME")

    p.add_argument("--benchmark", action="store_true", help="Run benchmark comparing strategies")
    p.add_argument(
        "--num_benchmark_samples", type=int, default=5000, help="Number of samples to benchmark on (0 = all)"
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["text_only"],
        choices=["baseline", "text_only"],
        help="Strategies to benchmark",
    )

    p.add_argument("--cache_dir", type=str, default=None, help="Directory to save filtered dataset")
    p.add_argument("--no_cache", action="store_true", help="Skip caching (benchmark only)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading (mirrors chat_dataset._load_openai_messages for HF)
# ---------------------------------------------------------------------------


def load_hf_dataset(dataset_id, split, name=None, shuffle_seed=None):
    from datasets import Dataset

    from nemo_automodel.components.datasets.llm.chat_dataset import _load_openai_messages

    data = _load_openai_messages(dataset_id, split=split, name=name, shuffle_seed=shuffle_seed)
    # _load_openai_messages returns an HF Dataset for repo IDs, or a plain
    # list of dicts for local files.  Wrap lists so .filter()/.map() work.
    if isinstance(data, list):
        return Dataset.from_list(data)
    return data


# ---------------------------------------------------------------------------
# Filtering strategies
# ---------------------------------------------------------------------------


def _auto_num_proc(dataset_len, requested=None):
    if requested is not None:
        return max(1, requested)
    cpu_count = os.cpu_count() or 4
    # Use up to 80% of CPUs, but at least 1 and at most dataset_len
    return max(1, min(int(cpu_count * 0.8), dataset_len))


def filter_baseline(dataset, tokenizer, seq_length, num_proc=None):
    """Current strategy: tokenize every sample, check length."""

    def _fits(messages) -> bool:
        try:
            result = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                padding=False,
                truncation=False,
            )
            return len(result["input_ids"]) <= seq_length
        except Exception:
            return False

    np = _auto_num_proc(len(dataset), num_proc)
    return dataset.filter(
        lambda row: _fits(row["messages"]),
        num_proc=np,
        desc=f"baseline (np={np})",
    )


def filter_text_only(dataset, tokenizer, seq_length, num_proc=None):
    """Render text (no tokenize), then use tokenizer.encode() on the rendered string.

    apply_chat_template(tokenize=False) is faster than tokenize=True for many
    tokenizers because it skips the return_dict / assistant_mask overhead.
    We then call len(tokenizer.encode()) which is a fast C++ path for most
    HF tokenizers.
    """
    np = _auto_num_proc(len(dataset), num_proc)

    def _fits(row) -> bool:
        try:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            return len(tokenizer.encode(text, add_special_tokens=False)) <= seq_length
        except Exception:
            return False

    return dataset.filter(_fits, num_proc=np, desc=f"text_only (np={np})")


STRATEGIES = {
    "baseline": filter_baseline,
    "text_only": filter_text_only,
}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def compute_token_lengths(dataset, tokenizer, num_proc=None):
    """Compute token lengths for all samples using the fast text_only path."""
    np = _auto_num_proc(len(dataset), num_proc)

    def _token_len(row):
        try:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            return {"__token_len__": len(tokenizer.encode(text, add_special_tokens=False))}
        except Exception:
            return {"__token_len__": -1}

    ds = dataset.map(_token_len, num_proc=np, desc="Computing token lengths")
    return [r for r in ds["__token_len__"] if r >= 0]


def print_length_distribution(lengths, seq_length):
    """Print head (shortest) and tail (longest) sequence length distribution."""
    if not lengths:
        logger.info("No samples to analyze.")
        return

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def _percentile(p):
        idx = min(int(n * p / 100), n - 1)
        return lengths_sorted[idx]

    logger.info("\n" + "=" * 70)
    logger.info("SEQUENCE LENGTH DISTRIBUTION (%d samples, seq_length=%d)", n, seq_length)
    logger.info("=" * 70)

    logger.info("  Min:       %8d tokens", lengths_sorted[0])
    logger.info("  p1:        %8d tokens", _percentile(1))
    logger.info("  p5:        %8d tokens", _percentile(5))
    logger.info("  p10:       %8d tokens", _percentile(10))
    logger.info("  p25:       %8d tokens", _percentile(25))
    logger.info("  Median:    %8d tokens", _percentile(50))
    logger.info("  Mean:      %8.0f tokens", sum(lengths) / n)
    logger.info("  p75:       %8d tokens", _percentile(75))
    logger.info("  p90:       %8d tokens", _percentile(90))
    logger.info("  p95:       %8d tokens", _percentile(95))
    logger.info("  p99:       %8d tokens", _percentile(99))
    logger.info("  Max:       %8d tokens", lengths_sorted[-1])

    # Head distribution — shortest 10 samples
    logger.info("\n  HEAD (10 shortest samples):")
    for i, length in enumerate(lengths_sorted[:10]):
        logger.info("    #%-4d  %8d tokens", i + 1, length)

    # Tail distribution — longest 10 samples
    logger.info("\n  TAIL (10 longest samples):")
    tail = lengths_sorted[-10:]
    for i, length in enumerate(reversed(tail)):
        rank = n - i
        logger.info("    #%-4d  %8d tokens", rank, length)

    # Truncation / filter impact at seq_length
    over = sum(1 for length in lengths if length > seq_length)
    under = n - over
    logger.info("\n  FILTER IMPACT (seq_length=%d):", seq_length)
    logger.info("    Kept:     %8d  (%5.1f%%)", under, 100 * under / n)
    logger.info("    Removed:  %8d  (%5.1f%%)", over, 100 * over / n)

    # Chunked average + EMA (dataset order, not sorted)
    chunk_size = 100
    ema_alpha = 0.1
    if n >= chunk_size:
        logger.info("\n  CHUNKED AVERAGE & EMA (chunk_size=%d, alpha=%.2f):", chunk_size, ema_alpha)
        logger.info("    %-14s %10s %10s %10s %10s", "Samples", "Avg", "Min", "Max", "EMA")
        logger.info("    " + "-" * 58)
        ema = float(sum(lengths[:chunk_size]) / chunk_size)
        for start in range(0, n, chunk_size):
            chunk = lengths[start : start + chunk_size]
            if not chunk:
                break
            avg = sum(chunk) / len(chunk)
            if start == 0:
                ema = avg
            else:
                ema = ema_alpha * avg + (1 - ema_alpha) * ema
            label = f"[{start}:{start + len(chunk)}]"
            logger.info(
                "    %-14s %10.0f %10d %10d %10.0f",
                label,
                avg,
                min(chunk),
                max(chunk),
                ema,
            )

    # Bucket histogram
    buckets = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    # Only show buckets up to 2x the max length
    max_len = lengths_sorted[-1]
    buckets = [b for b in buckets if b <= max_len * 2]
    if not buckets or buckets[-1] < max_len:
        buckets.append(max_len)

    logger.info("\n  BUCKET HISTOGRAM:")
    logger.info("    %-16s %10s %7s  %s", "Range", "Count", "Pct", "Bar")
    logger.info("    " + "-" * 60)
    prev = 0
    for b in buckets:
        count = sum(1 for length in lengths if prev < length <= b)
        pct = 100 * count / n
        bar = "#" * int(pct / 2)
        label = f"({prev}, {b}]"
        logger.info("    %-16s %10d %6.1f%%  %s", label, count, pct, bar)
        prev = b
    if prev < max_len:
        count = sum(1 for length in lengths if length > prev)
        pct = 100 * count / n
        bar = "#" * int(pct / 2)
        label = f"({prev}, +inf)"
        logger.info("    %-16s %10d %6.1f%%  %s", label, count, pct, bar)


def run_benchmark(dataset, tokenizer, seq_length, strategies, num_proc=None):
    results = {}
    for name in strategies:
        fn = STRATEGIES[name]
        logger.info("--- Benchmarking strategy: %s ---", name)
        t0 = time.perf_counter()
        filtered = fn(dataset, tokenizer, seq_length, num_proc=num_proc)
        elapsed = time.perf_counter() - t0
        kept = len(filtered)
        removed = len(dataset) - kept
        results[name] = {
            "time_s": elapsed,
            "kept": kept,
            "removed": removed,
        }
        logger.info(
            "%s: %.1fs | kept %d / %d (removed %d, %.1f%%)",
            name,
            elapsed,
            kept,
            len(dataset),
            removed,
            100 * removed / max(len(dataset), 1),
        )
    return results


def print_benchmark_summary(results, dataset_len):
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY (%d samples)", dataset_len)
    logger.info("=" * 70)
    logger.info("%-15s %10s %10s %10s %10s", "Strategy", "Time (s)", "Kept", "Removed", "Speedup")
    logger.info("-" * 70)

    baseline_time = results.get("baseline", {}).get("time_s", None)
    for name, r in results.items():
        speedup = (baseline_time / r["time_s"]) if baseline_time and r["time_s"] > 0 else float("nan")
        logger.info(
            "%-15s %10.1f %10d %10d %9.2fx",
            name,
            r["time_s"],
            r["kept"],
            r["removed"],
            speedup,
        )

    # Verify consistency: all strategies should keep the same count
    counts = {name: r["kept"] for name, r in results.items()}
    unique_counts = set(counts.values())
    if len(unique_counts) == 1:
        logger.info("\nAll strategies agree: %d samples kept.", unique_counts.pop())
    else:
        logger.warning(
            "\nWARNING: strategies disagree on kept count! %s",
            {n: c for n, c in counts.items()},
        )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def get_cache_path(cache_dir, dataset_id, split, seq_length, model_name, shuffle_seed, chat_template=None):
    """Deterministic cache path based on filtering parameters."""
    # Sanitize dataset name for filesystem
    ds_name = dataset_id.replace("/", "_").replace("\\", "_")
    split_name = split.replace("[", "_").replace("]", "_").replace(":", "-")
    model_short = model_name.split("/")[-1]

    # Include a hash of the full config for uniqueness (chat_template changes tokenization)
    config_str = f"{dataset_id}|{split}|{seq_length}|{model_name}|{shuffle_seed}|{chat_template}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    dirname = f"{ds_name}_{split_name}_seq{seq_length}_{model_short}_{config_hash}"
    return os.path.join(cache_dir, dirname)


def cache_filtered_dataset(dataset, cache_path):
    """Save filtered HF dataset as Parquet (loadable via load_dataset)."""
    os.makedirs(cache_path, exist_ok=True)
    parquet_file = os.path.join(cache_path, "data.parquet")
    logger.info("Saving filtered dataset to: %s", parquet_file)
    t0 = time.perf_counter()
    dataset.to_parquet(parquet_file)
    elapsed = time.perf_counter() - t0
    logger.info("Saved %d samples in %.1fs to %s", len(dataset), elapsed, parquet_file)
    return cache_path


def verify_cached_dataset(cache_path):
    """Load back and verify the cached Parquet dataset via load_dataset."""
    from datasets import load_dataset

    logger.info("Verifying cached dataset at: %s", cache_path)
    t0 = time.perf_counter()
    ds = load_dataset(cache_path, split="train")
    elapsed = time.perf_counter() - t0
    logger.info("Loaded %d samples in %.1fs", len(ds), elapsed)
    if len(ds) > 0:
        row = ds[0]
        assert "messages" in row, f"First row missing 'messages' key, got: {list(row.keys())}"
        logger.info("Verification passed. First sample has %d messages.", len(row["messages"]))
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Disable tokenizers parallelism to avoid deadlocks with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.chat_template:
        from nemo_automodel.components.datasets.llm.formatting_utils import _resolve_chat_template

        resolved = _resolve_chat_template(args.chat_template)
        tokenizer.chat_template = resolved
        logger.info("Using custom chat template from: %s", args.chat_template)

    logger.info("Loading dataset: %s (split=%s)", args.dataset, args.split)
    dataset = load_hf_dataset(
        args.dataset,
        args.split,
        name=args.name,
        shuffle_seed=args.shuffle_seed,
    )
    logger.info("Loaded %d samples", len(dataset))

    # --- Benchmark ---
    if args.benchmark:
        bench_dataset = dataset
        if args.num_benchmark_samples > 0 and args.num_benchmark_samples < len(dataset):
            bench_dataset = dataset.select(range(args.num_benchmark_samples))
            logger.info("Benchmarking on first %d samples", args.num_benchmark_samples)

        # Collect token lengths and print distribution
        logger.info("Computing token lengths for distribution stats...")
        lengths = compute_token_lengths(bench_dataset, tokenizer, num_proc=args.num_proc)
        print_length_distribution(lengths, args.seq_length)

        results = run_benchmark(
            bench_dataset,
            tokenizer,
            args.seq_length,
            strategies=args.strategies,
            num_proc=args.num_proc,
        )
        print_benchmark_summary(results, len(bench_dataset))

    # --- Cache ---
    if not args.no_cache:
        strategy = "text_only"
        logger.info("\nFiltering full dataset (%d samples) with strategy: %s", len(dataset), strategy)
        t0 = time.perf_counter()
        filtered = STRATEGIES[strategy](dataset, tokenizer, args.seq_length, num_proc=args.num_proc)
        elapsed = time.perf_counter() - t0

        original_len = len(dataset)
        kept = len(filtered)
        removed = original_len - kept

        logger.info(
            "Filtered in %.1fs: kept %d / %d (removed %d)",
            elapsed,
            kept,
            original_len,
            removed,
        )

        # Determine cache path
        if args.cache_dir:
            cache_dir = args.cache_dir
        else:
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            cache_dir = os.path.join(hf_home, "filtered_datasets")

        cache_path = get_cache_path(
            cache_dir,
            args.dataset,
            args.split,
            args.seq_length,
            args.model,
            args.shuffle_seed,
            chat_template=args.chat_template,
        )
        cache_filtered_dataset(filtered, cache_path)
        verify_cached_dataset(cache_path)

        # --- Print summary statistics ---
        logger.info("\n" + "=" * 70)
        logger.info("FILTERING STATISTICS")
        logger.info("=" * 70)
        logger.info("  Dataset:        %s (split=%s)", args.dataset, args.split)
        logger.info("  Model:          %s", args.model)
        logger.info("  Seq length:     %d", args.seq_length)
        if args.chat_template:
            logger.info("  Chat template:  %s", args.chat_template)
        logger.info("  Strategy:       %s", strategy)
        logger.info("-" * 70)
        logger.info("  Original:       %d samples", original_len)
        logger.info("  Kept:           %d samples (%.1f%%)", kept, 100 * kept / max(original_len, 1))
        logger.info("  Removed:        %d samples (%.1f%%)", removed, 100 * removed / max(original_len, 1))
        logger.info("  Filter time:    %.1fs", elapsed)
        logger.info("-" * 70)
        logger.info("  Cached to:      %s", cache_path)
        logger.info("")
        logger.info("  To use in training config:")
        logger.info("    dataset:")
        logger.info("      path_or_dataset_id: %s", cache_path)
        logger.info("      filter_long: false")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
