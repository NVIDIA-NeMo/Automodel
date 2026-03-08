#!/usr/bin/env python3
"""Benchmark greedy_knapsack: speed and packing efficiency across scales and distributions."""

import random
import statistics
import time

from nemo_automodel.components.datasets.llm.neat_packing import greedy_knapsack


def _run_benchmark(name: str, lengths: list[int], max_length: int) -> dict:
    """Run knapsack and return stats."""
    N = len(lengths)

    t0 = time.perf_counter()
    bins = greedy_knapsack(lengths, max_length)
    elapsed = time.perf_counter() - t0

    # ── Packing stats ──
    n_bins = len(bins)
    bin_fills = [sum(lengths[i] for i in b) for b in bins]
    total_tokens = sum(lengths)
    total_capacity = n_bins * max_length
    utilization = 100.0 * total_tokens / total_capacity if total_capacity else 0

    samples_per_bin = [len(b) for b in bins]

    result = {
        "name": name,
        "N": N,
        "max_length": max_length,
        "n_bins": n_bins,
        "time_s": elapsed,
        "rate_samples_per_s": N / elapsed if elapsed > 0 else float("inf"),
        "utilization_pct": utilization,
        "avg_fill": statistics.mean(bin_fills) if bin_fills else 0,
        "min_fill": min(bin_fills) if bin_fills else 0,
        "max_fill": max(bin_fills) if bin_fills else 0,
        "avg_samples_per_bin": statistics.mean(samples_per_bin) if samples_per_bin else 0,
        "min_samples_per_bin": min(samples_per_bin) if samples_per_bin else 0,
        "max_samples_per_bin": max(samples_per_bin) if samples_per_bin else 0,
        "avg_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
    }
    return result


def _print_result(r: dict):
    print(f"\n{'=' * 70}")
    print(f"  {r['name']}")
    print(f"{'=' * 70}")
    print(f"  Samples:       {r['N']:>10,}")
    print(f"  Pack size:     {r['max_length']:>10,}")
    print(f"  Bins:          {r['n_bins']:>10,}")
    print(f"  Time:          {r['time_s']:>10.3f}s")
    print(f"  Throughput:    {r['rate_samples_per_s']:>10,.0f} samples/s")
    print("  ──────────────────────────────────────")
    print(f"  Utilization:   {r['utilization_pct']:>10.1f}%")
    print(f"  Avg fill:      {r['avg_fill']:>10.0f} / {r['max_length']}")
    print(f"  Min fill:      {r['min_fill']:>10,}")
    print(f"  Max fill:      {r['max_fill']:>10,}")
    print("  ──────────────────────────────────────")
    print(
        f"  Samples/bin:   avg={r['avg_samples_per_bin']:.1f}  min={r['min_samples_per_bin']}  max={r['max_samples_per_bin']}"
    )
    print(f"  Sample length: avg={r['avg_length']:.0f}  median={r['median_length']:.0f}")


def main():
    rng = random.Random(42)
    pack_size = 8192

    benchmarks = []

    # ── 1. Varying scale ──────────────────────────────────────
    for N in [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]:
        lengths = [rng.randint(100, 4000) for _ in range(N)]
        r = _run_benchmark(f"Uniform(100-4000) N={N:,}", lengths, pack_size)
        benchmarks.append(r)
        _print_result(r)

    # ── 2. Different distributions (N=100K) ───────────────────
    N = 100_000

    # Mostly short samples
    lengths = [rng.randint(50, 500) for _ in range(N)]
    r = _run_benchmark("Mostly short (50-500)", lengths, pack_size)
    benchmarks.append(r)
    _print_result(r)

    # Mostly long samples
    lengths = [rng.randint(4000, 7500) for _ in range(N)]
    r = _run_benchmark("Mostly long (4000-7500)", lengths, pack_size)
    benchmarks.append(r)
    _print_result(r)

    # Bimodal: 50% short + 50% long
    lengths = [rng.randint(50, 500) for _ in range(N // 2)] + [rng.randint(4000, 7500) for _ in range(N // 2)]
    rng.shuffle(lengths)
    r = _run_benchmark("Bimodal (short+long)", lengths, pack_size)
    benchmarks.append(r)
    _print_result(r)

    # Heavy tail (log-normal-ish)
    lengths = [min(pack_size, int(rng.lognormvariate(6, 1.2))) for _ in range(N)]
    lengths = [max(10, x) for x in lengths]
    r = _run_benchmark("Heavy-tail (lognormal)", lengths, pack_size)
    benchmarks.append(r)
    _print_result(r)

    # Near pack_size (worst case for bin creation)
    lengths = [rng.randint(pack_size - 500, pack_size) for _ in range(N)]
    r = _run_benchmark("Near pack_size (7692-8192)", lengths, pack_size)
    benchmarks.append(r)
    _print_result(r)

    # ── Summary table ─────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'Name':<35} {'N':>10} {'Bins':>8} {'Time':>8} {'Rate':>12} {'Util%':>7}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 12} {'-' * 7}")
    for r in benchmarks:
        print(
            f"  {r['name']:<35} {r['N']:>10,} {r['n_bins']:>8,} "
            f"{r['time_s']:>7.2f}s {r['rate_samples_per_s']:>11,.0f}/s "
            f"{r['utilization_pct']:>6.1f}%"
        )


if __name__ == "__main__":
    main()
