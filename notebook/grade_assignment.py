#!/usr/bin/env python3
"""
Grading Script for LLM Optimization Labs

Usage:
    python grade_assignment.py submission/

Expected files in submission directory:
    - answers.yaml              (30 pts - Arithmetic Intensity)
    - automodel_profile_*.nsys-rep  (5 pts - Nsight profile)
    - benchmark_results.json    (35 pts - MFU score)
"""

import argparse
import json
import math
from pathlib import Path

import yaml
import torch


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def calc_arithmetic_intensity(n, dtype):
    """Calculate arithmetic intensity for NxN matrix multiplication."""
    flops = 2 * n ** 3
    bytes_per_elem = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    bytes_accessed = 3 * (n ** 2) * bytes_per_elem
    return flops / bytes_accessed


def calc_mfu_points(mfu):
    """MFU scoring: >=20.5% = 35pts, 1-20.5% = linear, <1% = 0."""
    if mfu >= 20.5:
        return 35.0
    elif mfu >= 1.0:
        return (mfu - 1.0) / 19.5 * 35.0
    return 0.0


def grade(submission_dir):
    path = Path(submission_dir)
    results = {"ai": 0, "nsys": 0, "mfu": 0, "details": []}

    # --- Arithmetic Intensity (30 pts) ---
    answers_file = path / "answers.yaml"
    if answers_file.exists():
        answers = load_yaml(answers_file)
        for n in [16, 64, 256, 1024, 4096]:
            for dtype, name in [(torch.float16, "fp16"), (torch.float32, "fp32")]:
                key = f"ai_{name}_n{n}"
                expected = calc_arithmetic_intensity(n, dtype)
                if key in answers and math.isclose(answers[key], expected, rel_tol=1e-3):
                    results["ai"] += 3
                else:
                    results["details"].append(f"❌ {key}")
    else:
        results["details"].append("❌ answers.yaml not found")

    # --- Nsight Profile (5 pts) ---
    nsys_files = list(path.glob("automodel_profile_*.nsys-rep"))
    if nsys_files:
        results["nsys"] = 5
    else:
        results["details"].append("❌ automodel_profile_*.nsys-rep not found")

    # --- MFU Score (35 pts) ---
    benchmark_file = path / "benchmark_results.json"
    if benchmark_file.exists():
        data = load_json(benchmark_file)
        mfu = data.get("avg_mfu_percent", 0)
        results["mfu"] = calc_mfu_points(mfu)
        results["mfu_percent"] = mfu
    else:
        results["details"].append("❌ benchmark_results.json not found")

    results["total"] = results["ai"] + results["nsys"] + results["mfu"]
    return results


def main():
    parser = argparse.ArgumentParser(description="Grade LLM Optimization Labs")
    parser.add_argument("submission_dir", help="Path to submission directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    path = Path(args.submission_dir)
    if not path.is_dir():
        print(f"Error: {args.submission_dir} is not a directory")
        return

    r = grade(args.submission_dir)

    print("=" * 50)
    print("GRADING RESULTS")
    print("=" * 50)
    print(f"  Arithmetic Intensity:  {r['ai']}/30")
    print(f"  Nsight Profile:        {r['nsys']}/5")
    print(f"  MFU Score:             {r['mfu']:.1f}/35", end="")
    if "mfu_percent" in r:
        print(f"  ({r['mfu_percent']:.1f}%)")
    else:
        print()
    print("=" * 50)
    print(f"  TOTAL: {r['total']:.1f}/70 ({r['total']/70*100:.1f}%)")
    print("=" * 50)

    if r["details"]:
        print("\nIssues:")
        for d in r["details"]:
            print(f"  {d}")

    # Save results
    output = path / "results.yaml"
    with open(output, "w") as f:
        yaml.dump(r, f)
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
