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

"""Inference quality analysis for lm_eval benchmark outputs.

Scans lm_eval sample JSONL files and results JSON files to detect common
failure modes in model-generated responses:

  - Death looping:   duplicate sentence ratio > 0.3, any sentence repeats
                     >= 5x, or a contiguous block >= 4 identical sentences.
  - Missing EOS:     response length is at the generation limit without a
                     termination token.
  - Empty response:  response is empty or whitespace-only after stripping
                     ``<think>...</think>`` tags.
  - Abrupt ending:   response ends mid-sentence (no terminal punctuation).

Outputs a human-readable report and optionally writes structured JSON for CI
consumption.  When ``--threshold-file`` is provided the script exits non-zero
if any metric violates the configured threshold, enabling automated quality
gating.

Usage examples
--------------
Analyze a single JSONL file::

    python analyze_quality.py samples_ifeval_2026-03-05.jsonl

Analyze all samples in a results directory::

    python analyze_quality.py results/ --word-threshold 300

Export structured JSON and gate on quality thresholds::

    python analyze_quality.py results/ \\
        --export quality_report.json \\
        --threshold-file thresholds.yaml

Example ``thresholds.yaml``::

    death_loop_rate: 0.01
    empty_response_rate: 0.005
    missing_eos_rate: 0.02
    abrupt_ending_rate: 0.05
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------


def resolve_jsonl_files(input_path: str) -> list[Path]:
    """Accept a file or directory.  Returns sorted list of sample JSONL files."""
    p = Path(input_path)
    if p.is_file() and p.suffix == ".jsonl":
        return [p]
    if p.is_dir():
        files = sorted(p.rglob("samples_*.jsonl"))
        if not files:
            files = sorted(p.rglob("*.jsonl"))
        return files
    raise FileNotFoundError(f"Input path does not exist or has no JSONL files: {input_path}")


def resolve_results_dir(input_path: str) -> Path | None:
    """Return the directory that may contain ``results_*.json`` files."""
    p = Path(input_path)
    if p.is_file():
        p = p.parent
    return p if p.is_dir() else None


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def _parse_sample_filename(jsonl_path: str) -> tuple[str, str]:
    """Extract ``(task_name, timestamp)`` from a samples JSONL filename.

    Example: ``samples_ifeval_2026-03-05T15-23-33.122647.jsonl``
    returns  ``("ifeval", "2026-03-05T15-23-33.122647")``.
    """
    stem = Path(jsonl_path).stem
    name = stem.removeprefix("samples_")
    m = re.search(r"_(\d{4}-\d{2}-\d{2}T.*)$", name)
    if m:
        timestamp = m.group(1)
        task = name[: m.start()]
    else:
        timestamp = "unknown"
        task = name
    return task or stem, timestamp


def load_samples(jsonl_path: str) -> list[dict]:
    """Load all samples from a single JSONL file."""
    task, timestamp = _parse_sample_filename(jsonl_path)
    samples: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                s["_task"] = task
                s["_timestamp"] = timestamp
                samples.append(s)
    return samples


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def extract_thinking_and_answer(text: str) -> tuple[str, str]:
    """Split a response into *thinking* and *answer* parts.

    Everything inside ``<think>...</think>`` is the thinking portion; the
    remainder is the answer.
    """
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
    else:
        thinking = ""
        answer = text.strip()
    return thinking, answer


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from *text*."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_prompt_text(sample: dict) -> str:
    """Extract the prompt/question text from a sample document."""
    doc = sample.get("doc", {})
    for key in ("prompt", "question", "input"):
        if key in doc:
            return str(doc[key])
    return "(no prompt found)"


def get_passed(sample: dict) -> bool:
    """Extract pass/fail from a sample using common benchmark metric keys."""
    for key in ("prompt_level_strict_acc", "exact_match"):
        if key in sample:
            return bool(sample[key])
    # NIAH / ruler benchmarks use numeric length keys like "4096"
    for key, val in sample.items():
        if key.isdigit():
            return bool(val)
    return False


# ---------------------------------------------------------------------------
# Failure-mode detectors
# ---------------------------------------------------------------------------


def detect_death_loop(text: str, min_sentence_len: int = 5) -> dict:
    """Detect repeated sentences / phrases indicating a death loop.

    A sample is flagged as looping when *any* of the following hold:

    * duplicate sentence ratio > 0.3
    * any single sentence repeats >= 5 times
    * a contiguous block of >= 4 identical sentences exists
    """
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [s.strip() for s in raw if len(s.strip().split()) >= min_sentence_len]

    if not sentences:
        return {
            "is_looping": False,
            "loop_ratio": 0.0,
            "total_sentences": 0,
            "unique_sentences": 0,
            "duplicate_sentences": 0,
            "top_repeats": [],
            "longest_repeated_block": 0,
        }

    total = len(sentences)
    counts = Counter(sentences)
    unique = len(counts)
    duplicates = total - unique
    repeated = [(s, c) for s, c in counts.most_common(10) if c > 1]

    # Longest contiguous block of the same sentence
    max_block = 1
    cur_block = 1
    for i in range(1, len(sentences)):
        if sentences[i] == sentences[i - 1]:
            cur_block += 1
            max_block = max(max_block, cur_block)
        else:
            cur_block = 1

    loop_ratio = duplicates / total if total > 0 else 0.0
    max_repeat = repeated[0][1] if repeated else 0
    is_looping = loop_ratio > 0.3 or max_repeat >= 5 or max_block >= 4

    return {
        "is_looping": is_looping,
        "loop_ratio": round(loop_ratio, 4),
        "total_sentences": total,
        "unique_sentences": unique,
        "duplicate_sentences": duplicates,
        "top_repeats": repeated,
        "longest_repeated_block": max_block,
    }


def detect_missing_eos(
    response: str,
    gen_limit: int = 32768,
    eos_tokens: tuple[str, ...] = ("</s>", "<|endoftext|>", "<|im_end|>", "<|end|>"),
) -> bool:
    """Return ``True`` when the response appears to have hit the generation
    length limit without producing a termination token."""
    # Approximate: if the response is very close to the token limit in
    # characters (avg ~4 chars/token) and does not end with an EOS token.
    char_estimate = gen_limit * 4
    if len(response) < char_estimate * 0.9:
        return False
    return not any(response.rstrip().endswith(tok) for tok in eos_tokens)


def detect_empty_response(response: str) -> bool:
    """Return ``True`` when the response is empty after stripping thinking tags."""
    return strip_thinking(response).strip() == ""


def detect_abrupt_ending(answer: str) -> bool:
    """Return ``True`` when the answer ends without terminal punctuation."""
    text = answer.rstrip()
    if not text:
        return False
    # Allow common terminal characters
    terminal = {".", "!", "?", '"', "'", ")", "]", "}", ">", "*", "`", ":", ";"}
    return text[-1] not in terminal


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _percentiles(data: list[int | float]) -> dict[str, float]:
    """Return common percentile statistics for a sorted list of numbers."""
    if not data:
        return {k: 0.0 for k in ("min", "p25", "p50", "p75", "p95", "max", "mean")}
    s = sorted(data)
    n = len(s)
    return {
        "min": float(s[0]),
        "p25": float(s[n // 4]),
        "p50": float(s[n // 2]),
        "p75": float(s[3 * n // 4]),
        "p95": float(s[int(n * 0.95)]),
        "max": float(s[-1]),
        "mean": round(sum(s) / n, 2),
    }


def annotate_sample(sample: dict) -> dict:
    """Compute per-sample annotations (failure modes, lengths)."""
    resp = sample["filtered_resps"][0] if sample.get("filtered_resps") else sample["resps"][0][0]
    thinking, answer = extract_thinking_and_answer(resp)

    loop_info = detect_death_loop(answer)
    missing_eos = detect_missing_eos(resp)
    empty = detect_empty_response(resp)
    abrupt = detect_abrupt_ending(answer)

    failure_modes: list[str] = []
    if loop_info["is_looping"]:
        failure_modes.append("death_loop")
    if missing_eos:
        failure_modes.append("missing_eos")
    if empty:
        failure_modes.append("empty_response")
    if abrupt:
        failure_modes.append("abrupt_ending")

    prompt_text = get_prompt_text(sample)
    passed = get_passed(sample)

    return {
        "task": sample.get("_task", "unknown"),
        "doc_id": sample.get("doc_id"),
        "prompt": prompt_text,
        "passed": passed,
        "response": resp,
        "answer": answer,
        "thinking": thinking,
        "resp_words": len(resp.split()),
        "answer_words": len(answer.split()),
        "thinking_words": len(thinking.split()) if thinking else 0,
        "resp_chars": len(resp),
        "failure_modes": failure_modes,
        "loop_info": loop_info,
        "missing_eos": missing_eos,
        "empty_response": empty,
        "abrupt_ending": abrupt,
    }


def analyze_samples(
    samples: list[dict],
    word_threshold: int = 500,
) -> dict[str, Any]:
    """Annotate every sample and compute aggregate statistics."""
    records = [annotate_sample(s) for s in samples]
    total = len(records)
    if total == 0:
        return {"total": 0, "records": [], "length_stats": {}, "failure_counts": {}}

    # Length stats
    length_stats = {
        "full_response": _percentiles([r["resp_words"] for r in records]),
        "answer_only": _percentiles([r["answer_words"] for r in records]),
        "thinking_only": _percentiles([r["thinking_words"] for r in records]),
    }

    # Failure counts
    mode_names = ("death_loop", "missing_eos", "empty_response", "abrupt_ending")
    failure_counts: dict[str, int] = {m: 0 for m in mode_names}
    for r in records:
        for m in r["failure_modes"]:
            failure_counts[m] = failure_counts.get(m, 0) + 1
    failure_rates = {f"{m}_rate": round(failure_counts[m] / total, 6) for m in mode_names}

    # Long-response breakdown
    long = [r for r in records if r["answer_words"] > word_threshold]
    short = [r for r in records if r["answer_words"] <= word_threshold]

    # Overall accuracy
    n_passed = sum(1 for r in records if r["passed"])

    return {
        "total": total,
        "records": records,
        "length_stats": length_stats,
        "failure_counts": failure_counts,
        "failure_rates": failure_rates,
        "n_passed": n_passed,
        "accuracy": round(n_passed / total, 4),
        "word_threshold": word_threshold,
        "n_long": len(long),
        "n_short": len(short),
        "long_pass_rate": (round(sum(1 for r in long if r["passed"]) / len(long), 4) if long else None),
        "short_pass_rate": (round(sum(1 for r in short if r["passed"]) / len(short), 4) if short else None),
    }


# ---------------------------------------------------------------------------
# Task-level metrics from results JSON
# ---------------------------------------------------------------------------


def load_task_metrics(results_dir: Path | None) -> list[dict]:
    """Load ``results_*.json`` files and extract per-task metrics."""
    if results_dir is None or not results_dir.is_dir():
        return []
    result_files = sorted(results_dir.glob("results_*.json"))
    all_metrics: list[dict] = []
    for rf in result_files:
        with open(rf) as f:
            data = json.load(f)
        results = data.get("results", {})
        for task_name, task_data in sorted(results.items()):
            n = task_data.get("sample_len", task_data.get("sample_count"))
            skip = {"name", "alias", "sample_len", "sample_count"}
            for key, val in sorted(task_data.items()):
                if key in skip or "stderr" in key:
                    continue
                if isinstance(val, (int, float)):
                    stderr_key = key.replace(key.split(",")[0], key.split(",")[0] + "_stderr", 1)
                    stderr = task_data.get(stderr_key)
                    all_metrics.append(
                        {
                            "file": rf.name,
                            "task": task_name,
                            "n_samples": n,
                            "metric": key,
                            "value": val,
                            "stderr": stderr,
                        }
                    )
    return all_metrics


# ---------------------------------------------------------------------------
# Reporting (stdout)
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


def report_length_stats(stats: dict) -> None:
    """Print response-length percentiles."""
    _section(f"RESPONSE LENGTH ANALYSIS ({stats['total']} samples)")
    for label, key in [
        ("Full Response (words)", "full_response"),
        ("Answer Only (words, excluding <think>)", "answer_only"),
        ("Thinking Only (words, inside <think>)", "thinking_only"),
    ]:
        print(f"\n--- {label} ---")
        for k, v in stats["length_stats"][key].items():
            print(f"  {k:>5}: {v:>10.1f}")


def report_failure_modes(stats: dict) -> None:
    """Print per-failure-mode counts and rates."""
    _section("FAILURE MODE DETECTION")
    total = stats["total"]
    print(f"\n  {'Mode':<20} | {'Count':>6} | {'Rate':>8}")
    print("  " + "-" * 42)
    for mode in ("death_loop", "missing_eos", "empty_response", "abrupt_ending"):
        count = stats["failure_counts"][mode]
        rate = stats["failure_rates"][f"{mode}_rate"]
        print(f"  {mode:<20} | {count:>6} | {rate:>7.4f}")

    # Show any-failure summary
    any_failure = sum(1 for r in stats["records"] if r["failure_modes"])
    print(f"\n  Samples with at least one failure: {any_failure} / {total} ({100 * any_failure / total:.1f}%)")


def report_death_loop_details(stats: dict, top_n: int = 10) -> None:
    """Print details of death-looping samples."""
    looping = [r for r in stats["records"] if "death_loop" in r["failure_modes"]]
    if not looping:
        return

    _section("DEATH LOOP DETAILS")
    looping_sorted = sorted(looping, key=lambda r: r["loop_info"]["loop_ratio"], reverse=True)

    print(
        f"\n  {'task':>20} | {'doc_id':>6} | {'words':>6} | {'sents':>5} | "
        f"{'uniq':>5} | {'dup%':>5} | {'max_blk':>7} | {'status':>6}"
    )
    print("  " + "-" * 80)

    for r in looping_sorted[:top_n]:
        li = r["loop_info"]
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['task']:>20} | {r['doc_id']:>6} | {r['answer_words']:>6} | "
            f"{li['total_sentences']:>5} | {li['unique_sentences']:>5} | "
            f"{100 * li['loop_ratio']:>4.1f}% | {li['longest_repeated_block']:>7} | "
            f"{status:>6}"
        )

    worst = looping_sorted[0]
    if worst["loop_info"]["top_repeats"]:
        sent, cnt = worst["loop_info"]["top_repeats"][0]
        sent_trunc = (sent[:100] + "...") if len(sent) > 100 else sent
        print(f"\n  Worst offender (doc_id={worst['doc_id']}): sentence repeated {cnt}x:")
        print(f'    "{sent_trunc}"')


def report_accuracy(stats: dict) -> None:
    """Print accuracy breakdown by response length."""
    _section("ACCURACY BREAKDOWN")
    total = stats["total"]
    wt = stats["word_threshold"]

    print(f"\n  Overall: {stats['n_passed']} / {total} ({100 * stats['accuracy']:.1f}%)")

    n_long = stats["n_long"]
    n_short = stats["n_short"]
    if n_long:
        print(f"\n  Long responses (answer > {wt} words): {n_long} / {total} ({100 * n_long / total:.1f}%)")
        print(f"    Pass rate: {100 * stats['long_pass_rate']:.1f}%")
    if n_short:
        print(f"\n  Short responses (answer <= {wt} words): {n_short} / {total} ({100 * n_short / total:.1f}%)")
        print(f"    Pass rate: {100 * stats['short_pass_rate']:.1f}%")


def report_task_metrics(metrics: list[dict]) -> None:
    """Print task-level metrics from results JSON files."""
    if not metrics:
        return
    _section("TASK-LEVEL ACCURACY SUMMARY (from results JSON)")
    print(f"\n  {'task':<25} | {'samples':>7} | {'metric':<35} | {'value':>8} | {'stderr':>8}")
    print("  " + "-" * 95)
    for m in metrics:
        n_str = str(m["n_samples"]) if m["n_samples"] is not None else "?"
        stderr_str = f"{m['stderr']:>8.4f}" if isinstance(m["stderr"], (int, float)) else f"{'N/A':>8}"
        print(f"  {m['task']:<25} | {n_str:>7} | {m['metric']:<35} | {m['value']:>8.4f} | {stderr_str}")


def report_top_longest(stats: dict, top_n: int = 10) -> None:
    """Print the N longest responses by answer word count."""
    records_sorted = sorted(stats["records"], key=lambda r: r["answer_words"], reverse=True)
    display = records_sorted[:top_n]
    if not display:
        return

    _section(f"TOP {len(display)} LONGEST RESPONSES (by answer words)")
    for i, r in enumerate(display, 1):
        status = "PASS" if r["passed"] else "FAIL"
        modes = ", ".join(r["failure_modes"]) if r["failure_modes"] else "none"
        prompt_short = (r["prompt"][:80] + "...") if len(r["prompt"]) > 80 else r["prompt"]
        print(
            f"\n  [{i}] task={r['task']} doc_id={r['doc_id']} | "
            f"answer={r['answer_words']}w thinking={r['thinking_words']}w | "
            f"{status} | failures: {modes}"
        )
        print(f"      Prompt: {prompt_short}")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def build_export(stats: dict, task_metrics: list[dict]) -> dict:
    """Build a structured JSON-serializable dict for CI consumption."""
    # Per-sample annotations (strip large text fields for size)
    per_sample = []
    for r in stats["records"]:
        per_sample.append(
            {
                "task": r["task"],
                "doc_id": r["doc_id"],
                "passed": r["passed"],
                "resp_words": r["resp_words"],
                "answer_words": r["answer_words"],
                "thinking_words": r["thinking_words"],
                "failure_modes": r["failure_modes"],
                "loop_ratio": r["loop_info"]["loop_ratio"],
                "longest_repeated_block": r["loop_info"]["longest_repeated_block"],
                "missing_eos": r["missing_eos"],
                "empty_response": r["empty_response"],
                "abrupt_ending": r["abrupt_ending"],
            }
        )

    return {
        "summary": {
            "total_samples": stats["total"],
            "accuracy": stats["accuracy"],
            "n_passed": stats["n_passed"],
            "word_threshold": stats["word_threshold"],
            "n_long": stats["n_long"],
            "n_short": stats["n_short"],
            "long_pass_rate": stats["long_pass_rate"],
            "short_pass_rate": stats["short_pass_rate"],
        },
        "length_stats": stats["length_stats"],
        "failure_counts": stats["failure_counts"],
        "failure_rates": stats["failure_rates"],
        "task_metrics": task_metrics,
        "samples": per_sample,
    }


# ---------------------------------------------------------------------------
# Threshold gating
# ---------------------------------------------------------------------------


def load_thresholds(path: str) -> dict[str, float]:
    """Load quality gate thresholds from a YAML file.

    Expected format (all keys optional)::

        death_loop_rate: 0.01
        empty_response_rate: 0.005
        missing_eos_rate: 0.02
        abrupt_ending_rate: 0.05
    """
    try:
        import yaml
    except ImportError:
        # Fall back to simple key: value parsing for environments without PyYAML
        thresholds: dict[str, float] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, val = line.partition(":")
                thresholds[key.strip()] = float(val.strip())
        return thresholds

    with open(path) as f:
        return yaml.safe_load(f) or {}


def check_thresholds(stats: dict, thresholds: dict[str, float]) -> list[str]:
    """Compare failure rates against thresholds.  Returns list of violations."""
    violations: list[str] = []
    for key, max_val in thresholds.items():
        rate_key = key if key.endswith("_rate") else f"{key}_rate"
        actual = stats["failure_rates"].get(rate_key)
        if actual is not None and actual > max_val:
            violations.append(f"{rate_key}={actual:.4f} exceeds threshold {max_val}")
    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the lm_eval quality analysis CLI."""

    parser = argparse.ArgumentParser(
        description="Analyze inference quality from lm_eval benchmark outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s results/\n"
            "  %(prog)s results/ --word-threshold 300 --export report.json\n"
            "  %(prog)s results/ --threshold-file thresholds.yaml\n"
        ),
    )
    parser.add_argument(
        "input",
        help="Path to a samples JSONL file or a directory containing lm_eval outputs.",
    )
    parser.add_argument(
        "--word-threshold",
        type=int,
        default=500,
        help="Word count above which an answer is considered 'long' (default: 500).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of longest / worst responses to display (default: 10).",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Write structured JSON report to this path.",
    )
    parser.add_argument(
        "--threshold-file",
        type=str,
        default=None,
        help="YAML file with pass/fail thresholds. Exit non-zero on violations.",
    )
    args = parser.parse_args(argv)

    # --- Discover files ---
    jsonl_files = resolve_jsonl_files(args.input)
    results_dir = resolve_results_dir(args.input)
    print(f"Found {len(jsonl_files)} sample JSONL file(s) in: {args.input}")

    # --- Load samples ---
    all_samples: list[dict] = []
    for jf in jsonl_files:
        print(f"  Loading: {jf}")
        all_samples.extend(load_samples(str(jf)))
    print(f"Loaded {len(all_samples)} total samples.")

    if not all_samples:
        print("No samples found. Nothing to analyze.")
        return 0

    # --- Analyze ---
    stats = analyze_samples(all_samples, word_threshold=args.word_threshold)

    # --- Report ---
    report_length_stats(stats)
    report_failure_modes(stats)
    report_death_loop_details(stats, top_n=args.top_n)
    report_accuracy(stats)
    report_top_longest(stats, top_n=args.top_n)

    task_metrics = load_task_metrics(results_dir)
    report_task_metrics(task_metrics)

    # --- Export ---
    if args.export:
        export_data = build_export(stats, task_metrics)
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\nExported structured report to {export_path}")

    # --- Threshold gating ---
    if args.threshold_file:
        thresholds = load_thresholds(args.threshold_file)
        violations = check_thresholds(stats, thresholds)
        if violations:
            print(f"\nQUALITY GATE FAILED -- {len(violations)} violation(s):")
            for v in violations:
                print(f"  - {v}")
            return 1
        else:
            print("\nQuality gate passed -- all metrics within thresholds.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
