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

"""Results of LIBERO evaluations: paired before/after summaries and the learning-curve figure.

Both commands read evaluation trees of ``run_libero_eval.py`` (one JSON record per episode under
``suite*/<task>/``; the reference simulator's ``run_sim_eval.py`` writes the same layout).

``python libero_results.py summarize BASE_DIR AFTER_DIR [AFTER_DIR2 ...]`` pairs the episodes of every
AFTER_DIR with those of BASE_DIR by task, initial state (or scene seed) and sampler seed, and prints for
every task and in total the successes before and after, the pairs won and lost, the exact McNemar p and
the paired difference with its 95 % interval. It needs only the standard library.

``python libero_results.py curve --point 0=eval/libero_spatial_base --point 100=eval/<run>_100 ... --out <png>``
plots the success rate of the checkpoints of one run against the optimizer step (step 0 for the base
checkpoint); ``--extra name=step=eval_dir`` adds a single marker, e.g. a LoRA control. Error bars are
95 % Wilson intervals over the episodes; the table is printed in Markdown. It needs matplotlib.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

EpisodeKey = tuple[str, int, int]


class EpisodeRecord(TypedDict, total=False):
    """The fields of an episode record this script reads."""

    success: bool
    progress: float
    scene_seed: int
    seed: int
    sampler_seed: int


def load(root: str) -> dict[EpisodeKey, EpisodeRecord]:
    """Episode records of an evaluation tree keyed by (task, initial state or scene seed, sampler seed)."""
    out: dict[EpisodeKey, EpisodeRecord] = {}
    for path in glob.glob(f"{root}/suite*/*/episode_*.json"):
        with open(path) as f:
            record: EpisodeRecord = json.load(f)
        task = path.split("/")[-2]
        seed = int(record.get("scene_seed", record.get("seed", path.split("_")[-1].split(".")[0])))
        out[(task, seed, int(record.get("sampler_seed", 0)))] = record
    return out


def mcnemar_p(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value of ``wins`` against ``losses`` discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(base_dir: str, after_dirs: list[str]) -> None:
    """Print the paired summaries of every ``after_dir`` against ``base_dir``."""
    base = load(base_dir)
    for after_dir in after_dirs:
        after = load(after_dir)
        keys = sorted(set(base) & set(after))
        print(f"\n=== {base_dir}  vs  {after_dir}  ({len(keys)} paired episodes)")
        print(f"{'task':24s} {'before':>7s} {'after':>7s} {'won':>4s} {'lost':>5s} {'p':>7s}  prog_before prog_after")
        per_task: dict[str, list[EpisodeKey]] = defaultdict(list)
        for key in keys:
            per_task[key[0]].append(key)
        tot_b = tot_a = tot_w = tot_l = 0
        diffs: list[int] = []
        for task, ks in sorted(per_task.items()):
            b = sum(bool(base[k]["success"]) for k in ks)
            a = sum(bool(after[k]["success"]) for k in ks)
            won = sum((not base[k]["success"]) and after[k]["success"] for k in ks)
            lost = sum(base[k]["success"] and (not after[k]["success"]) for k in ks)
            pb = sum(float(base[k].get("progress", 0)) for k in ks) / len(ks)
            pa = sum(float(after[k].get("progress", 0)) for k in ks) / len(ks)
            print(
                f"{task:24s} {b:3d}/{len(ks):<3d} {a:3d}/{len(ks):<3d} {won:4d} {lost:5d} {mcnemar_p(won, lost):7.3f}"
                f"  {pb:11.3f} {pa:10.3f}"
            )
            tot_b += b
            tot_a += a
            tot_w += won
            tot_l += lost
            diffs += [int(bool(after[k]["success"])) - int(bool(base[k]["success"])) for k in ks]
        n = len(diffs)
        mean = sum(diffs) / n
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(1, n - 1))
        ci = 1.96 * sd / math.sqrt(n)
        print(
            f"{'TOTAL':24s} {tot_b:3d}/{n:<3d} {tot_a:3d}/{n:<3d} {tot_w:4d} {tot_l:5d} {mcnemar_p(tot_w, tot_l):7.4f}"
        )
        print(
            f"success rate {100 * tot_b / n:.1f}% -> {100 * tot_a / n:.1f}%  "
            f"paired difference {100 * mean:+.1f} pts (95% CI +-{100 * ci:.1f})"
        )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval of ``k`` successes in ``n`` trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def read_summary(path: Path) -> tuple[int, int]:
    """``(successes, episodes)`` of an evaluation directory, counted over its episode records."""
    records = load(str(path))
    if not records:
        raise SystemExit(f"no episode records under {path}")
    return sum(bool(record["success"]) for record in records.values()), len(records)


def row(label: str, k: int, n: int) -> str:
    """One Markdown table row."""
    lo, hi = wilson(k, n)
    return f"| {label} | {k}/{n} ({100 * k / n:.1f} %) | {100 * lo:.1f} - {100 * hi:.1f} % |"


def curve(args: argparse.Namespace) -> None:
    """Print the table and write the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = sorted((int(s), Path(d)) for s, d in (p.split("=", 1) for p in args.point))
    rows = [(s, *read_summary(d)) for s, d in points]
    if args.expect_episodes and any(n != args.expect_episodes for _, _, n in rows):
        raise SystemExit(f"incomplete evaluation: episodes per point {[n for _, _, n in rows]}")
    print("| optimizer step | success | 95 % interval |")
    print("|---|---|---|")
    for s, k, n in rows:
        print(row("base (0)" if s == 0 else str(s), k, n))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = [s for s, _, _ in rows]
    y = [100 * k / n for _, k, n in rows]
    err = [
        [100 * (k / n - wilson(k, n)[0]) for _, k, n in rows],
        [100 * (wilson(k, n)[1] - k / n) for _, k, n in rows],
    ]
    ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=args.label)
    for extra in args.extra:
        name, step, d = extra.split("=", 2)
        k, n = read_summary(Path(d))
        if args.expect_episodes and n != args.expect_episodes:
            raise SystemExit(f"incomplete evaluation {d}: {n} episodes")
        lo, hi = wilson(k, n)
        ax.errorbar(
            [int(step)],
            [100 * k / n],
            yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]],
            marker="s",
            capsize=3,
            linestyle="none",
            label=name,
        )
        print(row(f"{name}, step {step}", k, n))
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("wrote", args.out)


def main() -> int:
    """Dispatch ``summarize`` or ``curve``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = ap.add_subparsers(dest="command", required=True)
    summary = commands.add_parser(
        "summarize", help="paired before/after summary of evaluation trees with the same episodes"
    )
    summary.add_argument("base_dir")
    summary.add_argument("after_dirs", nargs="+")
    figure = commands.add_parser("curve", help="success rate of the checkpoints of one run against the optimizer step")
    figure.add_argument(
        "--point", action="append", required=True, help="step=eval_dir (step 0 for the base checkpoint)"
    )
    figure.add_argument("--label", default="full fine-tune")
    figure.add_argument("--extra", action="append", default=[], help="name=step=eval_dir, drawn as a single marker")
    figure.add_argument("--out", required=True)
    figure.add_argument(
        "--expect-episodes", type=int, default=0, help="fail unless every point has this many episodes (0: no check)"
    )
    args = ap.parse_args()
    if args.command == "summarize":
        summarize(args.base_dir, args.after_dirs)
    else:
        curve(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
