# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EXPERIMENT ONLY (perf-gap trace; not for merge): summarize one profiled PP step from a chrome trace."""

import gzip
import json
import re


def _union(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def _overlap(merged, a, b):
    total = 0.0
    for start, end in merged:
        if end <= a:
            continue
        if start >= b:
            break
        total += min(end, b) - max(start, a)
    return total


def summarize(path: str, rank: int) -> list[str]:
    """Busy/idle per schedule phase, NCCL time and the largest GPU idle gaps (one line each)."""
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        events = json.load(f)
    events = events["traceEvents"] if isinstance(events, dict) else events
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    nccl = [e for e in kernels if "nccl" in e["name"].lower()]
    compute = [e for e in kernels if "nccl" not in e["name"].lower()]
    actions = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" and e["name"].startswith("PP:")
    ]
    t0 = min(e["ts"] for e in kernels)
    t1 = max(e["ts"] + e["dur"] for e in kernels)
    busy = _union([(e["ts"], e["ts"] + e["dur"]) for e in compute])

    def kind(name):
        m = re.search(r"\d+([FBIW])\d+$", name)
        return m.group(1) if m else "?"

    fwd = [a for a in actions if kind(a["name"]) == "F"]
    bwd = [a for a in actions if kind(a["name"]) in "BI"]
    first_b = min((a["ts"] for a in bwd), default=t0)
    last_f = max((a["ts"] + a["dur"] for a in fwd), default=t1)
    first_a = min((a["ts"] for a in actions), default=t0)
    last_a = max((a["ts"] + a["dur"] for a in actions), default=t1)
    lines = [
        f"EXP_TRACE rank={rank} window={(t1 - t0) / 1e6:.3f}s busy={_overlap(busy, t0, t1) / 1e6:.3f}s "
        f"idle={(t1 - t0 - _overlap(busy, t0, t1)) / 1e6:.3f}s actions={len(actions)}"
    ]
    for name, a, b in (
        ("pre", t0, first_a),
        ("warmup", first_a, first_b),
        ("steady", first_b, last_f),
        ("cooldown", last_f, last_a),
        ("post", last_a, t1),
    ):
        if b > a:
            bb = _overlap(busy, a, b)
            lines.append(
                f"EXP_TRACE rank={rank} phase={name} dur={(b - a) / 1e6:.3f}s busy={bb / 1e6:.3f}s idle={(b - a - bb) / 1e6:.3f}s"
            )
    per_nccl = {}
    for e in nccl:
        n = re.sub(r"\(.*", "", e["name"])[:48]
        per_nccl[n] = per_nccl.get(n, 0) + e["dur"]
    lines.append(
        f"EXP_TRACE rank={rank} nccl "
        + ", ".join(f"{n}={d / 1e6:.2f}s" for n, d in sorted(per_nccl.items(), key=lambda x: -x[1])[:6])
    )
    gaps = []
    for (s0, e0), (s1, e1) in zip(busy, busy[1:]):
        if s1 - e0 > 2000:
            before = [a["name"] for a in actions if a["ts"] <= e0 <= a["ts"] + a["dur"]]
            after = [a["name"] for a in actions if a["ts"] <= s1 <= a["ts"] + a["dur"]]
            gaps.append((s1 - e0, (e0 - t0) / 1e6, before[:1], after[:1]))
    gaps.sort(reverse=True)
    total_gap = sum(g[0] for g in gaps) / 1e6
    lines.append(
        f"EXP_TRACE rank={rank} gaps>2ms total={total_gap:.3f}s n={len(gaps)} top: "
        + "; ".join(f"{g / 1e3:.0f}ms@{t:.2f}s {b}->{a}" for g, t, b, a in gaps[:12])
    )
    per_kernel = {}
    for e in kernels:
        n = re.sub(r"<.*", "", e["name"])[:70]
        per_kernel[n] = per_kernel.get(n, 0) + e["dur"]
    for n, d in sorted(per_kernel.items(), key=lambda x: -x[1])[:30]:
        lines.append(f"EXP_TRACE rank={rank} kernel {d / 1e3:9.1f}ms {n}")
    return lines
