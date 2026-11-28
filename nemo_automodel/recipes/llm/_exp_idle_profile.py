# EXPERIMENT ONLY (perf-gap idle attribution; not for merge).
"""Profile one iteration and attribute GPU compute-stream idle to what each host thread was doing."""

import collections
import json
import os
import re

import torch

_COMM = re.compile(r"nccl|deep_ep|hybrid_ep", re.I)


def start():
    torch.cuda.synchronize()
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    )
    prof.__enter__()
    return prof


def _union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def finish(prof, tag="EXP_IDLE"):
    torch.cuda.synchronize()
    prof.__exit__(None, None, None)
    rows = sorted(
        ((e.key, e.self_cpu_time_total, e.count) for e in prof.key_averages() if e.self_cpu_time_total > 0),
        key=lambda r: -r[1],
    )
    for k, t, c in rows[:35]:
        print(f"{tag} cpu_self {t / 1e3:9.1f} ms n={c:6d} {k[:100]}", flush=True)
    path = f"/tmp/exp_idle_{os.getpid()}.json"
    prof.export_chrome_trace(path)
    with open(path) as f:
        ev = json.load(f)
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    kernels = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"]
    comp = [e for e in kernels if not _COMM.search(e["name"])]
    if not comp:
        print(f"{tag} no compute kernels", flush=True)
        return
    busy = _union([(e["ts"], e["ts"] + e["dur"]) for e in comp])
    t0, t1 = busy[0][0], busy[-1][1]
    host = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver", "user_annotation"):
            host[e.get("tid")].append(e)
    for v in host.values():
        v.sort(key=lambda e: e["ts"])
    per = collections.Counter()
    cnt = collections.Counter()
    total = 0.0
    for (s0, e0), (s1, e1) in zip(busy, busy[1:]):
        gap = s1 - e0
        if gap < 200:
            continue
        total += gap
        t = e0 + 1
        labels = []
        for tid, lst in host.items():
            active = [e for e in lst if e["ts"] <= t <= e["ts"] + e["dur"]]
            if not active:
                continue
            inner = min(active, key=lambda e: e["dur"])
            outer_ann = [e["name"] for e in active if e.get("cat") == "user_annotation"]
            labels.append(f"{inner['name'][:60]} | {outer_ann[0][:40] if outer_ann else '-'}")
        key = " || ".join(sorted(labels)) if labels else "(no host op active)"
        per[key] += gap
        cnt[key] += 1
    print(
        f"{tag} window={(t1 - t0) / 1e3:.1f} ms busy={sum(e - s for s, e in busy) / 1e3:.1f} ms gaps>=0.2ms={total / 1e3:.1f} ms",
        flush=True,
    )
    for k, v in per.most_common(30):
        print(f"{tag} gap {v / 1e3:8.1f} ms n={cnt[k]:5d} {k[:230]}", flush=True)
    os.remove(path)
