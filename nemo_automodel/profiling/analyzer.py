"""Standalone step-level profiling with automatic analysis.

Usage::

    from nemo_automodel.profiling import StepProfiler

    profiler = StepProfiler(active_step=2, num_steps=10)

    for step in range(10):
        with profiler.step(step):
            forward_backward()
            optimizer_step()

    profiler.report()          # prints to logger
    profiler.export("trace")   # writes trace.json for chrome://tracing
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _device_time(e) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def _cpu_time(e) -> float:
    return float(getattr(e, "self_cpu_time_total", 0) or 0)


class StepProfiler:
    """Lightweight profiler that captures one step and produces a full analysis.

    Args:
        active_step:  Which step index to actually profile (0-based).
        num_steps:    Total steps (only needed if you call ``step()`` in a loop).
        record_shapes: Record tensor shapes (adds overhead but useful for analysis).
        with_stack:   Capture Python call stacks on every op.
        with_flops:   Estimate FLOPs per operator.
        profile_memory: Track tensor memory allocations.
        rank:         Distributed rank.  ``None`` = auto-detect.
        report_rank:  Only print the report on this rank (default 0).
        on_trace_ready: Optional callback ``(profiler) -> None`` after profiling.
        top_n:        Max rows per report section.
        straggler_min_calls: Minimum calls before an op is eligible for straggler check.
        straggler_ratio:     min max/mean ratio to flag as straggler.
    """

    def __init__(
        self,
        active_step: int = 0,
        num_steps: int | None = None,
        record_shapes: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        profile_memory: bool = True,
        rank: int | None = None,
        report_rank: int = 0,
        on_trace_ready: Callable | None = None,
        top_n: int = 20,
        straggler_min_calls: int = 4,
        straggler_ratio: float = 3.0,
    ):
        self._active_step = active_step
        self._num_steps = num_steps
        self._record_shapes = record_shapes
        self._with_stack = with_stack
        self._with_flops = with_flops
        self._profile_memory = profile_memory
        self._on_trace_ready = on_trace_ready
        self._top_n = top_n
        self._straggler_min_calls = straggler_min_calls
        self._straggler_ratio = straggler_ratio

        if rank is not None:
            self._rank = rank
        elif dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
        else:
            self._rank = 0
        self._report_rank = report_rank

        self._prof: torch.profiler.profile | None = None
        self._current_step = -1
        self._step_wall_us: float = 0.0
        self._entered = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "StepProfiler":
        """Manually start the profiler (alternative to using ``step()`` context)."""
        self._prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=self._record_shapes,
            with_stack=self._with_stack,
            with_flops=self._with_flops,
            profile_memory=self._profile_memory,
            schedule=torch.profiler.schedule(
                wait=self._active_step,
                warmup=0,
                active=1,
                repeat=1,
            ),
            on_trace_ready=self._on_trace_ready,
        )
        self._prof.__enter__()
        self._entered = True
        return self

    def stop(self) -> None:
        """Stop the profiler."""
        if self._prof is not None and self._entered:
            self._prof.__exit__(None, None, None)
            self._entered = False

    @contextmanager
    def step(self, step_idx: int):
        """Context manager wrapping a single training step.

        The profiler is created lazily on the first call and torn down
        after the active step completes.
        """
        if self._prof is None:
            self.start()

        self._current_step = step_idx

        import time
        t0 = time.monotonic()
        try:
            yield
        finally:
            if step_idx == self._active_step:
                self._step_wall_us = (time.monotonic() - t0) * 1e6
            if self._prof is not None:
                self._prof.step()
            if step_idx == self._active_step:
                self.stop()

    def report(self, log_fn: Callable[..., None] | None = None) -> str:
        """Generate and print the full analysis report.

        Returns the report as a string regardless of rank.
        """
        if self._prof is None:
            return ""
        if self._entered:
            self.stop()

        lines = self._build_report()
        text = "\n".join(lines)

        if self._rank == self._report_rank and lines:
            _log = log_fn or logger.info
            for line in lines:
                _log(line)

        return text

    def export(self, path_prefix: str) -> str | None:
        """Export a Chrome trace JSON (only on report_rank)."""
        if self._prof is None or self._rank != self._report_rank:
            return None
        if self._entered:
            self.stop()
        out = f"{path_prefix}.json"
        self._prof.export_chrome_trace(out)
        logger.info("Exported trace to %s", out)
        return out

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(self) -> list[str]:
        events = list(self._prof.events())
        if not events:
            return []

        total_cpu_us = sum(_cpu_time(e) for e in events)
        total_device_us = sum(_device_time(e) for e in events)
        wall_us = self._step_wall_us or total_cpu_us

        lines: list[str] = []
        W = 80

        def hdr(title: str):
            lines.append("")
            lines.append("=" * W)
            lines.append(f"[{title}]  (profiled step {self._active_step},  wall={wall_us / 1e6:.3f}s)")
            lines.append("-" * W)

        def footer():
            lines.append("=" * W)

        # ----------------------------------------------------------
        # 1. Comm vs Compute CUDA breakdown
        # ----------------------------------------------------------
        comm_keys = ("nccl", "c10d", "all_reduce", "reduce_scatter", "all_gather")
        key_device: dict[str, float] = defaultdict(float)
        key_count: dict[str, int] = defaultdict(int)
        for e in events:
            d = _device_time(e)
            key_device[e.name] += d
            key_count[e.name] += 1

        comm_rows = [(k, v, key_count[k]) for k, v in key_device.items()
                     if any(ck in k.lower() for ck in comm_keys)]
        compute_rows = [(k, v, key_count[k]) for k, v in key_device.items()
                        if k.startswith("aten::") and v > 0]
        module_rows = [(k, v, key_count[k]) for k, v in key_device.items()
                       if ": " in k and v > 0]
        comm_total = sum(v for _, v, _ in comm_rows)
        compute_total = sum(v for _, v, _ in compute_rows)

        hdr("Comm vs Compute  (CUDA self-time)")
        lines.append(f"  {'Op':<55s}  {'CUDA us':>12s}  {'%wall':>6s}  {'calls':>7s}")
        for k, v, c in sorted(comm_rows, key=lambda x: -x[1]):
            lines.append(f"  {k:<55s}  {v:12.1f}  {100*v/wall_us:5.1f}%  {c:7d}")

        if module_rows:
            lines.append(f"  {'--- modules (autonvtx) ---':<55s}")
            for k, v, c in sorted(module_rows, key=lambda x: -x[1])[:self._top_n]:
                lines.append(f"  {k:<55s}  {v:12.1f}  {100*v/wall_us:5.1f}%  {c:7d}")

        lines.append(f"  {'--- compute (aten) ---':<55s}")
        for k, v, c in sorted(compute_rows, key=lambda x: -x[1])[:10]:
            lines.append(f"  {k:<55s}  {v:12.1f}  {100*v/wall_us:5.1f}%  {c:7d}")

        lines.append("-" * W)
        lines.append(
            f"  Comm CUDA: {comm_total:,.0f} us ({100*comm_total/wall_us:.1f}% wall)  "
            f"Compute CUDA: {compute_total:,.0f} us ({100*compute_total/wall_us:.1f}% wall)  "
            f"Overlap ratio: {100*comm_total/(compute_total+1e-9):.1f}%"
        )
        footer()

        # ----------------------------------------------------------
        # 2. CUDA runtime overhead
        # ----------------------------------------------------------
        rt: dict[str, list[float, int]] = defaultdict(lambda: [0.0, 0])
        for e in events:
            c = _cpu_time(e)
            if c > 0 and (e.name.startswith("cuda") or e.name.startswith("cu")):
                rt[e.name][0] += c
                rt[e.name][1] += 1

        if rt:
            total_rt = sum(v[0] for v in rt.values())
            hdr("CUDA Runtime Overhead  (CPU-side)")
            lines.append(f"  {'Runtime call':<55s}  {'CPU us':>12s}  {'%wall':>6s}  {'calls':>7s}")
            for name, (us, cnt) in sorted(rt.items(), key=lambda x: -x[1][0]):
                lines.append(f"  {name:<55s}  {us:12.1f}  {100*us/wall_us:5.1f}%  {cnt:7d}")
            lines.append("-" * W)
            lines.append(f"  Total CUDA runtime CPU cost: {total_rt:,.0f} us  ({100*total_rt/wall_us:.1f}% wall)")
            footer()

        # ----------------------------------------------------------
        # 3. Top CPU-stalling ops with Python stacks
        # ----------------------------------------------------------
        stacked: dict[str, dict] = defaultdict(lambda: {"cpu_us": 0.0, "count": 0, "stacks": defaultdict(float)})
        for e in events:
            c = _cpu_time(e)
            stack = getattr(e, "stack", None)
            if c > 100 and stack:
                rec = stacked[e.name]
                rec["cpu_us"] += c
                rec["count"] += 1
                user = [f for f in stack if "site-packages" not in f and "/lib/" not in f]
                if not user:
                    user = stack[:4]
                key = "\n            ".join(user[:5])
                rec["stacks"][key] += c

        if stacked:
            hdr("CPU Stalls  (top ops by CPU self-time, with Python stacks)")
            for name, rec in sorted(stacked.items(), key=lambda x: -x[1]["cpu_us"])[:self._top_n]:
                pct = 100 * rec["cpu_us"] / wall_us
                lines.append(f"  {name:<50s}  {rec['cpu_us']:10.1f} us  ({pct:4.1f}% wall)  x{rec['count']}")
                for sk, sv in sorted(rec["stacks"].items(), key=lambda x: -x[1])[:2]:
                    lines.append(f"        [{sv:.1f} us]\n            {sk}")
            footer()

        # ----------------------------------------------------------
        # 4. Straggler detection (CPU + CUDA)
        # ----------------------------------------------------------
        per_cpu: dict[str, list[float]] = defaultdict(list)
        per_device: dict[str, list[float]] = defaultdict(list)
        for e in events:
            c = _cpu_time(e)
            d = _device_time(e)
            if c > 0 or d > 0:
                per_cpu[e.name].append(c)
                per_device[e.name].append(d)

        def _stragglers(by_name: dict[str, list[float]]) -> list[tuple]:
            out = []
            for name, times in by_name.items():
                n = len(times)
                if n < self._straggler_min_calls:
                    continue
                mean = sum(times) / n
                mx = max(times)
                if mean > 0 and mx / mean > self._straggler_ratio and mx > 100:
                    std = math.sqrt(sum((t - mean) ** 2 for t in times) / n)
                    p99 = sorted(times)[int(n * 0.99)]
                    out.append((name, mean, mx, mx / mean, std, p99, n))
            return out

        cpu_sg = _stragglers(per_cpu)
        dev_sg = _stragglers(per_device)

        if cpu_sg or dev_sg:
            hdr("Stragglers  (ops with max >> mean)")
            if cpu_sg:
                lines.append("  --- CPU self-time ---")
                lines.append(f"  {'Op':<45s}  {'mean us':>10s}  {'max us':>10s}  {'p99 us':>10s}  {'std us':>10s}  {'ratio':>7s}  {'calls':>6s}")
                for name, mean, mx, ratio, std, p99, cnt in sorted(cpu_sg, key=lambda x: -x[2])[:self._top_n]:
                    lines.append(f"  {name:<45s}  {mean:10.1f}  {mx:10.1f}  {p99:10.1f}  {std:10.1f}  {ratio:6.1f}x  {cnt:6d}")
            if dev_sg:
                lines.append("  --- CUDA self-time ---")
                lines.append(f"  {'Op':<45s}  {'mean us':>10s}  {'max us':>10s}  {'p99 us':>10s}  {'std us':>10s}  {'ratio':>7s}  {'calls':>6s}")
                for name, mean, mx, ratio, std, p99, cnt in sorted(dev_sg, key=lambda x: -x[2])[:self._top_n]:
                    lines.append(f"  {name:<45s}  {mean:10.1f}  {mx:10.1f}  {p99:10.1f}  {std:10.1f}  {ratio:6.1f}x  {cnt:6d}")
            footer()

        # ----------------------------------------------------------
        # 5. Suspicious ops (hidden syncs, casts, extra work)
        # ----------------------------------------------------------
        suspicious_patterns = {
            "aten::item": "GPU→CPU sync (scalar fetch)",
            "aten::_local_scalar_dense": "GPU→CPU sync (scalar materialization)",
            "aten::nonzero": "GPU→CPU sync (dynamic shape)",
            "aten::where": "potential dynamic shape",
            "aten::to": "dtype/device cast",
            "aten::_to_copy": "hidden dtype cast / device transfer",
            "aten::clone": "unexpected tensor copy",
            "aten::contiguous": "memory layout conversion",
            "aten::empty": "allocation (may trigger cudaMalloc)",
            "aten::zeros": "allocation + memset",
            "aten::pin_memory": "page-locked allocation",
            "cudaDeviceSynchronize": "full device sync",
            "cudaStreamSynchronize": "stream sync",
            "Command Buffer Full": "GPU cmd queue full (back-pressure stall)",
        }

        found = []
        for e_name, times in per_cpu.items():
            for pattern, reason in suspicious_patterns.items():
                if pattern in e_name:
                    total_us = sum(times)
                    if total_us > 50:
                        stack_info = ""
                        for e in events:
                            if e.name == e_name:
                                st = getattr(e, "stack", None)
                                if st:
                                    user = [f for f in st if "site-packages" not in f and "/lib/" not in f]
                                    stack_info = " <- ".join((user or st)[:3])
                                    break
                        pct = 100 * total_us / wall_us
                        found.append((e_name, reason, total_us, pct, len(times), stack_info))
                    break

        if found:
            hdr("Suspicious Ops  (hidden syncs, casts, extra work)")
            for name, reason, total_us, pct, cnt, stack in sorted(found, key=lambda x: -x[2]):
                lines.append(f"  {name:<45s}  {total_us:10.1f} us  ({pct:4.1f}% wall)  x{cnt:<6d}  {reason}")
                if stack:
                    lines.append(f"      at: {stack}")
            footer()

        # ----------------------------------------------------------
        # 6. Memory (if profile_memory was on)
        # ----------------------------------------------------------
        if self._profile_memory:
            allocs = []
            for e in events:
                cpu_mem = getattr(e, "cpu_memory_usage", 0) or 0
                cuda_mem = getattr(e, "cuda_memory_usage", 0) or 0
                if cuda_mem > 0:
                    allocs.append((e.name, cuda_mem, getattr(e, "stack", None)))
            if allocs:
                by_name: dict[str, list[int, int]] = defaultdict(lambda: [0, 0])
                for name, mem, _ in allocs:
                    by_name[name][0] += mem
                    by_name[name][1] += 1
                hdr("CUDA Memory Allocations  (net positive)")
                lines.append(f"  {'Op':<50s}  {'Allocated':>12s}  {'calls':>7s}")
                for name, (mem, cnt) in sorted(by_name.items(), key=lambda x: -x[1][0])[:self._top_n]:
                    mb = mem / (1024 * 1024)
                    lines.append(f"  {name:<50s}  {mb:10.1f} MB  {cnt:7d}")
                footer()

        # ----------------------------------------------------------
        # 7. Summary
        # ----------------------------------------------------------
        lines.append("")
        lines.append("=" * W)
        lines.append("[Summary]")
        lines.append(f"  Wall time:             {wall_us/1e6:.3f} s")
        lines.append(f"  Total CPU self-time:   {total_cpu_us/1e6:.3f} s  (sum of all ops)")
        lines.append(f"  Total CUDA self-time:  {total_device_us/1e6:.3f} s  (sum of all ops)")
        lines.append(f"  Unique op types:       {len(set(e.name for e in events))}")
        lines.append(f"  Total events:          {len(events)}")
        if rt:
            lines.append(f"  CUDA runtime CPU:      {total_rt/1e6:.3f} s  ({100*total_rt/wall_us:.1f}% wall)")
        lines.append(f"  Comm CUDA:             {comm_total/1e6:.3f} s  ({100*comm_total/wall_us:.1f}% wall)")
        lines.append(f"  Compute CUDA:          {compute_total/1e6:.3f} s  ({100*compute_total/wall_us:.1f}% wall)")
        footer()

        return lines
