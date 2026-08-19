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

"""Temporary, environment-gated diagnostics for distributed training hangs."""

from __future__ import annotations

import json
import os
import pickle
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

_TRUE = {"1", "true", "yes", "on"}
_COUNTS: dict[str, int] = defaultdict(int)
_PRINT_LOCK = threading.Lock()
_PROBE_STARTED = False


def _enabled(name: str = "NEMO_AUTOMODEL_DIST_DEBUG") -> bool:
    return os.environ.get(name, "0").strip().lower() in _TRUE


def configure_distributed_debug_environment() -> None:
    """Configure flight recorder and NCCL logging before process groups are built."""
    if not _enabled():
        return

    pipeline_dir = Path(os.environ.get("PIPELINE_DIR", "/tmp"))
    test_name = os.environ.get("TEST_NAME", "automodel_dist_debug")
    debug_dir = Path(os.environ.get("NEMO_AUTOMODEL_DIST_DEBUG_DIR", pipeline_dir / test_name / "dist_debug"))
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        debug_dir = Path("/tmp") / test_name / "dist_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("NEMO_AUTOMODEL_DIST_DEBUG_DIR", str(debug_dir))
    os.environ.setdefault("TORCH_FR_DUMP_TEMP_FILE", str(debug_dir / "torch_fr_trace_"))
    os.environ.setdefault("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", str(debug_dir / "torch_nccl_debug_"))
    os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "5000")
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    os.environ.setdefault("TORCH_NCCL_TRACE_CPP_STACK", "1")
    os.environ.setdefault("TORCH_NCCL_ENABLE_TIMING", "1")
    os.environ.setdefault("TORCH_NCCL_DESYNC_DEBUG", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "1")
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "480")
    os.environ.setdefault("TORCH_NCCL_COORD_CHECK_MILSEC", "1000")
    os.environ.setdefault("TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC", "120000")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "INFO")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,ENV,GRAPH,NET,COLL,P2P")
    os.environ.setdefault("NCCL_DEBUG_TIMESTAMP_LEVELS", "WARN,INFO")


configure_distributed_debug_environment()


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "-1"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "-1"))


def _selected_rank(rank: int, local_rank: int) -> bool:
    selected = os.environ.get("NEMO_AUTOMODEL_DIST_DEBUG_RANKS", "all").strip().lower()
    if selected == "all":
        return True
    if selected in {"node_leaders", "local0"}:
        return local_rank in {-1, 0}
    try:
        return rank in {int(value) for value in selected.split(",")}
    except ValueError:
        return rank == 0


def _describe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": str(value.device),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, (tuple, list)) and value and all(isinstance(item, torch.Tensor) for item in value):
        return [_describe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def marker(name: str, /, **fields: Any) -> None:
    """Print a timestamped marker without adding a distributed collective."""
    if not _enabled():
        return

    rank = _rank()
    local_rank = _local_rank()
    if not _selected_rank(rank, local_rank):
        return

    max_per_marker = int(os.environ.get("NEMO_AUTOMODEL_DIST_DEBUG_MAX_PER_MARKER", "12"))
    count = _COUNTS[name]
    _COUNTS[name] += 1
    if count >= max_per_marker:
        return

    sync_error = None
    if _enabled("NEMO_AUTOMODEL_DIST_DEBUG_SYNC_MARKERS") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as exc:  # pragma: no cover - diagnostic path
            sync_error = repr(exc)

    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "mono": round(time.monotonic(), 6),
        "host": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "pid": os.getpid(),
        "marker": name,
        "occurrence": count,
        "cuda_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "sync_error": sync_error,
        **{key: _describe(value) for key, value in fields.items()},
    }
    with _PRINT_LOCK:
        print("[DIST_DEBUG_MARKER] " + json.dumps(payload, sort_keys=True), flush=True)


def _run_probe_command(command: list[str], timeout: float = 10.0) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"{command[0]} failed: {exc!r}"
    output = (completed.stdout + completed.stderr).strip()
    return output if output else f"{command[0]} exit={completed.returncode} no-output"


def _dump_python_stacks() -> None:
    frames = sys._current_frames()
    chunks = []
    for thread_id, frame in sorted(frames.items()):
        chunks.append(f"thread={thread_id}\n{''.join(traceback.format_stack(frame))}")
    with _PRINT_LOCK:
        print(
            f"[DIST_DEBUG_STACK] host={socket.gethostname()} rank={_rank()} local_rank={_local_rank()} "
            f"pid={os.getpid()}\n" + "\n".join(chunks),
            flush=True,
        )


def _read_rdma_counters() -> dict[str, str]:
    result: dict[str, str] = {}
    root = Path("/sys/class/infiniband")
    if not root.exists():
        return result
    names = ("port_rcv_data", "port_xmit_data", "port_rcv_errors", "port_xmit_discards")
    for port in root.glob("*/ports/*"):
        for name in names:
            path = port / "counters" / name
            try:
                result[str(path.relative_to(root))] = path.read_text().strip()
            except OSError:
                pass
    return result


def _flight_recorder_snapshot() -> None:
    try:
        dump_fn = getattr(torch._C._distributed_c10d, "_dump_nccl_trace", None)
        if dump_fn is None:
            marker("probe.flight_recorder.unavailable")
            return
        # The bytes are produced in-process by PyTorch's C++ flight recorder,
        # not loaded from an external or user-controlled source.
        trace = pickle.loads(dump_fn())  # noqa: S301
        entries = trace.get("entries", [])
        keys = (
            "record_id",
            "pg_id",
            "process_group",
            "collective_seq_id",
            "p2p_seq_id",
            "profiling_name",
            "state",
            "input_sizes",
            "output_sizes",
            "duration_ms",
        )
        tail = [{key: entry.get(key) for key in keys if key in entry} for entry in entries[-16:]]
        marker("probe.flight_recorder.snapshot", entry_count=len(entries), tail=json.dumps(tail, default=str))
    except Exception as exc:  # pragma: no cover - diagnostic path
        marker("probe.flight_recorder.error", error=repr(exc))


def _system_probe() -> None:
    gpu = _run_probe_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = _run_probe_command(["ps", "-ww", "-eo", "pid,ppid,stat,wchan:32,etime,comm,args", "--sort=pid"])
    interesting = "\n".join(
        line
        for line in processes.splitlines()
        if any(token in line for token in ("python", "nvcc", "cc1plus", "ptxas", "nvlink", "torchrun"))
    )
    with _PRINT_LOCK:
        print(
            f"[DIST_DEBUG_SYSTEM] host={socket.gethostname()} rank={_rank()} local_rank={_local_rank()} "
            f"pid={os.getpid()} gpu={json.dumps(gpu)} rdma={json.dumps(_read_rdma_counters(), sort_keys=True)} "
            f"processes={json.dumps(interesting)}",
            flush=True,
        )


def _probe_loop(heartbeat_group: dist.ProcessGroup | None) -> None:
    interval = float(os.environ.get("NEMO_AUTOMODEL_DIST_DEBUG_PROBE_INTERVAL_SEC", "30"))
    stack_interval = float(os.environ.get("NEMO_AUTOMODEL_DIST_DEBUG_STACK_INTERVAL_SEC", "120"))
    flight_interval = float(os.environ.get("NEMO_AUTOMODEL_DIST_DEBUG_FLIGHT_INTERVAL_SEC", "60"))
    next_stack = time.monotonic()
    next_flight = time.monotonic()
    heartbeat = 0
    while True:
        started = time.monotonic()
        if heartbeat_group is not None:
            try:
                value = torch.tensor([heartbeat], dtype=torch.int64)
                dist.all_reduce(value, group=heartbeat_group)
                marker(
                    "probe.gloo_heartbeat.ok",
                    heartbeat=heartbeat,
                    reduced=int(value.item()),
                    seconds=round(time.monotonic() - started, 4),
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                marker("probe.gloo_heartbeat.error", heartbeat=heartbeat, error=repr(exc))
                heartbeat_group = None

        now = time.monotonic()
        if _local_rank() in {-1, 0}:
            _system_probe()
            if now >= next_stack:
                _dump_python_stacks()
                next_stack = now + stack_interval
            if now >= next_flight:
                _flight_recorder_snapshot()
                next_flight = now + flight_interval

        heartbeat += 1
        time.sleep(max(1.0, interval - (time.monotonic() - started)))


def start_distributed_debug_probe() -> None:
    """Start one system probe per node and a separate Gloo all-rank heartbeat."""
    global _PROBE_STARTED
    if not _enabled() or _PROBE_STARTED:
        return
    _PROBE_STARTED = True

    heartbeat_group = None
    marker("probe.start.before_gloo_group")
    if _enabled("NEMO_AUTOMODEL_DIST_DEBUG_GLOO_HEARTBEAT") and dist.is_available() and dist.is_initialized():
        heartbeat_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=45))
    marker(
        "probe.start.after_gloo_group",
        heartbeat=heartbeat_group is not None,
        debug_dir=os.environ.get("NEMO_AUTOMODEL_DIST_DEBUG_DIR"),
    )
    thread = threading.Thread(
        target=_probe_loop,
        args=(heartbeat_group,),
        name="automodel-dist-debug-probe",
        daemon=True,
    )
    thread.start()
