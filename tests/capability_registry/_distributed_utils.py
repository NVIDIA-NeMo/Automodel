# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Distributed helpers for the capability registry CLI.

Includes:
- torchrun-worker detection
- process-group init
- a sufficiency check that fails fast before spawning torchrun
- an auto-spawn helper used by the parent process to run one capability per child

Adapted from
``tests/functional_tests/llm_pretrain_and_kd/run_tp_output_parity_minified.py``
``_init_distributed`` and ``tests/utils/test_utils.run_test_script``.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist


# Reference training on rank 0 can take several minutes for an 8B model; the
# other ranks wait at ``dist.broadcast`` for that long. Give collectives a
# generous default timeout so they don't trip while rank 0 is still computing.
_PG_TIMEOUT = datetime.timedelta(minutes=60)


def is_torchrun_worker() -> bool:
    """True iff this process was spawned by ``torch.distributed.run`` / ``torchrun``."""
    return "LOCAL_RANK" in os.environ and "TORCHELASTIC_RUN_ID" in os.environ


def init_distributed() -> tuple[torch.device, str]:
    """Initialise the process group when launched via torchrun.

    Returns:
        Tuple of ``(device, device_type)`` for this rank.
    """
    if not dist.is_available():
        return torch.device("cpu"), "cpu"

    if dist.is_initialized():
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if device_type == "cuda"
            else torch.device("cpu")
        )
        return device, device_type

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return torch.device("cpu"), "cpu"

    if torch.cuda.is_available():
        backend = "nccl"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        device_type = "cuda"
    else:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        backend = "gloo"
        device = torch.device("cpu")
        device_type = "cpu"

    dist.init_process_group(backend=backend, timeout=_PG_TIMEOUT)
    return device, device_type


def check_sufficient_gpus(world_size: int) -> str | None:
    """Return ``None`` when enough CUDA GPUs are available, else an explanatory message.

    Use the return value to produce a clean SKIP result rather than crashing the
    user's whole run when GPUs are missing.

    Args:
        world_size: Number of GPUs required.

    Returns:
        ``None`` if requirements are satisfied; a human-readable string otherwise.
    """
    if not torch.cuda.is_available():
        return f"requires {world_size} CUDA GPUs but CUDA is unavailable"
    n = torch.cuda.device_count()
    if n < world_size:
        return f"requires {world_size} GPUs (got {n}); re-run on a machine with at least {world_size} GPUs"
    return None


def broadcast_object_from_rank0(obj, *, src: int = 0):
    """Broadcast a pickleable Python object from ``src`` to all ranks."""
    payload = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def broadcast_tensor_from_rank0(
    tensor: torch.Tensor | None, *, device: torch.device, src: int = 0
) -> torch.Tensor:
    """Broadcast ``tensor`` from rank ``src`` to all ranks.

    Non-source ranks pass ``None``; this helper first broadcasts the shape and
    dtype via ``broadcast_object_list``, then ``dist.broadcast``\\ s the tensor.

    Args:
        tensor: The source tensor on rank ``src``, else ``None``.
        device: Device to allocate the receiving buffer on.
        src: Source rank.

    Returns:
        The broadcast tensor on every rank.
    """
    rank = dist.get_rank()
    meta: list[object | None]
    if rank == src:
        assert tensor is not None
        meta = [(tuple(tensor.shape), str(tensor.dtype))]
    else:
        meta = [None]
    dist.broadcast_object_list(meta, src=src)
    shape, dtype_str = meta[0]  # type: ignore[misc]
    dtype = _torch_dtype_from_str(dtype_str)
    if rank == src:
        buf = tensor.to(device).contiguous()  # type: ignore[union-attr]
    else:
        buf = torch.empty(shape, dtype=dtype, device=device)
    dist.broadcast(buf, src=src)
    return buf


def _torch_dtype_from_str(s: str) -> torch.dtype:
    """Convert a stringified torch dtype (``'torch.bfloat16'``) back to the enum."""
    if s.startswith("torch."):
        s = s[len("torch.") :]
    return getattr(torch, s)


def spawn_torchrun_for_capability(
    *,
    capability: str,
    world_size: int,
    parent_argv: list[str],
    script_path: str,
) -> dict:
    """Spawn a torchrun child to run a single capability and return its JSON result.

    The child writes its ``CapabilityTestResult`` to the path specified by the
    ``NAR_RESULT_PATH`` env var (rank 0 only); the parent reads it back.

    Args:
        capability: Name of the capability to validate (``"tp"``, ``"cp"``, ...).
        world_size: Number of processes per node (= number of GPUs needed).
        parent_argv: ``sys.argv`` of the parent (the same args are passed through).
        script_path: Absolute path to the CLI entry-point script.

    Returns:
        Dict (the JSON-serialised CapabilityTestResult). The caller is
        responsible for wrapping in the dataclass.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
        result_path = fp.name

    env = os.environ.copy()
    env["NAR_CAPABILITY_UNDER_TEST"] = capability
    env["NAR_RESULT_PATH"] = result_path

    # Pass everything after the script name; the child re-parses argv and
    # detects worker mode via is_torchrun_worker().
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        "--nnodes=1",
        script_path,
        *parent_argv,
    ]
    completed = subprocess.run(cmd, env=env, check=False)

    if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
        # Child died before writing the result (e.g. OOM, NCCL init failure).
        return {
            "capability": capability,
            "passed": False,
            "skipped": False,
            "max_kl": None,
            "threshold": 0.0,
            "variant_label": f"{capability.upper()}={world_size}",
            "error": f"torchrun child exited with code {completed.returncode} before writing a result",
        }

    with open(result_path) as f:
        result = json.load(f)
    os.unlink(result_path)
    return result


def write_result_json(path: str, result_dict: dict) -> None:
    """Atomically write the result dict as JSON (used by rank 0 of the child)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result_dict, f)
    os.replace(tmp, path)
