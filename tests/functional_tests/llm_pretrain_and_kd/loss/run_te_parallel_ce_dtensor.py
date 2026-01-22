#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Standalone distributed test for DTensor + TEParallelCrossEntropy.

This reproduces the common TP setup where the model returns vocab-sharded logits as a DTensor
(e.g. TP plans with `use_local_output=False`), and validates that
`nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy` can:

- unwrap DTensor logits to the local shard (no full gather)
- infer the TP process group from the DTensor mesh
- compute a loss matching a full-logits PyTorch reference

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/training/loss/run_te_parallel_ce_dtensor.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from nemo_automodel.components.loss.te_parallel_ce import (
    HAVE_TE_PARALLEL_CE,
    MISSING_TE_PARALLEL_CE_MSG,
    TEParallelCrossEntropy,
)


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _init_distributed() -> None:
    if not dist.is_available():
        return
    if dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def main() -> int:
    _init_distributed()
    rank = _get_rank()
    world_size = _get_world_size()

    if not torch.cuda.is_available():
        if rank == 0:
            print("SKIP: CUDA not available", file=sys.stderr)
        return 0

    if not HAVE_TE_PARALLEL_CE:
        if rank == 0:
            print(f"SKIP: TEParallelCrossEntropy unavailable: {MISSING_TE_PARALLEL_CE_MSG}", file=sys.stderr)
        return 0

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: This test requires world_size=2, got {world_size}", file=sys.stderr)
        return 1

    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    device = torch.device(f"cuda:{local_rank}")

    # Deterministic inputs across ranks (so labels match).
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    batch_size = 2
    seq_len = 8
    vocab_size = 128
    assert vocab_size % world_size == 0
    vocab_local = vocab_size // world_size
    dtype = torch.bfloat16
    ignore_index = -100

    logits_local = torch.randn(batch_size, seq_len, vocab_local, device=device, dtype=dtype)
    logits_local_for_ref = logits_local.detach().clone()

    # Vocab-sharded DTensor logits (global shape is [B, T, V]).
    tp_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("tp",))
    logits_dt = DTensor.from_local(logits_local, device_mesh=tp_mesh, placements=[Shard(-1)], run_check=True)

    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    # Sprinkle some ignored labels.
    labels.view(-1)[::7] = ignore_index

    # Reference: all-gather full logits and compute PyTorch CE.
    gathered: list[torch.Tensor] = [torch.empty_like(logits_local_for_ref) for _ in range(world_size)]
    dist.all_gather(gathered, logits_local_for_ref)
    full_logits = torch.cat(gathered, dim=-1)
    ref_loss = F.cross_entropy(
        full_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction="mean",
        ignore_index=ignore_index,
    )

    # Under test: DTensor logits + inferred tp_group.
    te_loss = TEParallelCrossEntropy(ignore_index=ignore_index, reduction="mean", tp_group=None)(logits_dt, labels)

    ok = torch.allclose(te_loss.float(), ref_loss.float(), rtol=1e-2, atol=1e-2)
    ok_tensor = torch.tensor(1 if ok else 0, device=device, dtype=torch.int)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    all_ok = bool(ok_tensor.item())

    if rank == 0:
        if all_ok:
            print("PASS: DTensor + TEParallelCrossEntropy matches reference")
        else:
            print(
                "FAIL: Loss mismatch\n"
                f"  ref_loss={ref_loss.item()}\n"
                f"  te_loss={te_loss.item()}\n",
                file=sys.stderr,
            )

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

