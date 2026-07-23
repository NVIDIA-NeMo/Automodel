#!/usr/bin/env python
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

"""Functional test for dtype-specific FSDP2 mixed precision policies.

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/training/run_fully_shard_by_dtype_param_dtype.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


class ParamDTypeCheckingModule(nn.Module):
    def __init__(self, name: str, dtype: torch.dtype):
        super().__init__()
        self.name = name
        self.expected_dtype = dtype
        self.weight = nn.Parameter(torch.ones(4, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != self.expected_dtype:
            raise AssertionError(
                f"{self.name} expected param dtype {self.expected_dtype}, got {self.weight.dtype}"
            )
        return x


class TwoDTypeRoot(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp32_module = ParamDTypeCheckingModule("fp32_module", torch.float32)
        self.bf16_module = ParamDTypeCheckingModule("bf16_module", torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fp32_module(x)
        return self.bf16_module(x)


def _init_distributed() -> tuple[int, torch.device]:
    if not dist.is_available():
        return 1, torch.device("cpu")
    if not dist.is_initialized():
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            return 1, torch.device("cpu")
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available", file=sys.stderr)
        return 0

    world_size, device = _init_distributed()
    rank = _rank()
    if world_size != 2:
        if rank == 0:
            print(f"ERROR: This test requires world_size=2, got {world_size}", file=sys.stderr)
        return 1

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    model = TwoDTypeRoot().to(device)
    mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    # Start with the wrong param_dtype; fully_shard_by_dtype should override it
    # per wrapped module based on the underlying parameter dtype.
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    fully_shard_by_dtype(model, mesh=mesh, mp_policy=mp_policy, offload_policy=None)

    model(torch.ones(4, device=device, dtype=torch.float32))
    torch.cuda.synchronize()

    if rank == 0:
        print("PASS: fully_shard_by_dtype preserved forward parameter dtypes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if _is_distributed():
            dist.destroy_process_group()
