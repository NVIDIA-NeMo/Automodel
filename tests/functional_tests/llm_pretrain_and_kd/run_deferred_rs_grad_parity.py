#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Gradient parity test for defer_rs_grad_accum with FSDP2 (DP=2).

Verifies that accumulating gradients across 2 microbatches with
defer_rs_grad_accum=True produces identical gradients to the baseline
(defer_rs_grad_accum=False, standard per-microbatch RS).

Uses fully_shard directly on a tiny model with ModuleList so the test
works without a full transformer model.

Usage:
    PYTHONPATH=$(pwd) torchrun --nproc_per_node=2 tests/functional_tests/llm_pretrain_and_kd/run_deferred_rs_grad_parity.py
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from nemo_automodel.components.distributed.deferred_rs import DeferredShardedReduceScatter, GAState


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    return rank, device


class _TinyModel(nn.Module):
    """Two-layer MLP with ModuleList so fully_shard can wrap each layer."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(64, 128, bias=False),
            nn.Linear(128, 64, bias=False),
        ])

    def forward(self, x):
        x = torch.relu(self.layers[0](x))
        return self.layers[1](x)


def _build_tiny_model(device):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    return _TinyModel().to(device=device, dtype=torch.bfloat16)


def _apply_fsdp2(model, dp_mesh, mp_policy, defer_rs: bool):
    """Apply fully_shard per layer then to the root, optionally installing deferred RS."""
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)

    ga_state = None
    if defer_rs:
        ga_state = GAState()
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.set_custom_reduce_scatter(DeferredShardedReduceScatter(ga_state))
    return ga_state


def _run_ga(model, inputs, ga_state):
    """Run GA=2 microbatches and return accumulated gradients."""
    model.zero_grad()
    num_microbatches = len(inputs)

    for i, x in enumerate(inputs):
        is_final = (i == num_microbatches - 1)
        if ga_state is not None:
            ga_state.is_final = is_final

        out = model(x)
        loss = out.mean()
        loss.backward()

    grads = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.detach()
            # FSDP2 grads are DTensors (sharded); extract the local shard for comparison
            if hasattr(g, "to_local"):
                g = g.to_local()
            grads[n] = g.clone()
    return grads


def test_fsdp_dp2(rank, device):
    """DP=2 FSDP2: defer_rs_grad_accum=True must produce same grads as False."""
    print(f"[Rank {rank}] === FSDP DP=2 test ===")

    # 3-D mesh required by fsdp2_strategy_parallelize convention: (dp_replicate, dp_shard_cp, tp)
    # For pure DP=2 sharding: dp_replicate=1, dp_shard_cp=2, tp=1
    full_mesh = init_device_mesh("cuda", (1, 2, 1), mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"))
    dp_mesh = full_mesh["dp_shard_cp"]

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    inputs = [
        torch.randn(4, 64, device=device, dtype=torch.bfloat16),
        torch.randn(4, 64, device=device, dtype=torch.bfloat16),
    ]

    # --- Baseline: no deferred RS ---
    model_base = _build_tiny_model(device)
    _apply_fsdp2(model_base, dp_mesh, mp_policy, defer_rs=False)
    grads_base = _run_ga(model_base, inputs, ga_state=None)

    # --- Deferred RS ---
    model_deferred = _build_tiny_model(device)
    ga_state = _apply_fsdp2(model_deferred, dp_mesh, mp_policy, defer_rs=True)
    grads_deferred = _run_ga(model_deferred, inputs, ga_state=ga_state)

    # --- Compare ---
    ok = True
    for name in grads_base:
        if name not in grads_deferred:
            print(f"[Rank {rank}] MISSING grad for {name}")
            ok = False
            continue
        if not torch.allclose(grads_base[name], grads_deferred[name], atol=1e-5):
            max_diff = (grads_base[name] - grads_deferred[name]).abs().max().item()
            print(f"[Rank {rank}] MISMATCH {name}: max_diff={max_diff:.2e}")
            ok = False

    if ok:
        print(f"[Rank {rank}] FSDP DP=2 PASSED — defer_rs_grad_accum grads match baseline")
    return ok


def main():
    rank, device = setup_distributed()
    world_size = dist.get_world_size()

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: requires world_size=2, got {world_size}", file=sys.stderr)
        sys.exit(1)

    results = [test_fsdp_dp2(rank, device)]

    dist.barrier()
    if rank == 0:
        if all(results):
            print("\nAll tests PASSED")
            sys.exit(0)
        else:
            print("\nSome tests FAILED", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
