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

"""Standalone test script for Mamba layer context parallelism validation.

This script validates that the NemotronV3Mamba2Mixer produces identical forward
outputs and gradients when using context parallelism (CP=2, hidden-parallel
strategy) versus no context parallelism (CP=1).

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_mamba_cp.py
"""

import os
import sys

import torch
import torch.distributed as dist


def init_distributed():
    """Initialize distributed environment from torchrun env vars."""
    if not (dist.is_available() and dist.is_initialized()):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class MockNemotronV3Config:
    """Mock configuration for NemotronV3Mamba2Mixer."""

    def __init__(self):
        self.hidden_size = 256
        self.mamba_num_heads = 8
        self.mamba_head_dim = 32
        self.ssm_state_size = 16
        self.n_groups = 1
        self.chunk_size = 256
        self.conv_kernel = 4
        self.use_conv_bias = True
        self.mamba_hidden_act = "silu"
        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.use_bias = False
        self.layer_norm_epsilon = 1e-5
        self.num_hidden_layers = 4


def run_test():
    """Run the CP validation test for NemotronV3Mamba2Mixer."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: This test requires exactly 2 GPUs, got {world_size}", file=sys.stderr)
        return 1

    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

    config = MockNemotronV3Config()

    mixer_no_cp = NemotronV3Mamba2Mixer(config, layer_idx=0).to(device).to(torch.bfloat16)
    mixer_with_cp = NemotronV3Mamba2Mixer(config, layer_idx=0).to(device).to(torch.bfloat16)

    mixer_no_cp.eval()
    mixer_with_cp.eval()
    mixer_with_cp.load_state_dict(mixer_no_cp.state_dict())

    for param_no_cp, param_with_cp in zip(mixer_no_cp.parameters(), mixer_with_cp.parameters()):
        dist.broadcast(param_no_cp.data, src=0)
        dist.broadcast(param_with_cp.data, src=0)

    # Baseline: CP=1
    batch_size = 2
    seq_len = 64

    torch.manual_seed(42 + rank)
    x_full = torch.randn(
        batch_size, seq_len, config.hidden_size,
        device=device, dtype=torch.bfloat16, requires_grad=True,
    )

    dist.broadcast(x_full.data, src=0)
    x_no_cp = x_full.detach().clone().requires_grad_(True)

    output_no_cp = mixer_no_cp(x_no_cp)
    loss_no_cp = output_no_cp.sum()
    loss_no_cp.backward()

    output_baseline = output_no_cp.detach().clone()
    grad_baseline = x_no_cp.grad.detach().clone()
    in_proj_grad_baseline = mixer_no_cp.in_proj.weight.grad.detach().clone()

    dist.barrier()

    # Test: CP=2
    from torch.distributed.device_mesh import init_device_mesh
    from nemo_automodel.components.distributed.mamba_cp import MambaContextParallel

    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()

    mixer_with_cp.cp = MambaContextParallel(
        cp_group=cp_group,
        num_heads=config.mamba_num_heads,
        head_dim=config.mamba_head_dim,
        n_groups=config.n_groups,
        d_state=config.ssm_state_size,
        conv1d=mixer_with_cp.conv1d,
        dt_bias=mixer_with_cp.dt_bias,
        A_log=mixer_with_cp.A_log,
        D=mixer_with_cp.D,
    )

    half_seq = seq_len // world_size
    x_local = x_full[:, rank * half_seq : (rank + 1) * half_seq, :].detach().clone().requires_grad_(True)

    output_with_cp = mixer_with_cp(x_local)
    loss_with_cp = output_with_cp.sum()
    loss_with_cp.backward()

    output_gathered = [
        torch.zeros(batch_size, half_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    grad_gathered = [
        torch.zeros(batch_size, half_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]

    dist.all_gather(output_gathered, output_with_cp.contiguous())
    dist.all_gather(grad_gathered, x_local.grad.contiguous())

    output_with_cp_full = torch.cat(output_gathered, dim=1)
    grad_with_cp_full = torch.cat(grad_gathered, dim=1)

    in_proj_grad_cp = mixer_with_cp.in_proj.weight.grad.detach().clone()
    dist.all_reduce(in_proj_grad_cp, op=dist.ReduceOp.SUM)

    if rank == 0:
        output_diff = (output_with_cp_full - output_baseline).abs()
        grad_diff = (grad_with_cp_full - grad_baseline).abs()
        in_proj_grad_diff = (in_proj_grad_cp - in_proj_grad_baseline).abs()

        print(f"\n{'='*70}")
        print(f"Context Parallelism Validation Test - NemotronV3 Mamba2Mixer")
        print(f"{'='*70}")
        print(f"Output shape: CP={output_with_cp_full.shape}, Baseline={output_baseline.shape}")
        print(f"Output diff - mean: {output_diff.mean():.6f}, max: {output_diff.max():.6f}, std: {output_diff.std():.6f}")
        print(f"Output relative diff - mean: {(output_diff / (output_baseline.abs() + 1e-8)).mean():.6f}")
        print(f"\nInput gradient statistics:")
        print(f"  Baseline - min: {grad_baseline.abs().min():.6f}, max: {grad_baseline.abs().max():.6f}, mean: {grad_baseline.abs().mean():.6f}")
        print(f"  CP       - min: {grad_with_cp_full.abs().min():.6f}, max: {grad_with_cp_full.abs().max():.6f}, mean: {grad_with_cp_full.abs().mean():.6f}")
        print(f"Grad diff - mean: {grad_diff.mean():.6f}, max: {grad_diff.max():.6f}, std: {grad_diff.std():.6f}")
        print(f"\nin_proj.weight.grad statistics:")
        print(f"  Baseline - mean: {in_proj_grad_baseline.abs().mean():.6f}, max: {in_proj_grad_baseline.abs().max():.6f}")
        print(f"  CP       - mean: {in_proj_grad_cp.abs().mean():.6f}, max: {in_proj_grad_cp.abs().max():.6f}")
        print(f"  Diff     - mean: {in_proj_grad_diff.mean():.6f}, max: {in_proj_grad_diff.max():.6f}")

    try:
        torch.testing.assert_close(
            output_with_cp_full,
            output_baseline,
            rtol=1e-2,
            atol=0.01,
            msg=f"[Rank {rank}] Forward outputs differ between CP=1 and CP=2",
        )

        torch.testing.assert_close(
            grad_with_cp_full,
            grad_baseline,
            rtol=2e-2,
            atol=0.05,
            msg=f"[Rank {rank}] Input gradients differ between CP=1 and CP=2",
        )

        torch.testing.assert_close(
            in_proj_grad_cp,
            in_proj_grad_baseline,
            rtol=5e-2,
            atol=1.5,
            msg=f"[Rank {rank}] in_proj.weight.grad differs between CP=1 and CP=2",
        )

        if rank == 0:
            print(f"Test PASSED: Forward outputs and gradients match between CP=1 and CP=2")
            print(f"{'='*70}\n")
        return 0

    except AssertionError as e:
        if rank == 0:
            print(f"Test FAILED: {e}")
            print(f"Note: Some numerical differences are expected with bfloat16 precision")
            print(f"{'='*70}\n")
        return 1


def main():
    init_distributed()
    exit_code = run_test()
    if dist.is_initialized():
        dist.barrier()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
