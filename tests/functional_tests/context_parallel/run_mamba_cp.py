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
outputs and gradients when using context parallelism (CP=2) versus no context
parallelism (CP=1) across four configurations:

  Config 1 (bshd_te):        3D BSHD input, TE p2p CP, DualChunkSwap
  Config 2 (thd_te):         2D THD input, TE p2p CP, DualChunkSwap, cu_seqlens
  Config 3 (thd_te_packed):  2D THD input, TE p2p CP, multi-sequence packing, seq_idx
  Config 4 (bshd_sdpa):      3D BSHD input, DualChunkSwap split (matches context_parallel allgather)

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_mamba_cp.py
"""

import os
import sys

import torch
import torch.distributed as dist


def dual_chunk_swap_unsplit(chunks_per_rank, cp_size, seq_dim=1, cu_seqlens=None):
    """Reconstruct full sequence from DualChunkSwap-ordered rank outputs.

    When *cu_seqlens* is provided (packed THD layout), the unsplit is applied
    independently to each sequence and the results are concatenated.
    *cu_seqlens* should contain the **local** (per-rank) boundaries so that
    each sequence segment has length ``seq_len_i / cp_size``.
    """
    if cu_seqlens is not None:
        # Per-sequence unsplit for packed data.
        num_seqs = len(cu_seqlens) - 1
        seq_parts = []
        for s in range(num_seqs):
            start, end = cu_seqlens[s].item(), cu_seqlens[s + 1].item()
            per_seq_ranks = [r[start:end] for r in chunks_per_rank]
            seq_parts.append(dual_chunk_swap_unsplit(per_seq_ranks, cp_size, seq_dim=0))
        return torch.cat(seq_parts, dim=0)

    all_chunks = [None] * (2 * cp_size)
    for rank_idx, rank_output in enumerate(chunks_per_rank):
        c0, c1 = torch.chunk(rank_output, 2, dim=seq_dim)
        all_chunks[rank_idx] = c0
        all_chunks[2 * cp_size - rank_idx - 1] = c1
    return torch.cat(all_chunks, dim=seq_dim)


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
        self.n_groups = 2
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


def _create_mixer_pair(config, device):
    """Create a pair of identical mixers (baseline and CP) with synced weights."""
    from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

    mixer_baseline = NemotronV3Mamba2Mixer(config, layer_idx=0).to(device).to(torch.bfloat16)
    mixer_cp = NemotronV3Mamba2Mixer(config, layer_idx=0).to(device).to(torch.bfloat16)
    mixer_baseline.eval()
    mixer_cp.eval()
    mixer_cp.load_state_dict(mixer_baseline.state_dict())

    for p_base, p_cp in zip(mixer_baseline.parameters(), mixer_cp.parameters()):
        dist.broadcast(p_base.data, src=0)
        dist.broadcast(p_cp.data, src=0)

    return mixer_baseline, mixer_cp


def _compare_results(
    config_name,
    rank,
    output_cp_full,
    output_baseline,
    grad_cp_full,
    grad_baseline,
    param_grad_cp,
    param_grad_baseline,
    param_name,
    output_atol,
    output_rtol,
    grad_atol,
    grad_rtol,
    param_atol,
    param_rtol,
):
    """Compare CP vs baseline results and return 0 on pass, 1 on fail."""
    if rank == 0:
        output_diff = (output_cp_full - output_baseline).abs()
        grad_diff = (grad_cp_full - grad_baseline).abs()
        param_diff = (param_grad_cp - param_grad_baseline).abs()

        print(f"\n{'=' * 70}")
        print(f"Config: {config_name} - NemotronV3 Mamba2Mixer")
        print(f"{'=' * 70}")
        print(f"Output shape: CP={output_cp_full.shape}, Baseline={output_baseline.shape}")
        print(f"Output diff - mean: {output_diff.mean():.6f}, max: {output_diff.max():.6f}")
        print(f"Grad diff - mean: {grad_diff.mean():.6f}, max: {grad_diff.max():.6f}")
        print(f"{param_name} grad diff - mean: {param_diff.mean():.6f}, max: {param_diff.max():.6f}")

    try:
        torch.testing.assert_close(
            output_cp_full,
            output_baseline,
            rtol=output_rtol,
            atol=output_atol,
            msg=f"[{config_name}][Rank {rank}] Forward outputs differ",
        )
        torch.testing.assert_close(
            grad_cp_full,
            grad_baseline,
            rtol=grad_rtol,
            atol=grad_atol,
            msg=f"[{config_name}][Rank {rank}] Input gradients differ",
        )
        torch.testing.assert_close(
            param_grad_cp,
            param_grad_baseline,
            rtol=param_rtol,
            atol=param_atol,
            msg=f"[{config_name}][Rank {rank}] {param_name} grad differs",
        )
        if rank == 0:
            print("  PASSED")
            print(f"{'=' * 70}")
        return 0
    except AssertionError as e:
        if rank == 0:
            print(f"  FAILED: {e}")
            print(f"{'=' * 70}")
        return 1


# ---------------------------------------------------------------------------
# Config 1: BSHD + TE
# ---------------------------------------------------------------------------
def run_bshd_te(rank, world_size, device, config):
    """Config 1: 3D BSHD input with TE p2p CP and DualChunkSwap."""
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.distributed.mamba_cp import MambaContextParallel

    mixer_baseline, mixer_cp = _create_mixer_pair(config, device)

    # Baseline
    batch_size, seq_len = 2, 512
    torch.manual_seed(42 + rank)
    x_full = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    dist.broadcast(x_full.data, src=0)

    x_no_cp = x_full.detach().clone().requires_grad_(True)
    output_baseline = mixer_baseline(x_no_cp)
    output_baseline.sum().backward()
    out_base = output_baseline.detach().clone()
    grad_base = x_no_cp.grad.detach().clone()
    param_grad_base = mixer_baseline.in_proj.weight.grad.detach().clone()
    dist.barrier()

    # CP=2
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()
    mixer_cp.cp = MambaContextParallel(
        cp_group=cp_group,
        num_heads=config.mamba_num_heads,
        head_dim=config.mamba_head_dim,
        n_groups=config.n_groups,
        d_state=config.ssm_state_size,
        mixer=mixer_cp,
    )

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    indices = tex.thd_get_partitioned_indices(cu_seqlens, seq_len, world_size, rank)
    x_local = x_full.detach()[:, indices, :].clone().requires_grad_(True)
    half_seq = x_local.shape[1]

    output_cp = mixer_cp(x_local)
    output_cp.sum().backward()

    output_gathered = [
        torch.zeros(batch_size, half_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    grad_gathered = [
        torch.zeros(batch_size, half_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    dist.all_gather(output_gathered, output_cp.contiguous())
    dist.all_gather(grad_gathered, x_local.grad.contiguous())

    out_cp_full = dual_chunk_swap_unsplit(output_gathered, cp_size=world_size, seq_dim=1)
    grad_cp_full = dual_chunk_swap_unsplit(grad_gathered, cp_size=world_size, seq_dim=1)

    param_grad_cp = mixer_cp.in_proj.weight.grad.detach().clone()
    dist.all_reduce(param_grad_cp, op=dist.ReduceOp.SUM)

    return _compare_results(
        "bshd_te",
        rank,
        out_cp_full,
        out_base,
        grad_cp_full,
        grad_base,
        param_grad_cp,
        param_grad_base,
        "in_proj.weight",
        output_atol=0.01,
        output_rtol=1e-2,
        grad_atol=0.05,
        grad_rtol=2e-2,
        param_atol=1.5,
        param_rtol=5e-2,
    )


# ---------------------------------------------------------------------------
# Config 2: THD + TE
# ---------------------------------------------------------------------------
def run_thd_te(rank, world_size, device, config):
    """Config 2: 2D THD input with TE p2p CP and DualChunkSwap."""
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.distributed.mamba_cp import MambaContextParallel

    mixer_baseline, mixer_cp = _create_mixer_pair(config, device)

    # Baseline: squeeze to 2D [T, H]
    batch_size, seq_len = 1, 512
    torch.manual_seed(42 + rank)
    x_full = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    dist.broadcast(x_full.data, src=0)

    x_no_cp = x_full.squeeze(0).detach().clone().requires_grad_(True)  # [T, H]
    cu_seqlens_baseline = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    output_baseline = mixer_baseline(x_no_cp, cu_seqlens=cu_seqlens_baseline)
    output_baseline.sum().backward()
    out_base = output_baseline.detach().clone()  # [T, H]
    grad_base = x_no_cp.grad.detach().clone()
    param_grad_base = mixer_baseline.in_proj.weight.grad.detach().clone()
    dist.barrier()

    # CP=2
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()
    mixer_cp.cp = MambaContextParallel(
        cp_group=cp_group,
        num_heads=config.mamba_num_heads,
        head_dim=config.mamba_head_dim,
        n_groups=config.n_groups,
        d_state=config.ssm_state_size,
        mixer=mixer_cp,
    )

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex

    # cu_seqlens is GLOBAL (pre-TE-partitioning) — matches the convention in the recipe batch.
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    indices = tex.thd_get_partitioned_indices(cu_seqlens, seq_len, world_size, rank)
    x_local = x_full.squeeze(0)[indices].detach().clone().requires_grad_(True)  # [T/cp, H]
    local_len = x_local.shape[0]

    output_cp = mixer_cp(x_local, cu_seqlens=cu_seqlens)
    output_cp.sum().backward()

    # Gather 2D outputs (seq_dim=0 for THD)
    output_gathered = [
        torch.zeros(local_len, config.hidden_size, device=device, dtype=torch.bfloat16) for _ in range(world_size)
    ]
    grad_gathered = [
        torch.zeros(local_len, config.hidden_size, device=device, dtype=torch.bfloat16) for _ in range(world_size)
    ]
    dist.all_gather(output_gathered, output_cp.contiguous())
    dist.all_gather(grad_gathered, x_local.grad.contiguous())

    out_cp_full = dual_chunk_swap_unsplit(output_gathered, cp_size=world_size, seq_dim=0)
    grad_cp_full = dual_chunk_swap_unsplit(grad_gathered, cp_size=world_size, seq_dim=0)

    param_grad_cp = mixer_cp.in_proj.weight.grad.detach().clone()
    dist.all_reduce(param_grad_cp, op=dist.ReduceOp.SUM)

    return _compare_results(
        "thd_te",
        rank,
        out_cp_full,
        out_base,
        grad_cp_full,
        grad_base,
        param_grad_cp,
        param_grad_base,
        "in_proj.weight",
        output_atol=0.01,
        output_rtol=1e-2,
        grad_atol=0.05,
        grad_rtol=2e-2,
        param_atol=1.5,
        param_rtol=5e-2,
    )


# ---------------------------------------------------------------------------
# Config 3: THD + TE + packing
# ---------------------------------------------------------------------------
def run_thd_te_packed(rank, world_size, device, config):
    """Config 3: 2D THD input with TE p2p CP, multi-sequence packing, and seq_idx."""
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.distributed.mamba_cp import MambaContextParallel

    mixer_baseline, mixer_cp = _create_mixer_pair(config, device)

    # Two packed sequences. Each sequence length must be divisible by
    # 2 * cp_size * chunk_size for DualChunkSwap to work correctly.
    # With cp_size=2 and chunk_size=256, the minimum per-sequence length
    # that satisfies divisibility is 1024 (= 2 * 2 * 256).
    # Use two sequences of length 1024 each for a total of 2048 tokens.
    seq_len_a, seq_len_b = 1024, 1024
    total_len = seq_len_a + seq_len_b

    torch.manual_seed(42 + rank)
    x_full_2d = torch.randn(total_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    dist.broadcast(x_full_2d.data, src=0)

    cu_seqlens_full = torch.tensor([0, seq_len_a, total_len], dtype=torch.int32, device=device)

    # Baseline: run without CP but WITH seq_idx to validate packing correctness
    x_no_cp = x_full_2d.detach().clone().requires_grad_(True)
    output_baseline = mixer_baseline(x_no_cp, cu_seqlens=cu_seqlens_full)
    output_baseline.sum().backward()
    out_base = output_baseline.detach().clone()
    grad_base = x_no_cp.grad.detach().clone()
    param_grad_base = mixer_baseline.in_proj.weight.grad.detach().clone()
    dist.barrier()

    # CP=2
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()
    mixer_cp.cp = MambaContextParallel(
        cp_group=cp_group,
        num_heads=config.mamba_num_heads,
        head_dim=config.mamba_head_dim,
        n_groups=config.n_groups,
        d_state=config.ssm_state_size,
        mixer=mixer_cp,
    )

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex

    indices = tex.thd_get_partitioned_indices(cu_seqlens_full, total_len, world_size, rank)
    x_local = x_full_2d[indices].detach().clone().requires_grad_(True)
    local_len = x_local.shape[0]

    # Pass GLOBAL cu_seqlens to mixer — matches the convention in the recipe batch.
    # MambaContextParallel derives local cu_seqlens internally.
    output_cp = mixer_cp(x_local, cu_seqlens=cu_seqlens_full)
    output_cp.sum().backward()

    # Gather 2D outputs
    output_gathered = [
        torch.zeros(local_len, config.hidden_size, device=device, dtype=torch.bfloat16) for _ in range(world_size)
    ]
    grad_gathered = [
        torch.zeros(local_len, config.hidden_size, device=device, dtype=torch.bfloat16) for _ in range(world_size)
    ]
    dist.all_gather(output_gathered, output_cp.contiguous())
    dist.all_gather(grad_gathered, x_local.grad.contiguous())

    # dual_chunk_swap_unsplit needs LOCAL cu_seqlens (per-rank boundaries).
    local_seq_a = seq_len_a // world_size
    local_seq_b = seq_len_b // world_size
    cu_seqlens_local = torch.tensor([0, local_seq_a, local_seq_a + local_seq_b], dtype=torch.int32, device=device)
    out_cp_full = dual_chunk_swap_unsplit(output_gathered, cp_size=world_size, seq_dim=0, cu_seqlens=cu_seqlens_local)
    grad_cp_full = dual_chunk_swap_unsplit(grad_gathered, cp_size=world_size, seq_dim=0, cu_seqlens=cu_seqlens_local)

    param_grad_cp = mixer_cp.in_proj.weight.grad.detach().clone()
    dist.all_reduce(param_grad_cp, op=dist.ReduceOp.SUM)

    return _compare_results(
        "thd_te_packed",
        rank,
        out_cp_full,
        out_base,
        grad_cp_full,
        grad_base,
        param_grad_cp,
        param_grad_base,
        "in_proj.weight",
        output_atol=0.02,
        output_rtol=2e-2,
        grad_atol=0.1,
        grad_rtol=5e-2,
        param_atol=2.0,
        param_rtol=5e-2,
    )


# ---------------------------------------------------------------------------
# Config 4: BSHD + SDPA
# ---------------------------------------------------------------------------
def run_bshd_sdpa(rank, world_size, device, config):
    """Config 4: 3D BSHD input with DualChunkSwap split (matches context_parallel allgather)."""
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.distributed.mamba_cp import (
        MambaContextParallel,
        _redo_attention_load_balancing,
        _undo_attention_load_balancing,
    )

    mixer_baseline, mixer_cp = _create_mixer_pair(config, device)

    # Baseline
    batch_size, seq_len = 2, 512
    torch.manual_seed(42 + rank)
    x_full = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    dist.broadcast(x_full.data, src=0)

    x_no_cp = x_full.detach().clone().requires_grad_(True)
    output_baseline = mixer_baseline(x_no_cp)
    output_baseline.sum().backward()
    out_base = output_baseline.detach().clone()
    grad_base = x_no_cp.grad.detach().clone()
    param_grad_base = mixer_baseline.in_proj.weight.grad.detach().clone()
    dist.barrier()

    # CP=2 with DualChunkSwap (same reordering as context_parallel allgather)
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()
    mixer_cp.cp = MambaContextParallel(
        cp_group=cp_group,
        num_heads=config.mamba_num_heads,
        head_dim=config.mamba_head_dim,
        n_groups=config.n_groups,
        d_state=config.ssm_state_size,
        mixer=mixer_cp,
    )

    # Apply DualChunkSwap then take this rank's contiguous half
    x_swapped = _redo_attention_load_balancing(x_full.detach(), world_size)
    chunk = seq_len // world_size
    x_local = x_swapped[:, rank * chunk : (rank + 1) * chunk, :].clone().requires_grad_(True)

    output_cp = mixer_cp(x_local)
    output_cp.sum().backward()

    # Gather outputs and undo DualChunkSwap to restore sequential order
    half_seq = x_local.shape[1]
    output_gathered = [
        torch.zeros(batch_size, half_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    grad_gathered = [
        torch.zeros(batch_size, half_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    dist.all_gather(output_gathered, output_cp.contiguous())
    dist.all_gather(grad_gathered, x_local.grad.contiguous())

    out_cp_full = _undo_attention_load_balancing(torch.cat(output_gathered, dim=1), world_size)
    grad_cp_full = _undo_attention_load_balancing(torch.cat(grad_gathered, dim=1), world_size)

    param_grad_cp = mixer_cp.in_proj.weight.grad.detach().clone()
    dist.all_reduce(param_grad_cp, op=dist.ReduceOp.SUM)

    return _compare_results(
        "bshd_sdpa",
        rank,
        out_cp_full,
        out_base,
        grad_cp_full,
        grad_base,
        param_grad_cp,
        param_grad_base,
        "in_proj.weight",
        output_atol=0.01,
        output_rtol=1e-2,
        grad_atol=0.05,
        grad_rtol=2e-2,
        param_atol=1.5,
        param_rtol=5e-2,
    )


def main():
    init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: This test requires exactly 2 GPUs, got {world_size}", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    config = MockNemotronV3Config()

    configs = {
        "bshd_te": lambda: run_bshd_te(rank, world_size, device, config),
        "thd_te": lambda: run_thd_te(rank, world_size, device, config),
        "thd_te_packed": lambda: run_thd_te_packed(rank, world_size, device, config),
        "bshd_sdpa": lambda: run_bshd_sdpa(rank, world_size, device, config),
    }

    results = {}
    for name, fn in configs.items():
        dist.barrier()
        try:
            results[name] = fn()
        except Exception as e:
            if rank == 0:
                import traceback

                print(f"  {name}: ERROR - {e}")
                traceback.print_exc()
            results[name] = 1

    if rank == 0:
        print(f"\n{'=' * 70}")
        print("Summary - NemotronV3 Mamba2Mixer CP Tests")
        print(f"{'=' * 70}")
        for name, result in results.items():
            status = "PASSED" if result == 0 else "FAILED"
            print(f"  {name}: {status}")
        print(f"{'=' * 70}\n")

    if dist.is_initialized():
        dist.barrier()
    sys.exit(1 if any(r != 0 for r in results.values()) else 0)


if __name__ == "__main__":
    main()
