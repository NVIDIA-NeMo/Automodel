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

"""Standalone test script for Qwen3.5 MoE linear attention context parallelism.

Validates that CPAwareGatedDeltaNet produces identical forward outputs and
gradients when running with CP=2 vs the baseline HF forward with CP=1.

The linear attention CP path is fundamentally different from TE attention CP:
  - Works in BSHD format (not THD packed sequences)
  - Undoes PyTorch's load-balanced CP layout to dense order
  - Runs causal conv1d with cross-rank P2P boundary exchange
  - Runs FLA gated delta rule with FLA's CP context
  - Restores output to load-balanced CP layout

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_linear_attn_cp.py
"""

import os
import sys

import torch
import torch.distributed as dist


def init_distributed():
    """Initialize distributed environment."""
    if not (dist.is_available() and dist.is_initialized()):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def run_test():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: This test requires exactly 2 GPUs, got {world_size}", file=sys.stderr)
        return 1

    # -- import Qwen3.5 MoE dependencies --
    try:
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    except ImportError:
        if rank == 0:
            print("ERROR: transformers does not have qwen3_5_moe support", file=sys.stderr)
        return 1

    try:
        import fla  # noqa: F401
    except ImportError:
        if rank == 0:
            print("ERROR: fla library is required but not installed", file=sys.stderr)
        return 1

    from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

    # -- model config (small for testing) --
    config = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        layer_types=["full_attention", "linear_attention"],
    )

    # -- create two identical modules --
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    module_no_cp = CPAwareGatedDeltaNet(config, layer_idx=1).to(device).to(torch.bfloat16)
    module_with_cp = CPAwareGatedDeltaNet(config, layer_idx=1).to(device).to(torch.bfloat16)
    module_with_cp.load_state_dict(module_no_cp.state_dict())

    # broadcast weights so both ranks are identical
    for p_no, p_cp in zip(module_no_cp.parameters(), module_with_cp.parameters()):
        dist.broadcast(p_no.data, src=0)
        dist.broadcast(p_cp.data, src=0)

    module_no_cp.eval()
    module_with_cp.eval()

    # -- create input data --
    B = 2
    S_global = 32  # total sequence length (must be divisible by world_size)
    S_local = S_global // world_size
    D = config.hidden_size

    torch.manual_seed(42 + rank)
    # Full input (same on all ranks for baseline)
    torch.manual_seed(42)
    x_full = torch.randn(B, S_global, D, device=device, dtype=torch.bfloat16)

    from torch.distributed.device_mesh import init_device_mesh

    # ===== Baseline: CP=1 using _forward_with_cp on full sequence =====
    # We call _forward_with_cp directly (instead of super().forward()) so that
    # both baseline and CP=2 exercise the same code path / autograd graph.
    # The only variable being tested is the distributed CP communication.
    x_baseline = x_full.clone().detach().requires_grad_(True)

    # Dense positions for full sequence — no load-balancing needed
    dense_positions = torch.arange(S_global, device=device, dtype=torch.long)

    # Create a single-rank process group containing only this rank, so
    # _forward_with_cp runs the same code path as CP=2 but with all
    # distributed ops (all_gather, P2P) degenerating to single-rank no-ops.
    baseline_group = dist.new_group(ranks=[rank])

    class _SingleRankMesh:
        """Minimal mesh-like object wrapping a single-rank process group."""

        def size(self):
            return dist.get_world_size(baseline_group)

        def get_group(self):
            return baseline_group

    module_no_cp._cp_mesh = _SingleRankMesh()

    output_baseline = module_no_cp._forward_with_cp(
        x_baseline,
        position_ids=dense_positions.unsqueeze(0).expand(B, -1),
        seq_index=dense_positions,
    )
    loss_baseline = output_baseline.sum()
    loss_baseline.backward()

    output_baseline_detached = output_baseline.detach().clone()
    grad_baseline_detached = x_baseline.grad.detach().clone()

    # Reset _cp_mesh so module_no_cp isn't accidentally reused with CP
    module_no_cp._cp_mesh = None

    dist.barrier()

    # ===== Test: CP=2 =====

    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    module_with_cp._cp_mesh = cp_mesh["cp"]

    # Simulate load-balanced CP sharding:
    # PyTorch CP uses interleaved load-balancing: rank 0 gets even positions,
    # rank 1 gets odd positions (for world_size=2).
    all_positions = torch.arange(S_global, device=device, dtype=torch.long)
    # Interleaved assignment: position i goes to rank (i % world_size)
    local_mask = (all_positions % world_size) == rank
    local_positions = all_positions[local_mask]  # [S_local]

    assert local_positions.shape[0] == S_local, (
        f"Expected {S_local} local positions, got {local_positions.shape[0]}"
    )

    # Shard input according to load-balanced positions
    x_local = x_full[:, local_mask, :].clone().detach().requires_grad_(True)

    # position_ids for the local shard (2D: [B, S_local])
    position_ids = local_positions.unsqueeze(0).expand(B, -1)
    # seq_index is the same as local_positions (1D)
    seq_index = local_positions

    output_cp = module_with_cp.forward(
        x_local,
        position_ids=position_ids,
        seq_index=seq_index,
    )

    loss_cp = output_cp.sum()
    loss_cp.backward()

    # -- gather CP outputs from all ranks --
    output_gathered = [torch.zeros_like(output_cp) for _ in range(world_size)]
    grad_gathered = [torch.zeros_like(x_local.grad) for _ in range(world_size)]
    positions_gathered = [torch.zeros_like(local_positions) for _ in range(world_size)]

    dist.all_gather(output_gathered, output_cp.detach())
    dist.all_gather(grad_gathered, x_local.grad.detach())
    dist.all_gather(positions_gathered, local_positions)

    # Reconstruct full output in dense order
    all_pos = torch.cat(positions_gathered, dim=0)
    sort_order = torch.argsort(all_pos)

    output_cp_cat = torch.cat(output_gathered, dim=1)  # [B, S_global, D]
    grad_cp_cat = torch.cat(grad_gathered, dim=1)      # [B, S_global, D]

    # Reorder from interleaved to dense
    output_cp_full = output_cp_cat[:, sort_order, :]
    grad_cp_full = grad_cp_cat[:, sort_order, :]

    # ===== Compare =====
    if rank == 0:
        output_diff = (output_cp_full - output_baseline_detached).abs()
        grad_diff = (grad_cp_full - grad_baseline_detached).abs()

        print(f"\n{'=' * 70}")
        print("Context Parallelism Validation - Qwen3.5 MoE Linear Attention")
        print(f"{'=' * 70}")
        print(f"Config: B={B}, S_global={S_global}, D={D}, CP={world_size}")
        print(f"Output shape: CP={output_cp_full.shape}, Baseline={output_baseline_detached.shape}")
        print(f"Output diff - mean: {output_diff.mean():.6f}, max: {output_diff.max():.6f}")
        print(f"Grad diff   - mean: {grad_diff.mean():.6f}, max: {grad_diff.max():.6f}")

    try:
        torch.testing.assert_close(
            output_cp_full,
            output_baseline_detached,
            rtol=1e-2,
            atol=0.01,
            msg=f"[Rank {rank}] Forward outputs differ between CP=1 and CP=2",
        )

        torch.testing.assert_close(
            grad_cp_full,
            grad_baseline_detached,
            rtol=2e-2,
            atol=0.05,
            msg=f"[Rank {rank}] Gradients differ between CP=1 and CP=2",
        )

        if rank == 0:
            print("PASSED: Forward outputs and gradients match between CP=1 and CP=2")
            print(f"{'=' * 70}\n")
        return 0

    except AssertionError as e:
        if rank == 0:
            print(f"FAILED: {e}")
            print(f"{'=' * 70}\n")
        return 1


def main():
    init_distributed()
    exit_code = run_test()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
