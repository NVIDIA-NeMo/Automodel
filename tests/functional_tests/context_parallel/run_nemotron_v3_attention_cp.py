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

"""NemotronV3Attention CP validation: verifies CP=2 matches CP=1 for forward + grads.

Uses BSHD format with TE p2p CP and DualChunkSwap sequence distribution. Each rank
receives two non-contiguous chunks selected via _dual_chunk_swap_select. TE p2p CP
ring-exchanges K,V between neighboring ranks and applies the correct global causal
mask. This matches the production code path in _split_batch_bshd_for_cp /
NemotronHParallelizationStrategy.

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_nemotron_v3_attention_cp.py
"""

import os
import sys

import torch
import torch.distributed as dist

from nemo_automodel.components.distributed.cp_utils import _dual_chunk_swap_select


def dual_chunk_swap_unsplit(chunks_per_rank, cp_size, seq_dim=1):
    """Reconstruct full sequence from DualChunkSwap-ordered rank outputs."""
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


class MockNemotronV3AttentionConfig:
    """Mock configuration for NemotronV3Attention."""

    def __init__(self):
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.head_dim = 32
        self.hidden_size = 256  # num_attention_heads * head_dim
        self.attention_bias = False
        self.attention_dropout = 0.0


def run_test():
    """Run the CP validation test for NemotronV3Attention."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: This test requires exactly 2 GPUs, got {world_size}", file=sys.stderr)
        return 1

    try:
        import transformer_engine.pytorch  # noqa: F401
    except ImportError:
        if rank == 0:
            print("ERROR: transformer_engine is required but not installed", file=sys.stderr)
        return 1

    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Attention
    from nemo_automodel.components.models.common import BackendConfig

    config = MockNemotronV3AttentionConfig()
    backend = BackendConfig(linear="torch", attn="te")

    attn_no_cp = NemotronV3Attention(config, backend).to(device).to(torch.bfloat16)
    attn_with_cp = NemotronV3Attention(config, backend).to(device).to(torch.bfloat16)

    attn_no_cp.train()
    attn_with_cp.train()

    attn_with_cp.load_state_dict(attn_no_cp.state_dict())
    for param_no_cp, param_with_cp in zip(attn_no_cp.parameters(), attn_with_cp.parameters()):
        dist.broadcast(param_no_cp.data, src=0)
        dist.broadcast(param_with_cp.data, src=0)

    # ===== Baseline: CP=1 (no context parallelism) =====
    batch_size = 2
    seq_len = 128

    torch.manual_seed(42)
    x_full = torch.randn(
        batch_size, seq_len, config.hidden_size,
        device=device, dtype=torch.bfloat16,
    )
    dist.broadcast(x_full, src=0)

    x_no_cp = x_full.detach().clone().requires_grad_(True)

    output_no_cp = attn_no_cp(x_no_cp)
    loss_no_cp = output_no_cp.sum()
    loss_no_cp.backward()

    output_baseline = output_no_cp.detach().clone()
    grad_baseline = x_no_cp.grad.detach().clone()
    q_proj_grad_baseline = attn_no_cp.q_proj.weight.grad.detach().clone()

    dist.barrier()

    # ===== Test: CP=2 (context parallelism with p2p / DualChunkSwap) =====
    # DualChunkSwap split: rank r gets chunks [r] and [2*cp_size-r-1] from the
    # sequence partitioned into 2*cp_size equal pieces.
    # TE p2p CP ring-exchanges K,V between neighboring ranks and applies the
    # correct global causal mask.
    from torch.distributed.device_mesh import init_device_mesh
    from transformer_engine.pytorch.attention import DotProductAttention

    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()

    assert isinstance(attn_with_cp.attn_module, DotProductAttention)

    attn_with_cp.attn_module.set_context_parallel_group(
        cp_group,
        torch.distributed.get_process_group_ranks(cp_group),
        torch.cuda.Stream(),
        cp_comm_type="p2p",
    )

    cp_size = world_size
    x_local = _dual_chunk_swap_select(x_full, cp_size=cp_size, cp_rank=rank, seq_dim=1).detach().clone().requires_grad_(True)

    output_with_cp = attn_with_cp(x_local)
    loss_with_cp = output_with_cp.sum()
    loss_with_cp.backward()

    # Gather outputs and input grads from all ranks
    local_seq = x_local.shape[1]
    output_gathered = [
        torch.zeros(batch_size, local_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(cp_size)
    ]
    grad_gathered = [
        torch.zeros(batch_size, local_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(cp_size)
    ]

    dist.all_gather(output_gathered, output_with_cp.contiguous())
    dist.all_gather(grad_gathered, x_local.grad.contiguous())

    output_with_cp_full = dual_chunk_swap_unsplit(output_gathered, cp_size=cp_size, seq_dim=1)
    grad_with_cp_full = dual_chunk_swap_unsplit(grad_gathered, cp_size=cp_size, seq_dim=1)

    q_proj_grad_cp = attn_with_cp.q_proj.weight.grad.detach().clone()
    dist.all_reduce(q_proj_grad_cp, op=dist.ReduceOp.SUM)

    if rank == 0:
        output_diff = (output_with_cp_full - output_baseline).abs()
        grad_diff = (grad_with_cp_full - grad_baseline).abs()
        q_proj_grad_diff = (q_proj_grad_cp - q_proj_grad_baseline).abs()

        print(f"\n{'='*70}")
        print(f"Context Parallelism Validation Test - NemotronV3Attention")
        print(f"{'='*70}")
        print(f"Config: heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
              f"head_dim={config.head_dim}, hidden_size={config.hidden_size}")
        print(f"Sequence: batch={batch_size}, seq_len={seq_len} -> {local_seq} tokens/rank with CP=2")
        print(f"CP comm type: p2p (DualChunkSwap sequence split)")
        print(f"\nForward output statistics:")
        print(f"  Output shape: CP={output_with_cp_full.shape}, Baseline={output_baseline.shape}")
        print(f"  Output diff - mean: {output_diff.mean():.6f}, max: {output_diff.max():.6f}, "
              f"std: {output_diff.std():.6f}")
        print(f"  Output relative diff - mean: {(output_diff / (output_baseline.abs() + 1e-8)).mean():.6f}")
        print(f"\nInput gradient statistics:")
        print(f"  Baseline - min: {grad_baseline.abs().min():.6f}, max: {grad_baseline.abs().max():.6f}, "
              f"mean: {grad_baseline.abs().mean():.6f}")
        print(f"  CP       - min: {grad_with_cp_full.abs().min():.6f}, max: {grad_with_cp_full.abs().max():.6f}, "
              f"mean: {grad_with_cp_full.abs().mean():.6f}")
        print(f"  Grad diff - mean: {grad_diff.mean():.6f}, max: {grad_diff.max():.6f}, "
              f"std: {grad_diff.std():.6f}")
        print(f"\nq_proj.weight.grad statistics:")
        print(f"  Baseline - mean: {q_proj_grad_baseline.abs().mean():.6f}, "
              f"max: {q_proj_grad_baseline.abs().max():.6f}")
        print(f"  CP       - mean: {q_proj_grad_cp.abs().mean():.6f}, "
              f"max: {q_proj_grad_cp.abs().max():.6f}")
        print(f"  Diff     - mean: {q_proj_grad_diff.mean():.6f}, max: {q_proj_grad_diff.max():.6f}")

    try:
        torch.testing.assert_close(
            output_with_cp_full,
            output_baseline,
            rtol=1e-2,
            atol=1e-2,
            msg=f"[Rank {rank}] Forward outputs differ between CP=1 and CP=2",
        )

        torch.testing.assert_close(
            grad_with_cp_full,
            grad_baseline,
            rtol=1e-2,
            atol=5e-2,
            msg=f"[Rank {rank}] Input gradients differ between CP=1 and CP=2",
        )

        torch.testing.assert_close(
            q_proj_grad_cp,
            q_proj_grad_baseline,
            rtol=5e-2,
            atol=5e-2,
            msg=f"[Rank {rank}] q_proj.weight.grad differs between CP=1 and CP=2",
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
        dist.destroy_process_group()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
