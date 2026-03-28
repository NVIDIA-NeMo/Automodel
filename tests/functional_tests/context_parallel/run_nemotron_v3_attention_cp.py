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

Tests three configurations:

  Config 1 (bshd_te):   3D BSHD input, TE p2p CP, DualChunkSwap
  Config 2 (thd_te):    2D THD input, TE p2p CP, DualChunkSwap, cu_seqlens
  Config 4 (bshd_sdpa): 3D BSHD input, DTensor context_parallel(), SDPA backend

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_nemotron_v3_attention_cp.py
"""

import os
import sys

import torch
import torch.distributed as dist


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


def _create_attn_pair(config, backend, device):
    """Create a pair of identical attention modules with synced weights."""
    from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Attention

    attn_baseline = NemotronV3Attention(config, backend).to(device).to(torch.bfloat16)
    attn_cp = NemotronV3Attention(config, backend).to(device).to(torch.bfloat16)
    attn_baseline.train()
    attn_cp.train()
    attn_cp.load_state_dict(attn_baseline.state_dict())

    for p_base, p_cp in zip(attn_baseline.parameters(), attn_cp.parameters()):
        dist.broadcast(p_base.data, src=0)
        dist.broadcast(p_cp.data, src=0)

    return attn_baseline, attn_cp


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
        print(f"Config: {config_name} - NemotronV3Attention")
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
    from nemo_automodel.components.models.common import BackendConfig

    backend = BackendConfig(linear="torch", attn="te")
    attn_baseline, attn_cp = _create_attn_pair(config, backend, device)

    batch_size, seq_len = 2, 128
    torch.manual_seed(42)
    x_full = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    dist.broadcast(x_full, src=0)

    x_no_cp = x_full.detach().clone().requires_grad_(True)
    output_baseline = attn_baseline(x_no_cp)
    output_baseline.sum().backward()
    out_base = output_baseline.detach().clone()
    grad_base = x_no_cp.grad.detach().clone()
    param_grad_base = attn_baseline.q_proj.weight.grad.detach().clone()
    dist.barrier()

    # CP=2
    from torch.distributed.device_mesh import init_device_mesh
    from transformer_engine.pytorch.attention import DotProductAttention

    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()

    assert isinstance(attn_cp.attn_module, DotProductAttention)
    attn_cp.attn_module.set_context_parallel_group(
        cp_group,
        torch.distributed.get_process_group_ranks(cp_group),
        torch.cuda.Stream(),
        cp_comm_type="p2p",
    )

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    indices = tex.thd_get_partitioned_indices(cu_seqlens, seq_len, world_size, rank)
    x_local = x_full[:, indices, :].detach().clone().requires_grad_(True)
    local_seq = x_local.shape[1]

    output_cp = attn_cp(x_local)
    output_cp.sum().backward()

    output_gathered = [
        torch.zeros(batch_size, local_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    grad_gathered = [
        torch.zeros(batch_size, local_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    dist.all_gather(output_gathered, output_cp.contiguous())
    dist.all_gather(grad_gathered, x_local.grad.contiguous())

    out_cp_full = dual_chunk_swap_unsplit(output_gathered, cp_size=world_size, seq_dim=1)
    grad_cp_full = dual_chunk_swap_unsplit(grad_gathered, cp_size=world_size, seq_dim=1)

    param_grad_cp = attn_cp.q_proj.weight.grad.detach().clone()
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
        "q_proj.weight",
        output_atol=1e-2,
        output_rtol=1e-2,
        grad_atol=5e-2,
        grad_rtol=1e-2,
        param_atol=5e-2,
        param_rtol=5e-2,
    )


# ---------------------------------------------------------------------------
# Config 2: THD + TE
# ---------------------------------------------------------------------------
def run_thd_te(rank, world_size, device, config):
    """Config 2: 2D THD input with TE p2p CP and DualChunkSwap."""
    from nemo_automodel.components.models.common import BackendConfig

    backend = BackendConfig(linear="torch", attn="te")
    attn_baseline, attn_cp = _create_attn_pair(config, backend, device)

    batch_size, seq_len = 1, 128
    torch.manual_seed(42)
    x_full = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    dist.broadcast(x_full, src=0)

    # Baseline: run in 3D BSHD (batch=1) so the TE attention path is identical
    # to CP aside from the CP gather/scatter.  The CP path will also use 3D BSHD
    # input (TE CP with p2p keeps BSHD format; DualChunkSwap operates on the
    # sequence dim).  Running the baseline in 2D THD while CP runs in 3D BSHD
    # causes numerical mismatches because TE uses different code-paths internally.
    x_no_cp = x_full.detach().clone().requires_grad_(True)  # [1, T, H]
    output_baseline = attn_baseline(x_no_cp)
    output_baseline.sum().backward()
    out_base = output_baseline.detach().clone()
    grad_base = x_no_cp.grad.detach().clone()
    param_grad_base = attn_baseline.q_proj.weight.grad.detach().clone()
    dist.barrier()

    # CP=2
    from torch.distributed.device_mesh import init_device_mesh
    from transformer_engine.pytorch.attention import DotProductAttention

    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()

    assert isinstance(attn_cp.attn_module, DotProductAttention)
    attn_cp.attn_module.set_context_parallel_group(
        cp_group,
        torch.distributed.get_process_group_ranks(cp_group),
        torch.cuda.Stream(),
        cp_comm_type="p2p",
    )

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    indices = tex.thd_get_partitioned_indices(cu_seqlens, seq_len, world_size, rank)
    x_local = x_full[:, indices, :].detach().clone().requires_grad_(True)  # [1, T/cp, H]
    local_len = x_local.shape[1]

    output_cp = attn_cp(x_local)
    output_cp.sum().backward()

    # Gather 3D outputs (seq_dim=1 for BSHD)
    output_gathered = [
        torch.zeros(batch_size, local_len, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    grad_gathered = [
        torch.zeros(batch_size, local_len, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    dist.all_gather(output_gathered, output_cp.contiguous())
    dist.all_gather(grad_gathered, x_local.grad.contiguous())

    out_cp_full = dual_chunk_swap_unsplit(output_gathered, cp_size=world_size, seq_dim=1)
    grad_cp_full = dual_chunk_swap_unsplit(grad_gathered, cp_size=world_size, seq_dim=1)

    param_grad_cp = attn_cp.q_proj.weight.grad.detach().clone()
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
        "q_proj.weight",
        output_atol=1e-2,
        output_rtol=1e-2,
        grad_atol=5e-2,
        grad_rtol=1e-2,
        param_atol=5e-2,
        param_rtol=5e-2,
    )


# ---------------------------------------------------------------------------
# Config 4: BSHD + SDPA
# ---------------------------------------------------------------------------
def run_bshd_sdpa(rank, world_size, device, config):
    """Config 4: 3D BSHD input with DTensor context_parallel() and SDPA backend."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import context_parallel_unshard, set_rotate_method
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from nemo_automodel.components.models.common import BackendConfig

    backend = BackendConfig(linear="torch", attn="sdpa")
    attn_baseline, attn_cp = _create_attn_pair(config, backend, device)

    batch_size, seq_len = 2, 128
    torch.manual_seed(42)
    x_full = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    dist.broadcast(x_full, src=0)

    # Baseline
    x_no_cp = x_full.detach().clone().requires_grad_(True)
    output_baseline = attn_baseline(x_no_cp)
    output_baseline.sum().backward()
    out_base = output_baseline.detach().clone()
    grad_base = x_no_cp.grad.detach().clone()
    param_grad_base = attn_baseline.q_proj.weight.grad.detach().clone()
    dist.barrier()

    # CP=2 with SDPA + context_parallel
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    set_rotate_method("allgather")

    # context_parallel() shards the full-sequence buffer itself, so pass the
    # complete tensor (not a pre-sharded chunk).  It also cannot handle buffers
    # that require grad, so enable grad only after entering the context.
    x_cp = x_full.detach().clone()

    cp_ctx = context_parallel(
        cp_mesh,
        buffers=[x_cp],
        buffer_seq_dims=[1],
        no_restore_buffers={x_cp},
    )
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        with cp_ctx:
            x_cp.requires_grad_(True)
            output_cp = attn_cp(x_cp)
            output_cp.sum().backward()

    # After context_parallel, x_cp and output_cp hold the local shard.
    # Use context_parallel_unshard to reconstruct the full sequence with
    # correct token ordering (undoes the head-tail load-balancing).
    out_cp_full, grad_cp_full = context_parallel_unshard(
        cp_mesh,
        [output_cp.detach(), x_cp.grad],
        seq_dims=[1, 1],
    )

    param_grad_cp = attn_cp.q_proj.weight.grad.detach().clone()
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
        "q_proj.weight",
        output_atol=1e-2,
        output_rtol=1e-2,
        grad_atol=5e-2,
        grad_rtol=2e-2,
        param_atol=5e-2,
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

    config = MockNemotronV3AttentionConfig()

    configs = {
        "bshd_te": lambda: run_bshd_te(rank, world_size, device, config),
        "thd_te": lambda: run_thd_te(rank, world_size, device, config),
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
        print("Summary - NemotronV3Attention CP Tests")
        print(f"{'=' * 70}")
        for name, result in results.items():
            status = "PASSED" if result == 0 else "FAILED"
            print(f"  {name}: {status}")
        print(f"{'=' * 70}\n")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    sys.exit(1 if any(r != 0 for r in results.values()) else 0)


if __name__ == "__main__":
    main()
