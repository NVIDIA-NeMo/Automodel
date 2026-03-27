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

"""End-to-end hybrid NemotronV3 CP test.

Validates that a hybrid model with interleaved attention and mamba layers
produces matching outputs/gradients between CP=1 and CP=2 across four
configurations:

  Config 1 (bshd_te):        3D BSHD input, TE p2p CP, DualChunkSwap
  Config 2 (thd_te):         2D THD input, TE p2p CP, DualChunkSwap, cu_seqlens
  Config 3 (thd_te_packed):  2D THD input, TE p2p CP, multi-sequence packing, seq_idx
  Config 4 (bshd_sdpa):      3D BSHD input, DTensor context_parallel(), SDPA backend

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_hybrid_nemotron_v3_cp.py
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


class MockHybridConfig:
    """Mock configuration for a hybrid NemotronV3 model (attention + mamba layers).

    Provides only the fields required by NemotronV3Model and its block types.
    MoE-related fields are still required because NemotronV3Model constructs
    a MoEConfig in __init__ regardless of layer types; they are set to minimal
    values that avoid errors without activating MoE layers.
    """

    def __init__(self):
        # Attention config
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.head_dim = 32
        self.hidden_size = 256  # num_attention_heads * head_dim
        self.attention_bias = False
        self.attention_dropout = 0.0

        # Mamba config
        self.mamba_num_heads = 8
        self.mamba_head_dim = 32
        self.ssm_state_size = 16
        self.n_groups = 2  # must be >= cp_size for non-replicated mode
        self.chunk_size = 256
        self.conv_kernel = 4
        self.use_conv_bias = True
        self.mamba_hidden_act = "silu"
        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.use_bias = False

        # Shared norm / model config
        self.layer_norm_epsilon = 1e-5
        self.num_hidden_layers = 4
        self.vocab_size = 128
        self.torch_dtype = "bfloat16"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False

        # Hybrid layer schedule: interleaved attention and mamba
        self.layers_block_type = ["attention", "mamba", "attention", "mamba"]

        # MLP config (required by MLP block type, kept here for completeness)
        self.intermediate_size = 512
        self.mlp_bias = False
        self.mlp_hidden_act = "silu"

        # MoE config fields
        self.n_routed_experts = 1
        self.num_experts_per_tok = 1
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.moe_intermediate_size = self.intermediate_size
        self.norm_topk_prob = False
        self.moe_shared_expert_intermediate_size = self.intermediate_size


def _create_baseline_model(config, backend, device):
    """Create and sync a baseline model (CP=1)."""
    from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

    model = NemotronV3Model(config, backend=backend).to(device=device, dtype=torch.bfloat16)
    model.train()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    return model


def _create_cp_model(config, backend, baseline_model, device):
    """Create a CP model with weights copied from baseline."""
    from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

    model = NemotronV3Model(config, backend=backend).to(device=device, dtype=torch.bfloat16)
    model.train()
    model.load_state_dict(baseline_model.state_dict(), strict=False)
    model.zero_grad()
    return model


def _wire_te_cp(model, cp_group, config):
    """Wire TE-based CP on each hybrid layer (p2p for attention, hidden-parallel for mamba)."""
    from transformer_engine.pytorch.attention import DotProductAttention

    from nemo_automodel.components.distributed.mamba_cp import MambaContextParallel

    for layer in model.layers.values():
        if layer.block_type == "mamba":
            mixer = layer.mixer
            mixer.cp = MambaContextParallel(
                cp_group=cp_group,
                num_heads=mixer.num_heads,
                head_dim=mixer.head_dim,
                n_groups=mixer.n_groups,
                d_state=mixer.ssm_state_size,
                mixer=mixer,
            )
        elif layer.block_type == "attention":
            attn_module = layer.mixer.attn_module
            if isinstance(attn_module, DotProductAttention):
                attn_module.set_context_parallel_group(
                    cp_group,
                    torch.distributed.get_process_group_ranks(cp_group),
                    torch.cuda.Stream(),
                    cp_comm_type="p2p",
                )


def _wire_sdpa_cp(model, cp_group):
    """Wire SDPA-based CP on mamba layers. Attention uses context_parallel().

    MambaContextParallel always undoes/redoes DualChunkSwap around the SSM
    kernel, matching the reordering applied by both TE CP and PyTorch's
    context_parallel(allgather).
    """
    from nemo_automodel.components.distributed.mamba_cp import MambaContextParallel

    for layer in model.layers.values():
        if layer.block_type == "mamba":
            mixer = layer.mixer
            mixer.cp = MambaContextParallel(
                cp_group=cp_group,
                num_heads=mixer.num_heads,
                head_dim=mixer.head_dim,
                n_groups=mixer.n_groups,
                d_state=mixer.ssm_state_size,
                mixer=mixer,
            )
        # Attention layers use DTensor context_parallel() -- no explicit CP wiring needed


def _compare_results(
    config_name,
    rank,
    output_cp_full,
    output_baseline,
    grad_cp,
    grad_baseline,
    output_atol,
    output_rtol,
    grad_atol,
    grad_rtol,
):
    """Compare CP vs baseline results and return 0 on pass, 1 on fail."""
    output_diff = (output_cp_full - output_baseline).abs()
    grad_diff = (grad_cp - grad_baseline).abs()

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"Config: {config_name} - Hybrid NemotronV3 (Attention + Mamba)")
        print(f"{'=' * 70}")
        print(f"Output shape: CP={output_cp_full.shape}, Baseline={output_baseline.shape}")
        print(f"Output diff - mean: {output_diff.mean().item():.6f}, max: {output_diff.max().item():.6f}")
        print(f"Embed grad diff - mean: {grad_diff.mean().item():.6f}, max: {grad_diff.max().item():.6f}")

    try:
        torch.testing.assert_close(
            output_cp_full,
            output_baseline,
            rtol=output_rtol,
            atol=output_atol,
            msg=f"[{config_name}][Rank {rank}] Forward outputs differ",
        )
        torch.testing.assert_close(
            grad_cp,
            grad_baseline,
            rtol=grad_rtol,
            atol=grad_atol,
            msg=f"[{config_name}][Rank {rank}] embed_tokens.weight.grad differs",
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

    from nemo_automodel.components.models.common import BackendConfig

    backend = BackendConfig(linear="torch", attn="te", rms_norm="torch", enable_hf_state_dict_adapter=False)

    model_baseline = _create_baseline_model(config, backend, device)

    batch_size, seq_len = 2, 128
    torch.manual_seed(42)
    input_ids_full = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    dist.broadcast(input_ids_full, src=0)

    output_baseline = model_baseline(input_ids=input_ids_full)
    output_baseline.sum().backward()
    out_base = output_baseline.detach().clone()
    embed_grad_base = model_baseline.embed_tokens.weight.grad.detach().clone()
    dist.barrier()

    # CP=2
    model_cp = _create_cp_model(config, backend, model_baseline, device)
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()
    _wire_te_cp(model_cp, cp_group, config)

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    indices = tex.thd_get_partitioned_indices(cu_seqlens, seq_len, world_size, rank)
    input_ids_local = input_ids_full[:, indices]

    output_cp_local = model_cp(input_ids=input_ids_local)
    output_cp_local.sum().backward()

    local_seq = output_cp_local.shape[1]
    output_gathered = [
        torch.zeros(batch_size, local_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    dist.all_gather(output_gathered, output_cp_local.detach().contiguous(), group=cp_group)
    out_cp_full = dual_chunk_swap_unsplit(output_gathered, cp_size=world_size, seq_dim=1)

    embed_grad_cp = model_cp.embed_tokens.weight.grad.detach().clone()
    dist.all_reduce(embed_grad_cp, op=dist.ReduceOp.SUM, group=cp_group)

    return _compare_results(
        "bshd_te",
        rank,
        out_cp_full,
        out_base,
        embed_grad_cp,
        embed_grad_base,
        output_atol=5e-2,
        output_rtol=1e-2,
        grad_atol=1e-1,
        grad_rtol=5e-2,
    )


# ---------------------------------------------------------------------------
# Config 2: THD + TE
# ---------------------------------------------------------------------------
def run_thd_te(rank, world_size, device, config):
    """Config 2: 2D THD input with TE p2p CP and DualChunkSwap."""
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.models.common import BackendConfig

    backend = BackendConfig(linear="torch", attn="te", rms_norm="torch", enable_hf_state_dict_adapter=False)

    model_baseline = _create_baseline_model(config, backend, device)

    batch_size, seq_len = 1, 128
    torch.manual_seed(42)
    input_ids_full = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    dist.broadcast(input_ids_full, src=0)

    # Baseline: use batch=1 BSHD path (model.forward expects input_ids)
    output_baseline = model_baseline(input_ids=input_ids_full)
    output_baseline.sum().backward()
    out_base = output_baseline.detach().clone()
    embed_grad_base = model_baseline.embed_tokens.weight.grad.detach().clone()
    dist.barrier()

    # CP=2
    model_cp = _create_cp_model(config, backend, model_baseline, device)
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()
    _wire_te_cp(model_cp, cp_group, config)

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    indices = tex.thd_get_partitioned_indices(cu_seqlens, seq_len, world_size, rank)
    input_ids_local = input_ids_full[:, indices]

    # TE CP with p2p operates in BSHD format; do NOT pass cu_seqlens here
    # (passing cu_seqlens triggers THD squeeze in NemotronV3Model.forward which
    # is incompatible with TE p2p CP).  Single-sequence batch_size=1 BSHD is
    # numerically equivalent.
    output_cp_local = model_cp(input_ids=input_ids_local)
    output_cp_local.sum().backward()

    local_seq = output_cp_local.shape[1]
    output_gathered = [
        torch.zeros(batch_size, local_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    dist.all_gather(output_gathered, output_cp_local.detach().contiguous(), group=cp_group)
    out_cp_full = dual_chunk_swap_unsplit(output_gathered, cp_size=world_size, seq_dim=1)

    embed_grad_cp = model_cp.embed_tokens.weight.grad.detach().clone()
    dist.all_reduce(embed_grad_cp, op=dist.ReduceOp.SUM, group=cp_group)

    return _compare_results(
        "thd_te",
        rank,
        out_cp_full,
        out_base,
        embed_grad_cp,
        embed_grad_base,
        output_atol=5e-2,
        output_rtol=1e-2,
        grad_atol=1e-1,
        grad_rtol=5e-2,
    )


# ---------------------------------------------------------------------------
# Config 3: THD + TE + packing
# ---------------------------------------------------------------------------
def run_thd_te_packed(rank, world_size, device, config):
    """Config 3: 2D THD with TE p2p CP, multi-sequence packing, and seq_idx."""
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.models.common import BackendConfig

    backend = BackendConfig(linear="torch", attn="te", rms_norm="torch", enable_hf_state_dict_adapter=False)

    model_baseline = _create_baseline_model(config, backend, device)

    # Two packed sequences: each 64 tokens for total 128.
    # For the hybrid model, the mamba layers need sequence lengths divisible
    # by 2 * cp_size = 4. 64 satisfies this.
    seq_len_a, seq_len_b = 64, 64
    total_len = seq_len_a + seq_len_b

    torch.manual_seed(42)
    input_ids_full = torch.randint(0, config.vocab_size, (1, total_len), device=device)
    dist.broadcast(input_ids_full, src=0)

    # Baseline: pass seq_idx (not cu_seqlens) to keep BSHD format for attention
    # while letting mamba layers know about packed sequence boundaries.
    # Using cu_seqlens would trigger THD squeeze in NemotronV3Model.forward,
    # causing TE to use a different code path than the BSHD CP run.
    cu_seqlens_full = torch.tensor([0, seq_len_a, total_len], dtype=torch.int32, device=device)
    positions_full = torch.arange(total_len, device=device)
    seq_idx_full = torch.searchsorted(cu_seqlens_full[1:], positions_full).unsqueeze(0).to(torch.int32)
    output_baseline = model_baseline(input_ids=input_ids_full, seq_idx=seq_idx_full)
    output_baseline.sum().backward()
    out_base = output_baseline.detach().clone()
    embed_grad_base = model_baseline.embed_tokens.weight.grad.detach().clone()
    dist.barrier()

    # CP=2
    model_cp = _create_cp_model(config, backend, model_baseline, device)
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()
    _wire_te_cp(model_cp, cp_group, config)

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex

    # TE CP with p2p operates in BSHD format; passing cu_seqlens to the model
    # triggers THD squeeze which is incompatible.  Use single-sequence DCS
    # indices (treating the entire packed sequence as one) so that the attention
    # DCS reordering matches the full-causal mask applied by BSHD attention.
    cu_seqlens_single = torch.tensor([0, total_len], dtype=torch.int32, device=device)
    indices = tex.thd_get_partitioned_indices(cu_seqlens_single, total_len, world_size, rank)
    input_ids_local = input_ids_full[:, indices]
    local_len = input_ids_local.shape[1]

    # Pre-compute seq_idx so mamba layers know about packed sequence boundaries.
    # The mamba kernel sees the global sequence (after all-to-all gather), so
    # seq_idx must cover the full (global) sequence length.
    positions = torch.arange(total_len, device=device)
    seq_idx = torch.searchsorted(cu_seqlens_full[1:], positions).unsqueeze(0).to(torch.int32)

    output_cp_local = model_cp(input_ids=input_ids_local, seq_idx=seq_idx)
    output_cp_local.sum().backward()

    output_gathered = [
        torch.zeros(1, local_len, config.hidden_size, device=device, dtype=torch.bfloat16) for _ in range(world_size)
    ]
    dist.all_gather(output_gathered, output_cp_local.detach().contiguous(), group=cp_group)
    out_cp_full = dual_chunk_swap_unsplit(output_gathered, cp_size=world_size, seq_dim=1)

    embed_grad_cp = model_cp.embed_tokens.weight.grad.detach().clone()
    dist.all_reduce(embed_grad_cp, op=dist.ReduceOp.SUM, group=cp_group)

    return _compare_results(
        "thd_te_packed",
        rank,
        out_cp_full,
        out_base,
        embed_grad_cp,
        embed_grad_base,
        output_atol=1e-1,
        output_rtol=2e-2,
        grad_atol=2e-1,
        grad_rtol=1e-1,
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

    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", enable_hf_state_dict_adapter=False)

    model_baseline = _create_baseline_model(config, backend, device)

    batch_size, seq_len = 2, 128
    torch.manual_seed(42)
    input_ids_full = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    dist.broadcast(input_ids_full, src=0)

    output_baseline = model_baseline(input_ids=input_ids_full)
    output_baseline.sum().backward()
    out_base = output_baseline.detach().clone()
    embed_grad_base = model_baseline.embed_tokens.weight.grad.detach().clone()
    dist.barrier()

    # CP=2 with SDPA + context_parallel
    model_cp = _create_cp_model(config, backend, model_baseline, device)
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()
    _wire_sdpa_cp(model_cp, cp_group)

    set_rotate_method("allgather")

    # context_parallel() shards the full-sequence buffer itself, so pass the
    # complete embedding (not a pre-sharded chunk).  Embed on the full input_ids
    # (all ranks have the same data) and let context_parallel handle sharding.
    with torch.no_grad():
        x_full_embed = model_cp.embed_tokens(input_ids_full)
    x_cp = x_full_embed.detach().clone()

    # context_parallel() cannot handle buffers that require grad, so enable
    # grad only after entering the context.
    cp_ctx = context_parallel(
        cp_mesh,
        buffers=[x_cp],
        buffer_seq_dims=[1],
        no_restore_buffers={x_cp},
    )
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        with cp_ctx:
            x_cp.requires_grad_(True)
            # Forward through layers directly using inputs_embeds
            hidden_states = x_cp
            for layer in model_cp.layers.values():
                hidden_states = layer(hidden_states)
            hidden_states = model_cp.norm(hidden_states)
            output_cp_local = hidden_states
            # backward() must run inside cp_ctx so that the ring-attention
            # backward hooks registered by context_parallel are still active.
            output_cp_local.sum().backward()

    # After context_parallel, output_cp_local holds the local shard.
    # Use context_parallel_unshard to reconstruct the full sequence with
    # correct token ordering (undoes the head-tail load-balancing).
    (out_cp_full,) = context_parallel_unshard(
        cp_mesh,
        [output_cp_local.detach()],
        seq_dims=[1],
    )

    # Embedding was not in the backward graph (we used detach + inputs_embeds),
    # so compare output only and use matching zeros for the grad comparison.
    embed_grad_zeros = torch.zeros_like(embed_grad_base)

    return _compare_results(
        "bshd_sdpa",
        rank,
        out_cp_full,
        out_base,
        embed_grad_zeros,
        embed_grad_zeros,
        output_atol=5e-2,
        output_rtol=1e-2,
        grad_atol=1e-1,
        grad_rtol=5e-2,
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

    config = MockHybridConfig()

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
                print(f"  {name}: ERROR - {e}")
            results[name] = 1

    if rank == 0:
        print(f"\n{'=' * 70}")
        print("Summary - Hybrid NemotronV3 CP Tests")
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
