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

"""Standalone test script for attention layer context parallelism validation.

This script validates that attention layers produce identical forward outputs
and gradients when using different context parallel sizes with packed sequences.

Supported model types: qwen3_moe, deepseek_v3, nemotron_v3

Supported configs:
  bshd_te   - 3D BSHD input, TE p2p CP, DualChunkSwap
  thd_te    - 2D THD input, TE p2p CP (qwen3/deepseek use make_cp_batch_for_te;
              nemotron_v3 uses DualChunkSwap)
  bshd_sdpa - 3D BSHD input, DTensor context_parallel(), SDPA backend

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_attention_cp.py \
        --model_type qwen3_moe

    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_attention_cp.py \
        --model_type deepseek_v3

    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_attention_cp.py \
        --model_type nemotron_v3

    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_attention_cp.py \
        --model_type nemotron_v3 --configs bshd_te thd_te
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def dual_chunk_swap_unsplit(chunks_per_rank, cp_size, seq_dim=1):
    """Reconstruct full sequence from DualChunkSwap-ordered rank outputs."""
    all_chunks = [None] * (2 * cp_size)
    for rank_idx, rank_output in enumerate(chunks_per_rank):
        c0, c1 = torch.chunk(rank_output, 2, dim=seq_dim)
        all_chunks[rank_idx] = c0
        all_chunks[2 * cp_size - rank_idx - 1] = c1
    return torch.cat(all_chunks, dim=seq_dim)


def is_distributed():
    """Check if we're running in a distributed environment."""
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    """Get the number of processes in the distributed group."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank():
    """Get the current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def init_distributed():
    """Initialize distributed environment."""
    if not is_distributed():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _compare_results(
    config_name,
    model_name,
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
        print(f"Config: {config_name} - {model_name}")
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
# Packed-sequence batch creation (used by qwen3_moe / deepseek_v3 thd_te)
# ---------------------------------------------------------------------------

def create_packed_sequence_batch(batch_size, seq_lens_per_batch, device, padding_token_id=0):
    """
    Create a packed sequence batch for testing.

    Args:
        batch_size: Number of examples in the batch
        seq_lens_per_batch: List of lists, where each inner list contains sequence lengths
            for packed sequences in that batch example
        device: Device to place tensors on
        padding_token_id: Token ID to use for padding

    Returns:
        Dictionary containing batch tensors in BSHD format
    """
    # Calculate total sequence length needed
    max_total_len = max(sum(lens) for lens in seq_lens_per_batch)

    # Create input_ids and labels with padding
    input_ids = torch.full((batch_size, max_total_len), padding_token_id, dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_total_len), padding_token_id, dtype=torch.long, device=device)
    position_ids = torch.zeros((batch_size, max_total_len), dtype=torch.long, device=device)

    # Fill with actual data
    for i, lens in enumerate(seq_lens_per_batch):
        pos = 0
        for seq_len in lens:
            # Fill with non-padding values
            input_ids[i, pos : pos + seq_len] = torch.arange(1, seq_len + 1, device=device)
            labels[i, pos : pos + seq_len] = torch.arange(2, seq_len + 2, device=device)
            # Position IDs restart for each packed sequence
            position_ids[i, pos : pos + seq_len] = torch.arange(seq_len, device=device)
            pos += seq_len

    # Create seq_lens and seq_lens_padded tensors
    max_num_seqs = max(len(lens) for lens in seq_lens_per_batch)
    seq_lens = torch.full((batch_size, max_num_seqs), -1000, dtype=torch.long, device=device)
    seq_lens_padded = torch.full((batch_size, max_num_seqs), -1000, dtype=torch.long, device=device)

    for i, lens in enumerate(seq_lens_per_batch):
        for j, seq_len in enumerate(lens):
            seq_lens[i, j] = seq_len
            # seq_lens_padded should be max_total_len to reflect padding in BSHD format
            seq_lens_padded[i, j] = max_total_len

    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model_config_and_attention(model_type, device):
    """Get model configuration and attention layer based on model type.

    For qwen3_moe / deepseek_v3: returns (config, attn_no_cp, attn_with_cp, get_freqs_cis).
    For nemotron_v3: returns (config, None, None, None) because attention pairs
    are created per-config with different BackendConfig instances.
    """
    if model_type == "qwen3_moe":
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.models.common.utils import get_rope_config
        from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding
        from nemo_automodel.components.models.qwen3_moe.layers import Qwen3MoeAttention

        config = Qwen3MoeConfig(
            vocab_size=256,
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_hidden_layers=2,
            intermediate_size=512,
            moe_intermediate_size=256,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            router_aux_loss_coef=0.01,
            use_sliding_window=False,
        )

        backend = BackendConfig(
            linear="torch",
            attn="te",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

        rope_theta, _, _ = get_rope_config(config)
        rope = RotaryEmbedding(
            head_dim=config.head_dim,
            base=rope_theta,
            dtype=torch.float32,
        )

        attn_no_cp = Qwen3MoeAttention(config, backend).to(device).to(torch.bfloat16)
        attn_with_cp = Qwen3MoeAttention(config, backend).to(device).to(torch.bfloat16)

        from nemo_automodel.components.models.gpt_oss.rope_utils import position_ids_to_freqs_cis

        def get_freqs_cis(position_ids, qkv_format, cp_size=1):
            return position_ids_to_freqs_cis(rope, position_ids, qkv_format, cp_size=cp_size)

    elif model_type == "deepseek_v3":
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.models.common.utils import get_rope_config
        from nemo_automodel.components.models.deepseek_v3.layers import MLA
        from nemo_automodel.components.models.deepseek_v3.rope_utils import (
            freqs_cis_from_position_ids,
            precompute_freqs_cis,
        )

        config = DeepseekV3Config(
            vocab_size=256,
            hidden_size=256,
            num_attention_heads=4,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            num_hidden_layers=2,
            intermediate_size=512,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
        )

        backend = BackendConfig(
            linear="torch",
            attn="te",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

        # Precompute RoPE frequencies
        rope_theta, _, _ = get_rope_config(config)
        rope_freqs = precompute_freqs_cis(
            qk_rope_head_dim=config.qk_rope_head_dim,
            max_seq_len=config.max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        ).to(device)

        attn_no_cp = MLA(config, backend).to(device).to(torch.bfloat16)
        attn_with_cp = MLA(config, backend).to(device).to(torch.bfloat16)

        def get_freqs_cis(position_ids, qkv_format, cp_size=1):
            return freqs_cis_from_position_ids(position_ids, rope_freqs, qkv_format=qkv_format, for_fused_rope=True, cp_size=cp_size)

    elif model_type == "nemotron_v3":

        class MockNemotronV3AttentionConfig:
            def __init__(self):
                self.num_attention_heads = 8
                self.num_key_value_heads = 4
                self.head_dim = 32
                self.hidden_size = 256
                self.attention_bias = False
                self.attention_dropout = 0.0

        config = MockNemotronV3AttentionConfig()
        return config, None, None, None

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return config, attn_no_cp, attn_with_cp, get_freqs_cis


# ---------------------------------------------------------------------------
# NemotronV3 attention pair creation
# ---------------------------------------------------------------------------

def _create_nemotron_v3_attn_pair(config, backend, device):
    """Create a pair of identical NemotronV3Attention modules with synced weights."""
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


# ---------------------------------------------------------------------------
# Config: bshd_te
# ---------------------------------------------------------------------------

def run_bshd_te(model_type, config, rank, world_size, device,
                attn_no_cp=None, attn_with_cp=None, get_freqs_cis=None):
    """3D BSHD input with TE p2p CP and DualChunkSwap.

    Only supported for nemotron_v3 (qwen3_moe / deepseek_v3 do not use this config).
    """
    if model_type != "nemotron_v3":
        raise ValueError(f"bshd_te config is only supported for nemotron_v3, got {model_type}")

    from nemo_automodel.components.models.common import BackendConfig

    backend = BackendConfig(linear="torch", attn="te")
    attn_baseline, attn_cp = _create_nemotron_v3_attn_pair(config, backend, device)

    batch_size, seq_len = 2, 128
    torch.manual_seed(42)
    x_full = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    dist.broadcast(x_full, src=0)

    # Baseline: CP=1
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
        model_type,
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
# Config: thd_te
# ---------------------------------------------------------------------------

def run_thd_te(model_type, config, rank, world_size, device,
               attn_no_cp=None, attn_with_cp=None, get_freqs_cis=None):
    """THD input with TE p2p CP.

    For qwen3_moe / deepseek_v3: uses make_cp_batch_for_te + apply_cp flow.
    For nemotron_v3: uses set_context_parallel_group + DualChunkSwap flow.
    """
    if model_type == "nemotron_v3":
        return _run_thd_te_nemotron_v3(config, rank, world_size, device)
    else:
        return _run_thd_te_qwen_deepseek(model_type, config, rank, world_size, device,
                                         attn_no_cp, attn_with_cp, get_freqs_cis)


def _run_thd_te_qwen_deepseek(model_type, config, rank, world_size, device,
                               attn_no_cp, attn_with_cp, get_freqs_cis):
    """THD test flow for qwen3_moe / deepseek_v3 (preserves original run_test behavior)."""
    try:
        import transformer_engine.pytorch  # This creates transformer_engine_torch module
        import transformer_engine_torch as tex
    except ImportError:
        if rank == 0:
            print("ERROR: transformer_engine is required but not installed", file=sys.stderr)
        return 1

    # Set to eval mode to avoid dropout
    attn_no_cp.eval()
    attn_with_cp.eval()

    # Copy weights to ensure they're identical
    attn_with_cp.load_state_dict(attn_no_cp.state_dict())

    # Broadcast weights from rank 0
    for param_no_cp, param_with_cp in zip(attn_no_cp.parameters(), attn_with_cp.parameters()):
        dist.broadcast(param_no_cp.data, src=0)
        dist.broadcast(param_with_cp.data, src=0)

    # Create packed sequence batch
    from nemo_automodel.components.distributed.cp_utils import make_cp_batch_for_te

    batch_size = 4
    seq_lens_per_batch = [[32], [40], [36], [44]]

    batch_cpu = create_packed_sequence_batch(batch_size, seq_lens_per_batch, torch.device("cpu"))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_cpu.items()}

    # ===== Baseline: CP=1 (no context parallelism) =====
    torch.manual_seed(42)
    batch_no_cp = make_cp_batch_for_te(
        cp_mesh=None,
        batch=batch,
        qkv_format="thd",
        padding_token_id=0,
    )

    total_tokens_no_cp = batch_no_cp["input_ids"].shape[0]
    x_no_cp = torch.randn(total_tokens_no_cp, config.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)

    freqs_cis_no_cp = get_freqs_cis(batch_no_cp["position_ids"], qkv_format="thd")

    # Compute max_seqlen from cu_seqlens if not present
    if "max_seqlen" not in batch_no_cp:
        cu_seqlens = batch_no_cp["cu_seqlens"]
        max_seqlen_no_cp = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    else:
        max_seqlen_no_cp = batch_no_cp["max_seqlen"]
        if isinstance(max_seqlen_no_cp, torch.Tensor):
            max_seqlen_no_cp = max_seqlen_no_cp.item()

    output_no_cp = attn_no_cp(
        x_no_cp,
        freqs_cis=freqs_cis_no_cp,
        cu_seqlens=batch_no_cp["cu_seqlens"],
        max_seqlen=max_seqlen_no_cp,
        qkv_format=batch_no_cp.get("qkv_format", "thd"),
    )

    loss_no_cp = output_no_cp.sum()
    loss_no_cp.backward()

    # Store baseline results
    output_baseline = output_no_cp.detach().clone()
    grad_baseline = x_no_cp.grad.detach().clone()

    # Get a param grad for comparison
    param_name = "q_proj.weight"
    param_grad_baseline = None
    for name, p in attn_no_cp.named_parameters():
        if name == param_name:
            param_grad_baseline = p.grad.detach().clone()
            break
    if param_grad_baseline is None:
        # Fallback: use first parameter with grad
        for name, p in attn_no_cp.named_parameters():
            if p.grad is not None:
                param_name = name
                param_grad_baseline = p.grad.detach().clone()
                break

    dist.barrier()

    # ===== Test: CP=2 (context parallelism enabled) =====
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.moe.parallelizer import apply_cp

    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))

    # Apply CP to the attention module
    class DummyBlock(torch.nn.Module):
        def __init__(self, attn_layer):
            super().__init__()
            self.self_attn = attn_layer
            self.mlp = None

    class DummyModel(torch.nn.Module):
        def __init__(self, attn_layer):
            super().__init__()
            self.model = None
            self.layers = torch.nn.ModuleList([DummyBlock(attn_layer)])

    dummy_model = DummyModel(attn_with_cp)
    apply_cp(dummy_model, cp_mesh["cp"], cp_comm_type="p2p")

    # Verify CP was applied correctly
    assert hasattr(attn_with_cp.attn_module, "cp_group"), "CP group not set on attention module"

    # Process batch with CP
    torch.manual_seed(42)
    batch_with_cp = make_cp_batch_for_te(
        cp_mesh=cp_mesh["cp"],
        batch=batch,
        qkv_format="thd",
        padding_token_id=0,
    )

    total_tokens_with_cp = batch_with_cp["input_ids"].shape[0]

    # Use the exact same input as no_cp case
    x_full = x_no_cp.detach().clone()

    # Shard the full input according to CP partitioning using TE's actual indices
    cu_seqlens_padded = batch_with_cp["cu_seqlens"]
    if isinstance(cu_seqlens_padded, torch.Tensor) and cu_seqlens_padded.ndim == 1:
        # Filter padding sentinel values (-1000)
        cu_seqlens_padded_filtered = cu_seqlens_padded[cu_seqlens_padded != -1000]

        # Get the actual indices that TE uses for this rank
        indices = tex.thd_get_partitioned_indices(
            cu_seqlens_padded_filtered,
            total_tokens_no_cp,
            world_size,
            rank,
        )

        x_with_cp = x_full.index_select(0, indices).clone().detach().requires_grad_(True)
    else:
        # Fallback to simple slicing
        start_idx = rank * total_tokens_with_cp
        end_idx = start_idx + total_tokens_with_cp
        x_with_cp = x_full[start_idx:end_idx].clone().detach().requires_grad_(True)

    cp_size = batch_with_cp.get("cp_size", 1)
    cp_rank = batch_with_cp.get("cp_rank", 0)
    freqs_cis_with_cp = get_freqs_cis(batch_with_cp["position_ids"], qkv_format="thd", cp_size=cp_size)

    # Compute max_seqlen from cu_seqlens if not present
    if "max_seqlen" not in batch_with_cp:
        cu_seqlens = batch_with_cp["cu_seqlens"]
        max_seqlen_with_cp = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    else:
        max_seqlen_with_cp = batch_with_cp["max_seqlen"]
        if isinstance(max_seqlen_with_cp, torch.Tensor):
            max_seqlen_with_cp = max_seqlen_with_cp.item()

    output_with_cp = attn_with_cp(
        x_with_cp,
        freqs_cis=freqs_cis_with_cp,
        cu_seqlens=batch_with_cp["cu_seqlens"],
        max_seqlen=max_seqlen_with_cp,
        qkv_format=batch_with_cp.get("qkv_format", "thd"),
        cp_size=cp_size,
        cp_rank=cp_rank,
    )

    loss_with_cp = output_with_cp.sum()
    loss_with_cp.backward()

    # Gather results from all ranks along with indices
    output_with_cp_gathered = [
        torch.zeros(total_tokens_with_cp, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    grad_with_cp_gathered = [
        torch.zeros(total_tokens_with_cp, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(world_size)
    ]
    indices_gathered = [torch.zeros(total_tokens_with_cp, device=device, dtype=torch.int32) for _ in range(world_size)]

    dist.all_gather(output_with_cp_gathered, output_with_cp)
    dist.all_gather(grad_with_cp_gathered, x_with_cp.grad)
    dist.all_gather(indices_gathered, indices.to(torch.int32))

    # Concatenate results
    output_with_cp_concat = torch.cat(output_with_cp_gathered, dim=0)
    grad_with_cp_concat = torch.cat(grad_with_cp_gathered, dim=0)
    indices_concat = torch.cat(indices_gathered, dim=0)

    # Reorder gathered outputs to match original token order
    output_with_cp_full = torch.zeros(total_tokens_no_cp, config.hidden_size, device=device, dtype=torch.bfloat16)
    grad_with_cp_full = torch.zeros(total_tokens_no_cp, config.hidden_size, device=device, dtype=torch.bfloat16)

    output_with_cp_full[indices_concat] = output_with_cp_concat
    grad_with_cp_full[indices_concat] = grad_with_cp_concat

    # Get param grad for CP run (all-reduce across CP ranks since each rank
    # only computes gradients for its local sequence shard)
    param_grad_cp = None
    for name, p in attn_with_cp.named_parameters():
        if name == param_name:
            param_grad_cp = p.grad.detach().clone()
            break
    if param_grad_cp is None:
        for name, p in attn_with_cp.named_parameters():
            if p.grad is not None:
                param_name = name
                param_grad_cp = p.grad.detach().clone()
                break
    if param_grad_cp is not None:
        dist.all_reduce(param_grad_cp, op=dist.ReduceOp.SUM)

    return _compare_results(
        "thd_te",
        model_type,
        rank,
        output_with_cp_full,
        output_baseline,
        grad_with_cp_full,
        grad_baseline,
        param_grad_cp,
        param_grad_baseline,
        param_name,
        output_atol=0.01,
        output_rtol=1e-2,
        grad_atol=0.05,
        grad_rtol=2e-2,
        param_atol=0.05,
        param_rtol=2e-2,
    )


def _run_thd_te_nemotron_v3(config, rank, world_size, device):
    """THD test flow for nemotron_v3 (DualChunkSwap + set_context_parallel_group)."""
    from nemo_automodel.components.models.common import BackendConfig

    backend = BackendConfig(linear="torch", attn="te")
    attn_baseline, attn_cp = _create_nemotron_v3_attn_pair(config, backend, device)

    batch_size, seq_len = 1, 128
    torch.manual_seed(42)
    x_full = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    dist.broadcast(x_full, src=0)

    # Baseline: run in 3D BSHD (batch=1) so the TE attention path is identical
    # to CP aside from the CP gather/scatter.
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
        "nemotron_v3",
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
# Config: bshd_sdpa
# ---------------------------------------------------------------------------

def run_bshd_sdpa(model_type, config, rank, world_size, device,
                  attn_no_cp=None, attn_with_cp=None, get_freqs_cis=None):
    """3D BSHD input with DTensor context_parallel() and SDPA backend.

    Only supported for nemotron_v3 (qwen3_moe / deepseek_v3 do not use this config).
    """
    if model_type != "nemotron_v3":
        raise ValueError(f"bshd_sdpa config is only supported for nemotron_v3, got {model_type}")

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import context_parallel_unshard, set_rotate_method
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from nemo_automodel.components.models.common import BackendConfig

    backend = BackendConfig(linear="torch", attn="sdpa")
    attn_baseline, attn_cp = _create_nemotron_v3_attn_pair(config, backend, device)

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
        "nemotron_v3",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONFIG_RUNNERS = {
    "bshd_te": run_bshd_te,
    "thd_te": run_thd_te,
    "bshd_sdpa": run_bshd_sdpa,
}


def main():
    parser = argparse.ArgumentParser(description="Test attention layer with context parallelism")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["qwen3_moe", "deepseek_v3", "nemotron_v3"],
        help="Model type to test",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        choices=["bshd_te", "thd_te", "bshd_sdpa"],
        help="Which configs to run. Defaults: nemotron_v3 -> all three; others -> thd_te only.",
    )
    args = parser.parse_args()

    # Default configs per model type
    if args.configs is None:
        if args.model_type == "nemotron_v3":
            args.configs = ["bshd_te", "thd_te", "bshd_sdpa"]
        else:
            args.configs = ["thd_te"]

    # Initialize distributed
    init_distributed()

    world_size = get_world_size()
    rank = get_rank()

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: This test requires exactly 2 GPUs, got {world_size}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{rank}")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Get model configuration and attention layers
    config, attn_no_cp, attn_with_cp, get_freqs_cis = get_model_config_and_attention(
        args.model_type, device
    )

    # Run selected configs and collect results
    results = {}
    for config_name in args.configs:
        dist.barrier()
        runner = CONFIG_RUNNERS[config_name]
        try:
            results[config_name] = runner(
                args.model_type, config, rank, world_size, device,
                attn_no_cp=attn_no_cp,
                attn_with_cp=attn_with_cp,
                get_freqs_cis=get_freqs_cis,
            )
        except Exception as e:
            if rank == 0:
                import traceback

                print(f"  {config_name}: ERROR - {e}")
                traceback.print_exc()
            results[config_name] = 1

    # Print summary table
    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"Summary - {args.model_type} CP Tests")
        print(f"{'=' * 70}")
        for name, result in results.items():
            status = "PASSED" if result == 0 else "FAILED"
            print(f"  {name}: {status}")
        print(f"{'=' * 70}\n")

    # Cleanup
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if any(r != 0 for r in results.values()) else 0)


if __name__ == "__main__":
    main()
