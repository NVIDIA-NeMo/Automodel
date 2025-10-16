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

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_attention_cp.py \
        --model_type qwen3_moe

    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_attention_cp.py \
        --model_type deepseek_v3
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist


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


def get_model_config_and_attention(model_type, device):
    """Get model configuration and attention layer based on model type."""
    if model_type == "qwen3_moe":
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

        from nemo_automodel.components.models.qwen3_moe.layers import Qwen3MoeAttention
        from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding
        from nemo_automodel.components.moe.utils import BackendConfig

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
            enable_deepep=False,
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

        rope = RotaryEmbedding(
            head_dim=config.head_dim,
            base=config.rope_theta,
            dtype=torch.float32,
        )

        attn_no_cp = Qwen3MoeAttention(config, backend).to(device).to(torch.bfloat16)
        attn_with_cp = Qwen3MoeAttention(config, backend).to(device).to(torch.bfloat16)

        from nemo_automodel.components.models.gpt_oss.rope_utils import position_ids_to_freqs_cis

        def get_freqs_cis(position_ids, qkv_format):
            return position_ids_to_freqs_cis(rope, position_ids, qkv_format)

    elif model_type == "deepseek_v3":
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

        from nemo_automodel.components.models.deepseek_v3.layers import MLA
        from nemo_automodel.components.models.deepseek_v3.rope_utils import (
            precompute_freqs_cis,
            freqs_cis_from_position_ids,
        )
        from nemo_automodel.components.moe.utils import BackendConfig

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
            enable_deepep=False,
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

        # Precompute RoPE frequencies
        rope_freqs = precompute_freqs_cis(
            qk_rope_head_dim=config.qk_rope_head_dim,
            max_seq_len=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        ).to(device)

        attn_no_cp = MLA(config, backend).to(device).to(torch.bfloat16)
        attn_with_cp = MLA(config, backend).to(device).to(torch.bfloat16)

        def get_freqs_cis(position_ids, qkv_format):
            return freqs_cis_from_position_ids(position_ids, rope_freqs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return config, attn_no_cp, attn_with_cp, get_freqs_cis


def run_test(model_type):
    """Run the CP validation test for the specified model type."""
    world_size = get_world_size()
    rank = get_rank()

    try:
        import transformer_engine.pytorch  # This creates transformer_engine_torch module
        import transformer_engine_torch as tex
    except ImportError:
        if rank == 0:
            print("ERROR: transformer_engine is required but not installed", file=sys.stderr)
        return 1

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: This test requires exactly 2 GPUs, got {world_size}", file=sys.stderr)
        return 1

    device = torch.device(f"cuda:{rank}")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Get model configuration and attention layers
    config, attn_no_cp, attn_with_cp, get_freqs_cis = get_model_config_and_attention(model_type, device)

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

    freqs_cis_with_cp = get_freqs_cis(batch_with_cp["position_ids"], qkv_format="thd")

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

    # Compare outputs and gradients
    if rank == 0:
        output_diff = (output_with_cp_full - output_baseline).abs()
        grad_diff = (grad_with_cp_full - grad_baseline).abs()

        print(f"\n{'='*70}")
        print(f"Context Parallelism Validation Test - {model_type.upper()}")
        print(f"{'='*70}")
        print(f"Output shape: CP={output_with_cp_full.shape}, Baseline={output_baseline.shape}")
        print(f"Output diff - mean: {output_diff.mean():.6f}, max: {output_diff.max():.6f}, std: {output_diff.std():.6f}")
        print(f"Output relative diff - mean: {(output_diff / (output_baseline.abs() + 1e-8)).mean():.6f}")
        print(f"\nGradient statistics:")
        print(f"  Baseline - min: {grad_baseline.abs().min():.6f}, max: {grad_baseline.abs().max():.6f}, mean: {grad_baseline.abs().mean():.6f}")
        print(f"  CP       - min: {grad_with_cp_full.abs().min():.6f}, max: {grad_with_cp_full.abs().max():.6f}, mean: {grad_with_cp_full.abs().mean():.6f}")
        print(f"Grad diff - mean: {grad_diff.mean():.6f}, max: {grad_diff.max():.6f}, std: {grad_diff.std():.6f}")

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
            msg=f"[Rank {rank}] Gradients differ between CP=1 and CP=2",
        )

        if rank == 0:
            print(f"✓ Test PASSED: Forward outputs and gradients match between CP=1 and CP=2")
            print(f"{'='*70}\n")
        return 0

    except AssertionError as e:
        if rank == 0:
            print(f"✗ Test FAILED: {e}")
            print(f"Note: Some numerical differences are expected with bfloat16 precision")
            print(f"{'='*70}\n")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Test attention layer with context parallelism")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["qwen3_moe", "deepseek_v3"],
        help="Model type to test",
    )
    args = parser.parse_args()

    # Initialize distributed
    init_distributed()

    # Run test
    exit_code = run_test(args.model_type)

    # Cleanup
    if is_distributed():
        dist.barrier()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
