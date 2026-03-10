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

Validates that a hybrid model with interleaved attention (TE p2p CP) and
mamba (hidden-parallel CP) layers produces matching outputs/gradients
between CP=1 and CP=2 with DualChunkSwap sequence distribution.

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_hybrid_nemotron_v3_cp.py
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

        # MoE config fields — required by NemotronV3Model.__init__ even when
        # no MoE layers are present in layers_block_type.
        self.n_routed_experts = 1
        self.num_experts_per_tok = 1
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.moe_intermediate_size = self.intermediate_size
        self.norm_topk_prob = False
        self.moe_shared_expert_intermediate_size = self.intermediate_size


def run_test():
    """Run the end-to-end CP validation test for hybrid NemotronV3."""
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

    from torch.distributed.device_mesh import init_device_mesh
    from transformer_engine.pytorch.attention import DotProductAttention

    from nemo_automodel.components.distributed.mamba_cp import MambaContextParallel
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

    config = MockHybridConfig()
    # Use torch linear and rms_norm to avoid TE internal buffers that would
    # complicate state_dict transfers, but keep TE attention for p2p CP.
    backend = BackendConfig(
        linear="torch",
        attn="te",
        rms_norm="torch",
        enable_hf_state_dict_adapter=False,
    )

    # ===== Baseline: CP=1 (no context parallelism) =====
    model_baseline = NemotronV3Model(config, backend=backend).to(device=device, dtype=torch.bfloat16)
    model_baseline.train()

    # Sync weights across ranks so both start from identical parameters
    for p in model_baseline.parameters():
        dist.broadcast(p.data, src=0)

    batch_size = 2
    seq_len = 128  # must be divisible by 2 * cp_size = 4

    torch.manual_seed(42)
    input_ids_full = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    dist.broadcast(input_ids_full, src=0)

    output_baseline = model_baseline(input_ids=input_ids_full)
    loss_baseline = output_baseline.sum()
    loss_baseline.backward()

    # Save baseline results before any further operations
    output_baseline_detached = output_baseline.detach().clone()
    embed_grad_baseline = model_baseline.embed_tokens.weight.grad.detach().clone()

    dist.barrier()

    # ===== Test: CP=2 (context parallelism with p2p for attn, hidden-parallel for mamba) =====
    model_cp = NemotronV3Model(config, backend=backend).to(device=device, dtype=torch.bfloat16)
    model_cp.train()

    # Copy weights from baseline model; use strict=False to tolerate any TE
    # internal buffers that may not appear in the baseline's state_dict
    model_cp.load_state_dict(model_baseline.state_dict(), strict=False)

    # Zero out any gradients that may have accumulated during load
    model_cp.zero_grad()

    # Build CP process group
    cp_size = world_size  # 2
    cp_mesh = init_device_mesh("cuda", (cp_size,), mesh_dim_names=("cp",))
    cp_group = cp_mesh["cp"].get_group()

    # Wire CP on each hybrid layer
    for layer in model_cp.layers.values():
        if layer.block_type == "mamba":
            mixer = layer.mixer
            mixer.cp = MambaContextParallel(
                cp_group=cp_group,
                num_heads=mixer.num_heads,
                head_dim=mixer.head_dim,
                n_groups=mixer.n_groups,
                d_state=mixer.ssm_state_size,
                conv1d=mixer.conv1d,
                dt_bias=mixer.dt_bias,
                A_log=mixer.A_log,
                D=mixer.D,
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

    # DualChunkSwap: each rank gets two non-contiguous chunks
    input_ids_local = _dual_chunk_swap_select(
        input_ids_full, cp_size=cp_size, cp_rank=rank, seq_dim=1
    )

    output_cp_local = model_cp(input_ids=input_ids_local)
    loss_cp = output_cp_local.sum()
    loss_cp.backward()

    # Gather local outputs from all CP ranks
    local_seq = output_cp_local.shape[1]
    output_gathered = [
        torch.zeros(batch_size, local_seq, config.hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(cp_size)
    ]
    dist.all_gather(output_gathered, output_cp_local.detach().contiguous(), group=cp_group)

    # Reconstruct full-sequence output from DualChunkSwap ordering
    output_cp_full = dual_chunk_swap_unsplit(output_gathered, cp_size=cp_size, seq_dim=1)

    # Embedding weight gradient: each rank only sees a subset of the sequence,
    # so the embedding grad is only partial — all-reduce to get the full gradient.
    embed_grad_cp = model_cp.embed_tokens.weight.grad.detach().clone()
    dist.all_reduce(embed_grad_cp, op=dist.ReduceOp.SUM, group=cp_group)

    # ===== Comparison =====
    output_diff = (output_cp_full - output_baseline_detached).abs()
    grad_diff = (embed_grad_cp - embed_grad_baseline).abs()

    if rank == 0:
        print(f"\n{'='*70}")
        print("End-to-End Hybrid CP Test - NemotronV3 (Attention + Mamba)")
        print(f"{'='*70}")
        print(f"Config: {config.num_hidden_layers} layers {config.layers_block_type}")
        print(
            f"Sequence: batch={batch_size}, seq_len={seq_len} -> "
            f"{local_seq} tokens/rank with CP={cp_size}"
        )
        print(f"\nForward output:")
        print(f"  Shape: CP={output_cp_full.shape}, Baseline={output_baseline_detached.shape}")
        print(
            f"  Diff - mean: {output_diff.mean().item():.6f}, "
            f"max: {output_diff.max().item():.6f}, "
            f"std: {output_diff.std().item():.6f}"
        )
        print(
            f"  Relative diff - mean: "
            f"{(output_diff / (output_baseline_detached.abs() + 1e-8)).mean().item():.6f}"
        )
        print(f"\nEmbedding weight gradient:")
        print(
            f"  Baseline - mean: {embed_grad_baseline.abs().mean().item():.6f}, "
            f"max: {embed_grad_baseline.abs().max().item():.6f}"
        )
        print(
            f"  CP       - mean: {embed_grad_cp.abs().mean().item():.6f}, "
            f"max: {embed_grad_cp.abs().max().item():.6f}"
        )
        print(
            f"  Diff     - mean: {grad_diff.mean().item():.6f}, "
            f"max: {grad_diff.max().item():.6f}"
        )

    try:
        torch.testing.assert_close(
            output_cp_full,
            output_baseline_detached,
            rtol=1e-2,
            atol=5e-2,
            msg=f"[Rank {rank}] Forward outputs differ between CP=1 and CP=2",
        )

        torch.testing.assert_close(
            embed_grad_cp,
            embed_grad_baseline,
            rtol=5e-2,
            atol=1e-1,
            msg=f"[Rank {rank}] embed_tokens.weight.grad differs between CP=1 and CP=2",
        )

        if rank == 0:
            print(f"Test PASSED: Forward outputs and embedding gradients match between CP=1 and CP=2")
            print(f"{'='*70}\n")
        return 0

    except AssertionError as e:
        if rank == 0:
            print(f"Test FAILED: {e}")
            print(f"Note: Some numerical differences are expected with bfloat16 and multi-layer accumulation")
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
