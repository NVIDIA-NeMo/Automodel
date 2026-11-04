#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Two-rank packed CP parity for Qwen3.8-Flash-Next QSA on the CUDA flex path.

One packed THD row (two documents plus CP padding) is sharded contiguously
over CP=2 with the production sharder. Each rank runs its local shard through
``Qwen3_8_FlashNextQSAAttention`` with the flex backend, which all-gathers K/V
to the global length and builds the FlexAttention mask from the recorded
routes. The result is compared against the same layer run in one process on
the full packed row: routes are bitwise global, local outputs and hidden-state
gradients match the corresponding slice, parameter gradients match after the
cross-rank sum, and the CP padding tail is exactly zero.

This is the configuration that regressed when the route selection sized the
mask by the local shard instead of the gathered K/V (``mask=(8, 8)`` versus
``tensors=(8, 16)``); the CPU unit tests cannot reach it because the flex mask
is only built on CUDA.

Run::

    torchrun --standalone --nproc-per-node=2 \
        tests/functional_tests/context_parallel/run_qwen3_8_flash_next_packed_cp.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from nemo_automodel.components.models.qwen3_8_flash_next.cp import (
    Qwen3_8_FlashNextCPContext,
    shard_batch_for_qwen3_8_flash_next_cp,
)
from nemo_automodel.components.models.qwen3_8_flash_next.layers import Qwen3_8_FlashNextQSAAttention

WORLD_SIZE = 2
BOUNDARIES = (0, 3, 10)  # two documents; the row is CP-padded to 16 tokens
PADDED_LENGTH = 16
DTYPE = torch.bfloat16
# bf16 flex attention versus the same bf16 layer on the full row.
OUTPUT_TOL = dict(rtol=2e-2, atol=2e-2)
# Parameter gradients sum two bf16 partials across ranks; reduce in fp32 and allow more.
PARAM_TOL = dict(rtol=5e-2, atol=5e-2)


def _config() -> Qwen3_8_FlashNextTextConfig:
    """Kernel-shaped heads (24 query, 2 KV, dim 256) on a small hidden width."""
    return Qwen3_8_FlashNextTextConfig(
        vocab_size=32,
        hidden_size=512,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        layer_types=["full_attention"],
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=2,
        ple_layer_ids=[],
        indexer_budget=8,
        indexer_compress_ratio=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=256,
        max_position_embeddings=4096,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0},
        partial_rotary_factor=1.0,
        dtype="float32",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
    )


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="flex",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


def _document_relative_freqs(cu_seqlens: tuple[int, ...], rotary_width: int) -> torch.Tensor:
    """Build packed rotary values whose positions restart at every document."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_width, 2).float() / rotary_width))
    segments = []
    for start, end in zip(cu_seqlens, cu_seqlens[1:]):
        positions = torch.arange(end - start, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        segments.append(torch.cat((angles.cos(), angles.sin()), dim=-1))
    return torch.cat(segments, dim=0).unsqueeze(0)


def main() -> None:
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"}.issubset(os.environ):
        print("ERROR: launch this script with torchrun.", file=sys.stderr)
        sys.exit(1)
    if torch.cuda.device_count() < WORLD_SIZE:
        print(f"[skip] Qwen3.8-Flash-Next packed CP parity requires {WORLD_SIZE} CUDA devices", file=sys.stderr)
        return
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size != WORLD_SIZE:
        if rank == 0:
            print(f"ERROR: expected world size {WORLD_SIZE}, got {world_size}", file=sys.stderr)
        dist.destroy_process_group()
        sys.exit(1)

    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))["cp"]
    total_tokens = BOUNDARIES[-1]
    input_ids = torch.arange(1, total_tokens + 1, dtype=torch.long).view(1, -1)
    document_positions = torch.cat([torch.arange(end - start) for start, end in zip(BOUNDARIES, BOUNDARIES[1:])]).view(
        1, -1
    )
    _, local_batch, layout = shard_batch_for_qwen3_8_flash_next_cp(
        cp_mesh,
        None,
        {
            "input_ids": input_ids.clone(),
            "labels": input_ids.clone(),
            "position_ids": document_positions.clone(),
            "cu_seqlens": torch.tensor(BOUNDARIES, dtype=torch.int32),
            "qkv_format": "thd",
        },
        padding_token_id=0,
        pad_multiple=4,
    )
    assert layout.padded_seq_len == PADDED_LENGTH, layout.padded_seq_len
    batch_context = local_batch["_qwen3_8_flash_next_cp_context"]
    assert isinstance(batch_context, Qwen3_8_FlashNextCPContext)
    assert batch_context.global_cu_seqlens is not None
    local_length = PADDED_LENGTH // world_size
    local_start = rank * local_length

    torch.manual_seed(321)
    config = _config()
    reference = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=_backend())
    reference.init_weights(torch.device("cpu"))
    cp_attention = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=_backend())
    cp_attention.load_state_dict(reference.state_dict())
    reference = reference.to(device=device, dtype=DTYPE)
    cp_attention = cp_attention.to(device=device, dtype=DTYPE)

    full_freqs = _document_relative_freqs((*BOUNDARIES, PADDED_LENGTH), rotary_width=config.head_dim).to(
        device=device, dtype=DTYPE
    )
    torch.manual_seed(99)
    full_hidden_values = torch.randn(1, PADDED_LENGTH, config.hidden_size).to(device=device, dtype=DTYPE)
    grad_output = torch.randn(1, PADDED_LENGTH, config.hidden_size).to(device=device, dtype=DTYPE)

    # Reference: the whole packed row in one process (packed CP1).
    reference_hidden = full_hidden_values[:, :total_tokens].clone().requires_grad_(True)
    reference_routes: list[torch.Tensor] = []
    handle = reference.indexer.register_forward_hook(lambda _module, _args, output: reference_routes.append(output))
    reference_output = reference(
        reference_hidden,
        freqs_cis=full_freqs[:, :total_tokens],
        cu_seqlens=torch.tensor(BOUNDARIES, dtype=torch.int32, device=device),
    )
    handle.remove()
    reference_output.backward(grad_output[:, :total_tokens])

    # Under test: this rank's shard with the sharder-provided CP context (packed CP2).
    context = Qwen3_8_FlashNextCPContext(
        group=dist.group.WORLD,
        rank=rank,
        size=world_size,
        global_input_ids=torch.nn.functional.pad(input_ids, (0, PADDED_LENGTH - total_tokens)).to(device),
        global_padding_mask=(torch.arange(PADDED_LENGTH).view(1, -1) >= total_tokens).to(device),
        local_sequence_start=local_start,
        local_sequence_length=local_length,
        global_cu_seqlens=torch.tensor(BOUNDARIES, dtype=torch.long, device=device),
    )
    local_hidden = full_hidden_values[:, local_start : local_start + local_length].clone().requires_grad_(True)
    cp_routes: list[torch.Tensor] = []
    handle = cp_attention.indexer.register_forward_hook(lambda _module, _args, output: cp_routes.append(output))
    cp_output = cp_attention(
        local_hidden,
        freqs_cis=full_freqs[:, local_start : local_start + local_length],
        cp_context=context,
    )
    handle.remove()
    cp_output.backward(grad_output[:, local_start : local_start + local_length])

    # Routes: bitwise-global IDs on real tokens, all -1 on the CP pad tail.
    for local_idx in range(local_length):
        global_idx = local_start + local_idx
        if global_idx < total_tokens:
            torch.testing.assert_close(cp_routes[0][0, local_idx], reference_routes[0][0, global_idx], rtol=0, atol=0)
        else:
            assert bool((cp_routes[0][0, local_idx] == -1).all())

    real_length = max(0, min(total_tokens - local_start, local_length))
    if real_length:
        torch.testing.assert_close(
            cp_output[:, :real_length], reference_output[:, local_start : local_start + real_length], **OUTPUT_TOL
        )
        torch.testing.assert_close(
            local_hidden.grad[:, :real_length],
            reference_hidden.grad[:, local_start : local_start + real_length],
            **OUTPUT_TOL,
        )
    if real_length < local_length:
        assert bool((cp_output[:, real_length:] == 0).all())

    # Parameter gradients: rank-local contributions must sum to the packed CP1 gradient.
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in cp_attention.named_parameters():
        reference_gradient = reference_parameters[name].grad
        if parameter.grad is None:
            assert reference_gradient is None, name
            continue
        summed_gradient = parameter.grad.float()
        dist.all_reduce(summed_gradient, group=dist.group.WORLD)
        assert reference_gradient is not None, name
        torch.testing.assert_close(summed_gradient, reference_gradient.float(), **PARAM_TOL)

    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("[ok] Qwen3.8-Flash-Next packed CP2 flex forward/backward matches packed CP1")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
