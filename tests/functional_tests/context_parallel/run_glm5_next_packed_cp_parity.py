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

"""Eight-rank packed CP parity for GLM-5.3-Flash KDA plus KPool-DSA.

The reference and distributed models receive the same packed documents.  CP=8
uses the production contiguous sharder: input ids stay global until embedding
and image-splice time, while auxiliary token fields are sharded immediately.
The test compares reconstructed logits, input-embedding gradients, and every
used parameter gradient against CP=1.

Usage:
    torchrun --standalone --nproc_per_node=8 \
        tests/functional_tests/context_parallel/run_glm5_next_packed_cp_parity.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _config():
    """Return a small BF16 hybrid model that still exercises real FLA kernels."""
    from nemo_automodel.components.models.glm5_next.config import (
        Glm5NextConfig,
        Glm5NextTextConfig,
        Glm5NextVisionConfig,
    )

    text = Glm5NextTextConfig(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        qk_nope_head_dim=16,
        v_head_dim=16,
        index_topk=16,
        index_head_dim=16,
        index_n_heads=4,
        index_kpool=4,
        linear_head_dim=16,
        linear_num_heads=4,
        linear_conv_kernel_dim=4,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        mlp_layer_types=["dense"] * 4,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        indexer_types=["full"] * 4,
        pad_token_id=0,
        torch_dtype="bfloat16",
    )
    vision = Glm5NextVisionConfig(
        depth=1,
        hidden_size=16,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        intermediate_size=32,
        projection_intermediate_size=64,
        torch_dtype="bfloat16",
    )
    return Glm5NextConfig(text_config=text, vision_config=vision, image_token_id=95, pad_token_id=0)


def _model(device: torch.device):
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration

    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )
    model = Glm5NextForConditionalGeneration(_config(), backend=backend).to(device)
    model.initialize_weights(device, dtype=torch.bfloat16)
    return model.train()


def _sync_model(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)


def main() -> None:
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"}.issubset(os.environ):
        print("ERROR: launch this script with torchrun.", file=sys.stderr)
        sys.exit(1)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size != 8:
        if rank == 0:
            print(f"ERROR: this parity contract requires CP=8, got {world_size} ranks.", file=sys.stderr)
        dist.destroy_process_group()
        sys.exit(1)

    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.models.glm5_next.cp import shard_batch_for_glm5_next_cp
    from nemo_automodel.components.moe.parallelizer import apply_cp

    torch.manual_seed(1234)
    reference = _model(device)
    distributed = _model(device)
    distributed.load_state_dict(reference.state_dict())
    _sync_model(reference)
    distributed.load_state_dict(reference.state_dict())

    sequence = 128
    torch.manual_seed(4321)
    input_ids = torch.randint(1, 95, (1, sequence), device=device)
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    doc_ids = torch.tensor(
        [[1] * 19 + [2] * 37 + [3] * 72],
        dtype=torch.int32,
        device=device,
    )
    dist.broadcast(input_ids, src=0)
    dist.broadcast(targets, src=0)

    ref_logits = reference(input_ids=input_ids, _packed_seq_ids=doc_ids).logits
    ref_loss = F.cross_entropy(ref_logits.float().flatten(0, 1), targets.flatten())
    ref_loss.backward()

    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))["cp"]
    apply_cp(distributed, cp_mesh)

    _, local_batch, _ = shard_batch_for_glm5_next_cp(
        cp_mesh,
        None,
        {
            "input_ids": input_ids.clone(),
            "labels": input_ids.clone(),
            "_packed_seq_ids": doc_ids.clone(),
        },
        shard_primary=False,
    )
    context = local_batch["glm5_next_packed_context"]
    local_logits = distributed(
        input_ids=local_batch["input_ids"],
        padding_mask=local_batch["padding_mask"],
        glm5_next_packed_context=context,
    ).logits
    local_start = context.seq_start
    local_end = local_start + context.local_seq_len
    local_loss = F.cross_entropy(
        local_logits.float().flatten(0, 1),
        targets[:, local_start:local_end].flatten(),
    )
    (local_loss / world_size).backward()
    cp_loss = local_loss.detach().clone() / world_size
    dist.all_reduce(cp_loss, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(cp_loss, ref_loss.detach(), rtol=2e-3, atol=2e-3)

    gathered_logits = [torch.empty_like(local_logits) for _ in range(world_size)]
    dist.all_gather(gathered_logits, local_logits.detach())
    cp_logits = torch.cat(gathered_logits, dim=1)
    logit_diff = (cp_logits.float() - ref_logits.detach().float()).abs()
    if rank == 0:
        print(
            f"GLM-5.3 packed CP logits mean/max={logit_diff.mean().item():.3e}/{logit_diff.max().item():.3e}",
            flush=True,
        )
    # FLA transports recurrent state rank-to-rank in CP=8, changing BF16
    # accumulation order relative to the monolithic CP=1 chunked kernel.
    torch.testing.assert_close(cp_logits, ref_logits.detach(), rtol=1.5e-1, atol=1.5e-1)

    max_grad_abs = 0.0
    gradient_diff_sq = 0.0
    gradient_ref_sq = 0.0
    gradient_dot = 0.0
    gradient_cp_sq = 0.0
    per_parameter_relative_l2 = []
    compared = 0
    cp_parameters = dict(distributed.named_parameters())
    for name, ref_parameter in reference.named_parameters():
        cp_parameter = cp_parameters[name]
        if ref_parameter.grad is None:
            if cp_parameter.grad is not None:
                raise AssertionError(f"CP produced an unexpected gradient for {name}")
            continue
        if cp_parameter.grad is None:
            raise AssertionError(f"CP did not produce a gradient for {name}")
        cp_gradient = cp_parameter.grad.detach().float().clone()
        dist.all_reduce(cp_gradient, op=dist.ReduceOp.SUM)
        ref_gradient = ref_parameter.grad.detach().float()
        if not torch.isfinite(cp_gradient).all():
            raise AssertionError(f"CP produced a non-finite gradient for {name}")
        difference = cp_gradient - ref_gradient
        ref_sq = ref_gradient.double().square().sum().item()
        diff_sq = difference.double().square().sum().item()
        cp_sq = cp_gradient.double().square().sum().item()
        max_grad_abs = max(max_grad_abs, difference.abs().max().item())
        gradient_diff_sq += diff_sq
        gradient_ref_sq += ref_sq
        gradient_cp_sq += cp_sq
        gradient_dot += (cp_gradient.double() * ref_gradient.double()).sum().item()
        relative_l2 = diff_sq**0.5 / max(ref_sq**0.5, 1e-12)
        per_parameter_relative_l2.append((relative_l2, name))
        compared += 1

    gradient_relative_l2 = (gradient_diff_sq / gradient_ref_sq) ** 0.5
    gradient_cosine = gradient_dot / max((gradient_cp_sq * gradient_ref_sq) ** 0.5, 1e-12)
    if rank == 0:
        worst_parameters = ", ".join(
            f"{name}={relative_l2:.3e}" for relative_l2, name in sorted(per_parameter_relative_l2, reverse=True)[:5]
        )
        print(
            "GLM-5.3 packed CP parity PASS: "
            f"CP1 vs CP8, loss={cp_loss.item():.6f}/{ref_loss.item():.6f}, "
            f"logits mean/max={logit_diff.mean().item():.3e}/{logit_diff.max().item():.3e}, "
            f"parameter gradients compared={compared}, grad rel-L2/cosine/max-abs="
            f"{gradient_relative_l2:.3e}/{gradient_cosine:.6f}/{max_grad_abs:.3e}; "
            f"worst tensor rel-L2: {worst_parameters}"
        )
    if gradient_relative_l2 >= 5e-2 or gradient_cosine <= 0.998:
        raise AssertionError(
            f"CP gradient parity failed: relative L2={gradient_relative_l2:.3e}, cosine={gradient_cosine:.6f}"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
