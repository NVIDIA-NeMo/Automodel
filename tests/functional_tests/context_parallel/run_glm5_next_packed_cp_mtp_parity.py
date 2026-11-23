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

"""Packed CP parity for the GLM-5.3-Flash MTP head (CP=world size, at least 2 ranks).

Same packed documents through CP=1 and the production contiguous CP sharder. The
last document crosses the CP rank boundary, so the MTP head's shifted embeddings
and targets must cross it too. The loss is the backbone CE plus 0.1 x the MTP CE
on the globally prepared targets. The test compares both losses, the gathered
logits and MTP states, every parameter gradient (MTP head, shared embedding and
lm_head included), and the loss after one AdamW step taken from each side's
gradients.

The model runs in FP32 with TF32 off, so the comparison isolates the sharding
logic: in BF16 the rank-to-rank KDA state transport changes accumulation order,
and near-tie expert picks in the MTP MoE flip between the two runs.

Usage:
    torchrun --standalone --nproc_per_node=2 \
        tests/functional_tests/context_parallel/run_glm5_next_packed_cp_mtp_parity.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

_MTP_WEIGHT = 0.1


def _config():
    """Return a small FP32 hybrid model with one MTP layer that still exercises real FLA kernels."""
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
        num_nextn_predict_layers=1,
        torch_dtype="float32",
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
        torch_dtype="float32",
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
    model.initialize_weights(device, dtype=torch.float32)
    return model.train()


def _sync_model(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)


def _loss_terms(logits, mtp_hidden, lm_head, targets, mtp_targets, num_targets, num_mtp_targets):
    """Return summed main and MTP CE divided by the global target counts."""
    main = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="sum") / num_targets
    mtp_logits = lm_head(mtp_hidden).float()
    mtp = (
        F.cross_entropy(mtp_logits.flatten(0, 1), mtp_targets.flatten(), ignore_index=-100, reduction="sum")
        / num_mtp_targets
    )
    return main, mtp


def _gradient_parity(reference, distributed):
    """All-reduce CP gradients in place and return (rel-L2, cosine, worst tensors)."""
    diff_sq = ref_sq = cp_sq = dot = 0.0
    per_parameter = []
    cp_parameters = dict(distributed.named_parameters())
    for name, ref_parameter in reference.named_parameters():
        cp_parameter = cp_parameters[name]
        if ref_parameter.grad is None:
            if cp_parameter.grad is not None:
                raise AssertionError(f"CP produced an unexpected gradient for {name}")
            continue
        if cp_parameter.grad is None:
            raise AssertionError(f"CP did not produce a gradient for {name}")
        dist.all_reduce(cp_parameter.grad, op=dist.ReduceOp.SUM)
        cp_gradient = cp_parameter.grad.detach().double()
        ref_gradient = ref_parameter.grad.detach().double()
        if not torch.isfinite(cp_gradient).all():
            raise AssertionError(f"CP produced a non-finite gradient for {name}")
        tensor_diff_sq = (cp_gradient - ref_gradient).square().sum().item()
        tensor_ref_sq = ref_gradient.square().sum().item()
        diff_sq += tensor_diff_sq
        ref_sq += tensor_ref_sq
        cp_sq += cp_gradient.square().sum().item()
        dot += (cp_gradient * ref_gradient).sum().item()
        per_parameter.append((tensor_diff_sq**0.5 / max(tensor_ref_sq**0.5, 1e-12), name))
    mtp_with_grad = [name for _, name in per_parameter if name.startswith("mtp.")]
    if not mtp_with_grad:
        raise AssertionError("no MTP parameter received a gradient")
    return (diff_sq / ref_sq) ** 0.5, dot / max((cp_sq * ref_sq) ** 0.5, 1e-12), per_parameter


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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if world_size < 2:
        print("ERROR: this parity contract needs at least two CP ranks.", file=sys.stderr)
        dist.destroy_process_group()
        sys.exit(1)

    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.models.glm5_next.cp import shard_batch_for_glm5_next_cp
    from nemo_automodel.components.moe.parallelizer import apply_cp

    torch.manual_seed(1234)
    reference = _model(device)
    distributed = _model(device)
    _sync_model(reference)
    distributed.load_state_dict(reference.state_dict())
    assert reference.mtp is not None

    sequence = 128
    torch.manual_seed(4321)
    input_ids = torch.randint(1, 95, (1, sequence), device=device)
    dist.broadcast(input_ids, src=0)
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    doc_ids = torch.tensor([[1] * 19 + [2] * 37 + [3] * 72], dtype=torch.int32, device=device)
    batch = {"input_ids": input_ids.clone(), "labels": targets.clone(), "_packed_seq_ids": doc_ids.clone()}
    (mtp_targets,) = reference.prepare_mtp_inputs_for_cp(batch).targets
    num_targets = targets.numel()
    num_mtp_targets = int((mtp_targets != -100).sum())

    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))["cp"]
    apply_cp(distributed, cp_mesh)
    _, local_batch, _ = shard_batch_for_glm5_next_cp(
        cp_mesh,
        None,
        {"input_ids": input_ids.clone(), "labels": targets.clone(), "_packed_seq_ids": doc_ids.clone()},
        shard_primary=False,
    )
    context = local_batch["glm5_next_packed_context"]
    start, end = context.seq_start, context.seq_start + context.local_seq_len
    if rank == 0:
        print(f"CP={world_size}; local interval {start}:{end}; document starts at 0, 19, 56", flush=True)

    def reference_step():
        output = reference(input_ids=input_ids, _packed_seq_ids=doc_ids)
        main, mtp = _loss_terms(
            output.logits,
            output.mtp_per_depth_h[0],
            reference.lm_head,
            targets,
            mtp_targets,
            num_targets,
            num_mtp_targets,
        )
        return output, main, mtp

    def distributed_step():
        output = distributed(
            input_ids=local_batch["input_ids"],
            padding_mask=local_batch["padding_mask"],
            glm5_next_packed_context=context,
        )
        main, mtp = _loss_terms(
            output.logits,
            output.mtp_per_depth_h[0],
            distributed.lm_head,
            targets[:, start:end],
            mtp_targets[:, start:end],
            num_targets,
            num_mtp_targets,
        )
        return output, main, mtp

    def global_terms(main, mtp):
        terms = torch.stack((main.detach(), mtp.detach()))
        dist.all_reduce(terms, op=dist.ReduceOp.SUM)
        return terms

    ref_output, ref_main, ref_mtp = reference_step()
    (ref_main + _MTP_WEIGHT * ref_mtp).backward()
    cp_output, cp_main, cp_mtp = distributed_step()
    (cp_main + _MTP_WEIGHT * cp_mtp).backward()
    cp_terms = global_terms(cp_main, cp_mtp)
    ref_terms = torch.stack((ref_main.detach(), ref_mtp.detach()))
    torch.testing.assert_close(cp_terms, ref_terms, rtol=1e-4, atol=1e-4)

    def gather(local: torch.Tensor) -> torch.Tensor:
        pieces = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(pieces, local.detach())
        return torch.cat(pieces, dim=1)

    cp_logits = gather(cp_output.logits)
    cp_mtp_states = gather(cp_output.mtp_per_depth_h[0])
    logit_diff = (cp_logits.float() - ref_output.logits.detach().float()).abs()
    mtp_diff = (cp_mtp_states.float() - ref_output.mtp_per_depth_h[0].detach().float()).abs()
    torch.testing.assert_close(cp_logits, ref_output.logits.detach(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(cp_mtp_states, ref_output.mtp_per_depth_h[0].detach(), rtol=1e-3, atol=1e-3)

    relative_l2, cosine, per_parameter = _gradient_parity(reference, distributed)
    mtp_relative = [value for value, name in per_parameter if name.startswith("mtp.")]
    if relative_l2 >= 1e-3 or cosine <= 0.99999 or max(mtp_relative) >= 1e-3:
        raise AssertionError(
            f"CP gradient parity failed: relative L2={relative_l2:.3e}, cosine={cosine:.6f}, "
            f"worst MTP tensor relative L2={max(mtp_relative):.3e}"
        )

    before = {name: parameter.detach().clone() for name, parameter in distributed.named_parameters()}
    for model in (reference, distributed):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    moved = sum(
        not torch.equal(parameter, before[name])
        for name, parameter in distributed.named_parameters()
        if name.startswith("mtp.") and parameter.requires_grad
    )
    with torch.no_grad():
        _, ref_main2, ref_mtp2 = reference_step()
        _, cp_main2, cp_mtp2 = distributed_step()
    cp_terms2 = global_terms(cp_main2, cp_mtp2)
    ref_terms2 = torch.stack((ref_main2, ref_mtp2))
    torch.testing.assert_close(cp_terms2, ref_terms2, rtol=1e-4, atol=1e-4)

    if rank == 0:
        worst = ", ".join(f"{name}={value:.3e}" for value, name in sorted(per_parameter, reverse=True)[:5])
        print(
            "GLM-5.3 packed CP MTP parity PASS: "
            f"CP1 vs CP{world_size}; step 1 main/MTP loss={cp_terms[0].item():.6f}/{cp_terms[1].item():.6f} vs "
            f"{ref_terms[0].item():.6f}/{ref_terms[1].item():.6f}; logits mean/max={logit_diff.mean().item():.3e}/"
            f"{logit_diff.max().item():.3e}; MTP states mean/max={mtp_diff.mean().item():.3e}/"
            f"{mtp_diff.max().item():.3e}; gradients compared={len(per_parameter)} "
            f"(MTP {len(mtp_relative)}, worst MTP rel-L2 {max(mtp_relative):.3e}), "
            f"grad rel-L2/cosine={relative_l2:.3e}/{cosine:.6f}; worst tensors: {worst}; "
            f"AdamW moved {moved} MTP tensors; step 2 main/MTP loss={cp_terms2[0].item():.6f}/"
            f"{cp_terms2[1].item():.6f} vs {ref_terms2[0].item():.6f}/{ref_terms2[1].item():.6f}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
