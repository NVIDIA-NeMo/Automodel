# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Gemma4 Unified context parallelism: CP=1 vs CP=world loss and gradient parity (2 GPUs).

A tiny randomly initialized Gemma4 Unified (sliding and full attention layers, K=V full layers,
vision-bidirectional sliding layers, the encoder-free vision embedder) runs three batches without
CP, then through ``ContextParallelSharder`` (the recipe path: the model-owned aux-only contiguous
shard installs the p2p ring, the forward embeds and splices the image on the full sequence and
keeps this rank's slice):

* ``text``: plain causal text.
* ``image``: one image block across the rank boundary whose tokens attend bidirectionally on the
  sliding layers (the reference forward uses the same generation masks).
* ``packed``: two documents in one row; the reference runs each document on its own.

The loss is the token-sum cross entropy over the global token count, so the CP gradients summed
over ranks must equal the reference gradients.

Run:
    torchrun --standalone --nproc-per-node=2 tests/functional_tests/context_parallel/run_gemma4_unified_cp.py
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F

IMAGE_TOKEN_ID = 100
NUM_PATCHES = 8


def _build_model(device, dtype):
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedConfig,
        Gemma4UnifiedTextConfig,
        Gemma4UnifiedVisionConfig,
    )

    from nemo_automodel.components.models.gemma4_unified.model import Gemma4UnifiedForConditionalGeneration

    text = Gemma4UnifiedTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=16,
        attention_k_eq_v=True,
        use_bidirectional_attention="vision",
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        pad_token_id=0,
    )
    vision = Gemma4UnifiedVisionConfig(
        mm_embed_dim=32, output_proj_dims=32, patch_size=2, pooling_kernel_size=2, model_patch_size=4, mm_posemb_size=16
    )
    config = Gemma4UnifiedConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), audio_config=None)
    config.image_token_id = IMAGE_TOKEN_ID
    config._attn_implementation = "sdpa"
    config.get_text_config()._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Gemma4UnifiedForConditionalGeneration(config).to(device=device, dtype=dtype)
    with torch.no_grad():
        # Peaked logits make the comparison meaningful on a random model.
        model.get_input_embeddings().weight.mul_(4.0)
        model.model.embed_vision.pos_embedding.normal_(std=0.5)
    for tensor in [*model.parameters(), *model.buffers()]:
        dist.broadcast(tensor.data, src=0)
    return model


def _batches(seqlen, device, dtype):
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(2, IMAGE_TOKEN_ID, (1, seqlen), generator=gen)
    labels = torch.cat([ids[:, 1:], torch.full((1, 1), -100)], dim=1)
    positions = torch.arange(seqlen).unsqueeze(0)
    batches = {"text": {"input_ids": ids, "labels": labels, "position_ids": positions}}

    image_ids = ids.clone()
    start = seqlen // 2 - NUM_PATCHES // 2  # the image block straddles the CP rank boundary
    image_ids[0, start : start + NUM_PATCHES] = IMAGE_TOKEN_ID
    mm = (image_ids == IMAGE_TOKEN_ID).long()
    image_labels = torch.cat([image_ids[:, 1:], torch.full((1, 1), -100)], dim=1)
    image_labels[mm.bool()] = -100
    grid = torch.arange(NUM_PATCHES)
    batches["image"] = {
        "input_ids": image_ids,
        "labels": image_labels,
        "position_ids": positions,
        "mm_token_type_ids": mm,
        "pixel_values": torch.randn(1, NUM_PATCHES, 48, generator=gen),
        "image_position_ids": torch.stack([grid % 4, grid // 4], dim=-1).unsqueeze(0),
    }

    split = seqlen // 2 + 6  # the document boundary falls inside a rank's shard
    doc_ids = torch.cat([torch.ones(split), torch.full((seqlen - split,), 2)]).long().unsqueeze(0)
    packed_positions = torch.cat([torch.arange(split), torch.arange(seqlen - split)]).unsqueeze(0)
    packed_labels = labels.clone()
    packed_labels[0, split - 1] = -100  # no target across the document boundary
    batches["packed"] = {
        "input_ids": ids,
        "labels": packed_labels,
        "position_ids": packed_positions,
        "_packed_seq_ids": doc_ids,
    }
    for batch in batches.values():
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
    return batches


def _token_loss_sum(logits, labels):
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")


def _reference(model, batch):
    """Loss and gradients without CP; packed rows run one document at a time."""
    model.zero_grad(set_to_none=True)
    num_tokens = (batch["labels"] != -100).sum()
    if "_packed_seq_ids" in batch:
        split = int((batch["_packed_seq_ids"] == 1).sum())
        loss = 0.0
        for span in (slice(0, split), slice(split, None)):
            logits = model(input_ids=batch["input_ids"][:, span], position_ids=batch["position_ids"][:, span]).logits
            loss = loss + _token_loss_sum(logits, batch["labels"][:, span])
    else:
        kwargs = {k: v for k, v in batch.items() if k != "labels"}
        loss = _token_loss_sum(model(**kwargs).logits, batch["labels"])
    loss = loss / num_tokens
    loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _context_parallel(model, batch, device_mesh):
    from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder

    model.zero_grad(set_to_none=True)
    batch = {k: v.clone() for k, v in batch.items()}
    num_tokens = (batch["labels"] != -100).sum()
    cp_sharder = ContextParallelSharder(model, device_mesh, batch)
    train_ctx, batch = cp_sharder.shard(batch)
    labels = batch.pop("labels")
    with train_ctx():
        logits = model(**batch).logits
        loss = _token_loss_sum(logits, labels) / num_tokens
        loss.backward()
    group = device_mesh["cp"].get_group()
    dist.all_reduce(loss, group=group)
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, group=group)
            grads[name] = param.grad.detach().clone()
    return loss.detach(), grads


def main():
    # The flex ring compiles per layer type and chunk geometry on the first CP forward.
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.float32

    from torch.distributed.device_mesh import init_device_mesh

    model = _build_model(device, dtype)
    batches = _batches(32 * world, device, dtype)
    # The ring is installed for good by the first CP shard, so every reference runs first.
    references = {name: _reference(model, batch) for name, batch in batches.items()}
    device_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))
    results = {name: _context_parallel(model, batch, device_mesh) for name, batch in batches.items()}

    loss_tol, grad_tol = 2e-4, 2e-3
    failed = []
    for name, (ref_loss, ref_grads) in references.items():
        cp_loss, cp_grads = results[name]
        missing = sorted(set(ref_grads) ^ set(cp_grads))
        worst_name, worst = max(
            (
                (n, ((cp_grads[n].float() - g.float()).norm() / g.float().norm().clamp_min(1e-12)).item())
                for n, g in ref_grads.items()
                if n in cp_grads
            ),
            key=lambda item: item[1],
        )
        dloss = abs(ref_loss.item() - cp_loss.item())
        ok = not missing and dloss < loss_tol and worst < grad_tol
        if rank == 0:
            print(
                f"[{name}] loss ref={ref_loss.item():.6f} cp={cp_loss.item():.6f} |d|={dloss:.2e}  "
                f"worst grad rel={worst:.2e} ({worst_name})  missing={missing}  {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
        if not ok:
            failed.append(name)
    if rank == 0:
        print("RESULT:", "PASS" if not failed else f"FAIL {failed}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
