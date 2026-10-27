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
* ``long_image``: an image block longer than the sliding window across the rank boundary, so the
  window also cuts attention inside the block.
* ``packed``: two documents in one row; the reference runs each document on its own.
* ``padded``: right padding that reaches into the last rank's shard.

The CP pass runs with activation checkpointing, so the ring metadata is also read back during the
backward recompute.

The loss is the token-sum cross entropy over the global token count, so the CP gradients summed
over ranks must equal the reference gradients. A NaN in any gradient fails the gate, and a negative
control checks that it does.

Finally, an FSDP2-wrapped CP model and an unwrapped CP=1 model each take an AdamW step on the image
batch and then one on the text batch. In both, the text step must leave the vision embedder, which
no rank used, and its optimizer state untouched while it updates the text weights.

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
LONG_IMAGE_PATCHES = 24  # longer than the sliding window of 16


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

    for name, num_patches in (("image", NUM_PATCHES), ("long_image", LONG_IMAGE_PATCHES)):
        image_ids = ids.clone()
        start = seqlen // 2 - num_patches // 2  # the image block straddles the CP rank boundary
        image_ids[0, start : start + num_patches] = IMAGE_TOKEN_ID
        mm = (image_ids == IMAGE_TOKEN_ID).long()
        image_labels = torch.cat([image_ids[:, 1:], torch.full((1, 1), -100)], dim=1)
        image_labels[mm.bool()] = -100
        grid = torch.arange(num_patches)
        batches[name] = {
            "input_ids": image_ids,
            "labels": image_labels,
            "position_ids": positions,
            "mm_token_type_ids": mm,
            "pixel_values": torch.randn(1, num_patches, 48, generator=gen),
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

    real = seqlen * 3 // 4  # the pad tail covers part of the last rank's shard
    padded_ids = ids.clone()
    padded_ids[0, real:] = 0
    padded_labels = labels.clone()
    padded_labels[0, real - 1 :] = -100
    batches["padded"] = {
        "input_ids": padded_ids,
        "labels": padded_labels,
        "position_ids": positions,
        "attention_mask": (torch.arange(seqlen) < real).long().unsqueeze(0),
    }
    for batch in batches.values():
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
    return batches


def _token_loss_sum(logits, labels):
    """Summed cross entropy of ``logits`` ``[batch, sequence, vocab]`` against ``labels`` ``[batch, sequence]``."""
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


def _cp_backward(model, batch, device_mesh):
    """Backward of the token-mean loss through the CP sharder (the recipe path); returns this rank's loss."""
    from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder

    model.zero_grad(set_to_none=True)
    batch = {k: v.clone() for k, v in batch.items()}
    num_tokens = (batch["labels"] != -100).sum()
    train_ctx, batch = ContextParallelSharder(model, device_mesh, batch).shard(batch)
    labels = batch.pop("labels")
    with train_ctx():
        loss = _token_loss_sum(model(**batch).logits, labels) / num_tokens
        loss.backward()
    return loss


def _context_parallel(model, batch, device_mesh):
    loss = _cp_backward(model, batch, device_mesh)
    group = device_mesh["cp"].get_group()
    dist.all_reduce(loss, group=group)
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, group=group)
            grads[name] = param.grad.detach().clone()
    return loss.detach(), grads


def _image_then_text_updates(model, image_batch, text_batch, backward):
    """Take an AdamW step on the image batch, then one on the text batch, with ``backward(model, batch)``.

    Returns ``(vision_changed, text_changed, vision_steps)``: whether the text step changed any vision embedder
    weight, whether it changed any other weight, and each vision weight's AdamW step count.
    """
    from torch.distributed.tensor import DTensor

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    params = dict(model.named_parameters())

    def _step(batch):
        backward(model, batch)
        optimizer.step()

    def _snapshot():
        # Full tensors, not local shards: a shard can be empty on a rank, where ``torch.equal`` holds vacuously.
        return {n: (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone() for n, p in params.items()}

    _step(image_batch)
    before = _snapshot()
    _step(text_batch)
    after = _snapshot()
    changed = {n for n in params if not torch.equal(after[n], before[n])}
    vision = {n for n in params if ".embed_vision." in n}
    vision_steps = {n: int(optimizer.state[params[n]].get("step", 0)) for n in vision}
    return bool(changed & vision), bool(changed - vision), vision_steps


def _compare(reference, result, loss_tol, grad_tol):
    """Gate one batch's CP ``(loss, grads)`` against the reference; returns ``(ok, summary)``.

    A nonfinite loss or gradient fails the gate explicitly: a NaN after the first entry would never be picked
    by the ``max`` over relative errors.
    """
    (ref_loss, ref_grads), (cp_loss, cp_grads) = reference, result
    missing = sorted(set(ref_grads) ^ set(cp_grads))
    tensors = [("ref_loss", ref_loss), ("cp_loss", cp_loss), *ref_grads.items(), *cp_grads.items()]
    nonfinite = sorted({n for n, t in tensors if not torch.isfinite(t).all()})
    rel_errors = [
        (n, ((cp_grads[n].float() - g.float()).norm() / g.float().norm().clamp_min(1e-12)).item())
        for n, g in ref_grads.items()
        if n in cp_grads
    ]
    # No shared gradient name is a failure to report, not a ``max()`` of nothing to crash on.
    worst_name, worst = max(rel_errors, key=lambda item: item[1]) if rel_errors else ("<no shared grads>", float("inf"))
    dloss = abs(ref_loss.item() - cp_loss.item())
    ok = not missing and not nonfinite and dloss < loss_tol and worst < grad_tol
    summary = (
        f"loss ref={ref_loss.item():.6f} cp={cp_loss.item():.6f} |d|={dloss:.2e}  "
        f"worst grad rel={worst:.2e} ({worst_name})  missing={missing}  nonfinite={nonfinite}"
    )
    return ok, summary


def main():
    # The flex ring compiles per layer type and chunk geometry on the first CP forward.
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.float32
    # The tolerances assume true fp32; TF32, on by default in some containers, moves CP grads ~2e-2 off the reference.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from torch.distributed.device_mesh import init_device_mesh

    model = _build_model(device, dtype)
    batches = _batches(32 * world, device, dtype)
    # The ring is installed for good by the first CP shard, so every reference runs first.
    references = {name: _reference(model, batch) for name, batch in batches.items()}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    device_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))
    results = {name: _context_parallel(model, batch, device_mesh) for name, batch in batches.items()}

    loss_tol, grad_tol = 2e-4, 2e-3
    failed = []
    for name, reference in references.items():
        ok, summary = _compare(reference, results[name], loss_tol, grad_tol)
        if rank == 0:
            print(f"[{name}] {summary}  {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            failed.append(name)

    # Negative control: a NaN in a gradient after the first must fail the gate.
    cp_loss, cp_grads = results["text"]
    last = list(cp_grads)[-1]
    corrupted = {**cp_grads, last: torch.full_like(cp_grads[last], float("nan"))}
    control_ok, _ = _compare(references["text"], (cp_loss, corrupted), loss_tol, grad_tol)
    if rank == 0:
        print(f"[nan_control] NaN in {last}: {'FAIL (gate passed it)' if control_ok else 'PASS (gate rejected it)'}")
    if control_ok:
        failed.append("nan_control")

    from torch.distributed.fsdp import fully_shard

    from nemo_automodel.components.distributed.parallelizer_utils import configure_fsdp_unused_param_reduction

    cp1 = _image_then_text_updates(_build_model(device, dtype), batches["image"], batches["text"], _reference)
    fsdp_model = _build_model(device, dtype)
    fully_shard(fsdp_model, mesh=device_mesh)
    configure_fsdp_unused_param_reduction(fsdp_model)
    cp = _image_then_text_updates(
        fsdp_model, batches["image"], batches["text"], lambda model, batch: _cp_backward(model, batch, device_mesh)
    )
    vision_changed, text_changed, vision_steps = cp
    # The image step stepped the vision weights once in both; the text step must step none of them.
    updates_ok = cp == cp1 and not vision_changed and text_changed and 1 in vision_steps.values()
    if rank == 0:
        summary = "  ".join(
            f"{label} vision_changed={vc} text_changed={tc} vision_steps={sorted(set(steps.values()))}"
            for label, (vc, tc, steps) in (("cp1", cp1), ("cp", cp))
        )
        print(f"[image_then_text] {summary}  {'PASS' if updates_ok else 'FAIL'}", flush=True)
    if not updates_ok:
        failed.append("image_then_text")
    if rank == 0:
        print("RESULT:", "PASS" if not failed else f"FAIL {failed}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
