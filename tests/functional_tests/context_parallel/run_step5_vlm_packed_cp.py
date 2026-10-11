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

"""Packed image/crop CP parity through native VLM packing and CP hooks.

Random-weight component test; checkpoint loading is validated separately by the
native VLM recipe. Run with torchrun --standalone --nproc-per-node=2.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.nn import functional as F

from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import _build_packed_vlm_sample, _shift_sample
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import configure_packing_for_models
from nemo_automodel.components.models.step5.backend import Step5BackendConfig
from nemo_automodel.components.models.step5.configuration import Step5Config, Step5VLConfig
from nemo_automodel.components.models.step5.model import Step5ForConditionalGeneration
from nemo_automodel.components.moe.parallelizer import apply_cp


def sample(length: int, image_start: int | None, crops: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, 126, (length + 1,), generator=generator)
    output = dict(input_ids=tokens, labels=tokens.clone(), attention_mask=torch.ones_like(tokens))
    if image_start is not None:
        tokens[image_start : image_start + 4 + crops] = 127
        output["labels"][image_start : image_start + 4 + crops] = -100
        output["pixel_values"] = torch.randn(1, 3, 112, 112, generator=generator)
        output["num_patches"] = torch.tensor([crops])
        if crops:
            output["patch_pixel_values"] = torch.randn(crops, 3, 56, 56, generator=generator)
    return output


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    if reference.numel() == 0:
        return dict(max_abs=0.0, rms=0.0, kl_mean=0.0, kl_max=0.0, count=0)
    a, b = reference.float(), actual.float()
    delta = a - b
    log_a, log_b = a.log_softmax(-1), b.log_softmax(-1)
    kl = (log_a.exp() * (log_a - log_b)).sum(-1)
    return dict(
        max_abs=delta.abs().max().item(),
        rms=delta.square().mean().sqrt().item(),
        kl_mean=kl.mean().item(),
        kl_max=kl.max().item(),
    )


def build_model(topk: int, *, swa_varlen: bool = False, window: int = 65) -> Step5ForConditionalGeneration:
    torch.manual_seed(73)
    text = Step5Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=192,
        vocab_size=128,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=32,
        share_expert_dim=32,
        moe_layer_list=[3],
        dtype="bfloat16",
        num_nextn_predict_layers=0,
        sliding_window=window,
        sparse_config=dict(
            proxy_dim=32,
            sparse_indexer_rope_dim=8,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=topk,
            region_block_size=8,
        ),
    )
    model = Step5ForConditionalGeneration(
        Step5VLConfig(
            text_config=text,
            vision_config=dict(width=16, layers=2, heads=2, image_size=112, patch_size=14, mlp_ratio=2),
            image_token_id=127,
        ),
        backend=(
            Step5BackendConfig(attn="fa4", linear="torch", dispatcher="torch", experts="torch_mm", swa_varlen=True)
            if swa_varlen
            else BackendConfig(attn="fa4", linear="torch", dispatcher="torch", experts="torch_mm")
        ),
    )
    return model.cuda().train()


def collate(
    rows: list[list[dict[str, torch.Tensor]]], model: Step5ForConditionalGeneration, max_length: int | None = None
) -> dict[str, torch.Tensor]:
    packing = configure_packing_for_models([model])
    packs = [_build_packed_vlm_sample([_shift_sample(s) for s in row], 16384, 0) for row in rows]
    batch = neat_packed_vlm_collater(packs, packing=packing, max_length=max_length)
    return {k: v.cuda() for k, v in batch.items()}


def forward(model: Step5ForConditionalGeneration, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(**{k: v for k, v in batch.items() if k != "labels"}).logits


def check_case(
    name: str,
    rows: list[list[dict[str, torch.Tensor]]],
    topk: int,
    output: Path,
    max_length: int | None = None,
    *,
    swa_varlen: bool = False,
    window: int = 65,
) -> None:
    rank = dist.get_rank()
    model = build_model(topk, swa_varlen=swa_varlen, window=window)
    batch = collate(rows, model, max_length)
    labels = batch["labels"]
    count = (labels != -100).sum()
    valid = batch["attention_mask"] > 0
    selections = []
    handle = model.model.layers["3"].self_attn.indexer.register_forward_hook(
        lambda module, args, result: selections.append(result.region_ids.detach().cpu())
    )
    torch.cuda.reset_peak_memory_stats()
    logits = forward(model, batch)
    loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum") / count
    loss.backward()
    baseline = logits.detach().cpu()
    reference_loss = loss.item()
    baseline_grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    reference_selection = selections.pop()
    assert any(n.startswith("vision_model.") for n in baseline_grads)
    assert "vit_large_projector.weight" in baseline_grads
    peak1 = torch.cuda.max_memory_allocated()
    model.zero_grad(set_to_none=True)
    del logits, loss
    report = dict(
        name=name,
        rank=rank,
        swa_varlen=swa_varlen,
        window=window,
        doc_lengths=[[s["input_ids"].numel() - 1 for s in row] for row in rows],
        input_hash=hashlib.sha256(batch["input_ids"].cpu().numpy().tobytes()).hexdigest(),
        cp1_loss=reference_loss,
        gradients={},
        independent_documents=[],
        boundaries={},
    )
    # Different execution shapes and independently constructed per-document inputs.
    with torch.no_grad():
        for row_idx, row in enumerate(rows):
            offset = 0
            for doc in row:
                one = collate([[doc]], model)
                separate = forward(model, one).cpu()
                length = separate.shape[1]
                result = metrics(separate, baseline[row_idx : row_idx + 1, offset : offset + length])
                report["independent_documents"].append(result)
                assert result["kl_mean"] < 1e-4, ("independent document", result)
                offset += length
        # Change every input of document 0; later documents must be unchanged.
        changed_rows = [[dict(s) for s in row] for row in rows]
        first = changed_rows[0][0]
        first["input_ids"] = first["input_ids"].clone()
        text = first["input_ids"] != 127
        first["input_ids"][text] = 125
        if "pixel_values" in first:
            first["pixel_values"] = -3 * first["pixel_values"]
        if "patch_pixel_values" in first:
            first["patch_pixel_values"] = 2 * first["patch_pixel_values"]
        changed = forward(model, collate(changed_rows, model, max_length)).cpu()
        end0 = rows[0][0]["input_ids"].numel() - 1
        leak = (changed[0, end0:] - baseline[0, end0:]).abs().max().item()
        report["document_perturbation_leak_max"] = leak
        assert leak == 0, ("document leakage", leak)
        # Negative control: suppress the reset and demonstrate the oracle detects leakage.
        wrong = dict(batch)
        wrong.pop("_packed_seq_ids", None)
        wrong["attention_mask"] = wrong["attention_mask"].bool()
        wrong["position_ids"] = torch.arange(labels.shape[1], device="cuda")[None].expand_as(labels)
        wrong_logits = forward(model, wrong).cpu()
        wrong_error = (wrong_logits[0, end0:] - baseline[0, end0:]).abs().max().item()
        report["missing_reset_negative_control_max"] = wrong_error
        assert wrong_error > 1e-4
    selections.clear()
    gc.collect()
    torch.cuda.empty_cache()
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))
    apply_cp(model, mesh["cp"])
    local_batch = dict(batch)
    sharder = ContextParallelSharder(model, mesh, local_batch)
    context, local = sharder.shard(local_batch)
    torch.cuda.reset_peak_memory_stats()
    with context():
        actual = forward(model, local)
        local_loss = F.cross_entropy(actual.float().flatten(0, 1), local["labels"].flatten(), reduction="sum") / count
        local_loss.backward()
    peak2 = torch.cuda.max_memory_allocated()
    total_loss = local_loss.detach().clone()
    dist.all_reduce(total_loss)
    length = actual.shape[1]
    start, stop = rank * length, min((rank + 1) * length, labels.shape[1])
    actual_cpu = actual[:, : stop - start].detach().cpu()
    ref = baseline[:, start:stop]
    local_valid = valid[:, start:stop].cpu()
    report["logits"] = metrics(ref[local_valid], actual_cpu[local_valid])
    report["cp2_loss"] = total_loss.item()
    report["cp1_peak_bytes"], report["cp2_peak_bytes"] = peak1, peak2
    report["local_image_tokens"] = int((local["input_ids"] == 127).sum())
    report["local_padding_tokens"] = int(local["padding_mask"].sum())
    local_selection = selections.pop()[:, :, : stop - start]
    expected_selection = reference_selection[:, :, start:stop]
    # Pad changes only the invalid-region sentinel; compare actual selected sets.
    region_count = int(((batch["position_ids"] % 8 == 0) & valid).sum(1).max())
    a = local_selection.masked_fill(local_selection >= region_count, -1)
    b = expected_selection.masked_fill(expected_selection >= region_count, -1)
    same_set = (a == b).all(-1)
    report["valid_local_queries"] = int(local_valid.sum())
    report["selected_region_set_agreement"] = (
        same_set[local_valid[:, None].expand_as(same_set)].float().mean().item() if local_valid.any() else None
    )
    sparse = (batch["position_ids"][:, start:stop].cpu() >= (topk + 1) * 8) & local_valid
    report["queries_dropping_history"] = int(sparse.sum())
    for label, mask in (("before_sparse", local_valid & ~sparse), ("sparse", sparse)):
        if mask.any():
            report["boundaries"][label] = metrics(ref[mask], actual_cpu[mask])
    for row_idx, row in enumerate(rows):
        offset = 0
        for doc_idx, doc in enumerate(row):
            if doc_idx:
                left, right = max(start, offset - 8), min(stop, offset + 8)
                if left < right:
                    report["boundaries"][f"row{row_idx}_doc{doc_idx}"] = metrics(
                        baseline[row_idx, left:right], actual_cpu[row_idx, left - start : right - start]
                    )
            offset += doc["input_ids"].numel() - 1
    assert abs(total_loss.item() - reference_loss) < 0.003, report
    assert report["logits"]["kl_mean"] < 1e-4, report
    assert report["selected_region_set_agreement"] in (None, 1.0), report
    assert {n for n, p in model.named_parameters() if p.grad is not None} == set(baseline_grads)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad)
        a, b = p.grad.float().cpu(), baseline_grads.pop(name)
        relative = ((a - b).square().sum() / b.square().sum().clamp_min(1e-30)).sqrt().item()
        report["gradients"][name] = dict(relative_rms=relative, max_abs=(a - b).abs().max().item())
        assert relative < 0.025, (name, relative)
    handle.remove()
    (output / f"{report['name']}-rank{rank}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "gradients"}), flush=True)
    del model, baseline_grads, actual, local_loss
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def check_metadata_attacks(output: Path, *, swa_varlen: bool = False, window: int = 65) -> None:
    """Reject contradictory boundaries on every CP rank and isolate image gradients."""
    model = build_model(4, swa_varlen=swa_varlen, window=window)
    rows = [[sample(132, 1, 1, 10), sample(131, 1, 0, 20)]]
    base = collate(rows, model)
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("cp",))
    apply_cp(model, mesh["cp"])
    results = []
    for attack in (
        "missing_positions",
        "no_reset_at_cp_seam",
        "duplicate_position",
        "inconsistent_document_ids",
        "indexed_mask_without_positions",
    ):
        batch = {k: v.clone() for k, v in base.items()}
        if attack in ("missing_positions", "indexed_mask_without_positions"):
            batch.pop("position_ids")
            if attack == "indexed_mask_without_positions":
                batch.pop("_packed_seq_ids")
        elif attack == "no_reset_at_cp_seam":
            batch["position_ids"][:, 132] = 132
        elif attack == "duplicate_position":
            batch["position_ids"][:, 134] = batch["position_ids"][:, 133]
        else:
            batch["_packed_seq_ids"][:, 132] = 1
        rejected = False
        try:
            sharder = ContextParallelSharder(model, mesh, batch)
            context, local = sharder.shard(batch)
            with context(), torch.no_grad():
                forward(model, local)
        except ValueError as error:
            assert "packed" in str(error) or "document" in str(error), str(error)
            rejected = True
        assert rejected, attack
        results.append({"attack": attack, "rejected": rejected})
        dist.barrier()
    # Only document 1 contributes loss; document 0's pixels and crops must get zero gradient.
    batch = {k: v.clone() for k, v in base.items()}
    batch["pixel_values"] = batch["pixel_values"].detach().requires_grad_()
    batch["patch_pixel_values"] = batch["patch_pixel_values"].detach().requires_grad_()
    pixels, crops = batch["pixel_values"], batch["patch_pixel_values"]
    sharder = ContextParallelSharder(model, mesh, batch)
    context, local = sharder.shard(batch)
    with context():
        logits = forward(model, local)
        # Document IDs themselves are sharded by the native model hook.
        target = local["_packed_seq_ids"] == 2
        logits[target].float().square().sum().backward()
    for tensor in (pixels, crops):
        assert tensor.grad is not None
        dist.all_reduce(tensor.grad)
    assert torch.count_nonzero(pixels.grad[0]) == 0
    assert torch.count_nonzero(crops.grad) == 0
    assert pixels.grad[1].abs().sum() > 0
    report = {
        "rank": dist.get_rank(),
        "attacks": results,
        "cross_document_overview_gradient_max": pixels.grad[0].abs().max().item(),
        "cross_document_crop_gradient_max": crops.grad.abs().max().item(),
        "own_document_image_gradient_sum": pixels.grad[1].abs().sum().item(),
    }
    (output / f"metadata-attacks-rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--swa-varlen", action="store_true")
    parser.add_argument("--window", type=int, default=65)
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--metadata-attacks", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    if args.metadata_attacks:
        check_metadata_attacks(args.output, swa_varlen=args.swa_varlen, window=args.window)
        dist.destroy_process_group()
        return
    cases = {
        "image_cross_cp": [[sample(19, 3, 2, 1), sample(244, 111, 0, 2)]],
        "boundary_before_cp": [[sample(131, 3, 0, 1), sample(132, 2, 1, 2)]],
        "boundary_at_cp": [[sample(132, 3, 0, 1), sample(131, 2, 1, 2)]],
        "boundary_after_cp": [[sample(133, 3, 0, 1), sample(130, 2, 1, 2)]],
        "rank_without_image": [[sample(19, 3, 0, 1), sample(244, None, 0, 2)]],
        "two_rows_padding": [
            [sample(19, 3, 2, 1), sample(244, 111, 0, 2)],
            [sample(137, 3, 0, 3), sample(105, 2, 1, 4)],
        ],
    }
    if args.long:
        cases = {
            "packed_4k": [[sample(19, 3, 2, 1), sample(4077, 2027, 0, 2)]],
            "packed_8k": [[sample(19, 3, 2, 1), sample(8173, 4075, 0, 2)]],
        }
    if args.adversarial:
        cases = {}
        for boundary in (
            1,
            7,
            8,
            9,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            131,
            132,
            133,
            135,
            136,
            137,
            191,
            192,
            193,
            255,
            256,
            257,
            262,
        ):
            lengths = (boundary, 263 - boundary)
            docs = [
                sample(length, 1 if length >= 8 else None, 1 if length >= 10 else 0, 10 + i)
                for i, length in enumerate(lengths)
            ]
            cases[f"adversarial_boundary_{boundary}"] = [docs]
        generator = torch.Generator().manual_seed(1907)
        for case in range(8):
            cuts = sorted(torch.randperm(262, generator=generator)[:7].add(1).tolist())
            edges = [0, *cuts, 263]
            docs = [
                sample(b - a, 1 if b - a >= 8 else None, 1 if b - a >= 10 else 0, 100 + i)
                for i, (a, b) in enumerate(zip(edges, edges[1:]))
            ]
            cases[f"adversarial_random_{case}"] = [docs]
        cases["adversarial_short_documents"] = [[*[sample(1, None, 0, i) for i in range(8)], sample(255, 1, 2, 20)]]
        cases["adversarial_empty_rank"] = [[sample(7, 1, 0, 1), sample(12, 1, 1, 2)]]
    for name, rows in cases.items():
        check_case(
            name,
            rows,
            512 if args.long else 4,
            args.output,
            264 if name == "adversarial_empty_rank" else None,
            swa_varlen=args.swa_varlen,
            window=args.window,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
