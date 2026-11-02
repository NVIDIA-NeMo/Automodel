#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark Open four-layer packed-THD EP/CP parity on real MedPix data.

Run both commands in the same eight-GPU allocation::

    torchrun --standalone --nproc-per-node=8 \
      tests/functional_tests/context_parallel/run_muse_spark_te_cp_parity.py \
      --cp-size 1 --output /tmp/muse_spark_ep8_cp1.json --model /path/to/model

    torchrun --standalone --nproc-per-node=8 \
      tests/functional_tests/context_parallel/run_muse_spark_te_cp_parity.py \
      --cp-size 2 --output /tmp/muse_spark_ep4_cp2.json \
      --reference /tmp/muse_spark_ep8_cp1.json --model /path/to/model

The reference topology is EP8 x CP1 and the sharded topology is EP4 x CP2.
Both execute the released first four layers, production Transformer Engine THD
attention, torch-mm experts, HybridEP, FSDP/MoE resharding, frozen native vision tower, activation
checkpointing, masked CE, and backward.  Two independently tokenized MedPix
conversations are concatenated as separate documents in one THD pack, so the
test also proves that attention does not cross document boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn
from nemo_automodel.components.datasets.vlm.datasets import make_medpix_dataset
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.muse_spark.processing import build_muse_spark_processor
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config

PROBE_VOCAB_IDS = (0, 1, 2, 3, 7, 11, 42, 127, 1024, 4096, 8192, 32768, 65536, 100000, 150000, 202047)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cp-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--num-medpix-documents", type=int, default=2)
    parser.add_argument("--experts", choices=("gmm", "torch_mm"), default="torch_mm")
    parser.add_argument("--dispatcher", choices=("deepep", "hybridep"), default="hybridep")
    return parser.parse_args()


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _build_real_packed_batch(args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], list[int]]:
    """Tokenize MedPix documents independently, then concatenate one THD pack."""
    processor = build_muse_spark_processor(args.model, local_files_only=True)
    dataset = make_medpix_dataset("mmoukouba/MedPix-VQA", split="train")
    indices = [(args.dataset_index + offset) % len(dataset) for offset in range(args.num_medpix_documents)]
    documents = [default_collate_fn([dataset[index]], processor, max_length=None) for index in indices]
    lengths = [int(document["input_ids"].shape[1]) for document in documents]

    batch: dict[str, Any] = {
        "input_ids": torch.cat([document["input_ids"] for document in documents], dim=1),
        "labels": torch.cat([document["labels"] for document in documents], dim=1),
        "position_ids": torch.cat([torch.arange(length).unsqueeze(0) for length in lengths], dim=1),
        "seq_lens": torch.tensor([lengths], dtype=torch.long),
        "seq_lens_padded": torch.tensor([lengths], dtype=torch.long),
        "qkv_format": "thd",
    }
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        values = [document[key] for document in documents if document.get(key) is not None]
        if values:
            batch[key] = torch.cat(values, dim=0)
    return _to_device(batch, device), indices


def _materialize_grad(parameter: torch.nn.Parameter | None, name: str) -> tuple[str, torch.Tensor]:
    if parameter is None or parameter.grad is None:
        raise AssertionError(f"Expected a gradient for {name}")
    grad = parameter.grad
    if isinstance(grad, DTensor):
        grad = grad.full_tensor()
    grad = grad.detach().float().cpu()
    if not torch.isfinite(grad).all():
        raise AssertionError(f"Non-finite gradient in {name}")
    return name, grad


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0).item())


def _compare(reference_path: Path, result: dict[str, Any]) -> bool:
    reference = json.loads(reference_path.read_text())
    for field in (
        "dataset_indices",
        "document_lengths",
        "sequence_length",
        "num_label_tokens",
        "supervised_positions",
        "probe_vocab_ids",
        "expert_backend",
        "expert_dispatcher",
        "reshard_after_forward",
        "moe_reshard_after_forward",
    ):
        if reference[field] != result[field]:
            raise ValueError(f"Packed CP parity requires identical {field}: {reference[field]} != {result[field]}")

    logits_ref = torch.tensor(reference["logits_probe"], dtype=torch.float32)
    logits_cp = torch.tensor(result["logits_probe"], dtype=torch.float32)
    supervised = torch.tensor(reference["supervised_positions"], dtype=torch.long)
    logits_diff = (logits_cp - logits_ref).abs()
    supervised_ref = logits_ref.index_select(0, supervised)
    supervised_cp = logits_cp.index_select(0, supervised)
    supervised_diff = (supervised_cp - supervised_ref).abs()
    loss_diff = abs(float(result["loss"]) - float(reference["loss"]))
    parity: dict[str, float] = {
        "loss_abs_diff": loss_diff,
        "logits_mean_abs_diff": float(logits_diff.mean().item()),
        "logits_max_abs_diff": float(logits_diff.max().item()),
        "logits_relative_mean_abs_diff": float(logits_diff.mean().item() / logits_ref.abs().mean().clamp_min(1e-12)),
        "logits_cosine": _cosine(logits_cp, logits_ref),
        "supervised_logits_mean_abs_diff": float(supervised_diff.mean().item()),
        "supervised_logits_max_abs_diff": float(supervised_diff.max().item()),
        "supervised_logits_relative_mean_abs_diff": float(
            supervised_diff.mean().item() / supervised_ref.abs().mean().clamp_min(1e-12)
        ),
        "supervised_logits_cosine": _cosine(supervised_cp, supervised_ref),
    }
    for stem in ("first_layer_norm", "last_layer_norm", "final_norm"):
        grad_ref = torch.tensor(reference[f"{stem}_grad"], dtype=torch.float32)
        grad_cp = torch.tensor(result[f"{stem}_grad"], dtype=torch.float32)
        ref_norm = grad_ref.norm().clamp_min(1e-12)
        parity[f"{stem}_grad_mean_abs_diff"] = float((grad_cp - grad_ref).abs().mean().item())
        parity[f"{stem}_grad_norm_ratio"] = float((grad_cp.norm() / ref_norm).item())
        parity[f"{stem}_grad_cosine"] = _cosine(grad_cp, grad_ref)

    passed = (
        loss_diff < 2e-2
        and parity["logits_relative_mean_abs_diff"] < 2e-2
        and parity["logits_cosine"] > 0.998
        and parity["supervised_logits_relative_mean_abs_diff"] < 2e-2
        and parity["supervised_logits_cosine"] > 0.998
        and 0.9 < parity["last_layer_norm_grad_norm_ratio"] < 1.1
        and parity["last_layer_norm_grad_cosine"] > 0.95
        and 0.9 < parity["final_norm_grad_norm_ratio"] < 1.1
        and parity["final_norm_grad_cosine"] > 0.98
    )
    result["parity_vs_ep8_cp1"] = parity
    result["parity_passed"] = passed
    print(json.dumps(parity, indent=2), flush=True)
    print("MUSE_SPARK PACKED EP8_CP1/EP4_CP2 PARITY:", "PASS" if passed else "FAIL", flush=True)
    return passed


def main() -> None:
    args = _parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError(f"Muse Spark Open packed parity requires eight ranks, got {world}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    ep_size = world // args.cp_size
    setup = create_distributed_setup_from_config(
        world_size=world,
        strategy="fsdp2",
        dp_size=None,
        tp_size=1,
        cp_size=args.cp_size,
        pp_size=1,
        ep_size=ep_size,
        moe={"reshard_after_forward": True, "wrap_outer_model": True},
        activation_checkpointing=True,
        reshard_after_forward=True,
    )
    model = NeMoAutoModelForCausalLM.from_pretrained(
        args.model,
        config={"num_hidden_layers": 4, "schedule_num_hidden_layers": 62},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        backend={
            "attn": "te",
            "linear": "torch",
            "rms_norm": "torch_fp32",
            "experts": args.experts,
            "dispatcher": args.dispatcher,
            "enable_hf_state_dict_adapter": True,
            "enable_fsdp_optimizations": True,
        },
        distributed_setup=setup,
        freeze_config={
            "freeze_vision_tower": True,
            "freeze_audio_tower": True,
            "freeze_language_model": False,
        },
        use_liger_kernel=False,
    )
    model.train()
    batch, dataset_indices = _build_real_packed_batch(args, device)
    document_lengths = batch["seq_lens"][0].tolist()
    full_labels = batch["labels"].clone()
    full_num_labels = int((full_labels != -100).sum().item())
    supervised_positions = (full_labels.reshape(-1) != -100).nonzero(as_tuple=False).flatten().tolist()

    cp_sharder = ContextParallelSharder(
        model,
        setup.mesh_context.device_mesh,
        batch,
        padding_token_id=0,
    )
    train_ctx, batch = cp_sharder.shard(batch)
    labels = batch.pop("labels")

    loss_fn = MaskedCrossEntropy()
    probe_ids = torch.tensor(PROBE_VOCAB_IDS, dtype=torch.long, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    start = time.perf_counter()
    with train_ctx():
        output = model(**batch)
        logits = output.logits
        logits_probe_local = logits.index_select(-1, probe_ids).detach()
        logits_probe = cp_sharder.gather_token_tensor(logits_probe_local, seq_dim=1, trim=True)
        loss = loss_fn(logits=logits, labels=labels, num_label_tokens=full_num_labels)
        (loss * args.cp_size).backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    reported_loss = loss.detach().float()
    if args.cp_size > 1:
        dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM, group=setup.mesh_context.device_mesh["cp"].get_group())

    layers = (
        list(model.model.layers.values())
        if isinstance(model.model.layers, torch.nn.ModuleDict)
        else list(model.model.layers)
    )
    first_name, first_grad = _materialize_grad(
        layers[0].self_attn.pre_qkv_proj_norm.weight,
        "model.layers.0.self_attn.pre_qkv_proj_norm.weight",
    )
    last_name, last_grad = _materialize_grad(
        layers[-1].self_attn.pre_qkv_proj_norm.weight,
        f"model.layers.{layers[-1].layer_idx}.self_attn.pre_qkv_proj_norm.weight",
    )
    final_name, final_grad = _materialize_grad(model.model.norm.weight, "model.norm.weight")

    result = {
        "model": args.model,
        "num_hidden_layers": 4,
        "schedule_num_hidden_layers": 62,
        "world_size": world,
        "ep_size": ep_size,
        "cp_size": args.cp_size,
        "pp_size": 1,
        "qkv_format": "thd",
        "attention_backend": "te",
        "expert_backend": args.experts,
        "expert_dispatcher": args.dispatcher,
        "reshard_after_forward": True,
        "moe_reshard_after_forward": True,
        "activation_checkpointing": True,
        "actual_data": True,
        "dataset_indices": dataset_indices,
        "document_lengths": document_lengths,
        "sequence_length": int(logits_probe.shape[1]),
        "num_label_tokens": full_num_labels,
        "supervised_positions": supervised_positions,
        "probe_vocab_ids": list(PROBE_VOCAB_IDS),
        "loss": float(reported_loss.item()),
        "logits_probe": logits_probe[0].float().cpu().tolist(),
        "first_layer_norm_grad_name": first_name,
        "first_layer_norm_grad": first_grad.tolist(),
        "last_layer_norm_grad_name": last_name,
        "last_layer_norm_grad": last_grad.tolist(),
        "final_norm_grad_name": final_name,
        "final_norm_grad": final_grad.tolist(),
        "forward_backward_seconds": elapsed,
        "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }

    return_code = 0
    if rank == 0:
        if args.reference is not None and not _compare(args.reference, result):
            return_code = 1
        _write_result(args.output, result)
        print(f"Wrote {args.output}", flush=True)
    return_code_tensor = torch.tensor(return_code, device=device)
    dist.broadcast(return_code_tensor, src=0)
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(int(return_code_tensor.item()))


if __name__ == "__main__":
    main()
