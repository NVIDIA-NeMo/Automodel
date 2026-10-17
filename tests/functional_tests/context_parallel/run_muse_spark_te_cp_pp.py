#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-node Muse Spark Open packed THD CP x PP end-to-end training smoke.

This is the pre-multinode gate for the released first four layers.  On eight
GPUs its default topology is EP2 x CP2 x PP4; the production-schedule gate uses
EP4 x CP2 x PP2 with two virtual stages per rank and interleaved 1F1B.  Both
exercise real packed MedPix images, production TE p2p attention, configurable
HybridEP/DeepEP dispatch and grouped experts, activation checkpointing,
backward, and an optimizer step.  The smoke retains the same model/data paths
used by the full recipe; only layer count and topology degrees are scaled down.
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

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.datasets.vlm.datasets import MedPixDatasetConfig, PreTokenizedDatasetWrapperConfig
from nemo_automodel.components.datasets.vlm.loader import VlmDataloaderConfig, VlmProcessorConfig
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import NeatPackConfig
from nemo_automodel.components.datasets.vlm.pp_media import VLM_PP_MEDIA_KEY, stage_vlm_media_for_pp
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.muse_spark.processing import build_muse_spark_processor
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="mmoukouba/MedPix-VQA")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--pack-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--experts", choices=("gmm", "torch_mm"), default="torch_mm")
    parser.add_argument("--dispatcher", choices=("deepep", "hybridep"), default="hybridep")
    parser.add_argument("--pp-size", type=int, default=4)
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--pipeline-schedule", choices=("1f1b", "interleaved1f1b"), default="1f1b")
    parser.add_argument("--pipeline-batch-size", type=int, default=None)
    parser.add_argument("--layers-per-stage", type=int, default=1)
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


def _build_real_packed_batch(
    args: argparse.Namespace,
    device: torch.device,
    pipeline_batch_size: int,
) -> tuple[dict[str, Any], list[list[int]]]:
    """Materialize the first batch through the production MedPix input pipeline."""
    if args.pack_size % (2 * args.cp_size):
        raise ValueError(f"pack_size must be divisible by 2 * CP ({2 * args.cp_size})")

    dataloader_config = VlmDataloaderConfig(
        dataset_config=MedPixDatasetConfig(path_or_dataset=args.dataset, split=args.dataset_split),
        processor_config=VlmProcessorConfig(
            factory=build_muse_spark_processor,
            kwargs={"pretrained_model_name_or_path": args.model, "local_files_only": True},
        ),
        pretokenization=PreTokenizedDatasetWrapperConfig(
            max_length=args.pack_size,
            truncate=True,
            inject_fake_images=True,
        ),
        packing=NeatPackConfig(
            pack_size=args.pack_size,
            drop_long_samples=True,
            packing_ratio=0.9,
            balance_media_tokens=True,
            collate_max_length=args.pack_size,
            attn_implementation="sdpa",
            packing_format="thd",
        ),
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    build = dataloader_config.build(
        pretrained_model_name_or_path=args.model,
        dp_rank=0,
        dp_world_size=1,
        batch_size=pipeline_batch_size,
        get_rope_index=None,
        packing_attn_implementation="sdpa",
        pp_n_microbatches=pipeline_batch_size,
        cp_size=args.cp_size,
    )
    # Record the exact source rows selected by the production DistributedSampler.
    # Re-iterating it is deterministic and does not mutate sampler state.
    pack_indices = list(iter(build.dataloader.sampler))[:pipeline_batch_size]
    source_indices = [list(build.dataloader.dataset.bins[index]) for index in pack_indices]
    batch = next(iter(build.dataloader))
    if VLM_PP_MEDIA_KEY not in batch:
        raise AssertionError("Production PP-aware MedPix collate did not stage media chunks")
    return _to_device(batch, device), source_indices


def _trainable_parameters(parts: list[torch.nn.Module]) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    parameters: list[torch.nn.Parameter] = []
    for part in parts:
        for parameter in part.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


def _small_grad_evidence(parts: list[torch.nn.Module]) -> dict[str, Any]:
    """Return finite, nonzero evidence from a small local gradient tensor."""
    for part in parts:
        for name, parameter in part.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
            if grad.numel() > 1_000_000:
                continue
            finite = bool(torch.isfinite(grad).all().item())
            norm = float(grad.detach().float().norm().item())
            if finite and norm > 0:
                return {"name": name, "local_shape": list(grad.shape), "norm": norm, "finite": True}
    raise AssertionError("No finite nonzero small parameter gradient was found on this pipeline rank")


def main() -> None:
    args = _parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError(f"Muse Spark Open CP x PP smoke requires eight ranks, got {world}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    pp_size, cp_size, ep_size = args.pp_size, args.cp_size, args.ep_size
    pipeline_batch_size = args.pipeline_batch_size or pp_size
    if world % pp_size:
        raise ValueError(f"world_size={world} must be divisible by pp_size={pp_size}")
    if pipeline_batch_size < pp_size:
        raise ValueError("pipeline_batch_size must provide at least one microbatch per physical PP stage")
    normalized_label_count = {"value": 0}
    masked_ce = MaskedCrossEntropy()

    def loss_fn(prediction: Any, target: torch.Tensor) -> torch.Tensor:
        logits = prediction.logits if hasattr(prediction, "logits") else prediction
        loss = masked_ce(logits=logits, labels=target, num_label_tokens=normalized_label_count["value"])
        # CP ranks own disjoint target shards. FSDP averages over the combined
        # DP/CP mesh, so compensate the CP degree for the backward signal.
        return loss * cp_size

    pipeline = {
        "pp_schedule": args.pipeline_schedule,
        "pp_microbatch_size": 1,
        "pp_batch_size": pipeline_batch_size,
        "layers_per_stage": args.layers_per_stage,
        "scale_grads_in_schedule": False,
        "patch_inner_model": False,
        "patch_causal_lm_model": False,
        "loss_fn": loss_fn,
    }
    setup = create_distributed_setup_from_config(
        world_size=world,
        strategy="fsdp2",
        dp_size=None,
        tp_size=1,
        cp_size=cp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        pipeline=pipeline,
        moe={"reshard_after_forward": True, "wrap_outer_model": True},
        activation_checkpointing=True,
        defer_fsdp_grad_sync=False,
        reshard_after_forward=True,
    )
    pp = NeMoAutoModelForCausalLM.from_pretrained(
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
    if not isinstance(pp, AutoPipeline):
        raise AssertionError(f"Expected an AutoPipeline for pp_size={pp_size}")
    for part in pp.parts:
        part.train()
    optimizer = torch.optim.SGD(_trainable_parameters(pp.parts), lr=args.learning_rate, foreach=False)

    batch, dataset_indices = _build_real_packed_batch(args, device, pipeline_batch_size)
    document_lengths = batch["seq_lens"].tolist()
    materialized_sequence_length = int(batch["input_ids"].shape[1])
    normalized_label_count["value"] = int((batch["labels"] != -100).sum().item())
    cp_sharder = ContextParallelSharder(
        pp.parts[0],
        setup.mesh_context.device_mesh,
        batch,
        padding_token_id=0,
        num_chunks=pipeline_batch_size,
        invoke_pre_embed=True,
    )
    train_ctx, batch = cp_sharder.shard(batch)
    labels = batch.pop("labels")
    model_input = batch.pop("input_ids")

    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    start = time.perf_counter()
    losses = [] if pp.info.has_last_stage else None
    target = labels if pp.info.has_last_stage else None
    pp.update_seq_len(model_input.shape[1])
    with train_ctx(), stage_vlm_media_for_pp(pp, pp.parts, batch):
        pp.step(model_input, target=target, losses=losses, **batch)
    grad_evidence = _small_grad_evidence(pp.parts)
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    local_loss = torch.tensor(0.0, device=device)
    if pp.info.has_last_stage:
        # The schedule stores the CP-scaled local contribution. Undo that
        # scale, then sum the two CP shards to recover global normalized loss.
        local_loss = torch.stack(losses).sum().detach().float() / cp_size
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM, group=setup.mesh_context.device_mesh["cp"].get_group())
    dist.all_reduce(local_loss, op=dist.ReduceOp.MAX)

    local_result = {
        "rank": rank,
        "has_first_stage": bool(pp.info.has_first_stage),
        "has_last_stage": bool(pp.info.has_last_stage),
        "grad": grad_evidence,
        "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }
    gathered: list[dict[str, Any] | None] = [None] * world
    dist.all_gather_object(gathered, local_result)
    if rank == 0:
        result = {
            "model": args.model,
            "num_hidden_layers": 4,
            "schedule_num_hidden_layers": 62,
            "world_size": world,
            "ep_size": ep_size,
            "cp_size": cp_size,
            "pp_size": pp_size,
            "qkv_format": "thd",
            "attention_backend": "te",
            "expert_dispatcher": args.dispatcher,
            "expert_backend": args.experts,
            "pipeline_schedule": args.pipeline_schedule,
            "pipeline_microbatch_size": 1,
            "pipeline_batch_size": pipeline_batch_size,
            "layers_per_stage": args.layers_per_stage,
            "activation_checkpointing": True,
            "reshard_after_forward": True,
            "moe_reshard_after_forward": True,
            "defer_fsdp_grad_sync": False,
            "actual_data": True,
            "production_input_pipeline": True,
            "dataset_indices": dataset_indices,
            "document_lengths": document_lengths,
            "sequence_length_per_microbatch": materialized_sequence_length,
            "num_label_tokens": normalized_label_count["value"],
            "loss": float(local_loss.item()),
            "backward_completed": True,
            "optimizer_step_completed": True,
            "all_ranks_have_finite_nonzero_grad": True,
            "elapsed_seconds": elapsed,
            "rank_evidence": gathered,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "rank_evidence"}, indent=2))
        print(f"MUSE_SPARK PACKED EP{ep_size}_CP{cp_size}_PP{pp_size} TRAIN STEP: PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
