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

"""MuseGlimmer 30B TE context-parallel parity on deterministic real MedPix data.

Run both commands on the same 8-GPU node:

    torchrun --standalone --nproc-per-node=8 \
      tests/functional_tests/context_parallel/run_muse_glimmer_te_cp_parity.py \
      --cp-size 1 --output /tmp/muse_glimmer_cp1.json

    torchrun --standalone --nproc-per-node=8 \
      tests/functional_tests/context_parallel/run_muse_glimmer_te_cp_parity.py \
      --cp-size 8 --output /tmp/muse_glimmer_cp8.json \
      --reference /tmp/muse_glimmer_cp1.json

The CP1 launch uses DP8/FSDP2, with the identical example replicated on every
rank. The CP8 launch uses the same FSDP2 group as its CP group. Both use the
recipe's ContextParallelSharder, model-owned post-vision sequence shard, TE
attention, external masked CE normalization, and recipe-equivalent backward
scale. The compact parity artifact contains selected logits at every token,
loss, and first-layer, last-layer, and final-norm parameter gradients.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoProcessor

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn
from nemo_automodel.components.datasets.vlm.datasets import make_medpix_dataset
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config

DEFAULT_MODEL = "../muse-glimmer-final-hf"
PROBE_VOCAB_IDS = (0, 1, 2, 3, 7, 11, 42, 127, 1024, 4096, 8192, 32768, 65536, 100000, 150000, 202047)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-size", type=int, choices=(1, 8), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument(
        "--num-medpix-samples",
        type=int,
        default=10,
        help="Concatenate this many consecutive real MedPix conversations into one multimodal training sequence.",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    return parser.parse_args()


def _to_device(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _build_batch(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], int, list[int]]:
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    dataset = make_medpix_dataset("mmoukouba/MedPix-VQA", split="train")
    dataset_indices = [(args.dataset_index + offset) % len(dataset) for offset in range(args.num_medpix_samples)]
    conversations = [message for index in dataset_indices for message in dataset[index]["conversation"]]
    batch = default_collate_fn(
        [{"conversation": conversations}],
        processor,
        max_length=args.max_length,
    )
    if int((batch["labels"] != -100).sum().item()) == 0:
        raise ValueError(f"MedPix examples {dataset_indices} have no supervised labels after collation.")
    source_length = batch["input_ids"].shape[1]
    return _to_device(batch, device), source_length, dataset_indices


def _norm_grad(norm: torch.nn.Module, name: str) -> tuple[str, torch.Tensor]:
    """Materialize one small FSDP/DTensor norm gradient for parity."""
    norm = getattr(norm, "_checkpoint_wrapped_module", norm)
    parameter = norm.weight
    if parameter.grad is None:
        raise AssertionError(f"Expected a gradient for {name}.")
    grad = parameter.grad
    if isinstance(grad, DTensor):
        grad = grad.full_tensor()
    return name, grad.detach().float().cpu()


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def _compare(reference_path: Path, result: dict[str, Any]) -> bool:
    reference = json.loads(reference_path.read_text())
    for field in (
        "dataset_indices",
        "sequence_length",
        "num_label_tokens",
        "supervised_positions",
        "probe_vocab_ids",
    ):
        if reference[field] != result[field]:
            raise ValueError(
                f"CP1/CP8 parity requires identical {field}: "
                f"CP1={reference[field]!r}, CP{result['cp_size']}={result[field]!r}."
            )
    logits_ref = torch.tensor(reference["logits_probe"], dtype=torch.float32)
    logits_cp = torch.tensor(result["logits_probe"], dtype=torch.float32)
    grad_ref = torch.tensor(reference["input_layernorm_grad"], dtype=torch.float32)
    grad_cp = torch.tensor(result["input_layernorm_grad"], dtype=torch.float32)
    supervised_positions = torch.tensor(reference["supervised_positions"], dtype=torch.long)

    logits_diff = (logits_cp - logits_ref).abs()
    supervised_logits_ref = logits_ref.index_select(0, supervised_positions)
    supervised_logits_cp = logits_cp.index_select(0, supervised_positions)
    supervised_logits_diff = (supervised_logits_cp - supervised_logits_ref).abs()
    grad_diff = (grad_cp - grad_ref).abs()
    loss_diff = abs(result["loss"] - reference["loss"])
    logits_ref_mean_abs = logits_ref.abs().mean().item()
    logits_relative_mean_abs = logits_diff.mean().item() / max(logits_ref_mean_abs, torch.finfo(torch.float32).eps)
    logits_cosine = torch.nn.functional.cosine_similarity(logits_cp.flatten(), logits_ref.flatten(), dim=0).item()
    supervised_logits_ref_mean_abs = supervised_logits_ref.abs().mean().item()
    supervised_logits_relative_mean_abs = supervised_logits_diff.mean().item() / max(
        supervised_logits_ref_mean_abs,
        torch.finfo(torch.float32).eps,
    )
    supervised_logits_cosine = torch.nn.functional.cosine_similarity(
        supervised_logits_cp.flatten(),
        supervised_logits_ref.flatten(),
        dim=0,
    ).item()
    grad_ref_norm = grad_ref.norm().item()
    grad_cp_norm = grad_cp.norm().item()
    grad_norm_ratio = grad_cp_norm / max(grad_ref_norm, torch.finfo(torch.float32).eps)
    grad_cosine = torch.nn.functional.cosine_similarity(grad_cp, grad_ref, dim=0).item()

    parity = {
        "loss_abs_diff": loss_diff,
        "logits_mean_abs_diff": logits_diff.mean().item(),
        "logits_max_abs_diff": logits_diff.max().item(),
        "logits_relative_mean_abs_diff": logits_relative_mean_abs,
        "logits_cosine": logits_cosine,
        "supervised_logits_mean_abs_diff": supervised_logits_diff.mean().item(),
        "supervised_logits_max_abs_diff": supervised_logits_diff.max().item(),
        "supervised_logits_relative_mean_abs_diff": supervised_logits_relative_mean_abs,
        "supervised_logits_cosine": supervised_logits_cosine,
        "grad_mean_abs_diff": grad_diff.mean().item(),
        "grad_max_abs_diff": grad_diff.max().item(),
        "grad_cp1_norm": grad_ref_norm,
        "grad_cp8_norm": grad_cp_norm,
        "grad_norm_ratio": grad_norm_ratio,
        "grad_cosine": grad_cosine,
    }
    for stem in ("last_layer_input_norm", "final_norm"):
        grad_ref_extra = torch.tensor(reference[f"{stem}_grad"], dtype=torch.float32)
        grad_cp_extra = torch.tensor(result[f"{stem}_grad"], dtype=torch.float32)
        grad_ref_extra_norm = grad_ref_extra.norm().item()
        grad_cp_extra_norm = grad_cp_extra.norm().item()
        parity[f"{stem}_grad_cp1_norm"] = grad_ref_extra_norm
        parity[f"{stem}_grad_cp8_norm"] = grad_cp_extra_norm
        parity[f"{stem}_grad_norm_ratio"] = grad_cp_extra_norm / max(
            grad_ref_extra_norm,
            torch.finfo(torch.float32).eps,
        )
        parity[f"{stem}_grad_cosine"] = torch.nn.functional.cosine_similarity(
            grad_cp_extra,
            grad_ref_extra,
            dim=0,
        ).item()
    result["parity_vs_cp1"] = parity

    # TE CP changes reduction order relative to CP1, and the effect compounds
    # through a 52-layer 4K stream. Check all-token direction/relative error,
    # then hold the supervised MedPix answer and gradient scale more tightly.
    long_context = logits_ref.shape[0] >= 2048
    passed = (
        loss_diff < 2e-2
        and parity["logits_relative_mean_abs_diff"] < (5e-2 if long_context else 1e-2)
        and parity["logits_cosine"] > (0.995 if long_context else 0.999)
        and parity["supervised_logits_relative_mean_abs_diff"] < (3e-2 if long_context else 2e-2)
        and parity["supervised_logits_cosine"] > 0.999
        and 0.9 < parity["last_layer_input_norm_grad_norm_ratio"] < 1.1
        and parity["last_layer_input_norm_grad_cosine"] > (0.95 if long_context else 0.98)
        and 0.9 < parity["final_norm_grad_norm_ratio"] < 1.1
        and parity["final_norm_grad_cosine"] > (0.99 if long_context else 0.995)
    )
    result["parity_passed"] = passed
    print(json.dumps(parity, indent=2), flush=True)
    print("MUSE_GLIMMER TE CP1/CP8 PARITY:", "PASS" if passed else "FAIL", flush=True)
    return passed


def main() -> None:
    args = _parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 8:
        raise ValueError(f"MuseGlimmer parity requires world_size=8, got {world}.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    setup = create_distributed_setup_from_config(
        ConfigNode(
            {
                "distributed": {
                    "strategy": "fsdp2",
                    "dp_size": None,
                    "tp_size": 1,
                    "cp_size": args.cp_size,
                    "pp_size": 1,
                    "ep_size": 1,
                    "sequence_parallel": False,
                    "activation_checkpointing": True,
                }
            }
        ),
        world_size=world,
    )
    model = NeMoAutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        backend={"attn": "te"},
        use_liger_kernel=False,
        distributed_setup=setup,
        freeze_config={
            "freeze_vision_tower": True,
            "freeze_audio_tower": True,
            "freeze_language_model": False,
        },
    )
    model.train()
    batch, source_sequence_length, dataset_indices = _build_batch(args, device)
    full_num_labels = int((batch["labels"] != -100).sum().item())
    supervised_positions = (batch["labels"][0] != -100).nonzero(as_tuple=False).flatten().tolist()

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
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with train_ctx():
        output = model(**batch)
        logits = output.logits
        logits_probe_local = logits.index_select(-1, probe_ids).detach()
        logits_probe = cp_sharder.gather_token_tensor(logits_probe_local, seq_dim=1, trim=True)
        loss = loss_fn(logits=logits, labels=labels, num_label_tokens=full_num_labels)
        # CP ranks own disjoint loss shards but share the FSDP reduction group,
        # which averages their gradients. Compensate by CP size only. CP1/DP8
        # has a complete local loss on every rank and therefore needs no scale.
        (loss * args.cp_size).backward()
    end_event.record()
    torch.cuda.synchronize(device)
    forward_backward_seconds = start_event.elapsed_time(end_event) / 1000.0
    peak_memory_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_memory_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)

    # CP ranks own disjoint label shards, so each rank's scalar is only its
    # contribution to the globally normalized loss. Sum those contributions
    # for reporting; CP1 has the full replicated loss on every rank.
    reported_loss = loss.detach().float()
    if args.cp_size > 1:
        dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM)

    grad_name, input_layernorm_grad = _norm_grad(
        model.model.layers[0].input_layernorm,
        "model.layers.0.input_layernorm.weight",
    )
    last_grad_name, last_layer_input_norm_grad = _norm_grad(
        model.model.layers[-1].input_layernorm,
        f"model.layers.{len(model.model.layers) - 1}.input_layernorm.weight",
    )
    final_grad_name, final_norm_grad = _norm_grad(
        model.model.norm,
        "model.norm.weight",
    )
    result = {
        "cp_size": args.cp_size,
        "world_size": world,
        "dataset_index": args.dataset_index,
        "dataset_indices": dataset_indices,
        "num_medpix_samples": args.num_medpix_samples,
        "actual_data": True,
        "source_sequence_length": source_sequence_length,
        "max_length": args.max_length,
        "sequence_length": int(logits_probe.shape[1]),
        "num_label_tokens": full_num_labels,
        "supervised_positions": supervised_positions,
        "forward_backward_seconds": forward_backward_seconds,
        "peak_memory_allocated_gib": peak_memory_allocated_gib,
        "peak_memory_reserved_gib": peak_memory_reserved_gib,
        "probe_vocab_ids": list(PROBE_VOCAB_IDS),
        "loss": float(reported_loss.item()),
        "logits_probe": logits_probe[0].float().cpu().tolist(),
        "input_layernorm_grad_name": grad_name,
        "input_layernorm_grad": input_layernorm_grad.tolist(),
        "last_layer_input_norm_grad_name": last_grad_name,
        "last_layer_input_norm_grad": last_layer_input_norm_grad.tolist(),
        "final_norm_grad_name": final_grad_name,
        "final_norm_grad": final_norm_grad.tolist(),
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
