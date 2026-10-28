#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run released four-layer Muse EP8/PP1 versus EP4/PP2 parity.

Both topologies default to the production TE attention, torch-mm experts,
HybridEP dispatch, and FSDP/MoE resharding used by the full recipe. The PP2
mode uses two physical stages with two decoder layers on each stage. An
optional SGD step proves backward and optimizer execution without introducing
optimizer-state numerics into the topology comparison.
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
import torch.nn.functional as F

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.models.muse_spark.layers import MuseSparkDecoderLayer
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config

PROBE_VOCAB_IDS = (0, 1, 2, 3, 7, 11, 42, 127, 1024, 4096, 8192, 32768, 65536, 100000, 150000, 202047)
TRACE_HIDDEN_IDS = (0, 1, 2, 7, 31, 127, 511, 1023, 2047, 4095, 6143, 8191)


def parse_args() -> argparse.Namespace:
    """Parse topology and output options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pp-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--ep-size", type=int, choices=(4, 8))
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--experts", choices=("gmm", "torch_mm"), default="torch_mm")
    parser.add_argument("--dispatcher", choices=("deepep", "hybridep"), default="hybridep")
    parser.add_argument("--train-step", action="store_true")
    parser.add_argument("--trace-hidden", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    return parser.parse_args()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return flattened FP32 cosine similarity."""
    return float(F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item())


def _compare(reference_path: Path, result: dict[str, Any]) -> bool:
    """Attach and validate EP4/PP2 parity against an EP8/PP1 artifact."""
    reference = json.loads(reference_path.read_text())
    for field in (
        "sequence_length",
        "batch_size",
        "probe_vocab_ids",
        "expert_backend",
        "expert_dispatcher",
        "reshard_after_forward",
        "moe_reshard_after_forward",
    ):
        if reference[field] != result[field]:
            raise ValueError(f"EP/PP parity requires identical {field}: {reference[field]} != {result[field]}")
    reference_logits = torch.tensor(reference["logits_probe"], dtype=torch.float32)
    candidate_logits = torch.tensor(result["logits_probe"], dtype=torch.float32)
    difference = (candidate_logits - reference_logits).abs()
    reference_losses = torch.tensor(reference["losses"], dtype=torch.float32)
    candidate_losses = torch.tensor(result["losses"], dtype=torch.float32)
    parity = {
        "loss_max_abs_diff": float((candidate_losses - reference_losses).abs().max().item()),
        "logits_max_abs_diff": float(difference.max().item()),
        "logits_mean_abs_diff": float(difference.mean().item()),
        "logits_cosine": _cosine(candidate_logits, reference_logits),
    }
    passed = (
        parity["loss_max_abs_diff"] < 1e-2
        and parity["logits_max_abs_diff"] < 0.6
        and parity["logits_mean_abs_diff"] < 3e-2
        and parity["logits_cosine"] > 0.999
    )
    result["parity_vs_ep8_pp1"] = parity
    result["parity_passed"] = passed
    print(json.dumps(parity, indent=2), flush=True)
    print("MUSE_SPARK EP8_PP1/EP4_PP2 PARITY:", "PASS" if passed else "FAIL", flush=True)
    return passed


def _make_batch(sequence_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two deterministic equal-length sequences and shifted labels."""
    positions = torch.arange(sequence_length, device=device)
    first = (positions * 7919 + 17) % 199999
    second = (positions * 3571 + 29) % 199999
    input_ids = torch.stack((first, second)).to(torch.long)
    labels = input_ids.roll(-1, dims=1)
    labels[:, -1] = -100
    return input_ids, labels


def _trainable_parameters(parts: list[torch.nn.Module]) -> list[torch.nn.Parameter]:
    """Collect each local pipeline part's trainable parameters once."""
    seen: set[int] = set()
    parameters: list[torch.nn.Parameter] = []
    for part in parts:
        for parameter in part.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def main() -> None:
    """Load, execute, and serialize one topology's parity evidence."""
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError(f"Muse Spark Open EP/PP parity requires eight ranks, got {world}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    captured_logits: list[torch.Tensor] = []

    def loss_fn(prediction: Any, target: torch.Tensor) -> torch.Tensor:
        """Capture fixed logits and return mean next-token cross entropy."""
        logits = prediction.logits if hasattr(prediction, "logits") else prediction
        probe_ids = torch.tensor(PROBE_VOCAB_IDS, device=logits.device, dtype=torch.long)
        captured_logits.append(logits.detach().float().index_select(-1, probe_ids).cpu())
        return F.cross_entropy(
            logits.float().flatten(0, 1),
            target.flatten(),
            ignore_index=-100,
            reduction="mean",
        )

    ep_size = args.ep_size or world // args.pp_size
    if world % (args.pp_size * ep_size):
        raise ValueError(f"world size {world} is not divisible by pp={args.pp_size} * ep={ep_size}")
    pipeline = None
    if args.pp_size > 1:
        pipeline = {
            "pp_schedule": "1f1b",
            "pp_microbatch_size": 1,
            "pp_batch_size": 2,
            "layers_per_stage": 2,
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
        cp_size=1,
        pp_size=args.pp_size,
        ep_size=ep_size,
        pipeline=pipeline,
        moe={"reshard_after_forward": True, "wrap_outer_model": True},
        activation_checkpointing=True,
        reshard_after_forward=True,
    )
    model_or_pipeline = NeMoAutoModelForCausalLM.from_pretrained(
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
            "freeze_language_model": not args.train_step,
        },
        use_liger_kernel=False,
    )
    is_pp = isinstance(model_or_pipeline, AutoPipeline)
    if is_pp != (args.pp_size > 1):
        raise AssertionError(f"unexpected AutoPipeline state for pp_size={args.pp_size}: {is_pp}")
    parts = model_or_pipeline.parts if is_pp else [model_or_pipeline]
    for part in parts:
        part.train(args.train_step)
    hidden_traces: dict[str, list[torch.Tensor]] = {}
    hidden_input_traces: dict[str, list[torch.Tensor]] = {}
    parameter_traces: dict[str, dict[str, Any]] = {}
    stage_input_traces: list[torch.Tensor] = []
    stage_output_traces: list[torch.Tensor] = []
    stage_io_metadata: list[dict[str, Any]] = []
    if args.trace_hidden:
        for part in parts:
            for name, module in part.named_modules():
                if not isinstance(module, MuseSparkDecoderLayer):
                    continue
                layer_idx = str(module.layer_idx)
                hidden_traces.setdefault(layer_idx, [])
                hidden_input_traces.setdefault(layer_idx, [])

                def capture_hidden(_module, _inputs, output, *, key=layer_idx):
                    hidden = output[0] if isinstance(output, tuple) else output
                    indices = torch.tensor(TRACE_HIDDEN_IDS, device=hidden.device, dtype=torch.long)
                    hidden_traces[key].append(hidden.detach().float().index_select(-1, indices).cpu())

                def capture_hidden_input(_module, inputs, *, key=layer_idx):
                    hidden = inputs[0]
                    indices = torch.tensor(TRACE_HIDDEN_IDS, device=hidden.device, dtype=torch.long)
                    hidden_input_traces[key].append(hidden.detach().float().index_select(-1, indices).cpu())

                parameter_traces[layer_idx] = {}
                for parameter_name, parameter in module.named_parameters():
                    if not any(
                        marker in parameter_name
                        for marker in (
                            "q_proj.weight",
                            "gate.weight",
                            "fc1_latent_proj.weight",
                            "gate_and_up_projs",
                            "add_attn.gate",
                        )
                    ):
                        continue
                    local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
                    flat = local.detach().float().reshape(-1)
                    parameter_traces[layer_idx][parameter_name] = {
                        "shape": list(local.shape),
                        "first": flat[:16].cpu().tolist(),
                    }

                module.register_forward_pre_hook(capture_hidden_input)
                module.register_forward_hook(capture_hidden)

            def capture_stage_input(_module, inputs):
                value = inputs[0]
                if value.ndim != 3 or value.shape[-1] < max(TRACE_HIDDEN_IDS) + 1:
                    return
                indices = torch.tensor(TRACE_HIDDEN_IDS, device=value.device, dtype=torch.long)
                stage_input_traces.append(value.detach().float().index_select(-1, indices).cpu())
                stage_io_metadata.append(
                    {
                        "kind": "input",
                        "shape": list(value.shape),
                        "stride": list(value.stride()),
                        "dtype": str(value.dtype),
                        "type": type(value).__name__,
                        "contiguous": value.is_contiguous(),
                    }
                )

            def capture_stage_output(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                if hasattr(value, "logits"):
                    value = value.logits
                if value.ndim != 3 or value.shape[-1] < max(TRACE_HIDDEN_IDS) + 1:
                    return
                indices = torch.tensor(TRACE_HIDDEN_IDS, device=value.device, dtype=torch.long)
                stage_output_traces.append(value.detach().float().index_select(-1, indices).cpu())
                stage_io_metadata.append(
                    {
                        "kind": "output",
                        "shape": list(value.shape),
                        "stride": list(value.stride()),
                        "dtype": str(value.dtype),
                        "type": type(value).__name__,
                        "contiguous": value.is_contiguous(),
                    }
                )

            part.register_forward_pre_hook(capture_stage_input)
            part.register_forward_hook(capture_stage_output)
    optimizer = None
    if args.train_step:
        parameters = _trainable_parameters(parts)
        if not parameters:
            raise AssertionError(f"rank {rank} has no trainable parameters")
        optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, foreach=False)

    input_ids, labels = _make_batch(args.sequence_length, device)
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    start = time.perf_counter()
    if is_pp:
        model_or_pipeline.update_seq_len(input_ids.shape[1])
        losses = [] if model_or_pipeline.info.has_last_stage else None
        target = labels if model_or_pipeline.info.has_last_stage else None
        model_or_pipeline.step(input_ids, target=target, losses=losses)
        local_losses = [value.detach().float().cpu() for value in losses] if losses is not None else []
        has_last_stage = bool(model_or_pipeline.info.has_last_stage)
    else:
        local_losses = []
        for microbatch in range(input_ids.shape[0]):
            output = model_or_pipeline(input_ids=input_ids[microbatch : microbatch + 1])
            loss = loss_fn(output, labels[microbatch : microbatch + 1])
            if args.train_step:
                loss.backward()
            local_losses.append(loss.detach().float().cpu())
        has_last_stage = True
    if optimizer is not None:
        optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    payload = None
    if has_last_stage and captured_logits:
        payload = {
            "rank": rank,
            "losses": [float(value.item()) for value in local_losses],
            "logits_probe": torch.cat(captured_logits, dim=0).tolist(),
        }
    trace_payload = {
        "rank": rank,
        "layers": {key: torch.cat(value, dim=0).tolist() for key, value in hidden_traces.items() if value},
        "layer_inputs": {key: torch.cat(value, dim=0).tolist() for key, value in hidden_input_traces.items() if value},
        "parameters": parameter_traces,
        "stage_inputs": [value.tolist() for value in stage_input_traces],
        "stage_outputs": [value.tolist() for value in stage_output_traces],
        "stage_io_metadata": stage_io_metadata,
        "modules": [name for part in parts for name, _ in part.named_modules() if name],
    }
    gathered: list[dict[str, Any] | None] = [None] * world
    dist.all_gather_object(gathered, payload)
    gathered_traces: list[dict[str, Any]] = [{} for _ in range(world)]
    dist.all_gather_object(gathered_traces, trace_payload)
    if rank == 0:
        candidates = [item for item in gathered if item is not None]
        if not candidates:
            raise AssertionError("no final pipeline rank captured logits")
        canonical = min(candidates, key=lambda item: item["rank"])
        for item in candidates[1:]:
            torch.testing.assert_close(
                torch.tensor(item["logits_probe"]),
                torch.tensor(canonical["logits_probe"]),
                rtol=0,
                atol=0,
            )
        result = {
            "model": args.model,
            "num_hidden_layers": 4,
            "schedule_num_hidden_layers": 62,
            "world_size": world,
            "ep_size": ep_size,
            "pp_size": args.pp_size,
            "sequence_length": args.sequence_length,
            "batch_size": 2,
            "probe_vocab_ids": list(PROBE_VOCAB_IDS),
            "losses": canonical["losses"],
            "logits_probe": canonical["logits_probe"],
            "train_step": args.train_step,
            "elapsed_seconds": elapsed,
            "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "final_stage_replica_count": len(candidates),
            "expert_backend": args.experts,
            "expert_dispatcher": args.dispatcher,
            "reshard_after_forward": True,
            "moe_reshard_after_forward": True,
        }
        if args.trace_hidden:
            canonical_layers: dict[str, Any] = {}
            canonical_layer_inputs: dict[str, Any] = {}
            canonical_parameters: dict[str, Any] = {}
            trace_ranks: dict[str, int] = {}
            for item in gathered_traces:
                for layer_idx, values in item["layers"].items():
                    if layer_idx not in canonical_layers:
                        canonical_layers[layer_idx] = values
                        canonical_layer_inputs[layer_idx] = item["layer_inputs"][layer_idx]
                        canonical_parameters[layer_idx] = item["parameters"][layer_idx]
                        trace_ranks[layer_idx] = item["rank"]
            result["trace_hidden_ids"] = list(TRACE_HIDDEN_IDS)
            result["hidden_traces"] = canonical_layers
            result["hidden_input_traces"] = canonical_layer_inputs
            result["parameter_traces"] = canonical_parameters
            result["hidden_trace_ranks"] = trace_ranks
            result["stage_modules"] = {
                str(item["rank"]): item["modules"] for item in gathered_traces if item["modules"]
            }
            result["rank_traces"] = {str(item["rank"]): item for item in gathered_traces}
        return_code = 0
        if args.reference is not None and not _compare(args.reference, result):
            return_code = 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "logits_probe"}, indent=2))
        print(f"MUSE_SPARK_EP{ep_size}_PP{args.pp_size}: PASS", flush=True)
    else:
        return_code = 0
    return_code_tensor = torch.tensor(return_code, device=device)
    dist.broadcast(return_code_tensor, src=0)
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(int(return_code_tensor.item()))


if __name__ == "__main__":
    main()
