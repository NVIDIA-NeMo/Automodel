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

"""Compare a streamed, unmodified TP1 oracle against the native EP model.

The reference phase keeps one complete released Block resident at a time. Layer1
retains the full original FP8 Engram table, without compacting rows or replacing
any reference operations. The native phase runs separately to bound GPU memory.
It also repeats the native forward with official attention/FFN inputs to isolate
component error from errors propagated through preceding residual blocks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from run_reference_parity import (
    REFERENCE_HASHES,
    REVISION,
    _capture_outputs,
    _evaluate_gates,
    _import_reference,
    _load_native,
    _load_official,
    _logit_metrics,
    _native_source_manifest,
    _Options,
    _tensor_metrics,
    _tokenize,
)
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)


def _stream_official(options: _Options, artifact: Path, device: torch.device) -> None:
    """Execute the exact official prefix while loading only one Block at a time."""
    reference = _import_reference(options.reference_dir)
    values = json.loads((options.reference_dir / "config.json").read_text())
    values.update(
        n_layers=options.num_layers,
        n_mtp_layers=0,
        dspark_block_size=0,
        dtype="bf16",
        expert_dtype=None,
        max_batch_size=1,
        max_seq_len=options.sequence_length,
    )
    args = reference.ModelArgs(**values)
    reference.world_size, reference.rank, reference.default_dtype = 1, 0, torch.bfloat16
    active_engram = any(index < args.n_layers for index in args.engram_layer_ids)
    tokenizer = AutoTokenizer.from_pretrained(options.checkpoint, local_files_only=True) if active_engram else None
    layout = reference.EngramLayout.from_args(args) if active_engram else None
    base = nn.Module()
    with torch.device(device), reference.set_dtype(torch.bfloat16):
        base.engram_hash = reference.NgramHashState(args, layout, tokenizer) if layout is not None else None
        base.embed = reference.ParallelEmbedding(args.vocab_size, args.dim)
        base.norm = reference.RMSNorm(args.dim, args.norm_eps)
        base.head = reference.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        base.layers = nn.ModuleDict()
    base.eval().requires_grad_(False)
    audits = {"shared": asdict(_load_official(base, options.checkpoint))}
    input_ids = _tokenize(options)
    captured, handles = _capture_outputs(base, native=False)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.device(device):
        tokens = input_ids.to(device)
        hashes = base.engram_hash(tokens, 0) if active_engram else None
        hidden = base.embed(tokens).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        pre_mix = reference.make_identity_pre_mix(hidden, args.hc_mult)
        for index in range(args.n_layers):
            LOGGER.info("Allocating and loading official TP1 Block %d", index)
            with reference.set_dtype(torch.bfloat16):
                layer = reference.Block(index, args, layout).eval().requires_grad_(False)
            # The namespace preserves exact release parameter names for strict loading.
            owner = nn.Module()
            owner.layers = nn.ModuleDict({str(index): layer})
            # SafeTensors builds CPU views; do not let official's CUDA default
            # device context turn a lazy table slice into a full GPU copy.
            with torch.device("cpu"):
                audits[f"layer.{index}"] = asdict(_load_official(owner, options.checkpoint))
            base.layers[str(index)] = layer
            observed, block_handles = _capture_outputs(base, native=False)
            # Capture actual common inputs, including the carried mHC coefficients.
            captured[f"layer.{index}.stream_input"] = hidden.cpu()
            captured[f"layer.{index}.pre_mix_input"] = pre_mix.cpu()
            if layer.engram is not None:
                hidden = layer.engram(hidden, hashes[:, :, layer.engram.layer_hash_index, :], None)
            hidden, pre_mix = layer(hidden, 0, pre_mix, None)
            torch.cuda.synchronize()
            captured.update(observed)
            for handle in block_handles:
                handle.remove()
            LOGGER.info(
                "Official TP1 Block %d forward complete; allocated %.2f GB", index, torch.cuda.memory_allocated() / 1e9
            )
            if index + 1 == args.n_layers:
                captured["final_streams"] = hidden.cpu()
                captured["final_pre_mix"] = pre_mix.cpu()
                hidden = layer.hc_pre(hidden, pre_mix)
            del base.layers[str(index)], owner, layer
            gc.collect()
            torch.cuda.empty_cache()
        logits = base.head(base.norm(hidden), full_logits=True)
        torch.cuda.synchronize()
        captured["logits"] = logits.cpu()
    for handle in handles:
        handle.remove()
    captured["input_ids"] = input_ids
    artifact.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.contiguous() for name, tensor in captured.items()}, str(artifact))
    report = {
        "reference_kind": "unmodified official inference, TP1, streamed original Blocks and full raw FP8 Engram",
        "reference_revision": REVISION,
        "reference_source_hashes": REFERENCE_HASHES,
        "checkpoint": str(options.checkpoint),
        "configuration": {
            "layers": args.n_layers,
            "sequence_length": options.sequence_length,
            "reference_tp": 1,
            "weight_compute_dtype": "bfloat16",
            "expert_compute_dtype": "bfloat16",
            "activation_quantization": "all original cache/index QAT operations unchanged",
            "full_vocabulary": logits.shape[-1],
            "all_positions": True,
        },
        "input_sha256": hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
        "state_audits": audits,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
        },
        "tensor_shapes": {name: list(value.shape) for name, value in captured.items()},
    }
    artifact.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("Official TP1 artifact written: %s", artifact)


def _compare_native(
    options: _Options,
    artifact: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    *,
    global_batch_one: bool = False,
) -> int:
    """Compare native outputs and common-input components with the saved oracle."""
    reference = load_file(str(artifact), device="cpu")
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    if metadata["reference_revision"] != REVISION or metadata["reference_source_hashes"] != REFERENCE_HASHES:
        raise ValueError("The reference artifact is not from the pinned official source")
    if metadata["configuration"]["layers"] != options.num_layers or reference["input_ids"].shape != (
        1,
        options.sequence_length,
    ):
        raise ValueError("Reference artifact scope differs from requested native scope")
    source_hashes = _native_source_manifest()
    model, audit = _load_native(options, device)
    LOGGER.info("Native state audit: %s", asdict(audit))
    input_ids = reference["input_ids"].to(device)
    compare_outputs = not global_batch_one or rank == 0
    attention_mask = torch.ones_like(input_ids) if compare_outputs else torch.zeros_like(input_ids)
    model_kwargs = {"attention_mask": attention_mask} if global_batch_one else {}
    captured, handles = _capture_outputs(model, native=True) if compare_outputs else ({}, [])
    started = time.monotonic()
    with torch.inference_mode():
        logits = model(input_ids, **model_kwargs).logits
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(logits).all())
        if compare_outputs:
            metrics, positions = _logit_metrics(reference["logits"], logits, chunk_size=options.metric_chunk_size)
        else:
            metrics, positions = None, None
    LOGGER.info("Natural native full-vocabulary metrics: %s", json.dumps(metrics, sort_keys=True))
    for handle in handles:
        handle.remove()
    if "engram_hash" in captured and not torch.equal(captured["engram_hash"], reference["engram_hash"]):
        raise ValueError("Native hashes differ from the original official table coordinates")
    activations = {name: asdict(_tensor_metrics(reference[name], value)) for name, value in captured.items()}
    del logits, captured

    # Only native inputs are substituted. The official artifact came from a
    # literal unmodified execution. Both attn and FFN get their exact oracle input.
    def replace_input(name: str):
        def replace(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            """Substitute an official [batch, sequence, hidden] component input."""
            return (reference[name].to(device), *inputs[1:])

        return replace

    common_captured, common_handles = _capture_outputs(model, native=True) if compare_outputs else ({}, [])
    if compare_outputs:
        for index, layer in model.model.layers.items():
            for component in ("attn", "ffn"):
                common_handles.append(
                    getattr(layer, component).register_forward_pre_hook(
                        replace_input(f"layer.{index}.{component}_norm")
                    )
                )
    with torch.inference_mode():
        model(input_ids, **model_kwargs)
        torch.cuda.synchronize()
    for handle in common_handles:
        handle.remove()
    common_metrics = {
        name: asdict(_tensor_metrics(reference[name], value))
        for name, value in common_captured.items()
        if name.endswith((".attn", ".ffn"))
    }
    for name, values in common_metrics.items():
        LOGGER.info("Common-input %s: %s", name, json.dumps(values, sort_keys=True))
    audits = {"native": asdict(audit), **metadata["state_audits"]}
    if compare_outputs:
        gates = _evaluate_gates(metrics, audits)
        gates["numerical_comparison_performed"] = True
    else:
        gates = {
            "passed": None,
            "numerical_comparison_performed": False,
            "scope": "All-masked collective participant; no oracle numerical comparison",
            "checks": {
                "all_positions_finite": finite,
                "state_coverage": all(
                    value["expected"] == value["loaded"]
                    and not any(value[key] for key in ("missing", "unexpected", "shape_mismatches"))
                    for value in audits.values()
                ),
            },
        }
    final_source_hashes = _native_source_manifest()
    changed_sources = [name for name, value in source_hashes.items() if final_source_hashes.get(name) != value]
    if changed_sources:
        raise RuntimeError(f"Native source changed during parity: {changed_sources}")
    report = {
        "reference_artifact": str(artifact),
        "reference_metadata": metadata,
        "rank": rank,
        "native_source_hashes_before_import": source_hashes,
        "configuration": {
            "native_tp": 1,
            "native_ep": world_size,
            "native_engram_owners": world_size,
            "native_dispatcher": "hybridep" if world_size > 1 else "torch",
            "attention_backend": options.attention_backend,
            "expert_backend": options.expert_backend or ("torch_mm" if world_size > 1 else "torch"),
            "official_tp": 1,
            "all_positions_and_logits": compare_outputs,
            "native_global_valid_batch": 1 if global_batch_one else world_size,
            "input_distribution": "rank0_valid_other_ranks_all_masked"
            if global_batch_one
            else "replicated_valid_sample",
            "rank_has_valid_sample": compare_outputs,
        },
        "state_audit": audits,
        "logits": metrics,
        "activations": activations,
        "common_input_components": common_metrics,
        "gates": gates,
        "elapsed_seconds": time.monotonic() - started,
        "native_source_files_changed_during_forward": changed_sources,
    }
    rank_output = (
        options.output if world_size == 1 else options.output.with_name(f"{options.output.stem}.rank{rank}.json")
    )
    rank_output.write_text(json.dumps(report, indent=2) + "\n")
    if positions is not None:
        save_file(positions, str(rank_output.with_suffix(".positions.safetensors")))
    passed = bool(gates["passed"])
    if world_size > 1:
        reports = [None] * world_size if rank == 0 else None
        dist.gather_object(report, reports, dst=0)
        if rank == 0:
            report["all_rank_logits"] = {
                str(item["rank"]): item["logits"] for item in reports if item["logits"] is not None
            }
            report["all_rank_gates"] = {str(item["rank"]): item["gates"] for item in reports}
            report["all_rank_state_audits"] = {str(item["rank"]): item["state_audit"]["native"] for item in reports}
            passed = all(
                item["gates"]["passed"]
                if item["gates"]["numerical_comparison_performed"]
                else all(item["gates"]["checks"].values())
                for item in reports
            )
            report["all_rank_validation_passed"] = passed
            options.output.write_text(json.dumps(report, indent=2) + "\n")
        result = [passed if rank == 0 else None]
        dist.broadcast_object_list(result, src=0)
        passed = result[0]
    return 0 if passed else 1


def main() -> int:
    """Run one reference-generation or native-comparison phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "native"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--metric-chunk-size", type=int, default=32)
    parser.add_argument("--input-text", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-backend", choices=("eager", "sdpa", "tilelang"), default="sdpa")
    parser.add_argument("--expert-backend", choices=("torch", "torch_mm"))
    parser.add_argument(
        "--native-global-batch-one",
        action="store_true",
        help="Give rank0 the oracle sample and mask every auxiliary rank; compare rank0's entire output",
    )
    parsed = vars(parser.parse_args())
    mode, artifact = parsed.pop("mode"), parsed.pop("artifact")
    global_batch_one = parsed.pop("native_global_batch_one")
    if mode == "reference" and global_batch_one:
        raise ValueError("--native-global-batch-one applies only to native comparison")
    parsed["reference_dir"] = parsed["reference_dir"] or parsed["checkpoint"] / "inference"
    options = _Options(**parsed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if mode == "reference" and world_size != 1:
        raise ValueError("The streaming oracle must run TP1 in a single process")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s rank={rank} %(levelname)s %(message)s", stream=sys.stdout
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(options.seed)
    torch.set_float32_matmul_precision("highest")
    try:
        if mode == "reference":
            _stream_official(options, artifact, device)
            return 0
        return _compare_native(options, artifact, device, rank, world_size, global_batch_one=global_batch_one)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
