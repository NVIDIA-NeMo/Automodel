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

"""Native four-layer, fixed-EP packed CP1/CP2 comparison over three updates."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor

from tests.functional_tests.models.flint.run_four_layer_logits import sha256_file, tensor_sha256
from tests.functional_tests.models.flint.sdpa_reference import LogitAccumulator, ReferenceConfig

# Declared before any released CP measurements. Near-zero individual tensors
# have absolute summaries; relative acceptance applies to the entire gradient.
TOLERANCES = {
    "mean_kl_lt": 1e-3,
    "loss_abs_le": 1e-3,
    "all_gradient_relative_l2_le": 1e-2,
    "global_gradient_norm_relative_error_le": 5e-3,
    "all_gradient_cosine_ge": 0.99995,
}
OPTIMIZER = {"lr": 1e-5, "weight_decay": 0.01, "betas": (0.9, 0.95), "eps": 1e-8}


def comparison_topology(cp_size: int, ep_size: int) -> dict[str, int]:
    """Describe the two-rank PP1 control, including duplicated logical packs."""
    if cp_size not in (1, 2) or ep_size not in (1, 2):
        raise ValueError("the controlled comparison supports CP1/2 and fixed EP1/2")
    return {"world_size": 2, "dp_size": 2 // cp_size, "cp_size": cp_size, "ep_size": ep_size, "pp_size": 1}


def expected_global_targets(unique_targets: int, cp_size: int) -> int:
    """Count supervised targets over CP shards and identical DP pack repeats."""
    if unique_targets <= 0:
        raise ValueError("each logical pack must contain supervised targets")
    return unique_targets * comparison_topology(cp_size, 1)["dp_size"]


def normalize_gradients(model: torch.nn.Module, mesh_context: Any) -> tuple[Any, dict[str, Any]]:
    """Normalize expert sums once and report the unclipped global gradient norm.

    Returns:
        The shared norm and options for a subsequent direct clip_grad_norm call.
        Callers must not call scale_grads_and_clip_grad_norm again this step.
    """
    from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

    options = {
        "device_mesh": mesh_context.device_mesh,
        "moe_mesh": mesh_context.moe_mesh,
        "ep_axis_name": "ep" if mesh_context.moe_mesh is not None else None,
        "grad_norm_backend": "triton",
    }
    norm = scale_grads_and_clip_grad_norm(float("inf"), [model], dp_group_size=2, **options)
    return norm, options


def optimizer_update_receipt(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> dict[str, Any]:
    """Validate every local Adam destination after one real optimizer update.

    This is a complete finite-state/step/dtype receipt, not an assertion that
    different-topology parameters or Adam moments are elementwise identical.
    Subsequent full-vocabulary forwards measure accumulated update drift.
    """
    from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

    observed = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not parameter.requires_grad or state is None or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise AssertionError(f"every trainable parameter requires complete Adam state: {name}")
        tensors = [parameter, state["exp_avg"], state["exp_avg_sq"]]
        if any(value.dtype != torch.float32 for value in tensors):
            raise AssertionError(f"Adam master parameters and moments must remain FP32: {name}")
        if any(
            not bool(torch.isfinite(value.to_local() if isinstance(value, DTensor) else value).all())
            for value in tensors
        ):
            raise AssertionError(f"updated parameter/Adam state is nonfinite: {name}")
        if int(state["step"]) != step:
            raise AssertionError(f"Adam update count differs: {name}")
        observed[canonical_parameter_fqn(name)] = {"shape": list(parameter.shape), "step": step}
    return {"all_local_trainable_parameters": True, "parameter_count": len(observed), "parameters": observed}


def pack_valid_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Validate one physical pack and return true document positions [1,T].

    Args:
        batch: CPU input_ids/labels/position_ids [1,T], and positive document
            seq_lens/seq_lens_padded [1,N]. Padded lengths sum to T. Labels
            already predict the next token; this function never shifts them.

    Returns:
        Boolean [1,T] derived only from lengths, so a real EOS/pad token stays
        valid. Padding targets must be -100 and real positions reset per doc.
    """
    ids, labels, positions = (batch[key] for key in ("input_ids", "labels", "position_ids"))
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape != labels.shape or ids.shape != positions.shape:
        raise ValueError("packed token/target/position tensors must share [1,T]")
    lengths, padded = batch["seq_lens"].reshape(-1), batch["seq_lens_padded"].reshape(-1)
    if lengths.numel() != padded.numel() or bool((lengths <= 0).any()) or bool((padded < lengths).any()):
        raise ValueError("document lengths must be positive and bounded by their padded lengths")
    if int(padded.sum()) != ids.shape[1]:
        raise ValueError("padded document lengths must cover exactly the physical pack")
    valid = torch.zeros_like(ids, dtype=torch.bool)
    start = 0
    for length, extent in zip(lengths.tolist(), padded.tolist()):
        valid[0, start : start + length] = True
        if not torch.equal(positions[0, start : start + length], torch.arange(length)):
            raise ValueError("real position_ids must reset to zero at each document")
        start += extent
    if bool((labels[~valid] != -100).any()):
        raise ValueError("physical padding must not carry supervised targets")
    return valid


def dataset_batch(dataset: Any, index: int) -> dict[str, torch.Tensor]:
    """Copy one frozen dataset row to independent CPU tensors with [1,*] axes."""
    row = dataset[index]
    result = {}
    for key in ("input_ids", "labels", "position_ids", "seq_lens", "seq_lens_padded"):
        value = torch.as_tensor(row[key]).detach().cpu().clone().reshape(1, -1)
        result[key] = value.to(torch.int32 if key.startswith("seq_lens") else torch.int64)
    pack_valid_mask(result)
    return result


def boundary_batch(dataset: Any, padding_id: int) -> tuple[dict[str, torch.Tensor], list[dict[str, int]]]:
    """Crop frozen real documents into an 8192-slot window-boundary pack.

    Returns:
        CPU tensors in the same layout as dataset_batch, plus original row,
        document, token-offset and crop-length receipts. Source labels retain
        their original shift; only each cropped final target becomes -100.
    """
    documents = []
    for row_index in range(len(dataset)):
        batch = dataset_batch(dataset, row_index)
        offset = 0
        for document, (length, extent) in enumerate(
            zip(batch["seq_lens"][0].tolist(), batch["seq_lens_padded"][0].tolist())
        ):
            documents.append((row_index, document, offset, length))
            offset += extent
    lengths = [512, 513, 514, 3294, 2364, 832]
    padded = [((length + 63) // 64) * 64 for length in lengths]
    if sum(padded) != 8192:
        raise AssertionError("boundary pack physical extent changed")
    ids = torch.full((1, 8192), padding_id, dtype=torch.int64)
    labels = torch.full_like(ids, -100)
    positions = torch.zeros_like(ids)
    receipts, used, destination = [], set(), 0
    for length, extent in zip(lengths, padded):
        selected = min(
            (item for item in documents if item[3] >= length and item[:2] not in used),
            key=lambda item: item[3],
            default=None,
        )
        if selected is None:
            raise ValueError(f"frozen dataset lacks an unused document of at least {length} tokens")
        row_index, document, offset, source_length = selected
        source = dataset_batch(dataset, row_index)
        ids[0, destination : destination + length] = source["input_ids"][0, offset : offset + length]
        labels[0, destination : destination + length] = source["labels"][0, offset : offset + length]
        labels[0, destination + length - 1] = -100
        positions[0, destination : destination + length] = torch.arange(length)
        receipts.append(
            {"row": row_index, "document": document, "offset": offset, "source_length": source_length, "length": length}
        )
        used.add(selected[:2])
        destination += extent
    batch = {
        "input_ids": ids,
        "labels": labels,
        "position_ids": positions,
        "seq_lens": torch.tensor([lengths], dtype=torch.int32),
        "seq_lens_padded": torch.tensor([padded], dtype=torch.int32),
    }
    pack_valid_mask(batch)
    return batch, receipts


def full_tensor(value: torch.Tensor) -> torch.Tensor:
    """Materialize a DTensor collectively; plain global tensors pass through."""
    return value.full_tensor() if isinstance(value, DTensor) else value


def observe_compute_precision(model: torch.nn.Module, records: dict[str, Any]) -> list[Any]:
    """Observe actual projection/DPA/residual/head calls, rejecting precision drift.

    Args:
        model: Native four-layer model, already prepared for this topology.
        records: Mutable evidence mapping containing an active case name. Each
            call appends only dtype/layout metadata, never tensor payloads.

    Returns:
        Removable hooks. Bulk projections and DPA must actually run in BF16;
        protected offsets, residual outputs and the vocabulary head stay FP32.
        In particular, merely configuring an ignored world-one mp_policy fails.
    """
    handles = []
    records["calls"] = []

    def record(name: str, **values: Any) -> None:
        records["calls"].append({"case": records.get("active_case"), "module": name, **values})

    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(attention, projection)
            name = f"layer.{index}.{projection}"

            def linear_call(
                module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor, name: str = name
            ) -> None:
                record(
                    name,
                    input_dtype=str(inputs[0].dtype),
                    weight_dtype=str(module.weight.dtype),
                    output_dtype=str(output.dtype),
                    output_shape=list(output.shape),
                )
                # DPA pre-hook provides the consolidated Q/K/V rejection after
                # all three projection dtypes have first been observed.
                if name.endswith("o_proj") and (
                    module.weight.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
                ):
                    raise AssertionError("actual output projection compute must be BF16 in both CP topologies")

            handles.append(module.register_forward_hook(linear_call))
        module = attention.attn_module
        if module is None:
            raise AssertionError("controlled TE CP comparison requires actual shared DPA modules")

        def dpa_pre(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], index: int = index) -> None:
            projected = {
                item["module"]: item
                for item in records["calls"]
                if item["case"] == records.get("active_case")
                and item["module"] in {f"layer.{index}.{name}" for name in ("q_proj", "k_proj", "v_proj")}
            }
            dtypes = [str(value.dtype) for value in inputs[:3]]
            record(f"layer.{index}.dpa.input", qkv_dtypes=dtypes, sink_dtype=str(module.softmax_offset.dtype))
            if len(inputs) < 3 or any(value.dtype != torch.bfloat16 for value in inputs[:3]):
                raise AssertionError(
                    "actual TE Q/K/V must be BF16 in both CP topologies; mp_policy alone is insufficient"
                )
            if len(projected) != 3 or any(
                item[key] != "torch.bfloat16"
                for item in projected.values()
                for key in ("input_dtype", "weight_dtype", "output_dtype")
            ):
                raise AssertionError("actual Q/K/V projection inputs, weights and outputs must all be BF16")
            if module.softmax_offset.dtype != torch.float32:
                raise AssertionError("actual learned sink must remain FP32")

        def dpa_post(
            module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor, index: int = index
        ) -> None:
            record(f"layer.{index}.dpa.output", dtype=str(output.dtype), shape=list(output.shape))
            if output.dtype != torch.bfloat16:
                raise AssertionError("actual TE attention output must be BF16")

        def residual_post(
            module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor, index: int = index
        ) -> None:
            record(f"layer.{index}.residual", dtype=str(output.dtype))
            if output.dtype != torch.float32:
                raise AssertionError("controlled comparison requires actual FP32 residual outputs")

        handles.extend(
            (
                module.register_forward_pre_hook(dpa_pre),
                module.register_forward_hook(dpa_post),
                layer.register_forward_hook(residual_post),
            )
        )

    def head_call(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        record(
            "lm_head",
            input_dtype=str(inputs[0].dtype),
            weight_dtype=str(module.weight.dtype),
            output_dtype=str(output.dtype),
        )
        if (
            inputs[0].dtype != module.weight.dtype
            or output.dtype != torch.float32
            or module.weight.dtype != torch.float32
        ):
            raise AssertionError("actual vocabulary head compute must remain FP32")

    handles.append(model.lm_head.register_forward_hook(head_call))
    return handles


def prove_source_prefix(model: torch.nn.Module, checkpoint: Path, raw: dict[str, object]) -> dict[str, object]:
    """Compare every global resident FP32 prefix parameter to original BF16 bytes.

    All ranks visit native parameters in the same sorted order. Only one
    gathered native tensor and one source expert/projection are compared at a
    time. The full source index schema is checked, including omitted layers.
    """
    from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

    config = ReferenceConfig.from_checkpoint(raw)
    expected = config.expected_shapes()
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    full_shapes = ReferenceConfig.from_checkpoint(
        raw, num_hidden_layers=int(raw["num_hidden_layers"])
    ).expected_shapes()
    if set(index) != set(full_shapes):
        raise ValueError("source index differs from the full independently enumerated schema")
    observed, receipt = set(), []
    with contextlib.ExitStack() as stack:
        sources = {}
        for name, parameter in sorted(model.named_parameters()):
            native = full_tensor(parameter.detach())
            if native.dtype != torch.float32:
                raise AssertionError("training master parameters must be FP32")
            native = native.cpu()
            for source_name, actual in model.state_dict_adapter.convert_single_tensor_to_hf(
                canonical_parameter_fqn(name), native
            ):
                if source_name not in expected or source_name in observed:
                    raise AssertionError(f"unexpected/duplicate resident prefix tensor {source_name}")
                filename = index[source_name]
                if Path(filename).name != filename:
                    raise ValueError("source shard names must be local basenames")
                if filename not in sources:
                    sources[filename] = stack.enter_context(
                        safe_open(checkpoint / filename, framework="pt", device="cpu")
                    )
                source = sources[filename].get_tensor(source_name)
                if source.dtype != torch.bfloat16 or tuple(source.shape) != expected[source_name]:
                    raise AssertionError(f"unexpected source shape/dtype for {source_name}")
                if actual.dtype != torch.float32 or not torch.equal(actual, source.float()):
                    raise AssertionError(f"resident FP32 source value differs for {source_name}")
                receipt.append({"name": source_name, "sha256": tensor_sha256(source), "shape": list(source.shape)})
                observed.add(source_name)
    if observed != set(expected):
        raise AssertionError("resident prefix is incomplete")
    return {"all_source_values_exact": True, "tensor_count": len(observed), "tensors": receipt}


def tensor_stats(actual: torch.Tensor, reference: torch.Tensor | None) -> dict[str, float]:
    """Accumulate FP64 dot/error sums for one complete CPU parameter gradient."""
    observed = actual.reshape(-1)
    expected = reference.reshape(-1) if reference is not None else observed
    if observed.shape != expected.shape or not bool(torch.isfinite(observed).all() & torch.isfinite(expected).all()):
        raise ValueError("gradient shape differs or contains nonfinite values")
    totals = {"reference_ss": 0.0, "actual_ss": 0.0, "error_ss": 0.0, "dot": 0.0, "max_abs": 0.0}
    for start in range(0, observed.numel(), 1024 * 1024):
        x, y = observed[start : start + 1024 * 1024].double(), expected[start : start + 1024 * 1024].double()
        error = x - y
        totals["reference_ss"] += float(y.square().sum())
        totals["actual_ss"] += float(x.square().sum())
        totals["error_ss"] += float(error.square().sum())
        totals["dot"] += float((x * y).sum())
        totals["max_abs"] = max(totals["max_abs"], float(error.abs().max()))
    return totals


def capture_gradients(model: torch.nn.Module, output: Path, baseline: Path | None, rank: int) -> dict[str, object]:
    """Collect every global pre-clip gradient, saving CP1 or comparing CP2.

    Rank zero alone retains CPU archives. Every rank participates in exactly
    the same per-parameter DTensor full_tensor collectives. No sampled probe
    or parameter subset is used for the norm/error acceptance statistics.
    """
    from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

    names = [canonical_parameter_fqn(name) for name, _ in sorted(model.named_parameters())]
    distributed = dist.is_initialized()
    inventories = [None] * dist.get_world_size() if distributed else [names]
    if distributed:
        dist.all_gather_object(inventories, names)
    if any(inventory != names for inventory in inventories) or len(set(names)) != len(names):
        raise AssertionError("all PP1 ranks must expose identical unique global parameter FQNs")
    errors = [None]
    if rank == 0 and baseline:
        with safe_open(baseline, framework="pt", device="cpu") as source:
            missing, extra = sorted(set(source.keys()) - set(names)), sorted(set(names) - set(source.keys()))
            if missing or extra:
                errors[0] = f"complete gradient parameter sets differ: missing={missing}, extra={extra}"
    if distributed:
        dist.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise AssertionError(errors[0])
    missing = [
        name for name, parameter in model.named_parameters() if not parameter.requires_grad or parameter.grad is None
    ]
    missing_by_rank = [None] * dist.get_world_size() if distributed else [missing]
    if distributed:
        dist.all_gather_object(missing_by_rank, missing)
    if any(missing_by_rank):
        raise AssertionError(f"every prefix parameter must train: {missing_by_rank}")
    tensors, summaries = {}, {}
    totals = dict.fromkeys(("reference_ss", "actual_ss", "error_ss", "dot"), 0.0)
    with contextlib.ExitStack() as stack:
        source = (
            stack.enter_context(safe_open(baseline, framework="pt", device="cpu")) if rank == 0 and baseline else None
        )
        expected_names = set(source.keys()) if source is not None else None
        observed = set()
        for name, parameter in sorted(model.named_parameters()):
            if not parameter.requires_grad or parameter.grad is None:
                raise AssertionError(f"every released prefix parameter must train: {name}")
            value = full_tensor(parameter.grad.detach()).cpu()
            name = canonical_parameter_fqn(name)
            if rank == 0:
                if value.dtype != torch.float32 or name in observed:
                    raise AssertionError("gradients must be unique FP32 global tensors")
                expected = source.get_tensor(name) if source is not None else None
                stats = tensor_stats(value, expected)
                for key in totals:
                    totals[key] += stats[key]
                summaries[name] = {"elements": value.numel(), "max_abs_error": stats["max_abs"]}
                if baseline is None:
                    tensors[name] = value.contiguous()
                observed.add(name)
        if rank != 0:
            return {}
        if expected_names is not None and observed != expected_names:
            raise AssertionError("complete gradient parameter sets differ")
    reference_norm, actual_norm = math.sqrt(totals["reference_ss"]), math.sqrt(totals["actual_ss"])
    if reference_norm == 0 or actual_norm == 0:
        raise AssertionError("released training must produce a nonzero global gradient")
    result = {
        "all_parameters": True,
        "parameter_tensor_count": len(observed),
        "elements": sum(item["elements"] for item in summaries.values()),
        "reference_global_norm": reference_norm,
        "actual_global_norm": actual_norm,
        "relative_l2": math.sqrt(totals["error_ss"]) / reference_norm,
        "global_norm_relative_error": abs(actual_norm - reference_norm) / reference_norm,
        "cosine": totals["dot"] / (reference_norm * actual_norm),
        "per_parameter": summaries,
    }
    if baseline is None:
        save_file(tensors, output)
    return result


def archive_receipt(path: Path) -> dict[str, object]:
    """Record exact durable artifact length/hash without copying its payload."""
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_archive(directory: Path, receipt: dict[str, object]) -> Path:
    """Verify a baseline artifact before reading its tensors."""
    path = directory / Path(str(receipt["path"])).name
    if path.stat().st_size != receipt["bytes"] or sha256_file(path) != receipt["sha256"]:
        raise ValueError(f"baseline archive changed: {path}")
    return path


def compare_or_save_logits(
    logits: torch.Tensor, valid: torch.Tensor, output: Path, baseline: Path | None
) -> dict[str, object]:
    """Save all physical logits, or compare all real [position,vocab] entries."""
    if logits.ndim != 3 or logits.shape[:2] != valid.shape or logits.dtype != torch.float32:
        raise AssertionError("gathered logits must be FP32 [1,Tphysical,V]")
    if not bool(torch.isfinite(logits).all()):
        raise AssertionError("every physical logit must remain finite")
    if baseline is None:
        save_file({"logits": logits.detach().cpu().contiguous(), "valid_queries": valid.cpu()}, output)
        return {
            "valid_positions": int(valid.sum()),
            "physical_positions": valid.numel(),
            "vocab_size": logits.shape[-1],
        }
    accumulator = LogitAccumulator(logits.shape[-1])
    with safe_open(baseline, framework="pt", device="cpu") as source:
        if set(source.keys()) != {"logits", "valid_queries"} or not torch.equal(
            source.get_tensor("valid_queries"), valid.cpu()
        ):
            raise AssertionError("CP1/CP2 document-validity masks differ")
        expected = source.get_slice("logits")
        if expected.get_shape() != list(logits.shape):
            raise AssertionError("CP1/CP2 full-vocabulary logit shapes differ")
        for start in range(0, logits.shape[1], 32):
            end = min(start + 32, logits.shape[1])
            keep = valid[0, start:end]
            if bool(keep.any()):
                accumulator.update(
                    expected[0, start:end][keep.cpu()].to(logits.device),
                    logits[0, start:end][keep],
                    positions=[(0, i) for i in range(start, end) if bool(valid[0, i])],
                )
    return accumulator.result()


def main() -> None:
    """Run one sequential CP topology, keeping fixed data, EP, precision and gates."""
    from datasets import load_from_disk
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from nemo_automodel import NeMoAutoModelForCausalLM
    from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
    from nemo_automodel.components.distributed.config import MoEParallelizerConfig
    from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
    from nemo_automodel.components.loss.loss import MaskedCrossEntropyConfig
    from nemo_automodel.components.loss.utils import calculate_loss
    from nemo_automodel.components.optim.optimizer import AdamWConfig
    from nemo_automodel.components.training.utils import clip_grad_norm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cp-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--ep-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    topology = comparison_topology(args.cp_size, args.ep_size)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 2 or bool(args.baseline) != (args.cp_size == 2):
        raise ValueError("use exactly two ranks for both topologies; only CP2 requires the completed CP1 baseline")
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    if output.is_relative_to(checkpoint) or (output.exists() and any(output.iterdir())):
        raise ValueError("use a fresh task-owned output directory outside the checkpoint")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
        raise ValueError("TF32 override conflicts with the declared precision policy")
    repo = Path(__file__).resolve().parents[4]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "scope": f"source first4, packed8192, EP{args.ep_size} fixed, PP1, CP1/DP2 identical-pack repeats vs CP2/DP1, 3 AdamW updates and exact window-boundary pack",
        "official_hf_reference": False,
        "cp_size": args.cp_size,
        "dp_size": world // args.cp_size,
        "world_size": world,
        "logical_pack_replicas": world // args.cp_size,
        "objective_control": "CP1 repeats the same logical pack on both DP ranks; normalize duplicate targets then FSDP averages identical gradients",
        "ep_size": args.ep_size,
        "pp_size": 1,
        "experts_backend": "torch_mm" if args.ep_size == 2 else "torch",
        "dispatcher_backend": "deepep" if args.ep_size == 2 else "torch",
        "gradient_normalization": "shared expert normalization exactly once before capture; direct clipping afterward",
        "optimizer_equality_scope": "all parameters have finite FP32 masters/moments and exact step counts; no elementwise cross-topology optimizer-state equality gate",
        "optimizer": OPTIMIZER,
        "gradient_clipping": 1.0,
        "tolerances_fixed_before_measurement": TOLERANCES,
        "master_dtype": "torch.float32",
        "bulk_compute_dtype": "torch.bfloat16",
        "residual_dtype": "fp32",
        "gradient_reduction_dtype": "torch.float32",
        "labels_already_shifted_once": True,
        "tf32": False,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "hardware": {
            "name": torch.cuda.get_device_properties(device).name,
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "cuda": torch.version.cuda,
        },
        "script_sha256": sha256_file(Path(__file__)),
        "helper_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("sdpa_reference.py", "run_four_layer_logits.py")
        },
        "flint_implementation_sha256": {
            str(path.relative_to(repo)): sha256_file(path)
            for path in sorted((repo / "nemo_automodel/components/models/flint").glob("*.py"))
        },
        "shared_implementation_sha256": {
            str(path.relative_to(repo)): sha256_file(path)
            for path in sorted((repo / "nemo_automodel").rglob("*.py"))
            if not path.is_relative_to(repo / "nemo_automodel/components/models/flint")
        },
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_index_sha256": sha256_file(checkpoint / "model.safetensors.index.json"),
        "repository_commit_at_measurement": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "transformer-engine", "datasets")
        },
        "steps": [],
    }

    def flush() -> None:
        if rank == 0:
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )

    started = time.perf_counter()
    handles = []
    flush()
    try:
        raw = json.loads((checkpoint / "config.json").read_text())
        manifest["source_num_hidden_layers"] = int(raw["num_hidden_layers"])
        dataset = load_from_disk(str(args.dataset))
        if len(dataset) < 3:
            raise ValueError("the frozen real dataset must contain three original packed rows")
        original = [dataset_batch(dataset, index) for index in range(3)]
        if any(batch["input_ids"].shape != (1, 8192) for batch in original):
            raise ValueError("the controlled update rows must have physical8192 tokens")
        boundary, crops = boundary_batch(dataset, int(raw["pad_token_id"]))
        manifest["boundary_crop_receipts"] = crops
        manifest["dataset_fingerprint"] = dataset._fingerprint
        manifest["dataset_files_sha256"] = {
            str(path.relative_to(args.dataset)): sha256_file(path)
            for path in sorted(args.dataset.rglob("*"))
            if path.is_file()
        }
        manifest["inputs_tensor_sha256"] = {
            name: {key: tensor_sha256(value) for key, value in batch.items()}
            for name, batch in [
                ("boundary", boundary),
                *[(f"step{index}", batch) for index, batch in enumerate(original)],
            ]
        }
        baseline = json.loads((args.baseline / "manifest.json").read_text()) if args.baseline else None
        if baseline is not None:
            if baseline["status"] != "passed" or baseline["cp_size"] != 1:
                raise ValueError("CP2 requires a successful CP1 baseline")
            for key in (
                "ep_size",
                "pp_size",
                "experts_backend",
                "dispatcher_backend",
                "optimizer",
                "gradient_clipping",
                "tolerances_fixed_before_measurement",
                "master_dtype",
                "bulk_compute_dtype",
                "residual_dtype",
                "gradient_reduction_dtype",
                "script_sha256",
                "helper_sha256",
                "checkpoint_config_sha256",
                "checkpoint_index_sha256",
                "flint_implementation_sha256",
                "shared_implementation_sha256",
                "packages",
                "tf32",
                "bf16_reduced_precision_reduction",
                "hardware",
                "dataset_fingerprint",
                "dataset_files_sha256",
                "inputs_tensor_sha256",
            ):
                if json.loads(json.dumps(manifest[key])) != baseline[key]:
                    raise ValueError(f"CP1/CP2 comparison conditions differ: {key}")
        setup = DistributedSetup.build(
            strategy=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=None, cast_forward_inputs=False
                ),
                enable_fsdp2_prefetch=False,
            ),
            parallelism_sizes=ParallelismSizes(
                dp_size=topology["dp_size"], cp_size=args.cp_size, ep_size=args.ep_size, tp_size=1, pp_size=1
            ),
            moe_parallel_config=MoEParallelizerConfig(reshard_after_forward=True) if args.ep_size == 2 else None,
            world_size=world,
        )
        backend = {
            "attn": "te",
            "linear": "torch",
            "rms_norm": "torch_fp32",
            "rope": "torch",
            "rope_fusion": False,
            "experts": manifest["experts_backend"],
            "dispatcher": manifest["dispatcher_backend"],
            "gate_precision": None,
            "enable_fsdp_optimizations": False,
        }
        manifest["stage"] = "public_from_pretrained"
        flush()
        model = NeMoAutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            config={
                "num_hidden_layers": 4,
                "schedule_num_hidden_layers": int(raw["num_hidden_layers"]),
                "router_aux_loss_coef": 0.0,
                "residual_dtype": "fp32",
                "use_cache": False,
            },
            dtype=torch.float32,
            backend=backend,
            attn_implementation="sdpa",
            use_liger_kernel=False,
            use_sdpa_patching=False,
            distributed_setup=setup,
        ).train()
        receipt = prove_source_prefix(model, checkpoint, raw)
        manifest["source_prefix"] = {key: value for key, value in receipt.items() if key != "tensors"}
        if rank == 0:
            (output / "source_prefix_tensor_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        optimizer = AdamWConfig(**OPTIMIZER).build(model)[0]
        loss_fn = MaskedCrossEntropyConfig().build()
        observations: dict[str, Any] = {}
        handles = observe_compute_precision(model, observations)
        manifest["actual_compute_precision"] = observations
        passed = True
        for case_index, (case_name, cpu_batch) in enumerate(
            [("boundary", boundary), *[(f"step{index}", batch) for index, batch in enumerate(original)]]
        ):
            train = case_name != "boundary"
            manifest["stage"] = case_name
            observations["active_case"] = case_name
            flush()
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            batch["qkv_format"] = "thd"
            valid = pack_valid_mask(cpu_batch).to(device)
            sharder = ContextParallelSharder(
                model, setup.mesh_context.device_mesh, batch, padding_token_id=int(raw["pad_token_id"])
            )
            context, local = sharder.shard(batch)
            local_labels = local.pop("labels")
            for key in ("seq_lens", "seq_lens_padded", "loss_mask"):
                local.pop(key, None)
            local_valid = ~local["padding_mask"].bool().reshape(1, -1)
            gathered_valid = sharder.gather_token_tensor(local_valid, seq_dim=1, trim=True)
            if not torch.equal(gathered_valid, valid):
                raise AssertionError("sharded padding ownership differs from the independent document-length mask")
            num_labels = torch.tensor(int((local_labels != -100).sum()), device=device, dtype=torch.int64)
            dist.all_reduce(num_labels)
            unique_labels = int((cpu_batch["labels"] != -100).sum())
            if int(num_labels) != expected_global_targets(unique_labels, args.cp_size):
                raise AssertionError("sharding/repeated-DP control has the wrong number of already-shifted targets")
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train), context():
                logits = model(**local).logits
                loss = calculate_loss(loss_fn, logits=logits, labels=local_labels, num_label_tokens=num_labels)
                if train:
                    # FSDP averages the CP ranks; global-token-normalized local
                    # sums therefore need this same compensation as the recipe.
                    (loss * world).backward()
            reported_loss = loss.detach().clone()
            dist.all_reduce(reported_loss)
            if not bool(torch.isfinite(reported_loss)):
                raise AssertionError("global cross entropy must remain finite")
            with torch.no_grad():
                global_logits = sharder.gather_token_tensor(logits.detach(), seq_dim=1, trim=True)
            reference_case = baseline["steps"][case_index] if baseline else None
            archive = output / f"{case_name}_logits.safetensors"
            source_archive = (
                verify_archive(args.baseline, reference_case["logits_archive"]) if rank == 0 and baseline else None
            )
            result: dict[str, Any] = {
                "case": case_name,
                "loss": float(reported_loss),
                "unique_logical_supervised_tokens": unique_labels,
                "supervised_tokens_across_ranks_including_repeats": int(num_labels),
                "logical_pack_replicas": world // args.cp_size,
            }
            if rank == 0:
                result["logits"] = compare_or_save_logits(global_logits, valid, archive, source_archive)
                if baseline:
                    result["loss_abs_error"] = abs(result["loss"] - reference_case["loss"])
                    passed &= result["logits"]["mean_kl"] < TOLERANCES["mean_kl_lt"]
                    passed &= result["loss_abs_error"] <= TOLERANCES["loss_abs_le"]
                else:
                    result["logits_archive"] = archive_receipt(archive)
            del global_logits, logits, loss
            if train:
                # DeepEP sums expert-token contributions independently of dense
                # FSDP averaging. Normalize exactly once before observing them.
                norm, norm_options = normalize_gradients(model, setup.mesh_context)
                grad_archive = output / f"{case_name}_gradients.safetensors"
                source_grad = (
                    verify_archive(args.baseline, reference_case["gradients_archive"])
                    if rank == 0 and baseline
                    else None
                )
                gradients = capture_gradients(model, grad_archive, source_grad, rank)
                clipped_norm = clip_grad_norm(1.0, [model], **norm_options)
                norm = full_tensor(norm) if isinstance(norm, torch.Tensor) else norm
                clipped_norm = full_tensor(clipped_norm) if isinstance(clipped_norm, torch.Tensor) else clipped_norm
                if not math.isclose(float(norm), float(clipped_norm), rel_tol=5e-5, abs_tol=1e-7):
                    raise AssertionError("normalization and direct clipping disagree on the pre-clip norm")
                if rank == 0:
                    result["gradients"] = gradients
                    result["shared_global_norm"] = float(norm)
                    if not math.isclose(float(norm), gradients["actual_global_norm"], rel_tol=5e-5, abs_tol=1e-7):
                        raise AssertionError("shared global norm differs from independent all-gradient FP64 norm")
                    if baseline:
                        passed &= gradients["relative_l2"] <= TOLERANCES["all_gradient_relative_l2_le"]
                        passed &= (
                            gradients["global_norm_relative_error"]
                            <= TOLERANCES["global_gradient_norm_relative_error_le"]
                        )
                        passed &= gradients["cosine"] >= TOLERANCES["all_gradient_cosine_ge"]
                    else:
                        result["gradients_archive"] = archive_receipt(grad_archive)
                optimizer.step()
                local_update = optimizer_update_receipt(model, optimizer, case_index)
                updates = [None] * world
                dist.all_gather_object(updates, local_update)
                if rank == 0:
                    result["optimizer_update_by_rank"] = updates
            if rank == 0:
                manifest["steps"].append(result)
                manifest["disk_free_bytes"] = shutil.disk_usage(output).free
                flush()
            dist.barrier()
        verdict = torch.tensor(int(passed), device=device)
        dist.broadcast(verdict, src=0)
        peaks = [None] * world
        dist.all_gather_object(peaks, torch.cuda.max_memory_allocated(device))
        manifest.update(
            status="passed" if bool(verdict) else "failed",
            capture_completed=True,
            comparison_passed=bool(verdict) if baseline is not None else None,
            elapsed_seconds=time.perf_counter() - started,
            process_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            process_cuda_peak_allocated_bytes_by_rank=peaks,
            optimizer_updates=3,
        )
        flush()
    except BaseException as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()
        )
        flush()
        raise
    finally:
        for handle in handles:
            handle.remove()
        dist.destroy_process_group()
    if manifest.get("comparison_passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
