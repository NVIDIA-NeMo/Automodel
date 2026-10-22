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

"""Scheduled four-GPU PP2/CP2/EP2/VPP4 control; ordinary functional CI stays two-GPU."""

from __future__ import annotations

import argparse
import faulthandler
import gc
import hashlib
import json
import math
import os
import socket
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
from nemo_automodel.components.distributed.config import MoEParallelizerConfig
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.components.training.utils import (
    clip_grad_norm,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from tests.functional_tests.models.flint.sdpa_reference import LogitAccumulator
from tests.functional_tests.models.flint.test_attention import attention_backend, te_config
from tests.functional_tests.models.flint.test_parallel_training import _runtime_receipt


def _guard_parameter_inventory(names: list[str], directory: Path, *, compare: bool) -> dict:
    """Require exactly the baseline's global FQNs before any tensor comparison.

    Args:
        names: Sorted unique global native parameter names gathered across PP ranks.
        directory: Shared archive directory containing baseline tensors.
        compare: Read and validate the baseline inventory instead of writing it.

    Returns:
        Small metadata receipt broadcast to all ranks. Missing/extra names are
        rejected collectively before entering per-parameter DTensor operations.
    """
    path = directory / "parameter_inventory.json"
    payload = [None]
    if dist.get_rank() == 0:
        try:
            if compare:
                recorded = json.loads(path.read_text())
                expected = recorded["parameter_fqns"]
                assert isinstance(expected, list) and all(isinstance(name, str) for name in expected)
                assert len(set(expected)) == len(expected), "Baseline parameter inventory contains duplicate FQNs"
                missing, extra = sorted(set(expected) - set(names)), sorted(set(names) - set(expected))
                assert not missing and not extra, f"Parameter FQN sets differ: missing={missing}, extra={extra}"
            else:
                path.write_text(json.dumps({"schema_version": 1, "parameter_fqns": names}, indent=2) + "\n")
            payload[0] = {
                "status": "passed",
                "parameter_count": len(names),
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        except Exception as error:
            payload[0] = {"status": "failed", "error": str(error)}
    dist.broadcast_object_list(payload, src=0)
    assert payload[0]["status"] == "passed", payload[0].get("error", payload[0])
    return payload[0]


def _archive_parameters(
    parts: list[torch.nn.Module], directory: Path, *, kind: str, compare: bool, global_gradient_gate: bool = False
) -> dict:
    """Archive or compare every global native parameter with bounded live storage.

    Args:
        parts: Local PP stages. Parameters use native [out,in] or grouped
            [experts,hidden,2*intermediate]/[experts,intermediate,hidden] layouts.
        directory: Task-owned baseline archive directory on shared storage.
        kind: ``initial``, pre-clip ``gradients``, or post-SGD ``updated``.
        compare: Compare candidate tensors to baseline instead of writing them.
        global_gradient_gate: Use the released control's fixed all-gradient
            relative-L2/cosine criteria, retaining per-tensor errors as diagnostics.

    Returns:
        Small all-parameter statistics. Only names/shapes and scalar summaries
        use world object collectives. DTensors gather on their own mesh, one
        parameter at a time; only one owner retains one complete CPU parameter.
        The safetensors writer may stage an additional parameter-sized buffer.
        Baseline FP32 comparison slices cover at most 16 MiB except one row;
        their FP64 statistic temporaries are proportional to that bounded slice.
    """
    local = {}
    for part in parts:
        for name, parameter in part.named_parameters():
            name = canonical_parameter_fqn(name)
            assert name not in local, name
            local[name] = parameter
    # Resolve missing gradients collectively before any DTensor mesh gather:
    # an owner-only failure here would otherwise strand a different PP stage.
    missing = [name for name, value in local.items() if kind == "gradients" and value.grad is None]
    missing_by_rank = [None] * dist.get_world_size()
    dist.all_gather_object(missing_by_rank, missing)
    assert not any(missing_by_rank), {"missing_gradients_by_rank": missing_by_rank}
    inventories = [None] * dist.get_world_size()
    dist.all_gather_object(inventories, {name: list(value.shape) for name, value in local.items()})
    names = sorted(set().union(*(item.keys() for item in inventories)))
    directory.mkdir(parents=True, exist_ok=True)
    inventory = _guard_parameter_inventory(names, directory, compare=compare)
    summaries = {}
    for name in names:
        owner = next(rank for rank, inventory in enumerate(inventories) if name in inventory)
        parameter = local.get(name)
        value = None
        if parameter is not None:
            value = parameter.grad if kind == "gradients" else parameter
            assert value is not None, f"Missing {kind}: {name}"
            value = value.detach()
            value = value.full_tensor() if isinstance(value, DTensor) else value
        payload = [None]
        if dist.get_rank() == owner:
            try:
                assert value is not None and value.dtype == torch.float32, (
                    name,
                    None if value is None else value.dtype,
                )
                actual = value.cpu().contiguous()
                path = directory / f"{hashlib.sha256(name.encode()).hexdigest()}.safetensors"
                stats = {
                    "shape": list(actual.shape),
                    "numel": actual.numel(),
                    "bytes": actual.nbytes,
                    "actual_ss": 0.0,
                    "reference_ss": 0.0,
                    "error_ss": 0.0,
                    "dot": 0.0,
                    "max_abs": 0.0,
                    "nonzero": 0,
                }
                # Safetensors slices retain arbitrary grouped/linear layout.
                # Chunk the leading axis; at least one complete row is needed.
                row_elements = math.prod(actual.shape[1:]) if actual.ndim > 1 else 1
                rows = max(1, (4 * 1024 * 1024) // row_elements)
                source = safe_open(path, framework="pt", device="cpu") if compare else None
                if source is not None:
                    assert source.keys() == ["value"], name
                    reference = source.get_slice("value")
                    assert reference.get_shape() == list(actual.shape), name
                else:
                    reference = None
                for start in range(0, actual.shape[0], rows):
                    x = actual[start : start + rows]
                    y = reference[start : start + rows] if reference is not None else x
                    assert torch.isfinite(x).all() and torch.isfinite(y).all(), name
                    if compare:
                        if kind in ("initial", "updated"):
                            torch.testing.assert_close(x, y, atol=0 if kind == "initial" else 3e-6, rtol=0, msg=name)
                        elif not global_gradient_gate:
                            torch.testing.assert_close(x, y, atol=3e-4, rtol=0.15, msg=name)
                    xd, yd = x.double(), y.double()
                    stats["actual_ss"] += float(xd.square().sum())
                    stats["reference_ss"] += float(yd.square().sum())
                    stats["error_ss"] += float((xd - yd).square().sum())
                    stats["dot"] += float((xd * yd).sum())
                    stats["max_abs"] = max(stats["max_abs"], float((x - y).abs().max()))
                    stats["nonzero"] += int(torch.count_nonzero(x))
                stats["relative_l2"] = math.sqrt(stats["error_ss"] / max(stats["reference_ss"], 1e-40))
                if compare and kind == "gradients" and not global_gradient_gate:
                    assert stats["relative_l2"] < 0.12, (name, stats)
                if not compare:
                    save_file({"value": actual}, path)
                if source is not None:
                    source.__exit__(None, None, None)
                payload[0] = {"status": "passed", "name": name, **stats}
                del actual
            except Exception as error:
                payload[0] = {"status": "failed", "name": name, "error": repr(error)}
        # All ranks learn the same failure, avoiding an owner-only assertion
        # stranding another stage in a subsequent collective.
        dist.broadcast_object_list(payload, src=owner)
        assert payload[0]["status"] == "passed", payload[0]
        summaries[name] = payload[0]
        del value
    totals = {
        key: sum(item[key] for item in summaries.values())
        for key in ("numel", "bytes", "actual_ss", "reference_ss", "error_ss", "dot")
    }
    return {
        "tensor_count": len(names),
        "parameter_inventory": inventory,
        "tensors": summaries,
        **totals,
        "largest_parameter_bytes": max(item["bytes"] for item in summaries.values()),
        "actual_norm": math.sqrt(totals["actual_ss"]),
        "relative_l2": math.sqrt(totals["error_ss"] / max(totals["reference_ss"], 1e-40)),
        "cosine": totals["dot"] / max(math.sqrt(totals["actual_ss"] * totals["reference_ss"]), 1e-40),
        "retention_policy": "one gathered CUDA parameter and one owner CPU parameter; writer may stage another parameter-sized buffer; FP32 reference slices <=16MiB except one row, bounded FP64 statistic temporaries",
    }


def _observe_parts(parts: list[torch.nn.Module], records: list[dict]) -> list:
    """Observe actual protected F32 units, BF16 DPA tensors and F32 residuals."""
    handles = []
    for part in parts:
        owned = (
            list(part.model.layers.keys())
            if isinstance(part.model.layers, torch.nn.ModuleDict)
            else list(range(len(part.model.layers)))
        )

        def stage_pre(module: torch.nn.Module, inputs: tuple, kwargs: dict, owned: list = owned) -> None:
            value = inputs[0] if inputs else kwargs.get("input_ids", kwargs.get("inputs_embeds"))
            info = {
                "module": "stage_input",
                "rank": dist.get_rank(),
                "layers": owned,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            records.append(info)
            print(json.dumps(info), flush=True)

        def stage_post(module: torch.nn.Module, inputs: tuple, output: object, owned: list = owned) -> None:
            value = output.logits if hasattr(output, "logits") else output
            info = {
                "module": "stage_output",
                "rank": dist.get_rank(),
                "layers": owned,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            records.append(info)
            print(json.dumps(info), flush=True)

        handles.extend(
            (part.register_forward_pre_hook(stage_pre, with_kwargs=True), part.register_forward_hook(stage_post))
        )
        if part.lm_head is not None:

            def head(module: torch.nn.Module, inputs: tuple, output: torch.Tensor) -> None:
                records.append(
                    {
                        "module": "head",
                        "input": str(inputs[0].dtype),
                        "weight": str(module.weight.dtype),
                        "output": str(output.dtype),
                    }
                )
                assert inputs[0].dtype == module.weight.dtype == output.dtype == torch.float32

            handles.append(part.lm_head.register_forward_hook(head))
        layers = part.model.layers.values() if isinstance(part.model.layers, torch.nn.ModuleDict) else part.model.layers
        for layer in layers:

            def attention(module: torch.nn.Module, inputs: tuple) -> None:
                records.append(
                    {
                        "module": "attention",
                        "qkv": [str(t.dtype) for t in inputs[:3]],
                        "sink": str(module.softmax_offset.dtype),
                    }
                )
                assert all(t.dtype == torch.bfloat16 for t in inputs[:3])
                assert module.softmax_offset.dtype == torch.float32

            def residual(module: torch.nn.Module, inputs: tuple, output: torch.Tensor) -> None:
                records.append({"module": "residual", "dtype": str(output.dtype)})
                assert output.dtype == torch.float32

            handles.extend(
                (
                    layer.self_attn.attn_module.register_forward_pre_hook(attention),
                    layer.register_forward_hook(residual),
                )
            )
    return handles


def _released_batch(checkpoint: Path) -> dict:
    """Freeze short literal text into four physical packs without changing targets.

    Returns:
        JSON-compatible token/target/position tensors [4,128], document lengths
        [4,2], tokenizer inputs, and explicit pad/ignored/vocabulary metadata.
        Labels shift once within each 5/7-token document; pack4 is ignored.
    """
    from transformers import AutoTokenizer

    raw = json.loads((checkpoint / "config.json").read_text())
    assert raw["num_hidden_layers"] == 49, "This released-prefix gate preserves the audited49-layer source schedule"
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False, fix_mistral_regex=False
    )
    pad, ignored = int(raw["pad_token_id"]), int(raw["vocab_size"]) - 1
    texts = [
        "Explain why a model learns from examples and how careful validation reveals errors.",
        "Write a short explanation of context parallel training with document boundaries.",
        "Compare two mathematical expressions and describe their numerical precision.",
        "List the steps needed to load a checkpoint and preserve every parameter.",
        "Describe how an attention window changes which earlier tokens are visible.",
        "Explain the purpose of a gradient and a single optimizer update in training.",
    ]
    ids = torch.full((4, 128), pad, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    positions = torch.cat((torch.arange(64), torch.arange(64)))[None].expand(4, -1).clone()
    token_receipts = []
    for pack in range(4):
        for doc, (start, length) in enumerate(((0, 5), (64, 7))):
            if pack < 3:
                tokens = tokenizer.encode(texts[pack * 2 + doc], add_special_tokens=False)[:length]
                assert len(tokens) == length and pad not in tokens and ignored not in tokens
                values = torch.tensor(tokens)
                token_receipts.append({"text": texts[pack * 2 + doc], "input_ids": tokens})
                labels[pack, start : start + length - 1] = values[1:]
            else:
                values = torch.full((length,), ignored)
            ids[pack, start : start + length] = values
    return {
        "input_ids": ids.tolist(),
        "labels": labels.tolist(),
        "position_ids": positions.tolist(),
        "seq_lens": [[5, 7]] * 4,
        "seq_lens_padded": [[64, 64]] * 4,
        "pad_id": pad,
        "ignored_id": ignored,
        "vocab_size": int(raw["vocab_size"]),
        "source_layers": 49,
        "active_layers": 4,
        "fix_mistral_regex": False,
        "token_receipts": token_receipts,
        "checkpoint_metadata_sha256": {
            name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")
        },
    }


def _config() -> FlintConfig:
    original = te_config(dtype=torch.float32).to_dict()
    return FlintConfig(
        **{
            **original,
            "num_hidden_layers": 4,
            "schedule_num_hidden_layers": 4,
            "layer_types": ["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
            "no_rope_layer_indices": [0, 2],
        }
    )


def _packed(device: torch.device, *, flatten: bool) -> dict[str, torch.Tensor | str]:
    """Return four physical128-token packs, two documents each, one fully ignored pack.

    Args:
        device: Runtime CUDA device.
        flatten: Flatten only baseline's batch dimension; physical token order stays fixed.

    Returns:
        IDs, targets, positions [4,128] or [1,512], lengths [4,2] or [1,8].
        Supervised labels shift within each real document once, never across boundaries.
        Ignored-document ID29 and physical-padding ID30 never occur in supervised packs.
    """
    ids = torch.full((4, 128), 30, device=device, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    positions = (
        torch.cat((torch.arange(64, device=device), torch.arange(64, device=device)))[None].expand(4, -1).clone()
    )
    for pack in range(4):
        for start, length in [(0, 5), (64, 7)]:
            values = (
                ((torch.arange(length, device=device) + pack * 3 + 2) % 28)
                if pack < 3
                else torch.full((length,), 29, device=device)
            )
            ids[pack, start : start + length] = values
            if pack < 3:
                labels[pack, start : start + length - 1] = values[1:]
    real = torch.tensor([[5, 7]] * 4, device=device, dtype=torch.int32)
    padded = torch.full_like(real, 64)
    if flatten:
        ids, labels, positions = (value.reshape(1, -1) for value in (ids, labels, positions))
        real, padded = real.reshape(1, -1), padded.reshape(1, -1)
    return {
        "input_ids": ids,
        "labels": labels,
        "position_ids": positions,
        "seq_lens": real,
        "seq_lens_padded": padded,
        "qkv_format": "thd",
    }


def _worker(
    rank: int, port: int, source: str, evidence: str, released: bool = False, dispatcher: str = "torch"
) -> None:
    """Hold source, tokens, TE kernel and precision fixed while composing parallel axes."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="4", LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    faulthandler.enable()
    faulthandler.dump_traceback_later(120)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=4, timeout=timedelta(minutes=12))
    try:
        provenance = _runtime_receipt()
        provenance["composition_harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        runs = []
        archive_root = Path(evidence).parent / "parameter_archives"
        frozen = json.loads((Path(evidence).parent / "input_batch.json").read_text()) if released else None
        pad_id = frozen["pad_id"] if frozen else 30
        ignored_id = frozen["ignored_id"] if frozen else 29
        vocabulary = frozen["vocab_size"] if frozen else 31
        valid = torch.zeros(4, 128, dtype=torch.bool)
        valid[:, :5] = True
        valid[:, 64:71] = True
        for composed in [False, True]:
            print(f"rank{rank}: start composed={composed}", flush=True)
            baseline_ep = 2 if dispatcher == "deepep" else 1
            pp, cp, ep, dp = (2, 2, 2, 1) if composed else (1, 1, baseline_ep, 4)
            captures = []
            losses = []

            def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                """Normalize local [microbatch,Tlocal,V] logits by the same30 global targets."""
                logits = prediction.logits if hasattr(prediction, "logits") else prediction
                captures.append(logits.detach())
                value = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum") / 30
                losses.append(value.detach())
                return value * cp

            pipeline = (
                PipelineConfig(
                    pp_schedule="interleaved1f1b",
                    pp_microbatch_size=1,
                    pp_batch_size=4,
                    layers_per_stage=1,
                    scale_grads_in_schedule=False,
                    patch_inner_model=False,
                    patch_causal_lm_model=False,
                    loss_fn=loss_fn,
                )
                if composed
                else None
            )
            setup = DistributedSetup.build(
                strategy=FSDP2Config(
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        output_dtype=torch.bfloat16,
                        cast_forward_inputs=False,
                    ),
                    enable_fsdp2_prefetch=False,
                    defer_fsdp_grad_sync=False,
                ),
                parallelism_sizes=ParallelismSizes(dp_size=dp, pp_size=pp, cp_size=cp, ep_size=ep),
                pipeline_config=pipeline,
                moe_parallel_config=MoEParallelizerConfig(reshard_after_forward=True) if ep > 1 else None,
                world_size=4,
            )
            model = NeMoAutoModelForCausalLM.from_pretrained(
                source,
                **(
                    {
                        "config": {
                            "num_hidden_layers": 4,
                            "schedule_num_hidden_layers": 49,
                            "router_aux_loss_coef": 0.0,
                            "residual_dtype": "fp32",
                        }
                    }
                    if released
                    else {}
                ),
                dtype=torch.float32,
                backend=replace(attention_backend("te"), experts="torch_mm", dispatcher="deepep")
                if dispatcher == "deepep"
                else attention_backend("te"),
                distributed_setup=setup,
                attn_implementation="eager",
                use_liger_kernel=False,
                use_sdpa_patching=False,
                local_files_only=True,
            )
            parts = model.parts if isinstance(model, AutoPipeline) else [model]
            observations = []
            handles = _observe_parts(parts, observations)
            initial = _archive_parameters(parts, archive_root / "initial", kind="initial", compare=composed)
            print(f"rank{rank}: loaded and initial values checked composed={composed}", flush=True)
            source_values = None
            if released and not composed:
                from tests.functional_tests.models.flint.run_packed_cp_logits import prove_source_prefix

                raw = json.loads((Path(source) / "config.json").read_text())
                source_values = prove_source_prefix(parts[0], Path(source), raw)
            if frozen:
                batch = {
                    name: torch.tensor(
                        frozen[name], device=device, dtype=torch.int32 if name.startswith("seq_lens") else torch.long
                    )
                    for name in ("input_ids", "labels", "position_ids", "seq_lens", "seq_lens_padded")
                }
                if not composed:
                    batch = {name: value.reshape(1, -1) for name, value in batch.items()}
                batch["qkv_format"] = "thd"
            else:
                batch = _packed(device, flatten=not composed)
            sharder = ContextParallelSharder(
                parts[0],
                setup.mesh_context.device_mesh,
                batch,
                num_chunks=4 if composed else 1,
                padding_token_id=pad_id,
            )
            _, local = sharder.shard(batch)
            labels = local.pop("labels")
            for name in ["seq_lens", "seq_lens_padded", "loss_mask"]:
                local.pop(name, None)
            prepare_for_grad_accumulation(parts)
            prepare_for_final_backward(parts)
            if composed:
                assert isinstance(model, AutoPipeline)
                model.update_seq_len(128)
                for stage in model.info.stages:
                    inputs_meta, outputs_meta = stage.submod.get_pipeline_stage_metas(
                        is_first=stage.is_first, microbatch_size=1, seq_len=128, dtype=torch.bfloat16
                    )
                    print(
                        json.dumps(
                            {
                                "rank": rank,
                                "module": "stage_metadata",
                                "stage": stage.stage_index,
                                "installed_user_meta": str(getattr(stage, "_user_meta", None)),
                                "inputs": [
                                    {"shape": list(item.shape), "dtype": str(item.dtype)} for item in inputs_meta
                                ],
                                "outputs": [
                                    {"shape": list(item.shape), "dtype": str(item.dtype)} for item in outputs_meta
                                ],
                            }
                        ),
                        flush=True,
                    )
                model.step(
                    local.pop("input_ids"),
                    target=labels if model.info.has_last_stage else None,
                    losses=[] if model.info.has_last_stage else None,
                    **local,
                )
            else:
                output = model(**local).logits
                loss_fn(output, labels).backward()
            print(f"rank{rank}: forward/backward complete composed={composed}", flush=True)
            total_loss = sum(losses, torch.zeros((), device=device))
            dist.all_reduce(total_loss)
            total_loss = total_loss / dp
            norm_options = dict(
                pp_enabled=bool(pp > 1),
                device_mesh=setup.mesh_context.device_mesh,
                moe_mesh=setup.mesh_context.moe_mesh,
                ep_axis_name="ep" if setup.mesh_context.moe_mesh is not None else None,
                pp_axis_name="pp" if pp > 1 else None,
                foreach=True,
                grad_norm_backend="triton",
            )
            norm = scale_grads_and_clip_grad_norm(float("inf"), parts, dp_group_size=dp * cp, **norm_options)
            if isinstance(norm, DTensor):
                norm = norm.full_tensor()
            gradients = _archive_parameters(
                parts, archive_root / "gradients", kind="gradients", compare=composed, global_gradient_gate=released
            )
            torch.testing.assert_close(
                torch.tensor(float(norm)), torch.tensor(gradients["actual_norm"]), atol=1e-6, rtol=5e-5
            )
            sink_values = [value for name, value in gradients["tensors"].items() if name.endswith("softmax_offset")]
            assert len(sink_values) == 4 and all(value["nonzero"] > 0 for value in sink_values)
            # The ignored pack's unique token and physical-pad token must never
            # receive an embedding adjoint, independently of EOS identity.
            for part in parts:
                if part.model.embed_tokens is not None:
                    grad = part.model.embed_tokens.weight.grad
                    grad = grad.full_tensor() if isinstance(grad, DTensor) else grad
                    assert torch.count_nonzero(grad[[ignored_id, pad_id]]) == 0
            global_logits = None
            if captures:
                if composed:
                    local_logits = torch.cat(captures)
                    indices = local["_thd_local_indices"]
                    global_logits = torch.zeros(4, 128, vocabulary, device=device)
                    for chunk in range(4):
                        global_logits[chunk].index_copy_(0, indices[chunk].long(), local_logits[chunk])
                    dist.all_reduce(global_logits, group=setup.mesh_context.device_mesh["cp"].get_group())
                else:
                    global_logits = captures[0].reshape(4, 128, vocabulary)
            payload = [None if global_logits is None else global_logits[valid.to(device)].cpu()]
            # All final-stage CP ranks have identical complete logits after the above collective.
            dist.broadcast_object_list(payload, src=2 if composed else 0)
            clipped_norm = clip_grad_norm(0.2, parts, **norm_options)
            if isinstance(clipped_norm, DTensor):
                clipped_norm = clipped_norm.full_tensor()
            torch.testing.assert_close(clipped_norm.double(), norm.double(), atol=1e-8, rtol=1e-8)
            parameters = [parameter for part in parts for parameter in part.parameters() if parameter.requires_grad]
            optimizer = torch.optim.SGD(parameters, lr=0.01)
            optimizer.step()
            updated = _archive_parameters(parts, archive_root / "updated", kind="updated", compare=composed)
            runs.append(
                {
                    "loss": float(total_loss),
                    "logits": payload[0],
                    "gradients": gradients,
                    "norm": float(norm),
                    "updated": updated,
                    "initial": initial,
                    "source_values": source_values,
                    "observed_dtypes": observations,
                }
            )
            for handle in handles:
                handle.remove()
            captures.clear()
            del model, parts, optimizer, parameters, sharder, gradients, updated, handles, global_logits
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
        baseline, candidate = runs
        left, right = baseline["logits"], candidate["logits"]
        max_abs = float((left - right).abs().max())
        accumulator = LogitAccumulator(vocabulary)
        accumulator.update(left, right, positions=[tuple(index) for index in valid.nonzero().tolist()])
        logit_metrics = accumulator.result()
        # The streaming metric accumulates scalar products in FP64; an F32
        # reduction across millions of logits can even report cosine > 1.
        cosine = float(logit_metrics["logit_cosine"])
        if rank == 0:
            after = _runtime_receipt()
            Path(evidence).with_suffix(".observed.json").write_text(
                json.dumps(
                    {
                        "status": "observed_before_final_acceptance",
                        "provenance": provenance,
                        "source_sha256_after": after["source_sha256"],
                        "released_checkpoint": released,
                        "dispatcher": dispatcher,
                        "precision": {"master": "float32", "compute": "bfloat16", "residual": "float32"},
                        "loss_baseline": baseline["loss"],
                        "loss_candidate": candidate["loss"],
                        "norm_baseline": baseline["norm"],
                        "norm_candidate": candidate["norm"],
                        "logit_max_abs": max_abs,
                        "logit_cosine": cosine,
                        "logit_metrics": logit_metrics,
                        "initial_weights_exact": True,
                        "source_prefix_values": baseline["source_values"],
                        "gradient_comparison": {
                            name: value for name, value in candidate["gradients"].items() if name != "tensors"
                        },
                        "all_parameter_update_comparisons_completed": True,
                        "observed_dtypes_baseline": baseline["observed_dtypes"],
                        "observed_dtypes_candidate": candidate["observed_dtypes"],
                    },
                    indent=2,
                )
                + "\n"
            )
        torch.testing.assert_close(
            torch.tensor(candidate["loss"]), torch.tensor(baseline["loss"]), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            torch.tensor(candidate["norm"]),
            torch.tensor(baseline["norm"]),
            atol=1e-7 if released else 3e-3,
            rtol=0.005 if released else 0.03,
        )
        assert logit_metrics["acceptance"]["passed"] and cosine > 0.9999
        if not released:
            assert max_abs < 1e-2
        else:
            assert candidate["gradients"]["relative_l2"] <= 1e-2
            assert candidate["gradients"]["cosine"] >= 0.99995
        after = _runtime_receipt()
        assert after["source_sha256"] == provenance["source_sha256"], "Production source changed during composition"
        if rank == 0:
            Path(evidence).write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "provenance": provenance,
                        "source_sha256_after": after["source_sha256"],
                        "checkpoint_scope": "released original-weight layers0-3, original49-layer schedule"
                        if released
                        else "generated tiny checkpoint",
                        "source_prefix_values": baseline["source_values"],
                        "control": "same TE backend, source, precision, global packed tokens/targets",
                        "baseline": {"dp": 4, "pp": 1, "cp": 1, "ep": baseline_ep},
                        "candidate": {"dp": 1, "pp": 2, "cp": 2, "ep": 2, "virtual_stages": 4},
                        "compute_dtype": "bfloat16",
                        "expert_backend": "torch_mm" if dispatcher == "deepep" else "torch",
                        "dispatcher": dispatcher,
                        "master_dtype": "float32",
                        "residual_dtype": "float32",
                        "loss_baseline": baseline["loss"],
                        "loss_candidate": candidate["loss"],
                        "norm_baseline": baseline["norm"],
                        "norm_candidate": candidate["norm"],
                        "logit_max_abs": max_abs,
                        "logit_cosine": cosine,
                        "full_vocabulary_metrics": logit_metrics,
                        "initial_values": candidate["initial"],
                        "all_parameter_gradient_errors": candidate["gradients"],
                        "gradient_acceptance": {
                            "all_values_finite": True,
                            "released_global_relative_l2_le": 0.01,
                            "released_global_cosine_ge": 0.99995,
                            "per_tensor_errors": "diagnostic for released weights; tiny fixture also enforces atol3e-4/rtol.15 and tensor relativeL2<.12",
                        },
                        "all_parameter_updates": candidate["updated"],
                        "observed_dtypes": {
                            "baseline": baseline["observed_dtypes"],
                            "candidate": candidate["observed_dtypes"],
                        },
                        "global_supervised_target_count": 30,
                        "objective_policy": "identical logical packs on DP4; CP2 local sum/30 multiplied2 once; shared EP divisor dp*cp/ep_shard=2 once; PP global norm",
                        "input_batch": frozen,
                        "ignored_and_padding_embedding_gradient_exact_zero": True,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dispatcher", choices=("torch", "deepep"), default="torch")
    args = parser.parse_args()
    assert torch.cuda.device_count() >= 4, "Scheduled composed control requires four GPUs"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        source = args.checkpoint.resolve(strict=True)
        (args.output / "input_batch.json").write_text(json.dumps(_released_batch(source), indent=2) + "\n")
    else:
        source = args.output / "source"
        source.mkdir()
        torch.manual_seed(34375)
        config = _config()
        if args.dispatcher == "deepep":
            # These are real dispatch/grouped-MM alignment constraints, already
            # exercised by the isolated expert LoRA CUDA gate. Keep q-width512
            # unequal to hidden128 and shared192 unequal to routed128.
            config = FlintConfig(
                **{
                    **config.to_dict(),
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "moe_intermediate_size": 128,
                    "shared_expert_intermediate_size": 192,
                }
            )
        model = config.build(backend=attention_backend("eager"))
        model.config.save_pretrained(source)
        save_file(model.state_dict_adapter.to_hf(model.state_dict()), source / "model.safetensors")
        del model
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        _worker,
        args=(port, str(source), str(args.output / "composition.json"), bool(args.checkpoint), args.dispatcher),
        nprocs=4,
        join=True,
    )


if __name__ == "__main__":
    main()
