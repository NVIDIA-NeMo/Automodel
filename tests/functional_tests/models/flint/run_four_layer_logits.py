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

"""Released four-layer/full-vocabulary comparison against independent FP32 SDPA."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.functional_tests.models.flint.sdpa_reference import (
    FlintSDPAReference,
    LogitAccumulator,
    ReferenceConfig,
    compare_hidden,
)


def sha256_file(path: Path) -> str:
    """Hash a metadata/tool artifact; checkpoint payload files are not passed here."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash contiguous tensor bytes without an intermediate Python bytes copy."""
    cpu = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(memoryview(cpu.numpy())).hexdigest()


def load_reference_weights(
    checkpoint: Path, raw: dict[str, object], config: ReferenceConfig, device: torch.device
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    """Read exactly the selected source tensors, preserving their BF16 provenance.

    Args:
        checkpoint: Original read-only checkpoint directory.
        raw: Original parsed config with its complete released layer schedule.
        config: Explicit first-layer prefix and independently enumerated shapes.
        device: CPU/CUDA device for FP32 reference tensors.

    Returns:
        FP32 source-name weights and a per-tensor shape/dtype/content receipt.
        The complete index key schema is checked; omitted layer payloads are
        never read. No HF/native adapter participates in this source loading.
    """
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    full = ReferenceConfig.from_checkpoint(raw, num_hidden_layers=int(raw["num_hidden_layers"])).expected_shapes()
    if set(mapping) != set(full):
        raise ValueError("complete source index keys differ from independently enumerated architecture")
    shapes = config.expected_shapes()
    groups: dict[str, list[str]] = {}
    for name in shapes:
        filename = mapping[name]
        if Path(filename).name != filename:
            raise ValueError("source shard names must be local basenames")
        groups.setdefault(filename, []).append(name)
    weights, receipt = {}, []
    for filename, names in sorted(groups.items()):
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as source:
            for name in names:
                value = source.get_tensor(name)
                if tuple(value.shape) != shapes[name] or value.dtype != torch.bfloat16:
                    raise ValueError(f"released source tensor {name} must have BF16 shape {shapes[name]}")
                receipt.append(
                    {
                        "name": name,
                        "source_shard": filename,
                        "source_dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "source_tensor_sha256": tensor_sha256(value),
                    }
                )
                weights[name] = value.to(device=device, dtype=torch.float32)
    return weights, receipt


def native_candidate(
    args: argparse.Namespace, raw: dict[str, object], weights: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Construct the public native candidate and prove every prefix value loaded."""
    from nemo_automodel import NeMoAutoModelForCausalLM
    from nemo_automodel.components.models.flint.config import FlintConfig

    dtype = torch.float32 if args.candidate_dtype == "float32" else torch.bfloat16
    backend = {
        "attn": args.attn,
        "linear": "torch",
        "rms_norm": "torch_fp32",
        "rope": "torch",
        "rope_fusion": False,
        "experts": "torch",
        "dispatcher": "torch",
        "gate_precision": None,
        "enable_fsdp_optimizations": False,
    }
    overrides = {
        "num_hidden_layers": 4,
        "schedule_num_hidden_layers": int(raw["num_hidden_layers"]),
        "router_aux_loss_coef": 0.0,
        "use_cache": False,
    }
    if args.residual_dtype == "fp32":
        overrides["residual_dtype"] = "fp32"
    if args.attention_compute_dtype == "fp16":
        overrides["attention_compute_dtype"] = "fp16"
    common = {
        "dtype": dtype,
        "backend": backend,
        "attn_implementation": "eager" if args.attn == "eager" else "sdpa",
        "use_liger_kernel": False,
        "use_sdpa_patching": False,
    }
    if args.load_method == "from_pretrained":
        model = NeMoAutoModelForCausalLM.from_pretrained(str(args.checkpoint), config=overrides, **common)
        model.to(device=device)
    else:
        data = {key: value for key, value in raw.items() if key != "torch_dtype"}
        data.update(overrides, dtype=args.candidate_dtype)
        model = NeMoAutoModelForCausalLM.from_config(FlintConfig(**data), **common).to(device=device)
        converted = model.state_dict_adapter.from_hf(dict(weights))
        model.load_state_dict(converted, strict=True)
        del converted
    model.eval()
    if model.backend.attn != args.attn or any(
        layer.self_attn.backend.attn != args.attn for layer in model.model.layers
    ):
        raise AssertionError("actual native attention backend differs from the requested comparison")
    if len(model.model.layers) != 4 or tuple(model.config.layer_types[:4]) != (
        "full_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
    ):
        raise AssertionError("native candidate did not preserve released layers 0-3")
    if tuple(i for i in model.config.no_rope_layer_indices if i < 4) != (0, 1):
        raise AssertionError("native prefix RoPE phase changed")

    # Convert one native tensor at a time: no retained second model-sized export.
    observed: set[str] = set()
    for native_name, value in model.state_dict().items():
        if native_name.endswith("_extra_state"):
            continue
        for name, actual in model.state_dict_adapter.convert_single_tensor_to_hf(native_name, value):
            if name not in weights or name in observed:
                raise AssertionError(f"unexpected/duplicate exported native prefix tensor {name}")
            if actual.shape != weights[name].shape or not torch.equal(actual.float(), weights[name]):
                raise AssertionError(f"native prefix weight differs from released source: {name}")
            if actual.untyped_storage().data_ptr() == weights[name].untyped_storage().data_ptr():
                raise AssertionError(f"reference and native tensor storage alias: {name}")
            observed.add(name)
    if observed != set(weights):
        raise AssertionError(f"native prefix is missing source tensors: {sorted(set(weights) - observed)}")
    dtype_counts: dict[str, int] = {}
    for parameter in model.parameters():
        name = str(parameter.dtype)
        dtype_counts[name] = dtype_counts.get(name, 0) + parameter.numel()
    if dtype == torch.float32 and set(dtype_counts) != {"torch.float32"}:
        raise AssertionError(f"FP32 candidate contains other resident dtypes: {dtype_counts}")
    if model.lm_head.weight.dtype != torch.float32 or any(
        layer.self_attn._fp32_params.softmax_offset.dtype != torch.float32 for layer in model.model.layers
    ):
        raise AssertionError("candidate head or learned sink parameters lost protected FP32 dtype")
    return model, {
        "load_method": args.load_method,
        "backend": backend,
        "actual_attention_backend": model.backend.attn,
        "requested_attention_compute_dtype": args.attention_compute_dtype,
        "requested_bulk_dtype": str(dtype),
        "runtime_overrides": overrides,
        "exact_released_prefix_tensor_count": len(observed),
        "all_selected_source_values_match": True,
        "reference_native_storage_independent": True,
        "resident_parameter_count_by_dtype": dtype_counts,
        "resident_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }


def parse_args() -> argparse.Namespace:
    """Parse explicit source, candidate backend and evidence paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn", choices=("eager", "sdpa", "te"), default="sdpa")
    parser.add_argument("--candidate-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--residual-dtype", choices=("native", "fp32"), default="native")
    parser.add_argument("--attention-compute-dtype", choices=("native", "fp16"), default="native")
    parser.add_argument(
        "--load-method", choices=("released-prefix-copy", "from_pretrained"), default="released-prefix-copy"
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--rows-per-chunk", type=int, default=32)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    """Run the honest four-layer FP32-reference protocol and preserve failure evidence."""
    args = parse_args()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    if output.is_relative_to(checkpoint):
        raise ValueError("output must be outside the read-only source checkpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new output directory so earlier attempts remain intact")
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "methodology_reference_pr": "https://github.com/NVIDIA-NeMo/Automodel/pull/4375",
        "methodology_pr_head": "79996e4ef7a5ed234f91a804f8ce646e34cd0e22",  # pragma: allowlist secret
        "official_hf_reference": False,
        "reference": "independent test-owned FP32 PyTorch SDPA math architecture reference",
        "reference_math": "unfused FP32 projections, RMSNorm, NeoX RoPE, sigmoid top-k/shared MoE; SDPA math with zero-V sink",
        "scope": "first four actual released layers, every requested input position, complete source vocabulary",
        "primary_4096_position_protocol": args.sequence_length == 4096,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "query_chunk_size": args.query_chunk_size,
        "logit_rows_per_chunk": args.rows_per_chunk,
        "script_sha256": sha256_file(Path(__file__)),
        "helper_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in ("sdpa_reference.py",)},
        "platform": {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system()},
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")},
    }
    path = output / "manifest.json"

    def flush() -> None:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")

    flush()
    handles = []
    started = time.perf_counter()
    try:
        if min(args.sequence_length, args.rows_per_chunk, args.query_chunk_size) <= 0:
            raise ValueError("sequence and chunk sizes must be positive")
        if args.attention_compute_dtype == "fp16" and (args.attn != "te" or args.candidate_dtype != "float32"):
            raise ValueError("explicit FP16 attention requires --attn te --candidate-dtype float32")
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
            raise ValueError("NVIDIA_TF32_OVERRIDE=1 conflicts with the FP32 correctness reference")
        manifest["math_policy"] = {
            "tf32": False,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "autocast_reference": False,
            "reference_sdpa_backend": "MATH",
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        }
        repo = Path(__file__).resolve().parents[4]
        manifest["repository_commit_at_measurement"] = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        manifest["flint_implementation_sha256"] = {
            str(path.relative_to(repo)): sha256_file(path)
            for path in sorted((repo / "nemo_automodel/components/models/flint").glob("*.py"))
        }
        manifest["shared_implementation_sha256"] = {
            str(path.relative_to(repo)): sha256_file(path)
            for path in sorted((repo / "nemo_automodel").rglob("*.py"))
            if not path.is_relative_to(repo / "nemo_automodel/components/models/flint")
        }
        try:
            manifest["packages"]["transformer-engine"] = importlib.metadata.version("transformer-engine")
        except importlib.metadata.PackageNotFoundError:
            manifest["packages"]["transformer-engine"] = None
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            manifest["hardware"] = {
                "device": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "device_total_memory_bytes": properties.total_memory,
                "cuda": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        raw = json.loads((checkpoint / "config.json").read_text())
        config = ReferenceConfig.from_checkpoint(raw)
        released_architecture = (
            config.source_num_hidden_layers == 49
            and config.hidden_size == 2048
            and config.vocab_size == 150144
            and config.num_attention_heads == 32
            and config.num_key_value_heads == 4
            and config.head_dim == 128
            and config.num_experts == 256
            and config.num_experts_per_tok == 6
        )
        manifest["released_architecture_identified"] = released_architecture
        manifest["primary_4096_position_protocol"] = args.sequence_length == 4096 and released_architecture
        manifest["checkpoint_path"] = str(checkpoint)
        if args.sequence_length > int(raw["max_position_embeddings"]):
            raise ValueError("input exceeds source maximum position extent")
        manifest.update(
            {
                "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
                "checkpoint_index_sha256": sha256_file(checkpoint / "model.safetensors.index.json"),
                "source_num_hidden_layers": config.source_num_hidden_layers,
                "reference_config": dataclasses.asdict(config),
                "checkpoint_hf_modeling_files": sorted(p.name for p in checkpoint.glob("modeling*.py")),
                "source_auxiliary_coefficient_metadata": raw["router_aux_loss_coef"],
            }
        )
        supplied = checkpoint / "sglang/flint_sglang-src/src/flint_sglang/model.py"
        if supplied.is_file():
            manifest["supplied_sglang_model_sha256"] = sha256_file(supplied)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        ids_cpu = torch.randint(config.vocab_size, (1, args.sequence_length), generator=generator, dtype=torch.int64)
        input_tensors = {
            "input_ids": ids_cpu,
            "attention_mask": torch.ones_like(ids_cpu, dtype=torch.bool),
            "position_ids": torch.arange(args.sequence_length, dtype=torch.int64).unsqueeze(0),
        }
        save_file(input_tensors, output / "inputs.safetensors")
        manifest["input_artifact_sha256"] = sha256_file(output / "inputs.safetensors")
        manifest["input_ids_tensor_sha256"] = tensor_sha256(ids_cpu)
        manifest["all_input_positions_valid"] = True
        manifest["reference_load_started_seconds"] = time.perf_counter() - started
        manifest["stage"] = "source_prefix_load"
        flush()
        weights, receipt = load_reference_weights(checkpoint, raw, config, device)
        (output / "source_prefix_tensor_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        manifest["source_prefix_tensor_receipt_sha256"] = sha256_file(output / "source_prefix_tensor_receipt.json")
        manifest["reference_parameter_count"] = sum(x.numel() for x in weights.values())
        reference = FlintSDPAReference(config, weights, query_chunk_size=args.query_chunk_size)
        manifest["stage"] = "native_candidate_load"
        flush()
        model, candidate_manifest = native_candidate(args, raw, weights, device)
        manifest["candidate"] = candidate_manifest
        attention_calls: list[dict[str, object]] = []
        manifest["actual_attention_calls"] = attention_calls
        for index, layer in enumerate(model.model.layers):
            module = layer.self_attn.attn_module
            if args.attn != "te":
                if module is not None:
                    raise AssertionError("non-TE native comparison unexpectedly owns a TE DPA")
                continue
            if module is None:
                raise AssertionError("actual TE attention module is absent")

            def observe_inputs(module: torch.nn.Module, values: tuple[torch.Tensor, ...], index: int = index) -> None:
                attention_calls.append(
                    {
                        "layer": index,
                        "qkv_dtypes": [str(value.dtype) for value in values[:3]],
                        "qkv_shapes": [list(value.shape) for value in values[:3]],
                        "sink_dtype": str(module.softmax_offset.dtype),
                    }
                )
                if args.attention_compute_dtype == "fp16" and (
                    len(values) < 3 or any(value.dtype != torch.float16 for value in values[:3])
                ):
                    raise AssertionError("native typed FP16 attention policy did not reach actual TE Q/K/V")
                if module.softmax_offset.dtype != torch.float32:
                    raise AssertionError("actual TE sink lost protected FP32 compute")

            def observe_output(module: torch.nn.Module, values: tuple[torch.Tensor, ...], result: torch.Tensor) -> None:
                attention_calls[-1]["output_dtype"] = str(result.dtype)
                if args.attention_compute_dtype == "fp16" and result.dtype != torch.float16:
                    raise AssertionError("actual TE attention output did not use the requested FP16 compute")

            handles.extend(
                (module.register_forward_pre_hook(observe_inputs), module.register_forward_hook(observe_output))
            )
        manifest["stage"] = "forward"
        flush()
        native_layers: list[torch.Tensor] = []
        for layer in model.model.layers:
            handles.append(
                layer.register_forward_hook(
                    lambda module, inputs, result: native_layers.append(result.detach().clone())
                )
            )
        batch = {name: value.to(device) for name, value in input_tensors.items()}
        with torch.inference_mode():
            target = reference.forward_hidden(**batch)
            actual = model(**batch).logits
            if actual.shape != (1, args.sequence_length, config.vocab_size) or len(native_layers) != 4:
                raise AssertionError("native forward did not emit every layer and full-vocabulary prompt row")
            if args.attn == "te" and len(attention_calls) != 4:
                raise AssertionError("actual TE comparison did not observe exactly four released attention calls")
            accumulator = LogitAccumulator(config.vocab_size)
            for start in range(0, args.sequence_length, args.rows_per_chunk):
                end = min(start + args.rows_per_chunk, args.sequence_length)
                expected = reference.head(target.final_norm[:, start:end]).squeeze(0)
                accumulator.update(
                    expected,
                    actual[0, start:end],
                    positions=[(0, i) for i in range(start, end)],
                )
            metrics = accumulator.result()
        if metrics["evaluated_positions"] != args.sequence_length or metrics["vocab_size"] != config.vocab_size:
            raise AssertionError("metric coverage is incomplete")
        manifest["metrics"] = metrics
        manifest["hidden_layer_metrics"] = {
            str(i): compare_hidden(expected, observed)
            for i, (expected, observed) in enumerate(zip(target.layers, native_layers))
        }
        manifest["hidden_layer_tensor_sha256"] = {
            str(i): {"reference": tensor_sha256(expected), "candidate": tensor_sha256(observed)}
            for i, (expected, observed) in enumerate(zip(target.layers, native_layers))
        }
        passed = bool(metrics["acceptance"]["passed"])
        manifest["status"] = "passed" if passed else "failed"
        manifest["strict_forward_parity_passed"] = passed
        manifest["primary_protocol_passed"] = passed and manifest["primary_4096_position_protocol"]
        if not passed:
            save_file(
                {
                    **{f"reference.layer.{i}": value.cpu().contiguous() for i, value in enumerate(target.layers)},
                    **{f"candidate.layer.{i}": value.cpu().contiguous() for i, value in enumerate(native_layers)},
                    "reference.final_norm": target.final_norm.cpu().contiguous(),
                },
                output / "failure_hidden_states.safetensors",
            )
            manifest["failure_hidden_states_sha256"] = sha256_file(output / "failure_hidden_states.safetensors")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            manifest["process_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        manifest["elapsed_seconds"] = time.perf_counter() - started
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
    if not manifest["strict_forward_parity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
