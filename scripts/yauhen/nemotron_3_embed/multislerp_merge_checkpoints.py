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

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INDEX_FILE = "model.safetensors.index.json"
DEFAULT_STEP_NAME = "epoch_0_step_multislerp"
DEFAULT_MAX_SHARD_SIZE_GB = 2.0
HF_METADATA_EXCLUDES = {
    INDEX_FILE,
    "pytorch_model.bin.index.json",
}


@dataclass(frozen=True)
class CheckpointInput:
    """Resolved checkpoint input for a multi-checkpoint merge."""

    checkpoint: Path
    index: dict[str, Any]


@dataclass
class ShardState:
    """Mutable state for the output shard currently being assembled."""

    tensors: dict[str, Any]
    size_bytes: int
    shard_index: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge HuggingFace safetensors checkpoints with a tensor-wise weighted merge. "
            "Input paths may be consolidated model directories, AutoModel experiment directories, "
            "or checkpoint step directories."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=Path,
        help="Input checkpoint. Repeat once per checkpoint; at least two are required.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        type=float,
        help="Checkpoint weight. Repeat once per checkpoint. Values are normalized to sum to 1.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory to create.")
    parser.add_argument(
        "--layout",
        choices=("consolidated", "experiment"),
        default="consolidated",
        help="Write files directly to output-dir, or create output-dir/<step-name>/model/consolidated.",
    )
    parser.add_argument(
        "--step-name",
        default=DEFAULT_STEP_NAME,
        help="Step directory name used when --layout experiment is selected.",
    )
    parser.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=DEFAULT_MAX_SHARD_SIZE_GB,
        help="Maximum target size for each output safetensors shard.",
    )
    parser.add_argument(
        "--copy-non-floating",
        action="store_true",
        help="Copy non-floating tensors from the first checkpoint instead of requiring identical tensors.",
    )
    parser.add_argument(
        "--merge-method",
        choices=("karcher", "multislerp", "linear"),
        default="karcher",
        help="Merge floating tensors with Karcher mean, log/exp-map Multi-SLERP, or a weighted linear average.",
    )
    parser.add_argument(
        "--karcher-max-iterations",
        type=int,
        default=8,
        help="Maximum Karcher-mean refinement iterations per floating tensor.",
    )
    parser.add_argument(
        "--karcher-tolerance",
        type=float,
        default=1e-6,
        help="Stop Karcher refinement when the tangent update norm is at or below this value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-dir first if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and validate matching index keys without writing weights.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for merge computation (e.g. cuda, cuda:0, cpu). Defaults to cuda when available, else cpu.",
    )
    return parser.parse_args()


def _read_index(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / INDEX_FILE
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {INDEX_FILE}: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if "weight_map" not in index or not isinstance(index["weight_map"], dict):
        raise ValueError(f"{index_path} does not contain a valid weight_map.")
    return index


def _resolve_consolidated_dir(path: Path) -> Path:
    candidates = [
        path,
        path / "model" / "consolidated",
        path / "LATEST" / "model" / "consolidated",
        path / "LOWEST_VAL" / "model" / "consolidated",
    ]
    for candidate in candidates:
        if (candidate / INDEX_FILE).is_file():
            return candidate.resolve()

    safetensors_files = sorted(path.glob("*.safetensors")) if path.is_dir() else []
    if safetensors_files:
        raise FileNotFoundError(f"{path} has safetensors files but no {INDEX_FILE}. Indexed checkpoints are required.")

    raise FileNotFoundError(
        "Could not resolve a consolidated HF checkpoint. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _validate_indexes(indexes: list[dict[str, Any]]) -> list[str]:
    keys = list(indexes[0]["weight_map"].keys())
    expected = set(keys)
    for index_number, index in enumerate(indexes[1:], start=2):
        actual = set(index["weight_map"].keys())
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing from checkpoint {index_number}: {missing[:20]}")
            if extra:
                details.append(f"extra in checkpoint {index_number}: {extra[:20]}")
            raise ValueError("Checkpoint tensor keys do not match; " + "; ".join(details))
    return keys


def _normalize_weights(weights: list[float] | None, checkpoint_count: int) -> list[float]:
    if weights is None:
        return [1.0 / checkpoint_count] * checkpoint_count
    if len(weights) != checkpoint_count:
        raise ValueError(f"Expected {checkpoint_count} weights, got {len(weights)}.")
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"Weights must be non-negative, got {weights}.")

    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError(f"At least one weight must be positive, got {weights}.")
    return [weight / total_weight for weight in weights]


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def _resolve_output_model_dir(output_dir: Path, layout: str, step_name: str) -> Path:
    if layout == "consolidated":
        return output_dir
    return output_dir / step_name / "model" / "consolidated"


def _load_runtime_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "Multi-SLERP merging requires torch and safetensors. Run this inside the AutoModel training/eval "
            "environment or install the project dependencies with uv."
        ) from exc
    return torch, safe_open, save_file


def _resolve_device(torch: Any, device: str | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device!r} but CUDA is not available.")
    return device


def _copy_hf_metadata_files(source_dir: Path, output_dir: Path) -> None:
    for source_path in sorted(source_dir.iterdir()):
        if source_path.is_dir():
            continue
        if source_path.suffix == ".safetensors" or source_path.name in HF_METADATA_EXCLUDES:
            continue
        shutil.copy2(source_path, output_dir / source_path.name)


def _tensor_file(index: dict[str, Any], tensor_name: str) -> str:
    try:
        return index["weight_map"][tensor_name]
    except KeyError as exc:
        raise KeyError(f"Tensor {tensor_name!r} is missing from weight_map.") from exc


def _read_tensor(
    model_dir: Path,
    index: dict[str, Any],
    tensor_name: str,
    safe_open: Any,
    device: str,
) -> Any:
    tensor_path = model_dir / _tensor_file(index, tensor_name)
    with safe_open(tensor_path, framework="pt", device=device) as handle:
        return handle.get_tensor(tensor_name)


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _linear_average(tensors: list[Any], weights: list[float]) -> Any:
    dtype = tensors[0].dtype
    merged = tensors[0].float().mul(weights[0])
    for tensor, weight in zip(tensors[1:], weights[1:]):
        merged.add_(tensor.float(), alpha=weight)
    return merged.to(dtype=dtype)


def _unit_direction_and_norm(tensor: Any, torch: Any) -> tuple[Any, Any]:
    flat = tensor.float().reshape(-1)
    norm = torch.linalg.vector_norm(flat)
    if norm.item() == 0.0:
        return flat, norm
    return flat / norm, norm


def _karcher_mean_direction(
    unit_directions: list[Any],
    weights: list[float],
    max_iterations: int,
    tolerance: float,
    torch: Any,
) -> tuple[Any | None, int, bool]:
    direction = unit_directions[0].mul(weights[0])
    for unit_direction, weight in zip(unit_directions[1:], weights[1:]):
        direction.add_(unit_direction, alpha=weight)

    direction_norm = torch.linalg.vector_norm(direction)
    if direction_norm.item() == 0.0:
        return None, 0, False
    direction.div_(direction_norm)

    iterations = 0
    converged = False
    for _ in range(max_iterations):
        iterations += 1
        tangent = torch.zeros_like(direction)
        for unit_direction, weight in zip(unit_directions, weights):
            cosine = torch.dot(direction, unit_direction)
            cosine = torch.clamp(cosine, -1.0, 1.0)
            angle = torch.acos(cosine)
            angle_value = angle.item()
            if angle_value <= tolerance:
                continue

            sin_angle = torch.sin(angle)
            if abs(sin_angle.item()) <= tolerance:
                return None, iterations, False

            tangent.add_(unit_direction - cosine * direction, alpha=weight * (angle / sin_angle).item())

        tangent_norm = torch.linalg.vector_norm(tangent)
        tangent_norm_value = tangent_norm.item()
        if tangent_norm_value <= tolerance:
            converged = True
            break

        direction = torch.cos(tangent_norm) * direction + (torch.sin(tangent_norm) / tangent_norm) * tangent
        direction.div_(torch.linalg.vector_norm(direction))

    return direction, iterations, converged


def _multislerp_mean_direction(
    unit_directions: list[Any],
    weights: list[float],
    tolerance: float,
    torch: Any,
) -> Any | None:
    mean_direction = unit_directions[0].mul(weights[0])
    for unit_direction, weight in zip(unit_directions[1:], weights[1:]):
        mean_direction.add_(unit_direction, alpha=weight)

    mean_direction_norm = torch.linalg.vector_norm(mean_direction)
    if mean_direction_norm.item() <= tolerance:
        return None
    mean_direction.div_(mean_direction_norm)

    tangent = torch.zeros_like(mean_direction)
    for unit_direction, weight in zip(unit_directions, weights):
        cosine = torch.dot(mean_direction, unit_direction)
        cosine = torch.clamp(cosine, -1.0, 1.0)
        angle = torch.acos(cosine)
        angle_value = angle.item()
        if angle_value <= tolerance:
            continue

        sin_angle = torch.sin(angle)
        if abs(sin_angle.item()) <= tolerance:
            return None

        log_map = unit_direction - cosine * mean_direction
        log_map.mul_((angle / sin_angle).item())
        tangent.add_(log_map, alpha=weight)

    tangent_norm = torch.linalg.vector_norm(tangent)
    tangent_norm_value = tangent_norm.item()
    if tangent_norm_value <= tolerance:
        return mean_direction

    direction = torch.cos(tangent_norm) * mean_direction + (torch.sin(tangent_norm) / tangent_norm) * tangent
    direction_norm = torch.linalg.vector_norm(direction)
    if direction_norm.item() <= tolerance:
        return None
    return direction / direction_norm


def _spherical_mean(
    tensors: list[Any],
    weights: list[float],
    max_iterations: int,
    tolerance: float,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    dtype = tensors[0].dtype
    shape = tensors[0].shape
    unit_directions = []
    norms = []

    for tensor in tensors:
        unit_direction, norm = _unit_direction_and_norm(tensor, torch)
        if norm.item() == 0.0:
            return _linear_average(tensors, weights), {"method": "linear_fallback", "reason": "zero_norm_input"}
        unit_directions.append(unit_direction)
        norms.append(norm)

    direction, iterations, converged = _karcher_mean_direction(
        unit_directions, weights, max_iterations, tolerance, torch
    )
    if direction is None:
        return _linear_average(tensors, weights), {
            "method": "linear_fallback",
            "reason": "degenerate_direction",
            "iterations": iterations,
        }

    mean_norm = norms[0] * weights[0]
    for norm, weight in zip(norms[1:], weights[1:]):
        mean_norm = mean_norm + norm * weight
    merged = (direction.reshape(shape) * mean_norm).to(dtype=dtype)
    return merged, {"method": "karcher", "iterations": iterations, "converged": converged}


def _multislerp_mean(
    tensors: list[Any],
    weights: list[float],
    tolerance: float,
    torch: Any,
) -> Any:
    dtype = tensors[0].dtype
    shape = tensors[0].shape
    unit_directions = []
    norms = []

    for tensor in tensors:
        unit_direction, norm = _unit_direction_and_norm(tensor, torch)
        if norm.item() == 0.0:
            return _linear_average(tensors, weights)
        unit_directions.append(unit_direction)
        norms.append(norm)

    direction = _multislerp_mean_direction(unit_directions, weights, tolerance, torch)
    if direction is None:
        return _linear_average(tensors, weights)

    mean_norm = norms[0] * weights[0]
    for norm, weight in zip(norms[1:], weights[1:]):
        mean_norm = mean_norm + norm * weight
    return (direction.reshape(shape) * mean_norm).to(dtype=dtype)


def _merge_tensor(
    tensor_name: str,
    inputs: list[CheckpointInput],
    weights: list[float],
    merge_method: str,
    copy_non_floating: bool,
    max_iterations: int,
    tolerance: float,
    torch: Any,
    safe_open: Any,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    tensors = [
        _read_tensor(input_checkpoint.checkpoint, input_checkpoint.index, tensor_name, safe_open, device)
        for input_checkpoint in inputs
    ]
    first = tensors[0]

    for tensor_number, tensor in enumerate(tensors[1:], start=2):
        if tensor.shape != first.shape:
            raise ValueError(
                f"Shape mismatch for {tensor_name} in checkpoint {tensor_number}: "
                f"{tuple(tensor.shape)} != {tuple(first.shape)}"
            )
        if tensor.dtype != first.dtype:
            raise ValueError(
                f"Dtype mismatch for {tensor_name} in checkpoint {tensor_number}: {tensor.dtype} != {first.dtype}"
            )

    if not torch.is_floating_point(first):
        if copy_non_floating:
            return first.contiguous(), {"method": "copy_non_floating"}
        for tensor_number, tensor in enumerate(tensors[1:], start=2):
            if not torch.equal(first, tensor):
                raise ValueError(
                    f"Non-floating tensor {tensor_name} differs in checkpoint {tensor_number}. "
                    "Pass --copy-non-floating to copy checkpoint 1's value."
                )
        return first.contiguous(), {"method": "copy_identical"}

    if merge_method == "linear":
        return _linear_average(tensors, weights).contiguous(), {"method": "linear"}
    if merge_method == "multislerp":
        return _multislerp_mean(tensors, weights, tolerance, torch).contiguous(), {"method": "multislerp"}

    merged, info = _spherical_mean(tensors, weights, max_iterations, tolerance, torch)
    return merged.contiguous(), info


def _save_current_shard(
    state: ShardState,
    output_model_dir: Path,
    weight_map: dict[str, str],
    save_file: Any,
) -> int:
    if not state.tensors:
        return 0

    shard_name = f"model-{state.shard_index:05d}-of-XXXXX.safetensors"
    shard_path = output_model_dir / shard_name
    save_file(state.tensors, shard_path, metadata={"format": "pt"})
    for tensor_name in state.tensors:
        weight_map[tensor_name] = shard_name
    saved_size = state.size_bytes
    state.tensors = {}
    state.size_bytes = 0
    state.shard_index += 1
    return saved_size


def _finalize_shard_names(output_model_dir: Path, weight_map: dict[str, str], shard_count: int) -> None:
    if shard_count == 0:
        raise ValueError("No shards were written.")

    old_to_new = {}
    for shard_index in range(1, shard_count + 1):
        old_name = f"model-{shard_index:05d}-of-XXXXX.safetensors"
        new_name = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        old_to_new[old_name] = new_name
        (output_model_dir / old_name).rename(output_model_dir / new_name)

    for tensor_name, old_name in list(weight_map.items()):
        weight_map[tensor_name] = old_to_new[old_name]


def _write_index(output_model_dir: Path, weight_map: dict[str, str], total_size: int) -> None:
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with (output_model_dir / INDEX_FILE).open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_merge_metadata(
    output_dir: Path,
    output_model_dir: Path,
    args: argparse.Namespace,
    inputs: list[CheckpointInput],
    weights: list[float],
    device: str,
) -> None:
    method_names = {
        "karcher": "weighted_tensorwise_karcher_spherical_mean",
        "linear": "weighted_tensorwise_linear_average",
        "multislerp": "weighted_tensorwise_multislerp_log_exp_mean",
    }
    metadata = {
        "checkpoints": [str(input_checkpoint.checkpoint) for input_checkpoint in inputs],
        "device": device,
        "karcher_max_iterations": args.karcher_max_iterations,
        "karcher_tolerance": args.karcher_tolerance,
        "layout": args.layout,
        "max_shard_size_gb": args.max_shard_size_gb,
        "method": method_names[args.merge_method],
        "merge_method": args.merge_method,
        "output_model_dir": str(output_model_dir),
        "step_name": args.step_name if args.layout == "experiment" else None,
        "weights": weights,
    }
    for target_dir in {output_dir, output_model_dir}:
        with (target_dir / "multislerp_merge_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")


def _create_experiment_links(output_dir: Path, step_name: str) -> None:
    for link_name in ("LATEST", "LOWEST_VAL"):
        link_path = output_dir / link_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(step_name, link_path)


def _merge_checkpoints(
    args: argparse.Namespace,
    inputs: list[CheckpointInput],
    tensor_names: list[str],
    weights: list[float],
) -> None:
    if len(inputs) < 2:
        raise ValueError("At least two checkpoints are required.")
    if args.max_shard_size_gb <= 0:
        raise ValueError(f"--max-shard-size-gb must be positive, got {args.max_shard_size_gb}")
    if args.merge_method == "karcher" and args.karcher_max_iterations <= 0:
        raise ValueError(f"--karcher-max-iterations must be positive, got {args.karcher_max_iterations}")
    if args.merge_method in {"karcher", "multislerp"} and args.karcher_tolerance <= 0:
        raise ValueError(f"--karcher-tolerance must be positive, got {args.karcher_tolerance}")

    torch, safe_open, save_file = _load_runtime_modules()
    device = _resolve_device(torch, args.device)
    output_model_dir = _resolve_output_model_dir(args.output_dir, args.layout, args.step_name)
    _prepare_output_dir(args.output_dir, args.overwrite)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    _copy_hf_metadata_files(inputs[0].checkpoint, output_model_dir)

    max_shard_bytes = int(args.max_shard_size_gb * (1024**3))
    weight_map: dict[str, str] = {}
    shard_state = ShardState(tensors={}, size_bytes=0, shard_index=1)
    total_size = 0

    logger.info(
        "Merging %d tensors from %d checkpoints with method %s, weights %s on device %s",
        len(tensor_names),
        len(inputs),
        args.merge_method,
        weights,
        device,
    )
    method_counts: dict[str, int] = {}
    karcher_iterations: list[int] = []
    karcher_converged = 0
    karcher_hit_cap = 0
    for index, tensor_name in enumerate(tensor_names, start=1):
        merged, info = _merge_tensor(
            tensor_name,
            inputs,
            weights,
            args.merge_method,
            args.copy_non_floating,
            args.karcher_max_iterations,
            args.karcher_tolerance,
            torch,
            safe_open,
            device,
        )
        merged = merged.cpu()
        tensor_size = _tensor_nbytes(merged)
        if shard_state.tensors and shard_state.size_bytes + tensor_size > max_shard_bytes:
            _save_current_shard(shard_state, output_model_dir, weight_map, save_file)
        shard_state.tensors[tensor_name] = merged
        shard_state.size_bytes += tensor_size
        total_size += tensor_size

        method_counts[info["method"]] = method_counts.get(info["method"], 0) + 1
        if info["method"] == "karcher":
            karcher_iterations.append(info["iterations"])
            if info["converged"]:
                karcher_converged += 1
            else:
                karcher_hit_cap += 1
            logger.debug(
                "Tensor %s merged via karcher in %d iteration(s) (%s)",
                tensor_name,
                info["iterations"],
                "converged" if info["converged"] else "hit max iterations",
            )
        else:
            logger.debug("Tensor %s merged via %s", tensor_name, info["method"])

        if index % 25 == 0 or index == len(tensor_names):
            logger.info("Merged %d/%d tensors", index, len(tensor_names))

    logger.info("Per-tensor merge method counts: %s", method_counts)
    if karcher_iterations:
        logger.info(
            "Karcher spherical mean applied to %d tensor(s): %d converged, %d hit max iterations "
            "(%d); iterations min/mean/max = %d/%.2f/%d",
            len(karcher_iterations),
            karcher_converged,
            karcher_hit_cap,
            args.karcher_max_iterations,
            min(karcher_iterations),
            sum(karcher_iterations) / len(karcher_iterations),
            max(karcher_iterations),
        )
    elif args.merge_method == "karcher":
        logger.warning("Merge method is 'karcher' but no tensor used the Karcher spherical mean (all fell back).")

    _save_current_shard(shard_state, output_model_dir, weight_map, save_file)
    shard_count = shard_state.shard_index - 1
    _finalize_shard_names(output_model_dir, weight_map, shard_count)
    _write_index(output_model_dir, weight_map, total_size)
    _write_merge_metadata(args.output_dir, output_model_dir, args, inputs, weights, device)

    if args.layout == "experiment":
        _create_experiment_links(args.output_dir, args.step_name)

    logger.info("Wrote multi-SLERP checkpoint to %s", output_model_dir)
    logger.info("Output shards: %d; total tensor bytes: %d", shard_count, total_size)


def main() -> None:
    """Run weighted multi-SLERP checkpoint merging from command-line arguments."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("Pass --checkpoint at least twice.")

    inputs = []
    for checkpoint_path in args.checkpoint:
        checkpoint = _resolve_consolidated_dir(checkpoint_path)
        inputs.append(CheckpointInput(checkpoint=checkpoint, index=_read_index(checkpoint)))

    tensor_names = _validate_indexes([input_checkpoint.index for input_checkpoint in inputs])
    weights = _normalize_weights(args.weight, len(inputs))
    for index, input_checkpoint in enumerate(inputs, start=1):
        logger.info("Resolved checkpoint %d: %s", index, input_checkpoint.checkpoint)
    logger.info("Matching tensor keys: %d", len(tensor_names))
    logger.info("Weights: %s", weights)

    if args.dry_run:
        output_model_dir = _resolve_output_model_dir(args.output_dir, args.layout, args.step_name)
        logger.info("Dry run only. Would write model to: %s", output_model_dir)
        return

    _merge_checkpoints(args, inputs, tensor_names, weights)


if __name__ == "__main__":
    main()
