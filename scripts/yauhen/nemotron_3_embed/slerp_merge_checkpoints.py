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
DEFAULT_STEP_NAME = "epoch_0_step_slerp"
DEFAULT_MAX_SHARD_SIZE_GB = 2.0
HF_METADATA_EXCLUDES = {
    INDEX_FILE,
    "pytorch_model.bin.index.json",
}


@dataclass(frozen=True)
class MergeInputs:
    """Resolved checkpoint inputs for a two-checkpoint merge."""

    checkpoint_a: Path
    checkpoint_b: Path
    index_a: dict[str, Any]
    index_b: dict[str, Any]


@dataclass
class ShardState:
    """Mutable state for the output shard currently being assembled."""

    tensors: dict[str, Any]
    size_bytes: int
    shard_index: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two HuggingFace safetensors checkpoints with tensor-wise SLERP. "
            "Input paths may be consolidated model directories, AutoModel experiment directories, "
            "or checkpoint step directories."
        )
    )
    parser.add_argument("--checkpoint-a", required=True, type=Path, help="First/source checkpoint.")
    parser.add_argument("--checkpoint-b", required=True, type=Path, help="Second/target checkpoint.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory to create.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Interpolation amount from checkpoint A to B. 0.0=A, 1.0=B.",
    )
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
        help="Copy non-floating tensors from checkpoint A instead of requiring identical tensors.",
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


def _validate_indexes(index_a: dict[str, Any], index_b: dict[str, Any]) -> list[str]:
    keys_a = list(index_a["weight_map"].keys())
    keys_b = list(index_b["weight_map"].keys())
    set_a = set(keys_a)
    set_b = set(keys_b)
    missing_from_b = sorted(set_a - set_b)
    extra_in_b = sorted(set_b - set_a)
    if missing_from_b or extra_in_b:
        details = []
        if missing_from_b:
            details.append(f"missing from checkpoint B: {missing_from_b[:20]}")
        if extra_in_b:
            details.append(f"extra in checkpoint B: {extra_in_b[:20]}")
        raise ValueError("Checkpoint tensor keys do not match; " + "; ".join(details))
    return keys_a


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
            "SLERP merging requires torch and safetensors. Run this inside the AutoModel training/eval "
            "environment or install the project dependencies with uv."
        ) from exc
    return torch, safe_open, save_file


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


def _read_tensor(model_dir: Path, index: dict[str, Any], tensor_name: str, safe_open: Any) -> Any:
    tensor_path = model_dir / _tensor_file(index, tensor_name)
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name)


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _linear_interpolate(tensor_a: Any, tensor_b: Any, alpha: float) -> Any:
    return tensor_a.lerp(tensor_b, alpha)


def _slerp(tensor_a: Any, tensor_b: Any, alpha: float, torch: Any) -> Any:
    dtype = tensor_a.dtype
    a = tensor_a.float()
    b = tensor_b.float()
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    a_norm = torch.linalg.vector_norm(a_flat)
    b_norm = torch.linalg.vector_norm(b_flat)
    if a_norm.item() == 0.0 or b_norm.item() == 0.0:
        return _linear_interpolate(a, b, alpha).to(dtype=dtype)

    cosine = torch.dot(a_flat, b_flat) / (a_norm * b_norm)
    cosine = torch.clamp(cosine, -1.0, 1.0)
    if abs(cosine.item()) > 0.9995:
        return _linear_interpolate(a, b, alpha).to(dtype=dtype)

    theta = torch.acos(cosine)
    sin_theta = torch.sin(theta)
    scale_a = torch.sin((1.0 - alpha) * theta) / sin_theta
    scale_b = torch.sin(alpha * theta) / sin_theta
    return (scale_a * a + scale_b * b).to(dtype=dtype)


def _merge_tensor(
    tensor_name: str,
    inputs: MergeInputs,
    alpha: float,
    copy_non_floating: bool,
    torch: Any,
    safe_open: Any,
) -> Any:
    tensor_a = _read_tensor(inputs.checkpoint_a, inputs.index_a, tensor_name, safe_open)
    tensor_b = _read_tensor(inputs.checkpoint_b, inputs.index_b, tensor_name, safe_open)

    if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"Shape mismatch for {tensor_name}: {tuple(tensor_a.shape)} != {tuple(tensor_b.shape)}")
    if tensor_a.dtype != tensor_b.dtype:
        raise ValueError(f"Dtype mismatch for {tensor_name}: {tensor_a.dtype} != {tensor_b.dtype}")

    if not torch.is_floating_point(tensor_a):
        if copy_non_floating:
            return tensor_a.contiguous()
        if not torch.equal(tensor_a, tensor_b):
            raise ValueError(
                f"Non-floating tensor {tensor_name} differs between checkpoints. "
                "Pass --copy-non-floating to copy checkpoint A's value."
            )
        return tensor_a.contiguous()

    return _slerp(tensor_a, tensor_b, alpha, torch).contiguous()


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
    output_dir: Path, output_model_dir: Path, args: argparse.Namespace, inputs: MergeInputs
) -> None:
    metadata = {
        "checkpoint_a": str(inputs.checkpoint_a),
        "checkpoint_b": str(inputs.checkpoint_b),
        "alpha": args.alpha,
        "layout": args.layout,
        "step_name": args.step_name if args.layout == "experiment" else None,
        "max_shard_size_gb": args.max_shard_size_gb,
        "output_model_dir": str(output_model_dir),
    }
    for target_dir in {output_dir, output_model_dir}:
        with (target_dir / "slerp_merge_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")


def _create_experiment_links(output_dir: Path, step_name: str) -> None:
    for link_name in ("LATEST", "LOWEST_VAL"):
        link_path = output_dir / link_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(step_name, link_path)


def _merge_checkpoints(args: argparse.Namespace, inputs: MergeInputs, tensor_names: list[str]) -> None:
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}")
    if args.max_shard_size_gb <= 0:
        raise ValueError(f"--max-shard-size-gb must be positive, got {args.max_shard_size_gb}")

    torch, safe_open, save_file = _load_runtime_modules()
    output_model_dir = _resolve_output_model_dir(args.output_dir, args.layout, args.step_name)
    _prepare_output_dir(args.output_dir, args.overwrite)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    _copy_hf_metadata_files(inputs.checkpoint_a, output_model_dir)

    max_shard_bytes = int(args.max_shard_size_gb * (1024**3))
    weight_map: dict[str, str] = {}
    shard_state = ShardState(tensors={}, size_bytes=0, shard_index=1)
    total_size = 0

    logger.info("Merging %d tensors with alpha=%s", len(tensor_names), args.alpha)
    for index, tensor_name in enumerate(tensor_names, start=1):
        merged = _merge_tensor(tensor_name, inputs, args.alpha, args.copy_non_floating, torch, safe_open)
        tensor_size = _tensor_nbytes(merged)
        if shard_state.tensors and shard_state.size_bytes + tensor_size > max_shard_bytes:
            _save_current_shard(shard_state, output_model_dir, weight_map, save_file)
        shard_state.tensors[tensor_name] = merged
        shard_state.size_bytes += tensor_size
        total_size += tensor_size

        if index % 25 == 0 or index == len(tensor_names):
            logger.info("Merged %d/%d tensors", index, len(tensor_names))

    _save_current_shard(shard_state, output_model_dir, weight_map, save_file)
    shard_count = shard_state.shard_index - 1
    _finalize_shard_names(output_model_dir, weight_map, shard_count)
    _write_index(output_model_dir, weight_map, total_size)
    _write_merge_metadata(args.output_dir, output_model_dir, args, inputs)

    if args.layout == "experiment":
        _create_experiment_links(args.output_dir, args.step_name)

    logger.info("Wrote SLERP checkpoint to %s", output_model_dir)
    logger.info("Output shards: %d; total tensor bytes: %d", shard_count, total_size)


def main() -> None:
    """Run SLERP checkpoint merging from command-line arguments."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    checkpoint_a = _resolve_consolidated_dir(args.checkpoint_a)
    checkpoint_b = _resolve_consolidated_dir(args.checkpoint_b)
    index_a = _read_index(checkpoint_a)
    index_b = _read_index(checkpoint_b)
    tensor_names = _validate_indexes(index_a, index_b)
    inputs = MergeInputs(checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b, index_a=index_a, index_b=index_b)

    logger.info("Resolved checkpoint A: %s", checkpoint_a)
    logger.info("Resolved checkpoint B: %s", checkpoint_b)
    logger.info("Matching tensor keys: %d", len(tensor_names))
    if args.dry_run:
        output_model_dir = _resolve_output_model_dir(args.output_dir, args.layout, args.step_name)
        logger.info("Dry run only. Would write model to: %s", output_model_dir)
        return

    _merge_checkpoints(args, inputs, tensor_names)


if __name__ == "__main__":
    main()
