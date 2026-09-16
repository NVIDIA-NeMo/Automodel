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

"""Internal packed reference checkpoint I/O, independent of LoRA preparation.

The QAT bridge imports this module only for export/load. These implementation
contracts are not package re-exports or additional user-facing entry points.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from itertools import chain
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components.quantization.weight_qat import QuantizedWeight, WeightQuantizationConfig
from nemo_automodel.shared.import_utils import safe_import

HAS_SAFETENSORS, _safetensors = safe_import("safetensors.torch")
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
_FORMAT = "nemo_automodel.lora_qat"
_MODEL_FILE = "model.safetensors"
_QUANTIZED_FILE = "quantized.safetensors"
_MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class QuantizedProjection:
    """A packed weight and its original model storage metadata.

    Args:
        weight_key: Original model state-dict key.
        dtype: Original float32, bfloat16, or float16 reconstruction dtype.
        layout: Original storage order: [out, in] or [experts, in, out].
        encoded: QuantizedWeight in canonical [out, in] or [experts, out, in]
            order. Its payload is uint8 with that shape for FP8, or the final
            dimension halved for MXFP4. Scales have shape [..., out/block_rows,
            in/block_cols], as uint8 E8M0 bytes or float32 multipliers according
            to encoded.config.scale_format. Construction borrows CPU or CUDA
            tensors without copying or mutating them. The QAT bridge supplies
            independent detached CPU snapshots at each projection boundary;
            saving still detaches and clones arbitrary callers' tensors into
            contiguous CPU storage.
    """

    weight_key: str
    dtype: torch.dtype
    layout: Literal["out_in", "experts_in_out"]
    encoded: QuantizedWeight


@dataclass(frozen=True)
class _WeightMetadata:
    """JSON description of one canonical packed weight, with no tensor storage.

    Args:
        weight_key: Original model state-dict key.
        dtype: Floating reconstruction dtype from the fixed safe dtype mapping.
        shape: Canonical [out, in] or [experts, out, in] logical shape.
        layout: Original storage axis order, out_in or experts_in_out.
        config: Resolved numerical configuration, including block dimensions.
        payload: Key in quantized.safetensors: uint8 [out, in] or
            [experts, out, in] for FP8; final dimension halved for packed MXFP4.
        scales: Key in quantized.safetensors: [..., out/block_rows,
            in/block_cols], uint8 E8M0 bytes or float32 multipliers according
            to config.scale_format.
    """

    weight_key: str
    dtype: str
    shape: tuple[int, ...]
    layout: Literal["out_in", "experts_in_out"]
    config: WeightQuantizationConfig
    payload: str
    scales: str


@dataclass(frozen=True)
class _Manifest:
    """Versioned fixed-file checkpoint metadata; hashes detect corruption, not authenticity."""

    schema_version: int
    format: str
    model_file: str
    quantized_file: str
    model_sha256: str
    quantized_sha256: str
    weights: tuple[_WeightMetadata, ...]


def _sha256(path: Path) -> str:
    """Hash a file in bounded memory without deserializing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(directory: Path, filename: str) -> Path:
    """Keep fixed checkpoint files inside the directory, including symlink resolution."""
    path = directory / filename
    if path.resolve().parent != directory.resolve():
        raise ValueError(f"checkpoint file escapes directory: {filename}")
    return path


def _json_fields(value: object, expected: set[str], label: str) -> dict[str, object]:
    """Check exact fields at the untrusted JSON boundary."""
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label}: missing or unknown metadata fields; expected {sorted(expected)}")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON object keys rather than silently taking the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON metadata key: {key}")
        result[key] = value
    return result


def _read_manifest(directory: Path) -> _Manifest:
    """Parse exact metadata fields, safe dtypes, canonical layouts, and fixed filenames."""
    raw = json.loads(_checkpoint_path(directory, _MANIFEST_FILE).read_text(), object_pairs_hook=_unique_json_object)
    raw = _json_fields(raw, {field.name for field in fields(_Manifest)}, "manifest")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1 or raw["format"] != _FORMAT:
        raise ValueError("unsupported checkpoint format or schema_version")
    if raw["model_file"] != _MODEL_FILE or raw["quantized_file"] != _QUANTIZED_FILE:
        raise ValueError("checkpoint filenames must be model.safetensors and quantized.safetensors")
    for key in ("model_sha256", "quantized_sha256"):
        digest = raw[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid {key}")
    if not isinstance(raw["weights"], list):
        raise ValueError("weights metadata must be a list")
    weights = []
    seen = set()
    for entry in raw["weights"]:
        entry = _json_fields(entry, {field.name for field in fields(_WeightMetadata)}, "weight")
        key = entry["weight_key"]
        if (
            not isinstance(key, str)
            or any(c in key for c in ("/", "\\", "\0"))
            or any(not part for part in key.split("."))
            or key in seen
        ):
            raise ValueError("invalid or duplicate weight_key")
        seen.add(key)
        if entry["payload"] != key + ".payload" or entry["scales"] != key + ".scales":
            raise ValueError(f"{key}: invalid payload/scales names")
        if not isinstance(entry["dtype"], str) or entry["dtype"] not in _DTYPES:
            raise ValueError(f"{key}: unsupported dtype")
        shape = entry["shape"]
        if not isinstance(shape, list) or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError(f"{key}: invalid canonical shape")
        if entry["layout"] == "out_in":
            valid = len(shape) == 2 and key.split(".")[-1] == "weight"
        elif entry["layout"] == "experts_in_out":
            valid = len(shape) == 3 and key.split(".")[-1] in ("gate_and_up_projs", "down_projs")
        else:
            valid = False
        if not valid:
            raise ValueError(f"{key}: invalid shape/layout/weight key combination")
        # Version 1 originally omitted scale_format; absence means E8M0.
        # Only this optional field is allowed, not arbitrary future metadata.
        config_fields = {"format", "block_size"}
        if isinstance(entry["config"], dict) and "scale_format" in entry["config"]:
            config_fields.add("scale_format")
        config = _json_fields(entry["config"], config_fields, "weight config")
        if not isinstance(config["block_size"], list):
            raise ValueError(f"{key}: block_size must be an explicit pair")
        weight_config = WeightQuantizationConfig(
            format=config["format"],
            block_size=tuple(config["block_size"]),
            scale_format=config.get("scale_format", "e8m0"),
        )
        weights.append(
            _WeightMetadata(
                weight_key=key,
                dtype=entry["dtype"],
                shape=tuple(shape),
                layout=entry["layout"],
                config=weight_config,
                payload=entry["payload"],
                scales=entry["scales"],
            )
        )
    return _Manifest(
        1, _FORMAT, _MODEL_FILE, _QUANTIZED_FILE, raw["model_sha256"], raw["quantized_sha256"], tuple(weights)
    )


class QATCheckpoint:
    """Own internal reference checkpoint serialization and validated restoration."""

    @staticmethod
    def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
        """Get complete model state without copying or materializing tensors.

        Args:
            model: Local materialized model; non-tensor extra state and DTensor
                parameters, buffers, or state-dict values are rejected.

        Returns:
            Borrowed state tensors in each original parameter/buffer's semantic
            layout: dense weights [out, in], grouped weights [experts, in, out],
            and arbitrary model-defined dimensions for other state. Neither the
            model nor tensor storage is mutated.
        """
        for name, tensor in chain(model.named_parameters(), model.named_buffers()):
            if isinstance(tensor, DTensor):
                raise TypeError(f"{name}: LoRA QAT preparation/export/load requires local tensors, not DTensor")
        state = model.state_dict()
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{key}: non-tensor model state cannot be exported/loaded")
            if isinstance(value, DTensor):
                raise TypeError(f"{key}: DTensor checkpoint state is unsupported")
            if value.is_meta:
                raise ValueError(f"{key}: checkpoint requires materialized, non-meta tensors")
        return state

    @staticmethod
    def save(state: dict[str, torch.Tensor], quantized: list[QuantizedProjection], directory: str | Path) -> None:
        """Write completed state and packed projections using the version-one schema.

        Args:
            state: Ordinary local materialized tensors in original model layout,
                with arbitrary ranks and strides. Dense weights are [out, in];
                grouped weights are [experts, in, out]. Adapter and quantized
                weight keys must already be removed. Values are detached and
                cloned into contiguous CPU storage; inputs are never mutated.
            quantized: Packed weights with the tensor layouts and ownership
                documented on QuantizedProjection. Payloads/scales are detached
                and cloned into contiguous CPU storage; configuration and shape
                are taken directly from each encoded weight.
            directory: Destination for the three fixed files. Both tensor files
                are written even when their corresponding state is empty.
        """
        packed = {}
        metadata = []
        for projection in quantized:
            key, encoded = projection.weight_key, projection.encoded
            payload, scales = key + ".payload", key + ".scales"
            packed[payload] = encoded.payload.detach().cpu().contiguous().clone()
            packed[scales] = encoded.scales.detach().cpu().contiguous().clone()
            metadata.append(
                _WeightMetadata(
                    key,
                    str(projection.dtype).removeprefix("torch."),
                    encoded.shape,
                    projection.layout,
                    encoded.config,
                    payload,
                    scales,
                )
            )
        ordinary = {key: value.detach().cpu().contiguous().clone() for key, value in state.items()}
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = _checkpoint_path(directory, _MODEL_FILE)
        quantized_path = _checkpoint_path(directory, _QUANTIZED_FILE)
        manifest_path = _checkpoint_path(directory, _MANIFEST_FILE)
        _safetensors.save_file(ordinary, str(model_path))
        _safetensors.save_file(packed, str(quantized_path))
        manifest = _Manifest(
            1,
            _FORMAT,
            _MODEL_FILE,
            _QUANTIZED_FILE,
            _sha256(model_path),
            _sha256(quantized_path),
            tuple(sorted(metadata, key=lambda entry: entry.weight_key)),
        )
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")

    @staticmethod
    @torch.no_grad()
    def load(model: nn.Module, directory: str | Path) -> None:
        """Validate and decode a checkpoint before strict state-dict restoration.

        Args:
            model: Fresh materialized local non-LoRA architecture, as validated
                by the QAT bridge. Keys, shapes and dtypes must match. Canonical
                [out, in] weights load directly; [experts, out, in] weights
                transpose to [experts, in, out]. Other state keeps its original
                arbitrary layout. All contracts are checked before mutation.
            directory: Directory containing the three fixed checkpoint files.
        """
        expected = QATCheckpoint.model_state(model)
        directory = Path(directory)
        manifest = _read_manifest(directory)
        for filename, digest in (
            (manifest.model_file, manifest.model_sha256),
            (manifest.quantized_file, manifest.quantized_sha256),
        ):
            if _sha256(_checkpoint_path(directory, filename)) != digest:
                raise ValueError(f"{filename}: SHA256 mismatch")
        state = _safetensors.load_file(str(_checkpoint_path(directory, _MODEL_FILE)), device="cpu")
        packed = _safetensors.load_file(str(_checkpoint_path(directory, _QUANTIZED_FILE)), device="cpu")
        if set(packed) != {key for entry in manifest.weights for key in (entry.payload, entry.scales)}:
            raise ValueError("quantized tensor keys do not match manifest")
        for entry in manifest.weights:
            if entry.weight_key in state:
                raise ValueError(f"{entry.weight_key}: weight occurs in both ordinary and quantized state")
            encoded = QuantizedWeight(packed[entry.payload], packed[entry.scales], entry.config, entry.shape)
            decoded = encoded.dequantize(dtype=_DTYPES[entry.dtype])
            state[entry.weight_key] = decoded if entry.layout == "out_in" else decoded.transpose(-2, -1).contiguous()
        if set(state) != set(expected):
            raise ValueError(
                f"checkpoint state keys mismatch: missing={set(expected) - set(state)}, "
                f"extra={set(state) - set(expected)}"
            )
        for key, tensor in expected.items():
            if tensor.shape != state[key].shape or tensor.dtype != state[key].dtype:
                raise ValueError(f"{key}: checkpoint shape/dtype does not match original model")
        aliases = {}
        for key, tensor in model.state_dict(keep_vars=True).items():
            if id(tensor) in aliases and not torch.equal(state[key], state[aliases[id(tensor)]]):
                raise ValueError(f"{key}: conflicting checkpoint values for tied weights")
            aliases[id(tensor)] = key
        model.load_state_dict(state, strict=True)
