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

"""Offline cache encoder for Qwen/Qwen-Image-Edit-2511."""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch

from nemo_automodel.shared.import_utils import safe_import

NUMPY_AVAILABLE, np = safe_import(
    "numpy",
    msg="Qwen image-edit preprocessing requires NumPy from the diffusion media dependencies",
)
PIL_AVAILABLE, Image = safe_import(
    "PIL.Image",
    msg="Qwen image-edit preprocessing requires Pillow from the diffusion media dependencies",
)
# diffusers is imported lazily in _load_pipeline: importing it initializes the
# CUDA driver, and this module is pulled in by the tools.diffusion.processors
# package, whose importers (e.g. the preprocessing CLI parent process) must
# stay CUDA-free so their multiprocessing workers can initialize CUDA.

logger = logging.getLogger(__name__)

_CONDITION_IMAGE_AREA = 384 * 384
_METADATA_SHARD_SIZE = 1_000
_RESOLUTION_PRESETS = {
    "256p": 256,
    "512p": 512,
    "768p": 768,
    "1024p": 1024,
    "1536p": 1536,
}
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
_DEFAULT_MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"


class _CompoundBucketSignature(TypedDict):
    target: list[int]
    contexts: list[list[int]]


class _CacheRecord(TypedDict):
    cache_file: str
    compound_bucket_signature: _CompoundBucketSignature
    original_id: str
    row_index: int
    target_token_length: int
    context_token_lengths: list[int]
    text_token_length: int


@dataclass(frozen=True)
class _MediaReference:
    role: str
    file_name: str
    path: Path


@dataclass(frozen=True)
class _ManifestSample:
    identifier: str
    prompt: str
    media: tuple[_MediaReference, ...]
    metadata: dict[str, object]
    row_index: int


@dataclass(frozen=True)
class _WorkerSettings:
    model_name: str
    max_sequence_length: int
    device: str | None
    torch_dtype: str


class QwenImageEditCacheEncoder:
    """Encode generic image-edit manifests into the cached contract.

    The encoder loads only the upstream Diffusers VAE and Qwen2.5-VL
    conditioning stack. The trainable transformer is explicitly omitted.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_sequence_length: int = 512,
        device: str | None = None,
        torch_dtype: str = "bfloat16",
    ) -> None:
        """Configure offline Qwen image-edit encoding.

        Args:
            model_name: Hugging Face model ID or local Diffusers pipeline path.
                ``None`` selects ``Qwen/Qwen-Image-Edit-2511`` so the generic
                preprocessing CLI can leave ``--model_name`` unset.
            max_sequence_length: Maximum cached Qwen2.5-VL token count.
            device: Optional explicit encoding device. By default each worker
                selects its assigned CUDA device, with CPU as a unit-test fallback.
            torch_dtype: VAE/text-encoder compute and cache dtype. Supported
                values are ``bfloat16``, ``float16``, and ``float32``.
        """
        model_name = _DEFAULT_MODEL_NAME if model_name is None else model_name
        if not model_name:
            raise ValueError("model_name must be non-empty when explicitly provided")
        if max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be positive, got {max_sequence_length}")
        if torch_dtype not in _DTYPES:
            raise ValueError(f"torch_dtype must be one of {sorted(_DTYPES)}, got {torch_dtype!r}")

        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.torch_dtype = torch_dtype

    def encode_manifest(
        self,
        *,
        manifest_path: Path,
        output_dir: Path,
        max_pixels: int,
        resolution_preset: str | None,
        num_gpus: int,
        verify: bool,
    ) -> Path:
        """Encode a generic manifest into cached target/context/text tensors.

        Args:
            manifest_path: JSONL manifest whose rows contain an ID, prompt,
                ordered role-to-file media mappings, and provenance metadata.
            output_dir: Destination for cache tensors and indexes. It may contain
                the materialized source tree that owns ``manifest_path`` but no
                prior cache artifacts.
            max_pixels: Maximum target/context pixel area when no fixed preset
                is selected.
            resolution_preset: Optional fixed square benchmark preset such as
                ``"1024p"``. When set, target and context images are center-
                cropped to that exact spatial shape.
            num_gpus: Number of independent encoder workers. Each worker owns
                one CUDA device and a private VAE/text encoder.
            verify: Whether to decode each target latent and validate its finite
                RGB tensor output.

        Returns:
            Path to ``metadata.json``, whose shards index target tensors of
            shape [channels, target_height, target_width], ordered context
            tensors of shape [channels, context_height, context_width], prompt
            embeddings of shape [sequence, hidden], and prompt masks of shape
            [sequence].
        """
        manifest_path = Path(manifest_path).resolve()
        output_dir = Path(output_dir).resolve()
        if not NUMPY_AVAILABLE or not PIL_AVAILABLE:
            raise ImportError("Qwen image-edit preprocessing requires NumPy and Pillow")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Image-edit manifest does not exist: {manifest_path}")
        if max_pixels < 32 * 32:
            raise ValueError(f"max_pixels must be at least {32 * 32} for 32-aligned Qwen images, got {max_pixels}")
        if resolution_preset is not None and resolution_preset not in _RESOLUTION_PRESETS:
            raise ValueError(
                f"resolution_preset must be one of {sorted(_RESOLUTION_PRESETS)}, got {resolution_preset!r}"
            )
        if num_gpus <= 0:
            raise ValueError(f"num_gpus must be positive, got {num_gpus}")
        if self.device is not None and num_gpus != 1:
            raise ValueError("An explicit device can only be used with num_gpus=1")

        samples = _read_manifest(manifest_path)
        dataset_name = _shared_metadata_string(samples, "dataset_name")
        dataset_config_name = _shared_metadata_optional_string(samples, "dataset_config_name")
        dataset_split = _shared_metadata_string(samples, "dataset_split")

        output_dir.mkdir(parents=True, exist_ok=True)
        _validate_output_directory(output_dir, manifest_path=manifest_path)
        (output_dir / "samples").mkdir()

        settings = _WorkerSettings(
            model_name=self.model_name,
            max_sequence_length=self.max_sequence_length,
            device=self.device,
            torch_dtype=self.torch_dtype,
        )
        if num_gpus == 1:
            records = self._encode_rows(
                samples,
                output_dir=output_dir,
                max_pixels=max_pixels,
                resolution_preset=resolution_preset,
                worker_index=0,
                verify=verify,
            )
        else:
            if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
                raise RuntimeError(
                    f"Requested {num_gpus} GPU preprocessing workers, but {torch.cuda.device_count()} CUDA devices "
                    "are available"
                )
            records = _encode_rows_multiprocess(
                settings,
                samples,
                output_dir=output_dir,
                max_pixels=max_pixels,
                resolution_preset=resolution_preset,
                num_workers=num_gpus,
                verify=verify,
            )

        records.sort(key=lambda record: record["row_index"])
        shard_names = _write_metadata_shards(records, output_dir)
        metadata = {
            "dataset_name": dataset_name,
            "dataset_config_name": dataset_config_name,
            "split": dataset_split,
            "row_limit": len(samples),
            "preprocessing_config": {
                "processor_target": (
                    "nemo_automodel.components.models.qwen_image_edit.preprocessing.QwenImageEditCacheEncoder"
                ),
                "model_name": self.model_name,
                "max_sequence_length": self.max_sequence_length,
                "max_pixels": max_pixels,
                "resolution_preset": resolution_preset,
                "resize_mode": (
                    "fixed_square_center_crop" if resolution_preset is not None else "aspect_preserving_max_pixels"
                ),
                "spatial_alignment": 32,
                "condition_image_pixels": _CONDITION_IMAGE_AREA,
                "vae_latent_sampling": "mode",
                "torch_dtype": self.torch_dtype,
                "num_gpus": num_gpus,
                "verify": verify,
            },
            "shards": shard_names,
        }
        metadata_path = output_dir / "metadata.json"
        _write_json_atomic(metadata_path, metadata)
        return metadata_path

    def _encode_rows(
        self,
        samples: list[_ManifestSample],
        *,
        output_dir: Path,
        max_pixels: int,
        resolution_preset: str | None,
        worker_index: int,
        verify: bool,
    ) -> list[_CacheRecord]:
        """Encode one worker's deterministic manifest shard.

        Args:
            samples: Validated manifest rows referencing local RGB images.
            output_dir: Cache root shared by workers; sample filenames are
                derived from globally unique row indices.
            max_pixels: Maximum target/context pixel area without a preset.
            resolution_preset: Optional fixed square spatial preset.
            worker_index: CUDA device index owned by this worker.
            verify: Whether to decode and validate each target latent tensor of
                shape [channels, target_height, target_width].

        Returns:
            Metadata records describing every emitted target/context tensor
            shape and prompt token count.
        """
        device = self._worker_device(worker_index)
        pipeline = self._load_pipeline(device)
        records = []
        try:
            for sample in samples:
                records.append(
                    self._encode_sample(
                        pipeline,
                        sample,
                        output_dir=output_dir,
                        max_pixels=max_pixels,
                        resolution_preset=resolution_preset,
                        device=device,
                        verify=verify,
                    )
                )
        finally:
            pipeline.vae.to("cpu")
            pipeline.text_encoder.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return records

    def _worker_device(self, worker_index: int) -> torch.device:
        """Resolve the runtime device owned by an encoder worker."""
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda", worker_index)
        return torch.device("cpu")

    def _load_pipeline(self, device: torch.device):
        """Load only upstream VAE and Qwen2.5-VL conditioning components."""
        diffusers_available, diffusers = safe_import(
            "diffusers",
            msg="Qwen image-edit preprocessing requires the diffusion optional dependencies",
        )
        if not diffusers_available:
            raise ImportError("Qwen image-edit preprocessing requires diffusers")

        dtype = _DTYPES[self.torch_dtype]
        pipeline = diffusers.QwenImageEditPlusPipeline.from_pretrained(
            self.model_name,
            transformer=None,
            torch_dtype=dtype,
        )
        pipeline.vae.requires_grad_(False).eval().to(device=device, dtype=dtype)
        pipeline.text_encoder.requires_grad_(False).eval().to(device=device, dtype=dtype)
        return pipeline

    def _encode_sample(
        self,
        pipeline,
        sample: _ManifestSample,
        *,
        output_dir: Path,
        max_pixels: int,
        resolution_preset: str | None,
        device: torch.device,
        verify: bool,
    ) -> _CacheRecord:
        """Encode and persist one instruction-based editing example.

        Args:
            pipeline: Upstream pipeline holding the frozen VAE and Qwen2.5-VL
                conditioning stack.
            sample: Validated target/context/condition media references.
            output_dir: Cache root for the emitted tensor payload.
            max_pixels: Maximum target/context pixel area without a preset.
            resolution_preset: Optional fixed square target/context preset.
            device: Runtime device for VAE and text encoding.
            verify: Whether to decode the target latent tensor of shape
                [channels, target_height, target_width].

        Returns:
            Index record containing the target/context tensor layouts and
            token counts for the emitted cache file.
        """
        images = _load_unique_images(sample.media)
        target_reference = next(reference for reference in sample.media if reference.role == "target")
        context_references = [reference for reference in sample.media if reference.role == "context"]
        condition_references = [reference for reference in sample.media if reference.role == "condition"]

        original_target_image = images[target_reference.file_name]
        original_context_images = [images[reference.file_name] for reference in context_references]
        original_condition_images = [images[reference.file_name] for reference in condition_references]
        target_image = _resize_vae_image(
            original_target_image,
            max_pixels=max_pixels,
            resolution_preset=resolution_preset,
        )
        context_images = [
            _resize_vae_image(
                image,
                max_pixels=max_pixels,
                resolution_preset=resolution_preset,
            )
            for image in original_context_images
        ]
        condition_images = [_resize_condition_image(image) for image in original_condition_images]

        dtype = _DTYPES[self.torch_dtype]
        with torch.no_grad():
            target_latent = _encode_vae_image(pipeline.vae, target_image, device=device, storage_dtype=dtype)
            context_latents = [
                _encode_vae_image(pipeline.vae, image, device=device, storage_dtype=dtype) for image in context_images
            ]
            prompt_embeddings, prompt_attention_mask = pipeline.encode_prompt(
                prompt=[sample.prompt],
                image=condition_images,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=self.max_sequence_length,
            )

        prompt_embeddings = prompt_embeddings[:, : self.max_sequence_length]
        if prompt_embeddings.shape[1] == 0:
            raise ValueError(f"Qwen2.5-VL produced no prompt tokens for manifest row {sample.row_index}")
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones(
                prompt_embeddings.shape[:2],
                dtype=torch.long,
                device=prompt_embeddings.device,
            )
        else:
            prompt_attention_mask = prompt_attention_mask[:, : self.max_sequence_length]

        prompt_embeddings = prompt_embeddings[0].detach().to(device="cpu", dtype=dtype)
        prompt_attention_mask = prompt_attention_mask[0].detach().to(device="cpu", dtype=torch.long)
        if verify:
            _verify_vae_latent(pipeline.vae, target_latent, device=device)

        target_shape = list(target_latent.shape)
        context_shapes = [list(latent.shape) for latent in context_latents]
        signature: _CompoundBucketSignature = {
            "target": target_shape,
            "contexts": context_shapes,
        }
        target_token_length = (target_shape[-2] // 2) * (target_shape[-1] // 2)
        context_token_lengths = [(shape[-2] // 2) * (shape[-1] // 2) for shape in context_shapes]
        text_token_length = int(prompt_attention_mask.sum().item())
        metadata = {
            "original_ids": {
                "id": sample.identifier,
                "row_index": sample.row_index,
            },
            "target_spatial_shape": target_shape[-2:],
            "context_spatial_shapes": [shape[-2:] for shape in context_shapes],
            "target_token_length": target_token_length,
            "context_token_lengths": context_token_lengths,
            "text_token_length": text_token_length,
            "compound_bucket_signature": signature,
            "conditioning_shapes": {},
            "manifest_metadata": sample.metadata,
            "media": [{"role": reference.role, "file_name": reference.file_name} for reference in sample.media],
            "original_pixel_shapes": {
                "target": [original_target_image.height, original_target_image.width],
                "contexts": [[image.height, image.width] for image in original_context_images],
                "conditions": [[image.height, image.width] for image in original_condition_images],
            },
            "preprocessed_pixel_shapes": {
                "target": [target_image.height, target_image.width],
                "contexts": [[image.height, image.width] for image in context_images],
                "conditions": [[image.height, image.width] for image in condition_images],
            },
        }
        payload = {
            "target_latent": target_latent,
            "context_latents": context_latents,
            "prompt_embeddings": prompt_embeddings,
            "prompt_attention_mask": prompt_attention_mask,
            "conditioning_tensors": {},
            "metadata": metadata,
        }

        relative_cache_file = Path("samples") / f"sample_{sample.row_index:08d}.pt"
        cache_file = output_dir / relative_cache_file
        temporary_file = cache_file.with_suffix(".pt.tmp")
        torch.save(payload, temporary_file)
        temporary_file.replace(cache_file)
        return {
            "cache_file": relative_cache_file.as_posix(),
            "compound_bucket_signature": signature,
            "original_id": sample.identifier,
            "row_index": sample.row_index,
            "target_token_length": target_token_length,
            "context_token_lengths": context_token_lengths,
            "text_token_length": text_token_length,
        }


def _read_manifest(manifest_path: Path) -> list[_ManifestSample]:
    """Read and validate every generic image-edit JSONL row."""
    root = manifest_path.parent.resolve()
    samples = []
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path} line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest line {line_number} must contain a JSON object")

            identifier = row.get("id")
            prompt = row.get("prompt")
            raw_media = row.get("media")
            metadata = row.get("metadata")
            if not isinstance(identifier, (str, int)) or isinstance(identifier, bool):
                raise ValueError(f"Manifest line {line_number} id must be a string or integer")
            if not isinstance(prompt, str):
                raise ValueError(f"Manifest line {line_number} prompt must be a string")
            if not isinstance(raw_media, list):
                raise ValueError(f"Manifest line {line_number} media must be an ordered list")
            if not isinstance(metadata, dict):
                raise ValueError(f"Manifest line {line_number} metadata must be a mapping")

            media = []
            for media_index, item in enumerate(raw_media):
                if not isinstance(item, dict):
                    raise ValueError(f"Manifest line {line_number} media[{media_index}] must be a mapping")
                role = item.get("role")
                file_name = item.get("file_name")
                if role not in {"target", "context", "condition"}:
                    raise ValueError(f"Manifest line {line_number} media[{media_index}] has unsupported role {role!r}")
                if not isinstance(file_name, str) or not file_name:
                    raise ValueError(f"Manifest line {line_number} media[{media_index}].file_name must be non-empty")
                relative_path = Path(file_name)
                if relative_path.is_absolute():
                    raise ValueError(
                        f"Manifest line {line_number} media[{media_index}] must use a manifest-relative path"
                    )
                resolved_path = (root / relative_path).resolve()
                try:
                    resolved_path.relative_to(root)
                except ValueError as exc:
                    raise ValueError(
                        f"Manifest line {line_number} media[{media_index}] resolves outside {root}"
                    ) from exc
                if not resolved_path.is_file():
                    raise FileNotFoundError(
                        f"Manifest line {line_number} media[{media_index}] does not exist: {resolved_path}"
                    )
                media.append(_MediaReference(role=role, file_name=file_name, path=resolved_path))

            target_count = sum(reference.role == "target" for reference in media)
            context_count = sum(reference.role == "context" for reference in media)
            condition_count = sum(reference.role == "condition" for reference in media)
            if target_count != 1 or context_count == 0 or condition_count == 0:
                raise ValueError(
                    f"Manifest line {line_number} requires exactly one target, at least one context, and at least "
                    f"one condition; got target={target_count}, context={context_count}, condition={condition_count}"
                )
            row_index_value = metadata.get("row_index", len(samples))
            if isinstance(row_index_value, bool) or not isinstance(row_index_value, int) or row_index_value < 0:
                raise ValueError(f"Manifest line {line_number} metadata.row_index must be a non-negative integer")
            samples.append(
                _ManifestSample(
                    identifier=str(identifier),
                    prompt=prompt,
                    media=tuple(media),
                    metadata=metadata,
                    row_index=row_index_value,
                )
            )

    if not samples:
        raise ValueError(f"Image-edit manifest contains no samples: {manifest_path}")
    row_indices = [sample.row_index for sample in samples]
    if len(set(row_indices)) != len(row_indices):
        raise ValueError("Image-edit manifest metadata.row_index values must be unique")
    return samples


def _shared_metadata_string(samples: list[_ManifestSample], field_name: str) -> str:
    """Return a required provenance string shared by all manifest rows."""
    values = [sample.metadata.get(field_name) for sample in samples]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"Manifest metadata.{field_name} must be a non-empty string")
    first_value = values[0]
    if any(value != first_value for value in values[1:]):
        raise ValueError(f"Manifest rows must share one {field_name}, got {values!r}")
    return cast(str, first_value)


def _shared_metadata_optional_string(samples: list[_ManifestSample], field_name: str) -> str | None:
    """Return an optional provenance string shared by all manifest rows."""
    values = [sample.metadata.get(field_name) for sample in samples]
    if any(value is not None and (not isinstance(value, str) or not value) for value in values):
        raise ValueError(f"Manifest metadata.{field_name} must be a non-empty string or null")
    first_value = values[0]
    if any(value != first_value for value in values[1:]):
        raise ValueError(f"Manifest rows must share one {field_name}, got {values!r}")
    return cast(str | None, first_value)


def _validate_output_directory(output_dir: Path, *, manifest_path: Path) -> None:
    """Require a fresh cache while allowing CLI-owned materialized input.

    The image-edit CLI may place its materialized source manifest inside the
    requested output directory before invoking the configured encoder. That
    source tree is intentionally retained for provenance. Cache-owned paths
    must not exist so encoding never overwrites prior outputs.
    """
    owned_paths = [output_dir / "metadata.json", output_dir / "samples"]
    owned_paths.extend(output_dir.glob("metadata_shard_*.json"))
    collisions = sorted(path.name for path in owned_paths if path.exists())

    allowed_source_entries: set[str] = set()
    try:
        relative_manifest = manifest_path.resolve().relative_to(output_dir.resolve())
    except ValueError:
        relative_manifest = None
    if relative_manifest is not None:
        if len(relative_manifest.parts) > 1:
            allowed_source_entries.add(relative_manifest.parts[0])
        else:
            allowed_source_entries.update({relative_manifest.name, "media", "metadata"})

    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in allowed_source_entries)
    if collisions or unexpected:
        names = sorted(set(collisions + unexpected))
        raise ValueError(
            f"Image-edit cache output directory contains existing cache artifacts: {output_dir} ({', '.join(names)})"
        )


def _load_unique_images(media: tuple[_MediaReference, ...]) -> dict[str, Image.Image]:
    """Load each manifest-relative media file once as an RGB image."""
    images = {}
    for reference in media:
        if reference.file_name not in images:
            with Image.open(reference.path) as image:
                images[reference.file_name] = image.convert("RGB")
    return images


def _resize_vae_image(
    image: Image.Image,
    *,
    max_pixels: int,
    resolution_preset: str | None,
) -> Image.Image:
    """Resize an RGB image for a cached Qwen VAE latent.

    Args:
        image: RGB image in channels-last pixel layout [height, width, channels].
        max_pixels: Maximum output pixel area without a preset.
        resolution_preset: Optional fixed square output preset.

    Returns:
        RGB image in channels-last layout [target_height, target_width,
        channels], with both spatial axes divisible by 32.
    """
    if resolution_preset is None:
        return _resize_to_area(image, target_area=max_pixels)

    side = _RESOLUTION_PRESETS[resolution_preset]
    scale = max(side / image.width, side / image.height)
    resized_width = max(side, round(image.width * scale))
    resized_height = max(side, round(image.height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    left = (resized_width - side) // 2
    top = (resized_height - side) // 2
    return resized.crop((left, top, left + side, top + side))


def _resize_condition_image(image: Image.Image) -> Image.Image:
    """Match the upstream Qwen edit vision-conditioning resize.

    Diffusers derives an aspect-preserving shape at the fixed 384-squared area
    and rounds both axes to the nearest multiple of 32. Matching that policy is
    required because the Qwen2.5-VL image grid changes the prompt embeddings.

    Args:
        image: RGB image in channels-last pixel layout [height, width, channels].

    Returns:
        RGB image with the upstream 32-aligned conditioning dimensions.
    """
    ratio = image.width / image.height
    width = max(32, round(math.sqrt(_CONDITION_IMAGE_AREA * ratio) / 32) * 32)
    height = max(32, round(math.sqrt(_CONDITION_IMAGE_AREA / ratio) / 32) * 32)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _resize_to_area(image: Image.Image, *, target_area: int) -> Image.Image:
    """Resize an RGB image to an aspect-preserving, 32-aligned pixel area.

    Args:
        image: RGB image in channels-last pixel layout [height, width, channels].
        target_area: Approximate output pixel count.

    Returns:
        RGB image in channels-last layout [target_height, target_width,
        channels], where each spatial axis is a positive multiple of 32 and
        their product does not exceed ``target_area``.
    """
    if target_area < 32 * 32:
        raise ValueError(f"target_area must be at least {32 * 32}, got {target_area}")

    scale = math.sqrt(target_area / (image.width * image.height))
    width = max(32, math.floor(image.width * scale / 32) * 32)
    height = max(32, math.floor(image.height * scale / 32) * 32)
    while width * height > target_area and (width > 32 or height > 32):
        if width >= height and width > 32:
            width -= 32
        elif height > 32:
            height -= 32
        else:
            break
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _image_to_tensor(image: Image.Image, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert an RGB image to a normalized channels-first tensor.

    Args:
        image: RGB image in channels-last layout [height, width, channels].
        device: Output tensor device.
        dtype: Output floating-point dtype.

    Returns:
        Tensor of shape [1, 3, height, width] in the range [-1, 1].
    """
    array = np.asarray(image, dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype).div_(127.5).sub_(1.0)


def _encode_vae_image(
    vae: Any,
    image: Image.Image,
    *,
    device: torch.device,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    """Encode one deterministic Qwen VAE latent.

    Args:
        vae: Upstream Qwen VAE. Its input layout is [batch, 3, frames,
            height, width], and its latent layout is [batch, channels, frames,
            latent_height, latent_width].
        image: RGB image in channels-last layout [height, width, channels].
        device: VAE execution device.
        storage_dtype: Floating-point dtype for the returned cache tensor.

    Returns:
        Normalized latent mode tensor of shape [channels, latent_height,
        latent_width] on CPU.
    """
    image_tensor = _image_to_tensor(image, device=device, dtype=storage_dtype).unsqueeze(2)
    latent = vae.encode(image_tensor).latent_dist.mode()
    latent_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=latent.dtype).reshape(
        1, vae.config.z_dim, 1, 1, 1
    )
    latent_std = torch.tensor(vae.config.latents_std, device=device, dtype=latent.dtype).reshape(
        1, vae.config.z_dim, 1, 1, 1
    )
    normalized = (latent - latent_mean) / latent_std
    return normalized[0, :, 0].detach().to(device="cpu", dtype=storage_dtype)


def _verify_vae_latent(vae: Any, latent: torch.Tensor, *, device: torch.device) -> None:
    """Decode a cached latent and require a finite RGB result.

    Args:
        vae: Upstream Qwen VAE whose decoded tensor layout is [batch, 3,
            frames, height, width].
        latent: Normalized cache tensor of shape [channels, latent_height,
            latent_width].
        device: VAE execution device.
    """
    vae_dtype = next(vae.parameters()).dtype
    latent_batch = latent.to(device=device, dtype=vae_dtype).unsqueeze(0).unsqueeze(2)
    latent_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae_dtype).reshape(
        1, vae.config.z_dim, 1, 1, 1
    )
    latent_std = torch.tensor(vae.config.latents_std, device=device, dtype=vae_dtype).reshape(
        1, vae.config.z_dim, 1, 1, 1
    )
    decoded = vae.decode(latent_batch * latent_std + latent_mean, return_dict=False)[0]
    if decoded.ndim != 5 or decoded.shape[1] != 3 or not torch.isfinite(decoded).all():
        raise ValueError(
            "Qwen VAE verification must produce a finite tensor with shape [batch, 3, frames, height, width], "
            f"got {tuple(decoded.shape)}"
        )


def _encode_rows_worker(
    settings: _WorkerSettings,
    samples: list[_ManifestSample],
    output_dir: Path,
    max_pixels: int,
    resolution_preset: str | None,
    worker_index: int,
    verify: bool,
) -> list[_CacheRecord]:
    """Construct a private encoder and run one spawned GPU shard."""
    encoder = QwenImageEditCacheEncoder(
        model_name=settings.model_name,
        max_sequence_length=settings.max_sequence_length,
        device=settings.device,
        torch_dtype=settings.torch_dtype,
    )
    return encoder._encode_rows(
        samples,
        output_dir=output_dir,
        max_pixels=max_pixels,
        resolution_preset=resolution_preset,
        worker_index=worker_index,
        verify=verify,
    )


def _encode_rows_multiprocess(
    settings: _WorkerSettings,
    samples: list[_ManifestSample],
    *,
    output_dir: Path,
    max_pixels: int,
    resolution_preset: str | None,
    num_workers: int,
    verify: bool,
) -> list[_CacheRecord]:
    """Encode round-robin manifest shards in spawned GPU workers."""
    shards = [samples[index::num_workers] for index in range(num_workers)]
    records = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as executor:
        futures = [
            executor.submit(
                _encode_rows_worker,
                settings,
                shard,
                output_dir,
                max_pixels,
                resolution_preset,
                worker_index,
                verify,
            )
            for worker_index, shard in enumerate(shards)
            if shard
        ]
        for future in futures:
            try:
                records.extend(future.result())
            except Exception as exc:
                raise RuntimeError("Qwen image-edit preprocessing worker failed") from exc
    return records


def _write_metadata_shards(records: list[_CacheRecord], output_dir: Path) -> list[str]:
    """Write deterministic cache-index shards and return relative names."""
    shard_names = []
    for shard_index, start in enumerate(range(0, len(records), _METADATA_SHARD_SIZE)):
        shard_name = f"metadata_shard_{shard_index:04d}.json"
        _write_json_atomic(output_dir / shard_name, records[start : start + _METADATA_SHARD_SIZE])
        shard_names.append(shard_name)
    return shard_names


def _write_json_atomic(path: Path, value: object) -> None:
    """Write one JSON value through a same-directory temporary file."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)
