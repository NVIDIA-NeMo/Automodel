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

"""Cached, model-agnostic image-edit datasets and dataloaders."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild

from .base_dataset import BaseMultiresolutionDataset
from .sampler import SequentialBucketSampler

logger = logging.getLogger(__name__)


class _CompoundBucketSignatureMetadata(TypedDict):
    target: list[int]
    contexts: list[list[int]]


class _ImageEditCacheMetadata(TypedDict, total=False):
    original_ids: dict[str, str | int]
    target_spatial_shape: list[int]
    context_spatial_shapes: list[list[int]]
    target_token_length: int
    context_token_lengths: list[int]
    text_token_length: int
    compound_bucket_signature: _CompoundBucketSignatureMetadata
    conditioning_shapes: dict[str, list[int]]


class _ImageEditSample(TypedDict):
    target_latent: torch.Tensor
    context_latents: list[torch.Tensor]
    prompt_embeddings: torch.Tensor
    prompt_attention_mask: torch.Tensor
    conditioning_tensors: dict[str, torch.Tensor]
    metadata: _ImageEditCacheMetadata


class _ImageEditBatchMetadata(TypedDict):
    samples: list[_ImageEditCacheMetadata]
    batch_size: int
    target_token_counts: list[int]
    context_token_counts: list[int]
    text_token_counts: list[int]
    total_token_counts: list[int]
    target_token_count: int
    context_token_count: int
    text_token_count: int
    total_token_count: int


class _ImageEditBatch(TypedDict):
    image_latents: torch.Tensor
    context_latents: list[torch.Tensor]
    text_embeddings: torch.Tensor
    text_attention_mask: torch.Tensor
    conditioning_tensors: dict[str, torch.Tensor]
    data_type: str
    metadata: _ImageEditBatchMetadata


@dataclass(frozen=True)
class _CompoundBucketSignature:
    target: tuple[int, int, int]
    contexts: tuple[tuple[int, int, int], ...]


def _normalize_shape(value: object, *, field_name: str, rank: int, allow_zero: bool = False) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != rank:
        raise ValueError(f"{field_name} must contain exactly {rank} dimensions, got {value!r}")

    shape: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise ValueError(f"{field_name} dimensions must be integers, got {value!r}")
        if dimension < 0 or (dimension == 0 and not allow_zero):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{field_name} dimensions must be {qualifier}, got {value!r}")
        shape.append(dimension)
    return tuple(shape)


def _parse_compound_bucket_signature(value: object, *, field_name: str) -> _CompoundBucketSignature:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping with 'target' and 'contexts' entries")

    signature = cast(Mapping[str, object], value)
    target = _normalize_shape(signature.get("target"), field_name=f"{field_name}.target", rank=3)
    raw_contexts = signature.get("contexts")
    if not isinstance(raw_contexts, list) or not raw_contexts:
        raise ValueError(f"{field_name}.contexts must be a non-empty ordered list of [channels, height, width] shapes")
    contexts = tuple(
        cast(
            tuple[int, int, int],
            _normalize_shape(shape, field_name=f"{field_name}.contexts[{index}]", rank=3),
        )
        for index, shape in enumerate(raw_contexts)
    )
    return _CompoundBucketSignature(
        target=cast(tuple[int, int, int], target),
        contexts=contexts,
    )


def _signature_to_metadata(signature: _CompoundBucketSignature) -> _CompoundBucketSignatureMetadata:
    return {
        "target": list(signature.target),
        "contexts": [list(shape) for shape in signature.contexts],
    }


def _nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")
    return value


def _validate_latent(tensor: torch.Tensor, *, field_name: str) -> tuple[int, int, int]:
    """Validate a cached image latent.

    Args:
        tensor: Tensor of shape [channels, height, width].
        field_name: Cache field name used in validation errors.

    Returns:
        Validated tensor shape in [channels, height, width] order.
    """
    if tensor.ndim != 3:
        raise ValueError(f"{field_name} must have shape [channels, height, width], got {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise ValueError(f"{field_name} must use a floating-point dtype, got {tensor.dtype}")
    return cast(tuple[int, int, int], tuple(tensor.shape))


def _require_tensor(data: Mapping[str, object], field_name: str) -> torch.Tensor:
    """Read a tensor field from a cache payload.

    Args:
        data: Cache mapping whose tensor fields may have arbitrary shapes.
        field_name: Name of the tensor field to read.

    Returns:
        Cached tensor with arbitrary shape; the field owner validates its layout.
    """
    value = data.get(field_name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Image-edit cache field '{field_name}' must be a tensor")
    return value


@dataclass
class ImageEditDatasetConfig:
    """Construction-time configuration for :class:`ImageEditDataset`."""

    cache_dir: str
    """Directory containing a preprocessed image-edit cache."""
    quantization: int = 64
    """Spatial quantization used for dynamic batch-size calculation."""

    def build(self) -> ImageEditDataset:
        """Build the configured image-edit dataset.

        Returns:
            Dataset backed by the configured preprocessed cache.
        """
        return ImageEditDataset(
            cache_dir=self.cache_dir,
            quantization=self.quantization,
        )


class ImageEditDataset(BaseMultiresolutionDataset):
    """Dataset for cached instruction-based image editing.

    Each cache payload contains a target latent ``[channels, height, width]``,
    an ordered list of context latents with the same axis order, prompt
    embeddings ``[sequence, hidden]``, an attention mask ``[sequence]``, and
    an optional explicitly named mapping of model-conditioning tensors.
    """

    def __init__(
        self,
        cache_dir: str,
        quantization: int = 64,
    ) -> None:
        """Load and organize a preprocessed image-edit cache.

        Args:
            cache_dir: Directory containing ``metadata.json``, metadata shards,
                and tensor cache files.
            quantization: Spatial quantization used for dynamic batch-size
                calculation.
        """
        if quantization <= 0:
            raise ValueError(f"quantization must be positive, got {quantization}")
        self.cache_manifest: dict[str, object] = {}
        super().__init__(cache_dir=cache_dir, quantization=quantization)

    def _resolve_contained_path(self, path_value: object, *, field_name: str) -> Path:
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"{field_name} must be a non-empty path string")

        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = self.cache_dir / candidate
        resolved = candidate.resolve()
        cache_dir = self.cache_dir.resolve()
        try:
            resolved.relative_to(cache_dir)
        except ValueError as exc:
            raise ValueError(f"{field_name} {resolved} is outside cache directory {cache_dir}") from exc
        return resolved

    def _load_metadata(self) -> list[dict[str, object]]:
        metadata_file = self.cache_dir / "metadata.json"
        if not metadata_file.is_file():
            raise FileNotFoundError(f"No metadata.json found in {self.cache_dir}")

        with metadata_file.open("r", encoding="utf-8") as file:
            raw_manifest = json.load(file)
        if not isinstance(raw_manifest, dict):
            raise ValueError(f"Invalid metadata format in {metadata_file}: expected a mapping")
        manifest = cast(dict[str, object], raw_manifest)

        raw_shards = manifest.get("shards")
        if not isinstance(raw_shards, list) or not all(isinstance(name, str) and name for name in raw_shards):
            raise ValueError(f"Invalid metadata format in {metadata_file}: 'shards' must be a list of file names")

        self.cache_manifest = manifest

        metadata: list[dict[str, object]] = []
        for shard_index, shard_name in enumerate(raw_shards):
            shard_path = self._resolve_contained_path(shard_name, field_name=f"shards[{shard_index}]")
            with shard_path.open("r", encoding="utf-8") as file:
                shard = json.load(file)
            if not isinstance(shard, list):
                raise ValueError(f"Metadata shard {shard_path} must contain a list of sample mappings")

            for item_index, raw_item in enumerate(shard):
                if not isinstance(raw_item, dict):
                    raise ValueError(f"Metadata shard {shard_path} item {item_index} must be a mapping")
                raw_item = cast(dict[str, object], raw_item)
                if "cache_file" not in raw_item:
                    raise ValueError(f"Metadata shard {shard_path} item {item_index} is missing 'cache_file'")
                _parse_compound_bucket_signature(
                    raw_item.get("compound_bucket_signature"),
                    field_name=f"{shard_path.name}[{item_index}].compound_bucket_signature",
                )
                metadata.append(raw_item)

        return metadata

    def _group_by_bucket(self) -> None:
        self.bucket_groups = {}

        for index, item in enumerate(self.metadata):
            signature = _parse_compound_bucket_signature(
                item.get("compound_bucket_signature"),
                field_name=f"metadata[{index}].compound_bucket_signature",
            )
            _, target_height, target_width = signature.target
            resolution = (target_width, target_height)
            aspect_ratio = target_width / target_height
            aspect_name = self._aspect_ratio_to_name(aspect_ratio)
            latent_pixels = target_height * target_width + sum(
                context_height * context_width for _, context_height, context_width in signature.contexts
            )
            if signature not in self.bucket_groups:
                self.bucket_groups[signature] = {
                    "indices": [],
                    "aspect_name": aspect_name,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "pixels": latent_pixels,
                    "compound_bucket_signature": _signature_to_metadata(signature),
                }
            self.bucket_groups[signature]["indices"].append(index)

        self.sorted_bucket_keys = sorted(
            self.bucket_groups,
            key=lambda signature: (self.bucket_groups[signature]["pixels"], signature.target, signature.contexts),
        )
        logger.info("Dataset organized into %d compound image-edit buckets", len(self.bucket_groups))

    def get_bucket_info(self) -> dict[str, object]:
        """Return target/context shape groups used by the bucket sampler.

        Returns:
            Mapping containing the bucket count and sample count per compound
            target/context signature.
        """
        buckets: dict[str, int] = {}
        for signature, group in self.bucket_groups.items():
            target = "x".join(str(dimension) for dimension in signature.target)
            contexts = ",".join("x".join(str(dimension) for dimension in shape) for shape in signature.contexts)
            buckets[f"target={target}/contexts=[{contexts}]"] = len(group["indices"])
        return {"total_buckets": len(self.bucket_groups), "buckets": buckets}

    def _normalize_cache_metadata(
        self,
        raw_metadata: object,
        *,
        signature: _CompoundBucketSignature,
        prompt_sequence_length: int,
        conditioning_tensors: Mapping[str, torch.Tensor],
        cache_file: Path,
    ) -> _ImageEditCacheMetadata:
        """Validate and normalize provenance and tensor-shape metadata.

        Args:
            raw_metadata: Serialized sample provenance and cache metadata.
            signature: Target shape [channels, height, width] and ordered
                context shapes [channels, context_height, context_width].
            prompt_sequence_length: Sequence axis length of prompt embeddings.
            conditioning_tensors: Named tensors with arbitrary, explicitly
                metadata-declared shapes.
            cache_file: Cache file used to identify invalid fields in errors.

        Returns:
            Validated metadata with spatial shapes, token lengths, and compound
            bucket signature normalized to JSON-safe values.
        """
        if not isinstance(raw_metadata, dict):
            raise ValueError(f"Image-edit cache {cache_file} field 'metadata' must be a mapping")
        cache_metadata = cast(dict[str, object], raw_metadata)

        original_ids = cache_metadata.get("original_ids")
        if not isinstance(original_ids, dict) or not original_ids:
            raise ValueError(f"Image-edit cache {cache_file} metadata.original_ids must be a non-empty mapping")
        original_id_values = cast(dict[object, object], original_ids)
        if any(
            not isinstance(name, str) or not name or isinstance(value, bool) or not isinstance(value, (str, int))
            for name, value in original_id_values.items()
        ):
            raise ValueError(
                f"Image-edit cache {cache_file} metadata.original_ids must map non-empty names to string or integer IDs"
            )

        target_spatial_shape = _normalize_shape(
            cache_metadata.get("target_spatial_shape"),
            field_name=f"{cache_file}.metadata.target_spatial_shape",
            rank=2,
        )
        if target_spatial_shape != signature.target[-2:]:
            raise ValueError(
                f"Image-edit cache {cache_file} target_spatial_shape {target_spatial_shape} does not match "
                f"target latent spatial shape {signature.target[-2:]}"
            )

        raw_context_spatial_shapes = cache_metadata.get("context_spatial_shapes")
        if not isinstance(raw_context_spatial_shapes, list):
            raise ValueError(f"Image-edit cache {cache_file} metadata.context_spatial_shapes must be an ordered list")
        context_spatial_shapes = [
            _normalize_shape(
                shape,
                field_name=f"{cache_file}.metadata.context_spatial_shapes[{index}]",
                rank=2,
            )
            for index, shape in enumerate(raw_context_spatial_shapes)
        ]
        expected_context_spatial_shapes = [shape[-2:] for shape in signature.contexts]
        if context_spatial_shapes != expected_context_spatial_shapes:
            raise ValueError(
                f"Image-edit cache {cache_file} context_spatial_shapes {context_spatial_shapes} do not match "
                f"context latent spatial shapes {expected_context_spatial_shapes}"
            )

        metadata_signature = _parse_compound_bucket_signature(
            cache_metadata.get("compound_bucket_signature"),
            field_name=f"{cache_file}.metadata.compound_bucket_signature",
        )
        if metadata_signature != signature:
            raise ValueError(
                f"Image-edit cache {cache_file} metadata compound bucket signature does not match its tensors"
            )

        conditioning_shapes = cache_metadata.get("conditioning_shapes", {})
        if not isinstance(conditioning_shapes, dict):
            raise ValueError(f"Image-edit cache {cache_file} metadata.conditioning_shapes must be a mapping")
        if any(not isinstance(name, str) or not name for name in conditioning_shapes):
            raise ValueError(
                f"Image-edit cache {cache_file} metadata.conditioning_shapes names must be non-empty strings"
            )
        conditioning_shapes = cast(dict[str, object], conditioning_shapes)
        if set(conditioning_shapes) != set(conditioning_tensors):
            raise ValueError(
                f"Image-edit cache {cache_file} conditioning_shapes names {sorted(conditioning_shapes)} do not match "
                f"conditioning_tensors names {sorted(conditioning_tensors)}"
            )
        normalized_conditioning_shapes: dict[str, list[int]] = {}
        for name, tensor in conditioning_tensors.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Image-edit cache {cache_file} conditioning tensor names must be non-empty strings")
            expected_shape = _normalize_shape(
                conditioning_shapes[name],
                field_name=f"{cache_file}.metadata.conditioning_shapes[{name!r}]",
                rank=tensor.ndim,
                allow_zero=True,
            )
            if expected_shape != tuple(tensor.shape):
                raise ValueError(
                    f"Image-edit cache {cache_file} conditioning tensor {name!r} has shape {tuple(tensor.shape)}, "
                    f"but metadata declares {expected_shape}"
                )
            normalized_conditioning_shapes[name] = list(expected_shape)

        target_token_length = cache_metadata.get("target_token_length")
        if target_token_length is None:
            target_token_length = signature.target[-2] * signature.target[-1]
        target_token_length = _nonnegative_int(
            target_token_length,
            field_name=f"{cache_file}.metadata.target_token_length",
        )

        raw_context_token_lengths = cache_metadata.get("context_token_lengths")
        if raw_context_token_lengths is None:
            context_token_lengths = [height * width for _, height, width in signature.contexts]
        else:
            if not isinstance(raw_context_token_lengths, list) or len(raw_context_token_lengths) != len(
                signature.contexts
            ):
                raise ValueError(
                    f"Image-edit cache {cache_file} metadata.context_token_lengths must contain one value per "
                    "ordered context latent"
                )
            context_token_lengths = [
                _nonnegative_int(
                    length,
                    field_name=f"{cache_file}.metadata.context_token_lengths[{index}]",
                )
                for index, length in enumerate(raw_context_token_lengths)
            ]

        text_token_length = cache_metadata.get("text_token_length", prompt_sequence_length)
        text_token_length = _nonnegative_int(
            text_token_length,
            field_name=f"{cache_file}.metadata.text_token_length",
        )
        if text_token_length > prompt_sequence_length:
            raise ValueError(
                f"Image-edit cache {cache_file} text_token_length {text_token_length} exceeds prompt sequence "
                f"length {prompt_sequence_length}"
            )

        normalized = cast(_ImageEditCacheMetadata, dict(cache_metadata))
        normalized["original_ids"] = cast(dict[str, str | int], original_ids)
        normalized["target_spatial_shape"] = list(target_spatial_shape)
        normalized["context_spatial_shapes"] = [list(shape) for shape in context_spatial_shapes]
        normalized["target_token_length"] = target_token_length
        normalized["context_token_lengths"] = context_token_lengths
        normalized["text_token_length"] = text_token_length
        normalized["compound_bucket_signature"] = _signature_to_metadata(signature)
        normalized["conditioning_shapes"] = normalized_conditioning_shapes
        return normalized

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Load and validate one cached image-edit sample.

        Args:
            idx: Zero-based sample index.

        Returns:
            Mapping containing a target latent tensor of shape [channels,
            height, width], ordered context latent tensors of shape [channels,
            context_height, context_width], prompt embeddings of shape
            [sequence, hidden], a prompt mask of shape [sequence], optional
            named conditioning tensors with metadata-declared shapes, and
            provenance metadata.
        """
        item = self.metadata[idx]
        cache_file = self._resolve_contained_path(item.get("cache_file"), field_name=f"metadata[{idx}].cache_file")
        data = torch.load(cache_file, map_location="cpu", weights_only=True)
        if not isinstance(data, Mapping):
            raise ValueError(f"Image-edit cache {cache_file} must contain a mapping")

        target_latent = _require_tensor(data, "target_latent")
        target_shape = _validate_latent(target_latent, field_name=f"{cache_file}.target_latent")

        raw_context_latents = data.get("context_latents")
        if not isinstance(raw_context_latents, list) or not raw_context_latents:
            raise ValueError(f"Image-edit cache {cache_file} field 'context_latents' must be a non-empty ordered list")
        context_latents: list[torch.Tensor] = []
        context_shapes: list[tuple[int, int, int]] = []
        for context_index, context_latent in enumerate(raw_context_latents):
            if not isinstance(context_latent, torch.Tensor):
                raise ValueError(f"Image-edit cache {cache_file} context_latents[{context_index}] must be a tensor")
            context_latents.append(context_latent)
            context_shapes.append(
                _validate_latent(
                    context_latent,
                    field_name=f"{cache_file}.context_latents[{context_index}]",
                )
            )

        signature = _CompoundBucketSignature(target=target_shape, contexts=tuple(context_shapes))
        shard_signature = _parse_compound_bucket_signature(
            item.get("compound_bucket_signature"),
            field_name=f"metadata[{idx}].compound_bucket_signature",
        )
        if shard_signature != signature:
            raise ValueError(
                f"Image-edit cache {cache_file} tensor shapes {_signature_to_metadata(signature)} do not match "
                f"metadata shard signature {_signature_to_metadata(shard_signature)}"
            )

        prompt_embeddings = _require_tensor(data, "prompt_embeddings")
        if prompt_embeddings.ndim != 2 or not prompt_embeddings.is_floating_point():
            raise ValueError(
                f"Image-edit cache {cache_file} prompt_embeddings must be a floating-point tensor of shape "
                f"[sequence, hidden], got {tuple(prompt_embeddings.shape)} with dtype {prompt_embeddings.dtype}"
            )
        prompt_attention_mask = _require_tensor(data, "prompt_attention_mask")
        if prompt_attention_mask.ndim != 1 or prompt_attention_mask.shape[0] != prompt_embeddings.shape[0]:
            raise ValueError(
                f"Image-edit cache {cache_file} prompt_attention_mask must have shape [sequence] matching "
                f"prompt_embeddings, got {tuple(prompt_attention_mask.shape)}"
            )
        if prompt_attention_mask.is_floating_point() or prompt_attention_mask.is_complex():
            raise ValueError(
                f"Image-edit cache {cache_file} prompt_attention_mask must use a boolean or integer dtype, "
                f"got {prompt_attention_mask.dtype}"
            )

        raw_conditioning_tensors = data.get("conditioning_tensors", {})
        if not isinstance(raw_conditioning_tensors, Mapping):
            raise ValueError(f"Image-edit cache {cache_file} conditioning_tensors must be a mapping")
        conditioning_tensors: dict[str, torch.Tensor] = {}
        for name, tensor in raw_conditioning_tensors.items():
            if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"Image-edit cache {cache_file} conditioning_tensors must map non-empty names to tensors"
                )
            conditioning_tensors[name] = tensor

        metadata = self._normalize_cache_metadata(
            data.get("metadata"),
            signature=signature,
            prompt_sequence_length=prompt_embeddings.shape[0],
            conditioning_tensors=conditioning_tensors,
            cache_file=cache_file,
        )
        sample: _ImageEditSample = {
            "target_latent": target_latent,
            "context_latents": context_latents,
            "prompt_embeddings": prompt_embeddings,
            "prompt_attention_mask": prompt_attention_mask,
            "conditioning_tensors": conditioning_tensors,
            "metadata": metadata,
        }
        return cast(dict[str, object], sample)


def _pad_prompts(
    embeddings: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad cached prompt encodings to a common sequence length.

    Args:
        embeddings: Prompt tensors, each of shape [sequence, hidden].
        masks: Attention-mask tensors, each of shape [sequence].

    Returns:
        Prompt embeddings of shape [batch, max_sequence, hidden] and attention
        masks of shape [batch, max_sequence].
    """
    hidden_sizes = {embedding.shape[1] for embedding in embeddings}
    embedding_dtypes = {embedding.dtype for embedding in embeddings}
    mask_dtypes = {mask.dtype for mask in masks}
    if len(hidden_sizes) != 1 or len(embedding_dtypes) != 1 or len(mask_dtypes) != 1:
        raise ValueError("Prompt embeddings and masks in a batch must have compatible hidden sizes and dtypes")

    max_sequence_length = max(embedding.shape[0] for embedding in embeddings)
    first_embedding = embeddings[0]
    first_mask = masks[0]
    padded_embeddings = first_embedding.new_zeros((len(embeddings), max_sequence_length, first_embedding.shape[1]))
    padded_masks = first_mask.new_zeros((len(masks), max_sequence_length))
    for index, (embedding, mask) in enumerate(zip(embeddings, masks)):
        padded_embeddings[index, : embedding.shape[0]] = embedding
        padded_masks[index, : mask.shape[0]] = mask
    return padded_embeddings, padded_masks


def _stack_conditioning_tensors(batch: Sequence[_ImageEditSample]) -> dict[str, torch.Tensor]:
    """Stack explicitly named model-conditioning tensors.

    Args:
        batch: Samples whose ``conditioning_tensors`` entries contain tensors
            with metadata-declared arbitrary shapes. A given name must have the
            same shape and dtype in every sample.

    Returns:
        Mapping whose tensors have shape [batch, ...], where trailing axes match
        each cache tensor's declared shape.
    """
    names = set(batch[0]["conditioning_tensors"])
    if any(set(sample["conditioning_tensors"]) != names for sample in batch[1:]):
        raise ValueError("All samples in an image-edit batch must provide the same conditioning tensor names")

    result: dict[str, torch.Tensor] = {}
    for name in sorted(names):
        tensors = [sample["conditioning_tensors"][name] for sample in batch]
        shapes = {tuple(tensor.shape) for tensor in tensors}
        dtypes = {tensor.dtype for tensor in tensors}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise ValueError(
                f"Conditioning tensor {name!r} must have the same shape and dtype in every image-edit sample"
            )
        result[name] = torch.stack(tensors)
    return result


def _collate_image_edit(batch: list[_ImageEditSample]) -> _ImageEditBatch:
    """Collate samples with compatible target and ordered-context shapes.

    Args:
        batch: Samples containing target tensors of shape [channels, height,
            width], ordered context tensors of shape [channels, context_height,
            context_width], prompt tensors of shape [sequence, hidden], prompt
            masks of shape [sequence], and optional named conditioning tensors.

    Returns:
        Batch containing ``image_latents`` of shape [batch, channels, height,
        width], an ordered list of context tensors of shape [batch, channels,
        context_height, context_width], ``text_embeddings`` of shape [batch,
        max_sequence, hidden], ``text_attention_mask`` of shape [batch,
        max_sequence], and conditioning tensors of shape [batch, ...]. Metadata
        token counts are ordinary CPU integers and do not require inspecting
        accelerator tensors.
    """
    if not batch:
        raise ValueError("Cannot collate an empty image-edit batch")

    target_shapes = {tuple(sample["target_latent"].shape) for sample in batch}
    target_dtypes = {sample["target_latent"].dtype for sample in batch}
    context_shapes = [tuple(tuple(tensor.shape) for tensor in sample["context_latents"]) for sample in batch]
    context_dtypes = [tuple(tensor.dtype for tensor in sample["context_latents"]) for sample in batch]
    if len(target_shapes) != 1 or len(target_dtypes) != 1 or len(set(context_shapes)) != 1:
        raise ValueError("Image-edit batches require identical target and ordered-context latent shapes")
    if len(set(context_dtypes)) != 1:
        raise ValueError("Image-edit batches require identical ordered-context latent dtypes")

    target_latents = torch.stack([sample["target_latent"] for sample in batch])
    context_latents = [
        torch.stack([sample["context_latents"][context_index] for sample in batch])
        for context_index in range(len(batch[0]["context_latents"]))
    ]
    prompt_embeddings, prompt_attention_mask = _pad_prompts(
        [sample["prompt_embeddings"] for sample in batch],
        [sample["prompt_attention_mask"] for sample in batch],
    )

    sample_metadata = [sample["metadata"] for sample in batch]
    target_token_counts = [metadata["target_token_length"] for metadata in sample_metadata]
    context_token_counts = [sum(metadata["context_token_lengths"]) for metadata in sample_metadata]
    text_token_counts = [metadata["text_token_length"] for metadata in sample_metadata]
    total_token_counts = [
        target + context + text
        for target, context, text in zip(target_token_counts, context_token_counts, text_token_counts)
    ]
    batch_metadata: _ImageEditBatchMetadata = {
        "samples": sample_metadata,
        "batch_size": len(batch),
        "target_token_counts": target_token_counts,
        "context_token_counts": context_token_counts,
        "text_token_counts": text_token_counts,
        "total_token_counts": total_token_counts,
        "target_token_count": sum(target_token_counts),
        "context_token_count": sum(context_token_counts),
        "text_token_count": sum(text_token_counts),
        "total_token_count": sum(total_token_counts),
    }
    return {
        "image_latents": target_latents,
        "context_latents": context_latents,
        "text_embeddings": prompt_embeddings,
        "text_attention_mask": prompt_attention_mask,
        "conditioning_tensors": _stack_conditioning_tensors(batch),
        "data_type": "image",
        "metadata": batch_metadata,
    }


@dataclass
class ImageEditDataloaderConfig:
    """Construction-time configuration for a cached image-edit dataloader."""

    cache_dir: str
    quantization: int = 64
    base_resolution: tuple[int, int] = (1024, 1024)
    drop_last: bool = True
    shuffle: bool = True
    dynamic_batch_size: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    seed: int = 42

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        """Build the configured dataset, bucket sampler, and stateful dataloader.

        Args:
            dp_rank: Data-parallel rank that consumes the dataloader.
            dp_world_size: Number of data-parallel ranks.
            batch_size: Base per-rank batch size.

        Returns:
            Materialized stateful dataloader and its resumable bucket sampler.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if dp_world_size <= 0:
            raise ValueError(f"dp_world_size must be positive, got {dp_world_size}")
        if dp_rank < 0 or dp_rank >= dp_world_size:
            raise ValueError(f"dp_rank must be in [0, {dp_world_size}), got {dp_rank}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.dynamic_batch_size:
            raise ValueError(
                "Image-edit dynamic_batch_size is not supported because batch cost depends on the compound "
                "target/context latent signature; use a fixed batch size"
            )
        if self.prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be positive, got {self.prefetch_factor}")
        if len(self.base_resolution) != 2 or any(dimension <= 0 for dimension in self.base_resolution):
            raise ValueError(f"base_resolution must contain two positive dimensions, got {self.base_resolution}")

        dataset = ImageEditDatasetConfig(
            cache_dir=self.cache_dir,
            quantization=self.quantization,
        ).build()
        sampler = SequentialBucketSampler(
            dataset,
            base_batch_size=batch_size,
            base_resolution=self.base_resolution,
            drop_last=self.drop_last,
            shuffle_buckets=self.shuffle,
            shuffle_within_bucket=self.shuffle,
            dynamic_batch_size=self.dynamic_batch_size,
            seed=self.seed,
            num_replicas=dp_world_size,
            rank=dp_rank,
        )
        dataloader = StatefulDataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=_collate_image_edit,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            generator=torch.Generator().manual_seed(self.seed + dp_rank),
        )
        logger.info("Built image-edit dataset with %d samples and %d batches", len(dataset), len(sampler))
        return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler)


__all__ = [
    "ImageEditDataloaderConfig",
    "ImageEditDataset",
    "ImageEditDatasetConfig",
]
