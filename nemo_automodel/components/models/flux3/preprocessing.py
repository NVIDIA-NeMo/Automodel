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

"""FLUX.3 image-edit (img2img) cache encoder.

Encodes generic image-edit manifests (one target image, one context/source
image, one instruction prompt per row) into the cached contract read by
``ImageEditDataset``:

- ``target_latent``: patchified + BN-normalized FLUX.2-VAE latent [128, H/16, W/16]
- ``context_latents``: single-entry list with the conditioning image latent in
  the same space (fed to the transformer's ``image_cond`` stream with l = -1
  position ids and timestep 0)
- ``prompt_embeddings`` / ``prompt_attention_mask``: stacked Qwen3-VL hidden
  states [sequence, joint_attention_dim] with an all-ones mask — FLUX.3 conditions
  on every padded position, matching the reference pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from nemo_automodel.components.models.flux3.text_encoding import (
    MAX_SEQUENCE_LENGTH,
    encode_text_qwen3vl,
    load_qwen3vl_text_stack,
)
from nemo_automodel.components.models.flux3.types import Flux3BucketSignature, Flux3EditRecord, Flux3Models
from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from multiprocessing.queues import Queue

    from PIL.Image import Image

logger = logging.getLogger(__name__)

PIL_AVAILABLE, PIL = safe_import("PIL")

_METADATA_SHARD_SIZE = 1000
_RESOLUTION_PRESETS = {
    "256p": (256, 256),
    "512p": (512, 512),
    "768p": (768, 768),
    "1024p": (1024, 1024),
    "1536p": (1536, 1536),
}
# VAE downsamples by 8 and latents are 2x2-patchified: pixel dims must be
# divisible by 16 so the patchified latent grid is integral.
_SPATIAL_ALIGNMENT = 16


@dataclass(frozen=True)
class _MediaReference:
    role: str
    file_name: str
    path: Path


@dataclass(frozen=True)
class _ManifestSample:
    identifier: str
    prompt: str
    target: _MediaReference
    context: _MediaReference
    metadata: dict[str, object]
    row_index: int


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Convert latents to patchified format (2x2 spatial patch -> channel).

    Args:
        latents: Tensor of shape [batch, channels, height, width].

    Returns:
        Tensor of shape [batch, channels*4, height//2, width//2].
    """
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)


class Flux3ImageEditCacheEncoder:
    """Encode image-edit manifests into the FLUX.3 img2img cached contract.

    Loads only the frozen FLUX.2 image VAE and Qwen3-VL text stack from the
    modular checkpoint; the trainable transformer is never loaded.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        device: str | None = None,
    ) -> None:
        """Configure offline FLUX.3 image-edit encoding.

        Args:
            model_name: FLUX.3 modular checkpoint repo id or local path. There
                is no public default checkpoint id yet, so this must be set.
            max_sequence_length: Maximum prompt token count. The image-small checkpoint uses
                an 8192-token limit and per-prompt padding to multiples of 80.
            device: Optional explicit encoding device. By default each worker
                selects its assigned CUDA device, with CPU as a fallback.
        """
        if not model_name:
            raise ValueError("model_name must be set: FLUX.3 has no public default checkpoint id")
        if max_sequence_length != MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"FLUX.3 image-small uses a {MAX_SEQUENCE_LENGTH}-token prompt limit; "
                f"pass --max_sequence_length {MAX_SEQUENCE_LENGTH} (got {max_sequence_length})"
            )
        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.device = device

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
        """Encode a generic image-edit manifest into cached FLUX.3 tensors.

        Args:
            manifest_path: JSONL manifest whose rows contain an ID, prompt,
                ordered role-to-file media mappings, and provenance metadata.
            output_dir: Destination for cache tensors and indexes.
            max_pixels: Maximum pixel area when no fixed preset is selected.
            resolution_preset: Optional fixed square preset such as ``"512p"``;
                target and context images are center-cropped to that shape.
            num_gpus: Number of encoder workers (one CUDA device each).
            verify: Whether to run finite-value checks on each cached latent.

        Returns:
            Path to ``metadata.json`` whose shards index target latents of
            shape [128, H/16, W/16], single-entry context latent lists in the
            same space, prompt embeddings of shape [sequence, joint_attention_dim],
            and all-ones prompt masks of shape [sequence].
        """
        manifest_path = Path(manifest_path).resolve()
        output_dir = Path(output_dir).resolve()
        if not PIL_AVAILABLE:
            raise ImportError("FLUX.3 image-edit preprocessing requires Pillow")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Image-edit manifest does not exist: {manifest_path}")
        if resolution_preset is not None and resolution_preset not in _RESOLUTION_PRESETS:
            raise ValueError(
                f"resolution_preset must be one of {sorted(_RESOLUTION_PRESETS)}, got {resolution_preset!r}"
            )
        if num_gpus <= 0:
            raise ValueError(f"num_gpus must be positive, got {num_gpus}")
        if self.device is not None and num_gpus != 1:
            raise ValueError("An explicit device can only be used with num_gpus=1")

        samples = _read_manifest(manifest_path)
        (output_dir / "samples").mkdir(parents=True, exist_ok=True)

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
                    f"Requested {num_gpus} GPU preprocessing workers, but {torch.cuda.device_count()} CUDA "
                    "devices are available"
                )
            records = _encode_rows_multiprocess(
                self,
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
            "row_limit": len(samples),
            "preprocessing_config": {
                "processor_target": "nemo_automodel.components.models.flux3.preprocessing.Flux3ImageEditCacheEncoder",
                "model_name": self.model_name,
                "max_sequence_length": self.max_sequence_length,
                "max_pixels": max_pixels,
                "resolution_preset": resolution_preset,
                "spatial_alignment": _SPATIAL_ALIGNMENT,
                "vae_latent_sampling": "mode",
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
    ) -> list[Flux3EditRecord]:
        """Encode one worker's manifest shard on its assigned device."""
        device = self._worker_device(worker_index)
        models = self._load_models(device)
        records = []
        for sample in samples:
            records.append(
                self._encode_sample(
                    models,
                    sample,
                    output_dir=output_dir,
                    max_pixels=max_pixels,
                    resolution_preset=resolution_preset,
                    device=device,
                    verify=verify,
                )
            )
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

    def _load_models(self, device: torch.device) -> Flux3Models:
        """Load the frozen FLUX.2 image VAE and Qwen3-VL text stack."""
        diffusers_available, diffusers = safe_import(
            "diffusers",
            msg="FLUX.3 image-edit preprocessing requires the diffusion optional dependencies",
        )
        if not diffusers_available:
            raise ImportError("FLUX.3 image-edit preprocessing requires diffusers")

        from nemo_automodel._diffusers._hf_cache import resolve_diffusion_model_dir

        model_dir = resolve_diffusion_model_dir(self.model_name)
        vae = diffusers.AutoencoderKLFlux2.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32)
        vae = vae.requires_grad_(False).eval().to(device)
        models = {
            "vae": vae,
            "bn_mean": vae.bn.running_mean.view(1, -1, 1, 1).float().to(device),
            "bn_std": torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).float().to(device),
        }
        models.update(load_qwen3vl_text_stack(model_dir, str(device)))
        return models

    def _encode_latent(self, models: Flux3Models, image: Image, device: torch.device) -> torch.Tensor:
        """Encode one RGB image into the patchified + BN-normalized latent space.

        Args:
            models: Dict containing 'vae', 'bn_mean', 'bn_std'.
            image: PIL RGB image, already resized to 16-aligned dimensions.
            device: Encoding device.

        Returns:
            Latent tensor of shape [128, H/16, W/16], FP32, on CPU.
        """
        import numpy as np

        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = ((tensor - 0.5) / 0.5).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            raw = models["vae"].encode(tensor).latent_dist.mode()
            latent = patchify_latents(raw.float())
            latent = (latent - models["bn_mean"]) / models["bn_std"]
        return latent.squeeze(0).detach().cpu()

    def _encode_sample(
        self,
        models: Flux3Models,
        sample: _ManifestSample,
        *,
        output_dir: Path,
        max_pixels: int,
        resolution_preset: str | None,
        device: torch.device,
        verify: bool,
    ) -> Flux3EditRecord:
        """Encode and persist one image-edit example.

        Args:
            models: Loaded VAE + text stack from ``_load_models``.
            sample: Validated target/context media references and prompt.
            output_dir: Cache root for the emitted tensor payload.
            max_pixels: Maximum pixel area without a preset.
            resolution_preset: Optional fixed square preset.
            device: Runtime device for VAE and text encoding.
            verify: Whether to run finite-value checks on the cached latents.

        Returns:
            Index record with cache path, compound bucket signature, and token
            counts for the emitted cache file.
        """
        target_image = _resize_image(PIL.Image.open(sample.target.path), max_pixels, resolution_preset)
        context_image = _resize_image(PIL.Image.open(sample.context.path), max_pixels, resolution_preset)

        target_latent = self._encode_latent(models, target_image, device)
        context_latent = self._encode_latent(models, context_image, device)
        prompt_embeddings = encode_text_qwen3vl(sample.prompt, models, str(device))
        # All padded positions carry conditioning values (FLUX.3 does not mask
        # padded prompt tokens), so the cached mask is all-ones to keep the
        # collate path from trimming the per-prompt window.
        prompt_attention_mask = torch.ones(prompt_embeddings.shape[0], dtype=torch.long)

        if verify:
            for name, tensor in (("target_latent", target_latent), ("context_latents[0]", context_latent)):
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite values in {name} for manifest row {sample.row_index}")

        target_shape = list(target_latent.shape)
        context_shape = list(context_latent.shape)
        signature: Flux3BucketSignature = {"target": target_shape, "contexts": [context_shape]}
        # FLUX.3 latents are already patchified: one token per latent pixel.
        target_token_length = target_shape[-2] * target_shape[-1]
        context_token_lengths = [context_shape[-2] * context_shape[-1]]
        metadata = {
            "original_ids": {"id": sample.identifier, "row_index": sample.row_index},
            "target_spatial_shape": target_shape[-2:],
            "context_spatial_shapes": [context_shape[-2:]],
            "target_token_length": target_token_length,
            "context_token_lengths": context_token_lengths,
            "text_token_length": prompt_embeddings.shape[0],
            "compound_bucket_signature": signature,
            "conditioning_shapes": {},
            "manifest_metadata": sample.metadata,
        }
        payload = {
            "target_latent": target_latent,
            "context_latents": [context_latent],
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
            "text_token_length": prompt_embeddings.shape[0],
        }


def _resize_image(image: Image, max_pixels: int, resolution_preset: str | None) -> Image:
    """Resize an image for VAE encoding with 16-aligned dimensions.

    Args:
        image: PIL image (any mode; converted to RGB).
        max_pixels: Maximum pixel area when no preset is selected.
        resolution_preset: Optional fixed square preset key.

    Returns:
        RGB PIL image whose dimensions are multiples of 16. Presets produce an
        exact center-cropped square; otherwise the aspect ratio is preserved
        under the ``max_pixels`` area budget.
    """
    image = image.convert("RGB")
    if resolution_preset is not None:
        target_h, target_w = _RESOLUTION_PRESETS[resolution_preset]
        scale = max(target_h / image.height, target_w / image.width)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), PIL.Image.LANCZOS)
        left = (resized.width - target_w) // 2
        top = (resized.height - target_h) // 2
        return resized.crop((left, top, left + target_w, top + target_h))

    scale = min(1.0, (max_pixels / (image.width * image.height)) ** 0.5)
    new_w = max(_SPATIAL_ALIGNMENT, int(image.width * scale) // _SPATIAL_ALIGNMENT * _SPATIAL_ALIGNMENT)
    new_h = max(_SPATIAL_ALIGNMENT, int(image.height * scale) // _SPATIAL_ALIGNMENT * _SPATIAL_ALIGNMENT)
    return image.resize((new_w, new_h), PIL.Image.LANCZOS)


def _read_manifest(manifest_path: Path) -> list[_ManifestSample]:
    """Read and validate every image-edit JSONL row.

    Each row must reference exactly one ``target`` and exactly one ``context``
    image; ``condition`` entries (used by VL-conditioned models like
    Qwen-Image-Edit) are not part of the FLUX.3 contract and are rejected.
    """
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

            references: dict[str, _MediaReference] = {}
            for media_index, item in enumerate(raw_media):
                if not isinstance(item, dict):
                    raise ValueError(f"Manifest line {line_number} media[{media_index}] must be a mapping")
                role = item.get("role")
                file_name = item.get("file_name")
                if role not in {"target", "context"}:
                    raise ValueError(
                        f"Manifest line {line_number} media[{media_index}] has unsupported role {role!r}; "
                        "FLUX.3 image-edit uses exactly one 'target' and one 'context' image"
                    )
                if role in references:
                    raise ValueError(f"Manifest line {line_number} has more than one {role!r} image")
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
                references[role] = _MediaReference(role=role, file_name=file_name, path=resolved_path)

            if set(references) != {"target", "context"}:
                raise ValueError(
                    f"Manifest line {line_number} requires exactly one target and one context image, "
                    f"got roles {sorted(references)}"
                )
            row_index_value = metadata.get("row_index", len(samples))
            if isinstance(row_index_value, bool) or not isinstance(row_index_value, int) or row_index_value < 0:
                raise ValueError(f"Manifest line {line_number} metadata.row_index must be a non-negative integer")
            samples.append(
                _ManifestSample(
                    identifier=str(identifier),
                    prompt=prompt,
                    target=references["target"],
                    context=references["context"],
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


def _encode_rows_worker(
    encoder: "Flux3ImageEditCacheEncoder",
    worker_index: int,
    samples: list[_ManifestSample],
    output_dir: Path,
    max_pixels: int,
    resolution_preset: str | None,
    verify: bool,
    results: Queue[tuple[int, list[Flux3EditRecord]]],
) -> None:
    """Encode one worker's shard in a spawned subprocess."""
    records = encoder._encode_rows(
        samples,
        output_dir=output_dir,
        max_pixels=max_pixels,
        resolution_preset=resolution_preset,
        worker_index=worker_index,
        verify=verify,
    )
    results.put((worker_index, records))


def _encode_rows_multiprocess(
    encoder: "Flux3ImageEditCacheEncoder",
    samples: list[_ManifestSample],
    *,
    output_dir: Path,
    max_pixels: int,
    resolution_preset: str | None,
    num_workers: int,
    verify: bool,
) -> list[Flux3EditRecord]:
    """Split samples across CUDA workers and merge their index records."""
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    results = context.Queue()
    shards = [samples[index::num_workers] for index in range(num_workers)]
    processes = []
    for worker_index, shard in enumerate(shards):
        process = context.Process(
            target=_encode_rows_worker,
            args=(encoder, worker_index, shard, output_dir, max_pixels, resolution_preset, verify, results),
        )
        process.start()
        processes.append(process)

    records: list[Flux3EditRecord] = []
    for _ in processes:
        _worker_index, worker_records = results.get()
        records.extend(worker_records)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Image-edit encoding worker exited with code {process.exitcode}")
    return records


def _write_metadata_shards(records: list[Flux3EditRecord], output_dir: Path) -> list[str]:
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
