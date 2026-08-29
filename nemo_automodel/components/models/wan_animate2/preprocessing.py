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

"""Offline cache encoder for Wan-Animate-2 character-animation triplets.

Turns a triplet manifest (reference character image, driving video, target
video, caption) into the ``.meta`` cache files consumed by
:class:`~nemo_automodel.components.datasets.diffusion.text_to_video_dataset.TextToVideoDataset`.
Every encoding rule mirrors the upstream ``WanAnimate2Pipeline``, with one
inversion relative to the generic video processors: the resolution bucket is
derived from the *reference image*, and the driving and target frames are
letterboxed into it.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.shared.import_utils import safe_import

NUMPY_AVAILABLE, np = safe_import(
    "numpy",
    msg="Wan-Animate-2 preprocessing requires NumPy from the diffusion media dependencies",
)
CV2_AVAILABLE, cv2 = safe_import(
    "cv2",
    msg="Wan-Animate-2 preprocessing requires OpenCV from the diffusion media dependencies",
)
PIL_AVAILABLE, Image = safe_import(
    "PIL.Image",
    msg="Wan-Animate-2 preprocessing requires Pillow from the diffusion media dependencies",
)
# diffusers/transformers are imported lazily in _load_models: importing them
# initializes the CUDA driver, and this module is pulled in by the
# tools.diffusion.processors package, whose importers must stay CUDA-free so
# their spawned multiprocessing workers can initialize CUDA themselves.

logger = logging.getLogger(__name__)

# Upstream CLIP ViT-H preprocessing constants (pipeline_wan_animate_2.CLIP_MEAN/CLIP_STD).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_CLIP_INPUT_SIZE = 224
_CLIP_TOKENS = 257
_CLIP_HIDDEN = 1280

# The fixed driving-branch prompt used by the upstream pipeline (``prompt_ref``).
_REFERENCE_PROMPT = "人物动作的参考视频"

# The Wan VAE compresses 8x spatially and 4x temporally into 16 latent channels.
# WanAnimate2Adapter hard-codes the same two factors, so a VAE with a different
# latent grid would silently break the reference-RoPE and block-mask geometry.
_VAE_TEMPORAL_COMPRESSION = 4
_VAE_SPATIAL_COMPRESSION = 8
_LATENT_CHANNELS = 16

# resize_by_area(..., divisor=16) keeps both pixel axes divisible by 16 so the
# latent grid stays divisible by the transformer's (2, 2) spatial patch size.
_SPATIAL_DIVISOR = 16

# The transformer's text_len is 512; a longer cached embedding would make its
# zero-padding step allocate a negative length.
_TEXT_LEN = 512

_DEFAULT_NUM_FRAMES = 81
_DEFAULT_FPS = 24
_METADATA_SHARD_SIZE = 1_000
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
_HALF_DTYPES = (torch.bfloat16, torch.float16)


class _CacheRecord(TypedDict):
    """One metadata-shard entry describing a written ``.meta`` cache file."""

    cache_file: str
    bucket_resolution: list[int]
    aspect_ratio: float
    num_frames: int
    latent_frames: int
    prompt: str
    reference_image: str
    driving_video: str
    target_video: str
    original_id: str
    row_index: int
    model_type: str


@dataclass(frozen=True)
class _TripletSample:
    """One validated manifest row referencing a reference/driving/target triplet."""

    identifier: str
    caption: str
    reference_path: Path
    driving_path: Path
    target_path: Path
    row_index: int


@dataclass(frozen=True)
class _EncoderModels:
    """The frozen conditioning stack loaded once per worker.

    Attributes:
        vae: Upstream ``AutoencoderKLWan``. Its input layout is [batch, 3,
            frames, height, width] and its latent layout is [batch, 16,
            latent_frames, latent_height, latent_width].
        text_encoder: Upstream ``UMT5EncoderModel``.
        image_encoder: Upstream ``CLIPVisionModel`` (ViT-H/14 at 224 pixels).
        tokenizer: The UMT5 ``AutoTokenizer``, kept untyped to avoid a hard
            transformers import at module scope.
        latents_mean: VAE channel means of shape [1, 16, 1, 1, 1].
        latents_reciprocal_std: Reciprocal VAE channel standard deviations of
            shape [1, 16, 1, 1, 1].
    """

    vae: nn.Module
    text_encoder: nn.Module
    image_encoder: nn.Module
    tokenizer: Any
    latents_mean: torch.Tensor
    latents_reciprocal_std: torch.Tensor


class WanAnimate2CacheEncoder:
    """Encode Wan-Animate-2 triplet manifests into the cached training contract.

    Only the frozen conditioning stack is loaded (Wan VAE, UMT5 text encoder,
    CLIP ViT-H image encoder). The trainable transformer is never instantiated.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_sequence_length: int = _TEXT_LEN,
        device: str | None = None,
        torch_dtype: str = "bfloat16",
    ) -> None:
        """Configure offline Wan-Animate-2 encoding.

        Args:
            model_name: Diffusers-format directory or Hub repo id holding the
                ``vae``, ``text_encoder``, ``tokenizer``, and ``image_encoder``
                subfolders. It has no default, so :meth:`encode_manifest` raises
                when it is left unset.
            max_sequence_length: Cached UMT5 token count. Wan-Animate-2 uses
                512 (the transformer's ``text_len``), unlike the 226 used by the
                Wan2.1/2.2 text-to-video processors.
            device: Optional explicit encoding device. By default each worker
                selects its assigned CUDA device, with CPU as a test fallback.
            torch_dtype: VAE / image-encoder compute dtype and the dtype of every
                cached tensor. One of ``bfloat16``, ``float16``, ``float32``.

        Raises:
            ValueError: If ``max_sequence_length`` is outside ``[1, 512]`` or
                ``torch_dtype`` is unsupported.
        """
        if not 0 < max_sequence_length <= _TEXT_LEN:
            raise ValueError(
                f"max_sequence_length must be in [1, {_TEXT_LEN}] because the Wan-Animate-2 transformer "
                f"zero-pads text embeddings to text_len={_TEXT_LEN}; got {max_sequence_length}"
            )
        if torch_dtype not in _DTYPES:
            raise ValueError(f"torch_dtype must be one of {sorted(_DTYPES)}, got {torch_dtype!r}")

        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.torch_dtype = torch_dtype

    @property
    def model_type(self) -> str:
        """Return the cache's model-type tag."""
        return "wan_animate2"

    def encode_manifest(
        self,
        *,
        manifest_path: Path,
        output_dir: Path,
        max_pixels: int,
        num_frames: int = _DEFAULT_NUM_FRAMES,
        fps: int = _DEFAULT_FPS,
        num_gpus: int = 1,
        verify: bool = False,
    ) -> Path:
        """Encode a triplet manifest into a bucketed ``.meta`` cache.

        Args:
            manifest_path: JSONL manifest whose rows carry ``reference_image``,
                ``driving_video``, ``target_video``, and ``caption``. Relative
                media paths resolve against the manifest's parent directory.
            output_dir: Destination cache root. It receives one
                ``{width}x{height}`` subdirectory, ``metadata_shard_*.json``
                shards, and ``metadata.json``.
            max_pixels: Pixel-area budget handed to the upstream
                ``resize_by_area`` when deriving the bucket from the reference
                image.
            num_frames: Pixel-space frame count emitted for both the driving and
                the target video after resampling to ``fps``. Must be positive
                and satisfy ``4n + 1`` so the Wan VAE emits
                ``(num_frames - 1) // 4 + 1`` latent frames.
            fps: Target frame rate the cached clips represent. Both videos are
                resampled to it, matching the upstream pipeline's inference-time
                resampling, so training motion speed matches inference. Must be
                positive.
            num_gpus: Number of independent encoder workers, each owning one
                CUDA device and a private copy of the conditioning stack.
            verify: Whether to decode each target latent and require a finite
                RGB tensor of shape [1, 3, num_frames, height, width].

        Returns:
            Path to ``metadata.json``. Its shards index cache files holding
            ``video_latents`` / ``cond_zero_latents`` / ``driving_latents`` of
            shape [1, 16, latent_frames, latent_height, latent_width],
            ``reference_latents`` of shape [1, 16, 1, latent_height,
            latent_width], ``clip_fea`` / ``clip_fea_ref`` of shape [1, 257,
            1280], and ``text_embeddings`` / ``prompt_ref_embeddings`` of shape
            [1, max_sequence_length, 4096].

        Raises:
            ValueError: If the encoder is unconfigured, an argument is invalid,
                the destination already holds a cache, or the manifest's
                reference images do not all map to one resolution bucket.
        """
        if not self.model_name:
            raise ValueError(
                "WanAnimate2CacheEncoder requires an explicit model_name pointing at a Diffusers-format "
                "directory with vae/, text_encoder/, tokenizer/, and image_encoder/ subfolders"
            )
        if not (NUMPY_AVAILABLE and CV2_AVAILABLE and PIL_AVAILABLE):
            raise ImportError("Wan-Animate-2 preprocessing requires NumPy, OpenCV, and Pillow")

        manifest_path = Path(manifest_path).resolve()
        output_dir = Path(output_dir).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Wan-Animate-2 manifest does not exist: {manifest_path}")
        # Python's modulo is non-negative for a positive divisor, so -3 % 4 == 1;
        # the positivity check is what actually rejects a negative frame count.
        if num_frames <= 0 or num_frames % _VAE_TEMPORAL_COMPRESSION != 1:
            raise ValueError(
                f"num_frames must be positive and satisfy 4n + 1 for the Wan VAE's 4x temporal "
                f"compression, got {num_frames}"
            )
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if max_pixels < _SPATIAL_DIVISOR * _SPATIAL_DIVISOR:
            raise ValueError(
                f"max_pixels must be at least {_SPATIAL_DIVISOR * _SPATIAL_DIVISOR} for 16-aligned buckets, "
                f"got {max_pixels}"
            )
        if num_gpus <= 0:
            raise ValueError(f"num_gpus must be positive, got {num_gpus}")
        if self.device is not None and num_gpus != 1:
            raise ValueError("An explicit device can only be used with num_gpus=1")

        samples = _read_manifest(manifest_path)
        bucket_height, bucket_width = _resolve_shared_bucket(samples, max_pixels=max_pixels)
        logger.info(
            "Wan-Animate-2 bucket resolved from %d reference images: %dx%d (WxH), %d frames",
            len(samples),
            bucket_width,
            bucket_height,
            num_frames,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            raise ValueError(f"Wan-Animate-2 cache output directory already holds a cache: {metadata_path}")

        if num_gpus == 1:
            records = self._encode_rows(
                samples,
                output_dir=output_dir,
                max_pixels=max_pixels,
                bucket_height=bucket_height,
                bucket_width=bucket_width,
                num_frames=num_frames,
                fps=fps,
                worker_index=0,
                verify=verify,
            )
        else:
            if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
                raise RuntimeError(
                    f"Requested {num_gpus} GPU preprocessing workers, but {torch.cuda.device_count()} CUDA "
                    "devices are available"
                )
            records = self._encode_rows_multiprocess(
                samples,
                output_dir=output_dir,
                max_pixels=max_pixels,
                bucket_height=bucket_height,
                bucket_width=bucket_width,
                num_frames=num_frames,
                fps=fps,
                num_workers=num_gpus,
                verify=verify,
            )

        records.sort(key=lambda record: record["row_index"])
        shard_names = _write_metadata_shards(records, output_dir)
        latent_frames = (num_frames - 1) // _VAE_TEMPORAL_COMPRESSION + 1
        metadata = {
            "processor": "wan_animate2",
            "model_name": self.model_name,
            "model_type": self.model_type,
            "total_items": len(records),
            "num_shards": len(shard_names),
            "shard_size": _METADATA_SHARD_SIZE,
            "shards": shard_names,
            "preprocessing_config": {
                "processor_target": (
                    "nemo_automodel.components.models.wan_animate2.preprocessing.WanAnimate2CacheEncoder"
                ),
                "max_pixels": max_pixels,
                "num_frames": num_frames,
                "latent_frames": latent_frames,
                "bucket_resolution": [bucket_width, bucket_height],
                "max_sequence_length": self.max_sequence_length,
                "reference_prompt": _REFERENCE_PROMPT,
                "spatial_divisor": _SPATIAL_DIVISOR,
                "vae_latent_sampling": "mode",
                "torch_dtype": self.torch_dtype,
                "num_gpus": num_gpus,
                "verify": verify,
            },
        }
        _write_json_atomic(metadata_path, metadata)
        logger.info("Wrote %d Wan-Animate-2 cache samples to %s", len(records), output_dir)
        return metadata_path

    def _encode_rows(
        self,
        samples: list[_TripletSample],
        *,
        output_dir: Path,
        max_pixels: int,
        bucket_height: int,
        bucket_width: int,
        num_frames: int,
        fps: int,
        worker_index: int,
        verify: bool,
    ) -> list[_CacheRecord]:
        """Encode one worker's shard of manifest rows.

        Args:
            samples: Validated triplet rows owned by this worker.
            output_dir: Cache root shared by all workers.
            max_pixels: Pixel-area budget passed to ``resize_by_area``; it must
                be the same value that produced the shared bucket.
            bucket_height: Letterboxed pixel height shared by every sample.
            bucket_width: Letterboxed pixel width shared by every sample.
            num_frames: Pixel frames emitted per video after fps resampling.
            fps: Target frame rate the emitted frames represent.
            worker_index: CUDA device index owned by this worker.
            verify: Whether to decode each target latent of shape [1, 16,
                latent_frames, latent_height, latent_width].

        Returns:
            One metadata record per encoded manifest row.
        """
        device = self._worker_device(worker_index)
        models = self._load_models(device)
        records: list[_CacheRecord] = []
        try:
            # Both sample-invariant tensors are encoded once per worker instead of
            # per row: the all-zero previous-segment clip depends only on the
            # bucket and the frame count, the reference prompt is a constant, and
            # both encoders are deterministic. Every cached copy stays
            # byte-identical while a third of the VAE work and one UMT5 forward
            # disappear from every row.
            with torch.no_grad():
                cond_zero_latents = _encode_vae(
                    models,
                    torch.zeros(
                        1,
                        3,
                        num_frames,
                        bucket_height,
                        bucket_width,
                        device=device,
                        dtype=next(models.vae.parameters()).dtype,
                    ),
                )
                prompt_ref_embeddings = self._encode_text(models, _REFERENCE_PROMPT, device=device)
            for sample in samples:
                records.append(
                    self._encode_sample(
                        models,
                        sample,
                        output_dir=output_dir,
                        max_pixels=max_pixels,
                        bucket_height=bucket_height,
                        bucket_width=bucket_width,
                        num_frames=num_frames,
                        fps=fps,
                        cond_zero_latents=cond_zero_latents,
                        prompt_ref_embeddings=prompt_ref_embeddings,
                        device=device,
                        verify=verify,
                    )
                )
        finally:
            models.vae.to("cpu")
            models.text_encoder.to("cpu")
            models.image_encoder.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return records

    def _encode_rows_multiprocess(
        self,
        samples: list[_TripletSample],
        *,
        output_dir: Path,
        max_pixels: int,
        bucket_height: int,
        bucket_width: int,
        num_frames: int,
        fps: int,
        num_workers: int,
        verify: bool,
    ) -> list[_CacheRecord]:
        """Encode round-robin manifest shards in spawned GPU workers.

        Each worker receives a pickled copy of this encoder's bound
        :meth:`_encode_rows` and builds its own conditioning stack on the CUDA
        device matching its shard index.

        Args:
            samples: Every validated triplet row, dealt round-robin to workers.
            num_workers: Number of spawned workers, one per CUDA device.
            output_dir, max_pixels, bucket_height, bucket_width, num_frames,
                fps, verify: Forwarded verbatim to :meth:`_encode_rows`.

        Returns:
            One metadata record per encoded manifest row, in worker order.
        """
        shards = [samples[index::num_workers] for index in range(num_workers)]
        records: list[_CacheRecord] = []
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as executor:
            futures = [
                executor.submit(
                    self._encode_rows,
                    shard,
                    output_dir=output_dir,
                    max_pixels=max_pixels,
                    bucket_height=bucket_height,
                    bucket_width=bucket_width,
                    num_frames=num_frames,
                    fps=fps,
                    worker_index=worker_index,
                    verify=verify,
                )
                for worker_index, shard in enumerate(shards)
                if shard
            ]
            for future in futures:
                try:
                    records.extend(future.result())
                except Exception as exc:
                    raise RuntimeError("Wan-Animate-2 preprocessing worker failed") from exc
        return records

    def _worker_device(self, worker_index: int) -> torch.device:
        """Resolve the runtime device owned by an encoder worker."""
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda", worker_index)
        return torch.device("cpu")

    def _load_models(self, device: torch.device) -> _EncoderModels:
        """Load the frozen Wan-Animate-2 conditioning stack onto one device.

        Args:
            device: Device that owns the VAE, text encoder, and image encoder.

        Returns:
            The loaded stack together with standardization tensors of shape
            [1, 16, 1, 1, 1].
        """
        diffusers_available, diffusers = safe_import(
            "diffusers",
            msg="Wan-Animate-2 preprocessing requires the diffusion optional dependencies",
        )
        transformers_available, transformers = safe_import(
            "transformers",
            msg="Wan-Animate-2 preprocessing requires the transformers dependency",
        )
        if not diffusers_available or not transformers_available:
            raise ImportError("Wan-Animate-2 preprocessing requires diffusers and transformers")

        from nemo_automodel._diffusers._hf_cache import resolve_diffusion_model_dir

        # Pre-resolve to a local snapshot so the four per-subfolder loads below
        # reuse a warm HF cache instead of re-validating over the network.
        model_dir = resolve_diffusion_model_dir(self.model_name)
        compute_dtype = _DTYPES[self.torch_dtype]
        # UMT5 overflows to zeros in float16, so it mirrors the Wan text-to-video
        # processor and always runs in bfloat16 on CUDA / float32 on CPU.
        text_encoder_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        logger.info("[Wan-Animate-2] Loading conditioning stack from %s", model_dir)
        vae = diffusers.AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=compute_dtype)
        text_encoder = transformers.UMT5EncoderModel.from_pretrained(
            model_dir, subfolder="text_encoder", torch_dtype=text_encoder_dtype
        )
        # transformers>=5.0.0 no longer ties "shared.weight" to
        # "encoder.embed_tokens.weight" during from_pretrained(); without this the
        # embedding stays zero-initialized and every cached prompt becomes zeros.
        if (
            hasattr(text_encoder, "shared")
            and hasattr(text_encoder.encoder, "embed_tokens")
            and text_encoder.encoder.embed_tokens.weight.data_ptr() != text_encoder.shared.weight.data_ptr()
        ):
            text_encoder.encoder.embed_tokens.weight = text_encoder.shared.weight
        image_encoder = transformers.CLIPVisionModel.from_pretrained(
            model_dir, subfolder="image_encoder", torch_dtype=compute_dtype
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer")

        vae.requires_grad_(False).eval().to(device=device)
        text_encoder.requires_grad_(False).eval().to(device=device)
        image_encoder.requires_grad_(False).eval().to(device=device)

        # AutoencoderKLWan always registers both, so only their width is checked.
        if len(vae.config.latents_mean) != _LATENT_CHANNELS or len(vae.config.latents_std) != _LATENT_CHANNELS:
            raise ValueError(
                f"Wan-Animate-2 requires a {_LATENT_CHANNELS}-channel VAE, but the loaded config declares "
                f"{len(vae.config.latents_mean)} latents_mean and {len(vae.config.latents_std)} latents_std entries"
            )
        latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(
            1, _LATENT_CHANNELS, 1, 1, 1
        )
        latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(
            1, _LATENT_CHANNELS, 1, 1, 1
        )
        return _EncoderModels(
            vae=vae,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            tokenizer=tokenizer,
            latents_mean=latents_mean,
            latents_reciprocal_std=1.0 / latents_std,
        )

    def _encode_sample(
        self,
        models: _EncoderModels,
        sample: _TripletSample,
        *,
        output_dir: Path,
        max_pixels: int,
        bucket_height: int,
        bucket_width: int,
        num_frames: int,
        fps: int,
        cond_zero_latents: torch.Tensor,
        prompt_ref_embeddings: torch.Tensor,
        device: torch.device,
        verify: bool,
    ) -> _CacheRecord:
        """Encode and persist one reference/driving/target triplet.

        The written payload holds the cached tensors documented on
        :meth:`encode_manifest`, all on CPU in the storage dtype.

        Args:
            models: The frozen conditioning stack owned by this worker.
            sample: The validated manifest row to encode.
            output_dir: Cache root; the payload lands in
                ``{output_dir}/{bucket_width}x{bucket_height}``.
            max_pixels: Pixel-area budget passed to ``resize_by_area``; it must
                be the same value that produced the shared bucket, otherwise the
                reference image would land in a different bucket than the one
                the driving and target frames are letterboxed into.
            bucket_height: Letterboxed pixel height for all three media.
            bucket_width: Letterboxed pixel width for all three media.
            num_frames: Pixel frames emitted per video after fps resampling.
            fps: Target frame rate the emitted frames represent.
            cond_zero_latents: The run's shared all-zero conditioning latent of
                shape [1, 16, latent_frames, latent_height, latent_width],
                encoded once per worker by :meth:`_encode_rows`.
            prompt_ref_embeddings: The run's shared reference-prompt embedding of
                shape [1, max_sequence_length, 4096], encoded once per worker by
                :meth:`_encode_rows`.
            device: Encoding device.
            verify: Whether to decode the target latent of shape [1, 16,
                latent_frames, latent_height, latent_width].

        Returns:
            The metadata record describing the written cache file.
        """
        storage_dtype = _DTYPES[self.torch_dtype]
        vae_dtype = next(models.vae.parameters()).dtype

        reference_rgb = _load_rgb_image(sample.reference_path)
        reference_padded = _resize_by_area(reference_rgb, target_area=max_pixels)
        driving_frames = _read_resampled_frames(
            sample.driving_path, num_frames=num_frames, fps=fps, height=bucket_height, width=bucket_width
        )
        target_frames = _read_resampled_frames(
            sample.target_path, num_frames=num_frames, fps=fps, height=bucket_height, width=bucket_width
        )

        reference_pixels = _frames_to_tensor(reference_padded[None], device=device)
        driving_pixels = _frames_to_tensor(driving_frames, device=device)
        target_pixels = _frames_to_tensor(target_frames, device=device)

        with torch.no_grad():
            reference_latents = _encode_vae(models, reference_pixels.to(vae_dtype))
            driving_latents = _encode_vae(models, driving_pixels.to(vae_dtype))
            video_latents = _encode_vae(models, target_pixels.to(vae_dtype))
            clip_fea = _clip_visual_encode(models.image_encoder, reference_pixels[0, :, 0], device=device)
            clip_fea_ref = _clip_visual_encode(models.image_encoder, driving_pixels[0, :, 0], device=device)
            text_embeddings = self._encode_text(models, sample.caption, device=device)
            if verify:
                _verify_latent(models, video_latents, num_frames=num_frames)

        # The driving/target/zero clips all carry num_frames pixel frames through
        # the same VAE, and the adapter re-checks their latent frame counts against
        # each other at training time, so only the VAE's spatial compression -- the
        # one property nothing downstream can observe -- is asserted here.
        latent_frames = video_latents.shape[2]
        expected_latent_hw = (bucket_height // _VAE_SPATIAL_COMPRESSION, bucket_width // _VAE_SPATIAL_COMPRESSION)
        if tuple(video_latents.shape[-2:]) != expected_latent_hw:
            raise ValueError(
                f"Wan-Animate-2 requires a VAE with {_VAE_SPATIAL_COMPRESSION}x spatial compression, so a "
                f"{bucket_width}x{bucket_height} (WxH) clip must encode to a latent grid of "
                f"{expected_latent_hw[1]}x{expected_latent_hw[0]}; the loaded VAE produced "
                f"{video_latents.shape[-1]}x{video_latents.shape[-2]}"
            )

        # Cache tensors are stored on CPU: the dataset calls torch.load without a
        # map_location, so a CUDA-resident payload would pin every sample to the
        # preprocessing GPU at training time.
        payload = {
            "video_latents": _to_cache(video_latents, storage_dtype),
            "reference_latents": _to_cache(reference_latents, storage_dtype),
            "driving_latents": _to_cache(driving_latents, storage_dtype),
            "cond_zero_latents": _to_cache(cond_zero_latents, storage_dtype),
            "clip_fea": _to_cache(clip_fea, storage_dtype),
            "clip_fea_ref": _to_cache(clip_fea_ref, storage_dtype),
            "text_embeddings": _to_cache(text_embeddings, storage_dtype),
            "prompt_ref_embeddings": _to_cache(prompt_ref_embeddings, storage_dtype),
            "bucket_resolution": [bucket_width, bucket_height],
            "num_frames": num_frames,
            "prompt": sample.caption,
            "reference_image": str(sample.reference_path),
            "driving_video": str(sample.driving_path),
            "target_video": str(sample.target_path),
            "model_type": self.model_type,
        }

        bucket_dir = output_dir / f"{bucket_width}x{bucket_height}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        cache_file = bucket_dir / f"sample_{sample.row_index:08d}.meta"
        temporary_file = cache_file.with_suffix(".meta.tmp")
        torch.save(payload, temporary_file)
        temporary_file.replace(cache_file)

        return {
            "cache_file": str(cache_file),
            "bucket_resolution": [bucket_width, bucket_height],
            "aspect_ratio": bucket_width / bucket_height,
            "num_frames": num_frames,
            "latent_frames": latent_frames,
            "prompt": sample.caption,
            "reference_image": str(sample.reference_path),
            "driving_video": str(sample.driving_path),
            "target_video": str(sample.target_path),
            "original_id": sample.identifier,
            "row_index": sample.row_index,
            "model_type": self.model_type,
        }

    def _encode_text(self, models: _EncoderModels, prompt: str, *, device: torch.device) -> torch.Tensor:
        """Encode one prompt with UMT5 using the upstream padding policy.

        The embedding is trimmed to the unpadded token count and re-padded with
        zeros, which is what ``_get_t5_prompt_embeds`` does upstream.

        Args:
            models: The frozen conditioning stack owned by this worker.
            prompt: Caption or fixed reference prompt to encode.
            device: Encoding device.

        Returns:
            Tensor of shape [1, max_sequence_length, 4096] on CPU, where axis 1
            is the UMT5 token axis and trailing positions are exactly zero.
        """
        inputs = models.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_count = int(attention_mask.gt(0).sum(dim=1)[0].item())

        embeddings = models.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        trimmed = embeddings[0, :token_count]
        padded = torch.cat([trimmed, trimmed.new_zeros(self.max_sequence_length - token_count, trimmed.shape[1])])
        return padded.unsqueeze(0).detach().to("cpu")


def _read_manifest(manifest_path: Path) -> list[_TripletSample]:
    """Read and validate every Wan-Animate-2 triplet manifest row.

    Args:
        manifest_path: JSONL manifest. Each row must be an object with
            ``reference_image``, ``driving_video``, ``target_video``, and
            ``caption``; ``id`` is optional and defaults to the row index.
            Relative media paths resolve against the manifest's parent.

    Returns:
        The validated rows in manifest order.
    """
    root = manifest_path.parent
    samples: list[_TripletSample] = []
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

            caption = row.get("caption")
            if not isinstance(caption, str):
                raise ValueError(f"Manifest line {line_number} caption must be a string")
            row_index = len(samples)
            identifier = row.get("id", row_index)
            if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
                raise ValueError(f"Manifest line {line_number} id must be a string or integer")

            paths = {}
            for field_name in ("reference_image", "driving_video", "target_video"):
                value = row.get(field_name)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Manifest line {line_number} {field_name} must be a non-empty string")
                resolved = Path(value)
                resolved = (resolved if resolved.is_absolute() else root / resolved).resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(f"Manifest line {line_number} {field_name} does not exist: {resolved}")
                paths[field_name] = resolved

            samples.append(
                _TripletSample(
                    identifier=str(identifier),
                    caption=caption,
                    reference_path=paths["reference_image"],
                    driving_path=paths["driving_video"],
                    target_path=paths["target_video"],
                    row_index=row_index,
                )
            )

    if not samples:
        raise ValueError(f"Wan-Animate-2 manifest contains no samples: {manifest_path}")
    return samples


def _to_cache(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Detach a tensor onto CPU in the cache storage dtype.

    Args:
        tensor: Any encoder output tensor; its shape and axis order are
            preserved.
        dtype: Floating-point dtype used for every cached tensor.

    Returns:
        A CPU tensor with the same shape as ``tensor``, detached from the
        autograd graph. ``Tensor.to`` is a no-op when the input is already a CPU
        tensor of ``dtype``, so the result may alias the input; callers only hand
        the result to ``torch.save``, which serializes a fresh copy per file.
    """
    return tensor.detach().to(device="cpu", dtype=dtype)


def _bucket_dimensions(height: int, width: int, *, target_area: int) -> tuple[int, int]:
    """Compute the upstream ``resize_by_area`` output dimensions.

    Args:
        height: Source pixel height.
        width: Source pixel width.
        target_area: Pixel-area budget.

    Returns:
        ``(bucket_height, bucket_width)``, both positive multiples of 16.
    """
    aspect_ratio = width / height
    new_height = math.sqrt(target_area / aspect_ratio)
    new_width = target_area / new_height
    bucket_width = int((new_width // _SPATIAL_DIVISOR) * _SPATIAL_DIVISOR)
    bucket_height = int((new_height // _SPATIAL_DIVISOR) * _SPATIAL_DIVISOR)
    if bucket_width <= 0 or bucket_height <= 0:
        raise ValueError(
            f"target_area {target_area} is too small for a {width}x{height} (WxH) reference image; "
            f"resize_by_area produced {bucket_width}x{bucket_height}"
        )
    return bucket_height, bucket_width


def _resolve_shared_bucket(samples: list[_TripletSample], *, max_pixels: int) -> tuple[int, int]:
    """Derive the single training bucket from every reference image.

    Wan-Animate-2 caches its reference RoPE offset (``refer_offset_w``) on the
    first forward pass and never resets it, so a run must train on exactly one
    resolution bucket. This reads image headers only and fails fast when the
    manifest would produce more than one.

    Args:
        samples: Validated manifest rows.
        max_pixels: Pixel-area budget for ``resize_by_area``.

    Returns:
        ``(bucket_height, bucket_width)`` shared by every sample.
    """
    buckets: dict[tuple[int, int], list[str]] = {}
    for sample in samples:
        with Image.open(sample.reference_path) as image:
            width, height = image.size
        bucket = _bucket_dimensions(height, width, target_area=max_pixels)
        buckets.setdefault(bucket, []).append(str(sample.reference_path))

    if len(buckets) > 1:
        summary = ", ".join(
            f"{width}x{height} (WxH) from {len(paths)} reference image(s), e.g. {paths[0]}"
            for (height, width), paths in sorted(buckets.items())
        )
        raise ValueError(
            "Wan-Animate-2 training requires a single resolution bucket because the transformer caches its "
            f"reference RoPE offset on the first forward pass. The manifest produced {len(buckets)} buckets: "
            f"{summary}. Split the manifest by reference-image aspect ratio and preprocess each split separately."
        )
    return next(iter(buckets))


def _load_rgb_image(path: Path) -> np.ndarray:
    """Load one image file as a channels-last RGB array.

    Args:
        path: Image file path.

    Returns:
        ``numpy.ndarray`` of shape [height, width, 3] with dtype ``uint8``.
    """
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _padding_resize(image: np.ndarray, *, height: int, width: int, interpolation: int) -> np.ndarray:
    """Letterbox one frame into exact dimensions with black padding.

    Mirrors the upstream ``padding_resize``: the frame is scaled to fit while
    preserving its aspect ratio and centered inside a zero-filled canvas.

    Args:
        image: ``numpy.ndarray`` of shape [height, width, channels], dtype
            ``uint8``, channels-last RGB.
        height: Output pixel height.
        width: Output pixel width.
        interpolation: OpenCV interpolation flag.

    Returns:
        ``numpy.ndarray`` of shape [height, width, channels], dtype ``uint8``.
    """
    original_height, original_width = image.shape[:2]
    channels = image.shape[2]
    padded = np.zeros((height, width, channels), dtype=np.uint8)
    if original_height / original_width > height / width:
        new_width = int(height / original_height * original_width)
        resized = cv2.resize(image, (new_width, height), interpolation=interpolation)
        offset = (width - new_width) // 2
        padded[:, offset : offset + new_width, :] = resized
    else:
        new_height = int(width / original_width * original_height)
        resized = cv2.resize(image, (width, new_height), interpolation=interpolation)
        offset = (height - new_height) // 2
        padded[offset : offset + new_height, :, :] = resized
    return padded


def _resize_by_area(image: np.ndarray, *, target_area: int) -> np.ndarray:
    """Resize a reference image to the 16-aligned bucket, then letterbox it.

    Mirrors the upstream ``resize_by_area(image, target_area, divisor=16)``,
    including its area-dependent interpolation choice.

    Args:
        image: ``numpy.ndarray`` of shape [height, width, 3], dtype ``uint8``.
        target_area: Pixel-area budget.

    Returns:
        ``numpy.ndarray`` of shape [bucket_height, bucket_width, 3], dtype
        ``uint8``, with both spatial axes divisible by 16.
    """
    height, width = image.shape[:2]
    bucket_height, bucket_width = _bucket_dimensions(height, width, target_area=target_area)
    interpolation = cv2.INTER_AREA if bucket_width * bucket_height < width * height else cv2.INTER_LINEAR
    return _padding_resize(image, height=bucket_height, width=bucket_width, interpolation=interpolation)


def _resample_frame_indices(source_frame_count: int, source_fps: float, *, num_frames: int, fps: int) -> list[int]:
    """Select source frame indices that realize ``num_frames`` at ``fps``.

    Ports the upstream inference-time resampling (``get_frame_indices`` in the
    Wan-Animate-2 Diffusers pipeline) so cached clips carry the same physical
    motion rate the model is driven with at inference. This deliberately differs
    from the shared video processor in ``tools/diffusion/processors/base_video.py``,
    which spreads ``num_frames`` evenly across the whole clip and ignores fps:
    for text-to-video that only time-stretches the training clip, but here the
    driving video's temporal sampling *is* the motion signal, so a duration-
    dependent rate would train motion speeds the model never sees at inference.

    Args:
        source_frame_count: Total frames available in the source video.
        source_fps: Frame rate reported by the container.
        num_frames: Number of frames to emit.
        fps: Target frame rate the emitted frames should represent.

    Returns:
        ``num_frames`` source frame indices, clamped to the last available frame
        so short clips hold on their final frame rather than failing.
    """
    times = np.arange(0, num_frames) / float(fps)
    indices = np.round(times * source_fps).astype(int)
    return np.clip(indices, 0, max(source_frame_count - 1, 0)).tolist()


def _read_resampled_frames(video_path: Path, *, num_frames: int, fps: int, height: int, width: int) -> np.ndarray:
    """Decode a video at a target frame rate and letterbox it into the bucket.

    Args:
        video_path: Video file path.
        num_frames: Number of frames to emit.
        fps: Target frame rate the emitted frames should represent.
        height: Letterboxed output pixel height.
        width: Letterboxed output pixel width.

    Returns:
        ``numpy.ndarray`` of shape [num_frames, height, width, 3], dtype
        ``uint8``, channels-last RGB.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not source_fps or source_fps <= 0 or not np.isfinite(source_fps):
            raise ValueError(
                f"Video {video_path} reports an unusable frame rate ({source_fps!r}); Wan-Animate-2 "
                "preprocessing resamples to an explicit fps and cannot proceed without it"
            )
        indices = _resample_frame_indices(source_frame_count, source_fps, num_frames=num_frames, fps=fps)

        # Decode sequentially and hold the most recent frame: consecutive target
        # indices repeat whenever fps exceeds source_fps, and seeking per frame is
        # far slower than a linear read.
        frames: list[np.ndarray] = []
        decoded: np.ndarray | None = None
        current_index = -1
        for target_index in indices:
            while current_index < target_index:
                ok, frame = capture.read()
                if not ok:
                    break
                current_index += 1
                decoded = frame
            if decoded is None:
                raise ValueError(f"Video {video_path} yielded no decodable frames")
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            frames.append(_padding_resize(rgb, height=height, width=width, interpolation=cv2.INTER_LINEAR))
    finally:
        capture.release()

    if len(frames) != num_frames:
        raise ValueError(f"Video {video_path} yielded {len(frames)} frames but {num_frames} are required")
    return np.stack(frames)


def _frames_to_tensor(frames: np.ndarray, *, device: torch.device) -> torch.Tensor:
    """Convert decoded frames into the VAE's pixel layout.

    Args:
        frames: ``numpy.ndarray`` of shape [frames, height, width, 3], dtype
            ``uint8``, channels-last RGB.
        device: Output tensor device.

    Returns:
        Tensor of shape [1, 3, frames, height, width] in ``float32``, scaled to
        the range [-1, 1].
    """
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).to(device=device, dtype=torch.float32)
    tensor = tensor / 127.5 - 1.0
    return tensor.permute(3, 0, 1, 2).unsqueeze(0)


def _encode_vae(models: _EncoderModels, pixels: torch.Tensor) -> torch.Tensor:
    """Encode a pixel clip into standardized Wan latents.

    Args:
        models: The frozen conditioning stack; its ``latents_mean`` and
            ``latents_reciprocal_std`` have shape [1, 16, 1, 1, 1].
        pixels: Tensor of shape [1, 3, frames, height, width] in the range
            [-1, 1] and in the VAE's parameter dtype.

    Returns:
        Tensor of shape [1, 16, latent_frames, height // 8, width // 8] in
        ``float32``, standardized as ``(latent - mean) * (1 / std)``.
    """
    encoded = models.vae.encode(pixels)
    latent_dist = getattr(encoded, "latent_dist", None)
    if latent_dist is None:
        raise TypeError(
            "The Wan VAE encode() output must expose latent_dist; got "
            f"{type(encoded).__name__}. Check the installed diffusers version."
        )
    latents = latent_dist.mode().to(torch.float32)
    return (latents - models.latents_mean) * models.latents_reciprocal_std


def _clip_visual_encode(image_encoder: nn.Module, pixels: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """Encode one frame with CLIP ViT-H exactly as the upstream pipeline does.

    Args:
        image_encoder: Upstream ``CLIPVisionModel``.
        pixels: Tensor of shape [3, height, width] in the range [-1, 1],
            channels-first RGB.
        device: Encoding device.

    Returns:
        Penultimate hidden-state tensor of shape [1, 257, 1280] on CPU, where
        axis 1 holds the class token followed by 256 patch tokens.
    """
    encoder_dtype = next(image_encoder.parameters()).dtype
    # Reshape exactly as upstream does -- unsqueeze(1) then transpose(0, 1) rather
    # than the equivalent-looking unsqueeze(0). The two produce the same values but
    # different strides, and bicubic F.interpolate dispatches on memory layout, so
    # the shortcut costs 15 bfloat16 ulps against the reference pipeline.
    videos = F.interpolate(
        pixels.unsqueeze(1).transpose(0, 1),
        size=(_CLIP_INPUT_SIZE, _CLIP_INPUT_SIZE),
        mode="bicubic",
        align_corners=False,
    )
    videos = videos.mul_(0.5).add_(0.5)
    mean = torch.tensor(_CLIP_MEAN, device=device, dtype=videos.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=device, dtype=videos.dtype).view(1, 3, 1, 1)
    videos = (videos - mean) / std

    # Upstream feeds float32 pixels through an autocast region rather than a
    # hard cast, which keeps every layer norm in float32. Casting the input to a
    # half dtype instead would run those norms in bfloat16 and drift the cached
    # features away from what the model sees at inference.
    with torch.autocast(device_type=device.type, dtype=encoder_dtype, enabled=encoder_dtype in _HALF_DTYPES):
        outputs = image_encoder(pixel_values=videos, output_hidden_states=True)
    features = outputs.hidden_states[-2]
    if tuple(features.shape) != (1, _CLIP_TOKENS, _CLIP_HIDDEN):
        raise ValueError(
            f"Wan-Animate-2 expects CLIP ViT-H features of shape [1, {_CLIP_TOKENS}, {_CLIP_HIDDEN}], "
            f"got {tuple(features.shape)}"
        )
    return features.detach().to("cpu")


def _verify_latent(models: _EncoderModels, latents: torch.Tensor, *, num_frames: int) -> None:
    """Decode a standardized latent and require a finite RGB clip.

    Args:
        models: The frozen conditioning stack owned by this worker.
        latents: Tensor of shape [1, 16, latent_frames, latent_height,
            latent_width], standardized as in :func:`_encode_vae`.
        num_frames: Expected decoded pixel-frame count.
    """
    vae_dtype = next(models.vae.parameters()).dtype
    destandardized = latents / models.latents_reciprocal_std + models.latents_mean
    decoded = models.vae.decode(destandardized.to(vae_dtype), return_dict=False)[0]
    if decoded.ndim != 5 or decoded.shape[1] != 3 or decoded.shape[2] != num_frames:
        raise ValueError(
            f"Wan VAE verification expected a decoded tensor of shape [1, 3, {num_frames}, height, width], "
            f"got {tuple(decoded.shape)}"
        )
    if not torch.isfinite(decoded).all():
        raise ValueError("Wan VAE verification produced non-finite pixels")


def _write_metadata_shards(records: list[_CacheRecord], output_dir: Path) -> list[str]:
    """Write deterministic cache-index shards and return their relative names."""
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
