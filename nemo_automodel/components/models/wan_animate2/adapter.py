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

"""Flow-matching adapter for cached Wan-Animate-2 character-animation batches."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext, ModelAdapter
from nemo_automodel.components.models.wan_animate2.interleaved import install_forward_origin

# The upstream transformer uses a (1, 2, 2) patch embedding, so post-patch grids
# halve both spatial axes. The VAE applies 8x spatial and 4x temporal compression.
_PATCH_H = 2
_PATCH_W = 2
_VAE_SPATIAL_COMPRESSION = 8
_VAE_TEMPORAL_COMPRESSION = 4

# The 36-channel patch-embedding input is 16 latent channels concatenated with a
# 20-channel conditioning block (4 mask + 16 latent).
_LATENT_CHANNELS = 16


class WanAnimate2Adapter(ModelAdapter):
    """Adapt cached Wan-Animate-2 triplet batches to the upstream Diffusers transformer.

    Wan-Animate-2 transfers motion from a driving video onto a reference
    character image. Unlike every other diffusion model in this repository, its
    transformer runs two passes per step against the same weights:

    1. ``forward_ref`` embeds the driving-video latents and writes per-block
       key/value tensors into a cache.
    2. ``forward_gen`` denoises the generation stream, where each generated
       latent frame attends to the time-aligned cached driving frame.

    Both calls are issued inside :meth:`forward` so the shared
    :class:`~nemo_automodel.components.flow_matching.adapters.base.ModelAdapter`
    single-call contract is preserved for every other model. ``forward_ref``
    carries gradient, matching the reference training implementation, which
    backpropagates through the reference stream. Inference builds the cache under
    ``torch.no_grad`` because it is reused across denoising steps; training does
    not.

    The generation stream carries one extra leading latent frame that holds the
    reference-character slot. Its prediction is discarded before the loss,
    mirroring inference where the decoded frame 0 is dropped.
    """

    @staticmethod
    def _build_i2v_mask(
        *,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        mask_pixel_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the 4-channel temporal conditioning mask.

        The mask marks which leading *pixel-space* frames are supplied as clean
        conditioning. It is expanded into the VAE's 4x temporal compression so
        that the first latent frame carries four mask channels.

        Args:
            latent_frames: Number of latent frames the mask must cover.
            latent_height: Latent grid height.
            latent_width: Latent grid width.
            mask_pixel_length: Number of leading pixel-space frames marked as
                conditioning. Zero marks nothing.
            device: Device for the returned tensor.
            dtype: Dtype for the returned tensor.

        Returns:
            Tensor of shape [4, latent_frames, latent_height, latent_width],
            where axis 0 stores the four temporally-expanded mask channels.
        """
        pixel_frames = (latent_frames - 1) * _VAE_TEMPORAL_COMPRESSION + 1
        mask = torch.zeros(1, pixel_frames, latent_height, latent_width, device=device, dtype=dtype)
        if mask_pixel_length > 0:
            mask[:, :mask_pixel_length] = 1
        mask = torch.cat(
            [torch.repeat_interleave(mask[:, 0:1], repeats=_VAE_TEMPORAL_COMPRESSION, dim=1), mask[:, 1:]],
            dim=1,
        )
        mask = mask.view(
            1, mask.shape[1] // _VAE_TEMPORAL_COMPRESSION, _VAE_TEMPORAL_COMPRESSION, latent_height, latent_width
        )
        return mask.transpose(1, 2)[0]

    @staticmethod
    def _require_tensor(batch: dict[str, Any], key: str, *, ndim: int, batch_size: int) -> torch.Tensor:
        """Fetch and shape-check a required cached conditioning tensor.

        Args:
            batch: Cached flow-matching batch dictionary.
            key: Batch key to fetch.
            ndim: Required rank of the tensor.
            batch_size: Required size of axis 0.

        Returns:
            The validated tensor, aliasing the batch entry.
        """
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Wan-Animate-2 batches require '{key}' as a torch.Tensor; got {type(value)!r}. "
                "Re-run preprocessing with the 'wan_animate2' processor."
            )
        if value.ndim != ndim:
            raise ValueError(f"'{key}' must have {ndim} dimensions, got shape {tuple(value.shape)}")
        if value.shape[0] != batch_size:
            raise ValueError(f"'{key}' must have batch size {batch_size}, got {value.shape[0]}")
        return value

    def prepare_inputs(self, context: FlowMatchingContext) -> dict[str, Any]:
        """Build the two-phase Wan-Animate-2 conditioning from a cached batch.

        Args:
            context: Flow context whose ``noisy_latents`` and ``latents``
                tensors have shape [batch, 16, target_latent_frames,
                latent_height, latent_width]. Its ``batch`` must additionally
                contain ``reference_latents`` of shape [batch, 16, 1,
                latent_height, latent_width], ``driving_latents`` of shape
                [batch, 16, driving_latent_frames, latent_height,
                latent_width], ``cond_zero_latents`` of shape [batch, 16,
                target_latent_frames, latent_height, latent_width],
                ``clip_fea`` and ``clip_fea_ref`` of shape [batch, 257, 1280],
                ``text_embeddings`` of shape [batch, text_tokens, 4096], and
                ``prompt_ref_embeddings`` of shape [batch, ref_text_tokens,
                4096].

        Returns:
            Mapping holding the tensors and scalars consumed by :meth:`forward`.
            ``x`` is a list of ``batch`` tensors of shape [16,
            target_latent_frames + 1, latent_height, latent_width]; ``y`` is a
            list of [20, target_latent_frames + 1, latent_height,
            latent_width]; ``x_ref`` and ``condition_y`` are lists of [16,
            driving_latent_frames, ...] and [20, driving_latent_frames, ...]
            respectively. Keys prefixed with an underscore carry slicing
            metadata for :meth:`forward` and are not passed to the model.
        """
        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 5:
            raise ValueError(
                "WanAnimate2Adapter expects noisy latents with shape "
                f"[batch, channels, frames, height, width]; got {tuple(noisy_latents.shape)}"
            )

        batch_size, channels, target_latent_frames, latent_height, latent_width = noisy_latents.shape
        if channels != _LATENT_CHANNELS:
            raise ValueError(f"Wan-Animate-2 requires {_LATENT_CHANNELS} latent channels, got {channels}")
        if latent_height % _PATCH_H != 0 or latent_width % _PATCH_W != 0:
            raise ValueError(
                "Latent height and width must be divisible by the (2, 2) patch size, "
                f"got {(latent_height, latent_width)}"
            )
        if batch_size != 1:
            # forward_ref passes a length-1 k_lens to the upstream varlen attention,
            # which silently truncates the packed key/value sequence when batch > 1.
            raise ValueError(
                "Wan-Animate-2 requires local_batch_size=1; the upstream reference pass does not "
                f"support batched key/value packing. Got batch size {batch_size}. Scale with "
                "gradient accumulation and data parallelism instead."
            )

        batch = context.batch
        device = context.device
        dtype = context.dtype

        reference_latents = self._require_tensor(batch, "reference_latents", ndim=5, batch_size=batch_size)
        driving_latents = self._require_tensor(batch, "driving_latents", ndim=5, batch_size=batch_size)
        cond_zero_latents = self._require_tensor(batch, "cond_zero_latents", ndim=5, batch_size=batch_size)
        clip_fea = self._require_tensor(batch, "clip_fea", ndim=3, batch_size=batch_size)
        clip_fea_ref = self._require_tensor(batch, "clip_fea_ref", ndim=3, batch_size=batch_size)
        text_embeddings = self._require_tensor(batch, "text_embeddings", ndim=3, batch_size=batch_size)
        prompt_ref_embeddings = self._require_tensor(batch, "prompt_ref_embeddings", ndim=3, batch_size=batch_size)

        if reference_latents.shape[2] != 1:
            raise ValueError(f"reference_latents must hold exactly one latent frame, got {reference_latents.shape[2]}")
        if cond_zero_latents.shape[2] != target_latent_frames:
            raise ValueError(
                "cond_zero_latents must match the target latent frame count "
                f"({target_latent_frames}), got {cond_zero_latents.shape[2]}"
            )
        if driving_latents.shape[2] != target_latent_frames:
            # The upstream block mask sizes the query span from the DRIVING frame
            # count (origin_len -> origin_latent_f + 1 reference slot), while
            # forward_gen embeds a stream of target_latent_frames + 1. A mismatch
            # produces a seq_len that silently disagrees with the flex-attention
            # block mask instead of raising.
            raise ValueError(
                "driving_latents and the target clip must have the same latent frame count; got "
                f"driving={driving_latents.shape[2]}, target={target_latent_frames}. Re-run preprocessing "
                "so both videos are encoded with the same num_frames."
            )
        for name, tensor in (("reference_latents", reference_latents), ("driving_latents", driving_latents)):
            if tensor.shape[-2:] != (latent_height, latent_width):
                raise ValueError(
                    f"{name} spatial dims {tuple(tensor.shape[-2:])} must match the target "
                    f"{(latent_height, latent_width)}"
                )

        reference_latents = reference_latents.to(device=device, dtype=dtype, non_blocking=True)
        driving_latents = driving_latents.to(device=device, dtype=dtype, non_blocking=True)
        cond_zero_latents = cond_zero_latents.to(device=device, dtype=dtype, non_blocking=True)
        driving_latent_frames = driving_latents.shape[2]

        # The generation stream prepends one latent frame for the reference slot.
        # Its x_t is built from the reference latent at the batch's sigma so the
        # input stays in-distribution; its prediction is discarded before the loss.
        sigma = context.sigma.to(device=device, dtype=torch.float32).view(batch_size, 1, 1, 1, 1)
        reference_noise = torch.randn(reference_latents.shape, device=device, dtype=torch.float32)
        reference_slot = (1.0 - sigma) * reference_latents.float() + sigma * reference_noise
        generation_stream = torch.cat([reference_slot.to(dtype), noisy_latents.to(dtype)], dim=2)

        generation_latent_frames = target_latent_frames + 1
        mask_reference = self._build_i2v_mask(
            latent_frames=1,
            latent_height=latent_height,
            latent_width=latent_width,
            mask_pixel_length=1,
            device=device,
            dtype=dtype,
        )
        # Zero-conditioning case: no previous-segment frames are supplied.
        mask_target = self._build_i2v_mask(
            latent_frames=target_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            mask_pixel_length=0,
            device=device,
            dtype=dtype,
        )
        driving_pixel_frames = (driving_latent_frames - 1) * _VAE_TEMPORAL_COMPRESSION + 1
        mask_driving = self._build_i2v_mask(
            latent_frames=driving_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            mask_pixel_length=driving_pixel_frames,
            device=device,
            dtype=dtype,
        )

        conditioning = []
        driving_conditioning = []
        for index in range(batch_size):
            y_reference = torch.cat([mask_reference, reference_latents[index]], dim=0)
            y_target = torch.cat([mask_target, cond_zero_latents[index]], dim=0)
            conditioning.append(torch.cat([y_reference, y_target], dim=1))
            driving_conditioning.append(torch.cat([mask_driving, driving_latents[index]], dim=0))

        grid_height = latent_height // _PATCH_H
        grid_width = latent_width // _PATCH_W
        sequence_length = generation_latent_frames * grid_height * grid_width
        sequence_length_ref = driving_latent_frames * grid_height * grid_width
        grid_sizes_ref = torch.tensor(
            [[driving_latent_frames, grid_height, grid_width]], dtype=torch.long, device=device
        )

        return {
            "x": [generation_stream[index] for index in range(batch_size)],
            "y": conditioning,
            "x_ref": [driving_latents[index] for index in range(batch_size)],
            "condition_y": driving_conditioning,
            "clip_fea": clip_fea.to(device=device, dtype=dtype, non_blocking=True),
            "clip_fea_ref": clip_fea_ref.to(device=device, dtype=dtype, non_blocking=True),
            "context": [
                text_embeddings[index].to(device=device, dtype=dtype, non_blocking=True) for index in range(batch_size)
            ],
            "context_ref": [
                prompt_ref_embeddings[index].to(device=device, dtype=dtype, non_blocking=True)
                for index in range(batch_size)
            ],
            "seq_len": sequence_length,
            "seq_len_ref": sequence_length_ref,
            "grid_sizes_ref": grid_sizes_ref,
            "timestep": context.timesteps.to(device=device, dtype=dtype),
            "origin_len": driving_pixel_frames,
            "origin_area": [
                latent_height * _VAE_SPATIAL_COMPRESSION,
                latent_width * _VAE_SPATIAL_COMPRESSION,
            ],
            "_target_latent_frames": target_latent_frames,
            "_compute_dtype": dtype,
        }

    def forward(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
        """Run the reference and generation passes, returning target predictions.

        The reference pass populates a per-step key/value cache and the
        generation pass consumes it. Both carry gradient. A fresh cache is
        allocated on every call so no state leaks across steps.

        Args:
            model: Upstream Diffusers ``WanAnimate2Transformer3DModel``, whose
                ``forward`` dispatches on a required ``method`` keyword.
            inputs: Mapping returned by :meth:`prepare_inputs`. Tensor-bearing
                fields have the layouts documented by that method.

        Returns:
            Velocity prediction tensor of shape [batch, 16,
            target_latent_frames, latent_height, latent_width]. The leading
            reference-slot frame is sliced off so the prediction aligns
            element-wise with the pipeline's flow-matching target.
        """
        # One traversal of the blocks, each running the reference pass and then
        # the generation pass against a block-local cache. The Diffusers
        # integration exposes the two passes as separate entry points, each
        # walking all forty blocks; under FSDP2 that means two forwards per
        # module, whose resharding frees weights the second pass still needs, and
        # a caller-owned cache that the pre-forward path copies per block. The
        # interleaved traversal is what the reference training implementation
        # does and it avoids both. See interleaved.forward_origin.
        #
        # Autocast wraps the whole thing: the upstream blocks promote activations
        # to float32 around the modulation arithmetic
        # (`self.norm1(x).float() * (1 + e[1]) + e[0]`) and feed the result
        # straight into half-precision Linear layers, so without it the first
        # matmul raises "mat1 and mat2 must have the same dtype".
        if not install_forward_origin(model):
            raise RuntimeError(
                "This Wan-Animate-2 transformer does not expose the block methods the interleaved "
                "training forward needs (blocks with forward_ref/forward_gen). Check the installed "
                "diffusers version."
            )
        with torch.amp.autocast(device_type=inputs["x"][0].device.type, dtype=inputs["_compute_dtype"]):
            # Dispatched through the model's own `method` keyword so the call
            # goes through __call__ and FSDP2 gathers the parameters first.
            prediction = model(inputs, method="forward_origin")

        if not isinstance(prediction, (list, tuple)):
            raise TypeError(
                "WanAnimate2Transformer3DModel.forward_gen must return a list of per-sample tensors, "
                f"got {type(prediction)!r}"
            )
        stacked = torch.stack(list(prediction), dim=0)

        target_latent_frames = inputs["_target_latent_frames"]
        sliced = stacked[:, :, 1:]
        if sliced.shape[2] != target_latent_frames:
            raise ValueError(f"Sliced prediction has {sliced.shape[2]} latent frames, expected {target_latent_frames}")
        return sliced
