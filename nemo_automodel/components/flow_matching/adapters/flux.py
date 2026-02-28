# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""
Flux model adapter for FlowMatching Pipeline.

This adapter supports FLUX.1 style models with:
- T5 text embeddings (text_embeddings)
- CLIP pooled embeddings (pooled_prompt_embeds)
- 2D image latents (treated as 1-frame video: [B, C, 1, H, W])
"""

import random
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import FlowMatchingContext, ModelAdapter


class FluxAdapter(ModelAdapter):
    """
    Model adapter for FLUX.1 image generation models.

    Supports batch format from multiresolution dataloader:
    - image_latents: [B, C, H, W] for images
    - text_embeddings: T5 embeddings [B, seq_len, 4096]
    - pooled_prompt_embeds: CLIP pooled [B, 768]

    FLUX model forward interface:
    - hidden_states: Packed latents
    - encoder_hidden_states: T5 text embeddings
    - pooled_projections: CLIP pooled embeddings
    - timestep: Normalized timesteps [0, 1]
    - img_ids / txt_ids: Positional embeddings
    """

    def __init__(
        self,
        guidance_scale: float = 3.5,
        use_guidance_embeds: bool = True,
    ):
        """
        Initialize FluxAdapter.

        Args:
            guidance_scale: Guidance scale for classifier-free guidance
            use_guidance_embeds: Whether to use guidance embeddings
        """
        self.guidance_scale = guidance_scale
        self.use_guidance_embeds = use_guidance_embeds

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pack latents from [B, C, H, W] to Flux format [B, (H//2)*(W//2), C*4].

        Flux uses a 2x2 patch embedding, so latents are reshaped accordingly.
        """
        b, c, h, w = latents.shape
        # Reshape: [B, C, H, W] -> [B, C, H//2, 2, W//2, 2]
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        # Permute: -> [B, H//2, W//2, C, 2, 2]
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        # Reshape: -> [B, (H//2)*(W//2), C*4]
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int = 8) -> torch.Tensor:
        """
        Unpack latents from Flux format back to [B, C, H, W].

        Args:
            latents: Packed latents of shape [B, num_patches, channels]
            height: Original image height in pixels
            width: Original image width in pixels
            vae_scale_factor: VAE compression factor (default: 8)
        """
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def _prepare_latent_image_ids(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Prepare positional IDs for image latents.

        Returns tensor of shape [B, (H//2)*(W//2), 3] containing (batch_idx, y, x).
        """
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = torch.arange(width // 2)[None, :]

        latent_image_ids = latent_image_ids.reshape(-1, 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        """
        Prepare inputs for Flux model from FlowMatchingContext.

        Expects 4D image latents: [B, C, H, W]
        """
        batch = context.batch
        device = context.device
        dtype = context.dtype

        # Flux only supports 4D image latents [B, C, H, W]
        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 4:
            raise ValueError(f"FluxAdapter expects 4D latents [B, C, H, W], got {noisy_latents.ndim}D")

        batch_size, channels, height, width = noisy_latents.shape

        # Get text embeddings (T5)
        text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)

        # Get pooled embeddings (CLIP) - may or may not be present
        if "pooled_prompt_embeds" in batch:
            pooled_projections = batch["pooled_prompt_embeds"].to(device, dtype=dtype)
        elif "clip_pooled" in batch:
            pooled_projections = batch["clip_pooled"].to(device, dtype=dtype)
        else:
            # Create zero embeddings if not provided
            pooled_projections = torch.zeros(batch_size, 768, device=device, dtype=dtype)

        if pooled_projections.ndim == 1:
            pooled_projections = pooled_projections.unsqueeze(0)

        if random.random() < context.cfg_dropout_prob:
            text_embeddings = torch.zeros_like(text_embeddings)
            pooled_projections = torch.zeros_like(pooled_projections)

        # Pack latents for Flux transformer
        packed_latents = self._pack_latents(noisy_latents)

        # Prepare positional IDs
        img_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        # Text positional IDs
        text_seq_len = text_embeddings.shape[1]
        txt_ids = torch.zeros(batch_size, text_seq_len, 3, device=device, dtype=dtype)

        # Timesteps - Flux expects normalized [0, 1] range
        # The pipeline provides timesteps in [0, num_train_timesteps]
        timesteps = context.timesteps.to(dtype) / 1000.0

        # TODO: guidance scale is different across pretraining and finetuning, we need pass it as a hyperparamters.
        # needs verify by Pranav
        guidance = torch.full((batch_size,), self.guidance_scale, device=device, dtype=torch.float32)

        inputs = {
            "hidden_states": packed_latents,
            "encoder_hidden_states": text_embeddings,
            "pooled_projections": pooled_projections,
            "timestep": timesteps,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            # Store original shape for unpacking
            "_original_shape": (batch_size, channels, height, width),
            "guidance": guidance,
        }

        return inputs

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute forward pass for Flux model.

        Returns unpacked prediction in [B, C, H, W] format.
        """
        original_shape = inputs.pop("_original_shape")
        batch_size, channels, height, width = original_shape

        # Flux forward pass
        model_pred = model(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            pooled_projections=inputs["pooled_projections"],
            timestep=inputs["timestep"],
            img_ids=inputs["img_ids"],
            txt_ids=inputs["txt_ids"],
            guidance=inputs["guidance"],
            return_dict=False,
        )

        # Handle tuple output
        pred = self.post_process_prediction(model_pred)

        # Unpack from Flux format back to [B, C, H, W]
        # Pass pixel dimensions (latent * vae_scale_factor) to _unpack_latents
        vae_scale_factor = 8
        pred = self._unpack_latents(pred, height * vae_scale_factor, width * vae_scale_factor)

        return pred
