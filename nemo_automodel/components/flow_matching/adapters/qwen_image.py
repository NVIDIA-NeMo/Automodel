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
Qwen-Image model adapter for FlowMatching Pipeline.

This adapter supports Qwen/Qwen-Image style T2I models with:
- Qwen2 text embeddings (text_embeddings)
- 2D image latents ([B, C, H, W])
- 2x2 patch packing similar to Flux
"""

import random
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import FlowMatchingContext, ModelAdapter


class QwenImageAdapter(ModelAdapter):
    """
    Model adapter for Qwen-Image text-to-image models.

    Supports batch format from multiresolution dataloader:
    - image_latents: [B, C, H, W]
    - text_embeddings: Qwen2 embeddings [B, seq_len, hidden_dim]

    Qwen-Image transformer forward interface:
    - hidden_states: Packed latents [B, num_patches, C*4]
    - encoder_hidden_states: Qwen2 text embeddings [B, seq_len, hidden_dim]
    - timestep: Normalized timesteps [0, 1]
    - img_ids: Image positional embeddings [num_patches, 3]
    - txt_ids: Text positional embeddings [B, seq_len, 3]
    """

    def __init__(
        self,
        guidance_scale: float = 3.5,
        use_guidance_embeds: bool = False,
    ):
        """
        Initialize QwenImageAdapter.

        Args:
            guidance_scale: Guidance scale for classifier-free guidance
            use_guidance_embeds: Whether to use guidance embeddings
        """
        self.guidance_scale = guidance_scale
        self.use_guidance_embeds = use_guidance_embeds

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pack latents from [B, C, H, W] to [B, (H//2)*(W//2), C*4].

        Uses 2x2 patch grouping to match the transformer's patch embedding.
        """
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int = 8) -> torch.Tensor:
        """
        Unpack latents from [B, num_patches, channels] back to [B, C, H, W].

        Args:
            latents: Packed latents of shape [B, num_patches, channels]
            height: Original image height in pixels
            width: Original image width in pixels
            vae_scale_factor: VAE compression factor (default: 8)
        """
        batch_size, num_patches, channels = latents.shape

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

        Returns tensor of shape [(H//2)*(W//2), 3] containing (0, y, x).
        """
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = torch.arange(width // 2)[None, :]

        latent_image_ids = latent_image_ids.reshape(-1, 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        """
        Prepare inputs for Qwen-Image model from FlowMatchingContext.

        Expects 4D image latents: [B, C, H, W]
        """
        batch = context.batch
        device = context.device
        dtype = context.dtype

        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 4:
            raise ValueError(f"QwenImageAdapter expects 4D latents [B, C, H, W], got {noisy_latents.ndim}D")

        batch_size, channels, height, width = noisy_latents.shape

        # Get text embeddings from Qwen2 encoder
        text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)

        if random.random() < context.cfg_dropout_prob:
            text_embeddings = torch.zeros_like(text_embeddings)

        # Pack latents for patch-based transformer
        packed_latents = self._pack_latents(noisy_latents)

        # Prepare positional IDs for image patches
        img_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        # Text positional IDs (zeros, model uses learned text positions)
        text_seq_len = text_embeddings.shape[1]
        txt_ids = torch.zeros(batch_size, text_seq_len, 3, device=device, dtype=dtype)

        # Normalize timesteps to [0, 1]
        timesteps = context.timesteps.to(dtype) / 1000.0

        inputs = {
            "hidden_states": packed_latents,
            "encoder_hidden_states": text_embeddings,
            "timestep": timesteps,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "_original_shape": (batch_size, channels, height, width),
        }

        return inputs

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute forward pass for Qwen-Image model.

        Returns unpacked prediction in [B, C, H, W] format.
        """
        original_shape = inputs.pop("_original_shape")
        batch_size, channels, height, width = original_shape

        model_pred = model(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            timestep=inputs["timestep"],
            img_ids=inputs["img_ids"],
            txt_ids=inputs["txt_ids"],
            return_dict=False,
        )

        pred = self.post_process_prediction(model_pred)

        # Unpack from patch format back to [B, C, H, W]
        vae_scale_factor = 8
        pred = self._unpack_latents(pred, height * vae_scale_factor, width * vae_scale_factor)

        return pred
