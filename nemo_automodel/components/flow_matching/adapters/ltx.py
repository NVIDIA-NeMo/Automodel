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
LTX-Video model adapter for FlowMatching Pipeline.

This adapter supports LTX-Video style models with:
- T5 text embeddings with attention masks
- 3D latent packing/unpacking ([B, C, F, H, W] <-> [B, S, D])
- RoPE positional embeddings via num_frames, height, width
"""

import random
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import FlowMatchingContext, ModelAdapter


class LTXAdapter(ModelAdapter):
    """
    Model adapter for LTX-Video models.

    LTX-Video uses a DiT transformer with:
    - hidden_states: Packed latents [B, S, D]
    - encoder_hidden_states: T5 text embeddings [B, seq_len, dim]
    - encoder_attention_mask: Text attention mask [B, seq_len]
    - timestep: Timestep values
    - num_frames, height, width: For RoPE positional embeddings

    Expected batch keys:
    - text_embeddings: T5 encoder output [B, seq_len, dim]
    - text_mask: Attention mask for T5 [B, seq_len] (optional)

    Example:
        adapter = LTXAdapter()
        pipeline = FlowMatchingPipeline(model_adapter=adapter)
    """

    def __init__(
        self,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ):
        """
        Initialize LTXAdapter.

        Args:
            patch_size: Spatial patch size for latent packing (default 1)
            patch_size_t: Temporal patch size for latent packing (default 1)
        """
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t

    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        """
        Pack latents from [B, C, F, H, W] to [B, S, D] for the transformer.

        Matches the packing used in diffusers LTXPipeline._pack_latents.
        """
        batch_size, num_channels, num_frames, height, width = latents.shape
        latents = latents.reshape(
            batch_size,
            num_channels,
            num_frames // patch_size_t,
            patch_size_t,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        """
        Unpack latents from [B, S, D] back to [B, C, F, H, W].

        Matches the unpacking used in diffusers LTXPipeline._unpack_latents.
        """
        batch_size = latents.size(0)
        latents = latents.reshape(
            batch_size,
            num_frames,
            height,
            width,
            -1,
            patch_size_t,
            patch_size,
            patch_size,
        )
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
        latents = latents.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        """
        Prepare inputs for LTX-Video transformer.

        Args:
            context: FlowMatchingContext with batch data

        Returns:
            Dictionary of inputs for the transformer forward method
        """
        batch = context.batch
        device = context.device
        dtype = context.dtype

        # LTX expects 5D video latents [B, C, F, H, W]
        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 5:
            raise ValueError(f"LTXAdapter expects 5D latents [B, C, F, H, W], got {noisy_latents.ndim}D")

        batch_size, channels, num_frames, height, width = noisy_latents.shape

        # Get text embeddings
        text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)

        # Get attention mask
        text_mask = batch.get("text_mask")
        if text_mask is not None:
            text_mask = text_mask.to(device, dtype=dtype)
        else:
            # Create all-ones mask if not provided
            text_mask = torch.ones(batch_size, text_embeddings.shape[1], device=device, dtype=dtype)

        # Handle CFG dropout
        if random.random() < context.cfg_dropout_prob:
            text_embeddings = torch.zeros_like(text_embeddings)
            text_mask = torch.zeros_like(text_mask)

        # Pack latents: [B, C, F, H, W] -> [B, S, D]
        packed_latents = self._pack_latents(noisy_latents, self.patch_size, self.patch_size_t)

        # Compute post-patch dimensions for RoPE
        post_patch_num_frames = num_frames // self.patch_size_t
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size

        return {
            "hidden_states": packed_latents,
            "encoder_hidden_states": text_embeddings,
            "timestep": context.timesteps.to(dtype),
            "encoder_attention_mask": text_mask,
            "num_frames": post_patch_num_frames,
            "height": post_patch_height,
            "width": post_patch_width,
            # Store original shape for unpacking
            "_original_shape": (
                batch_size,
                channels,
                num_frames,
                height,
                width,
            ),
        }

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute forward pass for LTX-Video transformer.

        Returns unpacked prediction in [B, C, F, H, W] format.
        """
        original_shape = inputs.pop("_original_shape")
        _, _, num_frames, height, width = original_shape

        model_pred = model(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            timestep=inputs["timestep"],
            encoder_attention_mask=inputs["encoder_attention_mask"],
            num_frames=inputs["num_frames"],
            height=inputs["height"],
            width=inputs["width"],
            return_dict=False,
        )

        pred = self.post_process_prediction(model_pred)

        # Unpack from [B, S, D] back to [B, C, F, H, W]
        post_patch_num_frames = num_frames // self.patch_size_t
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size

        pred = self._unpack_latents(
            pred,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            self.patch_size,
            self.patch_size_t,
        )

        return pred
