# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
Qwen-Image-2.1 model adapter for FlowMatching Pipeline.

Qwen-Image-2.1 is a single-stream, block-causal DiT. Unlike Qwen-Image it:
- consumes 64-channel latents unpatched (one token per 16x16 pixel tile),
- runs text and image tokens through one joint sequence whose layout is
  described by ``img_mask`` (one slot per 2x2 group of target latent tokens),
- predicts every joint-sequence token; only the trailing target-image tokens
  are supervised.
"""

import random
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import FlowMatchingContext, ModelAdapter

# Each ``img_mask`` slot stands for a 2x2 group of latent tokens.
_IMG_TOKENS_PER_SLOT = 4


class QwenImage21Adapter(ModelAdapter):
    """
    Model adapter for Qwen-Image-2.1 text-to-image models.

    Supports batch format from multiresolution dataloader:
    - image_latents: [B, 64, H, W]
    - text_embeddings: Qwen3-VL embeddings [B, seq_len, 4096], right-padded
    - text_attention_mask: optional [B, seq_len] bool marking valid text tokens

    Qwen-Image-2.1 transformer forward interface:
    - hidden_states: Flattened latents [B, H*W, 64]
    - encoder_hidden_states: Text embeddings [B, seq_len, 4096]
    - encoder_hidden_states_mask: [B, seq_len] bool, or None when nothing is padded
    - timestep: Normalized timesteps [0, 1]
    - img_shapes: [[(1, H, W)]] per sample
    - img_mask: [B, seq_len + H*W/4] bool, True at target-image slots
    """

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Flatten latents from [B, C, H, W] to [B, H*W, C] (no patching in 2.1)."""
        b, c, h, w = latents.shape
        return latents.reshape(b, c, h * w).transpose(1, 2)

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Restore [B, H*W, C] token predictions to [B, C, H, W]."""
        b, n, c = latents.shape
        if n != height * width:
            raise ValueError(f"Expected {height * width} target tokens for a {height}x{width} latent, got {n}")
        return latents.transpose(1, 2).reshape(b, c, height, width)

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        """
        Prepare inputs for Qwen-Image-2.1 model from FlowMatchingContext.

        Expects 4D image latents: [B, C, H, W] with even H and W.
        """
        batch = context.batch
        device = context.device
        dtype = context.dtype

        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 4:
            raise ValueError(f"QwenImage21Adapter expects 4D latents [B, C, H, W], got {noisy_latents.ndim}D")

        batch_size, channels, height, width = noisy_latents.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                f"Qwen-Image-2.1 latent height and width must be even (image sides a multiple of 32 px), "
                f"got {(height, width)}"
            )

        text_embeddings = batch["text_embeddings"].to(device, dtype=dtype, non_blocking=True)
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)

        text_mask = batch.get("text_attention_mask")
        if text_mask is None:
            text_mask = torch.ones(text_embeddings.shape[:2], dtype=torch.bool, device=device)
        else:
            text_mask = text_mask.to(device, dtype=torch.bool, non_blocking=True)
        if text_mask.shape != text_embeddings.shape[:2]:
            raise ValueError(
                f"text_attention_mask shape {tuple(text_mask.shape)} does not match text_embeddings "
                f"{tuple(text_embeddings.shape[:2])}"
            )

        # Drop trailing padding shared by every sample: the collate pads to a multiple of 8, and every extra
        # position shifts the target image's RoPE frame index away from what inference sees.
        text_len = int(text_mask.sum(dim=1).max().item())
        text_embeddings = text_embeddings[:, :text_len]
        text_mask = text_mask[:, :text_len]

        if random.random() < context.cfg_dropout_prob:
            text_embeddings = torch.zeros_like(text_embeddings)

        packed_latents = self._pack_latents(noisy_latents)

        # Text positions first, then one slot per 2x2 group of target latent tokens.
        img_mask = torch.cat(
            [
                torch.zeros(batch_size, text_len, dtype=torch.bool, device=device),
                torch.ones(batch_size, height * width // _IMG_TOKENS_PER_SLOT, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        img_shapes = [[(1, height, width)]] * batch_size

        # A mask with no padding carries no information and forces the slower masked attention path.
        encoder_hidden_states_mask = None if bool(text_mask.all()) else text_mask

        # Normalize timesteps to [0, 1]
        timesteps = context.timesteps.to(dtype) / 1000.0

        return {
            "hidden_states": packed_latents,
            "encoder_hidden_states": text_embeddings,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timesteps,
            "img_shapes": img_shapes,
            "img_mask": img_mask,
            "_original_shape": (batch_size, channels, height, width),
        }

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute forward pass for Qwen-Image-2.1 model.

        Returns target-image prediction in [B, C, H, W] format.
        """
        _, _, height, width = inputs["_original_shape"]

        model_pred = model(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            encoder_hidden_states_mask=inputs["encoder_hidden_states_mask"],
            timestep=inputs["timestep"],
            img_shapes=inputs["img_shapes"],
            img_mask=inputs["img_mask"],
            return_dict=False,
        )
        pred = self.post_process_prediction(model_pred)

        # The transformer predicts the whole joint sequence; the target image is the trailing block.
        pred = pred[:, -height * width :]
        return self._unpack_latents(pred, height, width)
