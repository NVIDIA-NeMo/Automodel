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
    - encoder_hidden_states: Text embeddings [B, text_len, 4096]
    - timestep: Normalized timesteps [0, 1]
    - img_shapes: [[(1, H, W)]] per sample
    - img_mask: [B, text_len + H*W/4] bool, True at target-image slots

    The transformer lays out RoPE from row 0 of ``img_mask`` for the whole batch, and every text position
    (padding included) advances the target image's frame index. Each sample is therefore run as its own
    unpadded transformer call, so it sees exactly the positions of single-prompt inference.
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

    @staticmethod
    def _build_img_mask(batch_size: int, text_len: int, image_tokens: int, device: torch.device) -> torch.Tensor:
        """Text positions first, then one slot per 2x2 group of target latent tokens."""
        return torch.cat(
            [
                torch.zeros(batch_size, text_len, dtype=torch.bool, device=device),
                torch.ones(batch_size, image_tokens // _IMG_TOKENS_PER_SLOT, dtype=torch.bool, device=device),
            ],
            dim=1,
        )

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

        # Prompts are right-padded, so each sample's valid tokens are a prefix of its row.
        text_lengths = text_mask.sum(dim=1).tolist()
        text_embeddings = text_embeddings[:, : max(text_lengths)]

        if random.random() < context.cfg_dropout_prob:
            text_embeddings = torch.zeros_like(text_embeddings)

        # Normalize timesteps to [0, 1]
        timesteps = context.timesteps.to(dtype) / 1000.0

        return {
            "hidden_states": self._pack_latents(noisy_latents),
            "encoder_hidden_states": text_embeddings,
            "timestep": timesteps,
            "img_shapes": [[(1, height, width)]] * batch_size,
            "_text_lengths": text_lengths,
            "_original_shape": (batch_size, channels, height, width),
        }

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute forward pass for Qwen-Image-2.1 model.

        Runs one transformer call per sample, trimmed to that sample's prompt length, and returns the
        target-image prediction in [B, C, H, W] format.

        The call count is always the local batch size, never the number of distinct prompt lengths: under FSDP
        every call issues collectives, so all ranks must make the same number of calls, and the diffusion
        sampler gives every rank the same local batch size.
        """
        _, _, height, width = inputs["_original_shape"]
        image_tokens = height * width
        hidden_states = inputs["hidden_states"]

        preds = []
        for index, text_len in enumerate(inputs["_text_lengths"]):
            model_pred = model(
                hidden_states=hidden_states[index : index + 1],
                encoder_hidden_states=inputs["encoder_hidden_states"][index : index + 1, :text_len],
                encoder_hidden_states_mask=None,
                timestep=inputs["timestep"][index : index + 1],
                img_shapes=inputs["img_shapes"][index : index + 1],
                img_mask=self._build_img_mask(1, text_len, image_tokens, hidden_states.device),
                return_dict=False,
            )
            # The transformer predicts the whole joint sequence; the target image is the trailing block.
            preds.append(self.post_process_prediction(model_pred)[:, -image_tokens:])

        return self._unpack_latents(torch.cat(preds), height, width)
