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
Flux2 model adapter for FlowMatching Pipeline.

Supports FLUX.2-dev with:
- Mistral3 text embeddings (text_embeddings, shape [B, seq, 15360])
- No CLIP pooled projections
- Patchified + BN-normalized image latents ([B, 128, H/16, W/16])
  stored by the preprocessor, noise is added in this space
- 4D positional IDs (T, H, W, L) instead of Flux1's 3D (batch, y, x)
"""

import random
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import FlowMatchingContext, ModelAdapter


class Flux2Adapter(ModelAdapter):
    """
    Model adapter for FLUX.2-dev image generation models.

    The preprocessor stores latents already patchified (2x2 spatial patch → channel)
    and BN-normalized using vae.bn running statistics.  The FlowMatchingPipeline adds
    noise in that space, so this adapter only needs to flatten the spatial dims (pack)
    before the transformer call and reshape back (unpack) after.

    Batch format expected from multiresolution dataloader:
    - image_latents: patchified + BN-normalized latents [B, 128, H_p, W_p]
    - text_embeddings: Mistral3 stacked embeddings [B, seq_len, 15360]

    FLUX.2 transformer forward interface:
    - hidden_states: packed latents [B, H_p*W_p, 128]
    - encoder_hidden_states: text embeddings [B, seq_len, 15360]
    - timestep: normalized [0, 1]
    - img_ids: 4D spatial coords [B, H_p*W_p, 4]
    - txt_ids: 4D text coords [B, seq_len, 4]
    - guidance: guidance scale [B]
    """

    def __init__(
        self,
        guidance_scale: float = 3.5,
        use_guidance_embeds: bool = True,
    ):
        """
        Args:
            guidance_scale: Guidance scale value (3.5 matches FLUX.2-dev default).
            use_guidance_embeds: Whether to pass guidance tensor to the transformer.
        """
        self.guidance_scale = guidance_scale
        self.use_guidance_embeds = use_guidance_embeds

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Flatten spatial dims: [B, C, H, W] -> [B, H*W, C]."""
        b, c, h, w = latents.shape
        return latents.reshape(b, c, h * w).permute(0, 2, 1)

    def _unpack_latents(self, latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Restore spatial dims: [B, seq, C] -> [B, C, H, W]."""
        b, _, c = latents.shape
        return latents.permute(0, 2, 1).reshape(b, c, h, w)

    def _prepare_latent_ids(self, h_p: int, w_p: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Build 4D image positional IDs with coords (T=0, h, w, L=0).

        Returns [B, H_p*W_p, 4] long tensor.
        """
        ids = torch.zeros(h_p * w_p, 4, dtype=torch.long, device=device)
        # T and L remain 0; fill H and W coords
        ids[:, 1] = torch.arange(h_p, device=device).repeat_interleave(w_p)
        ids[:, 2] = torch.arange(w_p, device=device).repeat(h_p)
        return ids.unsqueeze(0).expand(batch_size, -1, -1)

    def _prepare_text_ids(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Build 4D text positional IDs with coords (T=0, H=0, W=0, position).

        Returns [B, seq_len, 4] long tensor.
        """
        ids = torch.zeros(seq_len, 4, dtype=torch.long, device=device)
        ids[:, 3] = torch.arange(seq_len, device=device)
        return ids.unsqueeze(0).expand(batch_size, -1, -1)

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        """
        Prepare inputs for FLUX.2 transformer from FlowMatchingContext.

        Expects 4D patchified+BN-normalized latents [B, 128, H_p, W_p].
        """
        batch = context.batch
        device = context.device
        dtype = context.dtype

        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 4:
            raise ValueError(f"Flux2Adapter expects 4D patchified latents [B, C, H_p, W_p], got {noisy_latents.ndim}D")

        batch_size, channels, h_p, w_p = noisy_latents.shape

        # Pack: [B, 128, H_p, W_p] -> [B, H_p*W_p, 128]
        packed = self._pack_latents(noisy_latents)

        # Text embeddings from Mistral3 [B, seq, 15360]
        text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)

        # CFG dropout: zero entire text condition with probability cfg_dropout_prob
        if random.random() < context.cfg_dropout_prob:
            text_embeddings = torch.zeros_like(text_embeddings)

        seq_len = text_embeddings.shape[1]

        # 4D positional IDs (computed dynamically — not stored in cache)
        img_ids = self._prepare_latent_ids(h_p, w_p, batch_size, device)
        txt_ids = self._prepare_text_ids(seq_len, batch_size, device)

        # Timesteps normalized to [0, 1]
        timesteps = context.timesteps.to(dtype) / 1000.0

        guidance = None
        if self.use_guidance_embeds:
            guidance = torch.full((batch_size,), self.guidance_scale, device=device, dtype=torch.float32)

        return {
            "hidden_states": packed,
            "encoder_hidden_states": text_embeddings,
            "timestep": timesteps,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance,
            "_h_p": h_p,
            "_w_p": w_p,
        }

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute FLUX.2 transformer forward pass.

        Returns prediction in [B, 128, H_p, W_p] — same space as the stored
        patchified latents, so MSE loss against (noise - latents) is correct.
        """
        h_p = inputs.pop("_h_p")
        w_p = inputs.pop("_w_p")

        model_pred = model(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            timestep=inputs["timestep"],
            img_ids=inputs["img_ids"],
            txt_ids=inputs["txt_ids"],
            guidance=inputs["guidance"],
            return_dict=False,
        )

        pred = self.post_process_prediction(model_pred)

        # Unpack: [B, H_p*W_p, 128] -> [B, 128, H_p, W_p]
        return self._unpack_latents(pred, h_p, w_p)
