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

"""
Flux3 model adapter for FlowMatching Pipeline.

Supports FLUX.3 finetuning for two image workloads, dispatched from the batch:
- Text-to-image: 4D patchified + BN-normalized image latents [B, 128, H_p, W_p]
- Image-to-image: t2i plus a clean conditioning image stream
  (``batch["context_latents"]``, position ids with l = -1, timestep 0)

Common FLUX.3 conventions (vs Flux1/2):
- Qwen3-VL text embeddings (text_embeddings, [B, seq, 20480])
- No CLIP pooled projections and no guidance embedding (FLUX.3 is not
  guidance-distilled)
- Per-sample timesteps ([B]) and unbatched float32 4D position ids
  ([seq, 4], axes (t, h, w, l)); image H/W axes use the 0.5 RoPE scale
"""

from typing import TypedDict

import torch
import torch.nn as nn

from .base import FlowMatchingContext, ModelAdapter

# FLUX.3 image RoPE scale applied to the H/W position-id axes (matches the
# flux3 modular pipeline and the upstream training reference).
_IMAGE_ROPE_SCALE = 0.5


class Flux3ConditioningInputs(TypedDict, total=False):
    """Optional reference-image tokens and their position IDs."""

    image_cond_hidden_states: torch.Tensor
    image_cond_ids: torch.Tensor


class Flux3Inputs(Flux3ConditioningInputs):
    """Image transformer inputs and the latent grid needed to unpack its output."""

    encoder_hidden_states: torch.Tensor
    txt_ids: torch.Tensor
    hidden_states: torch.Tensor
    img_ids: torch.Tensor
    timestep: torch.Tensor
    _h_p: int
    _w_p: int


class Flux3Adapter(ModelAdapter):
    """
    Model adapter for FLUX.3 text-to-image and image-to-image.

    The preprocessor stores latents already in the transformer's expected space
    (2x2-patchified + BN-normalized). The FlowMatchingPipeline adds noise in
    that space, so this adapter only packs tokens for the transformer call and
    unpacks the prediction back so MSE loss against (noise - latents) is
    elementwise aligned.

    Mode dispatch in ``prepare_inputs``:
    - 4D with ``batch["context_latents"]`` -> image-to-image (adds the clean
      ``image_cond_*`` stream: same latent space as the target, ids with l = -1,
      timestep 0; the transformer returns only generated tokens, so the loss
      covers only the generated stream)
    - 4D otherwise -> text-to-image
    """

    # ------------------------------------------------------------------
    # Packing helpers
    # ------------------------------------------------------------------

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Flatten spatial dims.

        Args:
            latents: Tensor of shape [batch, channels, height, width].

        Returns:
            Tensor of shape [batch, height*width, channels].
        """
        b, c, h, w = latents.shape
        return latents.reshape(b, c, h * w).permute(0, 2, 1).contiguous()

    def _unpack_latents(self, latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Restore spatial dims.

        Args:
            latents: Tensor of shape [batch, height*width, channels].
            h: Latent height.
            w: Latent width.

        Returns:
            Tensor of shape [batch, channels, height, width].
        """
        b, _, c = latents.shape
        return latents.permute(0, 2, 1).contiguous().reshape(b, c, h, w)

    # ------------------------------------------------------------------
    # Position-id helpers
    # ------------------------------------------------------------------

    def _prepare_latent_ids(self, h_p: int, w_p: int, device: torch.device) -> torch.Tensor:
        """
        Build 4D image position ids with axes (t=0, h, w, l=0).

        The H/W coordinates are scaled by the FLUX.3 image RoPE scale (0.5).
        FLUX.3 position ids are unbatched and shared across the batch.

        Returns:
            Float32 tensor of shape [h_p*w_p, 4].
        """
        ids = torch.zeros(h_p * w_p, 4, dtype=torch.float32, device=device)
        ids[:, 1] = torch.arange(h_p, device=device, dtype=torch.float32).repeat_interleave(w_p) * _IMAGE_ROPE_SCALE
        ids[:, 2] = torch.arange(w_p, device=device, dtype=torch.float32).repeat(h_p) * _IMAGE_ROPE_SCALE
        return ids

    def _prepare_image_cond_ids(self, h_p: int, w_p: int, device: torch.device) -> torch.Tensor:
        """
        Build 4D conditioning-image position ids: like image ids but l = -1.

        Returns:
            Float32 tensor of shape [h_p*w_p, 4].
        """
        ids = self._prepare_latent_ids(h_p, w_p, device)
        ids[:, 3] = -1.0
        return ids

    def _prepare_text_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build 4D text position ids with axes (t=0, h=0, w=0, l=position).

        Returns:
            Float32 tensor of shape [seq_len, 4].
        """
        ids = torch.zeros(seq_len, 4, dtype=torch.float32, device=device)
        ids[:, 3] = torch.arange(seq_len, device=device, dtype=torch.float32)
        return ids

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def prepare_inputs(self, context: FlowMatchingContext) -> Flux3Inputs:
        """
        Prepare FLUX.3 transformer inputs from FlowMatchingContext.

        Expects ``context.noisy_latents`` of shape [batch, 128, H_p, W_p],
        text embeddings in ``context.batch["text_embeddings"]`` of shape
        [batch, seq_len, 20480], and, for image-to-image,
        ``context.batch["context_latents"]`` as a single-entry list of clean
        conditioning latents [batch, 128, H_c, W_c].

        Returns:
            Dict of transformer forward kwargs plus the private latent grid
            dims (``_h_p``, ``_w_p``) used by ``forward()`` to unpack the
            prediction.
        """
        batch = context.batch
        device = context.device
        dtype = context.dtype

        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 4:
            raise ValueError(f"Flux3Adapter expects 4D image latents [B, C, H_p, W_p], got {noisy_latents.ndim}D")

        batch_size, _, h_p, w_p = noisy_latents.shape

        # Text embeddings from Qwen3-VL [B, seq, 20480]
        text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)

        # CFG dropout: independently zero each sample's text condition
        if context.cfg_dropout_prob > 0.0:
            mask = torch.rand(batch_size, 1, 1, device=device) < context.cfg_dropout_prob
            text_embeddings = text_embeddings.masked_fill(mask, 0.0)

        sigma = context.sigma.to(device=device, dtype=dtype)
        packed = self._pack_latents(noisy_latents)

        inputs: Flux3Inputs = {
            "encoder_hidden_states": text_embeddings,
            "txt_ids": self._prepare_text_ids(text_embeddings.shape[1], device),
            "hidden_states": packed,
            "img_ids": self._prepare_latent_ids(h_p, w_p, device),
            "timestep": sigma.reshape(batch_size),
            "_h_p": h_p,
            "_w_p": w_p,
        }

        context_latents = batch.get("context_latents")
        if context_latents:
            if len(context_latents) != 1:
                raise ValueError(
                    f"Flux3Adapter supports exactly one conditioning image stream, "
                    f"got {len(context_latents)} context latents"
                )
            # Clean reference stream: same latent space as the target, ids with
            # l = -1, and per-token conditioning timestep fixed at 0.
            cond = context_latents[0].to(device, dtype=dtype)
            if cond.ndim != 4:
                raise ValueError(f"Flux3Adapter expects 4D conditioning latents [B, C, H_c, W_c], got {cond.ndim}D")
            _, _, h_c, w_c = cond.shape
            packed_cond = self._pack_latents(cond)
            inputs.update(
                image_cond_hidden_states=packed_cond,
                image_cond_ids=self._prepare_image_cond_ids(h_c, w_c, device),
            )

        return inputs

    def forward(self, model: nn.Module, inputs: Flux3Inputs) -> torch.Tensor:
        """
        Execute the FLUX.3 transformer forward pass.

        Args:
            model: The Flux3Transformer2DModel to call.
            inputs: Dict from ``prepare_inputs()``; tensor layouts are documented
                there.

        Returns:
            Prediction tensor of shape [batch, 128, H_p, W_p], the same unpacked
            space as the stored latents, so MSE loss against (noise - latents)
            is correct. Only the generated stream is returned by the
            transformer; conditioning tokens produce no output.
        """
        h_p = inputs.pop("_h_p")
        w_p = inputs.pop("_w_p")

        model_pred = model(**inputs, return_dict=False)
        pred = self.post_process_prediction(model_pred)
        return self._unpack_latents(pred, h_p, w_p)
