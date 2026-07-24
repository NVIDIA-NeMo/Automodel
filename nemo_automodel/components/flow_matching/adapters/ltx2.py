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
LTX-2 dual-stream (video + audio) model adapter for the FlowMatching pipeline.

LTX-2's transformer jointly denoises a video token stream and an audio token
stream with cross-modal attention. The pipeline owns noising and loss for the
primary (video) latents; this adapter additionally noises the audio latents
with the SAME per-sample sigma (independent noise draw), runs the dual-stream
forward, and reports the audio flow-matching loss through the
``auxiliary_losses`` hook. Both streams therefore train from a single forward
and a single backward pass.

Expected batch keys (produced by ``tools/diffusion/processors/ltx2.py``):
- video_latents: [B, 128, F, H, W] (consumed by the pipeline)
- audio_latents: [B, 8, L, 16]
- text_embeddings: [B, T, D_v] video-stream connector output
- audio_text_embeddings: [B, T, D_a] audio-stream connector output
- text_mask: [B, T] post-connector attention mask
"""

import inspect
import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import FlowMatchingContext, ModelAdapter

logger = logging.getLogger(__name__)

# Fixed by the LTX-2 audio VAE: 8 latent channels x 16 latent mel bins.
_AUDIO_LATENT_MEL_BINS = 16

_MISSING_KEY_HINT = (
    "LTX2Adapter requires '{key}' in the batch. Preprocess your dataset with the "
    "'ltx2' processor (tools/diffusion/preprocessing_multiprocess.py --processor ltx2), "
    "which caches audio latents and per-stream text embeddings."
)


def _pack_video_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack video latents [B, C, F, H, W] -> tokens [B, F*H*W, C] (patch size 1)."""
    return latents.flatten(2).transpose(1, 2)


def _unpack_video_latents(tokens: torch.Tensor, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Unpack video tokens [B, F*H*W, C] -> latents [B, C, F, H, W] (patch size 1).

    Args:
        tokens: Video token sequence [B, F*H*W, C].
        num_frames: Latent frame count F.
        height: Latent height H.
        width: Latent width W.

    Returns:
        Video latents [B, C, F, H, W].
    """
    return tokens.transpose(1, 2).unflatten(2, (num_frames, height, width))


def _pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack audio latents [B, C, L, M] -> tokens [B, L, C*M]."""
    return latents.transpose(1, 2).flatten(2, 3)


def _unpack_audio_latents(tokens: torch.Tensor, num_mel_bins: int) -> torch.Tensor:
    """Unpack audio tokens [B, L, C*M] -> latents [B, C, L, M].

    Args:
        tokens: Audio token sequence [B, L, C*M].
        num_mel_bins: Latent mel bin count M (16 for LTX-2).

    Returns:
        Audio latents [B, C, L, M].
    """
    return tokens.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)


class LTX2Adapter(ModelAdapter):
    """
    Model adapter for LTX-2 dual-stream (video + audio) transformers.

    The adapter is stateless: all per-step tensors live in the ``inputs`` dict
    created fresh by each ``prepare_inputs()`` call, so gradient accumulation
    over micro-batches is safe.

    Args:
        audio_loss_weight: Multiplier on the audio flow-matching MSE loss
            added to the video loss (LTX-2 reference training uses 1.0).
        fps: Video frame rate the model was trained at (LTX-2 uses 24).
    """

    def __init__(self, audio_loss_weight: float = 1.0, fps: float = 24.0):
        self.audio_loss_weight = audio_loss_weight
        self.fps = fps

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        """
        Prepare dual-stream transformer inputs; noise the audio latents.

        The video latents are already noised by the pipeline
        (``context.noisy_latents``). Audio latents are noised here with the
        same per-sample sigma and an independent Gaussian draw, in float32
        (matching the pipeline's video noising), then cast to the model dtype.

        Args:
            context: FlowMatchingContext with batch data. ``context.timesteps``
                is the rescaled timestep (sigma * num_train_timesteps) [B];
                ``context.sigma`` is the raw sigma in [0, 1] [B].

        Returns:
            Dictionary containing the transformer kwargs:
            - hidden_states: packed noisy video tokens [B, F*H*W, 128]
            - audio_hidden_states: packed noisy audio tokens [B, L, 128]
            - encoder_hidden_states / audio_encoder_hidden_states: per-stream
              text embeddings [B, T, D]
            - encoder_attention_mask / audio_encoder_attention_mask: [B, T]
            - timestep: per-token video timesteps [B, F*H*W]
            - audio_timestep: per-token audio timesteps [B, L]
            - sigma / audio_sigma: rescaled scalar timesteps [B]
            - num_frames, height, width, fps, audio_num_frames: geometry
            plus the private key ``_audio_target`` (float32 [B, 8, L, 16],
            the audio flow velocity target) consumed by ``auxiliary_losses``.

        Raises:
            KeyError: If the batch lacks LTX-2 audio/text cache keys.
        """
        batch = context.batch
        device = context.device
        dtype = context.dtype

        for key in ("audio_latents", "text_embeddings", "audio_text_embeddings", "text_mask"):
            if batch.get(key) is None:
                raise KeyError(_MISSING_KEY_HINT.format(key=key))

        # ---- Audio stream: noise with the shared sigma, independent draw ----
        audio_latents = batch["audio_latents"].to(device=device, dtype=torch.float32, non_blocking=True)
        audio_noise = torch.randn_like(audio_latents)
        sigma = context.sigma.to(device=device, dtype=torch.float32).view(-1, 1, 1, 1)
        noisy_audio = (1.0 - sigma) * audio_latents + sigma * audio_noise

        # ---- Pack token streams ----
        batch_size, _, num_frames, height, width = context.noisy_latents.shape
        hidden_states = _pack_video_latents(context.noisy_latents)
        audio_hidden_states = _pack_audio_latents(noisy_audio.to(dtype))
        audio_num_frames = audio_hidden_states.shape[1]

        # ---- Text conditioning (per-stream connector outputs from the cache) ----
        text_embeddings = batch["text_embeddings"].to(device=device, dtype=dtype, non_blocking=True)
        audio_text_embeddings = batch["audio_text_embeddings"].to(device=device, dtype=dtype, non_blocking=True)
        text_mask = batch["text_mask"].to(device=device, non_blocking=True)

        # ---- Timesteps: rescaled scalar [B] expanded per-token per stream ----
        timestep_scalar = context.timesteps.to(device=device, dtype=torch.float32)
        video_timestep = timestep_scalar.unsqueeze(1).expand(batch_size, hidden_states.shape[1])
        audio_timestep = timestep_scalar.unsqueeze(1).expand(batch_size, audio_num_frames)

        return {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": text_embeddings,
            "audio_encoder_hidden_states": audio_text_embeddings,
            "encoder_attention_mask": text_mask,
            "audio_encoder_attention_mask": text_mask,
            "timestep": video_timestep,
            "audio_timestep": audio_timestep,
            "sigma": timestep_scalar,
            "audio_sigma": timestep_scalar,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "fps": self.fps,
            "audio_num_frames": audio_num_frames,
            # Private stash (not passed to the model): audio flow velocity target.
            "_audio_target": audio_noise - audio_latents,
        }

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute the dual-stream forward pass.

        Calls the LTX-2 transformer with the kwargs from ``prepare_inputs()``
        (filtered to the model's forward signature so minor diffusers-version
        differences in optional kwargs don't break the call). The raw
        transformer outputs ARE the flow predictions (no conversion).

        Args:
            model: The LTX-2 transformer (possibly FSDP-wrapped).
            inputs: Dictionary from prepare_inputs(). Mutated: the unpacked
                audio prediction is stashed under ``_audio_pred`` for
                ``auxiliary_losses()``.

        Returns:
            Video flow prediction [B, 128, F, H, W].
        """
        model_kwargs = self._filter_model_kwargs(model, inputs)
        output = model(**model_kwargs, return_dict=False)

        if isinstance(output, tuple):
            video_tokens, audio_tokens = output[0], output[1]
        else:
            video_tokens = output.sample
            audio_tokens = output.audio_sample

        inputs["_audio_pred"] = _unpack_audio_latents(audio_tokens, _AUDIO_LATENT_MEL_BINS)
        return _unpack_video_latents(video_tokens, inputs["num_frames"], inputs["height"], inputs["width"])

    def auxiliary_losses(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor] | None:
        """
        Compute the audio flow-matching loss from stashed tensors.

        Args:
            inputs: Dictionary from prepare_inputs() after forward() has run;
                must contain ``_audio_pred`` [B, 8, L, 16] and ``_audio_target``
                (float32 [B, 8, L, 16]).

        Returns:
            {"audio_loss": scalar} - unweighted MSE in float32 scaled by
            ``audio_loss_weight``, matching the LTX-2 reference training loss.
        """
        audio_loss = nn.functional.mse_loss(inputs["_audio_pred"].float(), inputs["_audio_target"])
        return {"audio_loss": self.audio_loss_weight * audio_loss}

    @staticmethod
    def _filter_model_kwargs(model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Drop private stash keys and kwargs the model's forward doesn't accept.

        Args:
            model: The transformer (possibly wrapped; unwrapped via ``.module``
                for signature inspection only - the wrapped module is called).
            inputs: Full inputs dict from prepare_inputs().

        Returns:
            Kwargs safe to splat into ``model(...)``.
        """
        model_kwargs = {k: v for k, v in inputs.items() if not k.startswith("_")}
        unwrapped = getattr(model, "module", model)
        try:
            parameters = inspect.signature(unwrapped.forward).parameters
        except (TypeError, ValueError):
            return model_kwargs
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return model_kwargs
        dropped = sorted(set(model_kwargs) - set(parameters))
        if dropped:
            logger.debug(
                "LTX2Adapter: dropping kwargs not accepted by %s.forward: %s", type(unwrapped).__name__, dropped
            )
        return {k: v for k, v in model_kwargs.items() if k in parameters}
