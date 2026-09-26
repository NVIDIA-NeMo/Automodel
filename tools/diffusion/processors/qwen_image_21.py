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
Qwen-Image-2.1 model processor for preprocessing.

Handles Qwen/Qwen-Image-2.1 T2I models with:
- RGBA VAE (4 input channels, 64 latent channels, 16x spatial compression)
- Qwen3-VL text encoder for text conditioning
"""

import logging
from typing import Any, Dict

import torch
from torch import autocast

from .base import BaseModelProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("qwen_image_21")
class QwenImage21Processor(BaseModelProcessor):
    """
    Processor for Qwen-Image-2.1 T2I models.

    Qwen-Image-2.1 encodes RGBA images with a 64-channel VAE and conditions on
    Qwen3-VL hidden states. RGB training images are given an opaque alpha
    channel before VAE encoding, matching how the diffusers pipeline prepares
    condition images.
    """

    @property
    def model_type(self) -> str:
        return "qwen_image_21"

    @property
    def default_model_name(self) -> str:
        return "Qwen/Qwen-Image-2.1"

    def load_models(self, model_name: str, device: str) -> Dict[str, Any]:
        """
        Load Qwen-Image-2.1 models.

        Args:
            model_name: HuggingFace model path (e.g., 'Qwen/Qwen-Image-2.1')
            device: Device to load models on

        Returns:
            Dict containing:
                - vae: AutoencoderKLQwenImage21
                - pipeline: QwenImage21Pipeline without transformer (owns the
                  processor, Qwen3-VL text encoder and prompt template)
        """
        from diffusers import QwenImage21Pipeline

        from nemo_automodel._diffusers._hf_cache import resolve_diffusion_model_dir

        logger.info("[Qwen-Image-2.1] Loading models from %s...", model_name)

        model_name = resolve_diffusion_model_dir(model_name)

        # Load pipeline without transformer (not needed for preprocessing)
        pipeline = QwenImage21Pipeline.from_pretrained(
            model_name,
            transformer=None,
            torch_dtype=torch.bfloat16,
        )

        models = {}

        logger.info("  Configuring VAE...")
        models["vae"] = pipeline.vae.to(device=device, dtype=torch.bfloat16)
        models["vae"].eval()

        logger.info("  Configuring Qwen3-VL text encoder...")
        pipeline.text_encoder.to(device)
        pipeline.text_encoder.eval()

        # Keep pipeline for encode_prompt — it owns the processor, text_encoder,
        # prompt template, system-token dropping and final-norm bypass.
        models["pipeline"] = pipeline

        torch.cuda.empty_cache()

        logger.info("[Qwen-Image-2.1] Models loaded successfully!")
        return models

    def encode_image(
        self,
        image_tensor: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        """
        Encode image to latent space using VAE.

        Args:
            image_tensor: Image tensor (1, 3, H, W) or (1, 4, H, W), normalized to [-1, 1]
            models: Dict containing 'vae'
            device: Device to use

        Returns:
            Latent tensor (64, H//16, W//16), FP16
        """
        vae = models["vae"]
        image_tensor = image_tensor.to(device, dtype=torch.bfloat16)

        # The VAE is RGBA; give RGB inputs a fully opaque alpha channel (1.0 in [-1, 1] space).
        if image_tensor.shape[1] == 3:
            alpha = torch.ones_like(image_tensor[:, :1])
            image_tensor = torch.cat([image_tensor, alpha], dim=1)

        # Qwen-Image-2.1 VAE expects 5D input (B, C, T, H, W) — add frame dim for single image
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(2)

        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.sample()

        # Normalize using per-channel latents_mean / latents_std
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
        latent = (latent - latents_mean) / latents_std

        # Remove frame dim, then batch dim → (C, H, W)
        return latent.detach().cpu().to(torch.float16).squeeze(2).squeeze(0)

    def encode_text(
        self,
        prompt: str,
        models: Dict[str, Any],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text using the QwenImage21Pipeline's encode_prompt.

        Args:
            prompt: Text prompt
            models: Dict containing 'pipeline' (QwenImage21Pipeline)
            device: Device to use

        Returns:
            Dict containing:
                - prompt_embeds: Qwen3-VL hidden states [1, seq_len, hidden_dim]
        """
        pipeline = models["pipeline"]

        with torch.no_grad():
            prompt_embeds, _, image_pad_mask = pipeline.encode_prompt(
                prompt=prompt,
                device=device,
            )

        if image_pad_mask.any():
            raise ValueError("Qwen-Image-2.1 T2I prompt encoding unexpectedly produced image tokens")

        return {
            "prompt_embeds": prompt_embeds.detach().cpu().to(torch.bfloat16),
        }

    def verify_latent(
        self,
        latent: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> bool:
        """
        Verify latent can be decoded back to reasonable image.

        Args:
            latent: Encoded latent (C, H, W)
            models: Dict containing 'vae'
            device: Device to use

        Returns:
            True if verification passes
        """
        try:
            vae = models["vae"]
            device_type = "cuda" if "cuda" in device else "cpu"

            # (C, H, W) → (B, C, T, H, W)
            latent = latent.unsqueeze(0).unsqueeze(2).to(device).float()

            with torch.no_grad(), autocast(device_type=device_type, dtype=torch.float32):
                latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, latent.dtype)
                latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, latent.dtype)
                latent = latent * latents_std + latents_mean
                decoded = vae.decode(latent.to(vae.dtype)).sample

            # decoded is 5D (B, C, T, H, W) — take first frame; RGBA output
            decoded = decoded[:, :, 0]
            if decoded.shape[1] != 4:
                return False

            if torch.isnan(decoded).any() or torch.isinf(decoded).any():
                return False

            return True

        except Exception as e:
            logger.warning("[Qwen-Image-2.1] Verification failed: %s", e)
            return False

    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct cache dictionary for Qwen-Image-2.1.

        Args:
            latent: Encoded latent
            text_encodings: Dict from encode_text()
            metadata: Additional metadata

        Returns:
            Dict to save with torch.save()
        """
        return {
            # Image latent
            "latent": latent,
            # Text embeddings
            "prompt_embeds": text_encodings["prompt_embeds"],
            # Metadata
            "original_resolution": metadata["original_resolution"],
            "bucket_resolution": metadata["bucket_resolution"],
            "crop_offset": metadata["crop_offset"],
            "prompt": metadata["prompt"],
            "image_path": metadata["image_path"],
            "bucket_id": metadata["bucket_id"],
            "aspect_ratio": metadata["aspect_ratio"],
            # Model info
            "model_type": self.model_type,
        }
