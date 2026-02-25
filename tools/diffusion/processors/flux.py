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
FLUX model processor for preprocessing.

Handles FLUX.1-dev and similar FLUX architecture models with:
- VAE for image encoding
- CLIP text encoder
- T5 text encoder
"""

import logging
from typing import Any, Dict

import torch
from torch import autocast

from .base import BaseModelProcessor
from .registry import ProcessorRegistry


logger = logging.getLogger(__name__)


@ProcessorRegistry.register("flux")
class FluxProcessor(BaseModelProcessor):
    """
    Processor for FLUX.1 architecture models.

    FLUX uses a VAE for image encoding and dual text encoders (CLIP + T5)
    for text conditioning.
    """

    @property
    def model_type(self) -> str:
        return "flux"

    @property
    def default_model_name(self) -> str:
        return "black-forest-labs/FLUX.1-dev"

    def load_models(self, model_name: str, device: str) -> Dict[str, Any]:
        """
        Load FLUX models from FluxPipeline.

        Args:
            model_name: HuggingFace model path (e.g., 'black-forest-labs/FLUX.1-dev')
            device: Device to load models on

        Returns:
            Dict containing:
                - vae: AutoencoderKL
                - clip_tokenizer: CLIPTokenizer
                - clip_encoder: CLIPTextModel
                - t5_tokenizer: T5TokenizerFast
                - t5_encoder: T5EncoderModel
        """
        from diffusers import FluxPipeline

        logger.info("[FLUX] Loading models from %s via FluxPipeline...", model_name)

        # Load pipeline without transformer (not needed for preprocessing)
        pipeline = FluxPipeline.from_pretrained(
            model_name,
            transformer=None,
            torch_dtype=torch.bfloat16,
        )

        models = {}

        logger.info("  Configuring VAE...")
        models["vae"] = pipeline.vae.to(device=device, dtype=torch.bfloat16)
        models["vae"].eval()
        logger.debug("VAE config: %s", models["vae"].config)
        logger.debug("VAE shift_factor: %s", models["vae"].config.shift_factor)
        logger.debug("VAE scaling_factor: %s", models["vae"].config.scaling_factor)

        # Extract CLIP components
        logger.info("  Configuring CLIP...")
        models["clip_tokenizer"] = pipeline.tokenizer
        models["clip_encoder"] = pipeline.text_encoder.to(device)
        models["clip_encoder"].eval()

        # Extract T5 components
        logger.info("  Configuring T5...")
        models["t5_tokenizer"] = pipeline.tokenizer_2
        models["t5_encoder"] = pipeline.text_encoder_2.to(device)
        models["t5_encoder"].eval()

        # Clean up pipeline reference to free memory
        del pipeline
        torch.cuda.empty_cache()

        logger.info("[FLUX] Models loaded successfully!")
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
            image_tensor: Image tensor (1, 3, H, W), normalized to [-1, 1]
            models: Dict containing 'vae'
            device: Device to use

        Returns:
            Latent tensor (C, H//8, W//8), FP16
        """
        vae = models["vae"]
        image_tensor = image_tensor.to(device, dtype=torch.bfloat16)

        device_type = "cuda" if "cuda" in device else "cpu"

        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.sample()

        # Apply scaling factor
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor

        # Return as FP16 to save space, remove batch dimension
        # Use detach() to ensure tensor can be serialized across process boundaries
        return latent.detach().cpu().to(torch.float16).squeeze(0)

    def encode_text(
        self,
        prompt: str,
        models: Dict[str, Any],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text using CLIP and T5.

        Args:
            prompt: Text prompt
            models: Dict containing tokenizers and encoders
            device: Device to use

        Returns:
            Dict containing:
                - clip_tokens: Token IDs
                - clip_hidden: Hidden states from CLIP
                - pooled_prompt_embeds: Pooled CLIP output
                - t5_tokens: T5 token IDs
                - prompt_embeds: T5 hidden states
        """
        device_type = "cuda" if "cuda" in device else "cpu"

        # CLIP encoding
        clip_tokens = models["clip_tokenizer"](
            prompt,
            padding="max_length",
            max_length=models["clip_tokenizer"].model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        clip_output = models["clip_encoder"](clip_tokens.input_ids.to(device_type), output_hidden_states=True)
        clip_hidden = clip_output.hidden_states[-2]
        pooled_prompt_embeds = clip_output.pooler_output

        # T5 encoding
        t5_tokens = models["t5_tokenizer"](
            prompt,
            padding="max_length",
            max_length=models["t5_tokenizer"].model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        t5_output = models["t5_encoder"](t5_tokens.input_ids.to(device_type), output_hidden_states=False)
        prompt_embeds = t5_output.last_hidden_state

        return {
            "clip_tokens": clip_tokens["input_ids"].cpu(),
            "clip_hidden": clip_hidden.detach().cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.detach().cpu(),
            "t5_tokens": t5_tokens["input_ids"].cpu(),
            "prompt_embeds": prompt_embeds.detach().cpu(),
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

            # Add batch dimension and move to device
            latent = latent.unsqueeze(0).to(device).float()

            with torch.no_grad(), autocast(device_type=device_type, dtype=torch.float32):
                # Undo scaling
                latent = latent / vae.config.scaling_factor
                decoded = vae.decode(latent).sample

            # Check shape
            _, c, h, w = decoded.shape
            if c != 3:
                return False

            # Check for NaN/Inf
            if torch.isnan(decoded).any() or torch.isinf(decoded).any():
                return False

            return True

        except Exception as e:
            logger.warning("[FLUX] Verification failed: %s", e)
            return False

    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct cache dictionary for FLUX.

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
            # CLIP embeddings
            "clip_tokens": text_encodings["clip_tokens"],
            "clip_hidden": text_encodings["clip_hidden"],
            "pooled_prompt_embeds": text_encodings["pooled_prompt_embeds"],
            # T5 embeddings
            "t5_tokens": text_encodings["t5_tokens"],
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
