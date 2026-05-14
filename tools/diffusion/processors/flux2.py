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
FLUX.2 model processor for preprocessing.

Handles FLUX.2-dev with:
- AutoencoderKLFlux2 VAE (patchify 2x2 + BatchNorm normalization)
- Mistral3ForConditionalGeneration text encoder (intermediate layers 10, 20, 30)
- No CLIP / no pooled projections

Stores latents pre-patchified and BN-normalized ([128, H/16, W/16]) so the
FlowMatchingPipeline can add noise directly in the correct space.
"""

import logging
from typing import Any, Dict

import torch

from .base import BaseModelProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("flux2")
class Flux2Processor(BaseModelProcessor):
    """
    Processor for FLUX.2-dev architecture.

    Image latents are stored patchified (2×2 spatial → channel) and
    BatchNorm-normalized using vae.bn running statistics — this is the space
    in which the FlowMatchingPipeline interpolates with noise.

    Text embeddings come from Mistral3 hidden states at layers 10/20/30,
    stacked to 15360-dim per token.
    """

    @property
    def model_type(self) -> str:
        return "flux2"

    @property
    def default_model_name(self) -> str:
        return "black-forest-labs/FLUX.2-dev"

    def load_models(self, model_name: str, device: str) -> Dict[str, Any]:
        """
        Load FLUX.2 models from Flux2Pipeline.

        Args:
            model_name: HuggingFace model path (e.g., 'black-forest-labs/FLUX.2-dev')
            device: Device to load models on

        Returns:
            Dict containing:
                - pipeline: Flux2Pipeline (text encoder + tokenizer, no transformer)
                - vae: AutoencoderKLFlux2
                - bn_mean: BN running mean [1, 128, 1, 1] on device
                - bn_std: BN running std  [1, 128, 1, 1] on device
        """
        from diffusers import Flux2Pipeline

        logger.info("[FLUX.2] Loading models from %s via Flux2Pipeline...", model_name)

        # Load without transformer (not needed for preprocessing)
        pipeline = Flux2Pipeline.from_pretrained(
            model_name,
            transformer=None,
            torch_dtype=torch.bfloat16,
        )

        models = {}

        logger.info("  Configuring VAE...")
        vae = pipeline.vae.to(device=device, dtype=torch.bfloat16)
        vae.eval()
        models["vae"] = vae
        logger.debug("VAE config: %s", vae.config)

        # Extract BatchNorm normalization stats from vae.bn
        # These are fixed running statistics — not gradients — so we detach and float them.
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).float().to(device)
        bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).float().to(device)
        models["bn_mean"] = bn_mean
        models["bn_std"] = bn_std
        logger.debug("BN mean shape: %s, std shape: %s", bn_mean.shape, bn_std.shape)

        logger.info("  Configuring Mistral3 text encoder...")
        pipeline.text_encoder = pipeline.text_encoder.to(device)
        models["pipeline"] = pipeline  # encode_prompt lives here

        torch.cuda.empty_cache()
        logger.info("[FLUX.2] Models loaded successfully!")
        return models

    def encode_image(
        self,
        image_tensor: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        """
        Encode image to patchified + BN-normalized latent space.

        Pipeline (order matches DreamBooth Flux2 script):
          1. VAE encode → [1, 32, H/8, W/8]
          2. Patchify 2×2  → [1, 128, H/16, W/16]
          3. BN normalize   → [1, 128, H/16, W/16]

        Args:
            image_tensor: Image tensor (1, 3, H, W), normalized to [-1, 1]
            models: Dict containing 'vae', 'bn_mean', 'bn_std'
            device: Device to use

        Returns:
            Latent tensor (128, H//16, W//16), FP16
        """
        from diffusers import Flux2Pipeline

        vae = models["vae"]
        bn_mean = models["bn_mean"]
        bn_std = models["bn_std"]

        image_tensor = image_tensor.to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            # Step 1: VAE encode
            raw = vae.encode(image_tensor).latent_dist.mode()  # [1, 32, H/8, W/8]

            # Step 2: Patchify 2×2 spatial → channel
            latent = Flux2Pipeline._patchify_latents(raw.float())  # [1, 128, H/16, W/16]

            # Step 3: BN normalize
            latent = (latent - bn_mean) / bn_std  # [1, 128, H/16, W/16]

        return latent.detach().cpu().to(torch.float16).squeeze(0)  # [128, H/16, W/16]

    def encode_text(
        self,
        prompt: str,
        models: Dict[str, Any],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text with Mistral3, extracting intermediate layers 10/20/30.

        Args:
            prompt: Text prompt
            models: Dict containing 'pipeline'
            device: Device to use

        Returns:
            Dict containing:
                - prompt_embeds: Stacked Mistral3 hidden states [seq, 15360], FP16
        """
        pipeline = models["pipeline"]

        with torch.no_grad():
            prompt_embeds, _text_ids = pipeline.encode_prompt(
                prompt=prompt,
                max_sequence_length=512,
                text_encoder_out_layers=(10, 20, 30),
            )
        # prompt_embeds: [1, seq_len, 15360]
        # _text_ids: [1, seq_len, 4] — discarded; computed dynamically in Flux2Adapter

        return {
            "prompt_embeds": prompt_embeds.squeeze(0).detach().cpu().to(torch.float16),
        }

    def verify_latent(
        self,
        latent: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> bool:
        """
        Verify patchified latent has the expected shape and no NaN/Inf.

        Full VAE decode is not performed because the stored latent is already
        patchified and BN-normalized — undoing both transforms to reconstruct
        a pixel image is expensive and not needed for a basic sanity check.

        Args:
            latent: Encoded latent (128, H, W)
            models: Unused (no decode needed)
            device: Unused

        Returns:
            True if shape and numerical checks pass
        """
        if latent.ndim != 3:
            logger.warning("[FLUX.2] verify_latent: expected 3D tensor, got %dD", latent.ndim)
            return False
        if latent.shape[0] != 128:
            logger.warning("[FLUX.2] verify_latent: expected 128 channels, got %d", latent.shape[0])
            return False
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            logger.warning("[FLUX.2] verify_latent: NaN or Inf detected in latent")
            return False
        return True

    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct cache dictionary for FLUX.2.

        Uses the same keys as the Flux1 processor ('latent', 'prompt_embeds') so
        that collate_fn_text_to_image works without modification — it renames
        'latent' → 'image_latents' and 'prompt_embeds' → 'text_embeddings'.

        Args:
            latent: Patchified + BN-normalized latent (128, H/16, W/16)
            text_encodings: Dict from encode_text()
            metadata: Standard metadata dict

        Returns:
            Dict to save with torch.save()
        """
        return {
            # Image latent (patchified + BN-normalized)
            "latent": latent,
            # Mistral3 embeddings (stacked layers 10/20/30)
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
