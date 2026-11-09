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
FLUX.3 model processor for preprocessing.

Handles FLUX.3 text-to-image with:
- AutoencoderKLFlux2 image VAE (patchify 2x2 + BatchNorm normalization; FLUX.3
  reuses the FLUX.2 image VAE)
- Qwen3-VL text encoder (8 stacked hidden layers -> 20480-dim per token)
- No CLIP / no pooled projections

FLUX.3 checkpoints are modular pipelines (modular_model_index.json), so the
components are loaded directly from their subfolders instead of via a classic
DiffusionPipeline class.

Stores latents pre-patchified and BN-normalized ([128, H/16, W/16]) so the
FlowMatchingPipeline can add noise directly in the correct space.
"""

import logging

import torch

from nemo_automodel.components.models.flux3.types import Flux3CacheData, Flux3CacheMetadata, Flux3Models

from .base import BaseModelProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)

# Shared FLUX.3 text-encoding helpers are owned by the model package.
# The underscore aliases preserve this module's original public surface.
from nemo_automodel.components.models.flux3.text_encoding import (  # noqa: E402
    encode_text_qwen3vl as _encode_text_qwen3vl,
)
from nemo_automodel.components.models.flux3.text_encoding import (  # noqa: E402
    load_qwen3vl_text_stack as _load_qwen3vl_text_stack,
)


@ProcessorRegistry.register("flux3")
class Flux3Processor(BaseModelProcessor):
    """
    Processor for the FLUX.3 text-to-image architecture.

    Image latents are stored patchified (2x2 spatial -> channel) and
    BatchNorm-normalized using vae.bn running statistics — this is the space
    in which the FlowMatchingPipeline interpolates with noise.

    Text embeddings come from evenly-spaced Qwen3-VL hidden states (layers
    4/8/.../32 for the released checkpoint, derived from the checkpoint
    geometry), stacked per token, with the pipeline's chat template applied.

    FLUX.3 has no public default checkpoint id yet, so ``--model_name`` must be
    passed explicitly (a local modular snapshot directory or HF repo id).
    """

    @property
    def model_type(self) -> str:
        return "flux3"

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Convert latents to patchified format (2x2 spatial patch -> channel).

        Args:
            latents: Tensor of shape [batch, channels, height, width].

        Returns:
            Tensor of shape [batch, channels*4, height//2, width//2].
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)

    def load_models(self, model_name: str, device: str) -> Flux3Models:
        """
        Load FLUX.3 components from the modular checkpoint subfolders.

        Args:
            model_name: HuggingFace repo id or local path of the FLUX.3 modular
                checkpoint (contains transformer/, vae/, text_encoder/, tokenizer/)
            device: Device to load models on

        Returns:
            Dict containing:
                - vae: AutoencoderKLFlux2
                - bn_mean: BN running mean [1, C_latent, 1, 1] on device
                - bn_std: BN running std  [1, C_latent, 1, 1] on device
                - text_encoder: Qwen3VLForConditionalGeneration
                - tokenizer: AutoProcessor (chat template + tokenizer)
                - text_encoder_out_layers: tuple of hidden-state indices to stack
        """
        from diffusers import AutoencoderKLFlux2

        from nemo_automodel._diffusers._hf_cache import resolve_diffusion_model_dir

        logger.info("[FLUX.3] Loading models from %s (modular checkpoint)...", model_name)

        # Resolve to a local snapshot dir so a warm HF cache is not re-validated
        # (and potentially re-downloaded) over the network on every run.
        model_name = resolve_diffusion_model_dir(model_name)

        models = {}

        logger.info("  Configuring VAE...")
        vae = AutoencoderKLFlux2.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
        vae = vae.to(device)
        vae.eval()
        models["vae"] = vae

        # Extract BatchNorm normalization stats from vae.bn
        # These are fixed running statistics — not gradients — so we detach and float them.
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).float().to(device)
        bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).float().to(device)
        models["bn_mean"] = bn_mean
        models["bn_std"] = bn_std

        models.update(_load_qwen3vl_text_stack(model_name, device))

        torch.cuda.empty_cache()
        logger.info("[FLUX.3] Models loaded successfully!")
        return models

    def encode_image(
        self,
        image_tensor: torch.Tensor,
        models: Flux3Models,
        device: str,
    ) -> torch.Tensor:
        """
        Encode image to patchified + BN-normalized latent space.

        Pipeline (matches the FLUX.3 modular pipeline / training reference):
          1. VAE encode      -> [1, 32, H/8, W/8]
          2. Patchify 2x2    -> [1, 128, H/16, W/16]
          3. BN normalize    -> [1, 128, H/16, W/16]

        Args:
            image_tensor: Image tensor of shape [1, 3, H, W], normalized to [-1, 1]
            models: Dict containing 'vae', 'bn_mean', 'bn_std'
            device: Device to use

        Returns:
            Latent tensor of shape [128, H//16, W//16], FP32
        """
        vae = models["vae"]
        bn_mean = models["bn_mean"]
        bn_std = models["bn_std"]

        image_tensor = image_tensor.to(device, dtype=torch.float32)

        with torch.no_grad():
            # Step 1: VAE encode
            raw = vae.encode(image_tensor).latent_dist.mode()  # [1, 32, H/8, W/8]

            # Step 2: Patchify 2x2 spatial -> channel
            latent = self._patchify_latents(raw.float())  # [1, 128, H/16, W/16]

            # Step 3: BN normalize
            latent = (latent - bn_mean) / bn_std  # [1, 128, H/16, W/16]

        return latent.detach().cpu().squeeze(0)  # [128, H/16, W/16]

    def encode_text(
        self,
        prompt: str,
        models: Flux3Models,
        device: str,
    ) -> dict[str, torch.Tensor]:
        """
        Encode text with Qwen3-VL, stacking the checkpoint's configured hidden layers.

        Applies the chat template used by the FLUX.3 modular pipeline, tokenizes
        to a multiple of 80 tokens (up to 8192), and stacks the hidden layers
        derived in ``load_models`` (4/8/.../32 for the released checkpoint) into
        a joint_attention_dim-wide per-token embedding.

        Args:
            prompt: Text prompt
            models: Dict containing 'text_encoder', 'tokenizer', and
                'text_encoder_out_layers'
            device: Device to use

        Returns:
            Dict containing:
                - prompt_embeds: Stacked Qwen3-VL hidden states
                  [sequence, joint_attention_dim] (20480 for the released checkpoint), FP16
        """
        return {
            "prompt_embeds": _encode_text_qwen3vl(prompt, models, device),
        }

    def verify_latent(
        self,
        latent: torch.Tensor,
        models: Flux3Models,
        device: str,
    ) -> bool:
        """
        Verify patchified latent has the expected shape and no NaN/Inf.

        Full VAE decode is not performed because the stored latent is already
        patchified and BN-normalized — undoing both transforms to reconstruct
        a pixel image is expensive and not needed for a basic sanity check.

        Args:
            latent: Encoded latent of shape [C_latent*4, H/16, W/16]
            models: Dict containing 'vae' (for the expected channel count)
            device: Unused

        Returns:
            True if shape and numerical checks pass
        """
        expected_channels = models["vae"].config.latent_channels * 4
        if latent.ndim != 3:
            logger.warning("[FLUX.3] verify_latent: expected 3D tensor, got %dD", latent.ndim)
            return False
        if latent.shape[0] != expected_channels:
            logger.warning("[FLUX.3] verify_latent: expected %d channels, got %d", expected_channels, latent.shape[0])
            return False
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            logger.warning("[FLUX.3] verify_latent: NaN or Inf detected in latent")
            return False
        return True

    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: dict[str, torch.Tensor],
        metadata: Flux3CacheMetadata,
    ) -> Flux3CacheData:
        """
        Construct cache dictionary for FLUX.3.

        Uses the same keys as the Flux1/Flux2 processors ('latent',
        'prompt_embeds') so that collate_fn_text_to_image works without
        modification — it renames 'latent' -> 'image_latents' and
        'prompt_embeds' -> 'text_embeddings'.

        Args:
            latent: Patchified + BN-normalized latent of shape [128, H/16, W/16]
            text_encodings: Dict from encode_text() with 'prompt_embeds' of
                shape [sequence, 20480]
            metadata: Standard metadata dict

        Returns:
            Dict to save with torch.save()
        """
        return {
            # Image latent (patchified + BN-normalized)
            "latent": latent,
            # Qwen3-VL embeddings (stacked layers 4/8/.../32)
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
