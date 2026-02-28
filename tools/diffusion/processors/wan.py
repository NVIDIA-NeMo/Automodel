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
Wan2.1 video model processor for preprocessing.

Handles Wan2.1-T2V models (1.3B and 14B variants) with:
- AutoencoderKLWan for video encoding
- UMT5 text encoder for text conditioning
- Latent normalization using latents_mean and latents_std
"""

import html
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base_video import BaseVideoProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)


def _basic_clean(text: str) -> str:
    """Fix text encoding issues and unescape HTML entities."""
    try:
        from diffusers.utils import is_ftfy_available

        if is_ftfy_available():
            import ftfy

            text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    """Normalize whitespace by replacing multiple spaces with single space."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _prompt_clean(text: str) -> str:
    """Clean prompt text exactly as done in WanPipeline."""
    return _whitespace_clean(_basic_clean(text))


@ProcessorRegistry.register("wan")
@ProcessorRegistry.register("wan2.1")
class WanProcessor(BaseVideoProcessor):
    """
    Processor for Wan2.1 T2V video models.

    Wan2.1 uses:
    - AutoencoderKLWan for video/image encoding with latents_mean/latents_std normalization
    - UMT5 text encoder with specific padding behavior (trim and re-pad to 226 tokens)
    """

    # Maximum sequence length for UMT5 text encoder
    MAX_SEQUENCE_LENGTH = 226

    @property
    def model_type(self) -> str:
        return "wan"

    @property
    def default_model_name(self) -> str:
        return "Wan-AI/Wan2.1-T2V-14B-Diffusers"

    @property
    def supported_modes(self) -> List[str]:
        return ["video", "frames"]

    @property
    def quantization(self) -> int:
        # Wan VAE downsamples by 8x and transformer has patch_size=2 in latent space
        # Therefore, pixel dimensions must be divisible by 8 * 2 = 16
        return 16

    def load_models(self, model_name: str, device: str) -> Dict[str, Any]:
        """
        Load Wan2.1 models.

        Args:
            model_name: HuggingFace model path (e.g., 'Wan-AI/Wan2.1-T2V-14B-Diffusers')
            device: Device to load models on

        Returns:
            Dict containing:
                - vae: AutoencoderKLWan
                - text_encoder: UMT5EncoderModel
                - tokenizer: AutoTokenizer
        """
        from diffusers import AutoencoderKLWan
        from transformers import AutoTokenizer, UMT5EncoderModel

        dtype = torch.float16 if "cuda" in device else torch.float32

        logger.info("[Wan] Loading models from %s...", model_name)

        # Load text encoder
        logger.info("  Loading UMT5 text encoder...")
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
        text_encoder.to(device)
        text_encoder.eval()

        # Load VAE
        logger.info("  Loading AutoencoderKLWan...")
        vae = AutoencoderKLWan.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=dtype,
        )
        vae.to(device)
        vae.eval()

        # Enable memory optimizations
        vae.enable_slicing()
        vae.enable_tiling()

        # Load tokenizer
        logger.info("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")

        logger.info("[Wan] Models loaded successfully!")
        logger.debug("  VAE latents_mean: %s", vae.config.latents_mean)
        logger.debug("  VAE latents_std: %s", vae.config.latents_std)

        return {
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "dtype": dtype,
        }

    def load_video(
        self,
        video_path: str,
        target_size: Tuple[int, int],
        num_frames: Optional[int] = None,
        resize_mode: str = "bilinear",
        center_crop: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load video from file and preprocess.

        Args:
            video_path: Path to video file
            target_size: Target (height, width)
            num_frames: Number of frames to extract (None = all frames)
            resize_mode: Interpolation mode for resizing
            center_crop: Whether to center crop

        Returns:
            Tuple of:
                - video_tensor: Tensor of shape (1, C, T, H, W), normalized to [-1, 1]
                - first_frame: First frame as numpy array (H, W, C) in uint8
        """
        # Use base class utility to load frames
        frames, info = self.load_video_frames(
            video_path,
            target_size,
            num_frames=num_frames,
            resize_mode=resize_mode,
            center_crop=center_crop,
        )

        # Save first frame before converting to tensor
        first_frame = frames[0].copy()

        # Convert to tensor
        video_tensor = self.frames_to_tensor(frames)

        return video_tensor, first_frame

    def encode_video(
        self,
        video_tensor: torch.Tensor,
        models: Dict[str, Any],
        device: str,
        deterministic: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode video tensor to latent space using Wan VAE.

        Uses latents_mean and latents_std normalization as per Wan2.1 specification.

        Args:
            video_tensor: Video tensor (1, C, T, H, W), normalized to [-1, 1]
            models: Dict containing 'vae'
            device: Device to use
            deterministic: If True, use mean instead of sampling

        Returns:
            Latent tensor (1, C, T', H', W'), FP16
        """
        vae = models["vae"]
        dtype = models.get("dtype", torch.float16)

        video_tensor = video_tensor.to(device=device, dtype=dtype)

        with torch.no_grad():
            latent_dist = vae.encode(video_tensor)

            if deterministic:
                video_latents = latent_dist.latent_dist.mean
            else:
                video_latents = latent_dist.latent_dist.sample()

        # Apply Wan-specific latent normalization
        if not hasattr(vae.config, "latents_mean") or not hasattr(vae.config, "latents_std"):
            raise ValueError("Wan2.1 VAE requires latents_mean and latents_std in config")

        latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)

        latents = (video_latents - latents_mean) / latents_std

        return latents.detach().cpu().to(torch.float16)

    def encode_text(
        self,
        prompt: str,
        models: Dict[str, Any],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text using UMT5.

        Implements the specific padding behavior for Wan:
        1. Tokenize with padding to max_length
        2. Encode with attention mask
        3. Trim embeddings to actual sequence length
        4. Re-pad with zeros to max_sequence_length (226)

        Args:
            prompt: Text prompt
            models: Dict containing tokenizer and text_encoder
            device: Device to use

        Returns:
            Dict containing:
                - text_embeddings: UMT5 embeddings (1, 226, hidden_dim)
        """
        tokenizer = models["tokenizer"]
        text_encoder = models["text_encoder"]

        # Clean prompt
        prompt = _prompt_clean(prompt)

        # Tokenize
        inputs = tokenizer(
            prompt,
            max_length=self.MAX_SEQUENCE_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Calculate actual sequence length (excluding padding)
        seq_lens = inputs["attention_mask"].gt(0).sum(dim=1).long()

        with torch.no_grad():
            prompt_embeds = text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).last_hidden_state

        # CRITICAL: Trim to actual length and re-pad with zeros
        # This matches the exact behavior in WanPipeline
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(self.MAX_SEQUENCE_LENGTH - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )

        return {
            "text_embeddings": prompt_embeds.detach().cpu(),
        }

    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct cache dictionary for Wan.

        Args:
            latent: Encoded latent tensor (1, C, T, H, W)
            text_encodings: Dict from encode_text()
            metadata: Additional metadata including first_frame

        Returns:
            Dict to save with torch.save() or pickle
        """
        return {
            # Video latent
            "video_latents": latent,
            # Text embeddings
            "text_embeddings": text_encodings["text_embeddings"],
            # First frame for image-to-video conditioning
            "first_frame": metadata.get("first_frame"),
            # Resolution and bucketing info
            "original_resolution": metadata.get("original_resolution"),
            "bucket_resolution": metadata.get("bucket_resolution"),
            "bucket_id": metadata.get("bucket_id"),
            "aspect_ratio": metadata.get("aspect_ratio"),
            # Video info
            "num_frames": metadata.get("num_frames"),
            "prompt": metadata.get("prompt"),
            "video_path": metadata.get("video_path"),
            # Processing settings
            "deterministic_latents": metadata.get("deterministic", True),
            "model_version": "wan2.1",
            "processing_mode": metadata.get("mode", "video"),
            "model_type": self.model_type,
        }
