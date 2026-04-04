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
LTX-Video model processor for preprocessing.

Handles LTX-Video models with:
- AutoencoderKLLTXVideo for video encoding
- T5 text encoder for text conditioning
- Latent normalization using latents_mean and latents_std
- 8x8x8 spatio-temporal compression (128 latent channels)
- Frame count constraint: 8n+1
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base_video import BaseVideoProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("ltx")
@ProcessorRegistry.register("ltx-video")
class LTXProcessor(BaseVideoProcessor):
    """
    Processor for LTX-Video T2V models.

    LTX-Video uses:
    - AutoencoderKLLTXVideo for video encoding with latents_mean/latents_std normalization
    - T5 text encoder (max 128 tokens)
    - 128 latent channels with 8x8x8 compression
    - Frame count constraint: 8n+1 (1, 9, 17, 25, ...)
    """

    MAX_SEQUENCE_LENGTH = 128

    @property
    def model_type(self) -> str:
        return "ltx"

    @property
    def default_model_name(self) -> str:
        return "Lightricks/LTX-Video"

    @property
    def supported_modes(self) -> List[str]:
        return ["video", "frames"]

    @property
    def quantization(self) -> int:
        # LTX VAE has 8x spatial compression, and resolution must be divisible by 32
        # So pixel dimensions must be divisible by 32
        return 32

    @property
    def frame_constraint(self) -> Optional[str]:
        return "8n+1"

    def get_closest_valid_frame_count(self, frame_count: int) -> int:
        """Get closest valid frame count satisfying 8n+1 constraint."""
        n = round((frame_count - 1) / 8)
        n = max(n, 1)  # At least 9 frames (n=1)
        return 8 * n + 1

    def adjust_frame_count(self, frames: np.ndarray, target_frames: int) -> np.ndarray:
        """Adjust frame count to satisfy 8n+1 constraint."""
        valid_target = self.get_closest_valid_frame_count(target_frames)
        current_frames = len(frames)

        if current_frames == valid_target:
            return frames

        indices = np.linspace(0, current_frames - 1, valid_target).astype(int)
        return frames[indices]

    def load_models(self, model_name: str, device: str) -> Dict[str, Any]:
        """
        Load LTX-Video models.

        Args:
            model_name: HuggingFace model path (e.g., 'Lightricks/LTX-Video')
            device: Device to load models on

        Returns:
            Dict containing vae, text_encoder, tokenizer
        """
        from diffusers import AutoencoderKLLTXVideo
        from transformers import T5EncoderModel, T5TokenizerFast

        dtype = torch.float16 if "cuda" in device else torch.float32
        # T5 works well in bfloat16
        text_encoder_dtype = torch.bfloat16 if "cuda" in device else torch.float32

        logger.info("[LTX] Loading models from %s...", model_name)

        # Load text encoder
        logger.info("  Loading T5 text encoder...")
        text_encoder = T5EncoderModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            torch_dtype=text_encoder_dtype,
        )
        text_encoder.to(device)
        text_encoder.eval()

        # Load VAE
        logger.info("  Loading AutoencoderKLLTXVideo...")
        vae = AutoencoderKLLTXVideo.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=dtype,
        )
        vae.to(device)
        vae.eval()

        # Enable memory optimizations
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

        # Load tokenizer
        logger.info("  Loading T5 tokenizer...")
        tokenizer = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer")

        logger.info("[LTX] Models loaded successfully!")
        if hasattr(vae, "latents_mean"):
            logger.debug("  VAE latents_mean available")
        if hasattr(vae, "latents_std"):
            logger.debug("  VAE latents_std available")

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
            resize_mode: Interpolation mode
            center_crop: Whether to center crop

        Returns:
            Tuple of (video_tensor [1, C, T, H, W], first_frame [H, W, C] uint8)
        """
        frames, info = self.load_video_frames(
            video_path,
            target_size,
            num_frames=num_frames,
            resize_mode=resize_mode,
            center_crop=center_crop,
        )

        # Adjust frame count to satisfy 8n+1 constraint
        valid_count = self.get_closest_valid_frame_count(len(frames))
        if len(frames) != valid_count:
            logger.info("  Adjusting frame count from %d to %d (8n+1 constraint)", len(frames), valid_count)
            frames = self.adjust_frame_count(frames, valid_count)

        first_frame = frames[0].copy()
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
        Encode video tensor to latent space using LTX VAE.

        Uses latents_mean and latents_std normalization with scaling_factor.

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

        # Apply LTX latent normalization: (latents - mean) * scaling_factor / std
        if not hasattr(vae, "latents_mean") or not hasattr(vae, "latents_std"):
            raise ValueError("LTX VAE requires latents_mean and latents_std")

        latents_mean = vae.latents_mean.to(device=device, dtype=dtype)
        latents_std = vae.latents_std.to(device=device, dtype=dtype)
        scaling_factor = vae.config.scaling_factor

        latents_mean = latents_mean.view(1, -1, 1, 1, 1)
        latents_std = latents_std.view(1, -1, 1, 1, 1)

        latents = (video_latents - latents_mean) * scaling_factor / latents_std

        return latents.detach().cpu().to(torch.float16)

    def encode_text(
        self,
        prompt: str,
        models: Dict[str, Any],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text using T5 encoder.

        Args:
            prompt: Text prompt
            models: Dict containing tokenizer and text_encoder
            device: Device to use

        Returns:
            Dict containing text_embeddings and text_mask
        """
        tokenizer = models["tokenizer"]
        text_encoder = models["text_encoder"]

        inputs = tokenizer(
            prompt,
            max_length=self.MAX_SEQUENCE_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            prompt_embeds = text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).last_hidden_state

        return {
            "text_embeddings": prompt_embeds.detach().cpu(),
            "text_mask": inputs["attention_mask"].detach().cpu(),
        }

    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct cache dictionary for LTX.

        Args:
            latent: Encoded latent tensor (1, C, T, H, W)
            text_encodings: Dict from encode_text()
            metadata: Additional metadata including first_frame

        Returns:
            Dict to save with torch.save()
        """
        return {
            # Video latent
            "video_latents": latent,
            # Text embeddings (required key)
            "text_embeddings": text_encodings["text_embeddings"],
            # Text attention mask (optional video field)
            "text_mask": text_encodings.get("text_mask"),
            # First frame for potential i2v conditioning
            "first_frame": torch.from_numpy(metadata["first_frame"])
            if metadata.get("first_frame") is not None
            else None,
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
            "model_version": "ltx-video",
            "processing_mode": metadata.get("mode", "video"),
            "model_type": self.model_type,
        }
