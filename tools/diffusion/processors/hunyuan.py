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
HunyuanVideo-1.5 model processor for preprocessing.

Handles HunyuanVideo-1.5 video models with:
- HunyuanVideo VAE for video encoding
- Dual text encoders (CLIP-like + LLaMA)
- Image encoder for first frame (i2v conditioning)
- 4n+1 frame constraint
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base_video import BaseVideoProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("hunyuan")
@ProcessorRegistry.register("hunyuanvideo")
@ProcessorRegistry.register("hunyuanvideo-1.5")
class HunyuanVideoProcessor(BaseVideoProcessor):
    """
    Processor for HunyuanVideo-1.5 video models.

    HunyuanVideo uses:
    - HunyuanVideo VAE with shift_factor/scaling_factor normalization
    - Dual text encoders (CLIP-like + LLaMA) via pipeline.encode_prompt()
    - Image encoder for first frame embeddings (i2v conditioning)
    - 4n+1 frame constraint (1, 5, 9, 13, 17, ... 121)

    Default image embedding shape is (729, 1152).
    """

    # Default image embedding shape for HunyuanVideo
    DEFAULT_IMAGE_EMBED_SHAPE = (729, 1152)

    @property
    def model_type(self) -> str:
        return "hunyuanvideo"

    @property
    def default_model_name(self) -> str:
        return "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v"

    @property
    def supported_modes(self) -> List[str]:
        return ["video"]

    @property
    def frame_constraint(self) -> str:
        return "4n+1"

    @property
    def quantization(self) -> int:
        # HunyuanVideo VAE requires 8-pixel aligned dimensions
        return 8

    def load_models(self, model_name: str, device: str) -> Dict[str, Any]:
        """
        Load HunyuanVideo-1.5 models via pipeline.

        Args:
            model_name: HuggingFace model path
            device: Device to load models on

        Returns:
            Dict containing pipeline and individual components
        """
        from diffusers import HunyuanVideo15ImageToVideoPipeline

        dtype = torch.float16 if "cuda" in device else torch.float32

        logger.info("[HunyuanVideo] Loading pipeline from %s...", model_name)

        # Load pipeline without transformer to save memory
        # cpu_offload=True helps manage VRAM
        pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            transformer=None,  # Don't load transformer for preprocessing
        )

        logger.info("  Configuring VAE...")
        vae = pipeline.vae
        vae.to(device)
        vae.eval()

        # Enable memory optimizations
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling(
                tile_sample_min_height=64,
                tile_sample_min_width=64,
                tile_overlap_factor=0.25,
            )
            logger.info("  VAE tiling enabled")

        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
            logger.info("  VAE slicing enabled")

        logger.info("[HunyuanVideo] Models loaded successfully!")

        return {
            "pipeline": pipeline,
            "vae": vae,
            "text_encoder": pipeline.text_encoder,
            "tokenizer": pipeline.tokenizer,
            "image_encoder": pipeline.image_encoder,
            "dtype": dtype,
            "device": device,
        }

    def adjust_frame_count(self, frames: np.ndarray, target_frames: int) -> np.ndarray:
        """
        Adjust frame count to meet 4n+1 constraint.

        Args:
            frames: Array of frames (T, H, W, C)
            target_frames: Target number of frames (must be 4n+1)

        Returns:
            Adjusted frames array with target_frames frames
        """
        # Validate target_frames is 4n+1
        if (target_frames - 1) % 4 != 0:
            raise ValueError(f"target_frames must be 4n+1 (e.g., 1, 5, 9, 13, ..., 121), got {target_frames}")

        num_frames = len(frames)

        if num_frames == target_frames:
            return frames

        # Sample frames uniformly to reach target
        indices = np.linspace(0, num_frames - 1, target_frames).astype(int)
        return frames[indices]

    def validate_frame_count(self, num_frames: int) -> bool:
        """
        Check if frame count satisfies 4n+1 constraint.

        Args:
            num_frames: Number of frames

        Returns:
            True if valid, False otherwise
        """
        return (num_frames - 1) % 4 == 0

    def get_closest_valid_frame_count(self, num_frames: int) -> int:
        """
        Get the closest valid 4n+1 frame count.

        Args:
            num_frames: Current number of frames

        Returns:
            Closest 4n+1 value
        """
        n = (num_frames - 1) // 4
        lower = 4 * n + 1
        upper = 4 * (n + 1) + 1

        if num_frames - lower <= upper - num_frames:
            return max(1, lower)
        else:
            return upper

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
        Load video from file and preprocess with 4n+1 frame handling.

        Args:
            video_path: Path to video file
            target_size: Target (height, width)
            num_frames: Target number of frames (should be 4n+1)
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
            num_frames=None,  # Load all frames first
            resize_mode=resize_mode,
            center_crop=center_crop,
        )

        # Adjust to 4n+1 if target specified
        if num_frames is not None:
            frames = self.adjust_frame_count(frames, num_frames)
        else:
            # Auto-adjust to closest 4n+1
            target = self.get_closest_valid_frame_count(len(frames))
            if target != len(frames):
                frames = self.adjust_frame_count(frames, target)

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
        Encode video tensor to latent space using HunyuanVideo VAE.

        Uses shift_factor and scaling_factor normalization.

        Args:
            video_tensor: Video tensor (1, C, T, H, W), normalized to [-1, 1]
            models: Dict containing 'vae'
            device: Device to use
            deterministic: If True, use mean instead of sampling from latent distribution

        Returns:
            Latent tensor (1, C, T', H', W'), FP16
        """
        vae = models["vae"]
        dtype = models.get("dtype", torch.float16)

        video_tensor = video_tensor.to(device=device, dtype=dtype)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device != "cpu")):
                latent_dist = vae.encode(video_tensor)

                # Use mean for deterministic encoding, sample otherwise
                if deterministic:
                    latents = latent_dist.latent_dist.mean
                else:
                    latents = latent_dist.latent_dist.sample()

                # Apply HunyuanVideo-specific latent normalization
                if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                else:
                    latents = latents * vae.config.scaling_factor

        return latents.detach().cpu().to(torch.float16)

    def encode_text(
        self,
        prompt: str,
        models: Dict[str, Any],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text using dual encoders via pipeline.encode_prompt().

        Args:
            prompt: Text prompt
            models: Dict containing pipeline
            device: Device to use

        Returns:
            Dict containing:
                - text_embeddings: Primary text encoder output
                - text_mask: Primary attention mask
                - text_embeddings_2: Secondary text encoder output
                - text_mask_2: Secondary attention mask
        """
        pipeline = models["pipeline"]
        dtype = models.get("dtype", torch.float16)

        # Move text encoder to device
        pipeline.text_encoder.to(device)
        pipeline.text_encoder.eval()

        with torch.no_grad():
            (
                prompt_embeds,
                prompt_embeds_mask,
                prompt_embeds_2,
                prompt_embeds_mask_2,
            ) = pipeline.encode_prompt(
                prompt=prompt,
                device=device,
                dtype=dtype,
                batch_size=1,
                num_videos_per_prompt=1,
            )

        # Move back to CPU to free VRAM
        pipeline.text_encoder.to("cpu")

        return {
            "text_embeddings": prompt_embeds.detach().cpu(),
            "text_mask": prompt_embeds_mask.detach().cpu(),
            "text_embeddings_2": prompt_embeds_2.detach().cpu(),
            "text_mask_2": prompt_embeds_mask_2.detach().cpu(),
        }

    def encode_first_frame(
        self,
        first_frame: np.ndarray,
        models: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        """
        Encode first frame using image encoder for i2v conditioning.

        Args:
            first_frame: First frame as numpy array (H, W, C) in uint8
            models: Dict containing pipeline with image_encoder
            device: Device to use

        Returns:
            Image embeddings tensor (1, 729, 1152)
        """
        pipeline = models["pipeline"]
        dtype = models.get("dtype", torch.float16)

        # Move image encoder to device
        pipeline.image_encoder.to(device)

        # Convert numpy to PIL Image if needed
        if isinstance(first_frame, np.ndarray):
            first_frame_pil = Image.fromarray(first_frame)
        else:
            first_frame_pil = first_frame

        with torch.no_grad():
            image_embeds = pipeline.encode_image(
                image=first_frame_pil,
                batch_size=1,
                device=device,
                dtype=dtype,
            )

        # Move back to CPU
        pipeline.image_encoder.to("cpu")

        return image_embeds.detach().cpu()

    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct cache dictionary for HunyuanVideo.

        Args:
            latent: Encoded latent tensor (1, C, T, H, W)
            text_encodings: Dict from encode_text()
            metadata: Additional metadata including image_embeds

        Returns:
            Dict to save with torch.save() or pickle
        """
        return {
            # Video latent
            "video_latents": latent,
            # Dual text embeddings
            "text_embeddings": text_encodings["text_embeddings"],
            "text_mask": text_encodings["text_mask"],
            "text_embeddings_2": text_encodings["text_embeddings_2"],
            "text_mask_2": text_encodings["text_mask_2"],
            # Image embeddings for i2v
            "image_embeds": metadata.get("image_embeds"),
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
            "model_version": "hunyuanvideo-1.5",
            "processing_mode": metadata.get("mode", "video"),
            "model_type": self.model_type,
        }
