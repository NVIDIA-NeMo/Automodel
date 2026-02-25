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
Preprocessing Script for HunyuanVideo-1.5 Training Data

This script preprocesses videos and text captions for HunyuanVideo-1.5 training by:
1. Loading videos from a folder with meta.json metadata
2. Processing videos to ensure 4n+1 frames (required by VAE)
3. Encoding videos with VAE to get latents
4. Encoding text captions with text encoders (CLIP-like + LLaMA) and byT5
5. Saving preprocessed data to .meta files for faster training

Usage:
    python preprocess_dataset.py \
        --data_dir /path/to/videos \
        --meta_file meta.json \
        --output_dir /path/to/output \
        --pretrained_model_root /path/to/models \
        --target_frames 121 \
        --target_height 720 \
        --target_width 1280
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import torch
from diffusers import HunyuanVideo15ImageToVideoPipeline
from PIL import Image
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resize_and_center_crop(image, target_width, target_height):
    if target_height == image.shape[0] and target_width == image.shape[1]:
        return image

    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def str_to_bool(value):
    """Convert string to boolean."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def format_text_input(prompt: List[str], system_message: str) -> List[Dict[str, Any]]:
    """
    Apply text to template.

    Args:
        prompt (List[str]): Input text.
        system_message (str): System message.

    Returns:
        List[Dict[str, Any]]: List of chat conversation.
    """

    template = [
        [{"role": "system", "content": system_message}, {"role": "user", "content": p if p else " "}] for p in prompt
    ]

    return template


def load_video(video_path: str, start_frame: int = 0, end_frame: Optional[int] = None) -> np.ndarray:
    """
    Load video from file.

    Args:
        video_path: Path to video file
        start_frame: Starting frame index
        end_frame: Ending frame index (None means to the end)

    Returns:
        Video frames as numpy array [F, H, W, C] in uint8 [0, 255]
    """
    reader = imageio.get_reader(video_path, "ffmpeg")
    frames = []

    try:
        for i, frame in enumerate(reader):
            if i < start_frame:
                continue
            if end_frame is not None and i >= end_frame:
                break
            frames.append(frame)
    finally:
        reader.close()

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_path}")

    return np.stack(frames, axis=0)  # [F, H, W, C]


def adjust_frames_to_4n_plus_1(frames: np.ndarray, target_frames: Optional[int] = None) -> np.ndarray:
    """
    Adjust number of frames to 4n+1 format required by VAE.

    Args:
        frames: Input frames [F, H, W, C]
        target_frames: Target number of frames (must be 4n+1). If None, adjust to closest 4n+1

    Returns:
        Adjusted frames [F', H, W, C] where F' = 4n+1
    """
    num_frames = frames.shape[0]

    if target_frames is not None:
        # Validate target_frames is 4n+1
        if (target_frames - 1) % 4 != 0:
            raise ValueError(f"target_frames must be 4n+1, got {target_frames}")

        if num_frames < target_frames:
            # Repeat frames if not enough
            logger.warning(
                f"Video has {num_frames} frames, but target is {target_frames}. Some frames will be repeated."
            )
            indices = np.linspace(0, num_frames - 1, target_frames).astype(int)
            frames = frames[indices]
        elif num_frames > target_frames:
            # Sample frames uniformly
            logger.debug(f"Sampling {target_frames} frames from {num_frames} total frames")
            indices = np.linspace(0, num_frames - 1, target_frames).astype(int)
            frames = frames[indices]

        return frames
    else:
        # Find closest 4n+1
        n = (num_frames - 1) // 4
        target = 4 * n + 1

        if target < 1:
            target = 1

        if num_frames != target:
            logger.debug(f"Adjusting {num_frames} frames to {target} frames (4n+1 format)")
            if num_frames < target:
                indices = np.linspace(0, num_frames - 1, target).astype(int)
            else:
                indices = np.linspace(0, num_frames - 1, target).astype(int)
            frames = frames[indices]

        return frames


def preprocess_video(frames: np.ndarray, target_height: int, target_width: int) -> torch.Tensor:
    """
    Preprocess video frames to target resolution and convert to tensor.

    Args:
        frames: Input frames [F, H, W, C] in uint8 [0, 255]
        target_height: Target height
        target_width: Target width

    Returns:
        Preprocessed video tensor [C, F, H, W] in float32 [-1, 1]
    """
    num_frames = frames.shape[0]
    processed_frames = []

    for i in range(num_frames):
        frame = frames[i]  # [H, W, C]
        # Resize and center crop
        frame = resize_and_center_crop(frame, target_width, target_height)
        processed_frames.append(frame)

    processed_frames = np.stack(processed_frames, axis=0)  # [F, H, W, C]

    # Convert to tensor and normalize to [-1, 1]
    video_tensor = torch.from_numpy(processed_frames).float() / 255.0  # [F, H, W, C] in [0, 1]
    video_tensor = video_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # [C, F, H, W]

    return video_tensor


class VideoPreprocessor:
    def __init__(
        self,
        pretrained_model_root: str,
        transformer_version: str = "720p_t2v",
        device: str = "cuda",
        dtype: str = "fp16",
    ):
        """
        Initialize video preprocessor with models.

        Args:
            pretrained_model_root: Path to pretrained models
            transformer_version: Transformer version (not loaded, only for pipeline setup)
            device: Device to use ('cuda' or 'cpu')
            dtype: Data type for encoding ('fp16' or 'bf16' or 'fp32')
        """
        self.device = torch.device(device)

        # Set dtype for VAE encoding
        if dtype == "fp16":
            self.vae_dtype = torch.float16
        elif dtype == "bf16":
            self.vae_dtype = torch.bfloat16
        else:
            self.vae_dtype = torch.float32

        logger.info(f"Loading models from {pretrained_model_root}")
        logger.info(f"Using device: {device}, dtype: {dtype}")

        # Load pipeline (we only need VAE and text encoders, not transformer)
        self.pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v", torch_dtype=torch.float16, cpu_offload=True
        )

        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer

        self.tokenizer_max_length = 1000
        self.crop_start = 108
        self.num_hidden_layers_to_skip = 2
        # Set models to eval mode and move to device
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling(tile_sample_min_height=64, tile_sample_min_width=64, tile_overlap_factor=0.25)
            logger.info("VAE tiling enabled")

        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()
            logger.info("VAE slicing enabled")

        logger.info("Models loaded successfully")

    @torch.no_grad()
    def encode_vae(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video with VAE.

        Args:
            video: Video tensor [C, F, H, W] in float32 [-1, 1]

        Returns:
            Latents tensor [C_latent, F, H_latent, W_latent]
        """
        if video.max() > 1.0 or video.min() < -1.0:
            raise ValueError(f"Video must be in range [-1, 1], got [{video.min()}, {video.max()}]")

        # Add batch dimension
        video = video.unsqueeze(0)  # [1, C, F, H, W]
        video = video.to(device=self.device)

        with torch.autocast(device_type="cuda", dtype=self.vae_dtype, enabled=(self.device.type == "cuda")):
            latents = self.vae.encode(video).latent_dist.sample()
            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            else:
                latents = latents * self.vae.config.scaling_factor

        # Remove batch dimension
        latents = latents.squeeze(0)  # [C_latent, F, H_latent, W_latent]
        latents = latents.detach().cpu()

        return latents

    def preprocess_single_video(
        self,
        video_path: str,
        caption: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        target_frames: Optional[int] = None,
        target_height: int = 720,
        target_width: int = 1280,
        data_type: str = "video",
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a single video and caption.

        Args:
            video_path: Path to video file
            caption: Text caption
            start_frame: Starting frame index
            end_frame: Ending frame index
            target_frames: Target number of frames (must be 4n+1)
            target_height: Target height
            target_width: Target width
            data_type: "video" or "image"

        Returns:
            Dictionary containing preprocessed data
        """
        # Load video
        logger.debug(f"Loading video: {video_path}")
        frames = load_video(video_path, start_frame, end_frame)
        logger.debug(f"Loaded {frames.shape[0]} frames from {video_path}")

        # Adjust frames to 4n+1
        frames = adjust_frames_to_4n_plus_1(frames, target_frames)
        logger.debug(f"Adjusted to {frames.shape[0]} frames (4n+1 format)")

        # Preprocess video

        video_tensor = preprocess_video(frames, target_height, target_width)
        logger.debug(f"Preprocessed video shape: {video_tensor.shape}")

        # Encode with VAE
        self.vae.to(self.device)
        self.vae.eval()
        logger.debug("Encoding with VAE...")
        latents = self.encode_vae(video_tensor)
        logger.debug(f"Latents shape: {latents.shape}")
        self.vae.to("cpu")

        # Encode text
        self.text_encoder.to(self.device)
        self.text_encoder.eval()
        logger.debug("Encoding text...")
        # text_encodings = self.encode_text(caption)
        prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = self.pipeline.encode_prompt(
            prompt=caption, device=self.device, dtype=torch.float16, batch_size=1, num_videos_per_prompt=1
        )
        self.text_encoder.to("cpu")

        logger.debug("Encoding first frame for image embedding...")
        self.pipeline.image_encoder.to(self.device)
        first_frame = frames[0]
        image_embeds = self.pipeline.encode_image(
            image=first_frame, batch_size=1, device=self.device, dtype=torch.float16
        )
        logger.info(f"!!!Image embeddings shape: {image_embeds.shape}")
        self.pipeline.image_encoder.to("cpu")

        result = {
            "video_latents": latents.unsqueeze(0),  # Add batch dim: [1, C, F, H, W] - already detached above
            "text_embeddings": prompt_embeds.detach().cpu(),  # Already [1, seq_len, dim]
            "text_mask": prompt_embeds_mask.detach().cpu(),  # [1, seq_len]
            "text_embeddings_2": prompt_embeds_2.detach().cpu(),  # [1, seq_len, dim]
            "text_mask_2": prompt_embeds_mask_2.detach().cpu(),  # [1, seq_len]
            "image_embeds": image_embeds.detach().cpu(),  # [1, 729, 1152]
            "metadata": {
                "text": caption,
                "data_type": data_type,
                "video_shape": list(video_tensor.shape),  # [C, F, H, W]
                "latent_shape": list(latents.shape),  # [C_latent, F, H_latent, W_latent]
            },
            "original_filename": Path(video_path).name,
            "original_video_path": str(video_path),
            "num_frames": video_tensor.shape[1],  # F
            "deterministic_latents": "vae_encoded",
            "memory_optimization": f"dtype_{self.vae_dtype}",
        }
        logger.debug(
            f"Result shapes - video_latents: {result['video_latents'].shape}, "
            f"text_embeddings: {result['text_embeddings'].shape}"
        )

        return result


def main():
    parser = argparse.ArgumentParser(description="Preprocess videos and captions for HunyuanVideo-1.5 training")

    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--meta_file", type=str, default="meta.json", help="Metadata JSON file name")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .meta files")

    # Model parameters
    parser.add_argument("--pretrained_model_root", type=str, required=False, help="Path to pretrained models")
    parser.add_argument(
        "--transformer_version", type=str, default="720p_t2v", help="Transformer version (default: 720p_t2v)"
    )

    # Processing parameters
    parser.add_argument(
        "--target_frames",
        type=int,
        default=9,
        help="Target number of frames (must be 4n+1, e.g., 1, 5, 9, 13, 17, 21, ..., 121)",
    )
    parser.add_argument("--target_height", type=int, default=720, help="Target video height")
    parser.add_argument("--target_width", type=int, default=1280, help="Target video width")
    parser.add_argument(
        "--data_type",
        type=str,
        default="video",
        choices=["video", "image"],
        help="Data type for text encoding (default: video)",
    )

    # Caption field
    parser.add_argument(
        "--caption_field",
        type=str,
        default="vila_caption",
        help="Field name in meta.json containing captions (default: vila_caption)",
    )

    # Device parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type for encoding (default: fp16)",
    )

    # Other parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently only 1 is supported)")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of worker processes (currently only 0 is supported)"
    )

    args = parser.parse_args()

    # Validate target_frames is 4n+1
    if (args.target_frames - 1) % 4 != 0:
        raise ValueError(f"target_frames must be 4n+1 (e.g., 1, 5, 9, 13, 17, 21, ..., 121), got {args.target_frames}")

    # Setup paths
    data_dir = Path(args.data_dir)
    meta_path = data_dir / args.meta_file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    logger.info(f"Loading metadata from {meta_path}")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    logger.info(f"Found {len(metadata)} videos in metadata")

    # Initialize preprocessor
    logger.info("Initializing preprocessor...")
    preprocessor = VideoPreprocessor(
        pretrained_model_root=args.pretrained_model_root,
        transformer_version=args.transformer_version,
        device=args.device,
        dtype=args.dtype,
    )

    # Process each video
    logger.info("Starting preprocessing...")
    successful = 0
    failed = 0

    for item in tqdm(metadata, desc="Processing videos"):
        try:
            file_name = item["file_name"]
            video_path = data_dir / file_name

            # Check if video exists
            if not video_path.exists():
                logger.warning(f"Video not found: {video_path}")
                failed += 1
                continue

            # Get caption
            caption = item.get(args.caption_field, "")
            if not caption:
                logger.warning(f"No caption found for {file_name}")
                failed += 1
                continue

            # Get frame range if specified
            start_frame = item.get("start_frame", 0)
            end_frame = item.get("end_frame", None)
            if end_frame is not None:
                end_frame = end_frame + 1  # end_frame is inclusive in meta.json, but exclusive in our code

            # Preprocess
            result = preprocessor.preprocess_single_video(
                video_path=str(video_path),
                caption=caption,
                start_frame=start_frame,
                end_frame=end_frame,
                target_frames=args.target_frames,
                target_height=args.target_height,
                target_width=args.target_width,
                data_type=args.data_type,
            )

            # Save .meta file using pickle (compatible with wan21.py dataloader)
            output_path = output_dir / f"{Path(file_name).stem}.meta"
            with open(output_path, "wb") as f:
                pickle.dump(result, f)

            successful += 1

        except Exception as e:
            logger.error(f"Failed to process {item.get('file_name', 'unknown')}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            failed += 1

    logger.info("=" * 80)
    logger.info("Preprocessing complete!")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
