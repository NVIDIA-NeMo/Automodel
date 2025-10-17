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

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5EncoderModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoPreprocessor:
    def __init__(
        self,
        video_folder: str = "clipped_video",
        wan21_model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        output_folder: str = "processed_meta",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        deterministic_latents: bool = True,
        enable_memory_optimization: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        resize_mode: str = "bilinear",
        maintain_aspect_ratio: bool = True,
        center_crop: bool = False,
    ):
        """
        Initialize the video preprocessor for Wan2.1 fine-tuning.

        Args:
            video_folder: Path to folder containing videos and meta.json
            wan21_model_id: Hugging Face model ID for Wan2.1 (e.g., "Wan-AI/Wan2.1-T2V-14B-Diffusers")
            output_folder: Path to folder where .meta files will be saved
            device: Device to run inference on
            deterministic_latents: If True, use posterior mean instead of sampling (recommended for clean reconstructions)
            enable_memory_optimization: Enable Wan's built-in slicing and tiling
            target_size: Target (height, width) for resizing. If None, no resizing is performed
            resize_mode: Interpolation mode for resizing ('bilinear', 'bicubic', 'nearest', 'area')
            maintain_aspect_ratio: If True, maintain aspect ratio when resizing
            center_crop: If True, center crop to target size after resizing (when maintaining aspect ratio)
        """
        self.video_folder = Path(video_folder)
        self.output_folder = Path(output_folder)
        self.device = device
        self.wan21_model_id = wan21_model_id
        self.deterministic_latents = deterministic_latents
        self.enable_memory_optimization = enable_memory_optimization

        # Resize parameters
        self.target_size = target_size
        self.resize_mode = resize_mode
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.center_crop = center_crop

        # Validate resize parameters
        if self.target_size is not None:
            if len(self.target_size) != 2 or any(s <= 0 for s in self.target_size):
                raise ValueError("target_size must be a tuple of (height, width) with positive values")
            logger.info(f"Video resizing enabled: target size = {self.target_size} (H x W)")
            logger.info(f"Resize mode: {self.resize_mode}")
            logger.info(f"Maintain aspect ratio: {self.maintain_aspect_ratio}")
            if self.maintain_aspect_ratio and self.center_crop:
                logger.info("Center crop enabled (will crop to exact target size after aspect-preserving resize)")
        else:
            logger.info("Video resizing disabled - using original video dimensions")

        # Map resize modes to OpenCV interpolation flags
        self.interpolation_map = {
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }

        if self.resize_mode not in self.interpolation_map:
            raise ValueError(
                f"Invalid resize_mode '{self.resize_mode}'. Choose from: {list(self.interpolation_map.keys())}"
            )

        # Log the encoding mode
        if self.deterministic_latents:
            logger.info("Using DETERMINISTIC latents (posterior mean) - no flares expected")
        else:
            logger.info("Using STOCHASTIC latents (sampling) - may cause temporal flares")

        # Log memory optimization setting
        if self.enable_memory_optimization:
            logger.info("Using Wan's built-in memory optimization (slicing + tiling)")
        else:
            logger.info("Memory optimization disabled - using full tensors")

        # Create output directory if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder created/verified: {self.output_folder}")

        # Load Wan2.1 components
        logger.info(f"Loading Wan2.1 components from {wan21_model_id}...")
        self.text_encoder = self._load_text_encoder()
        self.vae = self._load_vae()
        self.tokenizer = self._load_tokenizer()

        # Load metadata
        self.metadata = self._load_metadata()

    def _load_text_encoder(self):
        """Load Wan2.1 UMT5 text encoder from Hugging Face."""
        logger.info("Loading UMT5 text encoder...")
        text_encoder = UMT5EncoderModel.from_pretrained(
            self.wan21_model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        text_encoder.to(self.device)
        text_encoder.eval()
        logger.info("UMT5 text encoder loaded successfully")
        return text_encoder

    def _load_vae(self):
        """Load Wan2.1 VAE from Hugging Face with memory optimization."""
        logger.info("Loading Wan2.1 VAE...")

        # Load Wan2.1 VAE with correct subfolder
        try:
            vae = AutoencoderKLWan.from_pretrained(
                self.wan21_model_id,
                subfolder="vae",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
        except Exception as e:
            logger.error(f"Failed to load VAE from {self.wan21_model_id}/vae")
            logger.error(f"Error: {e}")
            logger.info("Make sure you're using a valid Wan2.1 model:")
            logger.info("- Wan-AI/Wan2.1-T2V-14B-Diffusers")
            logger.info("- Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
            raise

        vae.to(self.device)
        vae.eval()

        # Enable Wan's built-in memory optimization
        if self.enable_memory_optimization:
            logger.info("Enabling Wan VAE memory optimization...")
            vae.enable_slicing()  # Reduce peak memory by slicing batch
            vae.enable_tiling()  # Tile H/W during encode+decode
            logger.info("✅ Enabled slicing and tiling for memory efficiency")
        else:
            logger.info("Memory optimization disabled - using full tensors")

        # Log VAE configuration
        logger.info("Wan2.1 VAE loaded successfully")
        logger.info(f"VAE config type: {type(vae.config)}")

        # Log the input/output channels to verify correctness
        if hasattr(vae.config, "in_channels"):
            logger.info(f"VAE in_channels: {vae.config.in_channels}")
        if hasattr(vae.config, "out_channels"):
            logger.info(f"VAE out_channels: {vae.config.out_channels}")
        if hasattr(vae.config, "z_dim"):
            logger.info(f"VAE z_dim (latent channels): {vae.config.z_dim}")

        # Wan2.1 uses per-channel normalization with latents_mean and latents_std
        if hasattr(vae.config, "latents_mean"):
            logger.info(f"VAE latents_mean: {vae.config.latents_mean}")
        if hasattr(vae.config, "latents_std"):
            logger.info(f"VAE latents_std: {vae.config.latents_std}")

        # Log scale factors (Wan2.1 typically uses 4x temporal, 8x spatial)
        scale_factor_temporal = getattr(vae.config, "scale_factor_temporal", 4)
        scale_factor_spatial = getattr(vae.config, "scale_factor_spatial", 8)
        logger.info(f"VAE scale_factor_temporal: {scale_factor_temporal}")
        logger.info(f"VAE scale_factor_spatial: {scale_factor_spatial}")

        return vae

    def _load_tokenizer(self):
        """Load UMT5 tokenizer from Hugging Face."""
        logger.info("Loading UMT5 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.wan21_model_id, subfolder="tokenizer")
        logger.info("Tokenizer loaded successfully")
        return tokenizer

    def _load_metadata(self) -> List[Dict]:
        """Load video metadata from meta.json."""
        meta_path = self.video_folder / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {self.video_folder}")

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        logger.info(f"Loaded metadata for {len(metadata)} videos")
        return metadata

    def _calculate_resize_dimensions(self, original_height: int, original_width: int) -> Tuple[int, int]:
        """
        Calculate the target dimensions for resizing based on the resize strategy.

        Args:
            original_height: Original frame height
            original_width: Original frame width

        Returns:
            Tuple of (target_height, target_width)
        """
        if self.target_size is None:
            return original_height, original_width

        target_height, target_width = self.target_size

        if not self.maintain_aspect_ratio:
            # Direct resize to target dimensions
            return target_height, target_width

        # Calculate aspect-preserving dimensions
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height

        if original_aspect > target_aspect:
            # Original is wider - fit to target width
            new_width = target_width
            new_height = int(target_width / original_aspect)
        else:
            # Original is taller - fit to target height
            new_height = target_height
            new_width = int(target_height * original_aspect)

        return new_height, new_width

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize a single frame according to the resize settings.

        Args:
            frame: Input frame as numpy array (H, W, C)

        Returns:
            Resized frame as numpy array
        """
        if self.target_size is None:
            return frame

        original_height, original_width = frame.shape[:2]
        resize_height, resize_width = self._calculate_resize_dimensions(original_height, original_width)

        # Resize the frame
        interpolation = self.interpolation_map[self.resize_mode]
        resized_frame = cv2.resize(frame, (resize_width, resize_height), interpolation=interpolation)

        # Apply center crop if needed
        if self.maintain_aspect_ratio and self.center_crop:
            target_height, target_width = self.target_size

            if resize_height != target_height or resize_width != target_width:
                # Calculate crop coordinates
                y_start = max(0, (resize_height - target_height) // 2)
                x_start = max(0, (resize_width - target_width) // 2)
                y_end = min(resize_height, y_start + target_height)
                x_end = min(resize_width, x_start + target_width)

                # Crop the frame
                resized_frame = resized_frame[y_start:y_end, x_start:x_end]

                # Pad if necessary (in case the resized frame is smaller than target)
                if resized_frame.shape[0] < target_height or resized_frame.shape[1] < target_width:
                    pad_height = max(0, target_height - resized_frame.shape[0])
                    pad_width = max(0, target_width - resized_frame.shape[1])

                    # Pad with zeros (black)
                    resized_frame = np.pad(
                        resized_frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant", constant_values=0
                    )

        return resized_frame

    def extract_first_frame(self, video_path: str, start_frame: int) -> np.ndarray:
        """
        Extract the first frame from the video as RGB image.

        Args:
            video_path: Path to video file
            start_frame: Starting frame index

        Returns:
            First frame as RGB numpy array (H, W, C) with values in [0, 255]
        """
        cap = cv2.VideoCapture(video_path)

        # Set to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read the first frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {start_frame} from {video_path}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply the same resize transformations as video frames
        frame_rgb = self._resize_frame(frame_rgb)

        logger.info(f"Extracted first frame with shape: {frame_rgb.shape}")
        return frame_rgb

    def load_video_frames(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """
        Load video frames and convert to tensor for Wan VAE.

        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index

        Returns:
            Video tensor of shape (batch, channels, num_frames, height, width)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Get original video dimensions for logging
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.target_size is not None:
            resize_height, resize_width = self._calculate_resize_dimensions(original_height, original_width)
            logger.info(f"Original dimensions: {original_height}x{original_width}")
            logger.info(f"Resize dimensions: {resize_height}x{resize_width}")
            if self.maintain_aspect_ratio and self.center_crop:
                logger.info(f"Final dimensions (after crop): {self.target_size[0]}x{self.target_size[1]}")

        # Set to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame if needed
            frame = self._resize_frame(frame)

            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError(f"No frames loaded from {video_path}")

        logger.info(f"Loaded {len(frames)} frames from {video_path}")

        # Convert to numpy array: (num_frames, height, width, channels)
        video_array = np.array(frames)
        logger.info(f"Video array shape: {video_array.shape}")

        # Convert to tensor and rearrange to: (batch, channels, num_frames, height, width)
        video_tensor = torch.from_numpy(video_array)
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (channels, num_frames, height, width)
        video_tensor = video_tensor.unsqueeze(0)  # (batch, channels, num_frames, height, width)

        # Convert to the same dtype as VAE (float16 for GPU, float32 for CPU)
        target_dtype = torch.float16 if self.device == "cuda" else torch.float32
        video_tensor = video_tensor.to(dtype=target_dtype)

        logger.info(f"Final video tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")
        return video_tensor.to(self.device)

    def encode_text(self, caption: str) -> torch.Tensor:
        """
        Encode text caption using Wan2.1 UMT5 text encoder.

        Args:
            caption: Text description of the video

        Returns:
            Text embedding tensor
        """
        # Clean the prompt (as done in Wan2.1 pipeline)
        caption = caption.strip()

        # Tokenize text with UMT5 settings (max_length=512 for Wan2.1)
        inputs = self.tokenizer(
            caption,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode text using UMT5 encoder
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).last_hidden_state

        logger.info(f"Text embeddings shape: {text_embeddings.shape}")
        return text_embeddings

    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        CRITICAL FIX: Encode video using Wan2.1 VAE without manual normalization.
        The VAE encode() already returns normalized latents!

        Args:
            video_tensor: Video tensor of shape (batch, channels, num_frames, height, width)

        Returns:
            Video latent tensor (already normalized by VAE)
        """
        logger.info(f"Input video tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")

        B, C, T, H, W = video_tensor.shape

        # Ensure tensor is on correct device and dtype
        video_tensor = video_tensor.to(device=self.device, dtype=self.vae.dtype)

        # Convert to [-1, 1] range for VAE
        video_tensor = video_tensor * 2.0 - 1.0

        # CRITICAL FIX: Check if VAE encode() returns normalized latents
        # According to Wan2.1 code, the VAE.encode() should handle normalization internally
        # We should NOT manually normalize again!

        logger.info("Encoding with Wan2.1 VAE...")

        with torch.no_grad():
            latent_dist = self.vae.encode(video_tensor)

            if self.deterministic_latents:
                # Use posterior mean for deterministic, flare-free encoding
                video_latents = latent_dist.latent_dist.mean
                logger.info("Using deterministic posterior mean (no flares)")
            else:
                # Use random sampling (training-style, but causes flares in reconstruction)
                video_latents = latent_dist.latent_dist.sample()
                logger.info("Using stochastic sampling (may cause flares)")

        # CRITICAL: Check if we need to normalize
        # If VAE already normalizes, we should NOT normalize again
        # Let's check the latent statistics
        latent_mean = video_latents.mean().item()
        latent_std = video_latents.std().item()

        logger.info(f"Raw latent statistics - mean: {latent_mean:.4f}, std: {latent_std:.4f}")

        # Check if latents are already normalized (mean~0, std~1)
        if abs(latent_mean) < 0.5 and 0.5 < latent_std < 2.0:
            logger.warning("⚠️  Latents appear already normalized! Skipping manual normalization.")
            logger.warning("⚠️  If you see training issues, the VAE might already normalize internally.")
            # Don't normalize again - use raw latents
            final_latents = video_latents
        else:
            # Latents need normalization - apply Wan2.1 per-channel normalization
            if not hasattr(self.vae.config, "latents_mean") or not hasattr(self.vae.config, "latents_std"):
                raise ValueError("Wan2.1 VAE requires latents_mean and latents_std in config")

            latents_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.vae.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.vae.dtype)

            # Reshape for broadcasting: (1, C, 1, 1, 1) for 5D tensors
            latents_mean = latents_mean.view(1, -1, 1, 1, 1)
            latents_std = latents_std.view(1, -1, 1, 1, 1)

            logger.info("Applying Wan2.1 VAE per-channel normalization")
            logger.info(f"latents_mean shape: {latents_mean.shape}, latents_std shape: {latents_std.shape}")

            # Apply Wan2.1 per-channel normalization: (z - mean) / std
            final_latents = (video_latents - latents_mean) / latents_std
            logger.info("Applied Wan2.1 VAE per-channel normalization")

        logger.info(f"Output video latents shape: {final_latents.shape}, dtype: {final_latents.dtype}")
        logger.info(f"Encoding mode: {'deterministic' if self.deterministic_latents else 'stochastic'}")
        logger.info(f"Memory optimization: {'enabled' if self.enable_memory_optimization else 'disabled'}")

        # Final statistics
        final_mean = final_latents.mean().item()
        final_std = final_latents.std().item()
        logger.info(f"Final latent statistics - mean: {final_mean:.4f}, std: {final_std:.4f}")

        return final_latents

    def save_processed_data(
        self,
        video_name: str,
        text_embeddings: torch.Tensor,
        video_latents: torch.Tensor,
        first_frame: np.ndarray,
        metadata: Dict,
    ):
        """
        Save processed text embeddings, video latents, and first frame to binary file.

        Args:
            video_name: Original video filename
            text_embeddings: Encoded text embeddings
            video_latents: Encoded video latents
            first_frame: First frame of the video as RGB numpy array
            metadata: Original metadata for the video
        """
        # Create output filename in the output folder
        video_stem = Path(video_name).stem
        output_path = self.output_folder / f"{video_stem}.meta"
        num_frames = metadata["end_frame"] - metadata["start_frame"] + 1

        # Prepare data for saving
        processed_data = {
            "text_embeddings": text_embeddings.cpu(),
            "video_latents": video_latents.cpu(),
            "first_frame": first_frame,  # Save as numpy array (H, W, C) in [0, 255]
            "metadata": metadata,
            "num_frames": num_frames,
            "original_filename": video_name,
            "original_video_path": str(self.video_folder / video_name),
            "deterministic_latents": self.deterministic_latents,  # Save encoding mode
            "memory_optimization": self.enable_memory_optimization,  # Save memory setting
            "model_version": "wan2.1",  # Mark as Wan2.1 format
            "resize_settings": {  # Save resize settings
                "target_size": self.target_size,
                "resize_mode": self.resize_mode,
                "maintain_aspect_ratio": self.maintain_aspect_ratio,
                "center_crop": self.center_crop,
            },
        }

        # Save as pickle file
        with open(output_path, "wb") as f:
            pickle.dump(processed_data, f)

        logger.info(f"Saved processed data to {output_path}")
        logger.info(f"First frame shape: {first_frame.shape}, dtype: {first_frame.dtype}")
        logger.info(f"Encoding mode saved: {'deterministic' if self.deterministic_latents else 'stochastic'}")
        logger.info(f"Memory optimization: {'enabled' if self.enable_memory_optimization else 'disabled'}")
        logger.info("Model version: Wan2.1")
        if self.target_size is not None:
            logger.info(f"Resize settings saved: {self.target_size} ({self.resize_mode})")

        # Log file sizes for reference
        video_path = self.video_folder / video_name
        if video_path.exists():
            original_size = video_path.stat().st_size / (1024 * 1024)  # MB
            meta_size = output_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Compression: {original_size:.1f}MB → {meta_size:.1f}MB ({meta_size / original_size:.2%})")

    def process_single_video(self, video_metadata: Dict):
        """Process a single video and save the results."""
        video_name = video_metadata["file_name"]
        video_path = self.video_folder / video_name

        if not video_path.exists():
            logger.warning(f"Video file {video_path} not found, skipping...")
            return

        logger.info(f"Processing {video_name}...")

        try:
            # Extract first frame
            logger.info("Step 0: Extracting first frame...")
            first_frame = self.extract_first_frame(str(video_path), video_metadata["start_frame"])
            logger.info(f"Step 0 completed: first_frame shape = {first_frame.shape}")

            # Load video frames
            logger.info("Step 1: Loading video frames...")
            video_tensor = self.load_video_frames(
                str(video_path), video_metadata["start_frame"], video_metadata["end_frame"]
            )
            logger.info(f"Step 1 completed: video_tensor shape = {video_tensor.shape}")

            # Encode text caption
            logger.info("Step 2: Encoding text caption...")
            text_embeddings = self.encode_text(video_metadata["vila_caption"])
            logger.info(f"Step 2 completed: text_embeddings shape = {text_embeddings.shape}")

            # Encode video
            logger.info("Step 3: Encoding video with VAE...")
            video_latents = self.encode_video(video_tensor)
            logger.info(f"Step 3 completed: video_latents shape = {video_latents.shape}")

            # Save processed data
            logger.info("Step 4: Saving processed data...")
            self.save_processed_data(video_name, text_embeddings, video_latents, first_frame, video_metadata)
            logger.info("Step 4 completed")

            logger.info(f"Successfully processed {video_name}")

        except Exception as e:
            import traceback

            logger.error(f"Error processing {video_name}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

    def process_all_videos(self):
        """Process all videos in the folder."""
        logger.info(f"Starting to process {len(self.metadata)} videos...")
        logger.info(f"Model: Wan2.1 ({self.wan21_model_id})")
        logger.info(
            f"Encoding mode: {'deterministic (flare-free)' if self.deterministic_latents else 'stochastic (may have flares)'}"
        )
        logger.info(
            f"Memory optimization: {'enabled (slicing + tiling)' if self.enable_memory_optimization else 'disabled'}"
        )
        if self.target_size is not None:
            logger.info(f"Video resizing: {self.target_size} using {self.resize_mode} interpolation")

        for i, video_metadata in enumerate(self.metadata):
            logger.info(f"Progress: {i + 1}/{len(self.metadata)}")
            self.process_single_video(video_metadata)

        logger.info("Finished processing all videos!")

    def load_processed_data(self, meta_file: str) -> Dict:
        """
        Load processed data from .meta file.

        Args:
            meta_file: Path to .meta file (can be relative to output_folder or absolute path)

        Returns:
            Dictionary containing text_embeddings, video_latents, first_frame, and metadata
        """
        meta_path = Path(meta_file)

        # If it's not an absolute path, assume it's in the output folder
        if not meta_path.is_absolute():
            meta_path = self.output_folder / meta_file

        with open(meta_path, "rb") as f:
            data = pickle.load(f)

        # Check encoding mode and memory optimization of loaded data
        encoding_mode = data.get("deterministic_latents", "unknown")
        memory_opt = data.get("memory_optimization", "unknown")
        model_version = data.get("model_version", "unknown")
        resize_settings = data.get("resize_settings", {})

        logger.info(f"Loaded .meta file with model version: {model_version}")
        logger.info(f"Encoding mode: {encoding_mode}, memory optimization: {memory_opt}")
        if resize_settings.get("target_size"):
            logger.info(f"Resize settings: {resize_settings}")
        if "first_frame" in data:
            logger.info(f"First frame shape: {data['first_frame'].shape}")

        return data

    def list_processed_files(self) -> List[str]:
        """
        List all .meta files in the output folder.

        Returns:
            List of .meta filenames
        """
        meta_files = list(self.output_folder.glob("*.meta"))
        return [f.name for f in meta_files]


def main():
    """Main function to run the preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess videos for Wan2.1 fine-tuning")
    parser.add_argument(
        "--video_folder", default="clipped_video", help="Path to folder containing videos and meta.json"
    )
    parser.add_argument(
        "--output_folder", default="processed_meta", help="Path to folder where .meta files will be saved"
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help="Wan2.1 model ID (e.g., Wan-AI/Wan2.1-T2V-14B-Diffusers or Wan-AI/Wan2.1-T2V-1.3B-Diffusers)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic encoding (sampling) instead of deterministic (may cause flares)",
    )
    parser.add_argument(
        "--no-memory-optimization", action="store_true", help="Disable Wan's built-in memory optimization"
    )

    # Resize arguments
    parser.add_argument("--height", type=int, default=None, help="Target height for video frames")
    parser.add_argument("--width", type=int, default=None, help="Target width for video frames")
    parser.add_argument(
        "--resize_mode",
        default="bilinear",
        choices=["bilinear", "bicubic", "nearest", "area", "lanczos"],
        help="Interpolation mode for resizing",
    )
    parser.add_argument(
        "--no-aspect-ratio",
        action="store_true",
        help="Disable aspect ratio preservation (stretch to exact target size)",
    )
    parser.add_argument(
        "--center-crop", action="store_true", help="Center crop to exact target size after aspect-preserving resize"
    )

    args = parser.parse_args()

    # Set target size
    target_size = None
    if args.height is not None and args.width is not None:
        target_size = (args.height, args.width)
    elif args.height is not None or args.width is not None:
        parser.error("Both --height and --width must be specified together")

    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        video_folder=args.video_folder,
        wan21_model_id=args.model,
        output_folder=args.output_folder,
        device=args.device,
        deterministic_latents=not args.stochastic,  # Default to deterministic
        enable_memory_optimization=not args.no_memory_optimization,  # Default to enabled
        target_size=target_size,
        resize_mode=args.resize_mode,
        maintain_aspect_ratio=not args.no_aspect_ratio,
        center_crop=args.center_crop,
    )

    # Process all videos
    preprocessor.process_all_videos()


if __name__ == "__main__":
    main()
