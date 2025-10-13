import logging
import os
import pickle
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKLWan

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDecoder:
    def __init__(
        self,
        wan21_model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enable_memory_optimization: bool = True,
    ):
        """
        Initialize the video decoder for converting .meta files back to videos.

        Args:
            wan21_model_id: Hugging Face model ID for Wan2.1 VAE
            device: Device to run inference on
            enable_memory_optimization: Enable Wan's built-in slicing and tiling
        """
        self.device = device
        self.wan21_model_id = wan21_model_id
        self.enable_memory_optimization = enable_memory_optimization

        # Load Wan2.1 VAE for decoding
        logger.info(f"Loading Wan2.1 VAE from {wan21_model_id}...")
        self.vae = self._load_vae()

    def _load_vae(self):
        """Load Wan2.1 VAE from Hugging Face with memory optimization."""
        logger.info("Loading Wan2.1 VAE for decoding...")
        vae = AutoencoderKLWan.from_pretrained(
            self.wan21_model_id, subfolder="vae", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
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

        # Log VAE config for debugging
        logger.info("Wan2.1 VAE loaded successfully")

        # Verify this is a Wan2.1 VAE
        if hasattr(vae.config, "latents_mean") and hasattr(vae.config, "latents_std"):
            logger.info("✅ Found latents_mean and latents_std in VAE config (Wan2.1 format)")
            logger.info(f"   z_dim: {vae.config.z_dim if hasattr(vae.config, 'z_dim') else 'unknown'}")
        else:
            logger.error("❌ No latents_mean/latents_std found in VAE config")
            logger.error("This doesn't appear to be a Wan2.1 VAE!")
            raise ValueError("VAE config missing latents_mean and latents_std (required for Wan2.1)")

        return vae

    def load_meta_file(self, meta_path: str) -> Dict:
        """
        Load processed data from .meta file.

        Args:
            meta_path: Path to .meta file

        Returns:
            Dictionary containing text_embeddings, video_latents, and metadata
        """
        logger.info(f"Loading .meta file: {meta_path}")

        with open(meta_path, "rb") as f:
            data = pickle.load(f)

        # Log information about the loaded data
        logger.info(f"Loaded data keys: {list(data.keys())}")
        logger.info(f"Video latents shape: {data['video_latents'].shape}")
        logger.info(f"Text embeddings shape: {data['text_embeddings'].shape}")
        logger.info(f"Original filename: {data.get('original_filename', 'N/A')}")

        # Check model version
        model_version = data.get("model_version", "unknown")
        logger.info(f"Model version: {model_version}")
        if model_version != "wan2.1":
            logger.warning(f"⚠️  This .meta file was created with {model_version}, but you're using a Wan2.1 decoder!")
            logger.warning("   Decoding may not work correctly if versions don't match.")

        # Check if first frame exists
        if "first_frame" in data:
            logger.info(f"First frame found: {data['first_frame'].shape}")

        # Check encoding mode
        encoding_mode = data.get("deterministic_latents", "unknown")
        logger.info(f"Encoding mode: {encoding_mode}")

        return data

    def save_first_frame_as_jpeg(self, first_frame: np.ndarray, output_path: str, quality: int = 95):
        """
        Save the first frame as a JPEG image.

        Args:
            first_frame: RGB numpy array (H, W, C) with values in [0, 255]
            output_path: Path to save the JPEG file
            quality: JPEG quality (1-100)
        """
        logger.info(f"Saving first frame to: {output_path}")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Convert RGB to BGR for OpenCV
        first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)

        # Save as JPEG with specified quality
        success = cv2.imwrite(output_path, first_frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

        if not success:
            raise RuntimeError(f"Failed to save JPEG to {output_path}")

        logger.info(f"✅ Successfully saved first frame JPEG: {output_path}")
        logger.info(f"   Image dimensions: {first_frame.shape[0]}x{first_frame.shape[1]}")

    def decode_video_latents(self, video_latents: torch.Tensor) -> torch.Tensor:
        """
        Decode video latents back to video frames using Wan2.1 VAE.
        Uses Wan's built-in memory optimization instead of manual chunking.

        Args:
            video_latents: Normalized video latents from .meta file

        Returns:
            Decoded video tensor in [0, 1] range
        """
        logger.info(f"Decoding video latents: {video_latents.shape}")
        logger.info(f"Input latents range: [{video_latents.min():.3f}, {video_latents.max():.3f}]")

        # Move to device and ensure correct dtype
        video_latents = video_latents.to(device=self.device, dtype=self.vae.dtype)

        # De-normalize latents (reverse the Wan2.1 encoding normalization)
        # Wan2.1 uses per-channel normalization: (z - mean) / std
        # So we reverse it: z * std + mean
        if not (hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std")):
            raise ValueError("Wan2.1 VAE requires latents_mean and latents_std in config")

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.vae.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.vae.dtype)

        logger.info("Using Wan2.1 per-channel de-normalization")
        logger.info(f"latents_mean: {latents_mean.tolist()}")
        logger.info(f"latents_std: {latents_std.tolist()}")

        # Reshape for broadcasting: (1, C, 1, 1, 1) for 5D tensors
        latents_mean = latents_mean.view(1, -1, 1, 1, 1)
        latents_std = latents_std.view(1, -1, 1, 1, 1)

        # De-normalize: z * std + mean (reverse of (z - mean) / std)
        video_latents = video_latents * latents_std + latents_mean
        logger.info("Applied Wan2.1 VAE de-normalization")
        logger.info(f"De-normalized latents range: [{video_latents.min():.3f}, {video_latents.max():.3f}]")

        B, C, T, H, W = video_latents.shape
        logger.info(f"De-normalized latents shape: {video_latents.shape}")

        # Use Wan's built-in memory optimization
        # No manual chunking needed - VAE handles it internally with slicing/tiling
        with torch.no_grad():
            logger.info("Decoding with Wan2.1 VAE (using built-in memory optimization)")
            decoded_video = self.vae.decode(video_latents).sample
            logger.info(f"Decoded video range: [{decoded_video.min():.3f}, {decoded_video.max():.3f}]")

        # Convert from [-1, 1] to [0, 1] range
        decoded_video = (decoded_video + 1.0) / 2.0
        logger.info(f"After [-1,1] to [0,1] conversion: [{decoded_video.min():.3f}, {decoded_video.max():.3f}]")

        # Clamp to ensure values are in [0, 1]
        decoded_video = torch.clamp(decoded_video, 0.0, 1.0)
        logger.info(f"After clamping: [{decoded_video.min():.3f}, {decoded_video.max():.3f}]")

        logger.info(f"Final decoded video shape: {decoded_video.shape}")
        return decoded_video

    def save_video_as_mp4(self, video_tensor: torch.Tensor, output_path: str, fps: int = 24):
        """
        Save decoded video tensor as MP4 file.

        Args:
            video_tensor: Decoded video tensor of shape (B, C, T, H, W) in [0, 1] range
            output_path: Path to save the MP4 file
            fps: Frames per second for the output video
        """
        logger.info(f"Saving video to: {output_path}")

        # Convert to numpy and rearrange dimensions
        # From (B, C, T, H, W) to (T, H, W, C)
        video_tensor = video_tensor.squeeze(0)  # Remove batch dimension: (C, T, H, W)
        video_tensor = video_tensor.permute(1, 2, 3, 0)  # (T, H, W, C)
        video_np = video_tensor.cpu().numpy()

        # Convert from [0, 1] to [0, 255] and to uint8
        video_np = (video_np * 255).astype(np.uint8)

        logger.info(f"Video array shape: {video_np.shape}")
        logger.info(f"Video array dtype: {video_np.dtype}")
        logger.info(f"Video array range: [{video_np.min()}, {video_np.max()}]")

        # Get video dimensions
        num_frames, height, width, channels = video_np.shape

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        # Write frames
        for frame_idx in range(num_frames):
            frame = video_np[frame_idx]

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write frame
            out.write(frame_bgr)

        # Release video writer
        out.release()

        logger.info(f"✅ Successfully saved video: {output_path}")
        logger.info(f"   Video info: {num_frames} frames, {height}x{width}, {fps} FPS")

    def decode_meta_to_video(self, meta_path: str, output_path: str, fps: int = 24, jpeg_quality: int = 95):
        """
        Complete pipeline: load .meta file, save first frame as JPEG, and save video as MP4.

        Args:
            meta_path: Path to input .meta file
            output_path: Path to output MP4 file
            fps: Frames per second for output video
            jpeg_quality: JPEG quality for first frame (1-100)
        """
        logger.info(f"Converting {meta_path} -> {output_path}")

        try:
            # Load meta file
            data = self.load_meta_file(meta_path)

            # Save first frame as JPEG if available
            if "first_frame" in data:
                # Generate first frame JPEG path (same directory and name as video, but .jpg)
                output_path_obj = Path(output_path)
                first_frame_path = output_path_obj.parent / f"{output_path_obj.stem}.jpg"
                self.save_first_frame_as_jpeg(data["first_frame"], str(first_frame_path), jpeg_quality)
            else:
                logger.warning("No 'first_frame' found in .meta file, skipping JPEG save")

            # Extract video latents
            video_latents = data["video_latents"]

            # Decode latents to video
            decoded_video = self.decode_video_latents(video_latents)

            # Save as MP4
            self.save_video_as_mp4(decoded_video, output_path, fps)

            logger.info(f"✅ Successfully converted {meta_path} to {output_path}")

        except Exception as e:
            logger.error(f"❌ Error converting {meta_path}: {e}")
            raise

    def decode_folder(self, meta_folder: str, output_folder: str, fps: int = 24, jpeg_quality: int = 95):
        """
        Decode all .meta files in a folder to MP4 videos and JPEG first frames.

        Args:
            meta_folder: Path to folder containing .meta files
            output_folder: Path to folder for output MP4 files and JPEG images
            fps: Frames per second for output videos
            jpeg_quality: JPEG quality for first frames (1-100)
        """
        meta_folder = Path(meta_folder)
        output_folder = Path(output_folder)

        # Create output directory
        output_folder.mkdir(parents=True, exist_ok=True)

        # Find all .meta files
        meta_files = list(meta_folder.glob("*.meta"))

        if not meta_files:
            logger.warning(f"No .meta files found in {meta_folder}")
            return

        logger.info(f"Found {len(meta_files)} .meta files to decode")

        for i, meta_file in enumerate(meta_files):
            logger.info(f"Progress: {i + 1}/{len(meta_files)}")

            # Generate output filename
            output_file = output_folder / f"{meta_file.stem}.mp4"

            try:
                self.decode_meta_to_video(str(meta_file), str(output_file), fps, jpeg_quality)
            except Exception as e:
                logger.error(f"Failed to decode {meta_file}: {e}")
                continue

        logger.info(f"✅ Finished decoding {len(meta_files)} videos to {output_folder}")


def main():
    """Main function to run the video decoding."""
    import argparse

    parser = argparse.ArgumentParser(description="Decode Wan2.1 .meta files to MP4 videos")
    parser.add_argument("--input", "-i", required=True, help="Input .meta file or folder containing .meta files")
    parser.add_argument("--output", "-o", required=True, help="Output MP4 file or folder for MP4 files")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for output video (default: 24)")
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help="Wan2.1 model ID (default: Wan-AI/Wan2.1-T2V-14B-Diffusers, also supports Wan-AI/Wan2.1-T2V-1.3B-Diffusers)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--no-memory-optimization", action="store_true", help="Disable Wan's built-in memory optimization"
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95, help="JPEG quality for first frame (1-100, default: 95)"
    )

    args = parser.parse_args()

    # Initialize decoder
    decoder = VideoDecoder(
        wan21_model_id=args.model, device=args.device, enable_memory_optimization=not args.no_memory_optimization
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file() and input_path.suffix == ".meta":
        # Single file mode
        if not output_path.suffix:
            output_path = output_path / f"{input_path.stem}.mp4"

        decoder.decode_meta_to_video(str(input_path), str(output_path), args.fps, args.jpeg_quality)

    elif input_path.is_dir():
        # Folder mode
        decoder.decode_folder(str(input_path), str(output_path), args.fps, args.jpeg_quality)

    else:
        logger.error(f"Invalid input: {input_path} (must be .meta file or directory)")


if __name__ == "__main__":
    main()


# Example usage:
"""
# Decode single .meta file (creates both .mp4 and .jpg)
python decode_wan21.py --input processed_meta/video1.meta --output decoded_videos/video1.mp4 --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers

# Decode all .meta files in a folder
python decode_wan21.py --input processed_meta/ --output decoded_videos/

# Use Wan2.1 1.3B model
python decode_wan21.py --input processed_meta/ --output decoded_videos/ --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers

# Custom JPEG quality and FPS
python decode_wan21.py --input processed_meta/ --output decoded_videos/ --fps 16 --jpeg-quality 90

# Disable memory optimization if you have enough VRAM
python decode_wan21.py --input processed_meta/ --output decoded_videos/ --no-memory-optimization

# Programmatic usage
from decode_wan21 import VideoDecoder

decoder = VideoDecoder("Wan-AI/Wan2.1-T2V-14B-Diffusers")

# Single file (creates video1.mp4 and video1.jpg)
decoder.decode_meta_to_video("processed_meta/video1.meta", "output/video1.mp4")

# Entire folder
decoder.decode_folder("processed_meta", "decoded_videos", fps=24, jpeg_quality=95)
"""
