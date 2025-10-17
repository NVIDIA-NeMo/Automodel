#!/usr/bin/env python3
# decode_debug_videos.py - Convert saved debug latents into actual videos
# FIXED: dtype mismatch and video export issues

import argparse
import glob
import os
from pathlib import Path

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Decode DMD Debug Videos")
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="./debug_videos",
        help="Directory containing .pt latent files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for MP4 files (default: {debug_dir}_decoded)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Model ID to load VAE from"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second for output videos"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only process files matching this pattern (e.g., 'step000000*' or '*final*')"
    )
    return parser.parse_args()


def decode_debug_videos(debug_output_dir: str, vae, output_video_dir: str, fps: int = 16, filter_pattern: str = None):
    """
    Decode saved debug latents into videos.
    
    Args:
        debug_output_dir: Directory with saved .pt latent files
        vae: WAN VAE model for decoding
        output_video_dir: Where to save decoded MP4 files
        fps: Frames per second for videos
        filter_pattern: Optional glob pattern to filter files
    """
    Path(output_video_dir).mkdir(exist_ok=True, parents=True)
    
    # Find all .pt files
    if filter_pattern:
        search_pattern = f"{debug_output_dir}/{filter_pattern}.pt"
    else:
        search_pattern = f"{debug_output_dir}/*.pt"
    
    latent_files = sorted(glob.glob(search_pattern))
    
    if not latent_files:
        print(f"❌ No .pt files found matching: {search_pattern}")
        return
    
    print(f"✓ Found {len(latent_files)} debug latent files")
    print(f"📁 Output directory: {output_video_dir}")
    print(f"🎬 FPS: {fps}")
    print()
    
    vae.eval()
    device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    
    print(f"VAE device: {device}, dtype: {vae_dtype}")
    print()
    
    # Process each file
    failed = []
    
    for latent_file in tqdm(latent_files, desc="Decoding videos"):
        try:
            # Load latent
            data = torch.load(latent_file, map_location='cpu')
            latent = data['latent']
            
            # Get metadata
            global_step = data.get('global_step', 0)
            denoising_step = data.get('denoising_step', 0)
            timestep = data.get('timestep', 0)
            stage = data.get('stage', 'unknown')
            
            # CRITICAL FIX: Match VAE dtype (usually bfloat16)
            latent = latent.to(device=device, dtype=vae_dtype)
            
            # Decode with VAE
            with torch.no_grad():
                video = vae.decode(latent).sample
            
            # Move to CPU and convert to float32 for video export
            video = video.cpu().float()
            
            # Prepare output filename
            input_name = Path(latent_file).stem
            output_name = f"{input_name}.mp4"
            output_path = Path(output_video_dir) / output_name
            
            # Save video
            # video shape: [B, C, F, H, W] -> take first batch item
            video_frames = video[0]  # [C, F, H, W]
            
            # Convert to numpy and correct format for export_to_video
            # Expected: List of PIL Images or numpy array [F, H, W, C]
            video_frames = video_frames.permute(1, 2, 3, 0)  # [F, H, W, C]
            
            # Normalize from [-1, 1] to [0, 255] uint8
            video_frames = ((video_frames + 1.0) / 2.0 * 255).clamp(0, 255)
            video_frames = video_frames.numpy().astype('uint8')
            
            export_to_video(video_frames, str(output_path), fps=fps)
            
            # Optional: print info for each file
            # print(f"  ✓ {output_name} (step={global_step}, denoise={denoising_step}, t={timestep}, stage={stage})")
            
        except Exception as e:
            failed.append((latent_file, str(e)))
            tqdm.write(f"  ✗ Failed: {Path(latent_file).name} - {e}")
    
    # Summary
    print()
    print("=" * 80)
    print(f"✓ Successfully decoded: {len(latent_files) - len(failed)}/{len(latent_files)}")
    if failed:
        print(f"✗ Failed: {len(failed)}/{len(latent_files)}")
        print("\nFailed files:")
        for file, error in failed[:5]:  # Show first 5 failures
            print(f"  - {Path(file).name}: {error}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    print(f"📁 Videos saved to: {output_video_dir}")
    print("=" * 80)


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🎥 DMD Debug Video Decoder")
    print("=" * 80)
    print()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"{args.debug_dir}_decoded"
    
    print(f"📂 Loading VAE from: {args.model_id}")
    
    # Load pipeline (we only need the VAE)
    pipe = WanPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        transformer=None,  # Don't load transformer to save memory
        text_encoder=None,  # Don't load text encoder
    )
    
    # Move VAE to GPU
    pipe.vae.to("cuda")
    
    # Enable VAE optimizations for memory efficiency
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    print("✓ VAE loaded successfully")
    print()
    
    # Decode videos
    decode_debug_videos(
        debug_output_dir=args.debug_dir,
        vae=pipe.vae,
        output_video_dir=args.output_dir,
        fps=args.fps,
        filter_pattern=args.filter,
    )


if __name__ == "__main__":
    main()