#!/usr/bin/env python3
"""
Debug script to identify VAE normalization issues.
This will help determine if preprocessing is correct.
"""

import torch
from diffusers import WanPipeline
from dataloader import MetaFilesDataset
import numpy as np

def debug_normalization(model_id: str, meta_folder: str):
    """Debug VAE normalization in preprocessing vs pipeline."""
    
    print("=" * 80)
    print("WAN 2.1 NORMALIZATION DEBUGGER")
    print("=" * 80)
    
    # Load pipeline
    print(f"\n[1] Loading pipeline: {model_id}")
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    
    # Load a sample from preprocessed dataset
    print(f"\n[2] Loading preprocessed sample from: {meta_folder}")
    dataset = MetaFilesDataset(meta_folder=meta_folder, device="cpu", max_files=1)
    
    if len(dataset) == 0:
        print("ERROR: No .meta files found!")
        return
    
    sample = dataset[0]
    video_latents = sample["video_latents"].to("cuda", dtype=torch.bfloat16)
    
    # Handle dimensions
    while video_latents.ndim > 5:
        video_latents = video_latents.squeeze(0)
    if video_latents.ndim == 4:
        video_latents = video_latents.unsqueeze(0)
    
    print(f"\n[3] Preprocessed latent statistics:")
    print(f"  - Shape: {video_latents.shape}")
    print(f"  - Mean: {video_latents.mean().item():.4f}")
    print(f"  - Std: {video_latents.std().item():.4f}")
    print(f"  - Min: {video_latents.min().item():.4f}")
    print(f"  - Max: {video_latents.max().item():.4f}")
    
    # Check per-channel statistics
    print(f"\n[4] Per-channel statistics (first 4 channels):")
    for i in range(min(4, video_latents.shape[1])):
        channel_data = video_latents[0, i]
        print(f"  Channel {i}:")
        print(f"    Mean: {channel_data.mean().item():.4f}")
        print(f"    Std: {channel_data.std().item():.4f}")
    
    # Get VAE config
    print(f"\n[5] VAE Configuration:")
    print(f"  - latents_mean: {pipe.vae.config.latents_mean[:4]}...")
    print(f"  - latents_std: {pipe.vae.config.latents_std[:4]}...")
    
    # Check if latents are normalized
    latents_mean_config = torch.tensor(pipe.vae.config.latents_mean).to("cuda")
    latents_std_config = torch.tensor(pipe.vae.config.latents_std).to("cuda")
    
    print(f"\n[6] Checking normalization status:")
    
    # Expected statistics if properly normalized
    # After normalization: (x - mean) / std should have mean~0, std~1 per channel
    
    # Check if already normalized (mean ~ 0, std ~ 1)
    global_mean = video_latents.mean().item()
    global_std = video_latents.std().item()
    
    if abs(global_mean) < 0.5 and 0.5 < global_std < 2.0:
        print(f"  ✅ Latents appear NORMALIZED (mean≈0, std≈1)")
        print(f"     This means preprocessing already normalized the latents")
        print(f"     Training should work with these as-is")
        normalized_status = "NORMALIZED"
    else:
        print(f"  ❌ Latents appear RAW/UNNORMALIZED")
        print(f"     Mean={global_mean:.4f} (should be ~0)")
        print(f"     Std={global_std:.4f} (should be ~1)")
        normalized_status = "RAW"
    
    # Try denormalizing and see what we get
    print(f"\n[7] Testing pipeline denormalization:")
    
    latents_mean = latents_mean_config.view(1, -1, 1, 1, 1).to(video_latents.dtype)
    latents_std = latents_std_config.view(1, -1, 1, 1, 1).to(video_latents.dtype)
    
    # Pipeline's denormalization (uses 1.0/std trick)
    pipeline_denorm = video_latents / (1.0 / latents_std) + latents_mean
    # This is equivalent to: video_latents * latents_std + latents_mean
    
    print(f"  Pipeline denormalization result:")
    print(f"    Mean: {pipeline_denorm.mean().item():.4f}")
    print(f"    Std: {pipeline_denorm.std().item():.4f}")
    print(f"    Min: {pipeline_denorm.min().item():.4f}")
    print(f"    Max: {pipeline_denorm.max().item():.4f}")
    
    # Try to decode with VAE
    print(f"\n[8] Testing VAE decode with preprocessed latents:")
    
    try:
        with torch.no_grad():
            # Test 1: Direct decode (if latents are already normalized)
            print(f"  Test 1: Direct decode (assuming latents are normalized)")
            decoded_direct = pipe.vae.decode(video_latents)
            print(f"    ✅ Direct decode successful")
            print(f"    Output range: [{decoded_direct.min().item():.3f}, {decoded_direct.max().item():.3f}]")
    except Exception as e:
        print(f"    ❌ Direct decode failed: {e}")
    
    try:
        with torch.no_grad():
            # Test 2: Decode after denormalization
            print(f"\n  Test 2: Decode after denormalization")
            decoded_denorm = pipe.vae.decode(pipeline_denorm)
            print(f"    ✅ Denormalized decode successful")
            print(f"    Output range: [{decoded_denorm.min().item():.3f}, {decoded_denorm.max().item():.3f}]")
    except Exception as e:
        print(f"    ❌ Denormalized decode failed: {e}")
    
    # Analysis
    print(f"\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATIONS:")
    print("=" * 80)
    
    if normalized_status == "NORMALIZED":
        print(f"""
✅ PREPROCESSED LATENTS ARE NORMALIZED
   
   Your preprocessing is CORRECT. The latents are stored in normalized form.
   
   WHAT THIS MEANS FOR TRAINING:
   1. Your training code should use these normalized latents directly ✅
   2. Add noise to these normalized latents ✅
   3. Model predicts in normalized space ✅
   4. Pipeline will denormalize before VAE decode ✅
   
   POTENTIAL ISSUE:
   - Check if the pipeline's denormalization matches your preprocessing!
   - If pipeline uses: latents * std + mean (correct)
   - Your preprocessing should use: (latents - mean) / std (correct)
   
   THE TIMESTEP MISMATCH might be your real issue!
        """)
    else:
        print(f"""
❌ PREPROCESSED LATENTS ARE NOT NORMALIZED!
   
   This is a CRITICAL BUG in your preprocessing!
   
   WHAT'S WRONG:
   - Latents should have mean≈0, std≈1 after normalization
   - Your latents have mean={global_mean:.4f}, std={global_std:.4f}
   
   POSSIBLE CAUSES:
   1. VAE.encode() might already normalize, then you normalize again
   2. Normalization formula might be wrong
   3. Using wrong mean/std values
   
   FIX:
   1. Check if VAE.encode() returns normalized latents already
   2. If yes, DON'T normalize again in preprocessing
   3. If no, verify normalization formula: (x - mean) / std
        """)
    
    print(f"\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--meta_folder", type=str, required=True)
    args = parser.parse_args()
    
    debug_normalization(args.model_id, args.meta_folder)


# Usage:
# python debug_normalization.py --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta