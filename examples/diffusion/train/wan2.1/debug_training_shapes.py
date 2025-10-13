#!/usr/bin/env python3
"""
Debug script to understand WAN 2.1 T2V model's expected shapes.
Run this BEFORE training to verify everything is correct.
"""

import torch
from diffusers import WanPipeline
from dataloader import MetaFilesDataset

def debug_model_shapes(model_id: str, meta_folder: str):
    """Debug the shapes expected by WAN 2.1 T2V model."""
    
    print("=" * 80)
    print("WAN 2.1 T2V SHAPE DEBUGGER")
    print("=" * 80)
    
    # Load pipeline
    print(f"\n[1] Loading pipeline: {model_id}")
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    
    # Print scheduler config
    print(f"\n[2] Scheduler Configuration:")
    print(f"  - Type: {type(pipe.scheduler).__name__}")
    print(f"  - num_train_timesteps: {pipe.scheduler.config.num_train_timesteps}")
    print(f"  - prediction_type: {getattr(pipe.scheduler.config, 'prediction_type', 'unknown')}")
    
    # Check if pipeline has expand_timesteps config
    print(f"\n[3] Pipeline Configuration:")
    print(f"  - expand_timesteps: {getattr(pipe.config, 'expand_timesteps', False)}")
    
    # Print transformer config
    print(f"\n[4] Transformer Configuration:")
    print(f"  - in_channels: {pipe.transformer.config.in_channels}")
    print(f"  - out_channels: {pipe.transformer.config.out_channels}")
    
    # Load a sample from dataset
    print(f"\n[5] Loading sample from dataset: {meta_folder}")
    dataset = MetaFilesDataset(meta_folder=meta_folder, device="cpu", max_files=1)
    
    if len(dataset) == 0:
        print("ERROR: No .meta files found!")
        return
    
    sample = dataset[0]
    video_latents = sample["video_latents"].to("cuda", dtype=torch.bfloat16)
    text_embeddings = sample["text_embeddings"].to("cuda", dtype=torch.bfloat16)
    
    print(f"\n[6] Sample Data Shapes (raw from dataset):")
    print(f"  - video_latents: {video_latents.shape}")
    print(f"  - text_embeddings: {text_embeddings.shape}")
    
    # Handle potential extra dimensions
    while video_latents.ndim > 5:
        print(f"  - Squeezing video_latents from {video_latents.shape}...")
        video_latents = video_latents.squeeze(0)
    
    while text_embeddings.ndim > 3:
        print(f"  - Squeezing text_embeddings from {text_embeddings.shape}...")
        text_embeddings = text_embeddings.squeeze(0)
    
    # Add batch dimension if needed
    if video_latents.ndim == 4:
        video_latents = video_latents.unsqueeze(0)
    if text_embeddings.ndim == 2:
        text_embeddings = text_embeddings.unsqueeze(0)
    
    print(f"\n[6b] Sample Data Shapes (after dimension handling):")
    print(f"  - video_latents: {video_latents.shape}")
    print(f"  - text_embeddings: {text_embeddings.shape}")
    
    batch_size, channels, frames, height, width = video_latents.shape
    
    # Generate noise and timesteps
    print(f"\n[7] Generating noise and timesteps...")
    noise = torch.randn_like(video_latents)
    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device="cuda")
    
    print(f"  - noise shape: {noise.shape}")
    print(f"  - base timesteps shape: {timesteps.shape}")
    
    # Add noise
    noisy_latents = pipe.scheduler.add_noise(video_latents, noise, timesteps)
    print(f"  - noisy_latents shape: {noisy_latents.shape}")
    
    # Test different timestep formats
    print(f"\n[8] Testing timestep formats...")
    
    # Format 1: Scalar timesteps (batch_size,)
    print(f"\n  Format 1: Scalar timesteps {timesteps.shape}")
    try:
        with torch.no_grad():
            output1 = pipe.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )
            if isinstance(output1, tuple):
                output1 = output1[0]
            print(f"  ✅ SUCCESS: Output shape = {output1.shape}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # Format 2: Expanded timesteps (batch_size, seq_len) with 2x2 patchification
    seq_len = frames * (height // 2) * (width // 2)
    expanded_timesteps = timesteps.unsqueeze(1).expand(batch_size, seq_len)
    print(f"\n  Format 2: Expanded timesteps {expanded_timesteps.shape} (2x2 patchification)")
    try:
        with torch.no_grad():
            output2 = pipe.transformer(
                hidden_states=noisy_latents,
                timestep=expanded_timesteps,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )
            if isinstance(output2, tuple):
                output2 = output2[0]
            print(f"  ✅ SUCCESS: Output shape = {output2.shape}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # Format 3: Expanded timesteps (batch_size, seq_len) with 1x1 patchification
    seq_len_full = frames * height * width
    expanded_timesteps_full = timesteps.unsqueeze(1).expand(batch_size, seq_len_full)
    print(f"\n  Format 3: Expanded timesteps {expanded_timesteps_full.shape} (1x1 patchification)")
    try:
        with torch.no_grad():
            output3 = pipe.transformer(
                hidden_states=noisy_latents,
                timestep=expanded_timesteps_full,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )
            if isinstance(output3, tuple):
                output3 = output3[0]
            print(f"  ✅ SUCCESS: Output shape = {output3.shape}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    # Determine which format worked
    working_format = None
    output_shape = None
    
    try:
        with torch.no_grad():
            test_out = pipe.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )
            if isinstance(test_out, tuple):
                test_out = test_out[0]
            working_format = "scalar"
            output_shape = test_out.shape
    except:
        pass
    
    if working_format is None:
        try:
            with torch.no_grad():
                test_out = pipe.transformer(
                    hidden_states=noisy_latents,
                    timestep=expanded_timesteps,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )
                if isinstance(test_out, tuple):
                    test_out = test_out[0]
                working_format = "expanded_2x2"
                output_shape = test_out.shape
        except:
            pass
    
    if working_format:
        print(f"\n✅ Working timestep format: {working_format}")
        print(f"   Model output shape: {output_shape}")
        print(f"   Expected target shape: {noise.shape}")
        
        if output_shape == noise.shape:
            print(f"\n✅ SHAPES MATCH! Training should work.")
        else:
            print(f"\n❌ SHAPE MISMATCH!")
            print(f"   Model output: {output_shape}")
            print(f"   Target (noise): {noise.shape}")
            print(f"\n   This will cause the error you're seeing!")
            print(f"   You may need to adjust how targets are calculated.")
    else:
        print(f"\n❌ NO WORKING TIMESTEP FORMAT FOUND!")
        print(f"   The model may require a different input format.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--meta_folder", type=str, required=True)
    args = parser.parse_args()
    
    debug_model_shapes(args.model_id, args.meta_folder)


# Usage:
# python debug_training_shapes.py --meta_folder /path/to/processed_meta