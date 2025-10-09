import os
import torch
from dist_utils import print0


def prepare_t2v_conditioning(pipe, video_latents: torch.Tensor, timesteps: torch.Tensor, bf16):
    """
    T2V conditioning for WAN 2.1 - simpler than I2V, just noisy latents.
    WAN 2.1 T2V uses 16-channel input (no conditioning concatenation).
    """
    # Only print debug info if DEBUG_TRAINING env var is set
    debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"
    
    if debug_mode:
        print0("[DEBUG] === T2V CONDITIONING (16-channel) ===")
        print0(f"[DEBUG] Input video_latents: {video_latents.shape}")
        print0(f"[DEBUG] Input timesteps: {timesteps.shape}")

    batch_size, channels, frames, height, width = video_latents.shape

    # Generate noise matching input latents (16 channels)
    noise = torch.randn_like(video_latents)

    # Apply scheduler noise based on timesteps
    noisy_latents = pipe.scheduler.add_noise(video_latents, noise, timesteps)
    
    if debug_mode:
        print0(f"[DEBUG] Noisy latents shape: {noisy_latents.shape}")
        print0(f"[DEBUG] Final T2V input shape: {noisy_latents.shape}")
        print0(f"[DEBUG] Input range: [{noisy_latents.min():.3f}, {noisy_latents.max():.3f}]")

    # Verify we have exactly 16 channels
    if noisy_latents.shape[1] != 16:
        raise RuntimeError(f"Expected 16 channels for T2V, got {noisy_latents.shape[1]}")

    return noisy_latents.to(bf16), noise.to(bf16)