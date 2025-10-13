import os
import torch
from dist_utils import print0

def prepare_t2v_conditioning(pipe, video_latents: torch.Tensor, timesteps: torch.Tensor, bf16):
    """
    WAN 2.1 T2V: 16-channel latents only (no concat conditioning).
    Compute noise/noising in fp32, return bf16 latents for model input,
    BUT keep noise in fp32 for loss stability.
    """
    debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

    if debug_mode:
        print0("[DEBUG] === T2V CONDITIONING (16-channel) ===")
        print0(f"[DEBUG] Input video_latents: {tuple(video_latents.shape)}  dtype={video_latents.dtype}")
        print0(f"[DEBUG] Input timesteps: {tuple(timesteps.shape)}  dtype={timesteps.dtype}")

    # Ensure fp32 math for the noising step
    latents_f32 = video_latents.float()

    # fp32 noise (do NOT bf16 this)
    noise = torch.randn(
        latents_f32.shape,
        device=latents_f32.device,
        dtype=torch.float32,
    )

    # Scheduler expects fp32 tensors for stable math
    noisy_latents = pipe.scheduler.add_noise(latents_f32, noise, timesteps)

    if debug_mode:
        nl = noisy_latents
        print0(f"[DEBUG] Noisy latents shape: {tuple(nl.shape)}  dtype={nl.dtype}")
        print0(f"[DEBUG] Noisy range: [{nl.min().item():.3f}, {nl.max().item():.3f}]")

    # WAN 2.1 T2V: enforce 16 channels
    if noisy_latents.shape[1] != 16:
        raise RuntimeError(f"Expected 16 channels for T2V, got {noisy_latents.shape[1]}")

    # Return: bf16 inputs for the transformer, fp32 noise for the loss/target
    return noisy_latents.to(bf16), noise  # keep noise fp32
