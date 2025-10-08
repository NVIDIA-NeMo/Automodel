import torch
from dist_utils import print0

def prepare_i2v_conditioning(pipe, video_latents: torch.Tensor, timesteps: torch.Tensor, bf16):
    """
    Enhanced I2V conditioning function that creates proper 36-channel input for WAN 2.2.
    Replicates the pipeline's prepare_latents logic to create mask + condition concatenation.
    """
    print0(f"[DEBUG] === I2V CONDITIONING (36-channel) ===")
    print0(f"[DEBUG] Input video_latents: {video_latents.shape}")
    print0(f"[DEBUG] Input timesteps: {timesteps.shape}")
    
    batch_size, channels, frames, height, width = video_latents.shape
    
    # Generate noise matching input latents (16 channels)
    noise = torch.randn_like(video_latents)
    
    # Apply scheduler noise based on timesteps
    noisy_latents = pipe.scheduler.add_noise(video_latents, noise, timesteps)
    print0(f"[DEBUG] Noisy latents shape: {noisy_latents.shape}")
    
    # === CREATE CONDITIONING (20 channels) ===
    
    # 1. Create mask (1 channel): 0 for first frame, 1 for other frames
    mask = torch.ones(batch_size, 1, frames, height, width, device=video_latents.device, dtype=bf16)
    mask[:, :, 0] = 0.0  # First frame is not masked (given)
    print0(f"[DEBUG] Mask shape: {mask.shape}")
    print0(f"[DEBUG] Mask first frame: {mask[:, :, 0].sum()}, other frames: {mask[:, :, 1:].sum()}")
    
    # 2. Create first frame conditioning (16 channels)
    # Take first frame and broadcast to all time positions
    first_frame = video_latents[:, :, :1]  # [B, 16, 1, H, W]
    first_frame_broadcast = first_frame.expand(-1, -1, frames, -1, -1)  # [B, 16, T, H, W]
    print0(f"[DEBUG] First frame conditioning shape: {first_frame_broadcast.shape}")
    
    # 3. Create additional conditioning channels (3 channels for padding to reach 20 total)
    # These might be zero padding or other model-specific conditioning
    extra_conditioning = torch.zeros(batch_size, 3, frames, height, width, 
                                   device=video_latents.device, dtype=bf16)
    print0(f"[DEBUG] Extra conditioning shape: {extra_conditioning.shape}")
    
    # Concatenate all conditioning: 1 + 16 + 3 = 20 channels
    conditioning = torch.cat([mask, first_frame_broadcast, extra_conditioning], dim=1)
    print0(f"[DEBUG] Full conditioning shape: {conditioning.shape}")
    
    # === FINAL CONCATENATION: 16 + 20 = 36 channels ===
    conditioned_input = torch.cat([noisy_latents, conditioning], dim=1)
    
    print0(f"[DEBUG] Final 36-channel input shape: {conditioned_input.shape}")
    print0(f"[DEBUG] Channel breakdown: 16 (noisy) + 20 (conditioning) = {conditioned_input.shape[1]} total")
    print0(f"[DEBUG] Input range: [{conditioned_input.min():.3f}, {conditioned_input.max():.3f}]")
    
    # Verify we have exactly 36 channels
    if conditioned_input.shape[1] != 36:
        raise RuntimeError(f"Expected 36 channels, got {conditioned_input.shape[1]}")
    
    return conditioned_input.to(bf16), noise.to(bf16), mask.to(bf16)