import torch
from typing import Dict
from conditioning import prepare_i2v_conditioning
from dist_utils import print0

def boundary_from_ratio(pipe, ratio: float):
    sch = pipe.scheduler
    num_train = getattr(sch, "num_train_timesteps", None)
    if num_train is None and hasattr(sch, "config") and hasattr(sch.config, "num_train_timesteps"):
        num_train = int(sch.config.num_train_timesteps)
    if num_train is None:
        num_train = 1000
    return max(0, int(ratio * num_train)), num_train

def step_single_transformer(
    pipe,
    model_map,
    transformer_names,
    batch: Dict,
    device,
    bf16,
    boundary_ratio: float,
    train_transformer_2: bool = False,
) -> torch.Tensor:
    """
    Train only one transformer at a time based on the flag.
    This version properly samples timesteps for the chosen transformer and only uses that transformer.
    
    Args:
        train_transformer_2: If False, train 'transformer' with high timesteps. 
                           If True, train 'transformer_2' with low timesteps.
    """
    text_embeddings = batch["text_embeddings"].to(device, dtype=bf16)
    video_latents = batch["video_latents"].to(device, dtype=torch.float32)
    
    print0(f"[DEBUG] Raw video_latents shape: {video_latents.shape}")
    
    if video_latents.ndim == 6:
        video_latents = video_latents.squeeze(1)
        print0(f"[DEBUG] After squeeze: {video_latents.shape}")
    if text_embeddings.ndim == 4:
        text_embeddings = text_embeddings.squeeze(1)

    B, C, T, H, W = video_latents.shape  # C should be 16
    print0(f"[DEBUG] Video latents dimensions: B={B}, C={C}, T={T}, H={H}, W={W}")

    # Reduce frames for memory (keep existing logic)
    if T > 2:  # If more than 16 frames
        video_latents = video_latents[:, :, :2, :, :]  # Take only first 16 frames
        T = 2
        print0(f"[DEBUG] Reduced video to {T} frames for memory: {video_latents.shape}")

    # Create I2V conditioning (36-channel input)
    # 1. Extract first frame and repeat for all timesteps (16 channels)
    first_frame = video_latents[:, :, 0:1, :, :].expand(-1, -1, T, -1, -1)  # [B, 16, T, H, W]
    print0(f"[DEBUG] First frame conditioning shape: {first_frame.shape}")
    
    # 2. Create 4-channel I2V mask (first frame = 1, others = 0)
    mask = torch.zeros(B, 4, T, H, W, device=device, dtype=torch.float32)
    mask[:, 0, 0, :, :] = 1.0  # Mark first frame in first channel
    print0(f"[DEBUG] I2V mask shape: {mask.shape}")
    
    # 3. Concatenate: 16 (video) + 16 (first_frame) + 4 (mask) = 36 channels
    conditioned_input = torch.cat([video_latents, first_frame, mask], dim=1)
    print0(f"[DEBUG] Conditioned input shape: {conditioned_input.shape} (should be 36 channels)")
    
    if conditioned_input.shape[1] != 36:
        raise ValueError(f"Expected 36 channels after conditioning, got {conditioned_input.shape[1]}")

    # Determine timestep boundary for sampling appropriate timesteps
    boundary_ts, num_train = boundary_from_ratio(pipe, boundary_ratio)
    
    # Sample timesteps based on which transformer we're training
    if train_transformer_2:
        # For transformer_2, sample from low timesteps (0 to boundary_ts)
        if boundary_ts <= 0:
            raise ValueError(f"Invalid boundary_ts {boundary_ts} for transformer_2 (should be > 0)")
        timesteps = torch.randint(0, boundary_ts, (B,), device=device, dtype=torch.long)
        active_transformer = "transformer_2"
        print0(f"[DEBUG] Training transformer_2 with timesteps in range [0, {boundary_ts})")
    else:
        # For transformer, sample from high timesteps (boundary_ts to num_train)
        if boundary_ts >= num_train:
            raise ValueError(f"Invalid boundary_ts {boundary_ts} for transformer (should be < {num_train})")
        timesteps = torch.randint(boundary_ts, num_train, (B,), device=device, dtype=torch.long)
        active_transformer = "transformer"
        print0(f"[DEBUG] Training transformer with timesteps in range [{boundary_ts}, {num_train})")

    print0(f"[DEBUG] Boundary timestep: {boundary_ts}")
    print0(f"[DEBUG] Sampled timesteps: {timesteps}")

    # Apply noise conditioning to the 36-channel input
    with torch.no_grad():
        cond, noise, cond_mask = prepare_i2v_conditioning(pipe, conditioned_input, timesteps, bf16)

    print0(f"[DEBUG] Final conditioning shapes:")
    print0(f"[DEBUG]   cond: {cond.shape}")
    print0(f"[DEBUG]   noise: {noise.shape}")
    print0(f"[DEBUG]   cond_mask: {cond_mask.shape}")

    # Verify the chosen transformer exists in model_map
    if active_transformer not in model_map:
        available_transformers = list(model_map.keys())
        raise RuntimeError(f"Active transformer '{active_transformer}' not found in model_map. Available: {available_transformers}")
    
    # Get the active transformer model
    model = model_map[active_transformer]["tp_transformer"]
    
    # Forward pass through the selected transformer only
    with torch.autocast(device_type="cuda", dtype=bf16):
        print0(f"[DEBUG] Calling {active_transformer} with input shape: {cond.shape}")
        
        try:
            out = model(
                hidden_states=cond, 
                timestep=timesteps, 
                encoder_hidden_states=text_embeddings, 
                return_dict=False
            )
            pred = out[0] if isinstance(out, tuple) else out
            print0(f"[DEBUG] {active_transformer} output shape: {pred.shape}")
            
        except Exception as e:
            print0(f"[ERROR] Forward pass failed for {active_transformer}: {e}")
            print0(f"[DEBUG] Input shapes - cond: {cond.shape}, timestep: {timesteps.shape}, text: {text_embeddings.shape}")
            raise
        
        # Apply conditioning mask to compute loss only on unconditioned regions
        # Note: pred is 16 channels (denoised video), but cond_mask is 36 channels (includes conditioning)
        # We only need the mask for the first 16 channels (the actual video latents)
        video_mask = (1 - cond_mask[:, :16])  # Only use first 16 channels of mask
        masked_pred = pred * video_mask
        masked_target = noise[:, :16] * video_mask  # Only use first 16 channels of noise
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(masked_pred, masked_target)
        
        print0(f"[DEBUG] Loss computation:")
        print0(f"[DEBUG]   pred shape: {pred.shape} (denoised video latents)")
        print0(f"[DEBUG]   noise shape: {noise.shape} (original noise)")
        print0(f"[DEBUG]   cond_mask shape: {cond_mask.shape} (conditioning mask)")
        print0(f"[DEBUG]   Using first 16 channels for loss computation")
        print0(f"[DEBUG]   video_mask coverage: {video_mask.mean():.3f} (1.0 = fully unconditioned)")
        print0(f"[DEBUG]   {active_transformer} loss: {loss.item():.6f}")
    
    # Validate loss
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(f"Invalid loss detected: {loss.item()}")
    
    if loss.item() == 0.0:
        print0(f"[WARNING] Zero loss detected - this might indicate a problem with conditioning masks")
    
    return loss

# Backward compatibility function - now uses single transformer training
def step_dual_transformer(
    pipe,
    model_map,
    transformer_names,
    batch: Dict,
    device,
    bf16,
    boundary_ratio: float,
    train_transformer_2: bool = False,
) -> torch.Tensor:
    """
    Backward compatibility wrapper that now implements single transformer training.
    The name is kept for compatibility but the implementation has changed.
    
    Args:
        train_transformer_2: Controls which transformer to train.
                           False = train 'transformer' (high timesteps)
                           True = train 'transformer_2' (low timesteps)
    """
    return step_single_transformer(
        pipe=pipe,
        model_map=model_map, 
        transformer_names=transformer_names,
        batch=batch,
        device=device,
        bf16=bf16,
        boundary_ratio=boundary_ratio,
        train_transformer_2=train_transformer_2
    )