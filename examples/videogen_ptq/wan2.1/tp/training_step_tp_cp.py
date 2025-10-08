# training_step_tp_cp.py - Fixed for correct tensor shapes and flow_prediction
import torch
import torch.distributed as dist
from typing import Dict, List
from dist_utils import print0
from conditioning import prepare_i2v_conditioning

def truncate_sequence_for_memory(video_latents, text_embeddings, max_frames=3):
    """Truncate sequences to prevent OOM when CP is disabled."""
    batch_size, channels, frames, height, width = video_latents.shape
    
    if frames > max_frames:
        print0(f"[MEMORY] Truncating sequence from {frames} to {max_frames} frames")
        video_latents = video_latents[:, :, :max_frames, :, :]
        # Text embeddings usually don't need truncation, but check
        if text_embeddings.shape[1] > max_frames and text_embeddings.shape[1] == frames:
            text_embeddings = text_embeddings[:, :max_frames, :]
    
    return video_latents, text_embeddings

def step_dual_transformer(
    pipe,
    model_map: Dict[str, Dict],
    transformer_names: List[str],
    batch,
    device,
    bf16,
    boundary_ratio: float = 0.0,
    train_transformer_2: bool = False,
):
    """
    Enhanced training step that works with RoPE-aware TP+CP.
    Fixed to handle correct tensor shapes from dataloader.
    """
    # Extract batch data
    video_latents = batch["video_latents"].to(device, dtype=bf16)
    text_embeddings = batch["text_embeddings"].to(device, dtype=bf16)
    
    # Handle tensor shapes correctly
    print0(f"[STEP DEBUG] Raw video_latents shape: {video_latents.shape}")
    print0(f"[STEP DEBUG] Raw text_embeddings shape: {text_embeddings.shape}")
    
    # Check if we need to squeeze extra dimensions
    if video_latents.ndim == 5:
        # Shape: [batch, channels, frames, height, width] - this is correct
        batch_size, channels, frames, height, width = video_latents.shape
    elif video_latents.ndim == 6:
        # Shape: [batch, 1, channels, frames, height, width] - squeeze extra dim
        video_latents = video_latents.squeeze(1)
        batch_size, channels, frames, height, width = video_latents.shape
        print0(f"[STEP DEBUG] Squeezed video_latents to: {video_latents.shape}")
    else:
        raise ValueError(f"Unexpected video_latents shape: {video_latents.shape}. Expected 5D [B,C,T,H,W] or 6D [B,1,C,T,H,W]")
    
    # Handle text embeddings similarly
    if text_embeddings.ndim == 4:
        # Shape: [batch, 1, seq_len, hidden] - squeeze middle dim
        text_embeddings = text_embeddings.squeeze(1)
        print0(f"[STEP DEBUG] Squeezed text_embeddings to: {text_embeddings.shape}")
    elif text_embeddings.ndim != 3:
        raise ValueError(f"Unexpected text_embeddings shape: {text_embeddings.shape}. Expected 3D [B,S,H]")
    
    # Enhanced debugging for RoPE-CP
    active_transformer_name = "transformer_2" if train_transformer_2 else "transformer"

    # Truncate the squence
    # if not model_map[active_transformer_name].get("use_rope_cp", False):
        # No CP enabled, truncate to prevent OOM
    video_latents, text_embeddings = truncate_sequence_for_memory(video_latents, text_embeddings, max_frames=2)
    print0(f"[MEMORY] Sequence truncated for TP-only mode: {video_latents.shape}")
    # Update frame count after truncation
    batch_size, channels, frames, height, width = video_latents.shape

    
    if active_transformer_name in model_map:
        cp_info = ""
        if model_map[active_transformer_name].get("use_rope_cp", False):
            cp_manager = model_map[active_transformer_name]["cp_manager"]
            min_frames_needed = model_map[active_transformer_name]["min_temporal_length"] * model_map[active_transformer_name]["cp_size"]
            cp_info = f" (RoPE-CP: needâ‰¥{min_frames_needed}, rope_dim={getattr(cp_manager, 'rope_dim', 'unknown')})"
        
        print0(f"[STEP] Using {active_transformer_name}: {video_latents.shape}{cp_info}")
    
    # Validate frame count for RoPE-CP
    if active_transformer_name in model_map and model_map[active_transformer_name].get("use_rope_cp", False):
        min_frames_needed = model_map[active_transformer_name]["min_temporal_length"] * model_map[active_transformer_name]["cp_size"]
        if frames < min_frames_needed:
            print0(f"[WARNING] Input has {frames} frames but RoPE-CP needs â‰¥{min_frames_needed}")
            print0("[INFO] Sequence may be too short for optimal RoPE positioning")
    
    # Generate random timesteps
    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device=device)
    
    # Prepare I2V conditioning with enhanced logging
    print0(f"[STEP] Preparing I2V conditioning for {frames} frames...")
    conditioned_latents, noise, condition_mask = prepare_i2v_conditioning(
        pipe, video_latents, timesteps, bf16
    )
    
    # Verify conditioning output
    print0(f"[STEP] Conditioned latents shape: {conditioned_latents.shape}")
    print0(f"[STEP] Expected channels for transformer: checking...")
    
    # Get the active transformer
    if active_transformer_name not in model_map:
        raise RuntimeError(f"Active transformer {active_transformer_name} not found in model_map")
    
    active_transformer = model_map[active_transformer_name]["tp_cp_transformer"]
    
    # Check expected input channels
    expected_channels = None
    try:
        # Handle both wrapped and unwrapped transformers
        base_transformer = active_transformer
        if hasattr(active_transformer, 'transformer'):
            # This is a wrapped transformer (RoPE-aware)
            base_transformer = active_transformer.transformer
        
        if hasattr(base_transformer, 'patch_embedding') and hasattr(base_transformer.patch_embedding, 'weight'):
            expected_channels = base_transformer.patch_embedding.weight.shape[1]
            print0(f"[STEP] {active_transformer_name} expects {expected_channels} input channels")
        
        if expected_channels and conditioned_latents.shape[1] != expected_channels:
            raise RuntimeError(f"Channel mismatch: conditioned={conditioned_latents.shape[1]}, expected={expected_channels}")
    
    except Exception as e:
        print0(f"[WARNING] Could not verify input channels: {e}")
    
    # Enhanced timestep selection for dual transformer setup
    if boundary_ratio > 0.0 and len(transformer_names) > 1:
        # Dual transformer: select based on timestep boundary
        boundary_timestep = int(pipe.scheduler.config.num_train_timesteps * boundary_ratio)
        
        if train_transformer_2:
            # transformer_2 handles low timesteps
            valid_timesteps = timesteps < boundary_timestep
            if not valid_timesteps.any():
                print0("[INFO] No low timesteps in batch, skipping transformer_2 step")
                return torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # transformer handles high timesteps  
            valid_timesteps = timesteps >= boundary_timestep
            if not valid_timesteps.any():
                print0("[INFO] No high timesteps in batch, skipping transformer step")
                return torch.tensor(0.0, device=device, requires_grad=True)
        
        print0(f"[STEP] {active_transformer_name} processing {valid_timesteps.sum().item()}/{batch_size} samples")
    else:
        # Single transformer or no boundary
        valid_timesteps = torch.ones(batch_size, dtype=torch.bool, device=device)
        print0(f"[STEP] {active_transformer_name} processing all {batch_size} samples")
    
    # Filter batch for valid timesteps
    if not valid_timesteps.all():
        conditioned_latents = conditioned_latents[valid_timesteps]
        timesteps = timesteps[valid_timesteps]
        text_embeddings = text_embeddings[valid_timesteps]
        noise = noise[valid_timesteps]
        
        if conditioned_latents.shape[0] == 0:
            print0("[INFO] No valid samples after timestep filtering")
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    print0(f"[STEP] Final batch size for {active_transformer_name}: {conditioned_latents.shape[0]}")
    
    # RoPE-aware forward pass
    print0(f"[STEP] Running forward pass through {active_transformer_name}...")
    
    try:
        # The RoPE-aware wrapper handles sequence sharding and gathering automatically
        model_pred = active_transformer(
            hidden_states=conditioned_latents,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )
        
        if isinstance(model_pred, tuple):
            model_pred = model_pred[0]
        
        print0(f"[STEP] Forward pass successful: {model_pred.shape}")
        
        # Validate RoPE positioning worked correctly
        if model_map[active_transformer_name].get("use_rope_cp", False):
            cp_manager = model_map[active_transformer_name]["cp_manager"]
            if hasattr(cp_manager, '_last_shard_meta') and cp_manager._last_shard_meta:
                was_sharded = cp_manager._last_shard_meta.get("was_sharded", False)
                if was_sharded:
                    print0("[STEP] RoPE-aware sequence sharding and gathering completed successfully")
                else:
                    print0("[STEP] Sequence too short for sharding, processed without CP")
        
    except Exception as e:
        print0(f"[ERROR] Forward pass failed: {e}")
        # Enhanced debugging
        print0(f"[DEBUG] Conditioned input shape: {conditioned_latents.shape}")
        print0(f"[DEBUG] Timestep shape: {timesteps.shape}")
        print0(f"[DEBUG] Text embedding shape: {text_embeddings.shape}")
        print0(f"[DEBUG] Expected input format: [B, C, T, H, W] = [batch, channels, frames, height, width]")
        raise
    
    # Compute loss based on prediction type - FIXED TO SUPPORT FLOW_PREDICTION
    if pipe.scheduler.config.prediction_type == "epsilon":
        target = noise
    elif pipe.scheduler.config.prediction_type == "v_prediction":
        target = pipe.scheduler.get_velocity(video_latents, noise, timesteps)
    elif pipe.scheduler.config.prediction_type == "flow_prediction":
        # Flow matching prediction - predict the flow field (difference)
        target = video_latents - noise
        print0(f"[STEP] Using flow_prediction target shape: {target.shape}")
    else:
        raise ValueError(f"Unknown prediction type: {pipe.scheduler.config.prediction_type}")
    
    # Apply loss only to valid timesteps if needed
    if not valid_timesteps.all():
        target = target[valid_timesteps]
    
    # Compute MSE loss
    loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
    
    # Handle CP loss reduction if needed
    if model_map[active_transformer_name].get("use_rope_cp", False):
        cp_manager = model_map[active_transformer_name]["cp_manager"]
        if cp_manager and hasattr(cp_manager, 'reduce_loss'):
            loss = cp_manager.reduce_loss(loss)
            print0(f"[STEP] Loss reduced across CP group: {loss.item():.6f}")
    
    print0(f"[STEP] Final loss: {loss.item():.6f}")
    
    return loss


# Backwards compatibility
def step_single_transformer(pipe, model_map, transformer_names, batch, device, bf16, train_transformer_2=False):
    """Legacy function - redirects to enhanced dual transformer step."""
    return step_dual_transformer(
        pipe, model_map, transformer_names, batch, device, bf16, 
        boundary_ratio=0.0, train_transformer_2=train_transformer_2
    )