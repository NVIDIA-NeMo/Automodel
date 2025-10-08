# training_step_fsdp2.py - Training step for FSDP2
import torch
from typing import Dict, List
from dist_utils import print0
from conditioning import prepare_i2v_conditioning


def step_fsdp2_transformer(
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
    Training step for FSDP2 wrapped transformer.
    
    Key differences from TP+CP:
    - No manual sequence sharding (FSDP2 handles parallelism)
    - No RoPE-aware positioning needed
    - Simpler gradient handling (FSDP2 does it automatically)
    """
    
    # Extract and prepare batch data
    video_latents = batch["video_latents"].to(device, dtype=bf16)
    text_embeddings = batch["text_embeddings"].to(device, dtype=bf16)
    
    # Handle tensor shapes
    print0(f"[FSDP2 STEP] Raw video_latents shape: {video_latents.shape}")
    print0(f"[FSDP2 STEP] Raw text_embeddings shape: {text_embeddings.shape}")
    
    # Squeeze extra dimensions if needed
    if video_latents.ndim == 6:
        video_latents = video_latents.squeeze(1)
        print0(f"[FSDP2 STEP] Squeezed video_latents to: {video_latents.shape}")
    elif video_latents.ndim != 5:
        raise ValueError(f"Unexpected video_latents shape: {video_latents.shape}")
    
    if text_embeddings.ndim == 4:
        text_embeddings = text_embeddings.squeeze(1)
        print0(f"[FSDP2 STEP] Squeezed text_embeddings to: {text_embeddings.shape}")
    elif text_embeddings.ndim != 3:
        raise ValueError(f"Unexpected text_embeddings shape: {text_embeddings.shape}")
    
    batch_size, channels, frames, height, width = video_latents.shape
    
    # Determine active transformer
    active_name = "transformer_2" if train_transformer_2 else "transformer"
    
    if active_name not in model_map:
        raise RuntimeError(f"Active transformer {active_name} not found")
    
    print0(f"[FSDP2 STEP] Using {active_name}: {video_latents.shape}")
    
    # Generate timesteps
    timesteps = torch.randint(
        0, pipe.scheduler.config.num_train_timesteps,
        (batch_size,), device=device
    )
    
    # Prepare I2V conditioning
    print0(f"[FSDP2 STEP] Preparing I2V conditioning...")
    conditioned_latents, noise, condition_mask = prepare_i2v_conditioning(
        pipe, video_latents, timesteps, bf16
    )
    
    print0(f"[FSDP2 STEP] Conditioned latents shape: {conditioned_latents.shape}")
    
    # Verify channel count
    expected_channels = 36  # For I2V
    if conditioned_latents.shape[1] != expected_channels:
        print0(f"[WARNING] Expected {expected_channels} channels, got {conditioned_latents.shape[1]}")
    
    # Handle dual transformer timestep selection
    if boundary_ratio > 0.0 and len(transformer_names) > 1:
        boundary_timestep = int(pipe.scheduler.config.num_train_timesteps * boundary_ratio)
        
        if train_transformer_2:
            # transformer_2 handles low timesteps
            valid_timesteps = timesteps < boundary_timestep
            if not valid_timesteps.any():
                print0("[INFO] No low timesteps, skipping transformer_2")
                return torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # transformer handles high timesteps
            valid_timesteps = timesteps >= boundary_timestep
            if not valid_timesteps.any():
                print0("[INFO] No high timesteps, skipping transformer")
                return torch.tensor(0.0, device=device, requires_grad=True)
        
        print0(f"[FSDP2 STEP] Processing {valid_timesteps.sum().item()}/{batch_size} samples")
    else:
        valid_timesteps = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    # Filter batch
    if not valid_timesteps.all():
        conditioned_latents = conditioned_latents[valid_timesteps]
        timesteps = timesteps[valid_timesteps]
        text_embeddings = text_embeddings[valid_timesteps]
        noise = noise[valid_timesteps]
        
        if conditioned_latents.shape[0] == 0:
            print0("[INFO] No valid samples after filtering")
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Get FSDP wrapped model
    fsdp_model = model_map[active_name]["fsdp_transformer"]
    
    print0(f"[FSDP2 STEP] Running forward pass...")
    
    try:
        # Forward pass - FSDP2 handles all parallelism automatically
        model_pred = fsdp_model(
            hidden_states=conditioned_latents,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )
        
        if isinstance(model_pred, tuple):
            model_pred = model_pred[0]
        
        print0(f"[FSDP2 STEP] Forward successful: {model_pred.shape}")
        
    except Exception as e:
        print0(f"[ERROR] Forward pass failed: {e}")
        print0(f"[DEBUG] Input shape: {conditioned_latents.shape}")
        print0(f"[DEBUG] Timestep shape: {timesteps.shape}")
        print0(f"[DEBUG] Text embedding shape: {text_embeddings.shape}")
        raise
    
    # Compute target based on prediction type
    if pipe.scheduler.config.prediction_type == "epsilon":
        target = noise
    elif pipe.scheduler.config.prediction_type == "v_prediction":
        target = pipe.scheduler.get_velocity(video_latents, noise, timesteps)
    elif pipe.scheduler.config.prediction_type == "flow_prediction":
        # Flow matching
        target = video_latents - noise
        print0(f"[FSDP2 STEP] Using flow_prediction target")
    else:
        raise ValueError(f"Unknown prediction type: {pipe.scheduler.config.prediction_type}")
    
    # Apply filtering to target if needed
    if not valid_timesteps.all():
        target = target[valid_timesteps]
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
    
    print0(f"[FSDP2 STEP] Loss: {loss.item():.6f}")
    
    return loss