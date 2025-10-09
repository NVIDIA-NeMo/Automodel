# training_step_t2v.py - Training step for WAN 2.1 T2V with FSDP
import os
from typing import Dict

import torch
from conditioning_t2v import prepare_t2v_conditioning
from dist_utils import print0


def step_fsdp_transformer_t2v(
    pipe,
    model_map: Dict,
    batch,
    device,
    bf16,
):
    """
    Training step for FSDP wrapped WAN 2.1 T2V transformer.
    
    Simpler than I2V - just text conditioning, no image conditioning.
    """
    # Only print debug info if DEBUG_TRAINING env var is set
    debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

    # Extract and prepare batch data
    video_latents = batch["video_latents"].to(device, dtype=bf16)
    text_embeddings = batch["text_embeddings"].to(device, dtype=bf16)

    # Handle tensor shapes
    if debug_mode:
        print0(f"[FSDP STEP] Raw video_latents shape: {video_latents.shape}")
        print0(f"[FSDP STEP] Raw text_embeddings shape: {text_embeddings.shape}")

    # Squeeze extra dimensions if needed
    if video_latents.ndim == 6:
        video_latents = video_latents.squeeze(1)
        if debug_mode:
            print0(f"[FSDP STEP] Squeezed video_latents to: {video_latents.shape}")
    elif video_latents.ndim != 5:
        raise ValueError(f"Unexpected video_latents shape: {video_latents.shape}")

    if text_embeddings.ndim == 4:
        text_embeddings = text_embeddings.squeeze(1)
        if debug_mode:
            print0(f"[FSDP STEP] Squeezed text_embeddings to: {text_embeddings.shape}")
    elif text_embeddings.ndim != 3:
        raise ValueError(f"Unexpected text_embeddings shape: {text_embeddings.shape}")

    batch_size, channels, frames, height, width = video_latents.shape

    if debug_mode:
        print0(f"[FSDP STEP] Processing batch: {video_latents.shape}")

    # Generate timesteps
    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device=device)

    # Prepare T2V conditioning (simpler than I2V - just noisy latents)
    if debug_mode:
        print0("[FSDP STEP] Preparing T2V conditioning...")
    
    noisy_latents, noise = prepare_t2v_conditioning(pipe, video_latents, timesteps, bf16)

    if debug_mode:
        print0(f"[FSDP STEP] Noisy latents shape: {noisy_latents.shape}")

    # Verify channel count (16 for T2V)
    expected_channels = 16
    if noisy_latents.shape[1] != expected_channels:
        raise RuntimeError(f"Expected {expected_channels} channels for T2V, got {noisy_latents.shape[1]}")

    # Get FSDP wrapped model
    fsdp_model = model_map["transformer"]["fsdp_transformer"]

    if debug_mode:
        print0("[FSDP STEP] Running forward pass...")

    try:
        # Forward pass - FSDP handles all parallelism automatically
        model_pred = fsdp_model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )

        if isinstance(model_pred, tuple):
            model_pred = model_pred[0]

        if debug_mode:
            print0(f"[FSDP STEP] Forward successful: {model_pred.shape}")

    except Exception as e:
        print0(f"[ERROR] Forward pass failed: {e}")
        print0(f"[DEBUG] Input shape: {noisy_latents.shape}")
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
        if debug_mode:
            print0("[FSDP STEP] Using flow_prediction target")
    else:
        raise ValueError(f"Unknown prediction type: {pipe.scheduler.config.prediction_type}")

    # Compute loss
    loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")

    if debug_mode:
        print0(f"[FSDP STEP] Loss: {loss.item():.6f}")

    return loss