# training_step_dmd_t2v.py - Fixed to use self-forcing pipeline

import os
from typing import Dict, Tuple

import torch
from dist_utils import print0


def step_dmd_generator(
    dmd_model,
    pipeline,  # DEPRECATED - no longer used, kept for compatibility
    batch,
    device,
    bf16,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    """
    DMD generator training step using self-forcing pipeline.
    
    KEY: The pipeline is now INSIDE dmd_model, not passed as argument.
    """
    debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

    # Extract batch data
    text_embeddings = batch["text_embeddings"].to(device, dtype=bf16)

    # Create negative (unconditional) embeddings
    negative_text_embeddings = torch.zeros_like(text_embeddings, device=device, dtype=bf16)

    # Handle tensor shapes
    while text_embeddings.ndim > 3:
        text_embeddings = text_embeddings.squeeze(0)
    if text_embeddings.ndim == 2:
        text_embeddings = text_embeddings.unsqueeze(0)

    while negative_text_embeddings.ndim > 3:
        negative_text_embeddings = negative_text_embeddings.squeeze(0)
    if negative_text_embeddings.ndim == 2:
        negative_text_embeddings = negative_text_embeddings.unsqueeze(0)

    batch_size = text_embeddings.shape[0]

    # Infer video shape
    channels = 16
    frames = 21
    height = 60
    width = 104
    image_or_video_shape = [batch_size, channels, frames, height, width]

    if debug_mode or (global_step % 100 == 0):
        print0(f"\n{'=' * 80}")
        print0(f"[GENERATOR STEP {global_step}] DMD Training with Self-Forcing")
        print0(f"{'=' * 80}")
        print0(f"[BATCH] Size: {batch_size}")
        print0(f"[VIDEO] Shape: {image_or_video_shape}")

    # Step 1: Run generator with self-forcing backward simulation
    # This internally calls the self-forcing pipeline
    generated_latent, gradient_mask, denoised_timestep_from, denoised_timestep_to = \
        dmd_model._run_generator(
            image_or_video_shape=image_or_video_shape,
            text_embeddings=text_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            initial_latent=None,
            clip_fea=None,
            y=None,
            global_step=global_step,
        )

    if debug_mode or (global_step % 100 == 0):
        print0("[SELF-FORCING] Complete")
        print0(f"  Generated shape: {generated_latent.shape}")
        print0(f"  Timestep range: [{denoised_timestep_to}, {denoised_timestep_from}]")
        print0(f"  Value range: [{generated_latent.min():.3f}, {generated_latent.max():.3f}]")

    # Step 2: Compute DMD loss
    dmd_loss, dmd_metrics = dmd_model.compute_distribution_matching_loss(
        generated_latent=generated_latent,
        text_embeddings=text_embeddings,
        negative_text_embeddings=negative_text_embeddings,
        gradient_mask=gradient_mask,
        denoised_timestep_from=denoised_timestep_from,
        denoised_timestep_to=denoised_timestep_to,
        global_step=global_step,
    )

    # Check for NaN or explosion
    if torch.isnan(dmd_loss) or dmd_loss > 100:
        print0(f"[ERROR] Generator loss explosion! Loss={dmd_loss.item():.3f}")
        print0(f"[DEBUG] Generated latent range: [{generated_latent.min():.3f}, {generated_latent.max():.3f}]")
        raise ValueError(f"Generator loss exploded: {dmd_loss.item()}")

    if debug_mode or (global_step % 100 == 0):
        print0(f"[GENERATOR LOSS] {dmd_loss.item():.6f}")
        print0(f"{'=' * 80}\n")

    # Compile metrics
    metrics = {
        "generator_loss": dmd_loss.item(),
        "dmd_gradient_norm": dmd_metrics.get("dmd_gradient_norm", 0.0),
        "generated_latent_min": generated_latent.min().item(),
        "generated_latent_max": generated_latent.max().item(),
        "generated_latent_mean": generated_latent.mean().item(),
    }

    return dmd_loss, metrics


def step_dmd_critic(
    dmd_model,
    pipeline,  # DEPRECATED - no longer used
    batch,
    device,
    bf16,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    """
    DMD critic training step using self-forcing pipeline.
    
    KEY: Samples are generated with torch.no_grad() to prevent feedback.
    """
    debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

    # Extract batch data
    text_embeddings = batch["text_embeddings"].to(device, dtype=bf16)

    # Handle tensor shapes
    while text_embeddings.ndim > 3:
        text_embeddings = text_embeddings.squeeze(0)
    if text_embeddings.ndim == 2:
        text_embeddings = text_embeddings.unsqueeze(0)

    batch_size = text_embeddings.shape[0]

    # Infer video shape
    channels = 16
    frames = 21
    height = 60
    width = 104
    image_or_video_shape = [batch_size, channels, frames, height, width]

    if debug_mode or (global_step % 100 == 0):
        print0(f"\n{'=' * 80}")
        print0(f"[CRITIC STEP {global_step}] DMD Training")
        print0(f"{'=' * 80}")
        print0(f"[BATCH] Size: {batch_size}")

    # Step 1: Generate samples (detached from generator)
    with torch.no_grad():
        generated_latent, _, denoised_timestep_from, denoised_timestep_to = \
            dmd_model._run_generator(
                image_or_video_shape=image_or_video_shape,
                text_embeddings=text_embeddings,
                negative_text_embeddings=None,
                initial_latent=None,
                clip_fea=None,
                y=None,
                global_step=global_step,
            )

    if debug_mode or (global_step % 100 == 0):
        print0("[SELF-FORCING] Generated samples (detached)")
        print0(f"  Shape: {generated_latent.shape}")
        print0(f"  Timestep range: [{denoised_timestep_to}, {denoised_timestep_from}]")

    # Step 2: Compute critic loss
    critic_loss, critic_metrics = dmd_model.compute_critic_loss(
        generated_latent=generated_latent,
        text_embeddings=text_embeddings,
        denoised_timestep_from=denoised_timestep_from,
        denoised_timestep_to=denoised_timestep_to,
        global_step=global_step,
    )

    # Check for issues
    explosion_threshold = 500.0 if global_step < 100 else 200.0

    if torch.isnan(critic_loss):
        print0("[ERROR] Critic loss is NaN!")
        print0(f"[DEBUG] Generated latent range: [{generated_latent.min():.3f}, {generated_latent.max():.3f}]")
        raise ValueError("Critic loss is NaN")

    if critic_loss > explosion_threshold:
        print0(f"[ERROR] Critic loss explosion! Loss={critic_loss.item():.3f}")
        print0(f"[DEBUG] Current threshold: {explosion_threshold}")
        
        if global_step >= 20:
            raise ValueError(f"Critic loss exploded: {critic_loss.item()}")
        else:
            print0(f"[WARNING] Allowing high loss during warmup (step {global_step}/20)")

    if debug_mode or (global_step % 100 == 0):
        print0(f"[CRITIC LOSS] {critic_loss.item():.6f}")
        print0(f"{'=' * 80}\n")

    # Compile metrics
    metrics = {
        "critic_loss": critic_loss.item(),
    }
    metrics.update(critic_metrics)

    return critic_loss, metrics


def step_dmd_alternating(
    dmd_model,
    pipeline,  # DEPRECATED - kept for compatibility
    batch,
    device,
    bf16,
    global_step: int = 0,
    update_generator: bool = True,
    update_critic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Combined DMD training step with optional alternating updates.
    
    NOTE: Pipeline argument is deprecated but kept for backward compatibility.
    The self-forcing pipeline is now internal to dmd_model.
    """
    metrics = {}
    generator_loss = None
    critic_loss = None

    if update_generator:
        generator_loss, gen_metrics = step_dmd_generator(
            dmd_model=dmd_model,
            pipeline=None,  # No longer used
            batch=batch,
            device=device,
            bf16=bf16,
            global_step=global_step,
        )
        metrics.update(gen_metrics)

    if update_critic:
        critic_loss, crit_metrics = step_dmd_critic(
            dmd_model=dmd_model,
            pipeline=None,  # No longer used
            batch=batch,
            device=device,
            bf16=bf16,
            global_step=global_step,
        )
        metrics.update(crit_metrics)

    return generator_loss, critic_loss, metrics