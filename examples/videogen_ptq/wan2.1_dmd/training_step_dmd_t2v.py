# training_step_dmd_t2v.py - DMD Training Step with Alternating Optimization (FIXED)

import os
from typing import Dict, Tuple

import torch
from dist_utils import print0


def step_dmd_generator(
    dmd_model,
    pipeline,
    batch,
    device,
    bf16,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    """
    DMD generator training step.

    This step:
    1. Runs backward simulation to generate synthetic samples
    2. Computes DMD loss to match real/fake distributions

    Args:
        dmd_model: DMDT2V instance
        pipeline: Backward simulation pipeline
        batch: Training batch
        device: Device
        bf16: BFloat16 dtype
        global_step: Current step

    Returns:
        loss: Generator loss
        metrics: Logging metrics
    """
    debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

    # Extract batch data
    text_embeddings = batch["text_embeddings"].to(device, dtype=bf16)

    # Create negative (unconditional) embeddings - zeros with same shape and device
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

    # Infer video shape from batch or use defaults
    # WAN 2.1 T2V: 16 channels, varying frames/height/width
    if "video_latents" in batch:
        video_latents = batch["video_latents"]
        while video_latents.ndim > 5:
            video_latents = video_latents.squeeze(0)
        if video_latents.ndim == 4:
            video_latents = video_latents.unsqueeze(0)

        _, channels, frames, height, width = video_latents.shape
    else:
        # Default shape for T2V
        channels = 16
        frames = 21  # 5 seconds at 24fps, downsampled
        height = 60  # 480p latent space
        width = 104  # 832p latent space

    if debug_mode or (global_step % 100 == 0):
        print0(f"\n{'=' * 80}")
        print0(f"[GENERATOR STEP {global_step}] DMD Training")
        print0(f"{'=' * 80}")
        print0(f"[BATCH] Size: {batch_size}")
        print0(f"[VIDEO] Shape: [{batch_size}, {channels}, {frames}, {height}, {width}]")

    # Step 1: Backward simulation - generate synthetic samples
    noise = torch.randn(batch_size, channels, frames, height, width, device=device, dtype=bf16)

    if debug_mode or (global_step % 100 == 0):
        print0("[BACKWARD SIM] Starting from noise...")

    # Run backward simulation through generator
    generated_latent, denoised_timestep_from, denoised_timestep_to = pipeline.inference_with_trajectory(
        noise=noise,
        text_embeddings=text_embeddings,
    )

    if debug_mode or (global_step % 100 == 0):
        print0("[BACKWARD SIM] Complete")
        print0(f"  Generated shape: {generated_latent.shape}")
        print0(f"  Timestep range: [{denoised_timestep_to}, {denoised_timestep_from}]")
        print0(f"  Value range: [{generated_latent.min():.3f}, {generated_latent.max():.3f}]")

    # Step 2: Compute DMD loss
    gradient_mask = None  # Can be used to mask certain frames

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
    pipeline,
    batch,
    device,
    bf16,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    """
    DMD critic training step.

    This step:
    1. Runs backward simulation to generate synthetic samples (detached)
    2. Trains critic to denoise generated samples

    STABILITY IMPROVEMENTS:
    - More lenient explosion threshold (500 instead of 100)
    - Better error messages
    - Additional diagnostics

    Args:
        dmd_model: DMDT2V instance
        pipeline: Backward simulation pipeline
        batch: Training batch
        device: Device
        bf16: BFloat16 dtype
        global_step: Current step

    Returns:
        loss: Critic loss
        metrics: Logging metrics
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
    if "video_latents" in batch:
        video_latents = batch["video_latents"]
        while video_latents.ndim > 5:
            video_latents = video_latents.squeeze(0)
        if video_latents.ndim == 4:
            video_latents = video_latents.unsqueeze(0)

        _, channels, frames, height, width = video_latents.shape
    else:
        channels = 16
        frames = 21
        height = 60
        width = 104

    if debug_mode or (global_step % 100 == 0):
        print0(f"\n{'=' * 80}")
        print0(f"[CRITIC STEP {global_step}] DMD Training")
        print0(f"{'=' * 80}")
        print0(f"[BATCH] Size: {batch_size}")

    # Step 1: Generate samples (detached from generator)
    noise = torch.randn(batch_size, channels, frames, height, width, device=device, dtype=bf16)

    with torch.no_grad():
        generated_latent, denoised_timestep_from, denoised_timestep_to = pipeline.inference_with_trajectory(
            noise=noise,
            text_embeddings=text_embeddings,
        )

    if debug_mode or (global_step % 100 == 0):
        print0("[BACKWARD SIM] Generated samples (detached)")
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

    # IMPROVED: More lenient explosion threshold and better diagnostics
    # Early in training, critic loss can be high but will stabilize
    explosion_threshold = 500.0 if global_step < 100 else 200.0

    if torch.isnan(critic_loss):
        print0("[ERROR] Critic loss is NaN!")
        print0("[DEBUG] Generated latent stats:")
        print0(f"  - Range: [{generated_latent.min():.3f}, {generated_latent.max():.3f}]")
        print0(f"  - Mean: {generated_latent.mean():.3f}")
        print0(f"  - Std: {generated_latent.std():.3f}")
        raise ValueError("Critic loss is NaN")

    if critic_loss > explosion_threshold:
        print0(f"[ERROR] Critic loss explosion! Loss={critic_loss.item():.3f}")
        print0("[DEBUG] This may be normal early in training")
        print0(f"[DEBUG] Current threshold: {explosion_threshold}")
        print0("[DEBUG] Consider:")
        print0("  1. Reducing critic learning rate (try 1e-6 or 5e-6)")
        print0("  2. Using warmup scheduler")
        print0("  3. Checking if loss stabilizes after ~10 steps")

        # Don't crash immediately - allow a few high loss steps early on
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
    pipeline,
    batch,
    device,
    bf16,
    global_step: int = 0,
    update_generator: bool = True,
    update_critic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Combined DMD training step with optional alternating updates.

    This is a convenience function that can run both generator and critic
    updates in a single step, or alternate between them.

    Args:
        dmd_model: DMDT2V instance
        pipeline: Backward simulation pipeline
        batch: Training batch
        device: Device
        bf16: BFloat16 dtype
        global_step: Current step
        update_generator: Whether to update generator this step
        update_critic: Whether to update critic this step

    Returns:
        generator_loss: Generator loss (or None)
        critic_loss: Critic loss (or None)
        metrics: Combined metrics
    """
    metrics = {}
    generator_loss = None
    critic_loss = None

    if update_generator:
        generator_loss, gen_metrics = step_dmd_generator(
            dmd_model=dmd_model,
            pipeline=pipeline,
            batch=batch,
            device=device,
            bf16=bf16,
            global_step=global_step,
        )
        metrics.update(gen_metrics)

    if update_critic:
        critic_loss, crit_metrics = step_dmd_critic(
            dmd_model=dmd_model,
            pipeline=pipeline,
            batch=batch,
            device=device,
            bf16=bf16,
            global_step=global_step,
        )
        metrics.update(crit_metrics)

    return generator_loss, critic_loss, metrics
