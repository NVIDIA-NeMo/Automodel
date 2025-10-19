# training_step_dmd_t2v.py - Pure DMD Training Step

import os
from typing import Dict, Tuple

import torch
from dist_utils import print0


def step_dmd_student(
    dmd_model,
    pipeline,
    batch,
    device,
    bf16,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    """
    DMD student training step (pure DMD, no self-forcing).
    
    Process:
    1. Sample random timestep
    2. Student predicts x0 from noise
    3. Create x_t from student's x0
    4. Get teacher and critic predictions at x_t (detached)
    5. Compute DMD loss using (critic - teacher) direction
    
    Args:
        dmd_model: DMD model
        pipeline: Simple DMD pipeline
        batch: Data batch
        device: Device
        bf16: BFloat16 dtype
        global_step: Current training step
        
    Returns:
        student_loss: Scalar loss
        metrics: Dict of metrics
    """
    debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"
    
    # Extract batch data
    text_embeddings = batch["text_embeddings"].to(device, dtype=bf16)
    
    # Create negative embeddings
    negative_text_embeddings = torch.zeros_like(text_embeddings)
    
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
    
    # Define video shape
    channels = 16
    frames = 21
    height = 60
    width = 104
    
    if debug_mode or (global_step % 100 == 0):
        print0(f"\n{'=' * 80}")
        print0(f"[STUDENT STEP {global_step}] Pure DMD Training")
        print0(f"{'=' * 80}")
        print0(f"[BATCH] Size: {batch_size}")
        print0(f"[VIDEO] Shape: [{batch_size}, {channels}, {frames}, {height}, {width}]")
    
    # Step 1: Sample random timestep
    timesteps = dmd_model.sample_timesteps(batch_size)
    
    if debug_mode or (global_step % 100 == 0):
        print0(f"[TIMESTEPS] {timesteps.tolist()}")
    
    # Step 2: Create pure noise
    noise = torch.randn(
        batch_size, channels, frames, height, width,
        device=device,
        dtype=bf16,
    )
    
    # Step 3: Student predicts x0 from noise, then create x_t
    x_t, student_x0 = pipeline.forward_one_step(
        noise=noise,
        timestep=timesteps,
        text_embeddings=text_embeddings,
    )
    
    if debug_mode or (global_step % 100 == 0):
        print0(f"[STUDENT] x0 range: [{student_x0.min():.3f}, {student_x0.max():.3f}]")
        print0(f"[NOISY] x_t range: [{x_t.min():.3f}, {x_t.max():.3f}]")
    
    # Step 4: Compute DMD loss
    student_loss, metrics = dmd_model.compute_student_dmd_loss(
        student_x0=student_x0,
        x_t=x_t,
        timesteps=timesteps,
        text_embeddings=text_embeddings,
        negative_text_embeddings=negative_text_embeddings,
        global_step=global_step,
    )
    
    if debug_mode or (global_step % 100 == 0):
        print0(f"[STUDENT LOSS] {student_loss.item():.6f}")
        print0(f"{'=' * 80}\n")
    
    return student_loss, metrics


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
    
    Process:
    1. Sample random timestep
    2. Student predicts x0 from noise (DETACHED)
    3. Create x_t from student's x0
    4. Critic learns to match teacher at x_t
    
    Args:
        dmd_model: DMD model
        pipeline: Simple DMD pipeline
        batch: Data batch
        device: Device
        bf16: BFloat16 dtype
        global_step: Current training step
        
    Returns:
        critic_loss: Scalar loss
        metrics: Dict of metrics
    """
    debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"
    
    # Extract batch data
    text_embeddings = batch["text_embeddings"].to(device, dtype=bf16)
    
    # Create negative embeddings
    negative_text_embeddings = torch.zeros_like(text_embeddings)
    
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
    
    # Define video shape
    channels = 16
    frames = 21
    height = 60
    width = 104
    
    if debug_mode or (global_step % 100 == 0):
        print0(f"\n{'=' * 80}")
        print0(f"[CRITIC STEP {global_step}] Pure DMD Training")
        print0(f"{'=' * 80}")
        print0(f"[BATCH] Size: {batch_size}")
    
    # Step 1: Sample random timestep
    timesteps = dmd_model.sample_timesteps(batch_size)
    
    # Step 2: Generate x_t from student (DETACHED)
    with torch.no_grad():
        noise = torch.randn(
            batch_size, channels, frames, height, width,
            device=device,
            dtype=bf16,
        )
        
        x_t, _ = pipeline.forward_one_step(
            noise=noise,
            timestep=timesteps,
            text_embeddings=text_embeddings,
        )
    
    if debug_mode or (global_step % 100 == 0):
        print0(f"[NOISY] x_t range: [{x_t.min():.3f}, {x_t.max():.3f}]")
    
    # Step 3: Train critic to match teacher
    critic_loss, metrics = dmd_model.compute_critic_loss(
        x_t=x_t,
        timesteps=timesteps,
        text_embeddings=text_embeddings,
        negative_text_embeddings=negative_text_embeddings,
        global_step=global_step,
    )
    
    if debug_mode or (global_step % 100 == 0):
        print0(f"[CRITIC LOSS] {critic_loss.item():.6f}")
        print0(f"{'=' * 80}\n")
    
    return critic_loss, metrics


def step_dmd_alternating(
    dmd_model,
    pipeline,
    batch,
    device,
    bf16,
    global_step: int = 0,
    update_student: bool = True,
    update_critic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Combined DMD training step with alternating updates.
    
    Args:
        dmd_model: DMD model
        pipeline: Simple DMD pipeline
        batch: Data batch
        device: Device
        bf16: BFloat16 dtype
        global_step: Current training step
        update_student: Whether to update student
        update_critic: Whether to update critic
        
    Returns:
        student_loss: Student loss (or None)
        critic_loss: Critic loss (or None)
        metrics: Combined metrics dict
    """
    metrics = {}
    student_loss = None
    critic_loss = None
    
    if update_student:
        student_loss, student_metrics = step_dmd_student(
            dmd_model=dmd_model,
            pipeline=pipeline,
            batch=batch,
            device=device,
            bf16=bf16,
            global_step=global_step,
        )
        metrics.update(student_metrics)
    
    if update_critic:
        critic_loss, critic_metrics = step_dmd_critic(
            dmd_model=dmd_model,
            pipeline=pipeline,
            batch=batch,
            device=device,
            bf16=bf16,
            global_step=global_step,
        )
        metrics.update(critic_metrics)
    
    return student_loss, critic_loss, metrics