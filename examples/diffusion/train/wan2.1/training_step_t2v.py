# training_step_t2v.py - Manual Flow Matching (DO NOT use scheduler.add_noise)

import os
from typing import Dict, Tuple

import torch
from dist_utils import print0
from time_shift_utils import compute_density_for_timestep_sampling


def step_fsdp_transformer_t2v(
    pipe,
    model_map: Dict,
    batch,
    device,
    bf16,
    # Flow matching parameters
    use_sigma_noise: bool = True,
    timestep_sampling: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    flow_shift: float = 3.0,
    mix_uniform_ratio: float = 0.1,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Pure flow matching training - DO NOT use scheduler.add_noise().
    
    The scheduler's add_noise() uses alpha_t/sigma_t which explodes at low timesteps.
    We use simple flow matching: x_t = (1-σ)x_0 + σ*ε
    """
    debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"
    detailed_log = (global_step % 100 == 0)
    summary_log = (global_step % 10 == 0)

    # Extract and prepare batch data
    video_latents = batch["video_latents"].to(device, dtype=bf16)
    text_embeddings = batch["text_embeddings"].to(device, dtype=bf16)

    # Handle tensor shapes
    while video_latents.ndim > 5:
        video_latents = video_latents.squeeze(0)
    if video_latents.ndim == 4:
        video_latents = video_latents.unsqueeze(0)

    while text_embeddings.ndim > 3:
        text_embeddings = text_embeddings.squeeze(0)
    if text_embeddings.ndim == 2:
        text_embeddings = text_embeddings.unsqueeze(0)

    batch_size = video_latents.shape[0]
    _, channels, frames, height, width = video_latents.shape

    # ========================================================================
    # Flow Matching Timestep Sampling
    # ========================================================================
    
    num_train_timesteps = pipe.scheduler.config.num_train_timesteps
    
    if use_sigma_noise:
        use_uniform = torch.rand(1).item() < mix_uniform_ratio
        
        if use_uniform or timestep_sampling == "uniform":
            # Pure uniform: u ~ U(0, 1)
            u = torch.rand(size=(batch_size,), device=device)
            sampling_method = "uniform"
        else:
            # Density-based sampling
            u = compute_density_for_timestep_sampling(
                weighting_scheme=timestep_sampling,
                batch_size=batch_size,
                logit_mean=logit_mean,
                logit_std=logit_std,
            ).to(device)
            sampling_method = timestep_sampling
        
        # Apply flow shift: σ = shift/(shift + (1/u - 1))
        u_clamped = torch.clamp(u, min=1e-5)  # Avoid division by zero
        sigma = flow_shift / (flow_shift + (1.0 / u_clamped - 1.0))
        sigma = torch.clamp(sigma, 0.0, 1.0)
        
    else:
        # Simple uniform without shift
        u = torch.rand(size=(batch_size,), device=device)
        sigma = u
        sampling_method = "uniform_no_shift"

    # ========================================================================
    # Manual Flow Matching Noise Addition
    # ========================================================================
    
    # Generate noise
    noise = torch.randn_like(video_latents, dtype=torch.float32)
    
    # CRITICAL: Manual flow matching (NOT scheduler.add_noise!)
    # x_t = (1 - σ) * x_0 + σ * ε
    sigma_reshaped = sigma.view(-1, 1, 1, 1, 1)
    noisy_latents = (
        (1.0 - sigma_reshaped) * video_latents.float() 
        + sigma_reshaped * noise
    )
    
    # Timesteps for model [0, 1000]
    timesteps = sigma * num_train_timesteps
    
    # ====================================================================
    # DETAILED LOGGING
    # ====================================================================
    if detailed_log or debug_mode:
        print0("\n" + "="*80)
        print0(f"[STEP {global_step}] MANUAL FLOW MATCHING")
        print0("="*80)
        print0(f"[WARNING] NOT using scheduler.add_noise() - it explodes!")
        print0(f"[INFO] Using manual: x_t = (1-σ)x_0 + σ*ε")
        print0("")
        print0(f"[SAMPLING] Method: {sampling_method}")
        print0(f"[FLOW] Shift: {flow_shift}")
        print0(f"[BATCH] Size: {batch_size}")
        print0("")
        print0(f"[U] Range: [{u.min():.4f}, {u.max():.4f}]")
        if u.numel() > 1:
            print0(f"[U] Mean: {u.mean():.4f}, Std: {u.std():.4f}")
        else:
            print0(f"[U] Value: {u.item():.4f}")
        print0("")
        print0(f"[SIGMA] Range: [{sigma.min():.4f}, {sigma.max():.4f}]")
        if sigma.numel() > 1:
            print0(f"[SIGMA] Mean: {sigma.mean():.4f}, Std: {sigma.std():.4f}")
        else:
            print0(f"[SIGMA] Value: {sigma.item():.4f}")
        print0("")
        print0(f"[TIMESTEPS] Range: [{timesteps.min():.2f}, {timesteps.max():.2f}]")
        print0("")
        print0(f"[WEIGHTS] Clean: {(1-sigma_reshaped).squeeze().cpu().numpy()}")
        print0(f"[WEIGHTS] Noise: {sigma_reshaped.squeeze().cpu().numpy()}")
        print0("")
        print0(f"[RANGES] Clean latents: [{video_latents.min():.4f}, {video_latents.max():.4f}]")
        print0(f"[RANGES] Noise:         [{noise.min():.4f}, {noise.max():.4f}]")
        print0(f"[RANGES] Noisy latents: [{noisy_latents.min():.4f}, {noisy_latents.max():.4f}]")
        
        # Sanity check
        max_expected = max(
            abs(video_latents.max().item()), 
            abs(video_latents.min().item()),
            abs(noise.max().item()),
            abs(noise.min().item())
        ) * 1.5
        if abs(noisy_latents.max()) > max_expected or abs(noisy_latents.min()) > max_expected:
            print0(f"\n⚠️  WARNING: Noisy range seems large! Expected ~{max_expected:.1f}")
        else:
            print0(f"\n✓ Noisy latents range is reasonable")
        print0("="*80 + "\n")
    
    elif summary_log:
        print0(f"[STEP {global_step}] σ=[{sigma.min():.3f},{sigma.max():.3f}] | "
               f"t=[{timesteps.min():.1f},{timesteps.max():.1f}] | "
               f"noisy=[{noisy_latents.min():.1f},{noisy_latents.max():.1f}] | "
               f"{sampling_method}")

    # Convert to bf16
    noisy_latents = noisy_latents.to(bf16)
    timesteps_for_model = timesteps.to(bf16)

    # ========================================================================
    # Forward Pass
    # ========================================================================
    
    fsdp_model = model_map["transformer"]["fsdp_transformer"]
    
    try:
        model_pred = fsdp_model(
            hidden_states=noisy_latents,
            timestep=timesteps_for_model,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )

        if isinstance(model_pred, tuple):
            model_pred = model_pred[0]

    except Exception as e:
        print0(f"[ERROR] Forward pass failed: {e}")
        print0(f"[DEBUG] noisy_latents: {noisy_latents.shape}, range: [{noisy_latents.min()}, {noisy_latents.max()}]")
        print0(f"[DEBUG] timesteps: {timesteps_for_model.shape}, range: [{timesteps_for_model.min()}, {timesteps_for_model.max()}]")
        raise

    # ========================================================================
    # Target: Flow Matching Velocity
    # ========================================================================
    
    # Flow matching target: v = ε - x_0
    target = noise - video_latents.float()
    
    # ========================================================================
    # Loss with Flow Weighting
    # ========================================================================
    
    loss = torch.nn.functional.mse_loss(
        model_pred.float(),
        target.float(),
        reduction="none"
    )
    
    # Flow weight: w = 1 + shift * σ
    loss_weight = 1.0 + flow_shift * sigma
    loss_weight = loss_weight.view(-1, 1, 1, 1, 1).to(device)
    
    unweighted_loss = loss.mean()
    weighted_loss = (loss * loss_weight).mean()
    
    # Safety check
    if torch.isnan(weighted_loss) or weighted_loss > 100:
        print0(f"[ERROR] Loss explosion! Loss={weighted_loss.item():.3f}")
        print0(f"[DEBUG] Stopping training - check hyperparameters")
        raise ValueError(f"Loss exploded: {weighted_loss.item()}")
    
    # ====================================================================
    # LOSS LOGGING
    # ====================================================================
    if detailed_log or debug_mode:
        print0("="*80)
        print0(f"[STEP {global_step}] LOSS DEBUG")
        print0("="*80)
        print0(f"[TARGET] Flow matching: v = ε - x_0")
        print0(f"[PREDICTION] Scheduler type (inference only): {type(pipe.scheduler).__name__}")
        print0("")
        print0(f"[RANGES] Model pred: [{model_pred.min():.4f}, {model_pred.max():.4f}]")
        print0(f"[RANGES] Target (v): [{target.min():.4f}, {target.max():.4f}]")
        print0("")
        print0(f"[WEIGHTS] Formula: 1 + {flow_shift} * σ")
        print0(f"[WEIGHTS] Range: [{loss_weight.min():.4f}, {loss_weight.max():.4f}]")
        if loss_weight.numel() > 1:
            print0(f"[WEIGHTS] Mean: {loss_weight.mean():.4f}")
        else:
            print0(f"[WEIGHTS] Value: {loss_weight.mean():.4f}")
        print0("")
        print0(f"[LOSS] Unweighted: {unweighted_loss.item():.6f}")
        print0(f"[LOSS] Weighted:   {weighted_loss.item():.6f}")
        print0(f"[LOSS] Impact:     {(weighted_loss/max(unweighted_loss, 1e-8)):.3f}x")
        print0("="*80 + "\n")
    
    elif summary_log:
        print0(f"[STEP {global_step}] Loss: {weighted_loss.item():.6f} | "
               f"w=[{loss_weight.min():.2f},{loss_weight.max():.2f}]")

    # Metrics
    metrics = {
        "loss": weighted_loss.item(),
        "unweighted_loss": unweighted_loss.item(),
        "sigma_min": sigma.min().item(),
        "sigma_max": sigma.max().item(),
        "sigma_mean": sigma.mean().item(),
        "weight_min": loss_weight.min().item(),
        "weight_max": loss_weight.max().item(),
        "timestep_min": timesteps.min().item(),
        "timestep_max": timesteps.max().item(),
        "noisy_min": noisy_latents.min().item(),
        "noisy_max": noisy_latents.max().item(),
        "sampling_method": sampling_method,
    }
    
    return weighted_loss, metrics