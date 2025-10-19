# training_step_t2v.py - Advanced Flow Matching with Karras Scheduling

import os
from typing import Dict, Tuple

import torch
from dist_utils import print0
from time_shift_utils import compute_density_for_timestep_sampling


def compute_karras_sigma(
    u: torch.Tensor,
    sigma_min: float = 1e-3,
    sigma_max: float = 1.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """
    Karras-style sigma scheduling for broader, smoother coverage.
    
    Args:
        u: Uniform or density-sampled values in [0, 1]
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value
        rho: Schedule parameter (higher = more emphasis on noisy timesteps)
    
    Returns:
        Sigma values with Karras scheduling
    """
    # Karras schedule: sigma = (sigma_max^rho + u * (sigma_min^rho - sigma_max^rho))^(1/rho)
    sigma_min_rho = sigma_min ** rho
    sigma_max_rho = sigma_max ** rho
    
    sigma = (sigma_max_rho + u * (sigma_min_rho - sigma_max_rho)) ** (1.0 / rho)
    
    # Normalize to [0, 1] for the convex combination
    sigma_normalized = torch.clamp(sigma / sigma_max, 0.0, 1.0)
    
    return sigma_normalized, sigma  # Return both normalized and continuous


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
    # Advanced pretrain parameters
    use_karras_schedule: bool = False,
    sigma_min: float = 1e-3,
    sigma_max: float = 1.0,
    rho: float = 7.0,
    loss_weight_k: float = 6.0,
    warmup_steps: int = 0,
    warmup_sigma_min_clamp: float = 0.05,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Advanced flow matching training with:
    - Karras-style sigma scheduling
    - Loss weight annealing
    - Warmup phase
    - fp32 timesteps for stability
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
        
        # Apply warmup clamping if in warmup phase
        if warmup_steps > 0 and global_step < warmup_steps:
            u = torch.clamp(u, min=warmup_sigma_min_clamp)
            if detailed_log:
                print0(f"[WARMUP] Step {global_step}/{warmup_steps}: Clamped u >= {warmup_sigma_min_clamp}")
        
        # Choose sigma mapping
        if use_karras_schedule:
            # Karras-style scheduling (recommended for pretraining)
            sigma, sigma_continuous = compute_karras_sigma(
                u, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
            )
            schedule_type = f"karras_rho{rho}"
        else:
            # Simple flow shift
            u_clamped = torch.clamp(u, min=1e-5)
            sigma = flow_shift / (flow_shift + (1.0 / u_clamped - 1.0))
            sigma = torch.clamp(sigma, 0.0, 1.0)
            sigma_continuous = sigma  # For simple mode, they're the same
            schedule_type = f"shift{flow_shift}"
        
    else:
        # Simple uniform without shift
        u = torch.rand(size=(batch_size,), device=device)
        sigma = u
        sigma_continuous = u
        sampling_method = "uniform_no_shift"
        schedule_type = "none"

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
    
    # Timesteps for model [0, num_train_timesteps-1] in fp32
    # Use fp32 to avoid bf16 quantization at small timesteps
    timesteps = (sigma * (num_train_timesteps - 1)).clamp(0, num_train_timesteps - 1)
    timesteps_fp32 = timesteps.to(torch.float32)  # Keep in fp32 for stability
    
    # ====================================================================
    # DETAILED LOGGING
    # ====================================================================
    if detailed_log or debug_mode:
        print0("\n" + "="*80)
        print0(f"[STEP {global_step}] ADVANCED FLOW MATCHING")
        print0("="*80)
        print0(f"[MODE] Schedule: {schedule_type}, Sampling: {sampling_method}")
        if warmup_steps > 0 and global_step < warmup_steps:
            print0(f"[WARMUP] {global_step}/{warmup_steps} (sigma >= {warmup_sigma_min_clamp})")
        print0(f"[BATCH] Size: {batch_size}")
        print0("")
        print0(f"[U] Range: [{u.min():.4f}, {u.max():.4f}]")
        if u.numel() > 1:
            print0(f"[U] Mean: {u.mean():.4f}, Std: {u.std():.4f}")
        print0("")
        print0(f"[SIGMA] Range: [{sigma.min():.4f}, {sigma.max():.4f}]")
        if sigma.numel() > 1:
            print0(f"[SIGMA] Mean: {sigma.mean():.4f}, Std: {sigma.std():.4f}")
        if use_karras_schedule:
            print0(f"[SIGMA_CONT] Range: [{sigma_continuous.min():.4f}, {sigma_continuous.max():.4f}]")
        print0("")
        print0(f"[TIMESTEPS] Range: [{timesteps.min():.2f}, {timesteps.max():.2f}] (fp32)")
        print0("")
        print0(f"[WEIGHTS] Clean: {(1-sigma_reshaped).squeeze().cpu().numpy()}")
        print0(f"[WEIGHTS] Noise: {sigma_reshaped.squeeze().cpu().numpy()}")
        print0("")
        print0(f"[RANGES] Clean latents: [{video_latents.min():.4f}, {video_latents.max():.4f}]")
        print0(f"[RANGES] Noise:         [{noise.min():.4f}, {noise.max():.4f}]")
        print0(f"[RANGES] Noisy latents: [{noisy_latents.min():.4f}, {noisy_latents.max():.4f}]")
        print0("="*80 + "\n")
    
    elif summary_log:
        warmup_str = f" WARMUP({global_step}/{warmup_steps})" if (warmup_steps > 0 and global_step < warmup_steps) else ""
        print0(f"[STEP {global_step}]{warmup_str} σ=[{sigma.min():.3f},{sigma.max():.3f}] | "
               f"t=[{timesteps.min():.1f},{timesteps.max():.1f}] | "
               f"{schedule_type} | {sampling_method}")

    # Convert noisy latents to bf16 for model
    noisy_latents = noisy_latents.to(bf16)

    # ========================================================================
    # Forward Pass
    # ========================================================================
    
    fsdp_model = model_map["transformer"]["fsdp_transformer"]
    
    try:
        model_pred = fsdp_model(
            hidden_states=noisy_latents,
            timestep=timesteps_fp32,  # Pass fp32 timesteps
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )

        if isinstance(model_pred, tuple):
            model_pred = model_pred[0]

    except Exception as e:
        print0(f"[ERROR] Forward pass failed: {e}")
        print0(f"[DEBUG] noisy_latents: {noisy_latents.shape}, range: [{noisy_latents.min()}, {noisy_latents.max()}]")
        print0(f"[DEBUG] timesteps: {timesteps_fp32.shape}, range: [{timesteps_fp32.min()}, {timesteps_fp32.max()}]")
        raise

    # ========================================================================
    # Target: Flow Matching Velocity
    # ========================================================================
    
    # Flow matching target: v = ε - x_0
    target = noise - video_latents.float()
    
    # ========================================================================
    # Loss with Advanced Weighting
    # ========================================================================
    
    loss = torch.nn.functional.mse_loss(
        model_pred.float(),
        target.float(),
        reduction="none"
    )
    
    # Advanced loss weighting
    if use_karras_schedule:
        # Karras mode: Weight based on continuous sigma
        # w = 1 + k * (sigma_cont / sigma_max)
        loss_weight = 1.0 + loss_weight_k * (sigma_continuous / sigma_max)
        weight_formula = f"1 + {loss_weight_k:.2f} * (σ_cont / {sigma_max})"
    else:
        # Simple mode: w = 1 + k * σ
        # For finetuning, k = flow_shift (default 3.0)
        # For pretrain without Karras, k can be custom (default 6.0)
        loss_weight = 1.0 + loss_weight_k * sigma
        weight_formula = f"1 + {loss_weight_k:.2f} * σ"
    
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
        print0("")
        print0(f"[RANGES] Model pred: [{model_pred.min():.4f}, {model_pred.max():.4f}]")
        print0(f"[RANGES] Target (v): [{target.min():.4f}, {target.max():.4f}]")
        print0("")
        if use_karras_schedule:
            print0(f"[WEIGHTS] Formula: 1 + {loss_weight_k:.2f} * (σ_cont / {sigma_max})")
        else:
            print0(f"[WEIGHTS] Formula: 1 + {loss_weight_k:.2f} * σ")
        print0(f"[WEIGHTS] Range: [{loss_weight.min():.4f}, {loss_weight.max():.4f}]")
        print0(f"[WEIGHTS] Mean: {loss_weight.mean():.4f}")
        print0("")
        print0(f"[LOSS] Unweighted: {unweighted_loss.item():.6f}")
        print0(f"[LOSS] Weighted:   {weighted_loss.item():.6f}")
        print0(f"[LOSS] Impact:     {(weighted_loss/max(unweighted_loss, 1e-8)):.3f}x")
        print0(f"[LOSS] k value:    {loss_weight_k:.3f}")
        print0("="*80 + "\n")
    
    elif summary_log:
        print0(f"[STEP {global_step}] Loss: {weighted_loss.item():.6f} | "
               f"k={loss_weight_k:.2f} | w=[{loss_weight.min():.2f},{loss_weight.max():.2f}]")

    # Metrics
    metrics = {
        "loss": weighted_loss.item(),
        "unweighted_loss": unweighted_loss.item(),
        "sigma_min": sigma.min().item(),
        "sigma_max": sigma.max().item(),
        "sigma_mean": sigma.mean().item(),
        "weight_min": loss_weight.min().item(),
        "weight_max": loss_weight.max().item(),
        "weight_mean": loss_weight.mean().item(),
        "timestep_min": timesteps.min().item(),
        "timestep_max": timesteps.max().item(),
        "noisy_min": noisy_latents.min().item(),
        "noisy_max": noisy_latents.max().item(),
        "sampling_method": sampling_method,
        "schedule_type": schedule_type,
        "loss_weight_k": loss_weight_k,
    }
    
    if use_karras_schedule:
        metrics["sigma_continuous_min"] = sigma_continuous.min().item()
        metrics["sigma_continuous_max"] = sigma_continuous.max().item()
        metrics["sigma_continuous_mean"] = sigma_continuous.mean().item()
    
    if warmup_steps > 0 and global_step < warmup_steps:
        metrics["in_warmup"] = True
        metrics["warmup_progress"] = global_step / warmup_steps
    else:
        metrics["in_warmup"] = False
    
    return weighted_loss, metrics