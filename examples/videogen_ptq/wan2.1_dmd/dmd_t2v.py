# dmd_t2v.py - DMD (Distribution Matching Distillation) for WAN 2.1 T2V
# FIXED VERSION - Gradient flow bug fixed

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from dist_utils import print0


class DMDT2V:
    """
    DMD (Distribution Matching Distillation) for WAN 2.1 T2V.

    FIXES APPLIED:
    1. Fixed tensor indexing in _sample_discrete_timestep()
    2. Fixed device handling in flow/noise conversion
    3. Consistent timestep ranges (0-999)
    4. Better numerical stability
    5. Removed double timestep shift application
    6. Fixed timestep shapes for transformer vs scheduler
    7. CRITICAL: Fixed gradient flow bug - removed torch.no_grad() wrapper
    """

    def __init__(
        self,
        model_map: Dict,
        scheduler,
        device,
        bf16,
        # DMD hyperparameters
        num_train_timestep: int = 1000,
        min_step: int = 20,
        max_step: int = 999,
        real_guidance_scale: float = 5.0,
        fake_guidance_scale: float = 0.0,
        timestep_shift: float = 3.0,
        ts_schedule: bool = True,
        ts_schedule_max: bool = False,
        min_score_timestep: int = 0,
        denoising_loss_type: str = "flow",
        denoising_step_list: list = None,
    ):
        """Initialize DMD trainer."""
        self.model_map = model_map
        self.scheduler = scheduler
        self.device = device
        self.bf16 = bf16

        # DMD hyperparameters
        self.num_train_timestep = num_train_timestep
        self.min_step = min_step
        self.max_step = max_step
        self.real_guidance_scale = real_guidance_scale
        self.fake_guidance_scale = fake_guidance_scale
        self.timestep_shift = timestep_shift
        self.ts_schedule = ts_schedule
        self.ts_schedule_max = ts_schedule_max
        self.min_score_timestep = min_score_timestep
        self.denoising_loss_type = denoising_loss_type

        if denoising_step_list is None:
            self.denoising_step_list = [999, 749, 499, 249, 0]
        else:
            self.denoising_step_list = denoising_step_list

        # Remove zero from list for sampling
        self.denoising_step_list_nonzero = [t for t in self.denoising_step_list if t > 0]

        print0("[DMD] Initialized with:")
        print0(f"  - Real guidance scale: {real_guidance_scale}")
        print0(f"  - Fake guidance scale: {fake_guidance_scale}")
        print0(f"  - Timestep shift: {timestep_shift}")
        print0(f"  - Denoising loss type: {denoising_loss_type}")
        print0(f"  - Discrete timesteps: {self.denoising_step_list}")
        print0(f"  - Non-zero timesteps for sampling: {self.denoising_step_list_nonzero}")

    def _sample_discrete_timestep(self, batch_size: int, num_frame: int, uniform_timestep: bool = True) -> torch.Tensor:
        """
        Sample DISCRETE timesteps from denoising_step_list for DMD training.

        Args:
            batch_size: Batch size
            num_frame: Number of frames
            uniform_timestep: Use same timestep for all frames (recommended)

        Returns:
            Timestep tensor of shape [batch_size, num_frame]
        """
        timestep_tensor = torch.tensor(self.denoising_step_list_nonzero, device=self.device, dtype=torch.long)

        if uniform_timestep:
            # Sample one discrete timestep per sample in batch
            indices = torch.randint(0, len(self.denoising_step_list_nonzero), (batch_size,), device=self.device)
            timesteps = timestep_tensor[indices]
            timestep = timesteps.unsqueeze(1).repeat(1, num_frame)
        else:
            # Sample different discrete timesteps for each frame
            indices = torch.randint(
                0, len(self.denoising_step_list_nonzero), (batch_size, num_frame), device=self.device
            )
            timestep = timestep_tensor[indices.flatten()].reshape(batch_size, num_frame)

        return timestep

    def _compute_kl_grad(
        self,
        noisy_latent: torch.Tensor,
        estimated_clean_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute KL gradient (Equation 7 in DMD paper).

        CRITICAL FIX: Fake score model needs gradients, only teacher should be in no_grad
        """
        batch_size, num_frame = noisy_latent.shape[:2]

        # Validate inputs
        assert noisy_latent.shape == estimated_clean_latent.shape, (
            f"Shape mismatch: {noisy_latent.shape} vs {estimated_clean_latent.shape}"
        )

        # Transformer expects timestep of shape [B], not [B, F]
        if timestep.ndim == 2:
            assert (timestep == timestep[:, 0:1]).all(), "All frames must have the same timestep"
            timestep_model = timestep[:, 0]
        else:
            timestep_model = timestep

        # Step 1: Compute fake score (critic prediction) - NEEDS GRADIENTS FOR GENERATOR
        fake_score_model = self.model_map["fake_score"]["fsdp_transformer"]

        pred_fake_cond = fake_score_model(
            hidden_states=noisy_latent,
            timestep=timestep_model.to(self.bf16),
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )
        if isinstance(pred_fake_cond, tuple):
            pred_fake_cond = pred_fake_cond[0]

        # Apply CFG for fake score if needed
        if abs(self.fake_guidance_scale) > 1e-6:
            pred_fake_uncond = fake_score_model(
                hidden_states=noisy_latent,
                timestep=timestep_model.to(self.bf16),
                encoder_hidden_states=negative_text_embeddings,
                return_dict=False,
            )
            if isinstance(pred_fake_uncond, tuple):
                pred_fake_uncond = pred_fake_uncond[0]

            pred_fake_image = pred_fake_cond + (pred_fake_cond - pred_fake_uncond) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_cond

        # Step 2: Compute real score (teacher prediction) - NO GRADIENTS NEEDED (FROZEN)
        real_score_model = self.model_map["real_score"]["fsdp_transformer"]

        with torch.no_grad():  # Only teacher is frozen
            pred_real_cond = real_score_model(
                hidden_states=noisy_latent,
                timestep=timestep_model.to(self.bf16),
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )
            if isinstance(pred_real_cond, tuple):
                pred_real_cond = pred_real_cond[0]

            pred_real_uncond = real_score_model(
                hidden_states=noisy_latent,
                timestep=timestep_model.to(self.bf16),
                encoder_hidden_states=negative_text_embeddings,
                return_dict=False,
            )
            if isinstance(pred_real_uncond, tuple):
                pred_real_uncond = pred_real_uncond[0]

            pred_real_image = pred_real_cond + (pred_real_cond - pred_real_uncond) * self.real_guidance_scale

        # Step 3: Compute DMD gradient (Equation 7)
        # CRITICAL: Keep gradients flowing from fake_score
        grad = pred_fake_image - pred_real_image

        # Step 4: Gradient normalization (Equation 8)
        if normalization:
            p_real = estimated_clean_latent - pred_real_image
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            normalizer = torch.clamp(normalizer, min=1e-5)
            grad = grad / normalizer

        # Handle NaN/Inf
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print0("[WARNING] NaN/Inf detected in gradient, clamping...")
            grad = torch.nan_to_num(grad, nan=0.0, posinf=10.0, neginf=-10.0)

        log_dict = {
            "dmd_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep_mean": timestep_model.float().mean().detach(),
            "timestep_min": timestep_model.float().min().detach(),
            "timestep_max": timestep_model.float().max().detach(),
        }

        return grad, log_dict

    def compute_distribution_matching_loss(
        self,
        generated_latent: torch.Tensor,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: Optional[int] = None,
        denoised_timestep_to: Optional[int] = None,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute DMD loss (Equation 7 in DMD paper).

        CRITICAL FIX: Removed torch.no_grad() wrapper to allow gradient flow
        """
        debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

        batch_size, num_channels, num_frame, height, width = generated_latent.shape

        # CRITICAL FIX: No torch.no_grad() here! We need gradients to flow!

        # Step 1: Sample DISCRETE timestep
        timestep = self._sample_discrete_timestep(batch_size, num_frame, uniform_timestep=True)
        timestep = timestep.clamp(self.min_step, self.max_step).long()

        # Step 2: Add noise to generated latent
        noise = torch.randn_like(generated_latent, dtype=torch.float32)

        # Permute and reshape for scheduler: [B, C, F, H, W] -> [B, F, C, H, W] -> [B*F, C, H, W]
        generated_permuted = generated_latent.permute(0, 2, 1, 3, 4)
        noise_permuted = noise.permute(0, 2, 1, 3, 4)

        generated_flat = generated_permuted.reshape(batch_size * num_frame, num_channels, height, width)
        noise_flat = noise_permuted.reshape(batch_size * num_frame, num_channels, height, width)
        timestep_flat = timestep.flatten()

        # IMPORTANT: Keep gradient flow through noising operation
        noisy_latent_flat = self.scheduler.add_noise(generated_flat.float(), noise_flat, timestep_flat)

        # Unflatten and permute back: [B*F, C, H, W] -> [B, F, C, H, W] -> [B, C, F, H, W]
        noisy_latent_unflat = noisy_latent_flat.reshape(batch_size, num_frame, num_channels, height, width)
        noisy_latent = noisy_latent_unflat.permute(0, 2, 1, 3, 4)

        # Convert back to bf16 for model forward
        noisy_latent = noisy_latent.to(self.bf16)

        # Step 3: Compute KL gradient
        grad, dmd_log_dict = self._compute_kl_grad(
            noisy_latent=noisy_latent,
            estimated_clean_latent=generated_latent,
            timestep=timestep,
            text_embeddings=text_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            normalization=True,
        )

        # Step 4: DMD loss (Equation 7)
        # CRITICAL FIX: Don't detach grad - we need gradients to flow!
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                generated_latent.double()[gradient_mask],
                (generated_latent.double() - grad.double())[gradient_mask],  # Removed .detach()
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                generated_latent.double(),
                (generated_latent.double() - grad.double()),  # Removed .detach()
                reduction="mean",
            )

        # Check for NaN/Inf in loss
        if torch.isnan(dmd_loss) or torch.isinf(dmd_loss):
            print0("[ERROR] Invalid DMD loss detected!")
            print0(f"  Generated latent range: [{generated_latent.min():.3f}, {generated_latent.max():.3f}]")
            print0(f"  Gradient range: [{grad.min():.3f}, {grad.max():.3f}]")
            raise ValueError("DMD loss is NaN or Inf")

        if debug_mode or (global_step % 100 == 0):
            print0(f"[DMD LOSS] Step {global_step}")
            print0(f"  Timestep range: [{timestep.min().item():.1f}, {timestep.max().item():.1f}]")
            print0(f"  Gradient norm: {dmd_log_dict['dmd_gradient_norm'].item():.6f}")
            print0(f"  DMD loss: {dmd_loss.item():.6f}")

        return dmd_loss, dmd_log_dict

    def compute_critic_loss(
        self,
        generated_latent: torch.Tensor,
        text_embeddings: torch.Tensor,
        denoised_timestep_from: Optional[int] = None,
        denoised_timestep_to: Optional[int] = None,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute critic (fake_score) training loss.

        FIXES FOR STABILITY:
        1. Detach generated_latent to prevent feedback loop
        2. Clamp predictions to reasonable range
        3. Use Huber loss for robustness
        4. Adaptive loss scaling
        """
        debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

        batch_size, num_channels, num_frame, height, width = generated_latent.shape

        # CRITICAL: Detach generated latent to prevent gradients flowing back to generator
        # The critic should learn to denoise, not affect the generator
        generated_latent = generated_latent.detach()

        # Step 1: Sample timestep for critic
        if self.ts_schedule and denoised_timestep_to is not None:
            min_timestep = max(denoised_timestep_to, self.min_score_timestep)
        else:
            min_timestep = self.min_score_timestep

        if self.ts_schedule_max and denoised_timestep_from is not None:
            max_timestep = min(denoised_timestep_from, self.num_train_timestep - 1)
        else:
            max_timestep = self.num_train_timestep - 1

        # Ensure valid range
        min_timestep = max(min_timestep, self.min_step)
        max_timestep = min(max_timestep, self.max_step)

        if min_timestep >= max_timestep:
            min_timestep = self.min_step
            max_timestep = self.max_step

        # Generate timestep as [B], not [B, F]
        critic_timestep = torch.randint(
            min_timestep, max_timestep + 1, (batch_size,), device=self.device, dtype=torch.long
        )

        # For scheduler, we need [B*F] timesteps
        critic_timestep_scheduler = critic_timestep.unsqueeze(1).repeat(1, num_frame).flatten()

        # Step 2: Add noise to generated latent
        critic_noise = torch.randn_like(generated_latent, dtype=torch.float32)

        # Permute and reshape for scheduler
        generated_permuted = generated_latent.permute(0, 2, 1, 3, 4)
        noise_permuted = critic_noise.permute(0, 2, 1, 3, 4)

        generated_flat = generated_permuted.reshape(batch_size * num_frame, num_channels, height, width)
        noise_flat = noise_permuted.reshape(batch_size * num_frame, num_channels, height, width)

        noisy_generated_flat = self.scheduler.add_noise(generated_flat.float(), noise_flat, critic_timestep_scheduler)

        # Unflatten and permute back
        noisy_generated_unflat = noisy_generated_flat.reshape(batch_size, num_frame, num_channels, height, width)
        noisy_generated_latent = noisy_generated_unflat.permute(0, 2, 1, 3, 4)

        # Convert to bf16 for model forward
        noisy_generated_latent = noisy_generated_latent.to(self.bf16)

        # Step 3: Critic forward pass
        fake_score_model = self.model_map["fake_score"]["fsdp_transformer"]

        pred_fake_image = fake_score_model(
            hidden_states=noisy_generated_latent,
            timestep=critic_timestep.to(self.bf16),
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )
        if isinstance(pred_fake_image, tuple):
            pred_fake_image = pred_fake_image[0]

        # STABILITY FIX: Clamp predictions to reasonable range
        pred_fake_image = torch.clamp(pred_fake_image, min=-50.0, max=50.0)

        # Step 4: Compute denoising loss
        if self.denoising_loss_type == "flow":
            # Flow matching loss
            pred_fake_permuted = pred_fake_image.permute(0, 2, 1, 3, 4)
            noisy_permuted = noisy_generated_latent.permute(0, 2, 1, 3, 4)
            generated_permuted = generated_latent.permute(0, 2, 1, 3, 4)

            pred_fake_flat = pred_fake_permuted.reshape(batch_size * num_frame, num_channels, height, width)
            noisy_flat = noisy_permuted.reshape(batch_size * num_frame, num_channels, height, width)
            generated_flat = generated_permuted.reshape(batch_size * num_frame, num_channels, height, width)

            flow_pred = self._convert_x0_to_flow_pred(
                x0_pred=pred_fake_flat,
                xt=noisy_flat,
                timestep=critic_timestep_scheduler,
            )

            target_flow = self._convert_x0_to_flow_pred(
                x0_pred=generated_flat,
                xt=noisy_flat,
                timestep=critic_timestep_scheduler,
            )

            # STABILITY FIX: Clamp flow predictions
            flow_pred = torch.clamp(flow_pred, min=-100.0, max=100.0)
            target_flow = torch.clamp(target_flow, min=-100.0, max=100.0)

            # Use Huber loss for robustness (less sensitive to outliers)
            critic_loss = F.huber_loss(flow_pred.float(), target_flow.float(), delta=1.0)

        elif self.denoising_loss_type == "x0":
            # Direct x0 prediction with Huber loss
            critic_loss = F.huber_loss(pred_fake_image.float(), generated_latent.float(), delta=1.0)

        elif self.denoising_loss_type == "epsilon":
            pred_fake_permuted = pred_fake_image.permute(0, 2, 1, 3, 4)
            noisy_permuted = noisy_generated_latent.permute(0, 2, 1, 3, 4)

            pred_fake_flat = pred_fake_permuted.reshape(batch_size * num_frame, num_channels, height, width)
            noisy_flat = noisy_permuted.reshape(batch_size * num_frame, num_channels, height, width)

            pred_noise = self._convert_x0_to_noise(
                x0=pred_fake_flat,
                xt=noisy_flat,
                timestep=critic_timestep_scheduler,
            )

            critic_noise_flat = critic_noise.permute(0, 2, 1, 3, 4).reshape(
                batch_size * num_frame, num_channels, height, width
            )

            # Clamp noise predictions
            pred_noise = torch.clamp(pred_noise, min=-50.0, max=50.0)

            critic_loss = F.huber_loss(pred_noise.float(), critic_noise_flat.float(), delta=1.0)

        else:
            raise ValueError(f"Unknown denoising_loss_type: {self.denoising_loss_type}")

        # STABILITY FIX: Scale loss if it's too large early in training
        if global_step < 100 and critic_loss > 10.0:
            critic_loss = critic_loss / 10.0
            if debug_mode:
                print0("  [SCALING] Applied 10x loss scaling for early training stability")

        # Check for NaN/Inf
        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            print0("[ERROR] Invalid critic loss detected!")
            print0(f"  pred_fake_image range: [{pred_fake_image.min():.3f}, {pred_fake_image.max():.3f}]")
            print0(f"  generated_latent range: [{generated_latent.min():.3f}, {generated_latent.max():.3f}]")
            raise ValueError("Critic loss is NaN or Inf")

        # STABILITY CHECK: Warn if loss is still high but don't crash
        if critic_loss > 100.0:
            print0(f"[WARNING] High critic loss: {critic_loss.item():.2f}")
            print0("  This may indicate instability. Consider:")
            print0("    - Reducing critic learning rate")
            print0("    - Using gradient clipping (already enabled)")
            print0("    - Checking timestep range")

        if debug_mode or (global_step % 100 == 0):
            print0(f"[CRITIC LOSS] Step {global_step}")
            print0(f"  Timestep range: [{critic_timestep.min().item():.1f}, {critic_timestep.max().item():.1f}]")
            print0(f"  Critic loss: {critic_loss.item():.6f}")
            print0(f"  Pred range: [{pred_fake_image.min().item():.2f}, {pred_fake_image.max().item():.2f}]")

        log_dict = {
            "critic_timestep_mean": critic_timestep.float().mean().detach(),
            "critic_loss_value": critic_loss.detach(),
            "pred_fake_range_min": pred_fake_image.min().detach(),
            "pred_fake_range_max": pred_fake_image.max().detach(),
        }

        return critic_loss, log_dict

    def _convert_x0_to_flow_pred(self, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Convert x0 prediction to flow prediction."""
        timestep = timestep.to(x0_pred.device, dtype=torch.float32)

        # Normalize timestep to [0, 1]
        t_norm = timestep / self.num_train_timestep
        t_norm = torch.clamp(t_norm, min=1e-5, max=1.0)

        # Compute sigma with shift
        sigma = self.timestep_shift / (self.timestep_shift + (1.0 / t_norm - 1.0))
        sigma = sigma.view(-1, 1, 1, 1)

        # Flow prediction
        flow_pred = (x0_pred - xt) / (1.0 - sigma + 1e-5)

        return flow_pred

    def _convert_x0_to_noise(self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Convert x0 to noise prediction."""
        timestep = timestep.to(x0.device, dtype=torch.long)

        if hasattr(self.scheduler, "alphas_cumprod") and self.scheduler.alphas_cumprod is not None:
            max_idx = len(self.scheduler.alphas_cumprod) - 1
            timestep = torch.clamp(timestep, 0, max_idx)

            alpha_bar = self.scheduler.alphas_cumprod.to(x0.device)[timestep]
            alpha_bar = alpha_bar.view(-1, 1, 1, 1)

            sqrt_alpha_bar = torch.sqrt(alpha_bar + 1e-8)
            sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))

            noise_pred = (xt - sqrt_alpha_bar * x0) / sqrt_one_minus_alpha_bar
        else:
            noise_pred = xt - x0

        return noise_pred
