# dmd_t2v.py - DMD with Self-Forcing Training Pipeline

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from dist_utils import print0


class DMDT2V:
    """
    DMD (Distribution Matching Distillation) for WAN 2.1 T2V with Self-Forcing.

    Key components:
    1. Generator (student) - trainable, learns to match teacher
    2. Teacher (real_score) - frozen, provides target distribution
    3. Critic (fake_score) - trainable, learns to score generated samples
    4. Self-forcing pipeline - generates training data via backward simulation

    CRITICAL FIXES:
    - Generator gradients flow through self-forcing pipeline
    - Teacher completely frozen and isolated
    - Critic uses detached generator outputs
    - Self-forcing: ONE random timestep per batch (memory efficient)
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
        """Initialize DMD trainer with self-forcing."""
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

        # Self-forcing pipeline (initialized lazily)
        self.inference_pipeline = None

        # Verify teacher is frozen
        self._verify_teacher_frozen()

        print0("[DMD] Initialized with self-forcing:")
        print0(f"  - Real guidance scale: {real_guidance_scale}")
        print0(f"  - Fake guidance scale: {fake_guidance_scale}")
        print0(f"  - Timestep shift: {timestep_shift}")
        print0(f"  - Denoising loss type: {denoising_loss_type}")
        print0(f"  - Discrete timesteps: {self.denoising_step_list}")

    def _verify_teacher_frozen(self):
        """Verify teacher is completely frozen."""
        teacher = self.model_map["real_score"]["fsdp_transformer"]
        
        trainable_count = sum(1 for p in teacher.parameters() if p.requires_grad)
        
        if trainable_count > 0:
            raise RuntimeError(
                f"[BUG] Teacher has {trainable_count} trainable parameters! "
                "Teacher MUST be frozen for DMD."
            )
        
        print0(f"[DMD] ✓ Teacher verified frozen (0 trainable params)")

    def _initialize_inference_pipeline(self):
        """Initialize self-forcing training pipeline (lazy initialization)."""
        from pipeline_t2v import SelfForcingTrainingPipeline
        
        try:
            model_name = self.model_map["generator"]["fsdp_transformer"].config._name_or_path
        except:
            model_name = "Wan2.1-T2V-1.3B"
        
        # CRITICAL: For 21 frames, use independent_first_frame=True
        # This gives: 1 independent frame + 20 frames (5 blocks of 4)
        self.inference_pipeline = SelfForcingTrainingPipeline(
            model_name=model_name,
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.model_map["generator"]["fsdp_transformer"],
            num_frame_per_block=4,
            independent_first_frame=True,  # FIXED: Was False, should be True for 21 frames
            same_step_across_blocks=True,
            last_step_only=True, ###TODO
            num_max_frames=21,
            context_noise=0,
        )
        
        print0("[DMD] Self-forcing pipeline initialized")
        print0(f"  - Configuration: 1 + {(21-1)} frames = {(21-1)//4} blocks of 4 frames")


    def _run_generator(
        self,
        image_or_video_shape,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        """
        Run generator with self-forcing backward simulation.
        
        KEY: This calls the self-forcing pipeline which:
        1. Starts from noise
        2. Iteratively denoises through discrete timesteps
        3. Computes gradients at ONE random timestep
        4. Returns final prediction WITH gradients
        
        Args:
            image_or_video_shape: [B, C, F, H, W]
            text_embeddings: Text conditioning
            negative_text_embeddings: Negative conditioning (unused in self-forcing)
            initial_latent: Optional I2V conditioning
            clip_fea: Optional CLIP features
            y: Optional VAE features
            global_step: Current training step
            
        Returns:
            generated_latent: Generated latent [B, C, F, H, W] WITH gradients
            gradient_mask: Optional mask for selective gradient computation
            denoised_timestep_from: Starting timestep
            denoised_timestep_to: Ending timestep
        """
        # Initialize pipeline lazily
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        # Create noise
        batch_size = image_or_video_shape[0]
        channels, frames, height, width = image_or_video_shape[1:]

        noise = torch.randn(
            batch_size, channels, frames, height, width,
            device=self.device, dtype=self.bf16
        )

        # Run self-forcing backward simulation WITH GRADIENTS
        generated_latent, denoised_timestep_from, denoised_timestep_to, _ = \
            self.inference_pipeline.inference_with_trajectory(
                noise=noise,
                text_embeddings=text_embeddings,
                clip_fea=clip_fea,
                y=y,
                initial_latent=initial_latent,
                global_step=global_step,
            )

        # Gradient mask (optional - for slicing last 21 frames)
        gradient_mask = None
        if generated_latent.shape[2] > 21:
            # Only compute loss on last 21 frames (for long video generation)
            gradient_mask = torch.zeros_like(generated_latent, dtype=torch.bool)
            gradient_mask[:, :, -21:] = True
            
            if global_step % 100 == 0:
                print0(f"[DMD] Using gradient mask: only last 21 frames")

        return generated_latent, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _assert_finite(self, tensor: torch.Tensor, name: str):
        """Assert tensor contains no NaN or Inf."""
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"[BUG] Non-finite values in {name}")

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

        CRITICAL: Fake score needs gradients for generator training!
        """
        batch_size = noisy_latent.shape[0]

        # Assert shapes
        assert noisy_latent.shape == estimated_clean_latent.shape
        assert noisy_latent.ndim == 5 and noisy_latent.shape[1] == 16
        assert timestep.ndim == 1 and timestep.shape[0] == batch_size

        # Step 1: Compute fake score (critic) - NEEDS GRADIENTS!
        fake_score_model = self.model_map["fake_score"]["fsdp_transformer"]

        pred_fake_cond = fake_score_model(
            hidden_states=noisy_latent,
            timestep=timestep.to(self.bf16),
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )
        if isinstance(pred_fake_cond, tuple):
            pred_fake_cond = pred_fake_cond[0]

        self._assert_finite(pred_fake_cond, "pred_fake_cond")

        # Apply CFG for fake score if needed
        if abs(self.fake_guidance_scale) > 1e-6:
            pred_fake_uncond = fake_score_model(
                hidden_states=noisy_latent,
                timestep=timestep.to(self.bf16),
                encoder_hidden_states=negative_text_embeddings,
                return_dict=False,
            )
            if isinstance(pred_fake_uncond, tuple):
                pred_fake_uncond = pred_fake_uncond[0]

            pred_fake_image = pred_fake_uncond + self.fake_guidance_scale * (pred_fake_cond - pred_fake_uncond)
        else:
            pred_fake_image = pred_fake_cond

        # Step 2: Compute real score (teacher) - NO GRADIENTS (frozen)
        real_score_model = self.model_map["real_score"]["fsdp_transformer"]

        with torch.no_grad():
            pred_real_cond = real_score_model(
                hidden_states=noisy_latent,
                timestep=timestep.to(self.bf16),
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )
            if isinstance(pred_real_cond, tuple):
                pred_real_cond = pred_real_cond[0]

            pred_real_uncond = real_score_model(
                hidden_states=noisy_latent,
                timestep=timestep.to(self.bf16),
                encoder_hidden_states=negative_text_embeddings,
                return_dict=False,
            )
            if isinstance(pred_real_uncond, tuple):
                pred_real_uncond = pred_real_uncond[0]

            pred_real_image = pred_real_uncond + self.real_guidance_scale * (pred_real_cond - pred_real_uncond)

        # Step 3: Compute DMD gradient (Equation 7)
        grad = pred_fake_image - pred_real_image

        # Step 4: Gradient normalization (Equation 8)
        if normalization:
            p_real = estimated_clean_latent - pred_real_image
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            normalizer = torch.clamp(normalizer, min=1e-5)
            grad = grad / normalizer

        self._assert_finite(grad, "dmd_gradient")

        log_dict = {
            "dmd_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep_mean": timestep.float().mean().detach(),
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
        
        Input: generated_latent from self-forcing pipeline (WITH gradients)
        Output: DMD loss that flows gradients to generator AND critic
        """
        debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

        batch_size, num_channels, num_frame, height, width = generated_latent.shape

        assert generated_latent.ndim == 5 and num_channels == 16

        # Step 1: Sample timestep for DMD loss
        if self.ts_schedule and denoised_timestep_to is not None:
            min_timestep = max(denoised_timestep_to, self.min_score_timestep)
        else:
            min_timestep = self.min_score_timestep

        if self.ts_schedule_max and denoised_timestep_from is not None:
            max_timestep = min(denoised_timestep_from, self.num_train_timestep - 1)
        else:
            max_timestep = self.num_train_timestep - 1

        min_timestep = max(min_timestep, self.min_step)
        max_timestep = min(max_timestep, self.max_step)

        if min_timestep >= max_timestep:
            min_timestep = self.min_step
            max_timestep = self.max_step

        timestep = torch.randint(
            min_timestep, max_timestep + 1, (batch_size,), device=self.device, dtype=torch.long
        )

        # Step 2: Add noise (in fp32)
        noise = torch.randn_like(generated_latent, dtype=torch.float32, device=self.device)
        generated_fp32 = generated_latent.float()

        # Permute and flatten for scheduler
        generated_permuted = generated_fp32.permute(0, 2, 1, 3, 4)
        noise_permuted = noise.permute(0, 2, 1, 3, 4)

        generated_flat = generated_permuted.reshape(batch_size * num_frame, num_channels, height, width)
        noise_flat = noise_permuted.reshape(batch_size * num_frame, num_channels, height, width)

        timestep_flat = timestep.unsqueeze(1).repeat(1, num_frame).flatten()

        noisy_latent_flat = self.scheduler.add_noise(generated_flat, noise_flat, timestep_flat)

        # Unflatten and permute back
        noisy_latent_unflat = noisy_latent_flat.reshape(batch_size, num_frame, num_channels, height, width)
        noisy_latent = noisy_latent_unflat.permute(0, 2, 1, 3, 4).to(self.bf16)

        self._assert_finite(noisy_latent, "noisy_latent")

        if debug_mode or (global_step % 100 == 0):
            print0(f"[DMD LOSS] Step {global_step}")
            print0(f"  Timestep range: [{timestep.min().item()}, {timestep.max().item()}]")

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
        # CRITICAL: grad is NOT detached - gradients flow to critic!
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                generated_latent.double()[gradient_mask],
                (generated_latent.double() - grad.double())[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                generated_latent.double(),
                (generated_latent.double() - grad.double()),  # NO .detach() here!
                reduction="mean",
            )

        # Check for NaN/Inf
        if torch.isnan(dmd_loss) or torch.isinf(dmd_loss):
            print0("[ERROR] Invalid DMD loss!")
            raise ValueError(f"DMD loss is NaN or Inf: {dmd_loss.item()}")

        if debug_mode or (global_step % 100 == 0):
            print0(f"  DMD loss: {dmd_loss.item():.6f}")

        return dmd_loss, dmd_log_dict

    # In dmd_t2v.py, update compute_critic_loss method

    def compute_critic_loss(
        self,
        generated_latent: torch.Tensor,
        text_embeddings: torch.Tensor,
        denoised_timestep_from: Optional[int] = None,
        denoised_timestep_to: Optional[int] = None,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute critic loss using custom Rectified Flow scheduler."""
        
        debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"
        
        batch_size, num_channels, num_frame, height, width = generated_latent.shape
        
        # Detach generated latent
        generated_latent = generated_latent.detach()
        
        # Clamp input
        generated_latent = torch.clamp(generated_latent, min=-10.0, max=10.0)
        
        # Sample timestep
        if self.ts_schedule and denoised_timestep_to is not None:
            min_timestep = max(denoised_timestep_to, self.min_score_timestep)
        else:
            min_timestep = self.min_score_timestep
        
        if self.ts_schedule_max and denoised_timestep_from is not None:
            max_timestep = min(denoised_timestep_from, self.num_train_timestep - 1)
        else:
            max_timestep = self.num_train_timestep - 1
        
        min_timestep = max(min_timestep, self.min_step)
        max_timestep = min(max_timestep, self.max_step)
        
        if min_timestep >= max_timestep:
            min_timestep = self.min_step
            max_timestep = self.max_step
        
        critic_timestep = torch.randint(
            min_timestep, max_timestep + 1, (batch_size,), device=self.device, dtype=torch.long
        )
        
        critic_timestep_scheduler = critic_timestep.unsqueeze(1).repeat(1, num_frame).flatten()
        
        # Generate noise
        noise = torch.randn_like(generated_latent, dtype=torch.float32, device=self.device)
        
        # Permute for scheduler
        generated_permuted = generated_latent.permute(0, 2, 1, 3, 4)
        noise_permuted = noise.permute(0, 2, 1, 3, 4)
        
        generated_flat = generated_permuted.reshape(batch_size * num_frame, num_channels, height, width)
        noise_flat = noise_permuted.reshape(batch_size * num_frame, num_channels, height, width)
        
        # Add noise using custom scheduler
        noisy_generated_flat = self.scheduler.add_noise(
            generated_flat.float(),
            noise_flat,
            critic_timestep_scheduler
        )
        
        # Unflatten
        noisy_generated_unflat = noisy_generated_flat.reshape(batch_size, num_frame, num_channels, height, width)
        noisy_generated_latent = noisy_generated_unflat.permute(0, 2, 1, 3, 4).to(self.bf16)
        
        if debug_mode or (global_step % 100 == 0):
            print0(f"[CRITIC] Noisy latent range: [{noisy_generated_latent.min():.3f}, {noisy_generated_latent.max():.3f}]")
        
        # Critic forward pass
        fake_score_model = self.model_map["fake_score"]["fsdp_transformer"]
        
        pred_x0 = fake_score_model(
            hidden_states=noisy_generated_latent,
            timestep=critic_timestep.to(self.bf16),
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )
        if isinstance(pred_x0, tuple):
            pred_x0 = pred_x0[0]
        
        # Clamp prediction
        pred_x0 = torch.clamp(pred_x0, min=-10.0, max=10.0)
        
        # Compute loss based on denoising_loss_type
        if self.denoising_loss_type == "flow":
            # Convert to velocity using CUSTOM SCHEDULER
            pred_x0_permuted = pred_x0.permute(0, 2, 1, 3, 4)
            noisy_permuted = noisy_generated_latent.permute(0, 2, 1, 3, 4)
            generated_permuted = generated_latent.permute(0, 2, 1, 3, 4)
            
            pred_x0_flat = pred_x0_permuted.reshape(batch_size * num_frame, num_channels, height, width)
            noisy_flat = noisy_permuted.reshape(batch_size * num_frame, num_channels, height, width)
            generated_flat = generated_permuted.reshape(batch_size * num_frame, num_channels, height, width)
            
            # KEY: Use custom scheduler's velocity computation
            velocity_pred = self.scheduler.get_velocity_from_x0_pred(
                x_t=noisy_flat.float(),
                x_0_pred=pred_x0_flat.float(),
                timestep=critic_timestep_scheduler,
            )
            
            velocity_target = self.scheduler.get_velocity_from_x0_pred(
                x_t=noisy_flat.float(),
                x_0_pred=generated_flat.float(),
                timestep=critic_timestep_scheduler,
            )
            
            if debug_mode or (global_step % 100 == 0):
                print0(f"[CRITIC] Velocity pred range: [{velocity_pred.min():.3f}, {velocity_pred.max():.3f}]")
                print0(f"[CRITIC] Velocity target range: [{velocity_target.min():.3f}, {velocity_target.max():.3f}]")
            
            # Compute loss
            critic_loss = F.huber_loss(velocity_pred, velocity_target, delta=1.0)
            
        elif self.denoising_loss_type == "x0":
            # Direct x0 loss
            critic_loss = F.huber_loss(pred_x0.float(), generated_latent.float(), delta=1.0)
        
        else:
            raise ValueError(f"Unknown denoising_loss_type: {self.denoising_loss_type}")
        
        # Check for issues
        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            print0("[ERROR] Invalid critic loss!")
            raise ValueError("Critic loss is NaN or Inf")
        
        # Adaptive explosion threshold
        if global_step < 50:
            explosion_threshold = 1000.0
        elif global_step < 100:
            explosion_threshold = 500.0
        else:
            explosion_threshold = 100.0
        
        if critic_loss > explosion_threshold:
            print0(f"[WARNING] High critic loss: {critic_loss.item():.3f} (threshold: {explosion_threshold})")
            
            if global_step >= 100:
                print0(f"[ERROR] Critic loss explosion after warmup!")
                raise ValueError(f"Critic loss exploded: {critic_loss.item()}")
        
        if debug_mode or (global_step % 100 == 0):
            print0(f"[CRITIC LOSS] {critic_loss.item():.6f}")
        
        log_dict = {
            "critic_timestep_mean": critic_timestep.float().mean().detach(),
            "critic_loss_value": critic_loss.detach(),
        }
        
        return critic_loss, log_dict