# dmd_t2v.py - DMD (Distribution Matching Distillation) for WAN 2.1 T2V

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from dist_utils import print0


class DMDT2V:
    """
    DMD (Distribution Matching Distillation) for WAN 2.1 T2V.

    This class handles:
    1. Generator loss: DMD loss on self-generated samples
    2. Critic loss: Denoising loss on generated samples

    Key differences from base training:
    - Uses three models: generator (student), real_score (teacher), fake_score (critic)
    - Backward simulation to generate training samples
    - Distribution matching via KL gradient
    - CRITICAL: Uses discrete timesteps matching inference!
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
        max_step: int = 980,
        real_guidance_scale: float = 5.0,
        fake_guidance_scale: float = 0.0,
        timestep_shift: float = 3.0,
        ts_schedule: bool = True,
        ts_schedule_max: bool = False,
        min_score_timestep: int = 0,
        denoising_loss_type: str = "flow",
        denoising_step_list: list = None,  # NEW: Store discrete timesteps
    ):
        """
        Initialize DMD trainer.

        Args:
            model_map: Dictionary containing generator, real_score, fake_score models
            scheduler: Diffusion scheduler
            device: Training device
            bf16: BFloat16 dtype
            num_train_timestep: Total training timesteps (1000)
            min_step: Minimum timestep for noise (20)
            max_step: Maximum timestep for noise (980)
            real_guidance_scale: CFG scale for teacher (5.0)
            fake_guidance_scale: CFG scale for critic (0.0)
            timestep_shift: Flow matching shift parameter (3.0)
            ts_schedule: Use dynamic timestep scheduling
            ts_schedule_max: Use max timestep scheduling
            min_score_timestep: Minimum timestep for critic
            denoising_loss_type: "flow" or "epsilon" or "x0"
            denoising_step_list: List of discrete timesteps for DMD [1000, 750, 500, 250, 0]
        """
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

        # CRITICAL: Store discrete timesteps for DMD
        if denoising_step_list is None:
            # Use 999 instead of 1000 to match WAN 2.1 scheduler
            self.denoising_step_list = [999, 749, 499, 249, 0]
        else:
            self.denoising_step_list = denoising_step_list

        # Remove zero from list for sampling (we don't add noise at t=0)
        self.denoising_step_list_nonzero = [t for t in self.denoising_step_list if t > 0]

        print0("[DMD] Initialized with:")
        print0(f"  - Real guidance scale: {real_guidance_scale}")
        print0(f"  - Fake guidance scale: {fake_guidance_scale}")
        print0(f"  - Timestep shift: {timestep_shift}")
        print0(f"  - Denoising loss type: {denoising_loss_type}")
        print0(f"  - Discrete timesteps: {self.denoising_step_list}")
        print0(f"  - Non-zero timesteps for sampling: {self.denoising_step_list_nonzero}")

    def _get_timestep(
        self, min_timestep: int, max_timestep: int, batch_size: int, num_frame: int, uniform_timestep: bool = True
    ) -> torch.Tensor:
        """
        Sample timesteps for DMD training.

        DEPRECATED: This method samples continuous timesteps.
        Use _sample_discrete_timestep() for DMD instead!

        Args:
            min_timestep: Minimum timestep
            max_timestep: Maximum timestep
            batch_size: Batch size
            num_frame: Number of frames
            uniform_timestep: Use same timestep for all frames

        Returns:
            Timestep tensor of shape [batch_size, num_frame]
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep, max_timestep, [batch_size, 1], device=self.device, dtype=torch.long
            ).repeat(1, num_frame)
        else:
            timestep = torch.randint(
                min_timestep, max_timestep, [batch_size, num_frame], device=self.device, dtype=torch.long
            )

        return timestep

    def _sample_discrete_timestep(self, batch_size: int, num_frame: int, uniform_timestep: bool = True) -> torch.Tensor:
        """
        Sample DISCRETE timesteps from denoising_step_list for DMD training.

        CRITICAL: DMD must use the same discrete timesteps during training and inference!
        This ensures train/test consistency.

        Args:
            batch_size: Batch size
            num_frame: Number of frames
            uniform_timestep: Use same timestep for all frames (recommended)

        Returns:
            Timestep tensor of shape [batch_size, num_frame]
        """
        if uniform_timestep:
            # Sample one discrete timestep per sample in batch
            indices = torch.randint(0, len(self.denoising_step_list_nonzero), (batch_size,), device=self.device)
            timesteps = torch.tensor(
                [self.denoising_step_list_nonzero[idx] for idx in indices], device=self.device, dtype=torch.long
            )
            # Repeat for all frames
            timestep = timesteps.unsqueeze(1).repeat(1, num_frame)
        else:
            # Sample different discrete timesteps for each frame
            indices = torch.randint(
                0, len(self.denoising_step_list_nonzero), (batch_size, num_frame), device=self.device
            )
            timestep = torch.tensor(
                [[self.denoising_step_list_nonzero[idx] for idx in row] for row in indices],
                device=self.device,
                dtype=torch.long,
            )

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

        This gradient measures the distribution divergence between the
        fake critic and real teacher at a given noise level.

        Args:
            noisy_latent: Noisy latents [B, F, C, H, W]
            estimated_clean_latent: Estimated clean latents [B, F, C, H, W]
            timestep: Timesteps [B, F]
            text_embeddings: Conditional text embeddings
            negative_text_embeddings: Unconditional text embeddings
            normalization: Whether to normalize gradient

        Returns:
            grad: KL gradient
            log_dict: Logging dictionary
        """
        batch_size, num_frame = noisy_latent.shape[:2]

        # Step 1: Compute fake score (critic prediction)
        fake_score_model = self.model_map["fake_score"]["fsdp_transformer"]

        pred_fake_cond = fake_score_model(
            hidden_states=noisy_latent,
            timestep=timestep.to(self.bf16),
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )
        if isinstance(pred_fake_cond, tuple):
            pred_fake_cond = pred_fake_cond[0]

        # Apply CFG for fake score if needed
        if self.fake_guidance_scale != 0.0:
            pred_fake_uncond = fake_score_model(
                hidden_states=noisy_latent,
                timestep=timestep.to(self.bf16),
                encoder_hidden_states=negative_text_embeddings,
                return_dict=False,
            )
            if isinstance(pred_fake_uncond, tuple):
                pred_fake_uncond = pred_fake_uncond[0]

            pred_fake_image = pred_fake_cond + (pred_fake_cond - pred_fake_uncond) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_cond

        # Step 2: Compute real score (teacher prediction)
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

            # Apply CFG for real score
            pred_real_image = pred_real_cond + (pred_real_cond - pred_real_uncond) * self.real_guidance_scale

        # Step 3: Compute DMD gradient (Equation 7)
        grad = pred_fake_image - pred_real_image

        # Step 4: Gradient normalization (Equation 8)
        if normalization:
            p_real = estimated_clean_latent - pred_real_image
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            normalizer = torch.clamp(normalizer, min=1e-8)  # Avoid division by zero
            grad = grad / normalizer

        grad = torch.nan_to_num(grad)

        log_dict = {"dmd_gradient_norm": torch.mean(torch.abs(grad)).detach(), "timestep": timestep.detach()}

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

        Args:
            generated_latent: Generated latents from backward simulation [B, F, C, H, W]
            text_embeddings: Conditional text embeddings
            negative_text_embeddings: Unconditional text embeddings
            gradient_mask: Optional mask for selective gradient computation
            denoised_timestep_from: Starting timestep from backward simulation
            denoised_timestep_to: Ending timestep from backward simulation
            global_step: Current training step

        Returns:
            dmd_loss: DMD loss
            log_dict: Logging dictionary
        """
        debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

        batch_size, num_frame = generated_latent.shape[:2]

        with torch.no_grad():
            # Step 1: Sample DISCRETE timestep from denoising_step_list
            # CRITICAL: Must use same discrete timesteps as inference!
            timestep = self._sample_discrete_timestep(batch_size, num_frame, uniform_timestep=True)

            # Optional: Apply timestep shift for flow matching
            # Note: With discrete timesteps, shift is less important
            if self.timestep_shift > 1:
                timestep = (
                    self.timestep_shift * (timestep / 1000) / (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
                )

            timestep = timestep.clamp(self.min_step, self.max_step).long()

            # Step 2: Add noise to generated latent
            noise = torch.randn_like(generated_latent, dtype=torch.float32)
            noisy_latent = self.scheduler.add_noise(
                generated_latent.flatten(0, 1).float(), noise.flatten(0, 1), timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))

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
        # Loss = 0.5 * ||x - (x - grad)||^2 = 0.5 * ||grad||^2
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                generated_latent.double()[gradient_mask],
                (generated_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                generated_latent.double(), (generated_latent.double() - grad.double()).detach(), reduction="mean"
            )

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

        The critic is trained to denoise generated samples, learning to
        score the distribution of student-generated videos.

        Args:
            generated_latent: Generated latents (detached) [B, F, C, H, W]
            text_embeddings: Conditional text embeddings
            denoised_timestep_from: Starting timestep from backward simulation
            denoised_timestep_to: Ending timestep from backward simulation
            global_step: Current training step

        Returns:
            critic_loss: Denoising loss for critic
            log_dict: Logging dictionary
        """
        debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

        batch_size, num_frame = generated_latent.shape[:2]

        # Step 1: Sample timestep for critic
        if self.ts_schedule and denoised_timestep_to is not None:
            min_timestep = denoised_timestep_to
        else:
            min_timestep = self.min_score_timestep

        if self.ts_schedule_max and denoised_timestep_from is not None:
            max_timestep = denoised_timestep_from
        else:
            max_timestep = self.num_train_timestep

        critic_timestep = self._get_timestep(min_timestep, max_timestep, batch_size, num_frame, uniform_timestep=True)

        # Apply timestep shift
        if self.timestep_shift > 1:
            critic_timestep = (
                self.timestep_shift
                * (critic_timestep / 1000)
                / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000))
                * 1000
            )

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        # Step 2: Add noise to generated latent
        critic_noise = torch.randn_like(generated_latent, dtype=torch.float32)
        noisy_generated_latent = self.scheduler.add_noise(
            generated_latent.flatten(0, 1).float(), critic_noise.flatten(0, 1), critic_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

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

        # Step 4: Compute denoising loss
        if self.denoising_loss_type == "flow":
            # Flow matching: v = (x0 - xt) / (1 - sigma)
            # Convert x0 prediction to flow prediction
            flow_pred = self._convert_x0_to_flow_pred(
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_latent.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )

            # Target flow
            target_flow = self._convert_x0_to_flow_pred(
                x0_pred=generated_latent.flatten(0, 1),
                xt=noisy_generated_latent.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )

            critic_loss = F.mse_loss(flow_pred.float(), target_flow.float())

        elif self.denoising_loss_type == "x0":
            # Direct x0 prediction
            critic_loss = F.mse_loss(pred_fake_image.float(), generated_latent.float())

        elif self.denoising_loss_type == "epsilon":
            # Epsilon prediction
            pred_noise = self._convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_latent.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            critic_loss = F.mse_loss(pred_noise.float(), critic_noise.flatten(0, 1).float())

        else:
            raise ValueError(f"Unknown denoising_loss_type: {self.denoising_loss_type}")

        if debug_mode or (global_step % 100 == 0):
            print0(f"[CRITIC LOSS] Step {global_step}")
            print0(f"  Timestep range: [{critic_timestep.min().item():.1f}, {critic_timestep.max().item():.1f}]")
            print0(f"  Critic loss: {critic_loss.item():.6f}")

        log_dict = {"critic_timestep": critic_timestep.detach()}

        return critic_loss, log_dict

    def _convert_x0_to_flow_pred(self, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow prediction.

        Flow matching: v = (x0 - xt) / (1 - sigma)
        where sigma = shift / (shift + (1/t - 1))

        Args:
            x0_pred: Predicted clean latent
            xt: Noisy latent
            timestep: Timestep

        Returns:
            Flow prediction
        """
        # Normalize timestep to [0, 1]
        t_norm = timestep.float() / self.num_train_timestep
        t_norm = torch.clamp(t_norm, min=1e-5, max=1.0)

        # Compute sigma with shift
        sigma = self.timestep_shift / (self.timestep_shift + (1.0 / t_norm - 1.0))
        sigma = sigma.view(-1, 1, 1, 1).to(x0_pred.device)

        # Flow prediction: v = (x0 - xt) / (1 - sigma)
        flow_pred = (x0_pred - xt) / (1.0 - sigma + 1e-8)

        return flow_pred

    def _convert_x0_to_noise(self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 to noise prediction.

        DDPM: xt = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * epsilon
        Therefore: epsilon = (xt - sqrt(alpha_bar) * x0) / sqrt(1 - alpha_bar)

        Args:
            x0: Clean latent
            xt: Noisy latent
            timestep: Timestep

        Returns:
            Noise prediction
        """
        if hasattr(self.scheduler, "alphas_cumprod") and self.scheduler.alphas_cumprod is not None:
            # Ensure timestep indices are valid
            timestep = torch.clamp(timestep, 0, len(self.scheduler.alphas_cumprod) - 1)
            alpha_bar = self.scheduler.alphas_cumprod[timestep].to(x0.device).view(-1, 1, 1, 1)
            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

            noise_pred = (xt - sqrt_alpha_bar * x0) / (sqrt_one_minus_alpha_bar + 1e-8)
        else:
            # Fallback for flow matching scheduler
            noise_pred = xt - x0

        return noise_pred
