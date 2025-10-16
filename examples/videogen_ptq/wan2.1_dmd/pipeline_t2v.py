# pipeline_t2v.py - Backward Simulation Pipeline for DMD Training (FIXED)

import os
from typing import Tuple

import torch
from dist_utils import print0


class BackwardSimulationPipeline:
    """
    Backward simulation pipeline for DMD training.

    This pipeline generates synthetic training samples by running multi-step
    inference during training, eliminating the need for real video data.

    Key features:
    - Consistency sampling with discrete timesteps
    - Returns denoising trajectory for loss computation
    - Memory-efficient implementation
    """

    def __init__(self, scheduler, generator_model, denoising_step_list, device, bf16, last_step_only: bool = True):
        """
        Initialize backward simulation pipeline.

        Args:
            scheduler: Diffusion scheduler
            generator_model: FSDP-wrapped generator model
            denoising_step_list: List of discrete timesteps [999, 749, 499, 249, 0]
            device: Training device
            bf16: BFloat16 dtype
            last_step_only: Only return last step (memory efficient)
        """
        self.scheduler = scheduler
        self.generator_model = generator_model
        self.denoising_step_list = torch.tensor(denoising_step_list, dtype=torch.long, device=device)
        self.device = device
        self.bf16 = bf16
        self.last_step_only = last_step_only

        print0("[PIPELINE] Initialized backward simulation")
        print0(f"  - Denoising steps: {denoising_step_list}")
        print0(f"  - Last step only: {last_step_only}")

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Run backward simulation (consistency sampling) to generate synthetic samples.

        This implements the consistency sampler from the DMD paper, which:
        1. Starts from pure noise
        2. Iteratively denoises through discrete timesteps
        3. Returns the final generated latent

        Args:
            noise: Pure noise [B, C, F, H, W]
            text_embeddings: Text conditioning [B, seq_len, hidden_dim]

        Returns:
            generated_latent: Final generated latent [B, C, F, H, W]
            denoised_timestep_from: Starting timestep (for dynamic scheduling)
            denoised_timestep_to: Ending timestep (for dynamic scheduling)
        """
        debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

        batch_size = noise.shape[0]
        num_channels = noise.shape[1]
        num_frame = noise.shape[2]
        height = noise.shape[3]
        width = noise.shape[4]

        if debug_mode:
            print0("[PIPELINE] Starting backward simulation")
            print0(f"  Noise shape: {noise.shape}")
            print0(f"  Text embeddings shape: {text_embeddings.shape}")

        # Start from pure noise at max timestep
        current_latent = noise

        # Iterate through denoising steps
        for i, timestep_value in enumerate(self.denoising_step_list):
            if timestep_value == 0:
                # Last step - no denoising needed
                break

            # CRITICAL: Timestep should be a scalar tensor or 1D tensor of shape [B]
            # NOT [B, F] as the transformer will broadcast it internally
            timestep = torch.full(
                (batch_size,),  # Shape: [B] not [B, F]
                timestep_value,
                device=self.device,
                dtype=torch.long,
            )

            if debug_mode:
                print0(f"[PIPELINE] Step {i}: timestep={timestep_value}")
                print0(f"  Current latent shape: {current_latent.shape}")
                print0(f"  Timestep shape: {timestep.shape}")

            # Generator forward pass
            # WAN transformer expects [B, C, F, H, W] format directly
            with torch.no_grad():
                pred_x0 = self.generator_model(
                    hidden_states=current_latent.to(self.bf16),
                    timestep=timestep.to(self.bf16),
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )
                if isinstance(pred_x0, tuple):
                    pred_x0 = pred_x0[0]

            if debug_mode:
                print0(f"  Predicted x0 shape: {pred_x0.shape}")

            # Get next timestep
            if i < len(self.denoising_step_list) - 1:
                next_timestep_value = self.denoising_step_list[i + 1]

                if next_timestep_value > 0:
                    # Add noise to predicted x0 for next step
                    # For scheduler, we need [B*F] timesteps
                    next_timestep = torch.full(
                        (batch_size * num_frame,),  # [B*F]
                        next_timestep_value,
                        device=self.device,
                        dtype=torch.long,
                    )

                    noise_to_add = torch.randn_like(pred_x0, dtype=torch.float32)

                    # CRITICAL FIX: Proper flattening for scheduler
                    # Scheduler expects [B*F, C, H, W]
                    # pred_x0 is [B, C, F, H, W] -> permute to [B, F, C, H, W] -> flatten to [B*F, C, H, W]
                    pred_x0_permuted = pred_x0.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    noise_permuted = noise_to_add.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

                    pred_x0_flat = pred_x0_permuted.reshape(batch_size * num_frame, num_channels, height, width)
                    noise_flat = noise_permuted.reshape(batch_size * num_frame, num_channels, height, width)

                    if debug_mode:
                        print0("  Adding noise:")
                        print0(f"    pred_x0_flat shape: {pred_x0_flat.shape}")
                        print0(f"    noise_flat shape: {noise_flat.shape}")
                        print0(f"    next_timestep shape: {next_timestep.shape}")

                    current_latent_flat = self.scheduler.add_noise(
                        pred_x0_flat.float(),
                        noise_flat,
                        next_timestep,  # [B*F]
                    )

                    # Unflatten back to [B, F, C, H, W] then permute to [B, C, F, H, W]
                    current_latent_unflat = current_latent_flat.reshape(
                        batch_size, num_frame, num_channels, height, width
                    )
                    current_latent = current_latent_unflat.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                else:
                    # Next step is 0 - use predicted x0 directly
                    current_latent = pred_x0
            else:
                # Last step
                current_latent = pred_x0

        # Return final generated latent and timestep range
        denoised_timestep_from = self.denoising_step_list[0].item()
        denoised_timestep_to = self.denoising_step_list[-1].item()

        if debug_mode:
            print0("[PIPELINE] Backward simulation complete")
            print0(f"  Final latent shape: {current_latent.shape}")

        return current_latent, denoised_timestep_from, denoised_timestep_to


class StochasticBackwardSimulationPipeline(BackwardSimulationPipeline):
    """
    Stochastic backward simulation with random timestep selection.

    Instead of always denoising through all steps, randomly select one
    timestep per training sample for gradient computation. This is more
    memory efficient and trains the model on all timesteps uniformly.
    """

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Run stochastic backward simulation.

        For each frame, randomly select one denoising timestep and compute
        the model's prediction at that timestep. This makes training more
        efficient and covers all timesteps uniformly.

        Args:
            noise: Pure noise [B, C, F, H, W]
            text_embeddings: Text conditioning [B, seq_len, hidden_dim]

        Returns:
            generated_latent: Generated latent at random timestep [B, C, F, H, W]
            denoised_timestep_from: Selected timestep
            denoised_timestep_to: Next timestep
        """
        batch_size = noise.shape[0]
        num_channels = noise.shape[1]
        num_frame = noise.shape[2]
        height = noise.shape[3]
        width = noise.shape[4]

        # Randomly select timestep index for each sample in batch
        timestep_indices = torch.randint(
            0,
            len(self.denoising_step_list) - 1,  # Exclude last step (0)
            (batch_size,),
            device=self.device,
        )

        # Get corresponding timestep values
        selected_timesteps = self.denoising_step_list[timestep_indices]

        # For transformer: timestep is [B]
        timestep = selected_timesteps

        # For scheduler: timestep is [B*F]
        timestep_scheduler = selected_timesteps.unsqueeze(1).repeat(1, num_frame).flatten()

        # Add noise to input according to selected timestep
        # Permute and reshape for scheduler
        noise_permuted = noise.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        noise_flat = noise_permuted.reshape(batch_size * num_frame, num_channels, height, width)

        noisy_input_flat = self.scheduler.add_noise(
            torch.zeros_like(noise_flat),  # Start from zero (will be x0 later)
            noise_flat,
            timestep_scheduler,  # [B*F]
        )

        # Unflatten and permute back to [B, C, F, H, W]
        noisy_input_unflat = noisy_input_flat.reshape(batch_size, num_frame, num_channels, height, width)
        noisy_input = noisy_input_unflat.permute(0, 2, 1, 3, 4)

        # Generator forward pass
        # WAN transformer expects [B, C, F, H, W] format and timestep [B]
        with torch.no_grad():
            pred_x0 = self.generator_model(
                hidden_states=noisy_input.to(self.bf16),
                timestep=timestep.to(self.bf16),  # [B] not [B, F]
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )
            if isinstance(pred_x0, tuple):
                pred_x0 = pred_x0[0]

        # Return predicted x0 and timestep range
        # Use min/max for dynamic scheduling
        denoised_timestep_from = selected_timesteps.max().item()
        denoised_timestep_to = selected_timesteps.min().item()

        return pred_x0, denoised_timestep_from, denoised_timestep_to
