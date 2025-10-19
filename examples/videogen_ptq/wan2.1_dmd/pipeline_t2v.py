# pipeline_t2v.py - Self-Forcing Training Pipeline for DMD

import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from dist_utils import print0


class SelfForcingTrainingPipeline:
    """
    Self-forcing training pipeline for DMD.

    Implements consistency backward simulation where we:
    1. Start from noise at timestep T_max
    2. Iteratively denoise through discrete timesteps
    3. RANDOMLY EXIT at one timestep (synchronized across ranks)
    4. Compute gradients ONLY at that exit timestep

    This is the key difference from standard backward simulation:
    - Standard: Compute gradients at ALL timesteps
    - Self-forcing: Compute gradients at ONE random timestep

    Benefits:
    - Much more memory efficient (single gradient computation)
    - Prevents overfitting to specific timesteps
    - Matches inference behavior better
    """

    def __init__(
        self,
        model_name: str,
        denoising_step_list: List[int],
        scheduler,
        generator,
        num_frame_per_block: int = 4,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = True,
        last_step_only: bool = True,
        num_max_frames: int = 21,
        context_noise: int = 0,
    ):
        """
        Initialize self-forcing training pipeline.

        Args:
            model_name: Model name (for config)
            denoising_step_list: Discrete timesteps [999, 749, 499, 249, 0]
            scheduler: Noise scheduler
            generator: FSDP-wrapped generator model
            num_frame_per_block: Frames per block for causal generation
            independent_first_frame: Whether first frame is independent
            same_step_across_blocks: Use same timestep for all blocks
            last_step_only: Only train on last step (most efficient)
            num_max_frames: Maximum number of frames
            context_noise: Noise level for context frames
        """
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.context_noise = context_noise

        # Store denoising steps
        self.denoising_step_list = torch.tensor(denoising_step_list, dtype=torch.long)

        # Remove zero timestep for sampling (we don't denoise at t=0)
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        print0("[PIPELINE] Self-forcing training pipeline initialized")
        print0(f"  - Denoising steps: {denoising_step_list}")
        print0(f"  - Frames per block: {num_frame_per_block}")
        print0(f"  - Same step across blocks: {same_step_across_blocks}")
        print0(f"  - Last step only: {last_step_only}")

    def generate_and_sync_list(self, num_blocks: int, num_denoising_steps: int, device) -> List[int]:
        """
        Generate random timestep indices and synchronize across all ranks.

        CRITICAL: All ranks must use the SAME random timestep for gradient computation.
        Otherwise, FSDP gradient synchronization will fail.

        Args:
            num_blocks: Number of frame blocks
            num_denoising_steps: Number of denoising steps
            device: Device for tensor creation

        Returns:
            List of timestep indices (one per block, or one for all blocks)
        """
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Rank 0 generates random indices
            if self.last_step_only:
                # Always use last step (most stable)
                indices = torch.ones(num_blocks, dtype=torch.long, device=device) * (num_denoising_steps - 1)
            else:
                # Random timestep per block
                indices = torch.randint(low=0, high=num_denoising_steps, size=(num_blocks,), device=device)
        else:
            # Other ranks create empty tensor
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        # Broadcast from rank 0 to all ranks
        if dist.is_initialized():
            dist.broadcast(indices, src=0)

        return indices.tolist()

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        text_embeddings: torch.Tensor,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, int, int, Optional[List[torch.Tensor]]]:
        """
        Run self-forcing backward simulation.

        Process:
        1. Start from pure noise at t_max (e.g., 999)
        2. Iteratively denoise through discrete timesteps WITHOUT gradients
        3. At ONE randomly selected timestep, compute WITH gradients
        4. Re-noise to next timestep and continue WITHOUT gradients
        5. Return final prediction and timestep range

        KEY: Only ONE denoising step has gradients enabled!

        Args:
            noise: Pure noise [B, C, F, H, W]
            text_embeddings: Text conditioning [B, seq_len, hidden_dim]
            clip_fea: Optional CLIP features
            y: Optional VAE features
            initial_latent: Optional initial latent (for I2V)
            global_step: Current training step (for logging)

        Returns:
            generated_latent: Final denoised latent [B, C, F, H, W]
            denoised_timestep_from: Starting timestep
            denoised_timestep_to: Ending timestep
            trajectory: None (not used in self-forcing)
        """
        debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"

        batch_size, num_channels, num_frames, height, width = noise.shape
        device = noise.device

        # Determine number of blocks
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        if debug_mode or (global_step % 100 == 0):
            print0(f"\n[SELF-FORCING] === Step {global_step} ===")
            print0(f"  Noise shape: {noise.shape}")
            print0(f"  Num blocks: {num_blocks}")
            print0(f"  Denoising steps: {self.denoising_step_list.tolist()}")

        # CRITICAL: Select random exit timestep (synchronized across all ranks)
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(num_blocks, num_denoising_steps, device)

        if debug_mode or (global_step % 100 == 0):
            print0(f"  Exit timestep index: {exit_flags[0]} (step {self.denoising_step_list[exit_flags[0]]})")

        # Start from pure noise
        current_latent = noise

        # Iterate through denoising steps
        for index, current_timestep in enumerate(self.denoising_step_list):
            # Check if this is the exit timestep
            if self.same_step_across_blocks:
                exit_flag = index == exit_flags[0]
            else:
                # Different exit for each block (rarely used)
                exit_flag = index in exit_flags

            # Create timestep tensor for generator
            # WAN expects [B] shape, NOT [B, F]
            timestep = torch.ones(batch_size, device=device, dtype=torch.long) * current_timestep

            if not exit_flag:
                # NO GRADIENTS - just keep denoising
                with torch.no_grad():
                    # Generator forward pass
                    output = self.generator(
                        hidden_states=current_latent,
                        timestep=timestep.to(self.generator.dtype),
                        encoder_hidden_states=text_embeddings,
                        return_dict=False,
                    )
                    if isinstance(output, tuple):
                        denoised_pred = output[0]
                    else:
                        denoised_pred = output

                    if debug_mode:
                        print0(f"  [No grad] Step {index}: t={current_timestep}")

                    # Re-noise to next timestep
                    if index < len(self.denoising_step_list) - 1:
                        next_timestep = self.denoising_step_list[index + 1]
                        current_latent = self._add_noise_to_timestep(
                            denoised_pred, next_timestep, batch_size, num_frames, num_channels, height, width
                        )
                    else:
                        current_latent = denoised_pred
            else:
                # WITH GRADIENTS - this is where we compute loss!
                if debug_mode or (global_step % 100 == 0):
                    print0(f"  [WITH GRAD] Step {index}: t={current_timestep} <- TRAINING HERE")

                output = self.generator(
                    hidden_states=current_latent,
                    timestep=timestep.to(self.generator.dtype),
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )
                if isinstance(output, tuple):
                    denoised_pred = output[0]
                else:
                    denoised_pred = output

                # Exit loop - we got our gradient-enabled prediction
                break

        # Compute timestep range for DMD loss
        if exit_flags[0] == len(self.denoising_step_list) - 1:
            # Last step
            denoised_timestep_to = 0
            denoised_timestep_from = self._convert_to_scheduler_timestep(self.denoising_step_list[exit_flags[0]])
        else:
            # Intermediate step
            denoised_timestep_to = self._convert_to_scheduler_timestep(self.denoising_step_list[exit_flags[0] + 1])
            denoised_timestep_from = self._convert_to_scheduler_timestep(self.denoising_step_list[exit_flags[0]])

        if debug_mode or (global_step % 100 == 0):
            print0(f"  Final latent range: [{denoised_pred.min():.3f}, {denoised_pred.max():.3f}]")
            print0(f"  Timestep range: [{denoised_timestep_from} -> {denoised_timestep_to}]")

        return denoised_pred, denoised_timestep_from, denoised_timestep_to, None

    def _add_noise_to_timestep(
        self,
        clean_latent: torch.Tensor,
        target_timestep: int,
        batch_size: int,
        num_frames: int,
        num_channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Add noise to clean latent to reach target timestep.

        Args:
            clean_latent: Clean latent [B, C, F, H, W]
            target_timestep: Target noise level
            batch_size, num_frames, num_channels, height, width: Tensor dimensions

        Returns:
            Noisy latent at target timestep [B, C, F, H, W]
        """
        # Generate noise
        noise = torch.randn_like(clean_latent, dtype=torch.float32)

        # Permute for scheduler: [B, C, F, H, W] -> [B, F, C, H, W] -> [B*F, C, H, W]
        clean_permuted = clean_latent.permute(0, 2, 1, 3, 4)
        noise_permuted = noise.permute(0, 2, 1, 3, 4)

        clean_flat = clean_permuted.reshape(batch_size * num_frames, num_channels, height, width)
        noise_flat = noise_permuted.reshape(batch_size * num_frames, num_channels, height, width)

        # Create timestep tensor [B*F]
        timestep_flat = (
            torch.ones(batch_size * num_frames, device=clean_latent.device, dtype=torch.long) * target_timestep
        )

        # Add noise (in fp32 for stability)
        noisy_flat = self.scheduler.add_noise(clean_flat.float(), noise_flat, timestep_flat)

        # Unflatten: [B*F, C, H, W] -> [B, F, C, H, W] -> [B, C, F, H, W]
        noisy_unflat = noisy_flat.reshape(batch_size, num_frames, num_channels, height, width)
        noisy_latent = noisy_unflat.permute(0, 2, 1, 3, 4)

        return noisy_latent

    def _convert_to_scheduler_timestep(self, discrete_step: int) -> int:
        """
        Convert discrete step to scheduler timestep.

        Discrete steps: [999, 749, 499, 249]
        Scheduler timesteps: [0, 1, 2, ..., 999]

        Args:
            discrete_step: Discrete timestep value

        Returns:
            Scheduler timestep index
        """
        # Find closest scheduler timestep
        timestep_idx = (
            1000
            - torch.argmin(
                (
                    self.scheduler.timesteps.to(
                        discrete_step.device if isinstance(discrete_step, torch.Tensor) else "cpu"
                    )
                    - discrete_step
                ).abs(),
                dim=0,
            ).item()
        )
        return timestep_idx


# Backward compatibility - keep old name as alias
BackwardSimulationPipeline = SelfForcingTrainingPipeline
