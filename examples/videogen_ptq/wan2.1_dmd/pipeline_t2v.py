# pipeline_t2v.py - Backward Simulation Pipeline for DMD Training

import torch
from dist_utils import print0
from typing import Tuple


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
    
    def __init__(
        self,
        scheduler,
        generator_model,
        denoising_step_list,
        device,
        bf16,
        last_step_only: bool = True
    ):
        """
        Initialize backward simulation pipeline.
        
        Args:
            scheduler: Diffusion scheduler
            generator_model: FSDP-wrapped generator model
            denoising_step_list: List of discrete timesteps [1000, 750, 500, 250, 0]
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
        
        print0(f"[PIPELINE] Initialized backward simulation")
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
            noise: Pure noise [B, F, C, H, W]
            text_embeddings: Text conditioning
        
        Returns:
            generated_latent: Final generated latent [B, F, C, H, W]
            denoised_timestep_from: Starting timestep (for dynamic scheduling)
            denoised_timestep_to: Ending timestep (for dynamic scheduling)
        """
        batch_size, num_frame = noise.shape[:2]
        
        # Start from pure noise at max timestep
        current_latent = noise
        
        # Iterate through denoising steps
        for i, timestep_value in enumerate(self.denoising_step_list):
            if timestep_value == 0:
                # Last step - no denoising needed
                break
            
            # Create timestep tensor [B, F]
            timestep = torch.full(
                (batch_size, num_frame),
                timestep_value,
                device=self.device,
                dtype=torch.long
            )
            
            # Generator forward pass
            with torch.no_grad():
                pred_x0 = self.generator_model(
                    hidden_states=current_latent.to(self.bf16),
                    timestep=timestep.to(self.bf16),
                    encoder_hidden_states=text_embeddings,
                    return_dict=False
                )
                if isinstance(pred_x0, tuple):
                    pred_x0 = pred_x0[0]
            
            # Get next timestep
            if i < len(self.denoising_step_list) - 1:
                next_timestep_value = self.denoising_step_list[i + 1]
                
                if next_timestep_value > 0:
                    # Add noise to predicted x0 for next step
                    next_timestep = torch.full(
                        (batch_size, num_frame),
                        next_timestep_value,
                        device=self.device,
                        dtype=torch.long
                    )
                    
                    noise_to_add = torch.randn_like(pred_x0, dtype=torch.float32)
                    current_latent = self.scheduler.add_noise(
                        pred_x0.flatten(0, 1).float(),
                        noise_to_add.flatten(0, 1),
                        next_timestep.flatten(0, 1)
                    ).unflatten(0, (batch_size, num_frame))
                else:
                    # Next step is 0 - use predicted x0 directly
                    current_latent = pred_x0
            else:
                # Last step
                current_latent = pred_x0
        
        # Return final generated latent and timestep range
        denoised_timestep_from = self.denoising_step_list[0].item()
        denoised_timestep_to = self.denoising_step_list[-1].item()
        
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
            noise: Pure noise [B, F, C, H, W]
            text_embeddings: Text conditioning
        
        Returns:
            generated_latent: Generated latent at random timestep [B, F, C, H, W]
            denoised_timestep_from: Selected timestep
            denoised_timestep_to: Next timestep
        """
        batch_size, num_frame = noise.shape[:2]
        
        # Randomly select timestep index for each sample in batch
        timestep_indices = torch.randint(
            0,
            len(self.denoising_step_list) - 1,  # Exclude last step (0)
            (batch_size,),
            device=self.device
        )
        
        # Get corresponding timestep values
        selected_timesteps = self.denoising_step_list[timestep_indices]
        
        # Create timestep tensor [B, F]
        timestep = selected_timesteps.unsqueeze(1).repeat(1, num_frame)
        
        # Add noise to input according to selected timestep
        noisy_input = self.scheduler.add_noise(
            torch.zeros_like(noise).flatten(0, 1),  # Start from zero (will be x0 later)
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        
        # Generator forward pass
        with torch.no_grad():
            pred_x0 = self.generator_model(
                hidden_states=noisy_input.to(self.bf16),
                timestep=timestep.to(self.bf16),
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )
            if isinstance(pred_x0, tuple):
                pred_x0 = pred_x0[0]
        
        # Return predicted x0 and timestep range
        # Use min/max for dynamic scheduling
        denoised_timestep_from = selected_timesteps.max().item()
        denoised_timestep_to = selected_timesteps.min().item()
        
        return pred_x0, denoised_timestep_from, denoised_timestep_to