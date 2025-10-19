# pipeline_t2v.py - Simple DMD Pipeline (No Self-Forcing)

import torch
from typing import Optional, Tuple


class SimpleDMDPipeline:
    """
    Simple DMD pipeline for single-step training.
    
    At each training step:
    1. Sample random timestep t
    2. Student predicts x0 from noise
    3. Add noise to x0 to get x_t
    4. Teacher and critic predict at x_t
    """
    
    def __init__(
        self,
        scheduler,
        student_model,
    ):
        """
        Initialize simple DMD pipeline.
        
        Args:
            scheduler: Noise scheduler (RectifiedFlowScheduler)
            student_model: Student transformer (FSDP-wrapped)
        """
        self.scheduler = scheduler
        self.student = student_model
        
    def forward_one_step(
        self,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step forward for DMD training.
        
        Args:
            noise: Pure noise [B, C, F, H, W]
            timestep: Timestep [B] (int64)
            text_embeddings: Text conditioning [B, L, D]
            
        Returns:
            x_t: Noisy latent at timestep t [B, C, F, H, W]
            student_x0: Student's x0 prediction [B, C, F, H, W]
        """
        batch_size, num_channels, num_frames, height, width = noise.shape
        
        # Step 1: Student predicts x0 from noise at timestep t
        # For DMD, we predict x0 directly (not epsilon or velocity)
        student_x0 = self.student(
            hidden_states=noise,
            timestep=timestep.to(self.student.dtype),
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )
        if isinstance(student_x0, tuple):
            student_x0 = student_x0[0]
            
        # Step 2: Add noise to student's x0 prediction to get x_t
        # This creates the noisy latent that teacher/critic will evaluate
        x_t = self._add_noise_to_x0(
            student_x0,
            timestep,
            batch_size,
            num_frames,
            num_channels,
            height,
            width,
        )
        
        return x_t, student_x0
    
    def _add_noise_to_x0(
        self,
        x0: torch.Tensor,
        timestep: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Add noise to x0 to create x_t at given timestep.
        
        Args:
            x0: Clean prediction [B, C, F, H, W]
            timestep: Timestep [B]
            batch_size, num_frames, etc: Tensor dimensions
            
        Returns:
            x_t: Noisy latent [B, C, F, H, W]
        """
        # Generate fresh noise
        noise = torch.randn_like(x0, dtype=torch.float32)
        
        # Reshape for scheduler: [B, C, F, H, W] -> [B*F, C, H, W]
        x0_permuted = x0.permute(0, 2, 1, 3, 4)
        noise_permuted = noise.permute(0, 2, 1, 3, 4)
        
        x0_flat = x0_permuted.reshape(batch_size * num_frames, num_channels, height, width)
        noise_flat = noise_permuted.reshape(batch_size * num_frames, num_channels, height, width)
        
        # Expand timestep: [B] -> [B*F]
        timestep_flat = timestep.unsqueeze(1).repeat(1, num_frames).flatten()
        
        # Add noise using scheduler
        x_t_flat = self.scheduler.add_noise(x0_flat.float(), noise_flat, timestep_flat)
        
        # Reshape back: [B*F, C, H, W] -> [B, C, F, H, W]
        x_t_unflat = x_t_flat.reshape(batch_size, num_frames, num_channels, height, width)
        x_t = x_t_unflat.permute(0, 2, 1, 3, 4)
        
        return x_t.to(x0.dtype)


# Helper function to create pipeline
def create_simple_dmd_pipeline(scheduler, student_model):
    """
    Create SimpleDMDPipeline instance.
    
    Args:
        scheduler: Noise scheduler (RectifiedFlowScheduler)
        student_model: Student transformer (FSDP-wrapped)
        
    Returns:
        SimpleDMDPipeline instance
    """
    return SimpleDMDPipeline(
        scheduler=scheduler,
        student_model=student_model,
    )