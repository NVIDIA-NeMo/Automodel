# dmd_t2v.py - Pure DMD (No Self-Forcing)

import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from dist_utils import print0


class DMDT2V:
    """
    Pure DMD (Distribution Matching Distillation) for WAN 2.1 T2V.
    
    Components:
    1. Student (generator) - trainable, learns to match teacher
    2. Teacher (real_score) - frozen, provides target distribution
    3. Critic (fake_score) - trainable, learns to distinguish real vs fake
    
    Training loop per step:
    1. Sample random timestep t
    2. Student predicts x0 from noise
    3. Create x_t by adding noise to student's x0
    4. Teacher predicts x0 at x_t (frozen, detached)
    5. Critic predicts x0 at x_t (trainable)
    6. Update critic to match teacher
    7. Update student using (critic - teacher) as direction
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
        loss_weight_type: str = "constant",  # "constant" or "sigma"
        loss_weight_scale: float = 1.0,
    ):
        """Initialize pure DMD trainer."""
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
        self.loss_weight_type = loss_weight_type
        self.loss_weight_scale = loss_weight_scale
        
        # Verify teacher is frozen
        self._verify_teacher_frozen()
        
        print0("[DMD] Initialized pure DMD:")
        print0(f"  - Real guidance scale: {real_guidance_scale}")
        print0(f"  - Fake guidance scale: {fake_guidance_scale}")
        print0(f"  - Timestep range: [{min_step}, {max_step}]")
        print0(f"  - Loss weight type: {loss_weight_type}")
    
    def _verify_teacher_frozen(self):
        """Verify teacher is completely frozen."""
        teacher = self.model_map["real_score"]["fsdp_transformer"]
        
        trainable_count = sum(1 for p in teacher.parameters() if p.requires_grad)
        
        if trainable_count > 0:
            raise RuntimeError(
                f"[BUG] Teacher has {trainable_count} trainable parameters! "
                "Teacher MUST be frozen for DMD."
            )
        
        print0("[DMD] ✓ Teacher verified frozen")
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample random timesteps for training.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Timesteps [B] in range [min_step, max_step]
        """
        timesteps = torch.randint(
            self.min_step,
            self.max_step + 1,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        return timesteps
    
    def get_loss_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute loss weights based on timestep/sigma.
        
        Args:
            timesteps: Timesteps [B]
            
        Returns:
            Weights [B, 1, 1, 1, 1] for broadcasting
        """
        if self.loss_weight_type == "sigma":
            # Weight by noise level (more noise = higher weight)
            sigmas = self.scheduler.get_sigma(timesteps)
            weights = 1.0 + self.loss_weight_scale * sigmas
        else:
            # Constant weight
            weights = torch.ones_like(timesteps, dtype=torch.float32)
        
        # Reshape for broadcasting: [B] -> [B, 1, 1, 1, 1]
        weights = weights.view(-1, 1, 1, 1, 1)
        return weights
    
    def compute_critic_loss(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute critic loss: train critic to match teacher.
        
        Args:
            x_t: Noisy latent [B, C, F, H, W] (DETACHED from student)
            timesteps: Timesteps [B]
            text_embeddings: Positive conditioning [B, L, D]
            negative_text_embeddings: Negative conditioning [B, L, D]
            global_step: Current step (for logging)
            
        Returns:
            critic_loss: Scalar loss
            metrics: Dict of metrics
        """
        debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"
        
        # Ensure x_t is detached (no gradients to student)
        x_t = x_t.detach()
        
        # Step 1: Get teacher prediction (frozen, no gradients)
        with torch.no_grad():
            teacher_x0 = self._get_teacher_prediction(
                x_t, timesteps, text_embeddings, negative_text_embeddings
            )
        
        # Step 2: Get critic prediction (with gradients)
        critic_x0 = self._get_critic_prediction(
            x_t, timesteps, text_embeddings, negative_text_embeddings
        )
        
        # Step 3: Compute weighted MSE loss
        weights = self.get_loss_weight(timesteps)
        critic_loss = ((critic_x0 - teacher_x0) ** 2 * weights).mean()
        
        if debug_mode or (global_step % 100 == 0):
            print0(f"[CRITIC] Step {global_step}")
            print0(f"  Timesteps: {timesteps.tolist()}")
            print0(f"  Loss: {critic_loss.item():.6f}")
        
        metrics = {
            "critic_loss_value": critic_loss.item(),
            "timestep_mean": timesteps.float().mean().item(),
        }
        
        return critic_loss, metrics
    
    def compute_student_dmd_loss(
        self,
        student_x0: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute DMD student loss: use (critic - teacher) as direction.
        
        CRITICAL: Critic and teacher outputs are DETACHED.
        Only student has gradients enabled.
        
        Args:
            student_x0: Student's x0 prediction [B, C, F, H, W] (WITH gradients)
            x_t: Noisy latent [B, C, F, H, W] (created from student_x0)
            timesteps: Timesteps [B]
            text_embeddings: Positive conditioning [B, L, D]
            negative_text_embeddings: Negative conditioning [B, L, D]
            global_step: Current step (for logging)
            
        Returns:
            dmd_loss: Scalar loss
            metrics: Dict of metrics
        """
        debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"
        
        # Step 1: Get teacher and critic predictions (DETACHED)
        with torch.no_grad():
            teacher_x0 = self._get_teacher_prediction(
                x_t, timesteps, text_embeddings, negative_text_embeddings
            )
            critic_x0 = self._get_critic_prediction(
                x_t, timesteps, text_embeddings, negative_text_embeddings
            )
            
            # Compute DMD direction: (critic - teacher)
            dmd_direction = critic_x0 - teacher_x0
        
        # Step 2: Apply direction to student
        # Loss: move student toward (student - direction)
        # This is equivalent to: ∇_θ = (critic - teacher)^T · ∂x_t/∂θ
        weights = self.get_loss_weight(timesteps)
        dmd_loss = 0.5 * ((student_x0 - (student_x0.detach() - dmd_direction)) ** 2 * weights).mean()
        
        if debug_mode or (global_step % 100 == 0):
            print0(f"[STUDENT] Step {global_step}")
            print0(f"  DMD direction norm: {dmd_direction.abs().mean().item():.6f}")
            print0(f"  Loss: {dmd_loss.item():.6f}")
        
        metrics = {
            "dmd_loss_value": dmd_loss.item(),
            "dmd_direction_norm": dmd_direction.abs().mean().item(),
        }
        
        return dmd_loss, metrics
    
    def _get_teacher_prediction(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Get teacher's x0 prediction with CFG."""
        teacher = self.model_map["real_score"]["fsdp_transformer"]
        
        # Conditional prediction
        pred_cond = teacher(
            hidden_states=x_t,
            timestep=timesteps.to(self.bf16),
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )
        if isinstance(pred_cond, tuple):
            pred_cond = pred_cond[0]
        
        # Unconditional prediction
        pred_uncond = teacher(
            hidden_states=x_t,
            timestep=timesteps.to(self.bf16),
            encoder_hidden_states=negative_text_embeddings,
            return_dict=False,
        )
        if isinstance(pred_uncond, tuple):
            pred_uncond = pred_uncond[0]
        
        # Apply CFG
        pred_x0 = pred_uncond + self.real_guidance_scale * (pred_cond - pred_uncond)
        
        return pred_x0
    
    def _get_critic_prediction(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Get critic's x0 prediction with optional CFG."""
        critic = self.model_map["fake_score"]["fsdp_transformer"]
        
        # Conditional prediction
        pred_cond = critic(
            hidden_states=x_t,
            timestep=timesteps.to(self.bf16),
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )
        if isinstance(pred_cond, tuple):
            pred_cond = pred_cond[0]
        
        # Optional CFG for critic (usually 0.0)
        if abs(self.fake_guidance_scale) > 1e-6:
            pred_uncond = critic(
                hidden_states=x_t,
                timestep=timesteps.to(self.bf16),
                encoder_hidden_states=negative_text_embeddings,
                return_dict=False,
            )
            if isinstance(pred_uncond, tuple):
                pred_uncond = pred_uncond[0]
            
            pred_x0 = pred_uncond + self.fake_guidance_scale * (pred_cond - pred_uncond)
        else:
            pred_x0 = pred_cond
        
        return pred_x0