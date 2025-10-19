# rectified_flow_scheduler.py - Custom scheduler for WAN 2.1 DMD training

from typing import Optional, Union

import torch


class RectifiedFlowScheduler:
    """
    Rectified Flow scheduler for WAN 2.1 T2V.

    WAN 2.1 uses Rectified Flow (RF) with timestep shifting.
    This scheduler implements proper noise scaling to prevent explosion.

    Key features:
    - Shift parameter (default 3.0 for WAN)
    - Proper sigma computation
    - Stable noise addition
    - Compatible with DMD training
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_discrete_timesteps: bool = True,
    ):
        """
        Initialize Rectified Flow scheduler.

        Args:
            num_train_timesteps: Total training timesteps (1000)
            shift: Timestep shift parameter (3.0 for WAN)
            use_discrete_timesteps: Use discrete timesteps for DMD
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_discrete_timesteps = use_discrete_timesteps

        # Initialize timesteps [0, 1, 2, ..., 999]
        self.timesteps = torch.arange(0, num_train_timesteps, dtype=torch.long)

        # Precompute sigmas for efficiency
        self._precompute_sigmas()

        print("[SCHEDULER] Rectified Flow initialized:")
        print(f"  - Num timesteps: {num_train_timesteps}")
        print(f"  - Shift: {shift}")
        print(f"  - Discrete timesteps: {use_discrete_timesteps}")

    def _precompute_sigmas(self):
        """Precompute sigma values for all timesteps."""
        # Normalize timesteps to [0, 1]
        t_norm = self.timesteps.float() / self.num_train_timesteps
        t_norm = torch.clamp(t_norm, min=1e-7, max=1.0 - 1e-7)

        # Compute sigma with shift
        # sigma(t) = shift / (shift + (1/t - 1))
        self.sigmas = self.shift / (self.shift + (1.0 / (t_norm + 1e-7) - 1.0))
        self.sigmas = torch.clamp(self.sigmas, min=0.0, max=1.0)

    def get_sigma(self, timestep: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Get sigma value for given timestep(s).

        Args:
            timestep: Timestep(s) as int or tensor

        Returns:
            Sigma value(s) in range [0, 1]
        """
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], dtype=torch.long)

        # Clamp to valid range
        timestep = torch.clamp(timestep, 0, self.num_train_timesteps - 1)

        # Normalize to [0, 1]
        t_norm = timestep.float() / self.num_train_timesteps
        t_norm = torch.clamp(t_norm, min=1e-7, max=1.0 - 1e-7)

        # Compute sigma
        sigma = self.shift / (self.shift + (1.0 / (t_norm + 1e-7) - 1.0))
        sigma = torch.clamp(sigma, min=0.0, max=1.0)

        return sigma

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples according to Rectified Flow.

        For Rectified Flow:
            x_t = (1 - sigma(t)) * x_0 + sigma(t) * noise

        where sigma(t) goes from 0 (clean) to 1 (pure noise).

        Args:
            original_samples: Original samples (x_0) [B*F, C, H, W]
            noise: Noise tensor (same shape as original_samples)
            timesteps: Timesteps [B*F]

        Returns:
            Noisy samples (x_t)
        """
        # Ensure inputs are float32 for numerical stability
        original_samples = original_samples.float()
        noise = noise.float()

        # Get sigma values
        sigmas = self.get_sigma(timesteps).to(original_samples.device)

        # Reshape for broadcasting [B*F] -> [B*F, 1, 1, 1]
        while sigmas.ndim < original_samples.ndim:
            sigmas = sigmas.unsqueeze(-1)

        # Rectified Flow interpolation
        # x_t = (1 - sigma) * x_0 + sigma * noise
        noisy_samples = (1.0 - sigmas) * original_samples + sigmas * noise

        # CRITICAL: Clamp to prevent extreme values
        noisy_samples = torch.clamp(noisy_samples, min=-20.0, max=20.0)

        return noisy_samples

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity target for Rectified Flow.

        For Rectified Flow, the velocity is simply:
            v = x_0 - noise

        This is constant throughout time (rectified!).

        Args:
            sample: Clean sample (x_0)
            noise: Noise
            timesteps: Timesteps (not used, but kept for API compatibility)

        Returns:
            Velocity target
        """
        # For Rectified Flow, velocity is constant
        velocity = sample - noise
        return velocity

    def predict_x0_from_velocity(
        self,
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from velocity prediction.

        Given: x_t = (1 - sigma) * x_0 + sigma * noise
               v = x_0 - noise

        We can solve for x_0:
            x_0 = x_t + sigma * v

        Args:
            x_t: Noisy sample
            velocity: Predicted velocity
            timestep: Current timestep

        Returns:
            Predicted x_0
        """
        sigmas = self.get_sigma(timestep).to(x_t.device)

        # Reshape for broadcasting
        while sigmas.ndim < x_t.ndim:
            sigmas = sigmas.unsqueeze(-1)

        # x_0 = x_t + sigma * v
        x_0 = x_t + sigmas * velocity

        # Clamp output
        x_0 = torch.clamp(x_0, min=-10.0, max=10.0)

        return x_0

    def predict_noise_from_x0(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise from x_0 prediction.

        Given: x_t = (1 - sigma) * x_0 + sigma * noise

        Solve for noise:
            noise = (x_t - (1 - sigma) * x_0) / sigma

        Args:
            x_t: Noisy sample
            x_0: Predicted clean sample
            timestep: Current timestep

        Returns:
            Predicted noise
        """
        sigmas = self.get_sigma(timestep).to(x_t.device)

        # Reshape for broadcasting
        while sigmas.ndim < x_t.ndim:
            sigmas = sigmas.unsqueeze(-1)

        # Avoid division by zero
        sigmas = torch.clamp(sigmas, min=1e-3)

        # noise = (x_t - (1 - sigma) * x_0) / sigma
        noise = (x_t - (1.0 - sigmas) * x_0) / sigmas

        # Clamp output
        noise = torch.clamp(noise, min=-10.0, max=10.0)

        return noise

    def get_velocity_from_x0_pred(
        self,
        x_t: torch.Tensor,
        x_0_pred: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert x_0 prediction to velocity.

        This is the KEY function for DMD loss computation.

        Given: x_t = (1 - sigma) * x_0 + sigma * noise
        And: v = x_0 - noise

        We can derive:
            v = (x_0 - x_t) / (1 - sigma) * (1/sigma) + x_0

        But simpler (and more stable):
            noise = (x_t - (1 - sigma) * x_0) / sigma
            v = x_0 - noise

        Args:
            x_t: Noisy sample
            x_0_pred: Predicted x_0
            timestep: Current timestep

        Returns:
            Velocity prediction
        """
        # Get noise prediction
        noise_pred = self.predict_noise_from_x0(x_t, x_0_pred, timestep)

        # Compute velocity
        velocity = x_0_pred - noise_pred

        # Clamp to prevent explosion
        velocity = torch.clamp(velocity, min=-50.0, max=50.0)

        return velocity

    def set_timesteps(self, num_inference_steps: Optional[int] = None):
        """
        Set timesteps for inference.

        Args:
            num_inference_steps: Number of inference steps
        """
        if num_inference_steps is not None:
            # Create evenly spaced timesteps
            self.timesteps = torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=torch.long)
        else:
            self.timesteps = torch.arange(0, self.num_train_timesteps, dtype=torch.long)


# Convenience function to create scheduler
def create_wan_scheduler(shift: float = 3.0) -> RectifiedFlowScheduler:
    """Create WAN 2.1 compatible Rectified Flow scheduler."""
    return RectifiedFlowScheduler(
        num_train_timesteps=1000,
        shift=shift,
        use_discrete_timesteps=True,
    )
