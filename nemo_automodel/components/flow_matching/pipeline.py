# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlowMatching Pipeline - Model-agnostic implementation with adapter pattern.

This module provides a unified FlowMatchingPipeline class that is completely
independent of specific model implementations through the ModelAdapter abstraction.

Features:
- Model-agnostic design via ModelAdapter protocol
- Various timestep sampling strategies (uniform, logit_normal, mode, lognorm)
- Flow shift transformation
- Sigma clamping for finetuning
- Loss weighting
- Detailed training logging
"""

import logging
import math
import os
import random
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# Import adapters from the adapters module
from .adapters import (
    FlowMatchingContext,
    FluxAdapter,
    HunyuanAdapter,
    ModelAdapter,
    SimpleAdapter,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Noise Schedule
# =============================================================================


class LinearInterpolationSchedule:
    """Simple linear interpolation schedule for flow matching."""

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation: x_t = (1 - σ) * x_0 + σ * x_1

        Args:
            x0: Starting point (clean latents)
            x1: Ending point (noise)
            sigma: Sigma values in [0, 1]

        Returns:
            Interpolated tensor at sigma
        """
        sigma = sigma.view(-1, *([1] * (x0.ndim - 1)))
        return (1.0 - sigma) * x0 + sigma * x1


# =============================================================================
# Flow Matching Pipeline
# =============================================================================


class FlowMatchingPipeline:
    """
    Flow Matching Pipeline - Model-agnostic implementation.

    This pipeline handles all flow matching training logic while delegating
    model-specific operations to a ModelAdapter. This allows adding support
    for new model architectures without modifying the pipeline code.

    Features:
    - Noise scheduling with linear interpolation
    - Timestep sampling with various strategies
    - Flow shift transformation
    - Sigma clamping for finetuning
    - Loss weighting
    - Detailed training logging

    Example:
        # Create pipeline with HunyuanVideo adapter
        from automodel.flow_matching.adapters import HunyuanAdapter

        pipeline = FlowMatchingPipeline(
            model_adapter=HunyuanAdapter(),
            flow_shift=3.0,
            timestep_sampling="logit_normal",
        )

        # Training step
        weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(model, batch, device, dtype, global_step)
    """

    def __init__(
        self,
        model_adapter: ModelAdapter,
        num_train_timesteps: int = 1000,
        timestep_sampling: str = "logit_normal",
        flow_shift: float = 3.0,
        i2v_prob: float = 0.3,
        cfg_dropout_prob: float = 0.1,
        # Logit-normal distribution parameters
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        # Mix sampling parameters
        mix_uniform_ratio: float = 0.1,
        # Sigma clamping for finetuning (pretrain uses [0.0, 1.0])
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        # Loss weighting
        use_loss_weighting: bool = True,
        # Logging
        log_interval: int = 100,
        summary_log_interval: int = 10,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the FlowMatching pipeline.

        Args:
            model_adapter: ModelAdapter instance for model-specific operations
            num_train_timesteps: Total number of timesteps for the flow
            timestep_sampling: Sampling strategy:
                - "uniform": Pure uniform sampling
                - "logit_normal": SD3-style logit-normal (recommended)
                - "mode": Mode-based sampling
                - "lognorm": Log-normal based sampling
                - "mix": Mix of lognorm and uniform
            flow_shift: Shift parameter for timestep transformation
            i2v_prob: Probability of using image-to-video conditioning
            cfg_dropout_prob: Probability of dropping text embeddings for CFG training
            logit_mean: Mean for logit-normal distribution
            logit_std: Std for logit-normal distribution
            mix_uniform_ratio: Ratio of uniform samples when using mix
            sigma_min: Minimum sigma (0.0 for pretrain)
            sigma_max: Maximum sigma (1.0 for pretrain)
            use_loss_weighting: Whether to apply flow-based loss weighting
            log_interval: Steps between detailed logs
            summary_log_interval: Steps between summary logs
            device: Device to use for computations
        """
        self.model_adapter = model_adapter
        self.num_train_timesteps = num_train_timesteps
        self.timestep_sampling = timestep_sampling
        self.flow_shift = flow_shift
        self.i2v_prob = i2v_prob
        self.cfg_dropout_prob = cfg_dropout_prob
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mix_uniform_ratio = mix_uniform_ratio
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.use_loss_weighting = use_loss_weighting
        self.log_interval = log_interval
        self.summary_log_interval = summary_log_interval
        self.device = device if device is not None else torch.device("cuda")

        # Initialize noise schedule
        self.noise_schedule = LinearInterpolationSchedule()

    def sample_timesteps(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Sample timesteps and compute sigma values with flow shift.

        Implements the flow shift transformation:
        σ = shift / (shift + (1/u - 1))

        Args:
            batch_size: Number of timesteps to sample
            device: Device for tensor operations

        Returns:
            sigma: Sigma values in [sigma_min, sigma_max]
            timesteps: Timesteps in [0, num_train_timesteps]
            sampling_method: Name of the sampling method used
        """
        if device is None:
            device = self.device

        # Determine if we should use uniform (for mix strategy)
        use_uniform = self.timestep_sampling == "uniform" or (
            self.mix_uniform_ratio > 0 and torch.rand(1).item() < self.mix_uniform_ratio
        )

        if use_uniform:
            u = torch.rand(size=(batch_size,), device=device)
            sampling_method = "uniform"
        else:
            u = self._sample_from_distribution(batch_size, device)
            sampling_method = self.timestep_sampling

        # Apply flow shift: σ = shift / (shift + (1/u - 1))
        u_clamped = torch.clamp(u, min=1e-5)  # Avoid division by zero
        sigma = self.flow_shift / (self.flow_shift + (1.0 / u_clamped - 1.0))

        # Apply sigma clamping
        sigma = torch.clamp(sigma, self.sigma_min, self.sigma_max)

        # Convert sigma to timesteps [0, T]
        timesteps = sigma * self.num_train_timesteps

        return sigma, timesteps, sampling_method

    def _sample_from_distribution(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample u values from the configured distribution."""
        if self.timestep_sampling == "logit_normal":
            u = torch.normal(
                mean=self.logit_mean,
                std=self.logit_std,
                size=(batch_size,),
                device=device,
            )
            u = torch.sigmoid(u)

        elif self.timestep_sampling == "lognorm":
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            u = torch.sigmoid(u)

        elif self.timestep_sampling == "mode":
            mode_scale = 1.29
            u = torch.rand(size=(batch_size,), device=device)
            u = 1.0 - u - mode_scale * (torch.cos(math.pi * u / 2.0) ** 2 - 1.0 + u)
            u = torch.clamp(u, 0.0, 1.0)

        elif self.timestep_sampling == "mix":
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            u = torch.sigmoid(u)

        else:
            u = torch.rand(size=(batch_size,), device=device)

        return u

    def determine_task_type(self, data_type: str) -> str:
        """Determine task type based on data type and randomization."""
        if data_type == "image":
            return "t2v"
        elif data_type == "video":
            return "i2v" if random.random() < self.i2v_prob else "t2v"
        else:
            return "t2v"

    def compute_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute flow matching loss with optional weighting.

        Loss weight: w = 1 + flow_shift * σ

        Args:
            model_pred: Model prediction
            target: Target (velocity = noise - clean)
            sigma: Sigma values for each sample
            batch: Optional batch dictionary containing loss_mask

        Returns:
            weighted_loss: Per-element weighted loss
            average_weighted_loss: Scalar average weighted loss
            unweighted_loss: Per-element raw MSE loss
            average_unweighted_loss: Scalar average unweighted loss
            loss_weight: Applied weights
            loss_mask: Loss mask from batch (or None if not present)
        """
        loss = nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_mask = batch.get("loss_mask") if batch is not None else None

        if self.use_loss_weighting:
            loss_weight = 1.0 + self.flow_shift * sigma
            loss_weight = loss_weight.view(-1, *([1] * (loss.ndim - 1)))
        else:
            loss_weight = torch.ones_like(sigma).view(-1, *([1] * (loss.ndim - 1)))

        loss_weight = loss_weight.to(model_pred.device)

        unweighted_loss = loss
        weighted_loss = loss * loss_weight
        average_unweighted_loss = unweighted_loss.mean()
        average_weighted_loss = weighted_loss.mean()

        return weighted_loss, average_weighted_loss, unweighted_loss, average_unweighted_loss, loss_weight, loss_mask

    def step(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Execute a single training step with flow matching.

        Expected batch format:
        {
            "video_latents": torch.Tensor,  # [B, C, F, H, W] for video
            OR
            "image_latents": torch.Tensor,  # [B, C, H, W] for image
            "text_embeddings": torch.Tensor,  # [B, seq_len, dim]
            "data_type": str,  # "video" or "image" (optional)
            # ... additional model-specific keys handled by adapter
        }

        Args:
            model: The model to train
            batch: Batch of training data
            device: Device to use
            dtype: Data type for operations
            global_step: Current training step (for logging)

        Returns:
            weighted_loss: Per-element weighted loss
            average_weighted_loss: Scalar average weighted loss
            loss_mask: Mask indicating valid loss elements (or None)
            metrics: Dictionary of training metrics
        """
        debug_mode = os.environ.get("DEBUG_TRAINING", "0") == "1"
        detailed_log = global_step % self.log_interval == 0
        summary_log = global_step % self.summary_log_interval == 0

        # Extract and prepare batch data (either image_latents or video_latents)
        if "video_latents" in batch:
            latents = batch["video_latents"].to(device, dtype=dtype)
        elif "image_latents" in batch:
            latents = batch["image_latents"].to(device, dtype=dtype)
        else:
            raise KeyError("Batch must contain either 'video_latents' or 'image_latents'")

        # latents can be 4D [B, C, H, W] for images or 5D [B, C, F, H, W] for videos
        batch_size = latents.shape[0]

        # Determine task type
        data_type = batch.get("data_type", "video")
        task_type = self.determine_task_type(data_type)

        # ====================================================================
        # Flow Matching: Sample Timesteps
        # ====================================================================
        sigma, timesteps, sampling_method = self.sample_timesteps(batch_size, device)

        # ====================================================================
        # Flow Matching: Add Noise
        # ====================================================================
        noise = torch.randn_like(latents, dtype=torch.float32)

        # x_t = (1 - σ) * x_0 + σ * ε
        noisy_latents = self.noise_schedule.forward(latents.float(), noise, sigma)

        # ====================================================================
        # Logging
        # ====================================================================
        if detailed_log and debug_mode:
            self._log_detailed(
                global_step, sampling_method, batch_size, sigma, timesteps, latents, noise, noisy_latents
            )
        elif summary_log and debug_mode:
            logger.info(
                f"[STEP {global_step}] σ=[{sigma.min():.3f},{sigma.max():.3f}] | "
                f"t=[{timesteps.min():.1f},{timesteps.max():.1f}] | "
                f"noisy=[{noisy_latents.min():.1f},{noisy_latents.max():.1f}] | "
                f"{sampling_method}"
            )

        # Convert to target dtype
        noisy_latents = noisy_latents.to(dtype)

        # ====================================================================
        # Forward Pass (via adapter)
        # ====================================================================
        context = FlowMatchingContext(
            noisy_latents=noisy_latents,
            latents=latents,
            timesteps=timesteps,
            sigma=sigma,
            task_type=task_type,
            data_type=data_type,
            device=device,
            dtype=dtype,
            cfg_dropout_prob=self.cfg_dropout_prob,
            batch=batch,
        )

        inputs = self.model_adapter.prepare_inputs(context)
        model_pred = self.model_adapter.forward(model, inputs)

        # ====================================================================
        # Target: Flow Matching Velocity
        # ====================================================================
        # v = ε - x_0
        target = noise - latents.float()

        # ====================================================================
        # Loss Computation
        # ====================================================================
        weighted_loss, average_weighted_loss, unweighted_loss, average_unweighted_loss, loss_weight, loss_mask = (
            self.compute_loss(model_pred, target, sigma, batch)
        )

        # Safety check
        if torch.isnan(average_weighted_loss) or average_weighted_loss > 100:
            logger.error(f"[ERROR] Loss explosion! Loss={average_weighted_loss.item():.3f}")
            raise ValueError(f"Loss exploded: {average_weighted_loss.item()}")

        # Logging
        if detailed_log and debug_mode:
            self._log_loss_detailed(
                global_step, model_pred, target, loss_weight, average_unweighted_loss, average_weighted_loss
            )
        elif summary_log and debug_mode:
            logger.info(
                f"[STEP {global_step}] Loss: {average_weighted_loss.item():.6f} | "
                f"w=[{loss_weight.min():.2f},{loss_weight.max():.2f}]"
            )

        # Collect metrics
        metrics = {
            "loss": average_weighted_loss.item(),
            "unweighted_loss": average_unweighted_loss.item(),
            "sigma_min": sigma.min().item(),
            "sigma_max": sigma.max().item(),
            "sigma_mean": sigma.mean().item(),
            "weight_min": loss_weight.min().item(),
            "weight_max": loss_weight.max().item(),
            "timestep_min": timesteps.min().item(),
            "timestep_max": timesteps.max().item(),
            "noisy_min": noisy_latents.min().item(),
            "noisy_max": noisy_latents.max().item(),
            "sampling_method": sampling_method,
            "task_type": task_type,
            "data_type": data_type,
        }

        return weighted_loss, average_weighted_loss, loss_mask, metrics

    def _log_detailed(
        self,
        global_step: int,
        sampling_method: str,
        batch_size: int,
        sigma: torch.Tensor,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
        noise: torch.Tensor,
        noisy_latents: torch.Tensor,
    ):
        """Log detailed training information."""
        logger.info("\n" + "=" * 80)
        logger.info(f"[STEP {global_step}] FLOW MATCHING")
        logger.info("=" * 80)
        logger.info("[INFO] Using: x_t = (1-σ)x_0 + σ*ε")
        logger.info("")
        logger.info(f"[SAMPLING] Method: {sampling_method}")
        logger.info(f"[FLOW] Shift: {self.flow_shift}")
        logger.info(f"[BATCH] Size: {batch_size}")
        logger.info("")
        logger.info(f"[SIGMA] Range: [{sigma.min():.4f}, {sigma.max():.4f}]")
        if sigma.numel() > 1:
            logger.info(f"[SIGMA] Mean: {sigma.mean():.4f}, Std: {sigma.std():.4f}")
        else:
            logger.info(f"[SIGMA] Value: {sigma.item():.4f}")
        logger.info("")
        logger.info(f"[TIMESTEPS] Range: [{timesteps.min():.2f}, {timesteps.max():.2f}]")
        logger.info("")
        logger.info(f"[RANGES] Clean latents: [{latents.min():.4f}, {latents.max():.4f}]")
        logger.info(f"[RANGES] Noise:         [{noise.min():.4f}, {noise.max():.4f}]")
        logger.info(f"[RANGES] Noisy latents: [{noisy_latents.min():.4f}, {noisy_latents.max():.4f}]")

        # Sanity check
        max_expected = (
            max(
                abs(latents.max().item()),
                abs(latents.min().item()),
                abs(noise.max().item()),
                abs(noise.min().item()),
            )
            * 1.5
        )
        if abs(noisy_latents.max()) > max_expected or abs(noisy_latents.min()) > max_expected:
            logger.info(f"\n⚠️  WARNING: Noisy range seems large! Expected ~{max_expected:.1f}")
        else:
            logger.info("\n✓ Noisy latents range is reasonable")
        logger.info("=" * 80 + "\n")

    def _log_loss_detailed(
        self,
        global_step: int,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        loss_weight: torch.Tensor,
        unweighted_loss: torch.Tensor,
        weighted_loss: torch.Tensor,
    ):
        """Log detailed loss information."""
        logger.info("=" * 80)
        logger.info(f"[STEP {global_step}] LOSS DEBUG")
        logger.info("=" * 80)
        logger.info("[TARGET] Flow matching: v = ε - x_0")
        logger.info("")
        logger.info(f"[RANGES] Model pred: [{model_pred.min():.4f}, {model_pred.max():.4f}]")
        logger.info(f"[RANGES] Target (v): [{target.min():.4f}, {target.max():.4f}]")
        logger.info("")
        logger.info(f"[WEIGHTS] Formula: 1 + {self.flow_shift} * σ")
        logger.info(f"[WEIGHTS] Range: [{loss_weight.min():.4f}, {loss_weight.max():.4f}]")
        logger.info(f"[WEIGHTS] Mean: {loss_weight.mean():.4f}")
        logger.info("")
        unweighted_val = unweighted_loss.item()
        weighted_val = weighted_loss.item()
        logger.info(f"[LOSS] Unweighted: {unweighted_val:.6f}")
        logger.info(f"[LOSS] Weighted:   {weighted_val:.6f}")
        logger.info(f"[LOSS] Impact:     {(weighted_val / max(unweighted_val, 1e-8)):.3f}x")
        logger.info("=" * 80 + "\n")


# =============================================================================
# Factory Functions
# =============================================================================


def create_adapter(adapter_type: str, **kwargs) -> ModelAdapter:
    """
    Factory function to create a model adapter by name.

    Args:
        adapter_type: Type of adapter ("hunyuan", "simple", "flux")
        **kwargs: Additional arguments passed to the adapter constructor

    Returns:
        ModelAdapter instance
    """
    adapters = {
        "hunyuan": HunyuanAdapter,
        "simple": SimpleAdapter,
        "flux": FluxAdapter,
    }

    if adapter_type not in adapters:
        raise ValueError(f"Unknown adapter type: {adapter_type}. Available: {list(adapters.keys())}")

    return adapters[adapter_type](**kwargs)


def create_pipeline(
    adapter_type: str,
    adapter_kwargs: Optional[Dict[str, Any]] = None,
    **pipeline_kwargs,
) -> FlowMatchingPipeline:
    """
    Factory function to create a pipeline with a specific adapter.

    Args:
        adapter_type: Type of adapter ("hunyuan", "simple")
        adapter_kwargs: Arguments for the adapter constructor
        **pipeline_kwargs: Arguments for the pipeline constructor

    Returns:
        FlowMatchingPipeline instance

    Example:
        pipeline = create_pipeline(
            adapter_type="hunyuan",
            adapter_kwargs={"use_condition_latents": True},
            flow_shift=3.0,
            timestep_sampling="logit_normal",
        )
    """
    adapter_kwargs = adapter_kwargs or {}
    adapter = create_adapter(adapter_type, **adapter_kwargs)
    return FlowMatchingPipeline(model_adapter=adapter, **pipeline_kwargs)
