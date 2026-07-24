# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Typed construction config for flow-matching training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from nemo_automodel.components.flow_matching.adapters.base import ModelAdapter

if TYPE_CHECKING:
    from nemo_automodel.components.flow_matching.pipeline import FlowMatchingPipeline


@dataclass(frozen=True)
class FlowMatchingAdapterConfig:
    """Declarative adapter factory resolved from ``flow_matching.adapter``."""

    target: Callable[..., ModelAdapter]
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def build(self) -> ModelAdapter:
        """Construct the configured model adapter."""
        adapter = self.target(**dict(self.kwargs))
        if not isinstance(adapter, ModelAdapter):
            raise TypeError(
                f"flow_matching.adapter._target_ must construct a ModelAdapter, got {type(adapter).__name__}"
            )
        return adapter


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Typed configuration for a :class:`FlowMatchingPipeline`."""

    adapter: FlowMatchingAdapterConfig | None = None
    adapter_type: str | None = None
    adapter_kwargs: Mapping[str, Any] = field(default_factory=dict)
    num_train_timesteps: int = 1000
    timestep_sampling: str = "logit_normal"
    flow_shift: float = 3.0
    i2v_prob: float = 0.3
    cfg_dropout_prob: float = 0.1
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mix_uniform_ratio: float = 0.1
    beta_alpha: float = 2.5
    beta_beta: float = 1.5
    use_sigma_noise: bool = True
    sigma_min: float = 0.0
    sigma_max: float = 1.0
    use_loss_weighting: bool = True
    loss_weighting_scheme: str = "linear"
    log_interval: int = 100
    summary_log_interval: int = 10

    def __post_init__(self) -> None:
        if self.adapter is not None and self.adapter_type is not None:
            raise ValueError("flow_matching.adapter cannot be combined with legacy flow_matching.adapter_type")

    def build_adapter(self) -> ModelAdapter:
        """Construct either the canonical target adapter or a legacy named adapter."""
        if self.adapter is not None:
            return self.adapter.build()

        from nemo_automodel.components.flow_matching.pipeline import create_adapter

        return create_adapter(self.adapter_type or "simple", **dict(self.adapter_kwargs))

    def build(
        self,
        *,
        model_adapter: ModelAdapter,
        device: torch.device,
        generator: torch.Generator,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
    ) -> "FlowMatchingPipeline":
        """Build the runtime flow pipeline from declarative settings.

        Args:
            model_adapter: Adapter constructed before model parallelization.
            device: Training device.
            generator: Checkpointable, rank-local recipe generator.
            sigma_min: Optional runtime override for staged diffusion training.
            sigma_max: Optional runtime override for staged diffusion training.

        Returns:
            Configured flow-matching pipeline.
        """
        from nemo_automodel.components.flow_matching.pipeline import FlowMatchingPipeline

        return FlowMatchingPipeline(
            model_adapter=model_adapter,
            num_train_timesteps=self.num_train_timesteps,
            timestep_sampling=self.timestep_sampling,
            flow_shift=self.flow_shift,
            i2v_prob=self.i2v_prob,
            cfg_dropout_prob=self.cfg_dropout_prob,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mix_uniform_ratio=self.mix_uniform_ratio,
            beta_alpha=self.beta_alpha,
            beta_beta=self.beta_beta,
            use_sigma_noise=self.use_sigma_noise,
            sigma_min=self.sigma_min if sigma_min is None else sigma_min,
            sigma_max=self.sigma_max if sigma_max is None else sigma_max,
            use_loss_weighting=self.use_loss_weighting,
            loss_weighting_scheme=self.loss_weighting_scheme,
            log_interval=self.log_interval,
            summary_log_interval=self.summary_log_interval,
            device=device,
            generator=generator,
        )
