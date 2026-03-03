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
Base classes and data structures for model adapters.

This module defines the abstract ModelAdapter class and the FlowMatchingContext
dataclass used to pass data between the pipeline and adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn


@dataclass
class FlowMatchingContext:
    """
    Context object passed to model adapters containing all necessary data.

    This provides a clean interface for adapters to access the data they need
    without coupling to the batch dictionary structure.

    Attributes:
        noisy_latents: [B, C, F, H, W] or [B, C, H, W] - Noisy latents after interpolation
        latents: [B, C, F, H, W] for video or [B, C, H, W] for image - Original clean latents
            (also accessible via deprecated 'video_latents' property for backward compatibility)
        timesteps: [B] - Sampled timesteps
        sigma: [B] - Sigma values
        task_type: "t2v" or "i2v"
        data_type: "video" or "image"
        device: Device for tensor operations
        dtype: Data type for tensor operations
        cfg_dropout_prob: Probability of dropping text embeddings (setting to 0) during
            training for classifier-free guidance (CFG). Defaults to 0.0 for backward compatibility.
        batch: Original batch dictionary (for model-specific data)
    """

    # Core tensors
    noisy_latents: torch.Tensor
    latents: torch.Tensor
    timesteps: torch.Tensor
    sigma: torch.Tensor

    # Task info
    task_type: str
    data_type: str

    # Device/dtype
    device: torch.device
    dtype: torch.dtype

    # Original batch (for model-specific data)
    batch: Dict[str, Any]

    # CFG dropout probability (optional with default for backward compatibility)
    cfg_dropout_prob: float = 0.0

    @property
    def video_latents(self) -> torch.Tensor:
        """Backward compatibility alias for 'latents' field."""
        return self.latents


class ModelAdapter(ABC):
    """
    Abstract base class for model-specific forward pass logic.

    Implement this class to add support for new model architectures
    without modifying the FlowMatchingPipeline.

    The adapter pattern decouples the flow matching logic from model-specific
    details like input preparation and forward pass conventions.

    Example:
        class MyCustomAdapter(ModelAdapter):
            def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
                return {
                    "x": context.noisy_latents,
                    "t": context.timesteps,
                    "cond": context.batch["my_conditioning"],
                }

            def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
                return model(**inputs)

        pipeline = FlowMatchingPipelineV2(model_adapter=MyCustomAdapter())
    """

    @abstractmethod
    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        """
        Prepare model-specific inputs from the context.

        Args:
            context: FlowMatchingContext containing all necessary data

        Returns:
            Dictionary of inputs to pass to the model's forward method
        """
        pass

    @abstractmethod
    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute the model forward pass.

        Args:
            model: The model to call
            inputs: Dictionary of inputs from prepare_inputs()

        Returns:
            Model prediction tensor
        """
        pass

    def post_process_prediction(self, model_pred: torch.Tensor) -> torch.Tensor:
        """
        Post-process model prediction if needed.

        Override this for models that return extra outputs or need transformation.

        Args:
            model_pred: Raw model output

        Returns:
            Processed prediction tensor
        """
        if isinstance(model_pred, tuple):
            return model_pred[0]
        return model_pred
