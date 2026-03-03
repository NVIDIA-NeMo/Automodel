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
Simple transformer model adapter for FlowMatching Pipeline.

This adapter supports simple transformer models with a basic interface,
such as Wan-style models.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import FlowMatchingContext, ModelAdapter


class SimpleAdapter(ModelAdapter):
    """
    Model adapter for simple transformer models (e.g., Wan).

    These models use a simple interface with:
    - hidden_states: noisy latents
    - timestep: timestep values
    - encoder_hidden_states: text embeddings

    Expected batch keys:
    - text_embeddings: Text encoder output [B, seq_len, dim]

    Example:
        adapter = SimpleAdapter()
        pipeline = FlowMatchingPipelineV2(model_adapter=adapter)
    """

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        """
        Prepare inputs for simple transformer model.

        Args:
            context: FlowMatchingContext with batch data

        Returns:
            Dictionary containing:
            - hidden_states: Noisy latents
            - timestep: Timestep values
            - encoder_hidden_states: Text embeddings
        """
        batch = context.batch
        device = context.device
        dtype = context.dtype

        # Get text embeddings
        text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)

        return {
            "hidden_states": context.noisy_latents,
            "timestep": context.timesteps.to(dtype),
            "encoder_hidden_states": text_embeddings,
        }

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute forward pass for simple transformer model.

        Args:
            model: Transformer model
            inputs: Dictionary from prepare_inputs()

        Returns:
            Model prediction tensor
        """
        model_pred = model(
            hidden_states=inputs["hidden_states"],
            timestep=inputs["timestep"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            return_dict=False,
        )
        return self.post_process_prediction(model_pred)
