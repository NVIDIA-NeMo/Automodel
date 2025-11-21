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

"""Combined gate_up MLP projection for SwiGLU and similar activations.

This module provides a combined gate_up projection that combines gate_proj and up_proj
into a single projection, reducing kernel launch overhead and improving memory efficiency.
"""

import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class CombinedGateUpMLP(nn.Module):
    """SwiGLU MLP with combined gate_up projection for efficiency.

    This module combines gate_proj and up_proj into a single projection,
    then splits the result. This can improve efficiency by reducing kernel launches,
    though the benefit depends on the specific hardware and tensor sizes.

    Works with any activation function that follows the gate * up pattern.

    Args:
        config: Model config with attributes:
            - hidden_size: Model hidden dimension
            - intermediate_size: MLP intermediate dimension
            - hidden_act: Activation function name (e.g., "silu", "gelu")
            - mlp_bias: Whether to use bias (optional, defaults to False)

    Example:
        # For Llama-style SwiGLU:
        mlp = CombinedGateUpMLP(config)  # config.hidden_act = "silu"

        # For Qwen2-style SwiGLU:
        mlp = CombinedGateUpMLP(config)  # config.hidden_act = "silu"
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Get bias setting (default to False if not specified)
        mlp_bias = getattr(config, "mlp_bias", False)

        # Combined gate and up projections
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with combined gate_up projection.

        Handles tensor parallelism by dynamically computing split sizes
        based on actual tensor dimensions.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Project and split into gate and up
        gate_up = self.gate_up_proj(x)

        # Handle tensor parallelism: split based on actual tensor size
        gate_up_size = gate_up.shape[-1]
        local_intermediate_size = gate_up_size // 2
        gate, up = gate_up.split([local_intermediate_size, local_intermediate_size], dim=-1)

        # SwiGLU: down(act(gate) * up)
        return self.down_proj(self.act_fn(gate) * up)
