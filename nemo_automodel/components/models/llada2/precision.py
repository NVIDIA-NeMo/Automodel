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

"""Preserve the publisher's FP32 LLaDA2 router under compute autocast."""

from __future__ import annotations

from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.mesh import MeshContext


def apply_precision_guards(model: nn.Module, *, mesh: MeshContext) -> nn.Module:
    """Protect the remote LLaDA2 router without changing attention or experts."""
    config = model.config
    if config.num_experts is None:
        return model
    for index in range(config.first_k_dense_replace, config.num_hidden_layers):
        gate = model.get_submodule(f"model.layers.{index}.mlp.gate")
        _guard_gate(gate)
    return model


def _guard_gate(gate: nn.Module) -> None:
    original = gate.forward.__func__

    @wraps(original)
    def forward(self: nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run routing with the publisher's explicit FP32 operands.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Expert indices and weights of shape [tokens, top_k], and FP32
            router logits of shape [tokens, experts]. Tokens flattens the leading dimensions.
        """
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return original(self, hidden_states)

    gate.forward = MethodType(forward, gate)
