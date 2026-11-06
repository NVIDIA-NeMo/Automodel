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

"""Preserve FP32 rotary angles in the publisher's diffusion model."""

from __future__ import annotations

from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.mesh import MeshContext


def apply_precision_guards(model: nn.Module, *, mesh: MeshContext) -> nn.Module:
    """Protect rotary angle construction while retaining the publisher forward."""
    rotary = model.get_submodule("encoder.rotary_emb")
    original = rotary.forward.__func__

    @wraps(original)
    def forward(self: nn.Module, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build rotary angles with the publisher's explicit FP32 operands.

        Args:
            x: Tensor of shape [batch, sequence, hidden], supplying device and output dtype.
            position_ids: Integer tensor of shape [batch, sequence].

        Returns:
            Cosine and sine tensors of shape [batch, sequence, head_dim], in x's dtype.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            return original(self, x, position_ids)

    rotary.forward = MethodType(forward, rotary)
    return model
