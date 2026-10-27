# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Fused normalization kernels for the dense Qwen3.5 decoder.

The reference Qwen3.5 norms run as chains of eager fp32 elementwise kernels, each a
full pass over the activation: about seven forward launches per zero-centered
RMSNorm and nine per gated RMSNorm, with more in backward. On CUDA these modules
compile the same math into fused Triton kernels for forward and backward. CPU and
DTensor inputs keep the eager chain, so results match the reference modules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNormGated

from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextRMSNorm


def _zero_centered_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute ``norm(x) * (1 + weight)`` in fp32, exactly as ``Qwen3NextRMSNorm``.

    Args:
        x: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        weight: Tensor of shape [hidden] holding the zero-centered gain.
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of shape [..., hidden] in x's dtype.
    """
    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    return (normalized * (1.0 + weight.float())).to(x.dtype)


def _rms_norm_gated_silu(
    hidden_states: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Compute ``weight * norm(hidden_states) * silu(gate)`` as ``Qwen3_5MoeRMSNormGated``.

    The normalized activations are rounded to the input dtype before the weight
    multiply, matching the reference module.

    Args:
        hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        gate: Tensor of shape [..., hidden] holding the output-gate logits.
        weight: Tensor of shape [hidden] holding the multiplicative gain.
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of shape [..., hidden] in hidden_states' dtype.
    """
    input_dtype = hidden_states.dtype
    normalized = hidden_states.float()
    normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + eps)
    gated = weight * normalized.to(input_dtype)
    return (gated * F.silu(gate.float())).to(input_dtype)


_zero_centered_rms_norm_compiled = torch.compile(_zero_centered_rms_norm, dynamic=True)
# Emulating the intermediate cast keeps the fused kernel's rounding identical to eager.
_rms_norm_gated_silu_compiled = torch.compile(
    _rms_norm_gated_silu, dynamic=True, options={"emulate_precision_casts": True}
)


def _fusable(*tensors: torch.Tensor) -> bool:
    return all(t.is_cuda and not isinstance(t, DTensor) for t in tensors)


class Qwen3_5FusedRMSNorm(Qwen3NextRMSNorm):
    """Zero-centered Qwen3.5 RMSNorm computed by one fused kernel on CUDA."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last axis in fp32 and scale by ``1 + weight``.

        Args:
            x: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., hidden] in x's dtype.
        """
        norm = _zero_centered_rms_norm_compiled if _fusable(x, self.weight) else _zero_centered_rms_norm
        return norm(x, self.weight, self.eps)


class Qwen3_5FusedRMSNormGated(Qwen3_5MoeRMSNormGated):
    """GatedDeltaNet output norm with SiLU gate computed by one fused kernel on CUDA."""

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize the last axis, apply the gain, then multiply by ``silu(gate)``.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.
            gate: Tensor of shape [..., hidden] holding the output-gate logits.

        Returns:
            Tensor of shape [..., hidden] in hidden_states' dtype.
        """
        # With a weight wider than the activations (no mixed-precision cast), eager rounds an
        # intermediate gradient to the activation dtype and the compiled backward does not.
        fused = _fusable(hidden_states, gate, self.weight) and self.weight.dtype == hidden_states.dtype
        norm = _rms_norm_gated_silu_compiled if fused else _rms_norm_gated_silu
        return norm(hidden_states, gate, self.weight, self.variance_epsilon)
