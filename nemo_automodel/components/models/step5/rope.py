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

"""Half-split RoPE with the checkpoint's explicit BF16 table boundary."""

import math

import torch
from torch import nn

from nemo_automodel.components.models.step5.configuration import Step5Config


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, span: int) -> torch.Tensor:
    """Apply already-rounded tables using FP32 half-split arithmetic."""
    half = span // 2
    a, b = x[..., :half].float(), x[..., half:span].float()
    return torch.cat(((a * cos - b * sin).to(x.dtype), (a * sin + b * cos).to(x.dtype), x[..., span:]), dim=-1)


@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"emulate_precision_casts": True},
)
def _apply_full_rotary_compiled(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return _apply_rotary(x, cos, sin, x.shape[-1])


@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"emulate_precision_casts": True},
)
def _apply_partial_rotary_compiled(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, span: int) -> torch.Tensor:
    return _apply_rotary(x, cos, sin, span)


@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"emulate_precision_casts": True},
)
def _apply_single_head_rotary_compiled(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, span: int
) -> torch.Tensor:
    return _apply_rotary(x, cos, sin, span)


def _apply_rotary_compiled(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, span: int) -> torch.Tensor:
    # Full/partial slices and singleton-head broadcasting have different
    # shape guards. Separate entries keep mixed main/indexer Q/K, singleton
    # token, and grad/no_grad shapes below Dynamo's specialization cache cap.
    if x.numel() == 0:
        return x
    if span == x.shape[-1]:
        return _apply_full_rotary_compiled(x, cos, sin)
    if x.shape[-2] == 1:
        return _apply_single_head_rotary_compiled(x, cos, sin, span)
    return _apply_partial_rotary_compiled(x, cos, sin, span)


class Step5RotaryEmbedding(nn.Module):
    """Recompute FP32 frequencies from config; never cast a frequency buffer."""

    def __init__(self, config: Step5Config, layer_idx: int, *, span: int) -> None:
        super().__init__()
        if span < 0 or span % 2:
            raise ValueError(f"Invalid rotary span {span}")
        self.span = span
        self.theta = config.rope_theta[layer_idx] if isinstance(config.rope_theta, list) else config.rope_theta
        self.scaling = config.rope_scaling if config.layer_types[layer_idx] in config.yarn_only_types else None

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate the leading half-split subspace.

        Args:
            x: Tensor of shape [batch, sequence, heads, head_dim].
            positions: Tensor of shape [batch, sequence] with document-local positions.

        Returns:
            Tensor of shape [batch, sequence, heads, head_dim] in the input dtype.
        """
        if self.span == 0 or x.numel() == 0:
            return x
        cos, sin = self._tables(x, positions)
        apply = _apply_rotary_compiled if x.is_cuda else _apply_rotary
        return apply(x, cos, sin, self.span)

    def forward_pair(
        self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate local Q/K with one shared, identically rounded table pair.

        Args:
            query: Query tensor [batch, sequence, query_heads, head_dim].
            key: Key tensor [batch, sequence, key_heads, head_dim], sharing
                query's dtype and device. Noncontiguous key views are supported.
            positions: Document-local positions [batch, sequence].

        Returns:
            Rotated query and key tensors, preserving their respective shapes.

        Raises:
            ValueError: Q/K have different batch/sequence extents, dtype or device.
        """
        if query.shape[:2] != key.shape[:2] or query.dtype != key.dtype or query.device != key.device:
            raise ValueError("Paired Step5 RoPE requires matching Q/K batch, sequence, dtype and device")
        if self.span == 0 or query.numel() == key.numel() == 0:
            return query, key
        cos, sin = self._tables(query, positions)
        apply = _apply_rotary_compiled if query.is_cuda else _apply_rotary
        return apply(query, cos, sin, self.span), apply(key, cos, sin, self.span)

    def _tables(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build tables once per call; preserve eager frequency/trig rounding."""
        inv = 1.0 / self.theta ** (torch.arange(0, self.span, 2, device=x.device, dtype=torch.float32) / self.span)
        if self.scaling is not None and self.scaling.get("rope_type", "default") != "default":
            factor = self.scaling["factor"]
            low, high = self.scaling["low_freq_factor"], self.scaling["high_freq_factor"]
            original = self.scaling["original_max_position_embeddings"]
            wavelength = 2 * math.pi / inv
            smooth = (original / wavelength - low) / (high - low)
            scaled = torch.where(wavelength > original / low, inv / factor, inv)
            inv = torch.where(
                (wavelength >= original / high) & (wavelength <= original / low),
                (1 - smooth) * inv / factor + smooth * inv,
                scaled,
            )
        angles = positions.float().unsqueeze(-1) * inv
        # Released inference stores FP32-generated tables in BF16.
        table_dtype = torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float32
        cos = angles.cos().to(table_dtype).float().unsqueeze(-2)
        sin = angles.sin().to(table_dtype).float().unsqueeze(-2)
        # Frequency construction and BF16 table materialization stay eager;
        # fusing trig would change checkpoint rounding. No cross-call cache.
        return cos, sin
