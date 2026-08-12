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

"""Positional and timestep embeddings for BAGEL modules."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

_SIN_COS_BASE = 10000.0


def _validate_positive(value: int, *, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_even(value: int, *, name: str) -> None:
    _validate_positive(value, name=name)
    if value % 2 != 0:
        raise ValueError(f"{name} must be even, got {value}")


def _geometric_frequencies(width: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return inverse periods for one sine/cosine axis."""
    _validate_even(width, name="width")
    pair_count = width // 2
    steps = torch.arange(pair_count, device=device, dtype=torch.float64)
    return torch.pow(torch.tensor(_SIN_COS_BASE, device=device, dtype=torch.float64), -steps / pair_count)


def _encode_scalar_positions(positions: torch.Tensor, width: int) -> torch.Tensor:
    """Encode scalar coordinates with sine features followed by cosine features."""
    frequencies = _geometric_frequencies(width, device=positions.device)
    phases = positions.reshape(-1, 1).to(dtype=torch.float64) * frequencies.reshape(1, -1)
    return torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)


def _build_square_grid_positions(grid_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_positive(grid_size, name="grid_size")
    axis_positions = torch.arange(grid_size, dtype=torch.float64)
    y_positions, x_positions = torch.meshgrid(axis_positions, axis_positions, indexing="ij")
    return x_positions, y_positions


def _build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Return ``(grid_size ** 2, embed_dim)`` sine/cosine features for a square grid."""
    _validate_even(embed_dim, name="embed_dim")
    x_positions, y_positions = _build_square_grid_positions(grid_size)
    x_features = _encode_scalar_positions(x_positions, embed_dim // 2)
    y_features = _encode_scalar_positions(y_positions, embed_dim // 2)
    return torch.cat((x_features, y_features), dim=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Return a 2D sin-cos embedding table as a NumPy array."""
    return _build_2d_sincos_pos_embed(embed_dim, grid_size).numpy()


class BagelGridPositionEmbedding(nn.Module):
    """Frozen 2D sine/cosine position table for patch and latent grids."""

    def __init__(self, max_num_patch_per_side: int, hidden_size: int) -> None:
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side**2, hidden_size),
            requires_grad=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        pos_embed = _build_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        self.pos_embed.data.copy_(pos_embed.to(dtype=torch.float32))

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pos_embed[position_ids]


def _timestep_features(timesteps: torch.Tensor, width: int, *, max_period: int = 10000) -> torch.Tensor:
    """Create sine/cosine features for scalar timesteps."""
    _validate_positive(width, name="width")
    half_width = width // 2
    step_ids = torch.arange(half_width, device=timesteps.device, dtype=torch.float32)
    frequencies = torch.exp(-math.log(max_period) * step_ids / half_width)
    phases = timesteps[:, None].float() * frequencies[None, :]
    features = torch.cat((torch.cos(phases), torch.sin(phases)), dim=-1)
    if width % 2:
        features = torch.cat((features, torch.zeros_like(features[:, :1])), dim=-1)
    return features


class BagelTimestepEmbedding(nn.Module):
    """Map scalar timesteps through sine/cosine features and a small MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = _timestep_features(t, self.frequency_embedding_size)
        return self.mlp(t_freq)
