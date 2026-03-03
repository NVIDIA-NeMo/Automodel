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

from __future__ import annotations

import math

import numpy as np
import torch


def time_shift(
    t: torch.Tensor,
    image_seq_len: int,
    shift_type: str = "constant",
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    constant: float = 3.0,
):
    """
    Convert timesteps to sigmas with sequence-length-aware shifting.

    Args:
        t: timesteps in range [0, 1]
        image_seq_len: number of tokens (frames * height * width / patch_size^2)
        shift_type: "linear", "sqrt", or "constant"
        base_shift: base shift for linear mode
        max_shift: max shift for linear mode
        constant: shift value for constant mode (default 3.0 matches Pika)

    Returns:
        sigma values for noise scheduling
    """
    if shift_type == "linear":
        # Linear interpolation based on sequence length
        mu = base_shift + (max_shift - base_shift) * (image_seq_len / 4096)
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1))

    elif shift_type == "sqrt":
        # Square root scaling (Flux-style)
        # Assuming 128x128 latent space (1024x1024 image) gives mu=3
        mu = np.maximum(1.0, np.sqrt(image_seq_len / (128.0 * 128.0)) * 3.0)
        return mu / (mu + (1 / t - 1))

    elif shift_type == "constant":
        # Constant shift (Pika default)
        return constant / (constant + (1 / t - 1))

    else:
        # No shift, return original t
        return t


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
):
    """
    Sample timesteps from different distributions for better training coverage.

    Args:
        weighting_scheme: "uniform", "logit_normal", or "mode"
        batch_size: number of samples to generate
        logit_mean: mean for logit-normal distribution
        logit_std: std for logit-normal distribution
        mode_scale: scale for mode-based sampling

    Returns:
        Tensor of shape (batch_size,) with values in [0, 1]
    """
    if weighting_scheme == "logit_normal":
        # SD3-style logit-normal sampling
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)

    elif weighting_scheme == "mode":
        # Mode-based sampling (concentrates around certain timesteps)
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)

    else:
        # Uniform sampling (default)
        u = torch.rand(size=(batch_size,), device="cpu")

    return u


def get_flow_match_loss_weight(sigma: torch.Tensor, shift: float = 3.0):
    """
    Compute loss weights for flow matching based on sigma values.

    Higher sigma (more noise) typically gets higher weight.

    Args:
        sigma: sigma values in range [0, 1]
        shift: weight scaling factor

    Returns:
        Loss weights with same shape as sigma
    """
    # Flow matching weight: weight = 1 + shift * sigma
    # This gives more weight to noisier timesteps
    weight = 1.0 + shift * sigma
    return weight
