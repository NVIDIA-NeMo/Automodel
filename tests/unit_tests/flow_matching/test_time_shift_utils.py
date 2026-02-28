# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for time_shift_utils.py: time_shift, compute_density_for_timestep_sampling, get_flow_match_loss_weight."""

import math

import numpy as np
import pytest
import torch

from nemo_automodel.components.flow_matching.time_shift_utils import (
    compute_density_for_timestep_sampling,
    get_flow_match_loss_weight,
    time_shift,
)


# =============================================================================
# TestTimeShift
# =============================================================================


class TestTimeShift:
    """Tests for time_shift() covering all shift_type branches."""

    # --- linear ---
    @pytest.mark.parametrize(
        "image_seq_len, base_shift, max_shift",
        [
            (256, 0.5, 1.15),
            (4096, 0.5, 1.15),
            (1024, 0.0, 2.0),
        ],
    )
    def test_linear_formula(self, image_seq_len, base_shift, max_shift):
        t = torch.tensor([0.1, 0.5, 0.9])
        result = time_shift(t, image_seq_len, shift_type="linear", base_shift=base_shift, max_shift=max_shift)

        mu = base_shift + (max_shift - base_shift) * (image_seq_len / 4096)
        expected = math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))
        assert torch.allclose(result, expected, atol=1e-6)

    def test_linear_boundary_seq_len_zero(self):
        t = torch.tensor([0.5])
        result = time_shift(t, 0, shift_type="linear", base_shift=0.5, max_shift=1.15)
        mu = 0.5
        expected = math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))
        assert torch.allclose(result, expected, atol=1e-6)

    # --- sqrt ---
    @pytest.mark.parametrize("image_seq_len", [256, 128 * 128, 100000])
    def test_sqrt_formula(self, image_seq_len):
        t = torch.tensor([0.2, 0.5, 0.8])
        result = time_shift(t, image_seq_len, shift_type="sqrt")

        mu = np.maximum(1.0, np.sqrt(image_seq_len / (128.0 * 128.0)) * 3.0)
        expected = mu / (mu + (1.0 / t - 1.0))
        assert torch.allclose(result, expected.float(), atol=1e-5)

    def test_sqrt_small_seq_len_clamps_mu_to_one(self):
        t = torch.tensor([0.5])
        result = time_shift(t, 1, shift_type="sqrt")
        # mu = max(1.0, sqrt(1/16384)*3) ≈ max(1.0, 0.023) = 1.0
        expected = 1.0 / (1.0 + (1.0 / t - 1.0))
        assert torch.allclose(result, expected, atol=1e-5)

    # --- constant ---
    @pytest.mark.parametrize("constant", [1.0, 3.0, 10.0])
    def test_constant_formula(self, constant):
        t = torch.tensor([0.1, 0.5, 0.9])
        result = time_shift(t, 256, shift_type="constant", constant=constant)
        expected = constant / (constant + (1.0 / t - 1.0))
        assert torch.allclose(result, expected, atol=1e-6)

    def test_constant_default_value(self):
        t = torch.tensor([0.5])
        result = time_shift(t, 256, shift_type="constant")
        expected = 3.0 / (3.0 + (1.0 / t - 1.0))
        assert torch.allclose(result, expected, atol=1e-6)

    # --- passthrough (else) ---
    def test_passthrough_returns_t_unchanged(self):
        t = torch.tensor([0.1, 0.5, 0.9])
        result = time_shift(t, 256, shift_type="none")
        assert torch.allclose(result, t)

    def test_passthrough_unknown_type(self):
        t = torch.tensor([0.3])
        result = time_shift(t, 100, shift_type="foobar")
        assert torch.allclose(result, t)


# =============================================================================
# TestComputeDensityForTimestepSampling
# =============================================================================


class TestComputeDensityForTimestepSampling:
    """Tests for compute_density_for_timestep_sampling() covering all weighting schemes."""

    @pytest.mark.parametrize("batch_size", [1, 8, 64])
    def test_logit_normal_shape_and_range(self, batch_size):
        u = compute_density_for_timestep_sampling("logit_normal", batch_size)
        assert u.shape == (batch_size,)
        assert u.device == torch.device("cpu")
        # sigmoid output is in (0, 1)
        assert (u >= 0).all() and (u <= 1).all()

    def test_logit_normal_custom_params(self):
        u = compute_density_for_timestep_sampling("logit_normal", 100, logit_mean=1.0, logit_std=0.5)
        assert u.shape == (100,)
        # With high mean, most values should be > 0.5
        assert u.mean() > 0.5

    @pytest.mark.parametrize("batch_size", [1, 8, 64])
    def test_mode_shape_and_range(self, batch_size):
        u = compute_density_for_timestep_sampling("mode", batch_size)
        assert u.shape == (batch_size,)
        assert u.device == torch.device("cpu")

    def test_mode_custom_scale(self):
        u = compute_density_for_timestep_sampling("mode", 100, mode_scale=2.0)
        assert u.shape == (100,)

    @pytest.mark.parametrize("batch_size", [1, 8, 64])
    def test_uniform_shape_and_range(self, batch_size):
        u = compute_density_for_timestep_sampling("uniform", batch_size)
        assert u.shape == (batch_size,)
        assert u.device == torch.device("cpu")
        assert (u >= 0).all() and (u < 1).all()

    def test_unknown_scheme_falls_through_to_uniform(self):
        u = compute_density_for_timestep_sampling("unknown_scheme", 32)
        assert u.shape == (32,)
        assert (u >= 0).all() and (u < 1).all()


# =============================================================================
# TestGetFlowMatchLossWeight
# =============================================================================


class TestGetFlowMatchLossWeight:
    """Tests for get_flow_match_loss_weight()."""

    def test_formula(self):
        sigma = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        shift = 3.0
        w = get_flow_match_loss_weight(sigma, shift)
        expected = 1.0 + shift * sigma
        assert torch.allclose(w, expected)

    def test_zero_sigma(self):
        sigma = torch.zeros(4)
        w = get_flow_match_loss_weight(sigma, shift=5.0)
        assert torch.allclose(w, torch.ones(4))

    def test_zero_shift(self):
        sigma = torch.tensor([0.3, 0.7])
        w = get_flow_match_loss_weight(sigma, shift=0.0)
        assert torch.allclose(w, torch.ones(2))

    @pytest.mark.parametrize("shift", [0.1, 1.0, 3.0, 10.0])
    def test_various_shifts(self, shift):
        sigma = torch.rand(16)
        w = get_flow_match_loss_weight(sigma, shift)
        expected = 1.0 + shift * sigma
        assert torch.allclose(w, expected, atol=1e-6)

    def test_default_shift(self):
        sigma = torch.tensor([0.5])
        w = get_flow_match_loss_weight(sigma)
        expected = torch.tensor([1.0 + 3.0 * 0.5])
        assert torch.allclose(w, expected)

    def test_output_shape_matches_input(self):
        sigma = torch.rand(32)
        w = get_flow_match_loss_weight(sigma, shift=3.0)
        assert w.shape == sigma.shape
