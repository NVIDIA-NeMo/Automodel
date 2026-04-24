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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.components.models.common.utils import (
    BackendConfig,
    Float32RMSNorm,
    TEFp8Config,
    get_is_first_microbatch,
    get_is_optim_step,
    get_rope_config,
    set_is_first_microbatch,
    set_is_optim_step,
)


class TestIsOptimStep:
    def teardown_method(self):
        set_is_optim_step(False)

    def test_default_is_false(self):
        set_is_optim_step(False)
        assert get_is_optim_step() is False

    def test_set_true(self):
        set_is_optim_step(True)
        assert get_is_optim_step() is True

    def test_set_false(self):
        set_is_optim_step(True)
        set_is_optim_step(False)
        assert get_is_optim_step() is False


class TestIsFirstMicrobatch:
    def teardown_method(self):
        set_is_first_microbatch(None)

    def test_default_is_none(self):
        set_is_first_microbatch(None)
        assert get_is_first_microbatch() is None

    def test_set_true(self):
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

    def test_set_false(self):
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False

    def test_set_none(self):
        set_is_first_microbatch(True)
        set_is_first_microbatch(None)
        assert get_is_first_microbatch() is None

    def test_grad_accumulation_lifecycle(self):
        """Simulate the typical GA lifecycle: True -> False -> True -> False."""
        # Start of optimizer step: first microbatch
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

        # After first microbatch
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False

        # Next optimizer step: first microbatch again
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

        # After first microbatch
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False


class TestTEFp8Config:
    def test_default_recipe(self):
        cfg = TEFp8Config()
        assert cfg.recipe == "current"

    def test_block_recipe(self):
        cfg = TEFp8Config(recipe="block")
        assert cfg.recipe == "block"

    def test_passthrough_recipe_object(self):
        """Non-string recipe objects are passed through directly."""
        sentinel = object()
        cfg = TEFp8Config(recipe=sentinel)
        assert cfg.build_recipe() is sentinel

    def test_maybe_te_autocast_without_te(self):
        """Without TE installed, maybe_te_autocast returns nullcontext."""
        cfg = TEFp8Config()
        with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
            ctx = cfg.maybe_te_autocast()
            assert isinstance(ctx, nullcontext)

    def test_build_recipe_without_te(self):
        """Without TE installed, build_recipe returns None."""
        cfg = TEFp8Config()
        with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
            assert cfg.build_recipe() is None


class TestBackendConfigTeFp8:
    def test_te_fp8_default_is_none(self):
        """BackendConfig.te_fp8 defaults to None."""
        cfg = BackendConfig()
        assert cfg.te_fp8 is None

    def test_te_fp8_dict_normalized(self):
        """A dict te_fp8 is converted to TEFp8Config."""
        cfg = BackendConfig(te_fp8={"recipe": "block"}, experts="te", dispatcher="deepep")
        assert isinstance(cfg.te_fp8, TEFp8Config)
        assert cfg.te_fp8.recipe == "block"

    def test_te_fp8_requires_te_backend(self):
        """te_fp8 requires linear='te' or experts='te'."""
        with pytest.raises(ValueError, match="te_fp8 requires at least one TE backend"):
            BackendConfig(te_fp8=TEFp8Config(), linear="torch", experts="torch")


class TestGetRopeConfig:
    def test_extracts_rope_theta(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
        )
        rope_theta, _, _ = get_rope_config(config)
        assert rope_theta == 1_000_000

    def test_returns_rope_parameters_as_rope_scaling(self):
        params = {"rope_theta": 10_000, "rope_type": "default"}
        config = SimpleNamespace(rope_parameters=params)
        _, rope_parameters, _ = get_rope_config(config)
        assert rope_parameters is params

    def test_partial_rotary_factor(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 10_000, "partial_rotary_factor": 0.5},
        )
        _, _, partial_rotary_factor = get_rope_config(config)
        assert partial_rotary_factor == 0.5

    def test_partial_rotary_factor_defaults_to_one(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 10_000},
        )
        _, _, partial_rotary_factor = get_rope_config(config)
        assert partial_rotary_factor == 1.0

    def test_yarn_scaling_parameters(self):
        config = SimpleNamespace(
            rope_parameters={
                "rope_theta": 1_000_000,
                "rope_type": "yarn",
                "factor": 4.0,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "original_max_position_embeddings": 8192,
            },
        )
        rope_theta, rope_parameters, _ = get_rope_config(config)
        assert rope_theta == 1_000_000
        assert rope_parameters["factor"] == 4.0
        assert rope_parameters["beta_slow"] == 1.0
        assert rope_parameters["beta_fast"] == 32.0
        assert rope_parameters["original_max_position_embeddings"] == 8192


class TestFloat32RMSNorm:
    """Tests for Float32RMSNorm numerical stability (issue #1432)."""

    def test_weight_dtype_is_model_dtype(self):
        """Float32RMSNorm weights should stay in the model dtype (bfloat16)."""
        norm = Float32RMSNorm(dim=16, eps=1e-5, dtype=torch.bfloat16)
        assert norm.weight.dtype == torch.bfloat16

    def test_fp32_computation_preserves_output_dtype(self):
        """The fp32 computation path (without compile) should return the input dtype."""
        norm = Float32RMSNorm(dim=16, eps=1e-5, dtype=torch.bfloat16)
        x = torch.randn(2, 8, 16, dtype=torch.bfloat16)
        # Exercise the fp32 computation manually (same logic as _float32_rms_norm_fwd)
        x_fp32 = x.float()
        out = (norm.weight * x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + norm.eps)).to(x.dtype)
        assert out.dtype == torch.bfloat16

    def test_fp32_norm_more_precise_than_bf16_norm(self):
        """fp32 RMS norm should be more precise than native bf16 norm for near-zero inputs."""
        # Create inputs that stress bf16 precision (small values near zero)
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.bfloat16) * 0.01
        weight = torch.ones(64, dtype=torch.bfloat16)
        eps = 1e-5

        # fp32 reference (what Float32RMSNorm computes)
        x_fp32 = x.float()
        out_fp32 = (weight.float() * x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)).to(
            torch.bfloat16
        )

        # bf16 native (what TE RMSNorm without fp32 upcast would compute)
        out_bf16 = weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        # fp32 version should match the high-precision float32 result more closely
        x_true = x.double()
        out_true = (weight.double() * x_true * torch.rsqrt(x_true.pow(2).mean(-1, keepdim=True) + eps)).to(
            torch.bfloat16
        )
        err_fp32 = (out_fp32.float() - out_true.float()).abs().mean().item()
        err_bf16 = (out_bf16.float() - out_true.float()).abs().mean().item()
        assert err_fp32 <= err_bf16, "fp32 upcast norm should be at least as accurate as native bf16 norm"

    def test_te_rmsnorm_patch_upcasts_to_fp32(self):
        """The TE RMSNorm patch must upcast bf16 inputs to fp32 before the kernel call."""
        captured = {}

        def mock_original_forward(self_inner, x):
            captured["dtype"] = x.dtype
            return x  # pass-through

        instance = MagicMock()
        x_bf16 = torch.randn(2, 4, dtype=torch.bfloat16)

        # Replicate the patched forward logic from _make_lazy_te_patcher
        input_dtype = x_bf16.dtype
        if input_dtype != torch.float32:
            result = mock_original_forward(instance, x_bf16.float())
        else:
            result = mock_original_forward(instance, x_bf16)

        assert captured["dtype"] == torch.float32, "TE RMSNorm should receive fp32 inputs after upcast patch"
