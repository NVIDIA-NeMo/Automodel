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
from unittest.mock import patch

import pytest

from nemo_automodel.components.models.common.utils import (
    BackendConfig,
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
    def test_newer_hf_format_with_rope_parameters(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
        )
        rope_theta, rope_scaling, partial_rotary_factor = get_rope_config(config)
        assert rope_theta == 1_000_000
        assert rope_scaling == {"rope_theta": 1_000_000, "rope_type": "default"}
        assert partial_rotary_factor == 1.0

    def test_no_rope_parameters_attribute(self):
        config = SimpleNamespace(rope_theta=10_000)
        rope_theta, rope_scaling, partial_rotary_factor = get_rope_config(config)
        assert rope_theta == 10_000
        assert rope_scaling == {}
        assert partial_rotary_factor == 1.0

    def test_rope_theta_falls_back_to_config_attribute(self):
        config = SimpleNamespace(
            rope_parameters={"rope_type": "default"},
            rope_theta=500_000,
        )
        rope_theta, _, _ = get_rope_config(config)
        assert rope_theta == 500_000

    def test_rope_theta_from_rope_parameters_preferred(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 1_000_000},
            rope_theta=10_000,
        )
        rope_theta, _, _ = get_rope_config(config)
        assert rope_theta == 1_000_000

    def test_rope_theta_not_on_config_when_in_rope_parameters(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 1_000_000},
        )
        rope_theta, _, _ = get_rope_config(config)
        assert rope_theta == 1_000_000

    def test_partial_rotary_factor_from_rope_parameters(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 10_000, "partial_rotary_factor": 0.5},
        )
        _, _, partial_rotary_factor = get_rope_config(config)
        assert partial_rotary_factor == 0.5

    def test_partial_rotary_factor_falls_back_to_config_attribute(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 10_000},
            partial_rotary_factor=0.25,
        )
        _, _, partial_rotary_factor = get_rope_config(config)
        assert partial_rotary_factor == 0.25

    def test_partial_rotary_factor_defaults_to_one(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 10_000},
        )
        _, _, partial_rotary_factor = get_rope_config(config)
        assert partial_rotary_factor == 1.0

    def test_rope_parameters_none_returns_empty_dict(self):
        config = SimpleNamespace(
            rope_parameters=None,
            rope_theta=10_000,
        )
        rope_theta, rope_scaling, _ = get_rope_config(config)
        assert rope_theta == 10_000
        assert rope_scaling == {}

    def test_rope_parameters_empty_dict(self):
        config = SimpleNamespace(
            rope_parameters={},
            rope_theta=10_000,
        )
        rope_theta, rope_scaling, _ = get_rope_config(config)
        assert rope_theta == 10_000
        assert rope_scaling == {}

    def test_yarn_scaling_through_rope_parameters(self):
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
        rope_theta, rope_scaling, _ = get_rope_config(config)
        assert rope_theta == 1_000_000
        assert rope_scaling["factor"] == 4.0
        assert rope_scaling["beta_slow"] == 1.0
        assert rope_scaling["beta_fast"] == 32.0
        assert rope_scaling["original_max_position_embeddings"] == 8192

    def test_rope_parameters_is_rope_scaling(self):
        params = {"rope_theta": 10_000, "factor": 2.0}
        config = SimpleNamespace(rope_parameters=params)
        _, rope_scaling, _ = get_rope_config(config)
        assert rope_scaling is params

    def test_rope_theta_fallback_when_rope_parameters_lacks_key(self):
        """When rope_scaling is passed to from_pretrained, HuggingFace's
        _init_rope_parameters() creates config.rope_parameters with
        inv_freq and attention_factor but NOT rope_theta. The function
        must fall back to config.rope_theta instead of raising KeyError.
        """
        config = SimpleNamespace(
            rope_parameters={"inv_freq": [0.1, 0.2], "attention_factor": 1.0},
            rope_theta=1_000_000,
        )
        rope_theta, rope_scaling, partial_rotary_factor = get_rope_config(config)
        assert rope_theta == 1_000_000
        assert rope_scaling == {"inv_freq": [0.1, 0.2], "attention_factor": 1.0}
        assert partial_rotary_factor == 1.0

    def test_yarn_parameters_propagated_from_rope_parameters(self):
        """When rope_parameters contains scaling fields (factor, beta_slow,
        beta_fast, original_max_position_embeddings), they must be
        accessible from the returned rope_scaling dict so that
        RotaryEmbedding can use them instead of hardcoded defaults.
        """
        config = SimpleNamespace(
            rope_parameters={
                "rope_theta": 1_000_000,
                "rope_type": "yarn",
                "factor": 5.5,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "original_max_position_embeddings": 40960,
            },
        )
        _, rope_scaling, _ = get_rope_config(config)
        assert rope_scaling.get("factor") == 5.5
        assert rope_scaling.get("original_max_position_embeddings") == 40960
        assert rope_scaling.get("beta_slow") == 1.0
        assert rope_scaling.get("beta_fast") == 32.0
