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

"""Tests for the Hy-MT2 config-shape fingerprint and its routing through the
shared ``_resolve_custom_model_cls_for_config`` entry point."""

from types import SimpleNamespace

from nemo_automodel._transformers.model_init import _resolve_custom_model_cls_for_config
from nemo_automodel.components.models.hy_mt2.dispatch import is_hy_mt2_config


def _hy_mt2_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["HYV3ForCausalLM"],
        model_type="hy_v3",
        hidden_size=2048,
        num_hidden_layers=48,
        num_experts=128,
        expert_hidden_dim=768,
        moe_intermediate_size=768,
        enable_lm_head_fp32=True,
    )


def _hy3_preview_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["HYV3ForCausalLM"],
        model_type="hy_v3",
        hidden_size=4096,
        num_hidden_layers=80,
        num_experts=192,
        moe_intermediate_size=1536,
    )


class TestIsHyMT2Config:
    """Direct tests of the fingerprint predicate."""

    def test_hy_mt2_fingerprint_matches(self):
        assert is_hy_mt2_config(_hy_mt2_config())

    def test_hy3_preview_fingerprint_does_not_match(self):
        assert not is_hy_mt2_config(_hy3_preview_config())

    def test_missing_enable_lm_head_fp32_does_not_match(self):
        config = _hy_mt2_config()
        del config.enable_lm_head_fp32
        assert not is_hy_mt2_config(config)

    def test_wrong_hidden_size_does_not_match(self):
        config = _hy_mt2_config()
        config.hidden_size = 4096
        assert not is_hy_mt2_config(config)

    def test_non_hy_v3_model_type_does_not_match(self):
        config = _hy_mt2_config()
        config.model_type = "llama"
        assert not is_hy_mt2_config(config)


class TestHyMT2ModelResolution:
    """Hy-MT2 shares ``HYV3ForCausalLM`` metadata but needs its own implementation."""

    def test_hy_mt2_config_resolves_to_hy_mt2_model(self):
        from nemo_automodel.components.models.hy_mt2.model import HyMT2ForCausalLM

        assert _resolve_custom_model_cls_for_config(_hy_mt2_config()) is HyMT2ForCausalLM

    def test_hy_v3_config_still_resolves_to_hy_v3_model(self):
        from nemo_automodel.components.models.hy_v3.model import HYV3ForCausalLM

        assert _resolve_custom_model_cls_for_config(_hy3_preview_config()) is HYV3ForCausalLM
