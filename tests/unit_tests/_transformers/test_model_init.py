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

"""Tests for nested config override handling in get_hf_config and _consume_config_overrides."""

from unittest.mock import MagicMock, patch

from nemo_automodel._transformers.model_init import (
    _consume_config_overrides,
    get_hf_config,
)


class TestConsumeConfigOverridesNestedDict:
    """Nested dict overrides should be deep-merged into sub-config objects."""

    def test_nested_dict_deep_merges_into_sub_config(self):
        """text_config={"key": val} should update sub-config fields, not replace the object."""
        sub_config = MagicMock()
        sub_config.to_dict.return_value = {"hidden_size": 2048, "router_aux_loss_coef": 0.001}
        sub_config.hidden_size = 2048
        sub_config.router_aux_loss_coef = 0.001

        config = MagicMock()
        config.to_dict.return_value = {"text_config": {}, "model_type": "some_vlm"}
        config.text_config = sub_config

        kwargs = {"text_config": {"router_aux_loss_coef": 0}}
        _consume_config_overrides(config, kwargs)

        # The override should be applied to the sub-config, not replace it
        assert sub_config.router_aux_loss_coef == 0
        # The sub-config object should NOT be replaced
        assert config.text_config is sub_config
        # hidden_size should be untouched
        assert sub_config.hidden_size == 2048
        # The key should be consumed from kwargs
        assert "text_config" not in kwargs

    def test_nested_dict_replaces_when_no_sub_config(self):
        """If the existing attribute has no to_dict, fall back to setattr."""
        config = MagicMock()
        config.to_dict.return_value = {"some_field": {}}
        config.some_field = "not_a_config_object"

        kwargs = {"some_field": {"key": "val"}}
        _consume_config_overrides(config, kwargs)

        config.__setattr__.assert_any_call("some_field", {"key": "val"})
        assert "some_field" not in kwargs


class TestGetHfConfigNestedKwargs:
    """get_hf_config should filter nested dict kwargs from AutoConfig.from_pretrained."""

    @patch("nemo_automodel._transformers.model_init.resolve_trust_remote_code", return_value=True)
    @patch("nemo_automodel._transformers.model_init.AutoConfig.from_pretrained")
    def test_nested_dict_kwargs_not_passed_to_auto_config(self, mock_from_pretrained, mock_trust):
        """Nested dict kwargs should be filtered out before calling AutoConfig.from_pretrained."""
        mock_from_pretrained.return_value = MagicMock()

        get_hf_config(
            "fake/vlm_model",
            attn_implementation="eager",
            text_config={"router_aux_loss_coef": 0},
            output_hidden_states=True,
        )

        call_kwargs = mock_from_pretrained.call_args[1]
        assert "text_config" not in call_kwargs
        assert call_kwargs["output_hidden_states"] is True
