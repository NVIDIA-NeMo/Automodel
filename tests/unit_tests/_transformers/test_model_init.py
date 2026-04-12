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
    _init_model,
    get_hf_config,
)
from nemo_automodel.components.models.common.utils import BackendConfig


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

        assert config.some_field == {"key": "val"}
        assert "some_field" not in kwargs


class TestBackendDictCoercion:
    """CLI overrides like --model.backend.attn sdpa produce a plain dict; _init_model should coerce it to BackendConfig."""

    def _make_config(self):
        config = MagicMock()
        config.architectures = ["SomeModel"]
        config.torch_dtype = "bfloat16"
        config.name_or_path = "fake/model"
        return config

    def _run_init_model(self, mock_resolve_cls, **extra_kwargs):
        """Helper to call _init_model with a fake model class and capture kwargs."""
        captured_kwargs = {}

        def fake_model_cls(config, **kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        fake_model_cls.__module__ = "nemo_automodel.components.models.fake"
        mock_resolve_cls.return_value = fake_model_cls

        _init_model(
            cls=MagicMock(),
            pretrained_model_name_or_path_or_config=self._make_config(),
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
            quantization_config=None,
            force_hf=False,
            **extra_kwargs,
        )
        return captured_kwargs

    @patch("nemo_automodel._transformers.model_init._download_model_weights")
    @patch("nemo_automodel._transformers.model_init._resolve_custom_model_cls_for_config")
    def test_dict_backend_coerced_to_backend_config(self, mock_resolve_cls, _mock_download):
        """A plain dict backend kwarg should become a BackendConfig with defaults filled in."""
        captured = self._run_init_model(mock_resolve_cls, backend={"attn": "sdpa"})
        defaults = BackendConfig()

        assert isinstance(captured["backend"], BackendConfig)
        assert captured["backend"].attn == "sdpa"
        # Unspecified fields should get their environment-dependent defaults
        assert captured["backend"].rms_norm == defaults.rms_norm
        assert captured["backend"].linear == defaults.linear

    @patch("nemo_automodel._transformers.model_init._download_model_weights")
    @patch("nemo_automodel._transformers.model_init._resolve_custom_model_cls_for_config")
    def test_backend_config_object_passed_through(self, mock_resolve_cls, _mock_download):
        """A proper BackendConfig should be passed through unchanged."""
        original_backend = BackendConfig(attn="te", linear="te")
        captured = self._run_init_model(mock_resolve_cls, backend=original_backend)

        assert captured["backend"] is original_backend

    @patch("nemo_automodel._transformers.model_init._download_model_weights")
    @patch("nemo_automodel._transformers.model_init._resolve_custom_model_cls_for_config")
    def test_no_backend_kwarg_unchanged(self, mock_resolve_cls, _mock_download):
        """When no backend is provided, kwargs should not gain one."""
        captured = self._run_init_model(mock_resolve_cls)

        assert "backend" not in captured


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
