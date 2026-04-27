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

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from nemo_automodel._transformers.model_init import (
    _consume_config_overrides,
    _has_safetensors,
    _init_model,
    _load_config_with_layer_types_fix,
    _propagate_torch_dtype_to_subconfigs,
    _resolve_model_dir,
    _setup_bnb_loading_kwargs,
    _stream_load_bnb_weights,
    _streaming_bnb_supported,
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


class TestPropagateTorchDtypeToSubconfigs:
    """Nested PretrainedConfig sub-configs must receive the requested torch_dtype."""

    def test_propagates_to_nested_multimodal_subconfigs(self):
        """Gemma4-style VLM configs expose text/vision/audio sub-configs that all need updating."""
        from transformers import PretrainedConfig

        text_config = PretrainedConfig()
        text_config.torch_dtype = torch.bfloat16
        vision_config = PretrainedConfig()
        vision_config.torch_dtype = torch.bfloat16
        audio_config = PretrainedConfig()
        audio_config.torch_dtype = torch.bfloat16

        top = PretrainedConfig()
        top.torch_dtype = torch.bfloat16
        top.text_config = text_config
        top.vision_config = vision_config
        top.audio_config = audio_config

        _propagate_torch_dtype_to_subconfigs(top, torch.float32)

        assert top.torch_dtype == torch.float32
        assert text_config.torch_dtype == torch.float32
        assert vision_config.torch_dtype == torch.float32
        assert audio_config.torch_dtype == torch.float32

    def test_ignores_non_config_attributes(self):
        """Non-config attributes (e.g. plain dicts, ints) must not be traversed or mutated."""
        from transformers import PretrainedConfig

        top = PretrainedConfig()
        top.torch_dtype = torch.bfloat16
        top.some_dict = {"not": "a_config"}
        top.hidden_size = 4096

        _propagate_torch_dtype_to_subconfigs(top, torch.float32)

        assert top.torch_dtype == torch.float32
        assert top.some_dict == {"not": "a_config"}
        assert top.hidden_size == 4096

    def test_handles_cycles(self):
        """Self-referencing configs must not cause infinite recursion."""
        from transformers import PretrainedConfig

        top = PretrainedConfig()
        top.torch_dtype = torch.bfloat16
        top.self_ref = top

        _propagate_torch_dtype_to_subconfigs(top, torch.float32)

        assert top.torch_dtype == torch.float32


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


class TestSetupBnbLoadingKwargs:
    """_setup_bnb_loading_kwargs sets a per-GPU device_map and disables HF async weight loading."""

    def test_sets_device_map_default_when_missing(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
        monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)

        kwargs: dict = {}
        _setup_bnb_loading_kwargs(kwargs)

        assert kwargs["device_map"] == {"": 3}
        assert os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "1"

    def test_respects_existing_device_map(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)

        kwargs = {"device_map": "auto"}
        _setup_bnb_loading_kwargs(kwargs)

        assert kwargs["device_map"] == "auto"

    def test_respects_explicit_env_var_even_when_zero(self, monkeypatch):
        """If the user explicitly set HF_DEACTIVATE_ASYNC_LOAD=0, don't silently flip it to 1."""
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "0")

        _setup_bnb_loading_kwargs({})

        assert os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "0"


class TestHasSafetensors:
    """_has_safetensors detects either a sharded index or a single safetensors file."""

    def test_index_file_present(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text("{}")
        assert _has_safetensors(str(tmp_path)) is True

    def test_single_safetensors_file_present(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert _has_safetensors(str(tmp_path)) is True

    def test_neither_present(self, tmp_path):
        (tmp_path / "pytorch_model.bin").write_bytes(b"")
        assert _has_safetensors(str(tmp_path)) is False


class TestResolveModelDir:
    """_resolve_model_dir passes local dirs through and falls back to offline snapshot_download."""

    def test_local_dir_passthrough(self, tmp_path):
        assert _resolve_model_dir(str(tmp_path)) == str(tmp_path)

    def test_repo_id_triggers_offline_snapshot_download(self, tmp_path):
        with patch("nemo_automodel._transformers.model_init.snapshot_download") as mock_sd:
            mock_sd.return_value = str(tmp_path)
            result = _resolve_model_dir("some/repo-id")

        mock_sd.assert_called_once_with("some/repo-id", local_files_only=True)
        assert result == str(tmp_path)


class _TinyModelOnMeta(nn.Module):
    """Minimal module used to exercise _stream_load_bnb_weights without bnb.

    Construction happens under ``with torch.device("meta")`` so all params/buffers
    start on meta device, mirroring the skeleton produced by _init_model_bnb_streaming.
    """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 8, bias=True)
        self.register_buffer("running_scale", torch.zeros(8))


class _TiedHeadModel(nn.Module):
    """Mimics HF tied-input/output-embedding layout where safetensors stores only one side."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16, bias=False)

    def tie_weights(self):
        self.head.weight = self.embed.weight


def _save_safetensors(path, tensors: dict):
    from safetensors.torch import save_file

    save_file(tensors, str(path))


class TestStreamLoadBnbWeights:
    """_stream_load_bnb_weights loads plain tensors, ties shared weights, and flags missing keys."""

    def test_loads_plain_params_and_buffers_from_meta(self, tmp_path):
        with torch.device("meta"):
            model = _TinyModelOnMeta()

        _save_safetensors(
            tmp_path / "model.safetensors",
            {
                "lin.weight": torch.randn(8, 4),
                "lin.bias": torch.randn(8),
                "running_scale": torch.arange(8, dtype=torch.float32),
            },
        )

        _stream_load_bnb_weights(model, str(tmp_path), torch.device("cpu"), torch.float32)

        assert model.lin.weight.device.type == "cpu"
        assert model.lin.bias.device.type == "cpu"
        assert model.running_scale.device.type == "cpu"
        assert model.lin.weight.dtype == torch.float32
        torch.testing.assert_close(model.running_scale, torch.arange(8, dtype=torch.float32))

    def test_missing_key_raises_runtime_error(self, tmp_path):
        with torch.device("meta"):
            model = _TinyModelOnMeta()

        # Omit "lin.bias" from the shard — it should be flagged as unmaterialized.
        _save_safetensors(
            tmp_path / "model.safetensors",
            {
                "lin.weight": torch.randn(8, 4),
                "running_scale": torch.zeros(8),
            },
        )

        with pytest.raises(RuntimeError, match="lin.bias"):
            _stream_load_bnb_weights(model, str(tmp_path), torch.device("cpu"), torch.float32)

    def test_tied_weights_pass_missing_check_after_tie(self, tmp_path):
        """Only embed.weight is in safetensors; head.weight must be tied post-load, not raise."""
        with torch.device("meta"):
            model = _TiedHeadModel()

        embed_weight = torch.randn(16, 8)
        _save_safetensors(tmp_path / "model.safetensors", {"embed.weight": embed_weight})

        _stream_load_bnb_weights(model, str(tmp_path), torch.device("cpu"), torch.float32)

        assert model.embed.weight.device.type == "cpu"
        assert model.head.weight.device.type == "cpu"
        # The whole point of the tie: both sides must share storage.
        assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()
        torch.testing.assert_close(model.embed.weight.detach(), embed_weight)

    def test_extra_safetensors_key_is_ignored(self, tmp_path):
        with torch.device("meta"):
            model = _TinyModelOnMeta()

        _save_safetensors(
            tmp_path / "model.safetensors",
            {
                "lin.weight": torch.randn(8, 4),
                "lin.bias": torch.randn(8),
                "running_scale": torch.zeros(8),
                "unused_stats.mean": torch.zeros(4),  # not in the model
            },
        )

        # Should complete without raising despite the extra key.
        _stream_load_bnb_weights(model, str(tmp_path), torch.device("cpu"), torch.float32)
        assert model.lin.weight.device.type == "cpu"

    def test_sharded_index_is_followed(self, tmp_path):
        """With a safetensors.index.json, _stream_load_bnb_weights should visit each unique shard."""
        import json

        with torch.device("meta"):
            model = _TinyModelOnMeta()

        _save_safetensors(
            tmp_path / "model-00001-of-00002.safetensors",
            {
                "lin.weight": torch.randn(8, 4),
                "lin.bias": torch.randn(8),
            },
        )
        _save_safetensors(
            tmp_path / "model-00002-of-00002.safetensors",
            {"running_scale": torch.arange(8, dtype=torch.float32)},
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        "lin.weight": "model-00001-of-00002.safetensors",
                        "lin.bias": "model-00001-of-00002.safetensors",
                        "running_scale": "model-00002-of-00002.safetensors",
                    },
                }
            )
        )

        _stream_load_bnb_weights(model, str(tmp_path), torch.device("cpu"), torch.float32)

        torch.testing.assert_close(model.running_scale, torch.arange(8, dtype=torch.float32))
        assert model.lin.weight.device.type == "cpu"
        assert model.lin.bias.device.type == "cpu"


class TestStreamingBnbSupported:
    """The streaming BnB path must skip custom Automodel classes that need a StateDictAdapter."""

    def _make_cls(self, model_cls):
        cfg_type = type("_Cfg", (), {})
        config = cfg_type()

        class _Cls:
            _model_mapping = {cfg_type: model_cls}

        return _Cls, config

    def test_vanilla_hf_class_is_supported(self):
        """A plain nn.Module (no HFCheckpointingMixin) resolves to supported."""
        cls, config = self._make_cls(nn.Linear)  # any class not inheriting the mixin
        assert _streaming_bnb_supported(cls, config) is True

    def test_custom_automodel_class_is_unsupported(self):
        """Classes that carry HFCheckpointingMixin rely on state_dict_adapter; skip streaming."""
        from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin

        class FakeCustomModel(HFCheckpointingMixin, nn.Module):
            pass

        cls, config = self._make_cls(FakeCustomModel)
        assert _streaming_bnb_supported(cls, config) is False

    def test_unknown_config_is_unsupported(self):
        """Missing entry in _model_mapping → unsupported (caller falls back to HF)."""
        cfg_type = type("_Cfg", (), {})

        class _Cls:
            _model_mapping: dict = {}

        assert _streaming_bnb_supported(_Cls, cfg_type()) is False

    def test_model_with_hf_conversion_mapping_is_unsupported(self):
        """HF conversion rules (Mixtral, Qwen MoE, …) reshape legacy safetensors at load
        time. The streaming path can't replay those ops, so it must opt out."""
        cfg_type = type("_Cfg", (), {"model_type": "mixtral"})
        cls, config = self._make_cls(nn.Linear)
        cls._model_mapping = {cfg_type: nn.Linear}
        assert _streaming_bnb_supported(cls, cfg_type()) is False

    def test_model_without_hf_conversion_mapping_is_supported(self):
        """Plain dense models (no MoE reshape, no mixin) should still use streaming."""
        cfg_type = type("_Cfg", (), {"model_type": "llama"})
        cls, config = self._make_cls(nn.Linear)
        cls._model_mapping = {cfg_type: nn.Linear}
        assert _streaming_bnb_supported(cls, cfg_type()) is True


class TestLayerTypesFix:
    """_load_config_with_layer_types_fix must truncate layer_types and resolve the right config class."""

    def _config_dict(self, n_layers=45, n_layer_types=48):
        return {
            "model_type": "step3p5",
            "num_hidden_layers": n_layers,
            "layer_types": ["full_attention"] + ["sliding_attention"] * (n_layer_types - 1),
            "hidden_size": 4096,
            "auto_map": {"AutoConfig": "configuration_step3p5.Step3p5Config"},
        }

    @patch("transformers.dynamic_module_utils.get_class_from_dynamic_module")
    @patch("nemo_automodel._transformers.model_init.PretrainedConfig.get_config_dict")
    def test_truncates_layer_types_via_dynamic_module(self, mock_get_dict, mock_get_cls):
        mock_get_dict.return_value = (self._config_dict(), {})
        built = MagicMock()
        fake_cls = MagicMock()
        fake_cls.from_dict.return_value = built
        mock_get_cls.return_value = fake_cls

        result = _load_config_with_layer_types_fix("stepfun-ai/Step-3.5-Flash", "sdpa", trust_remote_code=True)

        assert result is built
        passed_dict = fake_cls.from_dict.call_args[0][0]
        assert len(passed_dict["layer_types"]) == 45
        assert passed_dict["layer_types"][0] == "full_attention"
        assert fake_cls.from_dict.call_args[1]["attn_implementation"] == "sdpa"
        mock_get_cls.assert_called_once_with("configuration_step3p5.Step3p5Config", "stepfun-ai/Step-3.5-Flash")

    @patch("transformers.models.auto.configuration_auto.CONFIG_MAPPING", new_callable=MagicMock)
    @patch("nemo_automodel._transformers.model_init.PretrainedConfig.get_config_dict")
    def test_resolves_via_config_mapping_when_not_trust_remote_code(self, mock_get_dict, mock_mapping):
        cfg_dict = self._config_dict()
        cfg_dict.pop("auto_map")
        mock_get_dict.return_value = (cfg_dict, {})

        fake_cls = MagicMock()
        fake_cls.from_dict.return_value = "built"
        mock_mapping.get.side_effect = lambda k: fake_cls if k == "step3p5" else None

        result = _load_config_with_layer_types_fix("some/model", "flash_attention_2", trust_remote_code=False)

        assert result == "built"
        passed_dict = fake_cls.from_dict.call_args[0][0]
        assert len(passed_dict["layer_types"]) == 45
        mock_mapping.get.assert_called_once_with("step3p5")

    @patch("transformers.models.auto.configuration_auto.CONFIG_MAPPING", new_callable=MagicMock)
    @patch("nemo_automodel._transformers.model_init.PretrainedConfig.get_config_dict")
    def test_matching_lengths_leaves_layer_types_untouched(self, mock_get_dict, mock_mapping):
        cfg_dict = self._config_dict(n_layers=45, n_layer_types=45)
        original = list(cfg_dict["layer_types"])
        mock_get_dict.return_value = (cfg_dict, {})

        fake_cls = MagicMock()
        fake_cls.from_dict.return_value = MagicMock()
        mock_mapping.get.return_value = fake_cls

        _load_config_with_layer_types_fix("some/model", "sdpa", trust_remote_code=False)

        passed_dict = fake_cls.from_dict.call_args[0][0]
        assert passed_dict["layer_types"] == original

    @patch("transformers.models.auto.configuration_auto.CONFIG_MAPPING", new_callable=MagicMock)
    @patch("nemo_automodel._transformers.model_init.PretrainedConfig.get_config_dict")
    def test_raises_when_config_class_cannot_be_resolved(self, mock_get_dict, mock_mapping):
        cfg_dict = self._config_dict()
        cfg_dict.pop("auto_map")
        cfg_dict["model_type"] = "definitely_not_a_real_model_type_xyz"
        mock_get_dict.return_value = (cfg_dict, {})
        mock_mapping.get.return_value = None

        with pytest.raises(ValueError, match="Could not resolve config class"):
            _load_config_with_layer_types_fix("some/model", "sdpa", trust_remote_code=False)


class TestGetHfConfigLayerTypesRetry:
    """get_hf_config should retry via the layer_types fix helper when AutoConfig raises."""

    @patch("nemo_automodel._transformers.model_init._load_config_with_layer_types_fix")
    @patch("nemo_automodel._transformers.model_init.resolve_trust_remote_code", return_value=True)
    @patch("nemo_automodel._transformers.model_init.AutoConfig.from_pretrained")
    def test_retry_on_layer_types_mismatch(self, mock_from_pretrained, _mock_trust, mock_fix):
        mock_from_pretrained.side_effect = ValueError(
            "`num_hidden_layers` (45) must be equal to the number of layer types (48)."
        )
        fixed_cfg = MagicMock()
        mock_fix.return_value = fixed_cfg

        result = get_hf_config("stepfun-ai/Step-3.5-Flash", "sdpa")

        assert result is fixed_cfg
        mock_fix.assert_called_once()
        call_kwargs = mock_fix.call_args[1]
        assert call_kwargs["trust_remote_code"] is True

    @patch("nemo_automodel._transformers.model_init._load_config_with_layer_types_fix")
    @patch("nemo_automodel._transformers.model_init.resolve_trust_remote_code", return_value=True)
    @patch("nemo_automodel._transformers.model_init.AutoConfig.from_pretrained")
    def test_retry_on_strict_dataclass_validation_error(self, mock_from_pretrained, _mock_trust, mock_fix):
        """huggingface_hub wraps the validator ValueError in a non-ValueError error type."""
        from huggingface_hub.errors import StrictDataclassClassValidationError

        cause = ValueError("`num_hidden_layers` (45) must be equal to the number of layer types (48).")
        mock_from_pretrained.side_effect = StrictDataclassClassValidationError(
            validator="validate_layer_type", cause=cause
        )
        fixed_cfg = MagicMock()
        mock_fix.return_value = fixed_cfg

        result = get_hf_config("stepfun-ai/Step-3.5-Flash", "sdpa")

        assert result is fixed_cfg
        mock_fix.assert_called_once()

    @patch("nemo_automodel._transformers.model_init._load_config_with_layer_types_fix")
    @patch("nemo_automodel._transformers.model_init.resolve_trust_remote_code", return_value=False)
    @patch("nemo_automodel._transformers.model_init.AutoConfig.from_pretrained")
    def test_unrelated_value_error_is_reraised(self, mock_from_pretrained, _mock_trust, mock_fix):
        mock_from_pretrained.side_effect = ValueError("some totally unrelated failure")

        with pytest.raises(ValueError, match="totally unrelated failure"):
            get_hf_config("fake/model", "sdpa")
        mock_fix.assert_not_called()

    @patch("nemo_automodel._transformers.model_init._load_config_with_layer_types_fix")
    @patch("nemo_automodel._transformers.model_init.resolve_trust_remote_code", return_value=False)
    @patch("nemo_automodel._transformers.model_init.AutoConfig.from_pretrained")
    def test_unrecognized_architecture_still_raises_helpful_error(self, mock_from_pretrained, _mock_trust, mock_fix):
        mock_from_pretrained.side_effect = ValueError("Unknown model (fake/model) does not recognize this architecture")

        with pytest.raises(ValueError, match="pip install --upgrade nemo_automodel"):
            get_hf_config("fake/model", "sdpa")
        mock_fix.assert_not_called()
