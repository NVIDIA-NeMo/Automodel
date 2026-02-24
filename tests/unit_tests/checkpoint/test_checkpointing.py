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

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    _equally_divide_layers,
    _is_custom_model,
    _model_has_dtensors,
    _reinit_rope_buffers,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import _get_lm_head_weight_and_name


def _make_keys(count: int) -> list[str]:
    return [f"layer.{i}" for i in range(count)]


def _count_by_shard(mapping: dict[str, int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for shard_index in mapping.values():
        counts[shard_index] = counts.get(shard_index, 0) + 1
    return counts


def test_equally_divide_layers_num_shards_gt_num_layers():
    keys = _make_keys(3)

    mapping = _equally_divide_layers(5, keys)

    assert mapping == {keys[0]: 1, keys[1]: 2, keys[2]: 3}
    assert set(mapping.values()) == {1, 2, 3}


def test_equally_divide_layers_num_shards_eq_num_layers():
    keys = _make_keys(4)

    mapping = _equally_divide_layers(4, keys)

    assert mapping == {keys[0]: 1, keys[1]: 2, keys[2]: 3, keys[3]: 4}


def test_equally_divide_layers_num_shards_lt_num_layers():
    keys = _make_keys(10)

    mapping = _equally_divide_layers(3, keys)

    assert _count_by_shard(mapping) == {1: 4, 2: 3, 3: 3}
    assert [mapping[key] for key in keys] == [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_equally_divide_layers_num_shards_one():
    keys = _make_keys(5)

    mapping = _equally_divide_layers(1, keys)

    assert len(mapping) == len(keys)
    assert set(mapping.values()) == {1}


# =============================================================================
# Tests for _get_lm_head_weight_and_name
# =============================================================================


class TestGetLmHeadWeightAndName:
    """Test cases for _get_lm_head_weight_and_name name normalization."""

    def test_normal_model_returns_param_and_name(self):
        """Normal model without _orig_mod. prefix returns (param, 'lm_head.weight')."""
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(4, 4, bias=False)

        param, name = _get_lm_head_weight_and_name(model)

        assert name == "lm_head.weight"
        assert param is model.lm_head.weight

    def test_fp8_compiled_model_strips_orig_mod_prefix(self):
        """FP8/compiled model with _orig_mod. prefix returns stripped name."""
        # Simulate a compiled model where parameters have _orig_mod. prefix
        inner = torch.nn.Module()
        inner.lm_head = torch.nn.Linear(4, 4, bias=False)
        wrapper = torch.nn.Module()
        wrapper._orig_mod = inner

        param, name = _get_lm_head_weight_and_name(wrapper)

        assert name == "lm_head.weight"
        assert "_orig_mod" not in name
        assert param is inner.lm_head.weight

    def test_no_lm_head_returns_none(self):
        """Model without lm_head returns (None, None)."""
        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(4, 4)

        param, name = _get_lm_head_weight_and_name(model)

        assert param is None
        assert name is None

    def test_multiple_orig_mod_prefixes_all_stripped(self):
        """Multiple _orig_mod. prefixes are all stripped by .replace()."""
        # Create a deeply nested _orig_mod structure
        inner = torch.nn.Module()
        inner.lm_head = torch.nn.Linear(4, 4, bias=False)
        mid = torch.nn.Module()
        mid._orig_mod = inner
        outer = torch.nn.Module()
        outer._orig_mod = mid

        param, name = _get_lm_head_weight_and_name(outer)

        assert name == "lm_head.weight"
        assert "_orig_mod" not in name


# =============================================================================
# Tests for _reinit_rope_buffers
# =============================================================================


class TestReinitRopeBuffers:
    """Test cases for _reinit_rope_buffers RoPE buffer reinitialization."""

    def test_non_deci_model_returns_early(self):
        """Non-DeciLM model (e.g. llama) returns early without changes."""
        model = torch.nn.Module()
        config = MagicMock()
        config.model_type = "llama"
        model.config = config

        # Add a rope module that should NOT be touched
        rope = torch.nn.Module()
        rope.inv_freq = torch.ones(4)
        original_inv_freq = rope.inv_freq.clone()
        model.rope = rope

        _reinit_rope_buffers(model, torch.device("cpu"))

        assert torch.equal(model.rope.inv_freq, original_inv_freq)

    def test_deci_model_recomputes_inv_freq(self):
        """DeciLM model with rope modules gets inv_freq recomputed."""
        model = torch.nn.Module()
        config = MagicMock()
        config.model_type = "nemotron-nas"
        model.config = config

        new_inv_freq = torch.tensor([1.0, 2.0, 3.0, 4.0])

        rope = MagicMock()
        rope.rope_init_fn = MagicMock(return_value=(new_inv_freq, None))
        rope.inv_freq = torch.zeros(4)
        rope.rope_kwargs = {"seq_len": 128}
        rope.config = config
        # Make hasattr checks work
        rope.original_inv_freq = None
        del rope.original_inv_freq  # Remove so hasattr returns False

        # Use a real module so named_modules works
        real_model = torch.nn.Module()
        real_model.config = config
        # We need to mock named_modules to return our mock rope
        with patch.object(real_model, "named_modules", return_value=[("", real_model), ("layers.0.rotary", rope)]):
            _reinit_rope_buffers(real_model, torch.device("cpu"))

        rope.rope_init_fn.assert_called_once_with(rope.config, torch.device("cpu"), seq_len=128)
        assert rope.inv_freq is new_inv_freq

    def test_deci_model_updates_original_inv_freq(self):
        """DeciLM model with original_inv_freq gets both buffers updated."""
        model = torch.nn.Module()
        config = MagicMock()
        config.model_type = "nemotron-nas"
        model.config = config

        new_inv_freq = torch.tensor([1.0, 2.0, 3.0])

        rope = MagicMock()
        rope.rope_init_fn = MagicMock(return_value=(new_inv_freq, None))
        rope.inv_freq = torch.zeros(3)
        rope.rope_kwargs = {}
        rope.config = config
        rope.original_inv_freq = torch.zeros(3)

        with patch.object(model, "named_modules", return_value=[("", model), ("layers.0.rotary", rope)]):
            _reinit_rope_buffers(model, torch.device("cpu"))

        assert rope.inv_freq is new_inv_freq
        # original_inv_freq should be a clone of new_inv_freq
        assert torch.equal(rope.original_inv_freq, new_inv_freq)

    def test_deci_model_without_rope_attributes_no_crash(self):
        """DeciLM model without rope_init_fn/inv_freq/rope_kwargs gracefully skips."""
        model = torch.nn.Module()
        config = MagicMock()
        config.model_type = "nemotron-nas"
        model.config = config

        # Add a module without any rope attributes
        model.layer = torch.nn.Linear(4, 4)

        # Should not raise
        _reinit_rope_buffers(model, torch.device("cpu"))

    def test_no_config_returns_early(self):
        """Model without config attribute returns early."""
        model = torch.nn.Module()

        # Should not raise
        _reinit_rope_buffers(model, torch.device("cpu"))

    def test_rope_init_fn_failure_logs_warning(self):
        """If rope_init_fn raises, a warning is logged and other modules continue."""
        model = torch.nn.Module()
        config = MagicMock()
        config.model_type = "nemotron-nas"
        model.config = config

        rope = MagicMock()
        rope.rope_init_fn = MagicMock(side_effect=RuntimeError("bad init"))
        rope.inv_freq = torch.zeros(3)
        rope.rope_kwargs = {}
        rope.config = config

        with patch.object(model, "named_modules", return_value=[("", model), ("layers.0.rotary", rope)]):
            # Should not raise, just log a warning
            _reinit_rope_buffers(model, torch.device("cpu"))


# =============================================================================
# Tests for _is_custom_model
# =============================================================================


class TestIsCustomModel:
    """Test cases for _is_custom_model detection of nemo_automodel custom implementations."""

    def test_plain_nn_module_is_not_custom(self):
        """Standard nn.Module is not a custom model."""
        model = torch.nn.Module()
        assert _is_custom_model(model) is False

    def test_hf_linear_is_not_custom(self):
        """Standard PyTorch modules are not custom models."""
        model = torch.nn.Linear(4, 4)
        assert _is_custom_model(model) is False

    def test_module_from_custom_namespace_is_custom(self):
        """A class whose __module__ starts with nemo_automodel.components.models. is custom."""
        model = torch.nn.Module()
        # Simulate a custom model by patching __module__ on the class's MRO
        FakeCustom = type("FakeCustom", (torch.nn.Module,), {})
        FakeCustom.__module__ = "nemo_automodel.components.models.deepseek_v3.model"
        instance = FakeCustom()
        assert _is_custom_model(instance) is True

    def test_subclass_of_custom_model_is_custom(self):
        """A subclass of a custom model class is also detected as custom."""
        Base = type("Base", (torch.nn.Module,), {})
        Base.__module__ = "nemo_automodel.components.models.kimivl.model"
        Child = type("Child", (Base,), {})
        Child.__module__ = "some_other_module"
        instance = Child()
        assert _is_custom_model(instance) is True

    def test_none_module_attr_does_not_crash(self):
        """Classes where __module__ is None don't cause an error."""
        FakeClass = type("FakeClass", (torch.nn.Module,), {})
        FakeClass.__module__ = None
        instance = FakeClass()
        # Should not raise; the (c.__module__ or "") guard handles None
        assert _is_custom_model(instance) is False

    def test_similar_but_wrong_namespace_is_not_custom(self):
        """A class in a similar but different namespace is not custom."""
        FakeClass = type("FakeClass", (torch.nn.Module,), {})
        FakeClass.__module__ = "nemo_automodel.components.checkpoint.checkpointing"
        instance = FakeClass()
        assert _is_custom_model(instance) is False


# =============================================================================
# Tests for _model_has_dtensors
# =============================================================================


class TestModelHasDtensors:
    """Test cases for _model_has_dtensors detection of DTensor parameters."""

    def test_regular_model_has_no_dtensors(self):
        """A standard model with regular parameters returns False."""
        model = torch.nn.Linear(4, 4)
        assert _model_has_dtensors(model) is False

    def test_empty_model_has_no_dtensors(self):
        """A model with no parameters returns False."""
        model = torch.nn.Module()
        assert _model_has_dtensors(model) is False

    def test_model_with_dtensor_parameter_returns_true(self):
        """A model with a DTensor parameter returns True."""
        model = torch.nn.Module()
        # Create a mock DTensor-like object whose type name is "DTensor"
        DTensorLike = type("DTensor", (), {})
        mock_param = DTensorLike()
        with patch.object(model, "parameters", return_value=iter([mock_param])):
            assert _model_has_dtensors(model) is True

    def test_mixed_params_with_one_dtensor_returns_true(self):
        """If at least one parameter is DTensor, returns True."""
        model = torch.nn.Linear(4, 4)
        DTensorLike = type("DTensor", (), {})
        mock_dtensor = DTensorLike()
        regular_param = torch.nn.Parameter(torch.randn(4))
        with patch.object(model, "parameters", return_value=iter([regular_param, mock_dtensor])):
            assert _model_has_dtensors(model) is True

    def test_all_regular_params_returns_false(self):
        """If all parameters are regular tensors, returns False."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        assert _model_has_dtensors(model) is False


# =============================================================================
# Tests for load_model: custom model uses DCP path, not the fast safetensors path
# =============================================================================


class TestLoadModelCustomModelGuard:
    """Verify that custom models skip the fast safetensors path and use DCP instead.

    The fast safetensors path loads the full state dict directly and uses
    _load_full_state_dict_into_model, which bypasses the state_dict_adapter
    conversion needed by custom MoE models. Custom models must use the
    standard DCP path so that _maybe_adapt_state_dict_to_hf/from_hf handles
    the HF <-> native key and tensor format conversion.
    """

    def _make_checkpointer(self):
        """Create a minimally configured Checkpointer for testing."""
        from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig, Checkpointer

        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="/tmp/test",
            model_save_format="safetensors",
            model_cache_dir="/tmp/cache",
            model_repo_id="test/model",
            save_consolidated=False,
            is_peft=False,
        )
        with patch("torch.distributed.is_initialized", return_value=False):
            return Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)

    @patch("nemo_automodel.components.checkpoint.checkpointing._is_safetensors_checkpoint", return_value=True)
    @patch("nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype")
    @patch("nemo_automodel.components.checkpoint.checkpointing._load_full_state_dict_into_model")
    def test_non_custom_model_uses_fast_path(self, mock_load_full, mock_load_hf, mock_is_st):
        """Non-custom (HF) models use the fast safetensors loading path."""
        checkpointer = self._make_checkpointer()
        model = torch.nn.Linear(4, 4)

        mock_load_hf.return_value = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}

        with (
            patch("os.path.exists", return_value=True),
            patch.object(checkpointer, "_do_load") as mock_dcp_load,
        ):
            checkpointer.load_model(model, model_path="/fake/path", is_init_step=True)

        # Fast path should be used: _load_full_state_dict_into_model called
        mock_load_full.assert_called_once()
        # DCP path should NOT be used
        mock_dcp_load.assert_not_called()

    @patch("nemo_automodel.components.checkpoint.checkpointing._is_safetensors_checkpoint", return_value=True)
    @patch("nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype")
    @patch("nemo_automodel.components.checkpoint.checkpointing._load_full_state_dict_into_model")
    def test_custom_model_skips_fast_path_uses_dcp(self, mock_load_full, mock_load_hf, mock_is_st):
        """Custom models (nemo_automodel.components.models.*) must NOT use the fast path.

        They must use the standard DCP path so that state_dict_adapter handles
        the HF <-> native format conversion (e.g., merging individual MoE expert
        weights into grouped tensors).
        """
        checkpointer = self._make_checkpointer()

        # Create a model class in the custom namespace
        CustomModel = type("CustomModel", (torch.nn.Module,), {})
        CustomModel.__module__ = "nemo_automodel.components.models.kimivl.model"
        model = CustomModel()
        model.layer = torch.nn.Linear(4, 4)

        # Sanity check: model is detected as custom
        assert _is_custom_model(model) is True

        mock_state_dict = {"layer.weight": torch.randn(4, 4), "layer.bias": torch.randn(4)}

        with (
            patch("os.path.exists", return_value=True),
            patch(
                "nemo_automodel.components.checkpoint.checkpointing.ModelState"
            ) as MockModelState,
            patch(
                "nemo_automodel.components.checkpoint.checkpointing._maybe_adapt_state_dict_to_hf",
                side_effect=lambda m, sd, **kw: sd,
            ),
            patch(
                "nemo_automodel.components.checkpoint.checkpointing._maybe_adapt_state_dict_from_hf",
                side_effect=lambda m, sd, **kw: sd,
            ),
            patch.object(checkpointer, "_do_load", return_value=mock_state_dict) as mock_dcp_load,
            patch.object(checkpointer, "_get_storage_reader", return_value=None),
        ):
            mock_model_state = MockModelState.return_value
            mock_model_state.model = [model]
            mock_model_state.state_dict.return_value = mock_state_dict

            checkpointer.load_model(model, model_path="/fake/path", is_init_step=True)

        # Fast path should NOT be used
        mock_load_full.assert_not_called()
        # DCP path should be used
        mock_dcp_load.assert_called_once()


# =============================================================================
# Tests for Checkpointer.initialize_model_weights
# =============================================================================


class TestInitializeModelWeights:
    """Test cases for Checkpointer.initialize_model_weights static method."""

    def _make_meta_model(self):
        """Create a simple model on meta device with an _is_hf_initialized flag."""
        with torch.device("meta"):
            model = torch.nn.Linear(4, 4)
        model._is_hf_initialized = True
        model.config = SimpleNamespace(architectures=["TestModel"])
        return model

    def test_materializes_parameters_to_device(self):
        """Parameters should move from meta device to the target device."""
        model = self._make_meta_model()
        assert model.weight.device.type == "meta"

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        assert model.weight.device.type == "cpu"
        assert model.bias.device.type == "cpu"

    def test_materializes_meta_buffers(self):
        """Meta-device buffers should be materialized to the target device."""
        model = torch.nn.Module()
        model.config = SimpleNamespace(architectures=["TestModel"])
        model.register_buffer("buf", torch.empty(3, device="meta"))

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        assert model.buf.device.type == "cpu"

    def test_resets_is_hf_initialized(self):
        """_is_hf_initialized should be set to False on all submodules."""
        model = self._make_meta_model()
        assert model._is_hf_initialized is True

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        for _, module in model.named_modules():
            if hasattr(module, "_is_hf_initialized"):
                assert module._is_hf_initialized is False

    def test_calls_initialize_weights(self):
        """model.initialize_weights() should be called when available."""
        model = self._make_meta_model()
        model.initialize_weights = MagicMock()

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        model.initialize_weights.assert_called_once()

    def test_warns_when_no_initialize_weights_method(self):
        """Should log a warning when model lacks initialize_weights."""
        model = self._make_meta_model()
        assert not hasattr(model, "initialize_weights")

        with patch("nemo_automodel.components.checkpoint.checkpointing.logging") as mock_logging:
            Checkpointer.initialize_model_weights(model, torch.device("cpu"))
            mock_logging.warning.assert_called_once()

    def test_skips_for_nemotron_v2(self):
        """NemotronHForCausalLM v2 (no n_routed_experts) should skip init."""
        model = self._make_meta_model()
        model.config = SimpleNamespace(architectures=["NemotronHForCausalLM"])
        model._is_hf_initialized = True
        model.initialize_weights = MagicMock()

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        model.initialize_weights.assert_not_called()
        assert model._is_hf_initialized is True

    def test_does_not_skip_for_nemotron_v3_moe(self):
        """NemotronHForCausalLM v3 (with n_routed_experts) should NOT be skipped."""
        model = self._make_meta_model()
        model.config = SimpleNamespace(architectures=["NemotronHForCausalLM"], n_routed_experts=8)
        model.initialize_weights = MagicMock()

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        model.initialize_weights.assert_called_once()

    def test_handles_missing_config_gracefully(self):
        """Model without config.architectures should not raise."""
        with torch.device("meta"):
            model = torch.nn.Linear(4, 4)
        model.config = SimpleNamespace()
        model.initialize_weights = MagicMock()

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        model.initialize_weights.assert_called_once()

    def test_peft_init_method_calls_init_peft_adapters(self):
        """When peft_init_method is provided, _init_peft_adapters should be called."""
        model = self._make_meta_model()
        model.initialize_weights = MagicMock()

        with patch("nemo_automodel.components.checkpoint.checkpointing._init_peft_adapters") as mock_init_peft:
            Checkpointer.initialize_model_weights(model, torch.device("cpu"), peft_init_method="xavier")

        mock_init_peft.assert_called_once_with(model, "xavier")

    def test_peft_init_method_none_skips_init_peft_adapters(self):
        """When peft_init_method is None (default), _init_peft_adapters should NOT be called."""
        model = self._make_meta_model()
        model.initialize_weights = MagicMock()

        with patch("nemo_automodel.components.checkpoint.checkpointing._init_peft_adapters") as mock_init_peft:
            Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        mock_init_peft.assert_not_called()

    def test_load_base_model_does_not_accept_peft_init_method(self):
        """load_base_model should not accept peft_init_method as a parameter."""
        import inspect

        sig = inspect.signature(Checkpointer.load_base_model)
        assert "peft_init_method" not in sig.parameters
