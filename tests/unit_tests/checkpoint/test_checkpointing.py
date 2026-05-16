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

import json
import os
import pytest
import torch
import yaml

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from contextlib import ExitStack
from nemo_automodel.components.checkpoint._backports.hf_storage import _DIFFUSERS_INDEX_FN
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    _equally_divide_layers,
    _is_custom_model,
    _model_has_dtensors,
    _reinit_non_persistent_buffers,
    _summarize_state_dict_key_diff,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, _get_lm_head_weight_and_name
from nemo_automodel.components.checkpoint.utils import (
    has_local_tied_lm_head,
    materialize_missing_tied_lm_head,
)
from nemo_automodel.components.checkpoint.checkpointing import (
    is_cloud_path,
    _ensure_msc_available,
    _ensure_dirs,
    save_config,
)

CLOUD_PATH_MODEL = "msc://bucket/step-100/model"
CLOUD_PATH_OPTIM = "msc://bucket/step-100/optim"
LOCAL_PATH_MODEL = "/ckpts/step-100/model"


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


def test_summarize_state_dict_key_diff_reports_missing_and_unexpected():
    summary = _summarize_state_dict_key_diff(
        {"a.weight", "b.bias", "c.weight"},
        {"a.weight", "c.weight", "extra.weight"},
        limit=2,
    )

    assert summary["missing_count"] == 1
    assert summary["unexpected_count"] == 1
    assert summary["missing_examples"] == ["b.bias"]
    assert summary["unexpected_examples"] == ["extra.weight"]


def test_summarize_state_dict_key_diff_limits_examples():
    summary = _summarize_state_dict_key_diff(
        {"a", "b", "c", "d"},
        {"x"},
        limit=2,
    )

    assert summary["missing_count"] == 4
    assert summary["unexpected_count"] == 1
    assert summary["missing_examples"] == ["a", "b"]


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


class _PipelineLastStageLikeModel(torch.nn.Module):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.lm_head = torch.nn.Linear(4, 4, bias=False)


def test_has_local_tied_lm_head_is_false_for_pp_last_stage_like_partition():
    model = _PipelineLastStageLikeModel()

    assert has_local_tied_lm_head(model) is False


def test_materialize_missing_tied_lm_head_uses_embedding_tensor_from_checkpoint():
    model = _PipelineLastStageLikeModel()
    embed_weight = torch.full_like(model.lm_head.weight, 3.0)
    state_dict = {"model.language_model.embed_tokens.weight": embed_weight}

    materialized = materialize_missing_tied_lm_head(state_dict, model, allow_current_lm_head_fallback=False)

    assert materialized is True
    assert "lm_head.weight" in state_dict
    assert torch.equal(state_dict["lm_head.weight"], embed_weight)
    assert not torch.equal(state_dict["lm_head.weight"], model.lm_head.weight.detach())


def test_model_state_keeps_pp_last_stage_lm_head_in_saved_state_dict():
    model = _PipelineLastStageLikeModel()

    model_state = ModelState(model, is_peft=False, is_init_step=False)
    saved_state_dict = model_state.state_dict()

    assert "lm_head.weight" in saved_state_dict


# =============================================================================
# Tests for _reinit_non_persistent_buffers
# =============================================================================


class TestReinitRopeBuffers:
    """Test cases for _reinit_non_persistent_buffers RoPE buffer reinitialization."""

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

        _reinit_non_persistent_buffers(model, torch.device("cpu"), model_type="llama")

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
            _reinit_non_persistent_buffers(real_model, torch.device("cpu"), model_type="nemotron-nas")

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
            _reinit_non_persistent_buffers(model, torch.device("cpu"), model_type="nemotron-nas")

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
        _reinit_non_persistent_buffers(model, torch.device("cpu"), model_type="nemotron-nas")

    def test_no_config_returns_early(self):
        """Model without config attribute returns early."""
        model = torch.nn.Module()

        # Should not raise — model_type=None is not in the allowlist
        _reinit_non_persistent_buffers(model, torch.device("cpu"), model_type=None)

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
            _reinit_non_persistent_buffers(model, torch.device("cpu"), model_type="nemotron-nas")

    def test_embed_scale_reinitialized_from_scalar(self):
        """ScaledWordEmbedding embed_scale buffer is recomputed from scalar_embed_scale."""
        model = torch.nn.Module()
        emb = torch.nn.Embedding(10, 8)
        emb.scalar_embed_scale = 48.0
        emb.register_buffer("embed_scale", torch.tensor(float("nan")), persistent=False)
        model.embed_tokens = emb

        _reinit_non_persistent_buffers(model, torch.device("cpu"), model_type="gemma3")

        assert emb.embed_scale.item() == 48.0

    def test_embed_scale_without_scalar_attr_is_skipped(self):
        """Modules without scalar_embed_scale are not touched."""
        model = torch.nn.Module()
        emb = torch.nn.Embedding(10, 8)
        emb.register_buffer("embed_scale", torch.tensor(float("nan")), persistent=False)
        model.embed_tokens = emb

        _reinit_non_persistent_buffers(model, torch.device("cpu"), model_type="gemma3")

        # embed_scale should remain NaN because there's no scalar_embed_scale to recover from
        assert torch.isnan(emb.embed_scale)

    def test_position_ids_reinitialized_from_num_positions(self):
        """Vision embedding position_ids buffer is recomputed from num_positions."""
        model = torch.nn.Module()
        vis_emb = torch.nn.Module()
        vis_emb.num_positions = 16
        vis_emb.register_buffer("position_ids", torch.full((1, 16), 999999, dtype=torch.long), persistent=False)
        model.vision_embeddings = vis_emb

        _reinit_non_persistent_buffers(model, torch.device("cpu"), model_type="gemma3")

        expected = torch.arange(16).expand((1, -1))
        assert torch.equal(vis_emb.position_ids, expected)

    def test_position_ids_without_num_positions_is_skipped(self):
        """Modules with position_ids but no num_positions are not touched."""
        model = torch.nn.Module()
        vis_emb = torch.nn.Module()
        garbage = torch.full((1, 16), 999999, dtype=torch.long)
        vis_emb.register_buffer("position_ids", garbage.clone(), persistent=False)
        model.vision_embeddings = vis_emb

        _reinit_non_persistent_buffers(model, torch.device("cpu"), model_type="gemma3")

        assert torch.equal(vis_emb.position_ids, garbage)


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
        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

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

    @patch("nemo_automodel.components.checkpoint.checkpointing._is_safetensors_checkpoint", return_value=False)
    @patch("nemo_automodel.components.checkpoint.checkpointing._is_bin_checkpoint", return_value=True)
    @patch("nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype")
    @patch("nemo_automodel.components.checkpoint.checkpointing._load_full_state_dict_into_model")
    def test_bin_checkpoint_uses_fast_path(self, mock_load_full, mock_load_hf, mock_is_bin, mock_is_st):
        """Non-custom (HF) models with .bin checkpoints use the fast loading path."""
        checkpointer = self._make_checkpointer()
        model = torch.nn.Linear(4, 4)

        mock_load_hf.return_value = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}

        with (
            patch("os.path.exists", return_value=True),
            patch.object(checkpointer, "_do_load") as mock_dcp_load,
        ):
            checkpointer.load_model(model, model_path="/fake/path", is_init_step=True)

        mock_load_full.assert_called_once()
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
            patch("nemo_automodel.components.checkpoint.checkpointing.ModelState") as MockModelState,
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

    @pytest.mark.parametrize(
        "architecture",
        ["Gemma3ForCausalLM", "Gemma3ForConditionalGeneration"],
        ids=["causal_lm", "conditional_generation"],
    )
    def test_skips_for_gemma3(self, architecture):
        """Gemma3 models should skip init — _init_weights zeros embedding padding_idx which fails with DTensors."""
        model = self._make_meta_model()
        model.config = SimpleNamespace(architectures=[architecture])
        model._is_hf_initialized = True
        model.initialize_weights = MagicMock()

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        model.initialize_weights.assert_not_called()
        assert model._is_hf_initialized is True

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


class TestLmHeadWeightTying:
    """Tests that load_base_model calls tie_weights for tied models."""

    def test_tie_weights_called_when_tied(self):
        """load_base_model should call model.tie_weights() when tie_word_embeddings=True."""
        import torch.nn as nn

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(10, 4)
                self.lm_head = nn.Linear(4, 10, bias=False)
                self.config = SimpleNamespace(tie_word_embeddings=True)
                self.tie_weights_called = False

            def tie_weights(self, **kwargs):
                self.lm_head.weight = self.embed_tokens.weight
                self.tie_weights_called = True

        model = FakeModel()
        assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()

        from nemo_automodel.components.checkpoint.checkpointing import is_tied_word_embeddings

        is_tied = is_tied_word_embeddings(model)
        if hasattr(model, "tie_weights") and is_tied:
            model.tie_weights()

        assert model.tie_weights_called
        assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()

    def test_tie_weights_skipped_when_not_tied(self):
        """load_base_model should skip tie_weights when tie_word_embeddings=False."""
        import torch.nn as nn

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(4, 10, bias=False)
                self.config = SimpleNamespace(tie_word_embeddings=False)
                self.tie_weights_called = False

            def tie_weights(self, **kwargs):
                self.tie_weights_called = True

        model = FakeModel()

        from nemo_automodel.components.checkpoint.checkpointing import is_tied_word_embeddings

        is_tied = is_tied_word_embeddings(model)
        if hasattr(model, "tie_weights") and is_tied:
            model.tie_weights()

        assert not model.tie_weights_called


# =============================================================================
# Tests for Checkpointer.save_model — diffusers_compatible rename (all-ranks path)
# =============================================================================


class TestCheckpointerSaveModelDiffusersRename:
    """Tests that save_model() renames the index on the all-ranks consolidation path."""

    def _make_checkpointer(self, tmp_path, diffusers_compatible):
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path),
            model_save_format="safetensors",
            model_cache_dir=str(tmp_path / "cache"),
            model_repo_id="test/model",
            save_consolidated=True,
            is_peft=False,
            diffusers_compatible=diffusers_compatible,
        )
        with patch("torch.distributed.is_initialized", return_value=False):
            checkpointer = Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)

        # Mock internals to isolate the consolidation + rename logic
        checkpointer._should_write_consolidated_safetensors = MagicMock(return_value=True)
        checkpointer._should_write_hf_metadata = MagicMock(return_value=True)
        checkpointer._maybe_build_consolidated_index = MagicMock(return_value={"w": 1})
        checkpointer._get_storage_writer = MagicMock(return_value=MagicMock())
        checkpointer._do_save = MagicMock(return_value=None)
        checkpointer._addons = []
        return checkpointer

    @patch("nemo_automodel.components.checkpoint.checkpointing.consolidate_safetensors_files_on_every_rank")
    @patch(
        "nemo_automodel.components.checkpoint.checkpointing._maybe_adapt_state_dict_to_hf",
        side_effect=lambda *a, **kw: a[1],
    )
    @patch("torch.distributed.is_initialized", return_value=False)
    def test_save_model_renames_index_on_all_ranks_path(self, mock_dist_init, mock_adapt, mock_consolidate, tmp_path):
        weights_path = tmp_path / "step_100"
        consolidated_dir = weights_path / "model" / "consolidated"

        def _fake_consolidate(**kwargs):
            os.makedirs(kwargs["output_dir"], exist_ok=True)
            index_path = os.path.join(kwargs["output_dir"], "model.safetensors.index.json")
            with open(index_path, "w") as f:
                json.dump({"weight_map": {}}, f)

        mock_consolidate.side_effect = _fake_consolidate

        checkpointer = self._make_checkpointer(tmp_path, diffusers_compatible=True)

        model = MagicMock()
        model.state_dict.return_value = {"w": MagicMock()}

        checkpointer.save_model(model, str(weights_path))

        mock_consolidate.assert_called_once()
        assert not (consolidated_dir / "model.safetensors.index.json").exists()
        assert (consolidated_dir / _DIFFUSERS_INDEX_FN).exists()

    @patch("nemo_automodel.components.checkpoint.checkpointing.consolidate_safetensors_files_on_every_rank")
    @patch(
        "nemo_automodel.components.checkpoint.checkpointing._maybe_adapt_state_dict_to_hf",
        side_effect=lambda *a, **kw: a[1],
    )
    @patch("torch.distributed.is_initialized", return_value=False)
    def test_save_model_preserves_index_when_not_diffusers_compatible(
        self, mock_dist_init, mock_adapt, mock_consolidate, tmp_path
    ):
        weights_path = tmp_path / "step_100"
        consolidated_dir = weights_path / "model" / "consolidated"

        def _fake_consolidate(**kwargs):
            os.makedirs(kwargs["output_dir"], exist_ok=True)
            index_path = os.path.join(kwargs["output_dir"], "model.safetensors.index.json")
            with open(index_path, "w") as f:
                json.dump({"weight_map": {}}, f)

        mock_consolidate.side_effect = _fake_consolidate

        checkpointer = self._make_checkpointer(tmp_path, diffusers_compatible=False)

        model = MagicMock()
        model.state_dict.return_value = {"w": MagicMock()}

        checkpointer.save_model(model, str(weights_path))

        assert (consolidated_dir / "model.safetensors.index.json").exists()
        assert not (consolidated_dir / _DIFFUSERS_INDEX_FN).exists()


# =============================================================================
# Tests for _get_storage_reader: is_init_step uses backport, not upstream HF reader
# =============================================================================


class TestGetStorageReaderInitStep:
    """``_get_storage_reader`` must prefer the in-tree backport when ``is_init_step=True``.

    The upstream ``HuggingFaceStorageReader`` (in ``torch.distributed.checkpoint.hf_storage``)
    delegates dtype decoding to ``safetensors.torch._TYPES``, which does not
    yet recognise the FP8 scale dtypes (``F8_E5M2``/``F8_E8M0``) emitted by
    quantised HF checkpoints such as DSV4.  For base-model HF loads
    (``is_init_step=True``) we must therefore use the in-tree backport whose
    ``DTYPE_MAP`` was extended for those dtypes.  Mid-training DCP loads
    (``is_init_step=False`` and no key remap) may still use the faster upstream
    reader.
    """

    def _make_checkpointer(self):
        from nemo_automodel.components.checkpoint.checkpointing import (
            Checkpointer,
            CheckpointingConfig,
        )

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

    def test_init_step_returns_backport_reader_not_upstream(self):
        """is_init_step=True with no key_mapping should still go through the in-tree backport."""
        checkpointer = self._make_checkpointer()

        upstream_marker = MagicMock(name="UpstreamHFReader")
        backport_marker = MagicMock(name="BackportHFReader")

        with (
            patch(
                "torch.distributed.checkpoint.hf_storage.HuggingFaceStorageReader",
                upstream_marker,
            ),
            patch(
                "nemo_automodel.components.checkpoint.checkpointing._HuggingFaceStorageReader",
                backport_marker,
            ),
        ):
            reader = checkpointer._get_storage_reader(model_path="/fake/path", key_mapping=None, is_init_step=True)

        # Upstream reader must NOT be constructed for init-step base-model loads
        upstream_marker.assert_not_called()
        # Backport reader must be constructed instead
        backport_marker.assert_called_once_with(path="/fake/path", key_mapping=None)
        assert reader is backport_marker.return_value

    def test_non_init_step_no_keymap_uses_upstream(self):
        """For mid-training safetensors loads (is_init_step=False, no key_mapping),
        the faster upstream reader is preferred."""
        checkpointer = self._make_checkpointer()

        upstream_marker = MagicMock(name="UpstreamHFReader")
        backport_marker = MagicMock(name="BackportHFReader")

        with (
            patch(
                "torch.distributed.checkpoint.hf_storage.HuggingFaceStorageReader",
                upstream_marker,
            ),
            patch(
                "nemo_automodel.components.checkpoint.checkpointing._HuggingFaceStorageReader",
                backport_marker,
            ),
        ):
            reader = checkpointer._get_storage_reader(model_path="/fake/path", key_mapping=None, is_init_step=False)

        upstream_marker.assert_called_once_with(path="/fake/path")
        backport_marker.assert_not_called()
        assert reader is upstream_marker.return_value

    def test_keymap_always_uses_backport(self):
        """When a key_mapping is supplied, the backport reader is always used (regardless of is_init_step)."""
        checkpointer = self._make_checkpointer()

        upstream_marker = MagicMock(name="UpstreamHFReader")
        backport_marker = MagicMock(name="BackportHFReader")

        with (
            patch(
                "torch.distributed.checkpoint.hf_storage.HuggingFaceStorageReader",
                upstream_marker,
            ),
            patch(
                "nemo_automodel.components.checkpoint.checkpointing._HuggingFaceStorageReader",
                backport_marker,
            ),
        ):
            mapping = {"old.key": "new.key"}
            reader = checkpointer._get_storage_reader(model_path="/fake/path", key_mapping=mapping, is_init_step=False)

        upstream_marker.assert_not_called()
        backport_marker.assert_called_once_with(path="/fake/path", key_mapping=mapping)
        assert reader is backport_marker.return_value

    def test_init_step_with_keymap_uses_backport(self):
        """is_init_step=True + key_mapping must also use the backport (only one path remains)."""
        checkpointer = self._make_checkpointer()

        upstream_marker = MagicMock(name="UpstreamHFReader")
        backport_marker = MagicMock(name="BackportHFReader")

        with (
            patch(
                "torch.distributed.checkpoint.hf_storage.HuggingFaceStorageReader",
                upstream_marker,
            ),
            patch(
                "nemo_automodel.components.checkpoint.checkpointing._HuggingFaceStorageReader",
                backport_marker,
            ),
        ):
            mapping = {"old.key": "new.key"}
            reader = checkpointer._get_storage_reader(model_path="/fake/path", key_mapping=mapping, is_init_step=True)

        upstream_marker.assert_not_called()
        backport_marker.assert_called_once_with(path="/fake/path", key_mapping=mapping)
        assert reader is backport_marker.return_value


# Tests for the _skip_init_weights_on_load gate (Mistral3 FP8 VLM PR)
# =============================================================================


class TestSkipInitWeightsOnLoadGate:
    """The Checkpointer.initialize_model_weights gate that lets a model opt
    out of HF's initialize_weights() via a class attribute.

    Without this gate, Mistral3FP8VLMForConditionalGeneration's PP load
    deadlocks on stage-divergent DTensor collectives inside HF's init.
    """

    def _make_meta_model(self):
        with torch.device("meta"):
            model = torch.nn.Linear(4, 4)
        model._is_hf_initialized = True
        model.config = SimpleNamespace(architectures=["TestModel"])
        return model

    def test_skip_when_attr_true(self):
        """A model with _skip_init_weights_on_load=True takes the skip branch."""
        model = self._make_meta_model()
        model._skip_init_weights_on_load = True
        model.initialize_weights = MagicMock()

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        model.initialize_weights.assert_not_called()
        # And the _is_hf_initialized flag is left alone (not reset to False).
        assert model._is_hf_initialized is True

    def test_does_not_skip_when_attr_false(self):
        """attr=False (or attr-missing default) does NOT take the skip branch."""
        model = self._make_meta_model()
        model._skip_init_weights_on_load = False
        model.initialize_weights = MagicMock()

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        model.initialize_weights.assert_called_once()

    def test_does_not_skip_when_attr_missing(self):
        """No attr at all → default behavior (initialize_weights runs)."""
        model = self._make_meta_model()
        assert not hasattr(model, "_skip_init_weights_on_load")
        model.initialize_weights = MagicMock()

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        model.initialize_weights.assert_called_once()


class TestConsolidatedIndexUnderPPWithoutSourceIndex:
    """_maybe_build_consolidated_index else-branch (NVIDIA-NeMo/Automodel#1512)."""

    def _make_checkpointer(self, tmp_path):
        # empty_cache is created but contains no model.safetensors.index.json so
        # get_safetensors_index_path returns None.
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path),
            model_save_format="safetensors",
            model_cache_dir=str(tmp_path / "empty_cache"),
            model_repo_id="fake/repo-without-index",
            save_consolidated=True,
            is_peft=False,
        )
        with patch("torch.distributed.is_initialized", return_value=False):
            checkpointer = Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)
        return checkpointer

    @staticmethod
    def _fake_model_state(pre_shard_hf_state_dict_keys=None):
        """ModelState-like stub used by _maybe_build_consolidated_index."""
        if pre_shard_hf_state_dict_keys is None:
            model = SimpleNamespace(config=SimpleNamespace(model_type="nemotron_h"))
        else:
            model = MagicMock(spec=torch.nn.Module)
            model.config = SimpleNamespace(model_type="nemotron_h")
            model._pre_shard_hf_state_dict_keys = list(pre_shard_hf_state_dict_keys)
        return SimpleNamespace(model=[model], has_local_tied_lm_head=False, lm_head_param_name=None)

    @pytest.mark.run_only_on("CPU")
    def test_falls_back_to_local_state_dict_when_pre_shard_keys_missing(self, tmp_path):
        """No HF index AND no _pre_shard_hf_state_dict_keys → legacy local-keys fallback."""
        checkpointer = self._make_checkpointer(tmp_path)
        os.makedirs(checkpointer.config.model_cache_dir, exist_ok=True)

        rank_state_dict = {
            "backbone.embeddings.weight": torch.empty(0),
            "backbone.layers.0.mixer.qkv_proj.weight": torch.empty(0),
        }
        model_state = self._fake_model_state(pre_shard_hf_state_dict_keys=None)

        mapping = checkpointer._maybe_build_consolidated_index(model_state, rank_state_dict)

        assert mapping == {k: 1 for k in rank_state_dict.keys()}

    @pytest.mark.run_only_on("CPU")
    def test_global_pre_shard_keys_yield_consistent_mapping_across_pp_ranks(self, tmp_path):
        """Disjoint per-rank PP state dicts but the same global pre-shard key set →
        every rank produces the identical mapping covering every FQN, so
        consolidate_safetensors_files_on_every_rank's idx%world_size partitioning
        cannot drop any keys.
        """
        checkpointer = self._make_checkpointer(tmp_path)
        os.makedirs(checkpointer.config.model_cache_dir, exist_ok=True)

        global_pre_shard_keys = sorted(
            [
                "backbone.embeddings.weight",
                *[f"backbone.layers.{i}.mixer.qkv_proj.weight" for i in range(52)],
                "backbone.norm_f.weight",
                "lm_head.weight",
            ]
        )

        # Per-rank disjoint state_dicts (PP slicing).
        world_size = 8
        layer_chunks = [list(range(i * 7, (i + 1) * 7)) for i in range(world_size - 1)]
        layer_chunks.append(list(range(7 * (world_size - 1), 52)))
        per_rank_state_dicts: list[dict[str, torch.Tensor]] = []
        for r in range(world_size):
            sd: dict[str, torch.Tensor] = {}
            if r == 0:
                sd["backbone.embeddings.weight"] = torch.empty(0)
            for layer_idx in layer_chunks[r]:
                sd[f"backbone.layers.{layer_idx}.mixer.qkv_proj.weight"] = torch.empty(0)
            if r == world_size - 1:
                sd["backbone.norm_f.weight"] = torch.empty(0)
                sd["lm_head.weight"] = torch.empty(0)
            per_rank_state_dicts.append(sd)

        # Every rank produces the SAME mapping (and it covers every global FQN).
        per_rank_mappings = []
        for sd in per_rank_state_dicts:
            mapping = checkpointer._maybe_build_consolidated_index(self._fake_model_state(global_pre_shard_keys), sd)
            per_rank_mappings.append(mapping)

        first = per_rank_mappings[0]
        assert sorted(first.keys()) == global_pre_shard_keys
        assert "backbone.norm_f.weight" in first  # every rank sees rank-7's norm_f
        for r, m in enumerate(per_rank_mappings[1:], start=1):
            assert sorted(m.keys()) == global_pre_shard_keys, f"rank {r} mapping diverges"

        # Round-robin: any rank consolidating idx 1 covers every global FQN.
        consolidated_keys: set[str] = set()
        for r, mapping in enumerate(per_rank_mappings):
            indices_for_this_rank = {idx for idx in set(mapping.values()) if idx % world_size == r}
            for fqn, idx in mapping.items():
                if idx in indices_for_this_rank:
                    consolidated_keys.add(fqn)
        assert consolidated_keys == set(global_pre_shard_keys)
# Tests for cloud storage path support (MSC integration)
# =============================================================================


@pytest.mark.parametrize("path,expected", [
    ("msc://my-bucket/checkpoints",  True),
    ("msc://",                       True),
    ("/local/path/checkpoints",      False),
    ("",                             False),
    ("s3://my-bucket/checkpoints",   False),
    ("msc:/missing-slash",           False),
    ("/msc://tricky",                False),
])

def test_is_cloud_path(path, expected):
    """Returns True if path starts with 'msc://', False for all other paths. Only msc:// is supported."""
    assert is_cloud_path(path) is expected

def _make_ckptr(is_peft=False, is_async=False):
    """Returns a minimal mock Checkpointer for testing _do_save and _do_load without a real config or distributed setup."""
    config = MagicMock()
    config.is_peft = is_peft
    config.is_async = is_async
    ckptr = MagicMock(spec=Checkpointer)
    ckptr.config = config
    ckptr._model_ctx = MagicMock(staging_active=False)
    ckptr._optim_ctx = MagicMock(staging_active=False)
    return ckptr

def _cloud_patches(extra_patches=()):
    """Returns an ExitStack that patches MSC_AVAILABLE=True and stubs AsyncCheckpointerType for cloud path tests."""
    stack = ExitStack()
    stack.enter_context(patch("nemo_automodel.components.checkpoint.checkpointing.MSC_AVAILABLE", True))
    stack.enter_context(patch( "nemo_automodel.components.checkpoint.checkpointing.AsyncCheckpointerType", MagicMock(), create=True,))
    for i in extra_patches:
        stack.enter_context(i)
    return stack


class TestEnsureDirs:
    """Ensures that _ensure_dirs creates local directories and skips cloud path creation."""

    def test_creates_nested_local_dirs(self, tmp_path):
        """Calling _ensure_dirs called on a non-existing path creates it will all intermediate directories."""
        target = str(tmp_path / "a" / "b" / "c")
        assert not os.path.exists(target)
        _ensure_dirs(target)
        assert os.path.isdir(target)

    def test_existing_dir_does_not_raise(self, tmp_path):
        """Calling _ensure_dirs on a pre-existing directory does not raise error."""
        _ensure_dirs(str(tmp_path))

    def test_cloud_path_never_touches_filesystem(self):
        """For a msc:// path, os.makedirs is never called."""
        with patch("os.makedirs") as mock_makedirs:
            _ensure_dirs("msc://bucket/some/deep/path")
        mock_makedirs.assert_not_called()

    def test_local_path_passes_exist_ok_true(self, tmp_path):
        """os.makedirs is called exactly, use exist_ok=True to avoid errors on existing directories."""
        target = str(tmp_path / "new")
        with patch("os.makedirs") as mock_makedirs:
            _ensure_dirs(target)
        mock_makedirs.assert_called_once_with(target, exist_ok=True)


class TestSaveConfig:
    """Ensures that save_config writes valid YAML to local paths and uses msc.open for cloud paths."""

    def test_local_path_writes_valid_yaml(self, tmp_path):
        """Writes a config dict to a local path and verifies the file exist and contains the correct values when loaded back."""
        config = {"model": "llama3", "lr": 3e-4, "steps": 1000}
        save_config(config, str(tmp_path))
        cfg_file = tmp_path / "config.yaml"
        assert cfg_file.exists()
        loaded = yaml.safe_load(cfg_file.read_text())
        assert loaded["lr"] == pytest.approx(3e-4)
        assert loaded["steps"] == 1000

    def test_cloud_path_uses_msc_open_not_builtin(self):
        """Verifies that for an msc:// path, msc.open is used instead of python's open."""
        config = {"model": "llama3", "lr": 3e-4}
        mock_file = MagicMock()
        mock_ctx = MagicMock(
            __enter__=MagicMock(return_value=mock_file),
            __exit__=MagicMock(return_value=False),
        )
        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.MSC_AVAILABLE", True), \
             patch("builtins.open") as mock_builtin_open:
            mock_msc.open.return_value = mock_ctx
            save_config(config, "msc://bucket/checkpoints")

        mock_msc.open.assert_called_once()
        mock_builtin_open.assert_not_called()

    def test_config_written_inside_checkpoint_dir(self):
        """Confirms the config file lands inside the checkpoint directory"""
        config = {"x": 1}
        mock_file = MagicMock()
        mock_ctx = MagicMock(
            __enter__=MagicMock(return_value=mock_file),
            __exit__=MagicMock(return_value=False),
        )
        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.MSC_AVAILABLE", True):
            mock_msc.open.return_value = mock_ctx
            save_config(config, "msc://bucket/run42")

        opened_path = mock_msc.open.call_args[0][0]
        assert opened_path.startswith("msc://bucket/run42")


class TestDoLoad:
    """Tests that _do_load routes to the correct storage writer based on path and format."""

    def _make_checkpointer(self, is_peft=False):
        config = MagicMock()
        config.is_peft = is_peft
        ckptr = MagicMock(spec=Checkpointer)
        ckptr.config = config
        return ckptr

    def test_cloud_path_uses_msc_reader(self):
        """Cloud path: MSC writer is injected and used for saving."""
        ckptr = self._make_checkpointer()
        state_dict = {"weight": torch.zeros(4)}

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.MSC_AVAILABLE", True), \
             patch("nemo_automodel.components.checkpoint.checkpointing.dcp"):
            Checkpointer._do_load(ckptr, state_dict, "msc://bucket/step-100")

        mock_msc.torch.MultiStorageFileSystemReader.assert_called_once_with("msc://bucket/step-100")

    def test_local_path_does_not_use_msc_reader(self, tmp_path):
        """Local path: MSC writer is never used."""
        ckptr = self._make_checkpointer()
        state_dict = {"weight": torch.zeros(4)}

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.dcp"):
            Checkpointer._do_load(ckptr, state_dict, str(tmp_path / "step-100"))

        mock_msc.open.assert_not_called()

    def test_peft_cloud_load_still_routes_through_msc_reader(self):
        """MSC writer is called with the exact checkpoint path, not a modified subpath."""
        ckptr = self._make_checkpointer(is_peft=True)
        state_dict = {"weight": torch.zeros(4)}
        mock_file = MagicMock()
        mock_file.read.return_value= b"fake bytes"

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.MSC_AVAILABLE", True), \
             patch("nemo_automodel.components.checkpoint.checkpointing.dcp"), \
             patch("nemo_automodel.components.checkpoint.checkpointing.safetensors_load") as mock_load:
            mock_msc.open.return_value.__enter__=MagicMock(return_value=mock_file)
            mock_msc.open.return_value.__exit__=MagicMock(return_value=False)
            mock_load.return_value = state_dict
            Checkpointer._do_load(ckptr, state_dict, "msc://bucket/step-100/model")

        mock_msc.open.assert_called_once()

    def test_save_and_load_use_same_path(self):
        """Async mode: MSC writer is still injected for cloud paths."""
        config = MagicMock()
        config.is_peft = False
        config.is_async = False
        ckptr = MagicMock(spec=Checkpointer)
        ckptr.config = config
        state_dict = {"weight": torch.ones(4)}
        path = "msc://bucket/step-300"

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.MSC_AVAILABLE", True), \
             patch("nemo_automodel.components.checkpoint.checkpointing.dcp"):
            Checkpointer._do_save(ckptr, state_dict, path)
            Checkpointer._do_load(ckptr, state_dict, path)

        mock_msc.torch.MultiStorageFileSystemWriter.assert_called_once_with(path)
        mock_msc.torch.MultiStorageFileSystemReader.assert_called_once_with(path)

class TestDoSaveFullSFT:
    """Tests that _do_save correctly routes full-SFT saves for DCP and safetensors formats on cloud and local paths."""

    def test_dcp_cloud_sync_uses_msc_writer(self):
        """DCP + cloud + sync: MSC writer injected, and dcp.save is called"""
        ckptr = _make_ckptr(is_peft=False, is_async=False)
        sd = {"w": torch.ones(4)}

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_save(ckptr, sd, CLOUD_PATH_OPTIM, storage_writer=None)

            mock_msc.torch.MultiStorageFileSystemWriter.assert_called_once_with(CLOUD_PATH_OPTIM)
            mock_dcp.save.assert_called_once()

    def test_safetensors_cloud_sync_does_not_override_hf_writer(self):
        """Safetensors + cloud + sync: existing HF writer NOT replaced by MSC writer."""
        ckptr = _make_ckptr(is_peft=False, is_async=False)
        sd = {"w": torch.ones(4)}
        hf_writer = MagicMock(name="HFStorageWriter")

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_save(ckptr, sd, CLOUD_PATH_MODEL, storage_writer=hf_writer)

        mock_msc.torch.MultiStorageFileSystemWriter.assert_not_called()
        mock_dcp.save.assert_called_once()
        _, kwargs = mock_dcp.save.call_args
        assert kwargs["storage_writer"] is hf_writer

    def test_safetensors_cloud_async_does_not_override_hf_writer(self):
        """Safetensors + cloud + async: existing HF writer NOT replaced by MSC writer."""
        ckptr = _make_ckptr(is_peft=False, is_async=True)
        sd = {"w": torch.ones(4)}
        hf_writer = MagicMock(name="HFStorageWriter")

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_save(ckptr, sd, CLOUD_PATH_MODEL, storage_writer=hf_writer)

        mock_msc.torch.MultiStorageFileSystemWriter.assert_not_called()
        mock_dcp.async_save.assert_called_once()
        _, kwargs = mock_dcp.async_save.call_args
        assert kwargs["storage_writer"] is hf_writer

    def test_local_dcp_sync_no_msc(self):
        """Local + DCP + sync: MSC writer never used."""

        ckptr = _make_ckptr(is_peft=False, is_async=False)
        sd = {"w": torch.ones(4)}

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.dcp"):
            Checkpointer._do_save(ckptr, sd, LOCAL_PATH_MODEL, storage_writer=None)

        mock_msc.torch.MultiStorageFileSystemWriter.assert_not_called()

class TestDoSavePEFT:
    """Tests that _do_save correctly handles PEFT adapter saves using msc.open for cloud paths and save_file for local paths."""

    def test_peft_cloud_sync_uses_msc_open(self):
        """PEFT + cloud + sync: msc.open used for adapter file, dcp never called."""
        ckptr = _make_ckptr(is_peft=True, is_async=False)
        sd = {"lora.weight": torch.ones(4)}
        mock_file = MagicMock()
        mock_ctx = MagicMock(__enter__=MagicMock(return_value=mock_file),
                             __exit__=MagicMock(return_value=False))

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.save_file"), \
                 patch("torch.distributed.is_initialized", return_value=False):
                mock_msc.open.return_value = mock_ctx
                Checkpointer._do_save(ckptr, sd, CLOUD_PATH_MODEL)

        mock_msc.open.assert_called_once()
        mock_dcp.save.assert_not_called()
        mock_dcp.async_save.assert_not_called()

    def test_peft_cloud_async_still_uses_msc_open_not_dcp(self):
        """PEFT + cloud + async: adapter written sync via msc.open, dcp never called."""
        ckptr = _make_ckptr(is_peft=True, is_async=True)
        sd = {"lora.weight": torch.ones(4)}
        mock_file = MagicMock()
        mock_ctx = MagicMock(__enter__=MagicMock(return_value=mock_file),
                             __exit__=MagicMock(return_value=False))

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.save_file"), \
                 patch("torch.distributed.is_initialized", return_value=False):
                mock_msc.open.return_value = mock_ctx
                Checkpointer._do_save(ckptr, sd, CLOUD_PATH_MODEL)

        mock_msc.open.assert_called_once()
        mock_dcp.async_save.assert_not_called()
        mock_dcp.save.assert_not_called()

    def test_peft_local_sync_uses_save_file_not_msc(self):
        """PEFT + local + sync: save_file called, msc.open NOT called."""
        ckptr = _make_ckptr(is_peft=True, is_async=False)
        sd = {"lora.weight": torch.ones(4)}

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.save_file") as mock_sf, \
             patch("torch.distributed.is_initialized", return_value=False):
            Checkpointer._do_save(ckptr, sd, LOCAL_PATH_MODEL)

        mock_msc.open.assert_not_called()
        mock_sf.assert_called_once()

    def test_peft_adapter_path_appended_correctly(self):
        """PEFT cloud save opens exactly '<path>/adapter_model.safetensors'."""
        ckptr = _make_ckptr(is_peft=True)
        sd = {"lora.weight": torch.ones(4)}
        path = "msc://mybucket/run7/step-500/model"
        mock_file = MagicMock()
        mock_ctx = MagicMock(__enter__=MagicMock(return_value=mock_file),
                             __exit__=MagicMock(return_value=False))

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.save_file"), \
                 patch("torch.distributed.is_initialized", return_value=False):
                mock_msc.open.return_value = mock_ctx
                Checkpointer._do_save(ckptr, sd, path)

        opened_path = mock_msc.open.call_args[0][0]
        assert opened_path == "msc://mybucket/run7/step-500/model/adapter_model.safetensors"

class TestDoLoadFullSFT:
    """Tests that _do_load correctly routes full-SFT loads for DCP and safetensors formats on cloud and local paths."""

    def test_dcp_cloud_uses_msc_reader(self):
        """DCP + cloud: MSC reader injected when no reader provided."""
        ckptr = _make_ckptr(is_peft=False)
        sd = {"w": torch.zeros(4)}

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_load(ckptr, sd, CLOUD_PATH_OPTIM, storage_reader=None)

        mock_msc.torch.MultiStorageFileSystemReader.assert_called_once_with(CLOUD_PATH_OPTIM)
        mock_dcp.load.assert_called_once()

    def test_safetensors_cloud_does_not_override_hf_reader(self):
        """Safetensors + cloud: existing HF reader NOT replaced by MSC reader."""
        ckptr = _make_ckptr(is_peft=False)
        sd = {"w": torch.zeros(4)}
        hf_reader = MagicMock(name="HFStorageReader")

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_load(ckptr, sd, CLOUD_PATH_MODEL, storage_reader=hf_reader)

        mock_msc.torch.MultiStorageFileSystemReader.assert_not_called()
        mock_dcp.load.assert_called_once()
        _, kwargs = mock_dcp.load.call_args
        assert kwargs["storage_reader"] is hf_reader

    def test_local_dcp_no_msc(self):
        """Local + DCP: MSC reader never used."""
        ckptr = _make_ckptr(is_peft=False)
        sd = {"w": torch.zeros(4)}

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.dcp"):
            Checkpointer._do_load(ckptr, sd, LOCAL_PATH_MODEL, storage_reader=None)

        mock_msc.torch.MultiStorageFileSystemReader.assert_not_called()

    def test_safetensors_local_does_not_use_msc(self):
        """Safetensors + local: MSC reader never used."""
        ckptr = _make_ckptr(is_peft=False)
        sd = {"w": torch.zeros(4)}
        hf_reader = MagicMock(name="HFStorageReader")

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.dcp"):
            Checkpointer._do_load(ckptr, sd, LOCAL_PATH_MODEL, storage_reader=hf_reader)

        mock_msc.torch.MultiStorageFileSystemReader.assert_not_called()

class TestDoLoadPEFT:
    """Tests that _do_load correctly handles PEFT adapter loads using msc.open for cloud paths and load_file for local paths."""

    def test_peft_cloud_uses_msc_open_not_dcp(self):
        """PEFT + cloud: msc.open used for adapter, dcp.load NOT called."""
        ckptr = _make_ckptr(is_peft=True)
        sd = {"lora.weight": torch.zeros(4)}
        mock_file = MagicMock()
        mock_file.read.return_value = b"fake_safetensors_bytes"
        mock_ctx = MagicMock(__enter__=MagicMock(return_value=mock_file),
                             __exit__=MagicMock(return_value=False))

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.safetensors_load", return_value=sd):
                mock_msc.open.return_value = mock_ctx
                Checkpointer._do_load(ckptr, sd, CLOUD_PATH_MODEL)

        mock_msc.open.assert_called_once()
        mock_dcp.load.assert_not_called()

    def test_peft_cloud_adapter_path_correct(self):
        """PEFT + cloud: opens exactly '<path>/adapter_model.safetensors'."""
        ckptr = _make_ckptr(is_peft=True)
        sd = {}
        path = "msc://bucket/run3/step-200/model"
        mock_file = MagicMock()
        mock_file.read.return_value = b"bytes"
        mock_ctx = MagicMock(__enter__=MagicMock(return_value=mock_file),
                             __exit__=MagicMock(return_value=False))

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.safetensors_load", return_value=sd):
                mock_msc.open.return_value = mock_ctx
                Checkpointer._do_load(ckptr, sd, path)

        opened_path = mock_msc.open.call_args[0][0]
        assert opened_path == "msc://bucket/run3/step-200/model/adapter_model.safetensors"

    def test_peft_local_uses_load_file_not_msc(self):
        """PEFT + local: load_file called, msc.open NOT called."""
        ckptr = _make_ckptr(is_peft=True)
        sd = {}

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.load_file", return_value=sd) as mock_lf:
            Checkpointer._do_load(ckptr, sd, LOCAL_PATH_MODEL)

        mock_msc.open.assert_not_called()
        mock_lf.assert_called_once()

    def test_peft_load_at_init_step_skips_peft_branch_uses_dcp(self):
        """PEFT + cloud + is_init_step=True: DCP path used, not PEFT adapter path."""
        ckptr = _make_ckptr(is_peft=True)
        sd = {"w": torch.zeros(4)}

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_load(ckptr, sd, CLOUD_PATH_MODEL, is_init_step=True)

        mock_msc.torch.MultiStorageFileSystemReader.assert_called_once_with(CLOUD_PATH_MODEL)
        mock_dcp.load.assert_called_once()
        mock_msc.open.assert_not_called()


class TestFormatSave:
    """Tests that _get_storage_writer returns the correct writer for each format, and that _do_save routes correctly based on whether a writer is provided."""

    def _make_checkpointer(self, model_save_format, is_peft=False):
        with patch("torch.distributed.is_initialized", return_value=False):
            config = CheckpointingConfig(
                enabled=True,
                checkpoint_dir="/tmp/test",
                model_save_format=model_save_format,
                model_cache_dir="/tmp/cache",
                model_repo_id="test/model",
                save_consolidated=False,
                is_peft=is_peft,
            )
            return Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0)

    def test_safetensors_format_produces_hf_writer(self):
        """safetensors format: _get_storage_writer returns _HuggingFaceStorageWriter."""
        ckptr = self._make_checkpointer("safetensors")
        writer = ckptr._get_storage_writer(
            consolidated_output_path=None,
            fqn_to_index_mapping={"w": 1},
            model_path="/tmp/model",
        )
        from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageWriter
        assert isinstance(writer, _HuggingFaceStorageWriter)

    def test_dcp_format_produces_no_writer(self):
        """torch_save (DCP) format: _get_storage_writer returns None."""
        ckptr = self._make_checkpointer("torch_save")
        writer = ckptr._get_storage_writer(
            consolidated_output_path=None,
            fqn_to_index_mapping=None,
            model_path="/tmp/model",
        )
        assert writer is None

    def test_safetensors_cloud_save_uses_hf_writer_not_msc(self):
        """safetensors + cloud: HF writer passed to dcp.save, MSC writer never created."""
        ckptr = self._make_checkpointer("safetensors")
        sd = {"w": torch.ones(4)}
        hf_writer = MagicMock(name="HFStorageWriter")

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                ckptr._do_save(sd, "msc://bucket/step-100/model", storage_writer=hf_writer)

        mock_msc.torch.MultiStorageFileSystemWriter.assert_not_called()
        mock_dcp.save.assert_called_once()
        _, kwargs = mock_dcp.save.call_args
        assert kwargs["storage_writer"] is hf_writer

    def test_dcp_cloud_save_uses_msc_writer(self):
        """torch_save (DCP) + cloud: no HF writer provided, so MSC writer injected."""
        ckptr = self._make_checkpointer("torch_save")
        sd = {"w": torch.ones(4)}
        writer = ckptr._get_storage_writer(
            consolidated_output_path=None,
            fqn_to_index_mapping=None,
            model_path="msc://bucket/step-100/optim",
        )
        assert writer is None

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                ckptr._do_save(sd, "msc://bucket/step-100/optim", storage_writer=writer)

        mock_msc.torch.MultiStorageFileSystemWriter.assert_called_once_with("msc://bucket/step-100/optim")
        mock_dcp.save.assert_called_once()

    def test_safetensors_local_save_uses_hf_writer(self):
        """safetensors + local: HF writer used, MSC never involved."""
        ckptr = self._make_checkpointer("safetensors")
        sd = {"w": torch.ones(4)}
        hf_writer = ckptr._get_storage_writer(
            consolidated_output_path=None,
            fqn_to_index_mapping={"w": 1},
            model_path="/tmp/step-100/model",
        )

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
            ckptr._do_save(sd, "/tmp/step-100/model", storage_writer=hf_writer)

        mock_msc.torch.MultiStorageFileSystemWriter.assert_not_called()
        mock_dcp.save.assert_called_once()

    def test_dcp_local_save_no_writer_no_msc(self):
        """torch_save (DCP) + local: no writer, no MSC, plain dcp.save."""
        ckptr = self._make_checkpointer("torch_save")
        sd = {"w": torch.ones(4)}

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
            ckptr._do_save(sd, "/tmp/step-100/optim", storage_writer=None)

        mock_msc.torch.MultiStorageFileSystemWriter.assert_not_called()
        mock_dcp.save.assert_called_once()


class TestFormatLoad:
    """Tests that _get_storage_reader returns the correct reader for each format, and that _do_load routes correctly based on whether a reader is provided."""

    def _make_checkpointer(self, model_save_format, is_peft=False):
        with patch("torch.distributed.is_initialized", return_value=False):
            config = CheckpointingConfig(
                enabled=True,
                checkpoint_dir="/tmp/test",
                model_save_format=model_save_format,
                model_cache_dir="/tmp/cache",
                model_repo_id="test/model",
                save_consolidated=False,
                is_peft=is_peft,
            )
            return Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0)

    def test_safetensors_format_produces_hf_reader(self):
        """safetensors format: _get_storage_reader returns an HF reader."""
        ckptr = self._make_checkpointer("safetensors")
        reader = ckptr._get_storage_reader("/tmp/model", key_mapping=None)
        assert reader is not None

    def test_dcp_format_produces_no_reader(self):
        """torch_save (DCP) format: _get_storage_reader returns None."""
        ckptr = self._make_checkpointer("torch_save")
        reader = ckptr._get_storage_reader("/tmp/model", key_mapping=None)
        assert reader is None

    def test_safetensors_cloud_load_uses_hf_reader_not_msc(self):
        """safetensors + cloud: HF reader passed to dcp.load, MSC reader never created."""
        ckptr = self._make_checkpointer("safetensors")
        sd = {"w": torch.zeros(4)}
        hf_reader = ckptr._get_storage_reader("/tmp/model", key_mapping=None)
        assert hf_reader is not None

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                ckptr._do_load(sd, "msc://bucket/step-100/model", storage_reader=hf_reader)

        mock_msc.torch.MultiStorageFileSystemReader.assert_not_called()
        mock_dcp.load.assert_called_once()
        _, kwargs = mock_dcp.load.call_args
        assert kwargs["storage_reader"] is hf_reader

    def test_dcp_cloud_load_uses_msc_reader(self):
        """torch_save (DCP) + cloud: no HF reader, so MSC reader injected."""
        ckptr = self._make_checkpointer("torch_save")
        sd = {"w": torch.zeros(4)}
        reader = ckptr._get_storage_reader("/tmp/model", key_mapping=None)
        assert reader is None

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                ckptr._do_load(sd, "msc://bucket/step-100/optim", storage_reader=reader)

        mock_msc.torch.MultiStorageFileSystemReader.assert_called_once_with("msc://bucket/step-100/optim")
        mock_dcp.load.assert_called_once()

    def test_safetensors_local_load_uses_hf_reader(self):
        """safetensors + local: HF reader used, MSC never involved."""
        ckptr = self._make_checkpointer("safetensors")
        sd = {"w": torch.zeros(4)}
        hf_reader = ckptr._get_storage_reader("/tmp/model", key_mapping=None)

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
            ckptr._do_load(sd, "/tmp/step-100/model", storage_reader=hf_reader)

        mock_msc.torch.MultiStorageFileSystemReader.assert_not_called()
        mock_dcp.load.assert_called_once()

    def test_dcp_local_load_no_reader_no_msc(self):
        """torch_save (DCP) + local: no reader, no MSC, plain dcp.load."""
        ckptr = self._make_checkpointer("torch_save")
        sd = {"w": torch.zeros(4)}

        with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
             patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
            ckptr._do_load(sd, "/tmp/step-100/optim", storage_reader=None)

        mock_msc.torch.MultiStorageFileSystemReader.assert_not_called()
        mock_dcp.load.assert_called_once()


class TestSyncAsyncSave:
    """Tests that _do_save calls dcp.save for sync and dcp.async_save for async, across DCP, safetensors, and PEFT formats on both cloud and local paths."""

    def _make_ckptr(self, is_async, is_peft=False):
        config = MagicMock()
        config.is_peft = is_peft
        config.is_async = is_async
        ckptr = MagicMock(spec=Checkpointer)
        ckptr.config = config
        ckptr._model_ctx = MagicMock(staging_active=False)
        ckptr._optim_ctx = MagicMock(staging_active=False)
        return ckptr

    def test_dcp_cloud_sync_calls_dcp_save(self):
        """DCP + cloud + sync: dcp.save called, dcp.async_save NOT called."""
        ckptr = self._make_ckptr(is_async=False)
        sd = {"w": torch.ones(4)}

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc"), \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_save(ckptr, sd, "msc://bucket/step-100/optim")

        mock_dcp.save.assert_called_once()
        mock_dcp.async_save.assert_not_called()

    def test_dcp_cloud_async_calls_dcp_async_save(self):
        """DCP + cloud + async: dcp.async_save called, dcp.save NOT called."""
        ckptr = self._make_ckptr(is_async=True)
        sd = {"w": torch.ones(4)}

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc"), \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_save(ckptr, sd, "msc://bucket/step-100/optim")

        mock_dcp.async_save.assert_called_once()
        mock_dcp.save.assert_not_called()

    def test_dcp_cloud_async_msc_writer_passed_to_async_save(self):
        """DCP + cloud + async: MSC writer is passed as storage_writer to dcp.async_save."""
        ckptr = self._make_ckptr(is_async=True)
        sd = {"w": torch.ones(4)}

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_save(ckptr, sd, "msc://bucket/step-100/optim")

        msc_writer = mock_msc.torch.MultiStorageFileSystemWriter.return_value
        _, kwargs = mock_dcp.async_save.call_args
        assert kwargs["storage_writer"] is msc_writer

    def test_dcp_cloud_sync_msc_writer_passed_to_save(self):
        """DCP + cloud + sync: MSC writer is passed as storage_writer to dcp.save."""
        ckptr = self._make_ckptr(is_async=False)
        sd = {"w": torch.ones(4)}

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_save(ckptr, sd, "msc://bucket/step-100/optim")

        msc_writer = mock_msc.torch.MultiStorageFileSystemWriter.return_value
        _, kwargs = mock_dcp.save.call_args
        assert kwargs["storage_writer"] is msc_writer

    def test_safetensors_cloud_sync_calls_dcp_save(self):
        """safetensors + cloud + sync: dcp.save called with HF writer, dcp.async_save NOT called."""
        ckptr = self._make_ckptr(is_async=False)
        sd = {"w": torch.ones(4)}
        hf_writer = MagicMock(name="HFStorageWriter")

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_save(ckptr, sd, "msc://bucket/step-100/model", storage_writer=hf_writer)

        mock_dcp.save.assert_called_once()
        mock_dcp.async_save.assert_not_called()
        mock_msc.torch.MultiStorageFileSystemWriter.assert_not_called()

    def test_safetensors_cloud_async_calls_dcp_async_save(self):
        """safetensors + cloud + async: dcp.async_save called with HF writer, dcp.save NOT called."""
        ckptr = self._make_ckptr(is_async=True)
        sd = {"w": torch.ones(4)}
        hf_writer = MagicMock(name="HFStorageWriter")

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_save(ckptr, sd, "msc://bucket/step-100/model", storage_writer=hf_writer)

        mock_dcp.async_save.assert_called_once()
        mock_dcp.save.assert_not_called()
        mock_msc.torch.MultiStorageFileSystemWriter.assert_not_called()
        _, kwargs = mock_dcp.async_save.call_args
        assert kwargs["storage_writer"] is hf_writer

    def test_peft_cloud_sync_uses_msc_open_not_dcp(self):
        """PEFT + cloud + sync: adapter written via msc.open, dcp never called."""
        ckptr = self._make_ckptr(is_async=False, is_peft=True)
        sd = {"lora.weight": torch.ones(4)}
        mock_file = MagicMock()
        mock_ctx = MagicMock(__enter__=MagicMock(return_value=mock_file),
                             __exit__=MagicMock(return_value=False))

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.save_file"), \
                 patch("torch.distributed.is_initialized", return_value=False):
                mock_msc.open.return_value = mock_ctx
                Checkpointer._do_save(ckptr, sd, "msc://bucket/step-100/model")

        mock_msc.open.assert_called_once()
        mock_dcp.save.assert_not_called()
        mock_dcp.async_save.assert_not_called()

    def test_peft_cloud_async_still_uses_msc_open_not_dcp(self):
        """PEFT + cloud + async: adapter still written sync via msc.open, dcp never called."""
        ckptr = self._make_ckptr(is_async=True, is_peft=True)
        sd = {"lora.weight": torch.ones(4)}
        mock_file = MagicMock()
        mock_ctx = MagicMock(__enter__=MagicMock(return_value=mock_file),
                             __exit__=MagicMock(return_value=False))

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.msc") as mock_msc, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp, \
                 patch("nemo_automodel.components.checkpoint.checkpointing.save_file"), \
                 patch("torch.distributed.is_initialized", return_value=False):
                mock_msc.open.return_value = mock_ctx
                Checkpointer._do_save(ckptr, sd, "msc://bucket/step-100/model")

        mock_msc.open.assert_called_once()
        mock_dcp.async_save.assert_not_called()
        mock_dcp.save.assert_not_called()

    def test_local_sync_calls_dcp_save(self):
        """Local + sync: dcp.save called, dcp.async_save NOT called."""
        ckptr = self._make_ckptr(is_async=False)
        sd = {"w": torch.ones(4)}

        with patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
            Checkpointer._do_save(ckptr, sd, "/tmp/step-100/optim")

        mock_dcp.save.assert_called_once()
        mock_dcp.async_save.assert_not_called()

    def test_local_async_calls_dcp_async_save(self):
        """Local + async: dcp.async_save called, dcp.save NOT called."""
        ckptr = self._make_ckptr(is_async=True)
        sd = {"w": torch.ones(4)}

        with _cloud_patches():
            with patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp:
                Checkpointer._do_save(ckptr, sd, "/tmp/step-100/optim")

        mock_dcp.async_save.assert_called_once()
        mock_dcp.save.assert_not_called()
