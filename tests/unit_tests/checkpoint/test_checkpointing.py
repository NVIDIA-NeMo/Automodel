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
import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from nemo_automodel.components.checkpoint._backports.hf_storage import (
    _DIFFUSERS_INDEX_FN,
    _extract_file_index_with_status,
    get_fqn_to_dtype_mapping,
    get_fqn_to_file_index_mapping,
)
from nemo_automodel.components.checkpoint._backports.hf_utils import (
    FQN_TO_DTYPE_MAPPING_FILENAME,
    FQN_TO_FILE_INDEX_MAPPING_FILENAME,
)
from nemo_automodel.components.checkpoint.addons import ConsolidatedHFAddon
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    SaveConsolidatedMode,
    _divide_keys_by_size,
    _equally_divide_layers,
    _is_custom_model,
    _model_has_dtensors,
    _normalize_dtype_mapping_to_state_dict_keys,
    _reinit_non_persistent_buffers,
    _should_write_consolidated_safetensors,
    _summarize_state_dict_key_diff,
    _warn_if_large_inline_consolidation,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, _get_lm_head_weight_and_name
from nemo_automodel.components.checkpoint.utils import (
    has_local_tied_lm_head,
    materialize_missing_tied_lm_head,
)


def _make_keys(count: int) -> list[str]:
    return [f"layer.{i}" for i in range(count)]


def _count_by_shard(mapping: dict[str, int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for shard_index in mapping.values():
        counts[shard_index] = counts.get(shard_index, 0) + 1
    return counts


def test_extract_file_index_with_status_supports_hf_and_qwen35_patterns():
    assert _extract_file_index_with_status("model-00001-of-00008.safetensors") == (1, True)
    assert _extract_file_index_with_status("model.safetensors-00002-of-00008.safetensors") == (2, True)
    assert _extract_file_index_with_status("shard-00000-model-00003-of-00008.safetensors") == (3, True)
    assert _extract_file_index_with_status("model.safetensors") == (1, True)


def test_extract_file_index_with_status_rejects_invalid_encoded_index():
    assert _extract_file_index_with_status("model-00012-of-00008.safetensors") == (1, False)
    assert _extract_file_index_with_status("model-00000-of-00008.safetensors") == (1, False)
    assert _extract_file_index_with_status("weights-a.safetensors") == (1, False)


def test_get_fqn_to_file_index_mapping_uses_index_json_for_qwen35_names(tmp_path):
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "model.layers.0.weight": "model.safetensors-00001-of-00003.safetensors",
            "model.layers.1.weight": "model.safetensors-00002-of-00003.safetensors",
            "lm_head.weight": "model.safetensors-00003-of-00003.safetensors",
        },
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    mapping = get_fqn_to_file_index_mapping(str(tmp_path))

    assert mapping == {
        "model.layers.0.weight": 1,
        "model.layers.1.weight": 2,
        "lm_head.weight": 3,
    }


def test_get_fqn_to_dtype_mapping_reads_safetensors_headers_and_applies_key_mapping(tmp_path):
    save_file(
        {
            "orig.layers.0.weight": torch.ones(2, dtype=torch.bfloat16),
            "orig.layers.1.weight": torch.ones(2, dtype=torch.float32),
        },
        tmp_path / "model-00001-of-00001.safetensors",
    )
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "orig.layers.0.weight": "model-00001-of-00001.safetensors",
            "orig.layers.1.weight": "model-00001-of-00001.safetensors",
        },
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    mapping = get_fqn_to_dtype_mapping(str(tmp_path), {r"^orig\.": "model."})

    assert mapping == {
        "model.layers.0.weight": "BF16",
        "model.layers.1.weight": "F32",
    }


def test_get_fqn_to_file_index_mapping_warns_when_multiple_files_fallback_to_one(tmp_path, caplog):
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "model.layers.0.weight": "weights-a.safetensors",
            "model.layers.1.weight": "weights-b.safetensors",
        },
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    caplog.set_level(logging.WARNING)
    mapping = get_fqn_to_file_index_mapping(str(tmp_path))

    assert mapping == {
        "model.layers.0.weight": 1,
        "model.layers.1.weight": 2,
    }
    assert "parsing failed or produced unexpected indices" in caplog.text


def test_get_fqn_to_file_index_mapping_warns_when_only_some_files_parse(tmp_path, caplog):
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "weights-b.safetensors",
        },
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    caplog.set_level(logging.WARNING)
    mapping = get_fqn_to_file_index_mapping(str(tmp_path))

    assert mapping == {
        "model.layers.0.weight": 1,
        "model.layers.1.weight": 2,
    }
    assert "weights-b.safetensors" in caplog.text


def test_get_fqn_to_file_index_mapping_reserves_single_file_index_for_fallback_assignment(tmp_path, caplog):
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "model.layers.0.weight": "model.safetensors",
            "model.layers.1.weight": "weights-b.safetensors",
        },
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    caplog.set_level(logging.WARNING)
    mapping = get_fqn_to_file_index_mapping(str(tmp_path))

    assert mapping == {
        "model.layers.0.weight": 1,
        "model.layers.1.weight": 2,
    }
    assert "parsing failed or produced unexpected indices" in caplog.text


def test_get_fqn_to_file_index_mapping_reassigns_invalid_encoded_index(tmp_path, caplog):
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00008.safetensors",
            "model.layers.1.weight": "model-00012-of-00008.safetensors",
        },
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    caplog.set_level(logging.WARNING)
    mapping = get_fqn_to_file_index_mapping(str(tmp_path))

    assert mapping == {
        "model.layers.0.weight": 1,
        "model.layers.1.weight": 2,
    }
    assert "parsing failed or produced unexpected indices" in caplog.text
    assert "model-00012-of-00008.safetensors" in caplog.text


def test_get_fqn_to_file_index_mapping_remaps_when_parsed_indices_are_not_dense(tmp_path, caplog):
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00012.safetensors",
            "model.layers.1.weight": "model-00012-of-00012.safetensors",
        },
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    caplog.set_level(logging.WARNING)
    mapping = get_fqn_to_file_index_mapping(str(tmp_path))

    assert mapping == {
        "model.layers.0.weight": 1,
        "model.layers.1.weight": 2,
    }
    assert "Expected indices to be 1..2; falling back to sorted filename order for output indices" in caplog.text


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


def test_divide_keys_by_size_keeps_small_state_dict_in_one_shard():
    state_dict = {
        "a.weight": torch.empty(2, dtype=torch.float32),
        "b.weight": torch.empty(3, dtype=torch.float32),
    }

    mapping = _divide_keys_by_size(list(state_dict), state_dict, target_shard_bytes=32)

    assert mapping == {
        "a.weight": 1,
        "b.weight": 1,
    }


def test_divide_keys_by_size_splits_without_empty_shards():
    state_dict = {
        "a.weight": torch.empty(6, dtype=torch.uint8),
        "b.weight": torch.empty(6, dtype=torch.uint8),
        "c.weight": torch.empty(1, dtype=torch.uint8),
    }

    mapping = _divide_keys_by_size(list(state_dict), state_dict, target_shard_bytes=10)

    assert mapping == {
        "a.weight": 1,
        "b.weight": 2,
        "c.weight": 2,
    }


def test_divide_keys_by_size_places_oversized_tensor_alone():
    state_dict = {
        "huge.weight": torch.empty(12, dtype=torch.uint8),
        "small.weight": torch.empty(1, dtype=torch.uint8),
    }

    mapping = _divide_keys_by_size(list(state_dict), state_dict, target_shard_bytes=10)

    assert mapping == {
        "huge.weight": 1,
        "small.weight": 2,
    }


def test_missing_original_hf_index_uses_size_based_consolidated_mapping(tmp_path, caplog):
    class FakeTensor:
        def __init__(self, bytes_: int):
            self.bytes = bytes_

        def numel(self):
            return self.bytes

        def element_size(self):
            return 1

    config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(tmp_path),
        model_save_format="safetensors",
        model_cache_dir=str(tmp_path / "cache"),
        model_repo_id="config-only/model",
        save_consolidated=False,
        is_peft=False,
    )
    with patch("torch.distributed.is_initialized", return_value=False):
        checkpointer = Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)

    state_dict = {
        "a.weight": FakeTensor(4 * 1024**3),
        "b.weight": FakeTensor(4 * 1024**3),
        "c.weight": FakeTensor(1),
    }
    model_state = SimpleNamespace(model=[SimpleNamespace()])
    caplog.set_level(logging.INFO)

    with patch(
        "nemo_automodel.components.checkpoint.checkpointing._get_hf_safetensors_reference_path", return_value=None
    ):
        mapping = checkpointer._maybe_build_consolidated_index(model_state, state_dict)
        dtype_mapping = checkpointer._maybe_build_original_dtype_mapping(model_state, state_dict)

    assert mapping == {
        "a.weight": 1,
        "b.weight": 2,
        "c.weight": 2,
    }
    assert dtype_mapping is None
    assert "No original HF safetensors reference path found for config-only/model" in caplog.text
    assert "2 output shard(s)" in caplog.text


def test_normalize_dtype_mapping_to_state_dict_keys_uses_hf_base_model_prefix():
    dtype_mapping = {
        "h.0.ln_1.weight": "BF16",
        "wte.weight": "BF16",
        "lm_head.weight": "F32",
        "unused.weight": "BF16",
    }
    state_dict_keys = [
        "transformer.h.0.ln_1.weight",
        "transformer.wte.weight",
        "lm_head.weight",
    ]

    normalized = _normalize_dtype_mapping_to_state_dict_keys(dtype_mapping, state_dict_keys, "transformer")

    assert normalized == {
        "transformer.h.0.ln_1.weight": "BF16",
        "transformer.wte.weight": "BF16",
        "lm_head.weight": "F32",
    }


def test_original_dtype_mapping_is_keyed_by_export_state_dict(tmp_path):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    save_file(
        {
            "h.0.ln_1.weight": torch.ones(1, dtype=torch.bfloat16),
            "wte.weight": torch.ones(1, dtype=torch.bfloat16),
            "unused.weight": torch.ones(1, dtype=torch.float32),
        },
        reference_dir / "model.safetensors",
    )
    config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(tmp_path),
        model_save_format="safetensors",
        model_cache_dir=str(tmp_path / "cache"),
        model_repo_id="test/model",
        save_consolidated=False,
        is_peft=False,
    )
    with patch("torch.distributed.is_initialized", return_value=False):
        checkpointer = Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)
    model_state = SimpleNamespace(model=[SimpleNamespace(base_model_prefix="transformer")])
    state_dict = {
        "transformer.h.0.ln_1.weight": torch.ones(1),
        "transformer.wte.weight": torch.ones(1),
    }

    with patch(
        "nemo_automodel.components.checkpoint.checkpointing._get_hf_safetensors_reference_path",
        return_value=str(reference_dir),
    ):
        dtype_mapping = checkpointer._maybe_build_original_dtype_mapping(model_state, state_dict)

    assert dtype_mapping == {
        "transformer.h.0.ln_1.weight": "BF16",
        "transformer.wte.weight": "BF16",
    }


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


class _LocalUntiedButConfiguredModel(torch.nn.Module):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(4, 4)
        self.lm_head = torch.nn.Linear(4, 4, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens


def test_has_local_tied_lm_head_is_false_when_config_tied_but_storage_untied():
    model = _LocalUntiedButConfiguredModel()

    assert has_local_tied_lm_head(model) is False


def test_model_state_keeps_lm_head_when_config_tied_but_storage_untied():
    model = _LocalUntiedButConfiguredModel()

    model_state = ModelState(model, is_peft=False, is_init_step=False)
    saved_state_dict = model_state.state_dict()

    assert "lm_head.weight" in saved_state_dict
    assert "model.embed_tokens.weight" in saved_state_dict


def test_model_state_drops_lm_head_when_storage_is_actually_tied():
    model = _LocalUntiedButConfiguredModel()
    model.lm_head.weight = model.model.embed_tokens.weight

    model_state = ModelState(model, is_peft=False, is_init_step=False)
    saved_state_dict = model_state.state_dict()

    assert has_local_tied_lm_head(model) is True
    assert "lm_head.weight" not in saved_state_dict
    assert "model.embed_tokens.weight" in saved_state_dict


def test_model_state_refreshes_tied_lm_head_before_dropping_key():
    model = _LocalUntiedButConfiguredModel()
    model_state = ModelState(model, is_peft=False, is_init_step=False)
    assert model_state.has_local_tied_lm_head is False

    def fake_get_model_state_dict(model_part, options=None):
        model_part.lm_head.weight = model_part.model.embed_tokens.weight
        return {
            "lm_head.weight": model_part.lm_head.weight,
            "model.embed_tokens.weight": model_part.model.embed_tokens.weight,
        }

    with patch(
        "nemo_automodel.components.checkpoint.stateful_wrappers.get_model_state_dict",
        side_effect=fake_get_model_state_dict,
    ):
        saved_state_dict = model_state.state_dict()

    assert model_state.has_local_tied_lm_head is True
    assert "lm_head.weight" not in saved_state_dict
    assert "model.embed_tokens.weight" in saved_state_dict


def test_materialize_missing_tied_lm_head_uses_embedding_tensor_from_checkpoint():
    model = _PipelineLastStageLikeModel()
    embed_weight = torch.full_like(model.lm_head.weight, 3.0)
    state_dict = {"model.language_model.embed_tokens.weight": embed_weight}

    materialized = materialize_missing_tied_lm_head(state_dict, model, allow_current_lm_head_fallback=False)

    assert materialized is True
    assert "lm_head.weight" in state_dict
    assert torch.equal(state_dict["lm_head.weight"], embed_weight)
    assert not torch.equal(state_dict["lm_head.weight"], model.lm_head.weight.detach())


def test_model_state_retie_lm_head_after_load_state_dict():
    model = _LocalUntiedButConfiguredModel()
    model.lm_head.weight = model.model.embed_tokens.weight
    checkpoint_weight = torch.full_like(model.model.embed_tokens.weight, 5.0)
    state_dict = {"model.embed_tokens.weight": checkpoint_weight}
    model_state = ModelState(model, is_peft=False, is_init_step=False)

    def fake_set_model_state_dict(model_part, state_dict, options):
        model_part.model.embed_tokens.weight = torch.nn.Parameter(torch.empty_like(checkpoint_weight))
        model_part.lm_head.weight = torch.nn.Parameter(torch.empty_like(checkpoint_weight))
        model_part.model.embed_tokens.weight.data.copy_(state_dict["model.embed_tokens.weight"])
        model_part.lm_head.weight.data.copy_(state_dict["lm_head.weight"])
        assert model_part.lm_head.weight.data_ptr() != model_part.model.embed_tokens.weight.data_ptr()

    with patch(
        "nemo_automodel.components.checkpoint.stateful_wrappers.set_model_state_dict",
        side_effect=fake_set_model_state_dict,
    ):
        model_state.load_state_dict(state_dict, strict=False)

    assert has_local_tied_lm_head(model) is True
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert torch.equal(model.model.embed_tokens.weight, checkpoint_weight)


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
    """Verify custom-model load routing: sharded uses DCP, single-device uses the fast path.

    Under multi-rank (sharded) loading, custom models use the standard DCP path so each
    rank slices its local DTensor shard. On a single device (world_size == 1) there is no
    sharding, so a custom safetensors model takes the frugal full-state fast path instead
    (which still applies the state_dict_adapter from_hf conversion on CPU). See
    NOTE [nemotron-singlegpu-lora] in checkpointing.py.
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
        """Under sharded (multi-rank) loading, a custom model uses the standard DCP path.

        DCP lets each rank slice its local DTensor shard. The single-device exception is
        covered by test_single_device_custom_model_uses_fast_path.
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
            # Simulate multi-rank (sharded) load so the single-device fast path is not taken.
            patch("torch.distributed.is_initialized", return_value=False),
            patch.dict("os.environ", {"WORLD_SIZE": "2"}),
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

    @patch("nemo_automodel.components.checkpoint.checkpointing._is_safetensors_checkpoint", return_value=True)
    @patch("nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype")
    @patch("nemo_automodel.components.checkpoint.checkpointing._load_full_state_dict_into_model")
    def test_single_device_custom_model_uses_fast_path(self, mock_load_full, mock_load_hf, mock_is_st):
        """On a single device (world_size == 1) a custom safetensors model uses the fast path.

        The fast path applies the state_dict_adapter from_hf conversion on CPU (via
        _maybe_adapt_state_dict_from_hf) and copies into the model, keeping device memory at
        ~model size. DCP would transiently materialize a second on-device copy of the merged
        expert weights and OOM a 30B-class MoE on one 80GB GPU.
        See NOTE [nemotron-singlegpu-lora] in checkpointing.py.
        """
        checkpointer = self._make_checkpointer()

        CustomModel = type("CustomModel", (torch.nn.Module,), {})
        CustomModel.__module__ = "nemo_automodel.components.models.nemotron_v3.model"
        model = CustomModel()
        model.layer = torch.nn.Linear(4, 4)
        assert _is_custom_model(model) is True

        mock_load_hf.return_value = {"layer.weight": torch.randn(4, 4), "layer.bias": torch.randn(4)}

        with (
            patch("os.path.exists", return_value=True),
            # Single-device (non-sharded) load.
            patch("torch.distributed.is_initialized", return_value=False),
            patch.dict("os.environ", {"WORLD_SIZE": "1"}),
            patch.object(checkpointer, "_do_load") as mock_dcp_load,
        ):
            checkpointer.load_model(model, model_path="/fake/path", is_init_step=True)

        # Single-device custom model takes the frugal fast path, not DCP.
        mock_load_full.assert_called_once()
        mock_dcp_load.assert_not_called()


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

    def test_retie_weights_after_meta_initialization(self):
        """Tied embeddings should be re-applied after materializing and initializing meta params."""

        class FakeTiedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                with torch.device("meta"):
                    self.model = torch.nn.Module()
                    self.model.embed_tokens = torch.nn.Embedding(4, 4)
                    self.lm_head = torch.nn.Linear(4, 4, bias=False)
                self.config = SimpleNamespace(architectures=["FakeTiedModel"], tie_word_embeddings=True)
                self.tie_weights_called = False

            def get_input_embeddings(self):
                return self.model.embed_tokens

            def tie_weights(self):
                self.lm_head.weight = self.model.embed_tokens.weight
                self.tie_weights_called = True

            def initialize_weights(self):
                with torch.no_grad():
                    self.model.embed_tokens.weight.fill_(1.0)
                    self.lm_head.weight.fill_(2.0)

        model = FakeTiedModel()

        Checkpointer.initialize_model_weights(model, torch.device("cpu"))

        assert model.tie_weights_called is True
        assert model.lm_head.weight is model.model.embed_tokens.weight
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        assert torch.all(model.lm_head.weight == 1.0)

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

        from nemo_automodel.components.checkpoint.utils import is_tied_word_embeddings

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

        from nemo_automodel.components.checkpoint.utils import is_tied_word_embeddings

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


class TestOfflineConsolidationScriptAndWarnings:
    """Focused tests for offline consolidation helper generation and warnings."""

    def _make_checkpointer(self, tmp_path, save_consolidated=False, diffusers_compatible=False):
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path),
            model_save_format="safetensors",
            model_cache_dir=str(tmp_path / "cache"),
            model_repo_id="test/model",
            save_consolidated=save_consolidated,
            diffusers_compatible=diffusers_compatible,
            is_peft=False,
        )
        with patch("torch.distributed.is_initialized", return_value=False):
            return Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)

    def test_writes_conservative_consolidate_script(self, tmp_path, caplog):
        checkpointer = self._make_checkpointer(tmp_path, save_consolidated=False)
        model_dir = tmp_path / "epoch_0_step_1" / "model"
        model_dir.mkdir(parents=True)
        caplog.set_level(logging.DEBUG)

        checkpointer._maybe_write_offline_consolidation_script(str(model_dir))

        script_path = model_dir / "consolidate.sh"
        script = script_path.read_text()
        assert script_path.exists()
        assert os.access(script_path, os.X_OK)
        assert 'NPROC_PER_NODE="${NPROC_PER_NODE:-1}"' in script
        assert 'NUM_THREADS="${NUM_THREADS:-5}"' in script
        assert 'CAST_DTYPE="${CAST_DTYPE:-}"' in script
        assert 'PYTHON="${PYTHON:-python3}"' in script
        assert 'CONSOLIDATION_TOOL="${CONSOLIDATION_TOOL:-tools/offline_hf_consolidation.py}"' in script
        assert "PYTHON_MODULE" not in script
        assert 'NPROC_PER_NODE=16 NUM_THREADS=5 bash "$0"' in script
        assert "NPROC_PER_NODE * NUM_THREADS within your CPU allocation" in script
        assert "sbatch --cpus-per-task=80" in script
        assert "CAST_DTYPE=bf16" in script
        assert 'CAST_DTYPE_ARGS=(--cast-dtype "${CAST_DTYPE}")' in script
        assert '"${TORCHRUN}" --nproc-per-node="${NPROC_PER_NODE}" "${CONSOLIDATION_TOOL}" \\' in script
        assert '"${PYTHON}" "${CONSOLIDATION_TOOL}" \\' in script
        assert "Run from the AutoModel repo root or set CONSOLIDATION_TOOL=" in script
        assert "--backend gloo \\" in script
        assert '--model-name "test/model" \\' in script
        assert f'--input-dir "{model_dir}" \\' in script
        assert f'--output-dir "{model_dir / "consolidated"}"' in script
        assert "--diffusers-compatible" not in script
        assert f"Wrote offline HF safetensors consolidation helper script to {script_path}." in caplog.text

    def test_writes_diffusers_compatible_consolidate_script(self, tmp_path):
        checkpointer = self._make_checkpointer(tmp_path, save_consolidated=False, diffusers_compatible=True)
        model_dir = tmp_path / "epoch_0_step_1" / "model"
        model_dir.mkdir(parents=True)

        checkpointer._maybe_write_offline_consolidation_script(str(model_dir))

        script = (model_dir / "consolidate.sh").read_text()
        assert f'--output-dir "{model_dir / "consolidated"}" \\' in script
        assert "--diffusers-compatible" in script

    def test_writes_script_when_inline_consolidation_is_enabled(self, tmp_path):
        checkpointer = self._make_checkpointer(tmp_path, save_consolidated=True)
        model_dir = tmp_path / "epoch_0_step_1" / "model"
        model_dir.mkdir(parents=True)

        checkpointer._maybe_write_offline_consolidation_script(str(model_dir))

        assert (model_dir / "consolidate.sh").exists()

    def test_final_consolidation_mode_writes_script_for_non_final_checkpoints(self, tmp_path):
        checkpointer = self._make_checkpointer(tmp_path, save_consolidated="final")
        model_dir = tmp_path / "epoch_0_step_1" / "model"
        model_dir.mkdir(parents=True)

        checkpointer._maybe_write_offline_consolidation_script(str(model_dir))

        assert (model_dir / "consolidate.sh").exists()

    def test_final_consolidation_mode_writes_script_for_final_checkpoint(self, tmp_path):
        checkpointer = self._make_checkpointer(tmp_path, save_consolidated="final")
        model_dir = tmp_path / "epoch_0_step_9" / "model"
        model_dir.mkdir(parents=True)

        checkpointer._maybe_write_offline_consolidation_script(str(model_dir))

        assert (model_dir / "consolidate.sh").exists()

    def test_final_checkpoint_logs_helper_hint_for_sharded_only_export(self, tmp_path, caplog):
        checkpointer = self._make_checkpointer(tmp_path, save_consolidated=False)
        model_dir = tmp_path / "epoch_0_step_9" / "model"
        model_dir.mkdir(parents=True)
        caplog.set_level(logging.INFO)

        checkpointer._maybe_write_offline_consolidation_script(str(model_dir))
        checkpointer._maybe_log_final_offline_consolidation_hint(str(model_dir), is_final_checkpoint=True)

        assert "Final checkpoint was saved with checkpoint.save_consolidated=false" in caplog.text
        assert f"run bash {model_dir / 'consolidate.sh'}" in caplog.text

    def test_inline_consolidation_preserves_hf_metadata_for_offline_helper(self, tmp_path):
        hf_metadata_dir = tmp_path / "model" / ".hf_metadata"
        consolidated_dir = tmp_path / "model" / "consolidated"
        tokenizer_dir = hf_metadata_dir / "tokenizer"
        hf_metadata_dir.mkdir(parents=True)
        consolidated_dir.mkdir(parents=True)
        tokenizer_dir.mkdir()
        (hf_metadata_dir / "config.json").write_text("{}")
        (hf_metadata_dir / FQN_TO_FILE_INDEX_MAPPING_FILENAME).write_text('{"w": 1}')
        (hf_metadata_dir / FQN_TO_DTYPE_MAPPING_FILENAME).write_text('{"w": "BF16"}')
        (tokenizer_dir / "tokenizer.json").write_text("{}")

        with patch("torch.distributed.is_initialized", return_value=False):
            ConsolidatedHFAddon().post_save(
                consolidated_path=str(consolidated_dir),
                hf_metadata_path=str(hf_metadata_dir),
            )

        assert (hf_metadata_dir / "config.json").exists()
        assert (hf_metadata_dir / FQN_TO_FILE_INDEX_MAPPING_FILENAME).exists()
        assert (hf_metadata_dir / FQN_TO_DTYPE_MAPPING_FILENAME).exists()
        assert (tokenizer_dir / "tokenizer.json").exists()
        assert (consolidated_dir / "config.json").exists()
        assert (consolidated_dir / "tokenizer" / "tokenizer.json").exists()
        assert not (consolidated_dir / FQN_TO_FILE_INDEX_MAPPING_FILENAME).exists()
        assert not (consolidated_dir / FQN_TO_DTYPE_MAPPING_FILENAME).exists()

    def test_save_consolidated_normalizes_legacy_bools(self, tmp_path):
        assert self._make_checkpointer(tmp_path, save_consolidated=True).config.save_consolidated is (
            SaveConsolidatedMode.EVERY
        )
        assert self._make_checkpointer(tmp_path, save_consolidated=False).config.save_consolidated is (
            SaveConsolidatedMode.FALSE
        )

    def test_final_consolidation_only_exports_on_final_checkpoint(self, tmp_path):
        checkpointer = self._make_checkpointer(tmp_path, save_consolidated="final")

        assert _should_write_consolidated_safetensors(checkpointer.config, is_final_checkpoint=False) is False
        assert _should_write_consolidated_safetensors(checkpointer.config, is_final_checkpoint=True) is True

    def test_setup_warns_for_inline_consolidation(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("WORLD_SIZE", "1")
        caplog.set_level(logging.WARNING)

        self._make_checkpointer(tmp_path, save_consolidated=True)

        assert "checkpoint.save_consolidated=every exports HuggingFace safetensors during every checkpoint save" in (
            caplog.text
        )
        assert "world_size" not in caplog.text
        assert "can leave GPUs idle during consolidation and filesystem writes" in caplog.text
        assert "Recommended: checkpoint.save_consolidated=final" in caplog.text
        assert "bash <checkpoint>/model/consolidate.sh" in caplog.text

    def test_save_time_warns_for_large_inline_consolidation(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("WORLD_SIZE", "256")
        checkpointer = self._make_checkpointer(tmp_path, save_consolidated=True)
        caplog.clear()
        caplog.set_level(logging.WARNING)

        class FakeLargeTensor:
            def numel(self):
                return 50 * 1024**3 // 2

            def element_size(self):
                return 2

        _warn_if_large_inline_consolidation(checkpointer.config, {"w": FakeLargeTensor()}, {"w": 1})

        assert "may be exporting a large HF checkpoint" in caplog.text
        assert "this rank's local estimate is ~50.0 GiB" in caplog.text
        assert "full size may differ under distributed parallelism" in caplog.text
        assert "1 output file, world_size=256" in caplog.text
        assert "~50.0 GiB" in caplog.text
        assert "save_consolidated=final" in caplog.text
        assert "bash <checkpoint>/model/consolidate.sh" in caplog.text

    def test_save_time_uses_hf_index_size_before_distributed_fallback(self, tmp_path, caplog):
        cache_dir = tmp_path / "cache"
        model_dir = cache_dir / "models--test--model" / "snapshots" / "abc123"
        model_dir.mkdir(parents=True)
        with open(model_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": 64 * 1024**3}, "weight_map": {}}, f)

        checkpointer = self._make_checkpointer(tmp_path, save_consolidated=True)
        checkpointer.config.model_cache_dir = str(cache_dir)
        checkpointer.config.model_repo_id = "test/model"
        caplog.clear()
        caplog.set_level(logging.WARNING)

        class FakeSmallLocalTensor:
            def numel(self):
                return 1024

            def element_size(self):
                return 2

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=64),
            patch("torch.distributed.all_reduce") as mock_all_reduce,
        ):
            _warn_if_large_inline_consolidation(checkpointer.config, {"w": FakeSmallLocalTensor()}, {"w": 1})

        mock_all_reduce.assert_not_called()
        assert "checkpoint.save_consolidated=every is exporting ~64.0 GiB of HF safetensors" in caplog.text
        assert "size from HF index" in caplog.text
        assert "1 output file, world_size=64" in caplog.text
        assert "~64.0 GiB" in caplog.text


class TestOfflineHFConsolidationTool:
    """Focused tests for the root offline consolidation tool."""

    def test_main_renames_index_when_diffusers_compatible(self, tmp_path, monkeypatch, caplog):
        from tools import offline_hf_consolidation as tool

        input_dir = tmp_path / "model"
        metadata_dir = input_dir / ".hf_metadata"
        output_dir = input_dir / "consolidated"
        metadata_dir.mkdir(parents=True)
        with open(metadata_dir / FQN_TO_FILE_INDEX_MAPPING_FILENAME, "w") as f:
            json.dump({"w": 1}, f)
        with open(metadata_dir / FQN_TO_DTYPE_MAPPING_FILENAME, "w") as f:
            json.dump({"w": "BF16"}, f)
        metadata_file = metadata_dir / "config.json"
        metadata_file.write_text("{}")

        monkeypatch.setattr(
            "sys.argv",
            [
                "offline_hf_consolidation",
                "--backend",
                "gloo",
                "--model-name",
                "test/model",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--diffusers-compatible",
            ],
        )
        caplog.set_level(logging.INFO)

        with (
            patch.object(tool, "initialize_distributed"),
            patch.object(tool, "get_world_size_safe", return_value=1),
            patch.object(tool, "get_rank_safe", return_value=0),
            patch.object(tool, "consolidate_safetensors_files_on_every_rank") as mock_consolidate,
            patch.object(tool, "_maybe_rename_index_for_diffusers") as mock_rename,
        ):
            tool.main()

        mock_consolidate.assert_called_once_with(
            str(input_dir),
            str(output_dir),
            {"w": 1},
            num_threads=5,
            cast_dtype=None,
            fqn_to_dtype_mapping={"w": "BF16"},
        )
        mock_rename.assert_called_once_with(str(output_dir))
        assert metadata_dir.exists()
        assert metadata_file.exists()
        assert (output_dir / "config.json").exists()
        assert not (output_dir / FQN_TO_DTYPE_MAPPING_FILENAME).exists()
        assert f"Consolidating sharded HF safetensors from {input_dir} to {output_dir}." not in caplog.text
        assert f"Successfully exported consolidated HF safetensors to {output_dir}." in caplog.text

    def test_main_skips_when_output_exists_and_metadata_was_consumed(self, tmp_path, monkeypatch, caplog):
        from tools import offline_hf_consolidation as tool

        input_dir = tmp_path / "model"
        output_dir = input_dir / "consolidated"
        input_dir.mkdir(parents=True)
        output_dir.mkdir()
        (output_dir / "model-00001-of-00001.safetensors").write_bytes(b"")

        monkeypatch.setattr(
            "sys.argv",
            [
                "offline_hf_consolidation",
                "--backend",
                "gloo",
                "--model-name",
                "test/model",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )
        caplog.set_level(logging.INFO)

        with (
            patch.object(tool, "initialize_distributed"),
            patch.object(tool, "get_world_size_safe", return_value=1),
            patch.object(tool, "get_rank_safe", return_value=0),
            patch.object(tool, "consolidate_safetensors_files_on_every_rank") as mock_consolidate,
        ):
            tool.main()

        mock_consolidate.assert_not_called()
        assert f"Consolidated HF safetensors already exist at {output_dir}" in caplog.text

    def test_main_passes_cast_dtype_and_updates_config(self, tmp_path, monkeypatch, caplog):
        from tools import offline_hf_consolidation as tool

        input_dir = tmp_path / "model"
        metadata_dir = input_dir / ".hf_metadata"
        output_dir = input_dir / "consolidated"
        metadata_dir.mkdir(parents=True)
        with open(metadata_dir / FQN_TO_FILE_INDEX_MAPPING_FILENAME, "w") as f:
            json.dump({"w": 1}, f)
        with open(metadata_dir / "config.json", "w") as f:
            json.dump({"torch_dtype": "float32"}, f)

        monkeypatch.setattr(
            "sys.argv",
            [
                "offline_hf_consolidation",
                "--backend",
                "gloo",
                "--model-name",
                "test/model",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--cast-dtype",
                "bf16",
            ],
        )
        caplog.set_level(logging.INFO)

        with (
            patch.object(tool, "initialize_distributed"),
            patch.object(tool, "get_world_size_safe", return_value=1),
            patch.object(tool, "get_rank_safe", return_value=0),
            patch.object(tool, "consolidate_safetensors_files_on_every_rank") as mock_consolidate,
        ):
            tool.main()

        mock_consolidate.assert_called_once_with(
            str(input_dir),
            str(output_dir),
            {"w": 1},
            num_threads=5,
            cast_dtype=torch.bfloat16,
            fqn_to_dtype_mapping=None,
        )
        with open(output_dir / "config.json", "r") as f:
            assert json.load(f)["torch_dtype"] == "bfloat16"
        assert "Casting floating-point tensors" not in caplog.text


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
        # empty_cache is created but contains no HF snapshot directory, so
        # _get_hf_safetensors_reference_path returns None.
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
        every rank produces the identical mapping covering every FQN.
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
        assert set(first.values()) == {1}
        assert "backbone.norm_f.weight" in first  # every rank sees rank-7's norm_f
        for r, m in enumerate(per_rank_mappings[1:], start=1):
            assert sorted(m.keys()) == global_pre_shard_keys, f"rank {r} mapping diverges"
            assert set(m.values()) == {1}
