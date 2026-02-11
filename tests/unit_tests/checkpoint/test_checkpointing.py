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

from unittest.mock import MagicMock, patch

import torch

from nemo_automodel.components.checkpoint.checkpointing import _equally_divide_layers, _reinit_rope_buffers
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
        config.model_type = "deci"
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
        config.model_type = "deci"
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
        config.model_type = "deci"
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
        config.model_type = "deci"
        model.config = config

        rope = MagicMock()
        rope.rope_init_fn = MagicMock(side_effect=RuntimeError("bad init"))
        rope.inv_freq = torch.zeros(3)
        rope.rope_kwargs = {}
        rope.config = config

        with patch.object(model, "named_modules", return_value=[("", model), ("layers.0.rotary", rope)]):
            # Should not raise, just log a warning
            _reinit_rope_buffers(model, torch.device("cpu"))
