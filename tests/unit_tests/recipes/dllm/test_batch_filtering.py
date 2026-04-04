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

"""Tests for dLLM batch filtering logic.

Models like LLaDA have a fixed forward() signature (no **kwargs), so
extra keys in the batch dict (input_lengths, etc.) cause TypeError.
The recipe uses ``filter_forward_kwargs`` before calling model(**batch).
"""

from unittest.mock import MagicMock

import torch

from nemo_automodel.components.utils.model_utils import filter_forward_kwargs
from nemo_automodel.recipes.dllm.strategy import MDLMStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_with_kwargs():
    """Model whose forward() accepts **kwargs -- no filtering needed."""
    model = MagicMock()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        pass

    model.forward = forward.__get__(model, type(model))
    return model


def _make_model_without_kwargs():
    """Model whose forward() has a fixed param list -- filtering required."""
    model = MagicMock()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        pass

    model.forward = forward.__get__(model, type(model))
    return model


# ---------------------------------------------------------------------------
# Tests for filter_forward_kwargs with dLLM batches
# ---------------------------------------------------------------------------


class TestFilterForwardKwargs:
    def test_no_filtering_when_kwargs_accepted(self):
        """When model accepts **kwargs, all batch keys pass through."""
        model = _make_model_with_kwargs()
        batch = {
            "input_ids": torch.zeros(2, 4),
            "input_lengths": torch.tensor([3, 4]),
            "extra_key": torch.ones(2, 4),
        }
        result = filter_forward_kwargs(model, batch)
        assert "input_ids" in result
        assert "input_lengths" in result
        assert "extra_key" in result

    def test_filters_extra_keys(self):
        """Extra keys not in forward() params should be stripped."""
        model = _make_model_without_kwargs()
        batch = {
            "input_ids": torch.zeros(2, 4),
            "attention_mask": torch.ones(2, 4),
            "input_lengths": torch.tensor([3, 4]),
            "noise_mask": torch.ones(2, 4),
        }
        result = filter_forward_kwargs(model, batch)
        assert set(result.keys()) == {"input_ids", "attention_mask"}

    def test_preserves_accepted_keys(self):
        """Keys that match forward() params should be preserved."""
        model = _make_model_without_kwargs()
        batch = {
            "input_ids": torch.zeros(2, 4),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.zeros(2, 4, dtype=torch.long),
        }
        result = filter_forward_kwargs(model, batch)
        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_empty_batch(self):
        model = _make_model_without_kwargs()
        result = filter_forward_kwargs(model, {})
        assert result == {}


# ---------------------------------------------------------------------------
# Integration: strategy.prepare_batch + filter_forward_kwargs
# ---------------------------------------------------------------------------


class TestStrategyPlusFiltering:
    """End-to-end: MDLM strategy prepares batch, then filter_forward_kwargs
    removes unsupported keys for a LLaDA-like model."""

    def test_llada_compatible_batch(self):
        """Batch from collator -> prepare_batch -> filter -> only accepted keys."""
        strategy = MDLMStrategy()

        # Simulate batch from DLLMCollator
        batch = {
            "input_ids": torch.zeros(2, 8, dtype=torch.long),
            "loss_mask": torch.ones(2, 8),
            "attention_mask": torch.ones(2, 8),
            "input_lengths": torch.tensor([6, 8]),
        }
        noisy = torch.ones(2, 8, dtype=torch.long) * 126336
        noise_mask = torch.rand(2, 8) > 0.5
        clean = batch["input_ids"].clone()

        # Step 1: Strategy prepares batch (removes attention_mask for MDLM)
        batch.pop("loss_mask")  # popped by recipe before prepare_batch
        batch = strategy.prepare_batch(batch, noisy, noise_mask, clean)
        assert "attention_mask" not in batch  # removed by strategy
        assert "input_lengths" in batch  # still present

        # Step 2: filter_forward_kwargs removes remaining unsupported keys
        model = _make_model_without_kwargs()
        filtered = filter_forward_kwargs(model, batch)
        assert "input_lengths" not in filtered
        assert "input_ids" in filtered
