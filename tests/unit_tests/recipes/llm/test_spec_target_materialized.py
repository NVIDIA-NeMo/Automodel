# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the frozen-target materialization guard.

The failure this guards against is silent: a target initialized on the meta
device and never materialized becomes a random teacher, the draft trains
against noise, and nothing raises. The guard has to catch that without firing
on a genuinely loaded checkpoint.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.recipes.llm._spec_train_utils import raise_if_target_not_materialized

VOCAB = 512
HIDDEN = 64


class _Target(nn.Module):
    def __init__(self, std: float | None = None, initializer_range: float = 0.02):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        if std is not None:
            with torch.no_grad():
                self.embed.weight.normal_(0.0, std)
        self.config = SimpleNamespace(initializer_range=initializer_range)

    def get_input_embeddings(self):
        return self.embed


def test_untrained_embedding_is_rejected():
    """An embedding still at N(0, initializer_range) means nothing was loaded."""
    target = _Target(std=0.02)
    with pytest.raises(RuntimeError, match="never loaded from the checkpoint"):
        raise_if_target_not_materialized(target, "some/target")


def test_trained_embedding_passes():
    """A real checkpoint's norm sits well away from the initialization scale."""
    target = _Target(std=0.02)
    with torch.no_grad():
        target.embed.weight.mul_(3.0)
    raise_if_target_not_materialized(target, "some/target")


def test_meta_parameter_is_rejected_by_name():
    target = _Target(std=0.06)
    with torch.device("meta"):
        target.extra = nn.Linear(HIDDEN, HIDDEN)
    with pytest.raises(RuntimeError, match="meta device"):
        raise_if_target_not_materialized(target, "some/target")


def test_guard_follows_the_configured_initializer_range():
    """A model declaring a different initializer_range is judged against it."""
    target = _Target(std=0.05, initializer_range=0.05)
    with pytest.raises(RuntimeError, match="never loaded"):
        raise_if_target_not_materialized(target, "some/target")
    # The same weights are fine when the declared range is the usual 0.02,
    # because they no longer look like that initialization.
    target.config.initializer_range = 0.02
    raise_if_target_not_materialized(target, "some/target")


def test_expected_untrained_norm_matches_the_closed_form():
    """Guards the constant the check relies on."""
    target = _Target(std=0.02)
    observed = target.embed.weight.float().norm().item()
    assert observed == pytest.approx(0.02 * math.sqrt(VOCAB * HIDDEN), rel=0.05)


def test_model_without_embedding_weight_is_skipped():
    """A target whose embeddings expose no weight tensor cannot be judged."""
    target = _Target(std=0.06)
    target.get_input_embeddings = lambda: SimpleNamespace(weight=None)
    raise_if_target_not_materialized(target, "some/target")
