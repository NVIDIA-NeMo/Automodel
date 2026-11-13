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

"""Shared language/vision-language optimizer-step accounting."""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig, TokenCounts
from nemo_automodel.recipes.base_recipe import BaseRecipe

pytestmark = pytest.mark.cpu_only


def _recipe(*, stream):
    recipe = BaseRecipe.__new__(BaseRecipe)
    recipe.loss_fn = SimpleNamespace(ignore_index=-1)
    # Internal ignore values count toward throughput; only tail -100 padding
    # is excluded. Both dataset and loss ignore values are unsupervised.
    batches = [
        {"labels": torch.tensor([[1, -1, 3, -100]])},
        {"labels": torch.tensor([[-100, 4, -100, -100]])},
        {"labels": torch.tensor([[5, 6, 7, -100]])},
    ]
    recipe.step_scheduler = StepSchedulerConfig(global_batch_size=2, max_steps=10, preemption_signal=None).build(
        batches, 1, 1, synchronized_stream=stream
    )
    return recipe


def test_shared_counts_preserve_both_ignore_indices_and_tail_padding():
    recipe = _recipe(stream=False)
    batches, _ = next(recipe.step_scheduler.iter_steps(recipe._count_step_tokens))
    assert recipe._count_step_tokens(batches) == TokenCounts(3, 5)


def test_other_loaders_retain_partial_windows_without_preparing_or_reducing_counts():
    recipe = _recipe(stream=False)

    def unexpected_preparation(*args):
        raise AssertionError("Other loaders keep recipe-owned token counting")

    recipe._dp_allreduce = unexpected_preparation
    recipe._count_step_tokens = unexpected_preparation
    steps = list(recipe._iter_training_steps())
    assert [len(batches) for batches, _ in steps] == [2, 1]
    assert [counts for _, counts in steps] == [None, None]
    assert recipe.step_scheduler.epoch == 1


def test_stream_counts_are_not_reduced_again_by_shared_recipe():
    recipe = _recipe(stream=True)

    def unexpected_reduce(counts):
        raise AssertionError("Streaming readiness already reduced the token counts")

    recipe._dp_allreduce = unexpected_reduce
    steps = list(recipe._iter_training_steps())
    assert [counts for _, counts in steps] == [TokenCounts(3, 5)]
    assert recipe.step_scheduler.stream_exhausted
