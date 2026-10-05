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

"""Keep setup CUDA operations on the rank's selected device."""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.recipes.llm.benchmark import BenchmarkingRecipeForNextTokenPrediction
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


class SetupObserved(Exception):
    """Stop after the setup action under test, before model allocation."""


@pytest.mark.parametrize("cuda_available", [False, True])
def test_benchmark_selects_rank_before_setup_timer(monkeypatch: pytest.MonkeyPatch, cuda_available: bool) -> None:
    events: list[str] = []

    def set_device(rank: int) -> None:
        assert rank == 3
        events.append("rank3")

    @contextmanager
    def timer(name: str, *, log_level: int) -> Iterator[None]:
        assert name == "setup"
        assert log_level == 1
        events.append("timer")
        raise SetupObserved
        yield  # pragma: no cover

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    recipe = object.__new__(BenchmarkingRecipeForNextTokenPrediction)
    recipe.timers = timer
    with pytest.raises(SetupObserved):
        recipe.setup()
    assert events == (["rank3", "timer"] if cuda_available else ["timer"])


def test_training_selects_rank_before_memory_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def initialize(*, backend: str, timeout_minutes: int) -> SimpleNamespace:
        assert backend == "nccl"
        assert timeout_minutes == 1
        events.append("distributed")
        return SimpleNamespace(device=torch.device("cuda", 3))

    def reset_memory() -> None:
        events.append("reset")
        raise SetupObserved

    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.initialize_distributed", initialize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_memory)
    recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    recipe.cfg = {}
    with pytest.raises(SetupObserved):
        recipe.setup()
    assert events == ["distributed", "reset"]
