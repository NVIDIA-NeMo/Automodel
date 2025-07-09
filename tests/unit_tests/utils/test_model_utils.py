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

from __future__ import annotations

from typing import Dict

import pytest
import torch.nn as nn

import nemo_automodel.utils.model_utils as model_utils


@pytest.fixture()
def dummy_model() -> nn.Module:
    """
    Create a minimal but representative model containing:
    - An nn.Embedding layer
    - A `vision_tower` sub-module
    - Another module whose name contains “visual” (to test name-based pattern)
    - A `language_model` backbone
    - An additional unfrozen Linear layer (“other”) for sanity checks
    """

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embed = nn.Embedding(10, 3)  # embeddings
            self.vision_tower = nn.Sequential(nn.Linear(4, 4))  # vision tower attr
            self.visual_extra = nn.Sequential(nn.Linear(5, 5))  # triggers "visual" name pattern
            self.language_model = nn.Sequential(nn.Linear(6, 6))
            self.other = nn.Linear(7, 7)

        def forward(self, x):  # pragma: no cover
            pass

    return DummyModel()


def _all_requires_grad(module: nn.Module) -> bool:
    """Return True if every parameter in `module` requires gradients."""
    return all(p.requires_grad for p in module.parameters())


def _any_requires_grad(module: nn.Module) -> bool:
    """Return True if at least one parameter in `module` requires gradients."""
    return any(p.requires_grad for p in module.parameters())


def test_print_trainable_parameters_counts(dummy_model, capsys, monkeypatch):
    """
    Ensure the helper returns correct (trainable, total) counts
    and prints to stdout only when rank == 0.
    """
    # Mark one parameter as frozen beforehand to have different numbers
    dummy_model.other.weight.requires_grad = False

    # Force rank 0
    monkeypatch.setattr(model_utils, "get_rank_safe", lambda: 0)

    trainable, total = model_utils.print_trainable_parameters(dummy_model)
    captured = capsys.readouterr()

    assert trainable == sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    assert total == sum(p.numel() for p in dummy_model.parameters())

    # Basic sanity check on stdout
    assert "Trainable parameters" in captured.out
    assert "Total parameters" in captured.out
    assert trainable != total  # because we manually froze a param above


def test_print_trainable_parameters_non_zero_rank(dummy_model, capsys, monkeypatch):
    """
    Helper must stay silent for non-zero ranks.
    """
    monkeypatch.setattr(model_utils, "get_rank_safe", lambda: 1)
    _ = model_utils.print_trainable_parameters(dummy_model)
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.parametrize(
    "freeze_cfg, expect",
    [
        (
            {"freeze_embeddings": True, "freeze_vision_tower": False, "freeze_language_model": False},
            {"emb": False, "vision": True, "lang": True, "other": True},
        ),
        (
            {"freeze_embeddings": False, "freeze_vision_tower": True, "freeze_language_model": False},
            {"emb": True, "vision": False, "lang": True, "other": True},
        ),
        (
            {"freeze_embeddings": False, "freeze_vision_tower": False, "freeze_language_model": True},
            {"emb": True, "vision": True, "lang": False, "other": True},
        ),
        (
            {},  # rely on in-code defaults: embeddings=True, vision=True, language=False
            {"emb": False, "vision": False, "lang": True, "other": True},
        ),
    ],
)
def test_apply_parameter_freezing(dummy_model, freeze_cfg: Dict, expect: Dict):
    """
    Parametrized test to verify that each freeze flag affects the right sub-modules.

    `expect` dict uses:
        emb   -> require_grad status for Embedding
        vision-> require_grad status for *all* vision-related parameters
        lang  -> require_grad status for language_model
        other -> require_grad status for the unrelated `other` layer
    A value of True means gradients SHOULD be enabled; False means frozen.
    """
    # Reset all grads before every run (pytest reuses the same fixture instance)
    for p in dummy_model.parameters():
        p.requires_grad = True

    model_utils.apply_parameter_freezing(dummy_model, freeze_cfg)

    # embeddings
    assert dummy_model.token_embed.weight.requires_grad is expect["emb"]

    # vision tower(s)
    assert _all_requires_grad(dummy_model.vision_tower) is expect["vision"]
    assert _all_requires_grad(dummy_model.visual_extra) is expect["vision"]

    # language model
    assert _all_requires_grad(dummy_model.language_model) is expect["lang"]

    # unrelated layer
    assert dummy_model.other.weight.requires_grad is expect["other"]
    assert dummy_model.other.bias.requires_grad is expect["other"]
