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

"""Unit tests for EAGLE recipe _build_checkpointer and eagle3 main().

Covers the fix where EAGLE recipes pass ``model_state_dict_keys`` from the
draft model to ``CheckpointingConfig``, preventing ``TypeError: 'NoneType'
object is not iterable`` at consolidated checkpoint save time.

Also verifies that ``train_eagle3.main(None)`` no longer raises ``ValueError``
before ``parse_args_and_load_config`` has a chance to read ``sys.argv``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn

from nemo_automodel.recipes.llm.train_eagle1 import TrainEagle1Recipe
from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def _bare_recipe_for_checkpointer(cls, tmp_path):
    """Create a minimal recipe instance with just enough state for _build_checkpointer."""
    recipe = cls.__new__(cls)
    recipe.cfg = SimpleNamespace(get=lambda *_args, **_kw: None)
    recipe.output_dir = Path(tmp_path) / "output"
    recipe.output_dir.mkdir(parents=True, exist_ok=True)
    recipe.draft_model = _FakeDraftModel()
    return recipe


# ---------------------------------------------------------------------------
# EAGLE-1: _build_checkpointer sets model_state_dict_keys
# ---------------------------------------------------------------------------


@patch("nemo_automodel.recipes.llm.train_eagle1.Checkpointer")
def test_eagle1_build_checkpointer_sets_model_state_dict_keys(mock_checkpointer, tmp_path):
    """_build_checkpointer must populate model_state_dict_keys from the draft model."""
    recipe = _bare_recipe_for_checkpointer(TrainEagle1Recipe, tmp_path)

    recipe._build_checkpointer(target_path="/fake/target")

    expected_keys = list(recipe.draft_model.state_dict().keys())
    assert recipe.checkpoint_config.model_state_dict_keys is not None
    assert recipe.checkpoint_config.model_state_dict_keys == expected_keys


@patch("nemo_automodel.recipes.llm.train_eagle1.Checkpointer")
def test_eagle1_build_checkpointer_user_none_override_falls_back(mock_checkpointer, tmp_path):
    """If user config explicitly sets model_state_dict_keys=None, the fallback restores draft keys."""
    user_ckpt_cfg = SimpleNamespace(
        to_dict=lambda: {"model_state_dict_keys": None, "checkpoint_dir": str(tmp_path / "user_ckpt")},
    )
    recipe = _bare_recipe_for_checkpointer(TrainEagle1Recipe, tmp_path)
    recipe.cfg = SimpleNamespace(get=lambda key, default=None: user_ckpt_cfg if key == "checkpoint" else default)

    recipe._build_checkpointer(target_path="/fake/target")

    expected_keys = list(recipe.draft_model.state_dict().keys())
    assert recipe.checkpoint_config.model_state_dict_keys == expected_keys


# ---------------------------------------------------------------------------
# EAGLE-3: _build_checkpointer sets model_state_dict_keys
# ---------------------------------------------------------------------------


@patch("nemo_automodel.recipes.llm.train_eagle3.Checkpointer")
def test_eagle3_build_checkpointer_sets_model_state_dict_keys(mock_checkpointer, tmp_path):
    """EAGLE-3 _build_checkpointer must also populate model_state_dict_keys."""
    recipe = _bare_recipe_for_checkpointer(TrainEagle3Recipe, tmp_path)

    recipe._build_checkpointer(target_path="/fake/target")

    expected_keys = list(recipe.draft_model.state_dict().keys())
    assert recipe.checkpoint_config.model_state_dict_keys is not None
    assert recipe.checkpoint_config.model_state_dict_keys == expected_keys


@patch("nemo_automodel.recipes.llm.train_eagle3.Checkpointer")
def test_eagle3_build_checkpointer_user_none_override_falls_back(mock_checkpointer, tmp_path):
    """EAGLE-3: user config model_state_dict_keys=None triggers fallback to draft keys."""
    user_ckpt_cfg = SimpleNamespace(
        to_dict=lambda: {"model_state_dict_keys": None, "checkpoint_dir": str(tmp_path / "user_ckpt")},
    )
    recipe = _bare_recipe_for_checkpointer(TrainEagle3Recipe, tmp_path)
    recipe.cfg = SimpleNamespace(get=lambda key, default=None: user_ckpt_cfg if key == "checkpoint" else default)

    recipe._build_checkpointer(target_path="/fake/target")

    expected_keys = list(recipe.draft_model.state_dict().keys())
    assert recipe.checkpoint_config.model_state_dict_keys == expected_keys


# ---------------------------------------------------------------------------
# EAGLE-3: main(None) no longer raises ValueError
# ---------------------------------------------------------------------------


def test_eagle3_main_none_does_not_raise_valueerror():
    """main(None) must not raise ValueError; it should reach parse_args_and_load_config."""
    with patch("nemo_automodel.recipes.llm.train_eagle3.parse_args_and_load_config") as mock_parse:
        mock_parse.side_effect = SystemExit(2)
        with pytest.raises(SystemExit):
            from nemo_automodel.recipes.llm.train_eagle3 import main

            main(None)
    # The key assertion: no ValueError was raised before parse_args_and_load_config.
    mock_parse.assert_called_once_with(None)
