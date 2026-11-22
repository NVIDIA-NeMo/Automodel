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

"""The draft recipes have no per-module entry point; the ``automodel`` CLI launches them.

These tests pin the contract that replaced the per-recipe ``main()`` functions:
every draft recipe class resolves from its bare ``recipe:`` name, and every
speculative example config names a recipe the CLI can run.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from nemo_automodel.cli.utils import resolve_recipe_name
from nemo_automodel.components.launcher.interactive import resolve_recipe_cls
from nemo_automodel.recipes.speculative_decoding._spec_recipe_base import SpecDecodeRecipeBase

_ALGORITHMS = "nemo_automodel.recipes.speculative_decoding.algorithms"
_EXAMPLES = pathlib.Path(__file__).resolve().parents[4] / "examples" / "speculative"


@pytest.mark.parametrize(
    ("recipe_name", "module"),
    [
        ("TrainEagle1Recipe", "eagle"),
        ("TrainEagle2Recipe", "eagle"),
        ("TrainEagle3Recipe", "eagle3"),
        ("TrainDFlashRecipe", "dflash"),
        ("TrainDFlash2Recipe", "dflash2"),
        ("TrainDominoRecipe", "domino"),
        ("TrainJetSpecRecipe", "jetspec"),
        ("TrainDSparkRecipe", "dspark"),
        ("TrainVispecRecipe", "vispec"),
        ("TrainVispecStage1Recipe", "vispec"),
    ],
)
def test_cli_resolves_draft_recipe_by_bare_name(recipe_name, module):
    target = resolve_recipe_name(recipe_name)

    assert target == f"{_ALGORITHMS}.{module}.{recipe_name}"
    recipe_cls = resolve_recipe_cls(target)
    assert recipe_cls.__name__ == recipe_name
    assert issubclass(recipe_cls, SpecDecodeRecipeBase)


# Precompute and benchmark configs run their own entry point and name no recipe.
_RECIPE_CONFIGS = sorted(
    (path, recipe)
    for path in _EXAMPLES.rglob("*.yaml")
    if (recipe := (yaml.safe_load(path.read_text()) or {}).get("recipe")) is not None
)


def test_speculative_examples_name_recipes():
    assert len(_RECIPE_CONFIGS) >= 50


@pytest.mark.parametrize(("config_path", "recipe_name"), _RECIPE_CONFIGS, ids=lambda v: getattr(v, "name", v))
def test_speculative_example_recipe_resolves_to_a_draft_recipe(config_path, recipe_name):
    recipe_cls = resolve_recipe_cls(resolve_recipe_name(recipe_name))

    assert issubclass(recipe_cls, SpecDecodeRecipeBase), config_path
