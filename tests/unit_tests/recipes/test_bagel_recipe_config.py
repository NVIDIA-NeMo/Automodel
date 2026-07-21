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

"""Regression tests for BAGEL recipe typed-config initialization."""

from pathlib import Path

import pytest

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.multimodal.loader import BagelDataloaderConfig
from nemo_automodel.recipes.multimodal.finetune import FinetuneRecipeForMultimodal
from nemo_automodel.recipes.multimodal.pretrain import PretrainRecipeForMultimodal

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("recipe_path", "recipe_type"),
    [
        (
            "examples/multimodal_finetune/bagel/bagel_sft.yaml",
            FinetuneRecipeForMultimodal,
        ),
        (
            "examples/multimodal_pretrain/bagel/bagel_pretrain.yaml",
            PretrainRecipeForMultimodal,
        ),
    ],
)
def test_bagel_recipe_resolves_typed_dataloader_from_shipped_config(recipe_path, recipe_type):
    """Published BAGEL YAMLs must expose their typed dataloader through the recipe."""
    raw_config = parse_args_and_load_config(_REPO_ROOT / recipe_path)

    recipe = recipe_type(raw_config)
    dataloader_config = recipe.cfg.bagel_dataloader

    assert isinstance(dataloader_config, BagelDataloaderConfig)
    assert dataloader_config.num_workers == 1
    assert dataloader_config.pin_memory is True
    assert dataloader_config.prefetch_factor == 2
