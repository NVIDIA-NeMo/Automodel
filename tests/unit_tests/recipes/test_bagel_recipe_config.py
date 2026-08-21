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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.multimodal.loader import BagelDataloaderConfig
from nemo_automodel.components.distributed import init_utils
from nemo_automodel.recipes.multimodal import finetune as bagel_finetune
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


def test_bagel_recipe_skips_wandb_when_shipped_config_disables_it(monkeypatch):
    """A typed config must preserve the parser's disabled-wandb behavior."""
    raw_config = parse_args_and_load_config(_REPO_ROOT / "examples/multimodal_finetune/bagel/bagel_sft.yaml")
    assert raw_config.get("wandb", None) is None

    recipe = FinetuneRecipeForMultimodal(raw_config)
    monkeypatch.setattr(bagel_finetune.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bagel_finetune.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        init_utils,
        "initialize_distributed",
        lambda **_kwargs: SimpleNamespace(is_main=True, world_size=1, rank=0),
    )
    monkeypatch.setattr(bagel_finetune, "setup_logging", lambda: None)
    monkeypatch.setattr(
        bagel_finetune,
        "create_distributed_setup_from_config",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        recipe,
        "_distributed_setup_attributes",
        lambda _setup: (None, None, None, None, None, False, None, None, False),
    )

    class StopAfterWandb(Exception):
        pass

    build_wandb = Mock()
    monkeypatch.setattr(recipe, "_build_wandb", build_wandb)
    monkeypatch.setattr(recipe, "_log_experiment_details", Mock(side_effect=StopAfterWandb))

    with pytest.raises(StopAfterWandb):
        recipe.setup()

    build_wandb.assert_not_called()
