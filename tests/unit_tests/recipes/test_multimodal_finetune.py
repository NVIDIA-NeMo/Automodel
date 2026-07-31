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

import ast
from pathlib import Path

import pytest
import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.multimodal.loader import BagelDataloaderConfig
from nemo_automodel.components.loggers.loggers import WandbConfig
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.multimodal.finetune import FinetuneRecipeForMultimodal
from nemo_automodel.recipes.multimodal.pretrain import PretrainRecipeForMultimodal

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SFT_CONFIG = "examples/multimodal_finetune/bagel/bagel_sft.yaml"
_PRETRAIN_CONFIG = "examples/multimodal_pretrain/bagel/bagel_pretrain.yaml"


def test_bagel_auto_model_path_uses_distributed_setup_kwarg():
    """BAGEL's AutoModel path must match the shared VLM build_model API."""
    recipe_path = Path(__file__).resolve().parents[3] / "nemo_automodel/recipes/multimodal/finetune.py"
    tree = ast.parse(recipe_path.read_text())

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "build_vlm_model"
    ]
    assert len(calls) == 1

    keywords = {kw.arg for kw in calls[0].keywords}
    assert "distributed_setup" in keywords
    assert (
        not {
            "device_mesh",
            "moe_mesh",
            "distributed_config",
            "pipeline_config",
            "cfg_moe",
            "activation_checkpointing",
        }
        & keywords
    )


def test_bagel_finalizes_pending_checkpoint_before_closing_checkpointer():
    """Async BAGEL checkpoints must be published before the checkpointer closes."""
    from types import SimpleNamespace

    from nemo_automodel.recipes.multimodal.finetune import FinetuneRecipeForMultimodal

    events = []
    recipe = FinetuneRecipeForMultimodal.__new__(FinetuneRecipeForMultimodal)
    recipe.model = SimpleNamespace(train=lambda: events.append("train"))
    recipe.step_scheduler = SimpleNamespace(epochs=[])
    recipe.metric_logger_train = SimpleNamespace(close=lambda: events.append("train_logger_close"))
    recipe.metric_logger_valid = SimpleNamespace(close=lambda: events.append("valid_logger_close"))
    recipe.checkpointer = SimpleNamespace(close=lambda: events.append("checkpointer_close"))
    recipe._finalize_pending_checkpoint = lambda: events.append("finalize")

    FinetuneRecipeForMultimodal.run_train_validation_loop(recipe)

    assert events == ["train", "train_logger_close", "valid_logger_close", "finalize", "checkpointer_close"]


@pytest.mark.parametrize(
    ("config_path", "recipe_cls"),
    [
        (_SFT_CONFIG, FinetuneRecipeForMultimodal),
        (_PRETRAIN_CONFIG, PretrainRecipeForMultimodal),
    ],
)
def test_bagel_recipe_resolves_typed_sections_from_shipped_config(config_path, recipe_cls):
    """The parser hands the recipe a raw ConfigNode; setup() consumes typed sections.

    Both shipped YAMLs must therefore resolve ``bagel_dataloader`` while the
    sections the recipe still reads as raw nodes keep delegating.
    """
    recipe = recipe_cls(parse_args_and_load_config(_REPO_ROOT / config_path))

    assert isinstance(recipe.cfg, RecipeConfig)

    dataloader_config = recipe.cfg.bagel_dataloader
    assert isinstance(dataloader_config, BagelDataloaderConfig)
    assert dataloader_config.num_workers == 1
    assert dataloader_config.pin_memory is True
    assert dataloader_config.prefetch_factor == 2

    # Sections without a typed accessor still resolve against the raw node.
    assert recipe.cfg.get("step_scheduler.local_batch_size", None) == 1
    assert recipe.cfg.model.get("stage", None) == recipe.cfg.get("model.stage", None)


def test_bagel_recipe_wrapping_is_idempotent():
    """Re-entrant construction (recipe -> recipe) must not double-wrap the config."""
    raw = parse_args_and_load_config(_REPO_ROOT / _SFT_CONFIG)
    recipe = FinetuneRecipeForMultimodal(FinetuneRecipeForMultimodal(raw).cfg)

    assert recipe.cfg.bagel_dataloader.num_workers == 1


def test_bagel_recipe_builds_optimizer_over_trainable_params_only():
    """The typed optimizer config owns construction and skips frozen parameters."""
    recipe = FinetuneRecipeForMultimodal(parse_args_and_load_config(_REPO_ROOT / _SFT_CONFIG))

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model[1].requires_grad_(False)

    optimizers = recipe._build_optimizers(model, device_mesh=None)

    assert len(optimizers) == 1
    assert isinstance(optimizers[0], torch.optim.AdamW)
    (param_group,) = optimizers[0].param_groups
    assert param_group["lr"] == pytest.approx(2.0e-5)
    assert tuple(param_group["betas"]) == (0.9, 0.95)
    # The shipped YAML disables foreach; the typed path must forward it verbatim.
    assert param_group["foreach"] is False
    assert [id(p) for p in param_group["params"]] == [id(p) for p in model[0].parameters()]


def test_bagel_recipe_optimizer_build_requires_an_optimizer_section():
    """A config without an ``optimizer:`` block fails with an actionable error."""
    recipe = FinetuneRecipeForMultimodal(ConfigNode({"model": {"stage": 1}}))

    with pytest.raises(ValueError, match="optimizer"):
        recipe._build_optimizers(torch.nn.Linear(4, 4), device_mesh=None)


def test_bagel_recipe_skips_wandb_when_shipped_config_disables_it():
    """``wandb.enable: false`` is dropped by the parser, so setup() must see ``None``."""
    recipe = FinetuneRecipeForMultimodal(parse_args_and_load_config(_REPO_ROOT / _SFT_CONFIG))

    assert recipe.cfg.get("wandb", None) is None
    assert recipe.cfg.wandb is None


def test_bagel_recipe_resolves_typed_wandb_when_enabled():
    """Enabling W&B yields the typed config setup() builds the run from."""
    raw = parse_args_and_load_config(_REPO_ROOT / _SFT_CONFIG, ["--wandb.enable", "true"])
    recipe = FinetuneRecipeForMultimodal(raw)

    wandb_config = recipe.cfg.wandb
    assert isinstance(wandb_config, WandbConfig)
    assert wandb_config.project == "bagel-finetuning"
    assert wandb_config.name == "bagel_7b_mot_sft"
    # Non-field wandb.init kwargs stay available for the run.
    assert wandb_config.extra["mode"] == "disabled"
