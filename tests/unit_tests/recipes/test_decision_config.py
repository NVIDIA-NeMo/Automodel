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

"""Decision examples resolve their native recipes without creating models."""

import importlib
from pathlib import Path

import pytest
import yaml

from nemo_automodel.recipes.base_recipe import BaseRecipe


@pytest.mark.parametrize("filename", ["clm_soc_triage.yaml", "clef_soc_triage_lora.yaml", "clef_soc_triage_heads.yaml"])
def test_example_recipe_targets_resolve(filename):
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "examples" / "decision" / filename).read_text())
    module, _, name = config["recipe"].rpartition(".")
    recipe = getattr(importlib.import_module(module), name)
    assert issubclass(recipe, BaseRecipe)
    assert callable(recipe.setup)
    assert callable(recipe.run_train_validation_loop)


@pytest.mark.parametrize("local_batch,global_batch", [(0, 4), (2, 0), (2, 3), (2, 2)])
def test_clef_rejects_nonpositive_or_nondivisible_batches(local_batch, global_batch):
    from nemo_automodel.components.decision.losses import DecisionLossConfig
    from nemo_automodel.recipes.decision.train_clef import TrainClefDecisionRecipe

    recipe = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    recipe.loss_config = DecisionLossConfig()
    recipe.cfg = {"train.local_batch_size": local_batch, "train.global_batch_size": global_batch}
    recipe._get_dp_group_size = lambda: 2
    with pytest.raises(ValueError, match="batch_size"):
        recipe._train()


def test_clef_tiny_training_split_cannot_report_success_without_an_optimizer_step(tmp_path):
    from nemo_automodel.components.decision.losses import DecisionLossConfig
    from nemo_automodel.recipes.decision.train_clef import TrainClefDecisionRecipe

    recipe = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    recipe.loss_config = DecisionLossConfig()
    recipe.cfg = {"train.local_batch_size": 2, "train.global_batch_size": 32}
    recipe._get_dp_group_size = lambda: 1
    recipe._get_dp_rank = lambda: 0
    recipe.splits = {"train": list(range(16))}
    recipe.output_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="no complete optimizer batch"):
        recipe._train()
    assert not recipe.output_dir.exists()


def test_clef_reward_only_validation_does_not_require_accuracy():
    from nemo_automodel.recipes.decision.train_clef import TrainClefDecisionRecipe

    recipe = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    recipe.splits = {"validation": []}
    recipe.evaluate = lambda split: {"overall": {"reward": 0.75}}
    assert recipe._val_score("reward") == 0.75
    with pytest.raises(ValueError, match="validation metric 'accuracy' is unavailable"):
        recipe._val_score("accuracy")


def test_clm_reward_only_training_and_validation_use_explicit_utilities(tmp_path):
    import torch

    from nemo_automodel.components.decision.losses import DecisionLossConfig
    from nemo_automodel.components.models.clm.heads import CLMHeadConfig, CLMHeadPair, EmbeddingTable, collect_clm_texts
    from nemo_automodel.recipes.decision.train_clm_heads import TrainCLMHeadsRecipe

    record = {
        "id": "reward-only",
        "state": "A fictional event",
        "questions": {"pick": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
        "rewards": {"pick": {"a": 0.0, "b": 1.0}},
    }
    torch.manual_seed(7)
    recipe = TrainCLMHeadsRecipe.__new__(TrainCLMHeadsRecipe)
    recipe.cfg = {"train.epochs": 1, "train.batch_size": 1, "train.eval_metric": "reward"}
    recipe.loss_config = DecisionLossConfig(mode="rl", kl_coef=0.0)
    recipe.tag = "rl"
    recipe.seed = 7
    recipe.device = torch.device("cpu")
    recipe.output_dir = tmp_path
    recipe.slice_keys = []
    recipe.splits = {"train": [record], "validation": [record]}
    texts = collect_clm_texts([record])
    recipe.table = EmbeddingTable(texts, torch.randn(len(texts), 8))
    head = CLMHeadConfig(
        hidden_size=8, width=8, depth=2, projection_dim=4, activation="gelu", layer_norm=False, residual=False
    ).build()
    trained = recipe._train(head)
    assert isinstance(trained, CLMHeadPair)
    assert (tmp_path / "training_log.jsonl").exists()
