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
    from nemo_automodel.components.loss.decision import DecisionLossConfig
    from nemo_automodel.recipes.decision.train_clef import TrainClefDecisionRecipe

    recipe = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    recipe.loss_config = DecisionLossConfig()
    recipe.cfg = {"train.local_batch_size": local_batch, "train.global_batch_size": global_batch}
    recipe._get_dp_group_size = lambda: 2
    with pytest.raises(ValueError, match="batch_size"):
        recipe._train()


def test_clef_tiny_training_split_cannot_report_success_without_an_optimizer_step(tmp_path):
    from nemo_automodel.components.loss.decision import DecisionLossConfig
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


@pytest.mark.parametrize("metric", ["brier", "nll", "ece", "records", "n"])
def test_native_recipes_reject_metrics_that_cannot_be_maximized(metric):
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.components.loss.decision import DecisionLossConfig
    from nemo_automodel.components.models.clm.heads import CLMHeadConfig
    from nemo_automodel.recipes.decision.train_clef import TrainClefDecisionRecipe
    from nemo_automodel.recipes.decision.train_clm_heads import TrainCLMHeadsRecipe

    clef = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    with pytest.raises(ValueError, match="train.eval_metric"):
        clef._val_score(metric)
    clm = TrainCLMHeadsRecipe(ConfigNode({"train": {"eval_metric": metric}}))
    clm.loss_config = DecisionLossConfig()
    heads = CLMHeadConfig(4, 4, 2, 2, "gelu", False, False).build()
    with pytest.raises(ValueError, match="train.eval_metric"):
        clm._train(heads)


def test_clm_reward_only_training_and_validation_use_explicit_utilities(tmp_path):
    import torch

    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.components.loss.decision import DecisionLossConfig
    from nemo_automodel.components.models.clm.heads import CLMHeadConfig, CLMHeadPair, EmbeddingTable, collect_clm_texts
    from nemo_automodel.recipes.decision.train_clm_heads import TrainCLMHeadsRecipe

    record = {
        "id": "reward-only",
        "state": "A fictional event",
        "questions": {"pick": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
        "rewards": {"pick": {"a": 0.0, "b": 1.0}},
    }
    torch.manual_seed(7)
    recipe = TrainCLMHeadsRecipe(
        ConfigNode(
            {
                "train": {"epochs": 1, "batch_size": 1, "eval_metric": "reward", "train_logit_scale": False},
                "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
                "lr_scheduler": {"lr_decay_style": "constant", "lr_warmup_steps": 0},
            }
        )
    )
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
    before = {name: value.detach().clone() for name, value in head.named_parameters()}
    trained = recipe._train(head)
    assert isinstance(trained, CLMHeadPair)
    assert (tmp_path / "training_log.jsonl").exists()
    assert not head.logit_scale.requires_grad
    assert head.logit_scale.grad is None
    torch.testing.assert_close(head.logit_scale, before["logit_scale"], atol=0, rtol=0)
    assert any(not torch.equal(value, before[name]) for name, value in head.named_parameters() if name != "logit_scale")


def test_primary_and_head_optimizers_resolve_independently_from_yaml():
    import torch

    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes._typed_config import RecipeConfig

    raw = ConfigNode(
        {
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.02},
            "head_optimizer": {"_target_": "torch.optim.AdamW", "lr": 0.003, "weight_decay": 0.0},
        }
    )
    config = RecipeConfig(raw)
    first = torch.nn.Linear(2, 1)
    second = torch.nn.Linear(2, 1)
    primary = config.optimizer.build(first)
    head = config.head_optimizer.build(second)
    assert isinstance(primary[0], torch.optim.SGD)
    assert isinstance(head[0], torch.optim.AdamW)
    assert primary[0].param_groups[0]["lr"] == 0.02
    assert head[0].param_groups[0]["lr"] == 0.003
    assert config.head_optimizer is config.head_optimizer
    assert config.optimizer is config.optimizer
    assert config.head_optimizer is not config.optimizer
    assert RecipeConfig(ConfigNode({})).head_optimizer is None


def test_clm_example_native_optimizer_and_schedule_build_without_gpu():
    import torch

    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig
    from nemo_automodel.recipes._typed_config import RecipeConfig

    root = Path(__file__).resolve().parents[3]
    config = RecipeConfig(ConfigNode(yaml.safe_load((root / "examples/decision/clm_soc_triage.yaml").read_text())))
    model = torch.nn.Linear(2, 1)
    optimizer = config.optimizer.build(model)
    schedule = StepSchedulerConfig(global_batch_size=2, num_epochs=2, max_steps=6, preemption_signal=None).build(
        dataloader=None, dp_group_size=1, local_batch_size=2
    )
    schedulers = config.lr_scheduler.build(optimizer, schedule)
    assert len(schedulers) == 1
    assert schedulers[0].lr_decay_steps == 6
    assert schedulers[0].lr_warmup_steps == 0
    schedulers[0].step(1)
    assert 0 < optimizer[0].param_groups[0]["lr"] < 5e-4


def test_clm_configured_sgd_update_matches_unpadded_ce_reference(tmp_path):
    import copy

    import torch
    from torch.nn import functional

    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.components.loss.decision import DecisionLossConfig
    from nemo_automodel.components.models.clm.heads import CLMHeadConfig, EmbeddingTable, collect_clm_texts
    from nemo_automodel.recipes.decision.train_clm_heads import TrainCLMHeadsRecipe

    record = {
        "id": "sgd-step",
        "state": "A fictional event",
        "questions": {"pick": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
        "answers": {"pick": "b"},
    }
    torch.manual_seed(11)
    recipe = TrainCLMHeadsRecipe(
        ConfigNode(
            {
                "train": {"epochs": 1, "batch_size": 1, "grad_clip": 1e6},
                "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.03},
            }
        )
    )
    recipe.loss_config = DecisionLossConfig(label_smoothing=0.0, brier_weight=0.0, ordinal_weight=0.0)
    recipe.tag = "sft"
    recipe.seed = 11
    recipe.device = torch.device("cpu")
    recipe.output_dir = tmp_path
    recipe.splits = {"train": [record]}
    texts = collect_clm_texts([record])
    recipe.table = EmbeddingTable(texts, torch.randn(len(texts), 8))
    head = CLMHeadConfig(
        hidden_size=8, width=8, depth=2, projection_dim=4, activation="gelu", layer_norm=False, residual=False
    ).build()
    reference = copy.deepcopy(head)
    reference_logits, targets = recipe._forward(reference, [record])
    loss = functional.cross_entropy(reference_logits[0][0].values[None], torch.tensor([targets[0][0].label]))
    loss.backward()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    reference_optimizer.step()
    trained = recipe._train(head)
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(dict(trained.named_parameters())[name], parameter, atol=1e-7, rtol=1e-6)


def test_clm_setup_rejects_empty_split_before_loading_encoder(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.decision import train_clm_heads

    empty = tmp_path / "empty.jsonl"
    empty.touch()
    monkeypatch.setattr(
        train_clm_heads, "initialize_distributed", lambda **kwargs: SimpleNamespace(device=torch.device("cpu"))
    )
    recipe = train_clm_heads.TrainCLMHeadsRecipe(ConfigNode({"data": {"train": str(empty)}}))
    with pytest.raises(ValueError, match="data.train must contain at least one record"):
        recipe.setup()


def test_clef_setup_preserves_normalized_checkpoint_config(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from nemo_automodel import NeMoAutoModelForImageTextToText
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.components.models.clef.joint_schema_head import ClefHeadConfig
    from nemo_automodel.recipes.decision import train_clef

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Embedding(8, 8).requires_grad_(False)

        def get_output_embeddings(self) -> torch.nn.Embedding:
            """Return the frozen output table with weight of shape [vocabulary, hidden]."""
            return self.output

    root = Path(train_clef.__file__).resolve().parents[3]
    config = yaml.safe_load((root / "examples/decision/clef_soc_triage_heads.yaml").read_text())
    config.update(mode="eval", output_dir=str(tmp_path / "run"))
    record = {"id": "one", "state": "text", "questions": {"pick": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}}
    data = tmp_path / "data.jsonl"
    import json

    data.write_text(json.dumps(record) + "\n")
    for split in ("train", "validation", "test"):
        config["data"][split] = str(data)
    monkeypatch.setattr(
        train_clef, "initialize_distributed", lambda **kwargs: SimpleNamespace(device=torch.device("cpu"), world_size=1)
    )
    monkeypatch.setattr(train_clef, "create_distributed_setup_from_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_clef.TrainClefDecisionRecipe,
        "_distributed_setup_attributes",
        lambda *args: (None, SimpleNamespace(tp_size=1, cp_size=1), None, None, None, False, None, None, False),
    )
    monkeypatch.setattr(train_clef.TrainClefDecisionRecipe, "_get_dp_rank", lambda *args: 0)
    monkeypatch.setattr(train_clef.TrainClefDecisionRecipe, "_dp_group", lambda *args: None)
    monkeypatch.setattr(NeMoAutoModelForImageTextToText, "from_pretrained", lambda *args, **kwargs: Backbone())

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1])

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: Tokenizer())
    head = ClefHeadConfig(hidden_size=8, width=8, routing_layers=1, layers=1, heads=2, feedforward=16).build()
    monkeypatch.setattr(train_clef, "load_joint_head", lambda *args, **kwargs: head)
    recipe = train_clef.TrainClefDecisionRecipe(ConfigNode(config))
    try:
        recipe.setup()
        assert recipe.checkpointer.config is not recipe.cfg.checkpoint
        assert recipe.checkpointer.config.checkpoint_dir == str(tmp_path / "run")
        assert str(recipe.cfg.checkpoint.checkpoint_dir) == "checkpoints/"
        assert recipe.checkpointer.config.model_save_format == recipe.cfg.checkpoint.model_save_format
        assert len(recipe.splits["train"]) == 1
        assert recipe.lexical.rows.shape == (1, 8)
    finally:
        if hasattr(recipe, "checkpointer"):
            recipe.checkpointer.close()


@pytest.mark.parametrize("entrypoint", ["setup", "run_train_validation_loop"])
@pytest.mark.parametrize("stale_pointer", [False, True])
def test_clef_training_rejects_disabled_checkpoints_before_training_or_loading_old_stage(
    tmp_path, monkeypatch, entrypoint, stale_pointer
):
    from types import SimpleNamespace

    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.decision import train_clef

    output = tmp_path / "run"
    if stale_pointer:
        old_stage = output / "old-stage"
        old_stage.mkdir(parents=True)
        (output / "LOWEST_VAL").symlink_to(old_stage.name)
    recipe = train_clef.TrainClefDecisionRecipe(ConfigNode({"mode": "train", "checkpoint": {"enabled": False}}))
    recipe.mode = "train"
    recipe.output_dir = output
    recipe.splits = {}
    observed = []
    recipe._train = lambda: observed.append("train")
    recipe._load_stage = lambda path: observed.append(path)
    recipe.checkpointer = SimpleNamespace(finalize=lambda: None)
    monkeypatch.setattr(train_clef, "initialize_distributed", lambda **kwargs: SimpleNamespace(world_size=1))
    with pytest.raises(ValueError, match="checkpoint.enabled=true"):
        getattr(recipe, entrypoint)()
    assert observed == []
    assert not (output / "metrics_test.json").exists()
    assert (output / "LOWEST_VAL").is_symlink() == stale_pointer
