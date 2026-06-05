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

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.retrieval import train_bi_encoder as recipe_module
from nemo_automodel.recipes.retrieval.train_bi_encoder import TrainBiEncoderRecipe


class DummyCheckpointConfig:
    def build(self, **kwargs):
        return SimpleNamespace(config=SimpleNamespace(checkpoint_dir="/tmp/nemo-automodel-test-checkpoints"))


class DummyDistEnv:
    world_size = 1
    is_main = False
    device = torch.device("cpu")


class DummyDistSetup:
    strategy_config = SimpleNamespace(defer_fsdp_grad_sync=False)
    device_mesh = None
    moe_mesh = None
    pp_enabled = False
    pipeline_config = None


class DummyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class DummyStepSchedulerConfig:
    def build(self, dataloader, dp_group_size, local_batch_size):
        return SimpleNamespace(dataloader=dataloader)


class DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"


def test_setup_uses_raw_nested_dataloader_config(monkeypatch):
    cfg = ConfigNode(
        {
            "seed": 123,
            "dist_env": {},
            "model": {
                "_target_": "tests.unit_tests.recipes.test_train_bi_encoder.DummyModel",
                "pretrained_model_name_or_path": "dummy",
            },
            "optimizer": {"_target_": "torch.optim.AdamW", "lr": 0.001},
            "tokenizer": {"_target_": "tests.unit_tests.recipes.test_train_bi_encoder.DummyTokenizer"},
            "dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "dataset": {"_target_": "torch.utils.data.TensorDataset"},
            },
            "validation_dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "dataset": {"_target_": "torch.utils.data.TensorDataset"},
            },
            "step_scheduler": {"local_batch_size": 1},
        }
    )

    captured_dataloader_cfgs = []

    def fake_build_dataloader(cfg_dl, tokenizer, **kwargs):
        captured_dataloader_cfgs.append(cfg_dl)
        return [len(captured_dataloader_cfgs)]

    monkeypatch.setattr(recipe_module.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(recipe_module, "build_distributed", lambda cfg: DummyDistEnv())
    monkeypatch.setattr(recipe_module, "setup_logging", lambda: None)
    monkeypatch.setattr(recipe_module, "apply_cache_compatibility_patches", lambda: None)
    monkeypatch.setattr(recipe_module, "apply_te_patches", lambda: None)
    monkeypatch.setattr(recipe_module, "StatefulRNG", lambda *args, **kwargs: object())
    monkeypatch.setattr(recipe_module, "ScopedRNG", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(recipe_module, "setup_distributed", lambda cfg, world_size: DummyDistSetup())
    monkeypatch.setattr(recipe_module, "warn_if_torch_adam_with_bf16_params", lambda **kwargs: None)
    monkeypatch.setattr(recipe_module, "build_dataloader", fake_build_dataloader)
    monkeypatch.setattr(recipe_module, "build_metric_logger", lambda path: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(RecipeConfig, "wandb", property(lambda self: None))
    monkeypatch.setattr(RecipeConfig, "checkpoint", property(lambda self: DummyCheckpointConfig()))
    monkeypatch.setattr(RecipeConfig, "step_scheduler", property(lambda self: DummyStepSchedulerConfig()))
    monkeypatch.setattr(RecipeConfig, "lr_scheduler", property(lambda self: None))
    monkeypatch.setattr(TrainBiEncoderRecipe, "_get_dp_rank", lambda self, include_cp=False: 0)
    monkeypatch.setattr(TrainBiEncoderRecipe, "_get_dp_group_size", lambda self: 1)
    monkeypatch.setattr(TrainBiEncoderRecipe, "_get_tp_rank", lambda self: 0)
    monkeypatch.setattr(TrainBiEncoderRecipe, "_get_pp_rank", lambda self: 0)
    monkeypatch.setattr(TrainBiEncoderRecipe, "_log_experiment_details", lambda self: None)
    monkeypatch.setattr(TrainBiEncoderRecipe, "_log_library_versions", lambda self: None)
    monkeypatch.setattr(TrainBiEncoderRecipe, "_setup_garbage_collection", lambda self, step_scheduler: None)
    monkeypatch.setattr(TrainBiEncoderRecipe, "_log_model_and_optimizer_details", lambda self, *args: None)
    monkeypatch.setattr(TrainBiEncoderRecipe, "load_checkpoint", lambda self, restore_from: None)
    monkeypatch.setattr(TrainBiEncoderRecipe, "_log_step_scheduler_details", lambda self, step_scheduler: None)

    recipe = TrainBiEncoderRecipe(cfg)
    recipe.setup()

    assert recipe.cfg.dataloader is None
    assert captured_dataloader_cfgs == [recipe.cfg.get("dataloader"), recipe.cfg.get("validation_dataloader")]
