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

"""Tests for EAGLE recipe checkpoint resume.

Covers the recipe-level orchestration that ``BaseRecipe.save_checkpoint`` was
not handling correctly for EAGLE (multiple ``nn.Module`` attributes, frozen
target, ``LambdaLR`` not recognized by ``is_lr_scheduler``):

- ``_save_extra_state`` / ``_load_extra_state`` round-trip global_step, epoch,
  and EAGLE-3 vocab mapping tensors.
- ``load_checkpoint`` resolves ``"LATEST"``, named subdirs, and missing paths.
- The train loop honours ``_resume_epoch`` and skips already-completed epochs.

The DCP-backed ``Checkpointer.save_model`` / ``save_optimizer`` paths are
mocked because their numerical round-trip is covered upstream in the
checkpointer tests; this file only validates the EAGLE-specific wiring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.recipes.llm.train_eagle1 import TrainEagle1Recipe
from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe


@dataclass
class _StubCheckpointConfig:
    enabled: bool
    checkpoint_dir: str


def _build_stub_checkpointer(tmp_path) -> MagicMock:
    """Return a Checkpointer mock whose save/load methods are no-ops but whose
    config exposes the same attributes the recipe relies on."""
    ckpt_dir = str(tmp_path / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    mock = MagicMock()
    mock.config = _StubCheckpointConfig(enabled=True, checkpoint_dir=ckpt_dir)
    return mock


class _FakeDraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)


class _FakeEagle1TrainerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.draft_model = _FakeDraftModel()


class _FakeEagle3TrainerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.draft_model = _FakeDraftModel()
        self.register_buffer("selected_token_ids", torch.arange(8, dtype=torch.long))
        self.register_buffer("selected_token_mask", torch.ones(8, dtype=torch.bool))


def _bare_eagle1_recipe(tmp_path) -> TrainEagle1Recipe:
    recipe = TrainEagle1Recipe.__new__(TrainEagle1Recipe)
    recipe.cfg = SimpleNamespace(get=lambda *_args, **_kw: None, raw_config={})
    recipe.tokenizer = None
    recipe.dist_env = SimpleNamespace(is_main=True, world_size=1)
    recipe.trainer_module = _FakeEagle1TrainerModule()
    recipe.optimizer = torch.optim.AdamW(recipe.trainer_module.parameters(), lr=1e-4)
    recipe.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(recipe.optimizer, lambda s: 1.0)
    recipe.runtime = SimpleNamespace(global_step=0)
    recipe._resume_epoch = 0
    recipe.rng = MagicMock()
    recipe.rng.state_dict = MagicMock(return_value={"seed": 42})
    recipe.rng.load_state_dict = MagicMock()
    recipe.checkpointer = _build_stub_checkpointer(tmp_path)
    recipe.checkpoint_config = recipe.checkpointer.config
    return recipe


def _bare_eagle3_recipe(tmp_path) -> TrainEagle3Recipe:
    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe.cfg = SimpleNamespace(get=lambda *_args, **_kw: None, raw_config={})
    recipe.tokenizer = None
    recipe.dist_env = SimpleNamespace(is_main=True, world_size=1)
    recipe.trainer_module = _FakeEagle3TrainerModule()
    recipe.optimizer = torch.optim.AdamW(recipe.trainer_module.parameters(), lr=1e-4)
    recipe.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(recipe.optimizer, lambda s: 1.0)
    recipe.runtime = SimpleNamespace(global_step=0)
    recipe._resume_epoch = 0
    recipe.rng = MagicMock()
    recipe.rng.state_dict = MagicMock(return_value={"seed": 42})
    recipe.rng.load_state_dict = MagicMock()
    recipe.checkpointer = _build_stub_checkpointer(tmp_path)
    recipe.checkpoint_config = recipe.checkpointer.config
    return recipe


def test_eagle1_extra_state_roundtrip(tmp_path):
    """global_step + epoch survive save -> load on the EAGLE-1 recipe."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.runtime.global_step = 42
    save_dir = str(tmp_path / "ckpt")
    os.makedirs(save_dir, exist_ok=True)
    recipe._save_extra_state(save_dir, epoch=3)

    fresh = _bare_eagle1_recipe(tmp_path)
    assert fresh.runtime.global_step == 0
    assert fresh._resume_epoch == 0
    fresh._load_extra_state(save_dir)
    assert fresh.runtime.global_step == 42
    assert fresh._resume_epoch == 3


def test_eagle3_extra_state_roundtrip_includes_vocab_mapping(tmp_path):
    """selected_token_ids and selected_token_mask must round-trip on EAGLE-3."""
    recipe = _bare_eagle3_recipe(tmp_path)
    recipe.runtime.global_step = 17
    custom_ids = torch.tensor([5, 9, 11, 13, 17, 19, 23, 29], dtype=torch.long)
    custom_mask = torch.tensor([True, False, True, True, False, True, True, True])
    recipe._module().selected_token_ids.copy_(custom_ids)
    recipe._module().selected_token_mask.copy_(custom_mask)
    save_dir = str(tmp_path / "ckpt")
    os.makedirs(save_dir, exist_ok=True)
    recipe._save_extra_state(save_dir, epoch=2)

    fresh = _bare_eagle3_recipe(tmp_path)
    fresh._load_extra_state(save_dir)
    assert fresh.runtime.global_step == 17
    assert fresh._resume_epoch == 2
    assert torch.equal(fresh._module().selected_token_ids, custom_ids)
    assert torch.equal(fresh._module().selected_token_mask, custom_mask)


def test_eagle3_load_extra_state_accepts_legacy_filename(tmp_path):
    """Old checkpoints (pre-refactor) used ``eagle3_meta.pt``. New code reads either."""
    save_dir = str(tmp_path / "ckpt")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "global_step": 7,
            "epoch": 1,
            "selected_token_ids": torch.arange(8, dtype=torch.long),
            "selected_token_mask": torch.ones(8, dtype=torch.bool),
        },
        os.path.join(save_dir, "eagle3_meta.pt"),
    )
    recipe = _bare_eagle3_recipe(tmp_path)
    recipe._load_extra_state(save_dir)
    assert recipe.runtime.global_step == 7
    assert recipe._resume_epoch == 1


def test_eagle1_load_checkpoint_missing_dir_raises(tmp_path):
    """Explicit restore_from to a non-existent dir must raise, not silently start fresh."""
    recipe = _bare_eagle1_recipe(tmp_path)
    with pytest.raises(FileNotFoundError):
        recipe.load_checkpoint("/does/not/exist/epoch_1_step_5")


def test_eagle1_load_checkpoint_latest_with_no_checkpoints_is_noop(tmp_path):
    """``restore_from='LATEST'`` with an empty checkpoint dir starts fresh without raising."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.load_checkpoint("LATEST")
    assert recipe.runtime.global_step == 0
    assert recipe._resume_epoch == 0


def test_eagle1_load_checkpoint_auto_detects_latest(tmp_path):
    """When restore_from is None, the recipe auto-detects the most recent ``*_step_*`` dir."""
    recipe = _bare_eagle1_recipe(tmp_path)
    ckpt_dir = recipe.checkpoint_config.checkpoint_dir
    target = os.path.join(ckpt_dir, "epoch_2_step_10")
    os.makedirs(target, exist_ok=True)
    torch.save({"global_step": 10, "epoch": 2}, os.path.join(target, "eagle_meta.pt"))

    recipe.load_checkpoint(None)
    assert recipe.runtime.global_step == 10
    assert recipe._resume_epoch == 2
    recipe.checkpointer.load_model.assert_called_once()
    recipe.checkpointer.load_optimizer.assert_called_once()


def test_eagle1_load_checkpoint_skips_incompatible_auto_detected_checkpoint(tmp_path):
    """Auto-detected checkpoints should be skipped when config.yaml does not match the current run."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.cfg.raw_config = {"model": {"pretrained_model_name_or_path": "meta-llama/Llama-3.2-1B"}}
    ckpt_dir = recipe.checkpoint_config.checkpoint_dir
    target = os.path.join(ckpt_dir, "epoch_2_step_10")
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "config.yaml"), "w") as f:
        f.write("model:\n  pretrained_model_name_or_path: other/model\n")

    recipe.load_checkpoint(None)
    assert recipe.runtime.global_step == 0
    assert recipe._resume_epoch == 0
    recipe.checkpointer.load_model.assert_not_called()
    recipe.checkpointer.load_optimizer.assert_not_called()


def test_eagle1_save_checkpoint_skipped_when_disabled(tmp_path):
    """``checkpoint.enabled=False`` must be a true no-op (no dir created, no checkpointer call)."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.checkpointer.config = _StubCheckpointConfig(
        enabled=False, checkpoint_dir=recipe.checkpoint_config.checkpoint_dir
    )
    recipe.checkpoint_config = recipe.checkpointer.config

    recipe.save_checkpoint(epoch=1, step=5, train_loss=0.5)
    recipe.checkpointer.save_model.assert_not_called()
    recipe.checkpointer.save_optimizer.assert_not_called()


def test_eagle1_save_checkpoint_writes_expected_artifacts(tmp_path):
    """save_checkpoint must write losses.json + eagle_meta.pt and forward calls to the checkpointer."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.runtime.global_step = 5

    recipe.save_checkpoint(
        epoch=1,
        step=5,
        train_loss=0.7,
        val_loss={"val_loss": 0.6, "val_accuracy": 0.42},
    )

    ckpt_path = os.path.join(recipe.checkpoint_config.checkpoint_dir, "epoch_1_step_5")
    assert os.path.isfile(os.path.join(ckpt_path, "losses.json"))
    assert os.path.isfile(os.path.join(ckpt_path, "eagle_meta.pt"))
    latest = os.path.join(recipe.checkpoint_config.checkpoint_dir, "LATEST")
    assert os.path.islink(latest) or os.path.isfile(latest + ".txt")

    recipe.checkpointer.save_model.assert_called_once()
    recipe.checkpointer.save_optimizer.assert_called_once()
    recipe.checkpointer.save_on_dp_ranks.assert_called_once()

    meta = torch.load(os.path.join(ckpt_path, "eagle_meta.pt"), weights_only=False)
    assert meta["global_step"] == 5
    assert meta["epoch"] == 1


def test_eagle1_train_loop_skips_completed_epochs(tmp_path):
    """If ``_resume_epoch >= num_epochs`` the train loop returns early without touching the optimizer."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe._resume_epoch = 3
    recipe.num_epochs = 3
    recipe.train_dataloader = MagicMock()  # would explode if iterated
    recipe.val_dataloader = None
    recipe.target_wrapper = MagicMock()

    recipe.run_train_validation_loop()
    recipe.train_dataloader.__iter__.assert_not_called()
