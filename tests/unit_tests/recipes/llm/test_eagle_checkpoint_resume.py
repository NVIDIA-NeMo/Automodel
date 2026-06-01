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

import json
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


def test_eagle1_load_extra_state_accepts_legacy_filename(tmp_path):
    """Old checkpoints (pre-refactor) used ``eagle1_meta.pt``. New code reads either."""
    save_dir = str(tmp_path / "ckpt")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {"global_step": 5, "epoch": 2},
        os.path.join(save_dir, "eagle1_meta.pt"),
    )
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe._load_extra_state(save_dir)
    assert recipe.runtime.global_step == 5
    assert recipe._resume_epoch == 2


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


# ---------------------------------------------------------------------------
# save_checkpoint: train_loss only (no val_loss)
# ---------------------------------------------------------------------------


def test_eagle1_save_checkpoint_train_loss_only(tmp_path):
    """save_checkpoint with train_loss but no val_loss writes losses.json with train_loss only."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.runtime.global_step = 3

    recipe.save_checkpoint(epoch=1, step=3, train_loss=0.42, val_loss=None)

    ckpt_path = os.path.join(recipe.checkpoint_config.checkpoint_dir, "epoch_1_step_3")
    with open(os.path.join(ckpt_path, "losses.json")) as f:
        data = json.load(f)
    assert data == {"train_loss": 0.42}
    assert "val_loss" not in data


# ---------------------------------------------------------------------------
# save_checkpoint: multi-key val_loss with explicit best_metric_key
# ---------------------------------------------------------------------------


def test_eagle1_save_checkpoint_multi_key_val_loss(tmp_path):
    """Multi-key val_loss dict uses best_metric_key to pick the tracked metric."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe._best_val_loss = float("inf")
    recipe.runtime.global_step = 5

    recipe.save_checkpoint(
        epoch=1,
        step=5,
        val_loss={"val_loss": 0.6, "val_accuracy": 0.8, "val_perplexity": 1.5},
        best_metric_key="val_loss",
    )

    ckpt_path = os.path.join(recipe.checkpoint_config.checkpoint_dir, "epoch_1_step_5")
    with open(os.path.join(ckpt_path, "losses.json")) as f:
        data = json.load(f)
    assert "val_loss" in data
    assert "val_accuracy" in data
    assert "val_perplexity" in data


# ---------------------------------------------------------------------------
# save_checkpoint: single-key val_loss auto-selects the only key
# ---------------------------------------------------------------------------


def test_eagle1_save_checkpoint_single_key_val_loss(tmp_path):
    """Single-key val_loss dict auto-selects the only key for best_val_metric."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe._best_val_loss = float("inf")
    recipe.runtime.global_step = 2

    recipe.save_checkpoint(epoch=1, step=2, val_loss={"val_loss": 0.3})

    ckpt_path = os.path.join(recipe.checkpoint_config.checkpoint_dir, "epoch_1_step_2")
    assert os.path.isfile(os.path.join(ckpt_path, "losses.json"))


# ---------------------------------------------------------------------------
# save_checkpoint: async mode stores pending instead of immediate symlink
# ---------------------------------------------------------------------------


def test_eagle1_save_checkpoint_async_defers_symlink(tmp_path):
    """When is_async=True, save_checkpoint stores pending dir instead of creating LATEST symlink."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.runtime.global_step = 10
    recipe.checkpointer.config.is_async = True

    recipe.save_checkpoint(epoch=2, step=10, train_loss=0.5)

    assert recipe._last_pending_checkpoint_dir is not None
    assert "epoch_2_step_10" in recipe._last_pending_checkpoint_dir
    latest = os.path.join(recipe.checkpoint_config.checkpoint_dir, "LATEST")
    assert not os.path.islink(latest) and not os.path.isfile(latest + ".txt")


# ---------------------------------------------------------------------------
# save_checkpoint: prev_pending is flushed on next save
# ---------------------------------------------------------------------------


def test_eagle1_save_checkpoint_flushes_prev_pending(tmp_path):
    """When a prev_pending checkpoint dir exists, the next save creates its LATEST symlink."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.runtime.global_step = 5

    prev_path = os.path.join(recipe.checkpoint_config.checkpoint_dir, "epoch_1_step_5")
    os.makedirs(prev_path, exist_ok=True)
    recipe._last_pending_checkpoint_dir = prev_path

    recipe.save_checkpoint(epoch=2, step=10, train_loss=0.3)

    assert recipe._last_pending_checkpoint_dir is None
    latest = os.path.join(recipe.checkpoint_config.checkpoint_dir, "LATEST")
    assert os.path.islink(latest) or os.path.isfile(latest + ".txt")


# ---------------------------------------------------------------------------
# save_checkpoint: prev_best_pending is flushed on next save
# ---------------------------------------------------------------------------


def test_eagle1_save_checkpoint_flushes_prev_best_pending(tmp_path):
    """When a prev_best_pending exists, the next save flushes the best symlink."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe._best_val_loss = float("inf")
    recipe.runtime.global_step = 5

    prev_path = os.path.join(recipe.checkpoint_config.checkpoint_dir, "epoch_1_step_5")
    os.makedirs(prev_path, exist_ok=True)
    recipe._last_pending_best_checkpoint_info = {"path": prev_path, "val": 0.3}

    recipe.save_checkpoint(epoch=2, step=10, train_loss=0.3)

    assert recipe._last_pending_best_checkpoint_info is None


# ---------------------------------------------------------------------------
# save_checkpoint: async with val_loss stores both pending and best pending
# ---------------------------------------------------------------------------


def test_eagle1_save_checkpoint_async_stores_best_pending(tmp_path):
    """When is_async=True and val_loss given, both pending checkpoint and best pending are stored."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.runtime.global_step = 10
    recipe.checkpointer.config.is_async = True

    recipe.save_checkpoint(epoch=2, step=10, val_loss={"val_loss": 0.25})

    assert recipe._last_pending_checkpoint_dir is not None
    assert recipe._last_pending_best_checkpoint_info is not None
    assert recipe._last_pending_best_checkpoint_info["val"] == 0.25


# ---------------------------------------------------------------------------
# save_checkpoint: FileExistsError when checkpoint dir already exists
# ---------------------------------------------------------------------------


def test_eagle1_save_checkpoint_raises_on_existing_dir(tmp_path):
    """save_checkpoint raises FileExistsError if the target checkpoint dir already exists."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.runtime.global_step = 5
    ckpt_path = os.path.join(recipe.checkpoint_config.checkpoint_dir, "epoch_1_step_5")
    os.makedirs(ckpt_path, exist_ok=True)

    with pytest.raises(FileExistsError):
        recipe.save_checkpoint(epoch=1, step=5)


# ---------------------------------------------------------------------------
# save_checkpoint: config snapshot failure is non-fatal
# ---------------------------------------------------------------------------


def test_eagle1_save_checkpoint_config_snapshot_failure_nonfatal(tmp_path):
    """If save_config raises, save_checkpoint logs a warning but does not crash."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.runtime.global_step = 5
    del recipe.cfg.raw_config

    recipe.save_checkpoint(epoch=1, step=5, train_loss=0.5)

    ckpt_path = os.path.join(recipe.checkpoint_config.checkpoint_dir, "epoch_1_step_5")
    assert os.path.isdir(ckpt_path)


# ---------------------------------------------------------------------------
# load_checkpoint: incompatible with explicit restore_from (warns but proceeds)
# ---------------------------------------------------------------------------


def test_eagle1_load_checkpoint_incompatible_explicit_restore_proceeds(tmp_path):
    """Explicit restore_from with incompatible config warns but still loads the checkpoint."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.cfg.raw_config = {"model": {"pretrained_model_name_or_path": "meta-llama/Llama-3.2-1B"}}
    ckpt_dir = recipe.checkpoint_config.checkpoint_dir
    target = os.path.join(ckpt_dir, "epoch_2_step_10")
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "config.yaml"), "w") as f:
        f.write("model:\n  pretrained_model_name_or_path: other/model\n")
    torch.save({"global_step": 10, "epoch": 2}, os.path.join(target, "eagle_meta.pt"))

    recipe.load_checkpoint(target)

    assert recipe.runtime.global_step == 10
    assert recipe._resume_epoch == 2
    recipe.checkpointer.load_model.assert_called_once()
    recipe.checkpointer.load_optimizer.assert_called_once()


# ---------------------------------------------------------------------------
# load_checkpoint: RNG FileNotFoundError is handled gracefully
# ---------------------------------------------------------------------------


def test_eagle1_load_checkpoint_rng_missing_is_nonfatal(tmp_path):
    """Missing RNG state file should warn, not crash."""
    recipe = _bare_eagle1_recipe(tmp_path)
    ckpt_dir = recipe.checkpoint_config.checkpoint_dir
    target = os.path.join(ckpt_dir, "epoch_1_step_5")
    os.makedirs(target, exist_ok=True)
    torch.save({"global_step": 5, "epoch": 1}, os.path.join(target, "eagle_meta.pt"))
    recipe.checkpointer.load_on_dp_ranks.side_effect = FileNotFoundError("no rng")

    recipe.load_checkpoint(None)

    assert recipe.runtime.global_step == 5
    recipe.checkpointer.load_model.assert_called_once()


# ---------------------------------------------------------------------------
# load_checkpoint: disabled checkpointer is a no-op
# ---------------------------------------------------------------------------


def test_eagle1_load_checkpoint_disabled_is_noop(tmp_path):
    """load_checkpoint returns immediately when checkpoint is disabled."""
    recipe = _bare_eagle1_recipe(tmp_path)
    recipe.checkpointer.config.enabled = False

    recipe.load_checkpoint("some/path")

    assert recipe.runtime.global_step == 0
    recipe.checkpointer.load_model.assert_not_called()


# ---------------------------------------------------------------------------
# Eagle-3 specific: _load_extra_state without vocab mapping (old checkpoint)
# ---------------------------------------------------------------------------


def test_eagle3_load_extra_state_without_vocab_mapping(tmp_path):
    """Old checkpoints without vocab mapping tensors should still restore step/epoch."""
    save_dir = str(tmp_path / "ckpt")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {"global_step": 15, "epoch": 3},
        os.path.join(save_dir, "eagle_meta.pt"),
    )
    recipe = _bare_eagle3_recipe(tmp_path)
    original_ids = recipe._module().selected_token_ids.clone()

    recipe._load_extra_state(save_dir)

    assert recipe.runtime.global_step == 15
    assert recipe._resume_epoch == 3
    assert torch.equal(recipe._module().selected_token_ids, original_ids)


# ---------------------------------------------------------------------------
# Eagle-3 specific: _load_extra_state with missing meta file is no-op
# ---------------------------------------------------------------------------


def test_eagle3_load_extra_state_missing_meta_is_noop(tmp_path):
    """If neither eagle_meta.pt nor eagle3_meta.pt exists, _load_extra_state is a no-op."""
    save_dir = str(tmp_path / "empty_ckpt")
    os.makedirs(save_dir, exist_ok=True)
    recipe = _bare_eagle3_recipe(tmp_path)

    recipe._load_extra_state(save_dir)

    assert recipe.runtime.global_step == 0
    assert recipe._resume_epoch == 0


# ---------------------------------------------------------------------------
# Eagle-3 specific: save_checkpoint writes all expected artifacts
# ---------------------------------------------------------------------------


def test_eagle3_save_checkpoint_writes_expected_artifacts(tmp_path):
    """Eagle-3 save_checkpoint writes eagle_meta.pt with vocab mapping tensors."""
    recipe = _bare_eagle3_recipe(tmp_path)
    recipe.runtime.global_step = 7
    recipe.tokenizer = None

    recipe.save_checkpoint(
        epoch=1,
        step=7,
        train_loss=0.5,
        val_loss={"val_loss": 0.4, "val_accuracy": 0.6},
    )

    ckpt_path = os.path.join(recipe.checkpoint_config.checkpoint_dir, "epoch_1_step_7")
    assert os.path.isfile(os.path.join(ckpt_path, "eagle_meta.pt"))
    meta = torch.load(os.path.join(ckpt_path, "eagle_meta.pt"), weights_only=False)
    assert "selected_token_ids" in meta
    assert "selected_token_mask" in meta
    assert meta["global_step"] == 7
    assert meta["epoch"] == 1


# ---------------------------------------------------------------------------
# Eagle-3 specific: train loop skips completed epochs
# ---------------------------------------------------------------------------


def test_eagle3_train_loop_skips_completed_epochs(tmp_path):
    """Eagle-3 train loop returns early when all epochs are already completed."""
    recipe = _bare_eagle3_recipe(tmp_path)
    recipe._resume_epoch = 3
    recipe.num_epochs = 3
    recipe.train_dataloader = MagicMock()
    recipe.val_dataloader = None
    recipe.target_wrapper = MagicMock()
    recipe.device = torch.device("cpu")
    recipe.grad_accumulation_steps = 1
    recipe.max_grad_norm = 1.0
    recipe.log_every_steps = 1
    recipe.total_optim_steps = 10
    recipe.warmup_steps = 1
    recipe.peak_lr = 1e-4
    recipe.min_lr_ratio = 0.1

    recipe.run_train_validation_loop()
    recipe.train_dataloader.__iter__.assert_not_called()


# ---------------------------------------------------------------------------
# Eagle-3 specific: load_checkpoint with incompatible explicit proceeds
# ---------------------------------------------------------------------------


def test_eagle3_load_checkpoint_incompatible_explicit_proceeds(tmp_path):
    """Eagle-3 load_checkpoint with incompatible explicit restore_from warns but proceeds."""
    recipe = _bare_eagle3_recipe(tmp_path)
    recipe.cfg.raw_config = {"model": {"pretrained_model_name_or_path": "meta-llama/Llama-3.2-1B"}}
    ckpt_dir = recipe.checkpoint_config.checkpoint_dir
    target = os.path.join(ckpt_dir, "epoch_1_step_5")
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "config.yaml"), "w") as f:
        f.write("model:\n  pretrained_model_name_or_path: other/model\n")
    torch.save(
        {
            "global_step": 5,
            "epoch": 1,
            "selected_token_ids": torch.arange(8, dtype=torch.long),
            "selected_token_mask": torch.ones(8, dtype=torch.bool),
        },
        os.path.join(target, "eagle_meta.pt"),
    )

    recipe.load_checkpoint(target)

    assert recipe.runtime.global_step == 5
    assert recipe._resume_epoch == 1
    recipe.checkpointer.load_model.assert_called_once()


# ---------------------------------------------------------------------------
# Eagle-3 specific: load_checkpoint RNG missing is non-fatal
# ---------------------------------------------------------------------------


def test_eagle3_load_checkpoint_rng_missing_is_nonfatal(tmp_path):
    """Eagle-3: missing RNG state file warns but doesn't crash."""
    recipe = _bare_eagle3_recipe(tmp_path)
    ckpt_dir = recipe.checkpoint_config.checkpoint_dir
    target = os.path.join(ckpt_dir, "epoch_1_step_5")
    os.makedirs(target, exist_ok=True)
    torch.save(
        {
            "global_step": 5,
            "epoch": 1,
            "selected_token_ids": torch.arange(8, dtype=torch.long),
            "selected_token_mask": torch.ones(8, dtype=torch.bool),
        },
        os.path.join(target, "eagle_meta.pt"),
    )
    recipe.checkpointer.load_on_dp_ranks.side_effect = FileNotFoundError("no rng")

    recipe.load_checkpoint(None)

    assert recipe.runtime.global_step == 5
    recipe.checkpointer.load_model.assert_called_once()


# ---------------------------------------------------------------------------
# Eagle-3 specific: LATEST with no checkpoints is noop
# ---------------------------------------------------------------------------


def test_eagle3_load_checkpoint_latest_no_checkpoints_is_noop(tmp_path):
    """Eagle-3: restore_from='LATEST' with empty checkpoint dir starts fresh."""
    recipe = _bare_eagle3_recipe(tmp_path)
    recipe.load_checkpoint("LATEST")
    assert recipe.runtime.global_step == 0
    assert recipe._resume_epoch == 0


# ---------------------------------------------------------------------------
# Eagle-3 specific: save_checkpoint disabled is no-op
# ---------------------------------------------------------------------------


def test_eagle3_save_checkpoint_disabled_is_noop(tmp_path):
    """Eagle-3: checkpoint.enabled=False is a true no-op."""
    recipe = _bare_eagle3_recipe(tmp_path)
    recipe.checkpointer.config = _StubCheckpointConfig(
        enabled=False, checkpoint_dir=recipe.checkpoint_config.checkpoint_dir
    )
    recipe.checkpoint_config = recipe.checkpointer.config

    recipe.save_checkpoint(epoch=1, step=5, train_loss=0.5)
    recipe.checkpointer.save_model.assert_not_called()


# ---------------------------------------------------------------------------
# Eagle-3 specific: auto-detect latest checkpoint
# ---------------------------------------------------------------------------


def test_eagle3_load_checkpoint_auto_detects_latest(tmp_path):
    """Eagle-3: auto-detect the most recent checkpoint when restore_from is None."""
    recipe = _bare_eagle3_recipe(tmp_path)
    ckpt_dir = recipe.checkpoint_config.checkpoint_dir
    target = os.path.join(ckpt_dir, "epoch_2_step_10")
    os.makedirs(target, exist_ok=True)
    torch.save(
        {
            "global_step": 10,
            "epoch": 2,
            "selected_token_ids": torch.arange(8, dtype=torch.long),
            "selected_token_mask": torch.ones(8, dtype=torch.bool),
        },
        os.path.join(target, "eagle_meta.pt"),
    )

    recipe.load_checkpoint(None)
    assert recipe.runtime.global_step == 10
    assert recipe._resume_epoch == 2
    recipe.checkpointer.load_model.assert_called_once()


# ---------------------------------------------------------------------------
# Eagle-3 specific: skips incompatible auto-detected checkpoint
# ---------------------------------------------------------------------------


def test_eagle3_load_checkpoint_skips_incompatible_auto_detected(tmp_path):
    """Eagle-3: auto-detected incompatible checkpoint is skipped."""
    recipe = _bare_eagle3_recipe(tmp_path)
    recipe.cfg.raw_config = {"model": {"pretrained_model_name_or_path": "meta-llama/Llama-3.2-1B"}}
    ckpt_dir = recipe.checkpoint_config.checkpoint_dir
    target = os.path.join(ckpt_dir, "epoch_2_step_10")
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "config.yaml"), "w") as f:
        f.write("model:\n  pretrained_model_name_or_path: other/model\n")

    recipe.load_checkpoint(None)
    assert recipe.runtime.global_step == 0
    recipe.checkpointer.load_model.assert_not_called()


# ---------------------------------------------------------------------------
# Eagle-3 specific: missing dir raises with explicit restore_from
# ---------------------------------------------------------------------------


def test_eagle3_load_checkpoint_missing_dir_raises(tmp_path):
    """Eagle-3: explicit restore_from to a non-existent dir raises FileNotFoundError."""
    recipe = _bare_eagle3_recipe(tmp_path)
    with pytest.raises(FileNotFoundError):
        recipe.load_checkpoint("/does/not/exist/epoch_1_step_5")
