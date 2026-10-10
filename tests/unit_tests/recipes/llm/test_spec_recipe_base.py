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

"""Unit tests for the shared speculative-decoding recipe base."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.recipes.llm import _spec_recipe_base
from nemo_automodel.recipes.llm._spec_recipe_base import SpecDecodeRecipeBase


class _Recipe(SpecDecodeRecipeBase):
    def _save_extra_state(self, path, epoch):
        self.events.append(("extra_save", path, epoch))

    def _load_extra_state(self, ckpt_dir):
        self.events.append(("extra_load", ckpt_dir))


class _GlobalRngRecipe(_Recipe):
    rng_checkpoint_on_dp_ranks = False


class _Lifecycle:
    def __init__(self, events):
        self.events = events

    def complete_pending(self):
        self.events.append("complete_pending")

    def reserve(self, path):
        self.events.append(("reserve", path))

    def run_coordinator_step(self, fn, description):
        fn()

    def publish(self, path, best_val_metric, metric_key):
        self.events.append(("publish", path, best_val_metric, metric_key))

    def defer_publication(self, path, best_val_metric, metric_key):
        self.events.append(("defer", path, best_val_metric, metric_key))


class _Cfg(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _make_recipe(cls, tmp_path, *, is_async=False, cp_mesh=None):
    events = []
    obj = cls.__new__(cls)
    obj.events = events
    obj.checkpoint_config = SimpleNamespace(checkpoint_dir=str(tmp_path), allow_legacy_pickle_restore=False)
    obj.checkpointer = SimpleNamespace(
        config=SimpleNamespace(enabled=True, is_async=is_async),
        lifecycle=_Lifecycle(events),
        async_wait=lambda: events.append("wait"),
        save_model=lambda model, path, **kw: events.append(("save_model", kw)),
        save_optimizer=lambda *a: events.append("save_optimizer"),
        save_on_dp_ranks=lambda *a: events.append("rng_dp"),
        save_on_global_ranks=lambda *a: events.append("rng_global"),
        load_model=lambda *a: events.append("load_model"),
        load_optimizer=lambda *a: events.append("load_optimizer"),
        load_on_dp_ranks=lambda *a: events.append("load_rng_dp"),
        load_on_global_ranks=lambda *a: events.append("load_rng_global"),
    )
    obj.trainer_module = SimpleNamespace(draft_model="draft")
    obj.tokenizer = None
    obj.optimizer = object()
    obj.lr_scheduler = object()
    obj.rng = object()
    obj.cp_mesh = cp_mesh
    obj.cfg = SimpleNamespace(raw_config={})
    obj.dist_env = SimpleNamespace(is_main=True)
    return obj, events


# --------------------------------------------------------------------------- #
# _setup_training_state
# --------------------------------------------------------------------------- #
def _optim_recipe(tmp_path, dataloader):
    obj = _Recipe.__new__(_Recipe)
    obj.cfg = SimpleNamespace(optimizer=_Cfg(lr=1e-3, warmup_ratio=0.25, min_lr_ratio=0.2))
    obj.trainer_module = torch.nn.Linear(2, 2)
    obj.train_dataloader = dataloader
    recipe_cfg = _Cfg(num_epochs=2, grad_accumulation_steps=3, output_dir=str(tmp_path / "out"))
    return obj, recipe_cfg


def test_setup_training_state_builds_adamw_schedule_and_cadence(tmp_path):
    obj, recipe_cfg = _optim_recipe(tmp_path, dataloader=list(range(10)))

    obj._setup_training_state(recipe_cfg)
    assert obj.num_batches_per_epoch == 10

    assert isinstance(obj.optimizer, torch.optim.AdamW)
    assert obj.optimizer.param_groups[0]["betas"] == (0.9, 0.95)
    # ceil(10 / 3) = 4 optimizer steps per epoch, two epochs.
    assert obj.total_optim_steps == 8
    assert obj.warmup_steps == 2
    assert obj.min_lr_ratio == pytest.approx(0.2)
    assert isinstance(obj.lr_scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert obj.max_grad_norm == 1.0
    assert obj.log_every_steps == 10
    assert obj.ckpt_every_steps is None
    assert obj.save_checkpoint_every_epoch is False
    assert obj.output_dir.is_dir()
    assert obj.runtime.global_step == 0
    assert obj._resume_epoch == 0


def test_setup_training_state_handles_unsized_dataloader(tmp_path):
    obj, recipe_cfg = _optim_recipe(tmp_path, dataloader=iter(range(10)))

    obj._setup_training_state(recipe_cfg)
    assert obj.num_batches_per_epoch == 0
    assert obj.total_optim_steps == 1
    assert obj.warmup_steps == 1


def test_min_warmup_steps_floor_is_used(tmp_path):
    class _FlooredWarmup(_Recipe):
        min_warmup_steps = 5

    obj, recipe_cfg = _optim_recipe(tmp_path, dataloader=list(range(10)))
    obj.__class__ = _FlooredWarmup
    obj._setup_training_state(recipe_cfg)
    assert obj.warmup_steps == 5


# --------------------------------------------------------------------------- #
# _build_checkpointer / _module
# --------------------------------------------------------------------------- #
def test_build_checkpointer_marks_peft_drafts(tmp_path, monkeypatch):
    built = {}
    monkeypatch.setattr(
        "nemo_automodel.components.checkpoint.checkpointing.Checkpointer",
        lambda **kw: built.update(kw) or SimpleNamespace(**kw),
    )
    obj = _Recipe.__new__(_Recipe)
    obj.cfg = _Cfg()
    obj.output_dir = tmp_path
    obj.draft_model = torch.nn.Linear(2, 2)
    obj.peft_config = object()
    obj.dp_mesh = SimpleNamespace(get_local_rank=lambda: 2)

    obj._build_checkpointer("target/path")

    assert built["dp_rank"] == 2
    assert obj.checkpoint_config.is_peft is True
    assert obj.checkpoint_config.save_consolidated.value == "false"


def test_module_unwraps_ddp(monkeypatch):
    class _FakeDDP:
        module = "inner"

    monkeypatch.setattr(_spec_recipe_base, "DistributedDataParallel", _FakeDDP)
    obj = _Recipe.__new__(_Recipe)
    obj.trainer_module = _FakeDDP()
    assert obj._module() == "inner"
    obj.trainer_module = "plain"
    assert obj._module() == "plain"


# --------------------------------------------------------------------------- #
# save_checkpoint
# --------------------------------------------------------------------------- #
def test_save_checkpoint_dp_rng_and_hook_order(tmp_path, monkeypatch):
    obj, events = _make_recipe(_Recipe, tmp_path)
    obj.peft_config = "PEFT"
    hook_calls = []
    obj._on_draft_saved = lambda draft, path, **kw: hook_calls.append((draft, path, kw)) or events.append("hook")
    monkeypatch.setattr(_spec_recipe_base, "save_config", lambda cfg, path: events.append("config"))
    monkeypatch.setattr(_spec_recipe_base, "save_losses", lambda losses, path: events.append(("losses", losses)))

    obj.save_checkpoint(epoch=1, step=4, train_loss=0.5, is_final_checkpoint=True)

    path = str(tmp_path / "epoch_1_step_4")
    assert ("save_model", {"peft_config": "PEFT", "tokenizer": None, "is_final_checkpoint": True}) in events
    # The hook runs right after the draft weights, before the optimizer and RNG.
    hook = events.index("hook")
    assert events[hook - 1][0] == "save_model"
    assert hook < events.index("save_optimizer") < events.index("rng_dp") < events.index(("extra_save", path, 1))
    assert "rng_global" not in events
    assert hook_calls == [("draft", path, {"is_final_checkpoint": True, "is_rank_0": True})]
    assert events[-1] == ("publish", path, None, "default")
    assert ("losses", {"train_loss": 0.5}) in events


def test_save_checkpoint_global_rng_layout(tmp_path, monkeypatch):
    obj, events = _make_recipe(_GlobalRngRecipe, tmp_path, is_async=True)
    monkeypatch.setattr(_spec_recipe_base, "save_config", lambda cfg, path: None)

    monkeypatch.setattr(_spec_recipe_base, "save_losses", lambda losses, path: events.append(("losses", losses)))

    obj.save_checkpoint(epoch=0, step=2, val_loss={"val_loss": 0.25})

    assert "rng_global" in events and "rng_dp" not in events
    assert ("losses", {"val_loss": 0.25}) in events
    assert events[-1] == ("defer", str(tmp_path / "epoch_0_step_2"), 0.25, "val_loss")


def test_save_checkpoint_skips_rng_on_non_leading_cp_peer(tmp_path, monkeypatch):
    obj, events = _make_recipe(_Recipe, tmp_path, cp_mesh=SimpleNamespace(get_local_rank=lambda: 1))
    monkeypatch.setattr(_spec_recipe_base, "save_config", lambda cfg, path: None)

    obj.save_checkpoint(epoch=0, step=1)

    assert "rng_dp" not in events and "rng_global" not in events


def test_save_checkpoint_barriers_under_dist_and_warns_on_config_failure(tmp_path, monkeypatch, caplog):
    obj, events = _make_recipe(_Recipe, tmp_path)
    monkeypatch.setattr(_spec_recipe_base.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(_spec_recipe_base.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(_spec_recipe_base.dist, "barrier", lambda: events.append("barrier"))

    def _fail(cfg, path):
        raise OSError("disk full")

    monkeypatch.setattr(_spec_recipe_base, "save_config", _fail)
    monkeypatch.setattr(_spec_recipe_base, "save_losses", lambda losses, path: events.append("losses"))

    with caplog.at_level(logging.WARNING):
        obj.save_checkpoint(epoch=0, step=1, train_loss=1.0)

    assert events.count("barrier") == 2
    assert "Failed to save config snapshot" in caplog.text
    # Only rank 0 writes the loss summary.
    assert "losses" not in events


def test_save_checkpoint_noop_when_disabled(tmp_path):
    obj, events = _make_recipe(_Recipe, tmp_path)
    obj.checkpointer.config.enabled = False
    obj.save_checkpoint(epoch=0, step=1)
    assert events == []


def test_default_hooks():
    obj = SpecDecodeRecipeBase.__new__(SpecDecodeRecipeBase)
    assert obj._on_draft_saved(None, "p", is_final_checkpoint=True, is_rank_0=True) is None
    with pytest.raises(NotImplementedError):
        obj._save_extra_state("p", 0)
    with pytest.raises(NotImplementedError):
        obj._load_extra_state("p")


# --------------------------------------------------------------------------- #
# load_checkpoint
# --------------------------------------------------------------------------- #
def test_load_checkpoint_restores_dp_rng_and_meta(tmp_path, monkeypatch):
    ckpt = tmp_path / "epoch_0_step_3"
    ckpt.mkdir()
    obj, events = _make_recipe(_Recipe, tmp_path)
    monkeypatch.setattr(_spec_recipe_base, "find_latest_checkpoint", lambda root: ckpt)
    monkeypatch.setattr(_spec_recipe_base, "_is_checkpoint_model_config_compatible", lambda cfg, d: (True, ""))

    obj.load_checkpoint()

    assert events == ["load_model", "load_optimizer", "load_rng_dp", ("extra_load", str(ckpt))]


def test_load_checkpoint_explicit_incompatible_proceeds_with_global_rng(tmp_path, monkeypatch, caplog):
    ckpt = tmp_path / "epoch_0_step_3"
    ckpt.mkdir()
    obj, events = _make_recipe(_GlobalRngRecipe, tmp_path)
    monkeypatch.setattr(_spec_recipe_base, "resolve_restore_from_to_checkpoint_dir", lambda root, r: str(ckpt))
    monkeypatch.setattr(_spec_recipe_base, "_is_checkpoint_model_config_compatible", lambda cfg, d: (False, "x"))

    with caplog.at_level(logging.WARNING):
        obj.load_checkpoint(str(ckpt))

    assert "Proceeding with restore anyway" in caplog.text
    assert "load_rng_global" in events


def test_load_checkpoint_auto_incompatible_skips(tmp_path, monkeypatch, caplog):
    obj, events = _make_recipe(_Recipe, tmp_path)
    monkeypatch.setattr(_spec_recipe_base, "find_latest_checkpoint", lambda root: tmp_path / "ckpt")
    monkeypatch.setattr(_spec_recipe_base, "_is_checkpoint_model_config_compatible", lambda cfg, d: (False, "x"))

    with caplog.at_level(logging.WARNING):
        obj.load_checkpoint()

    assert "Skipping restore" in caplog.text
    assert events == []


def test_load_checkpoint_missing_rng_is_tolerated(tmp_path, monkeypatch, caplog):
    ckpt = tmp_path / "epoch_0_step_3"
    ckpt.mkdir()
    obj, events = _make_recipe(_Recipe, tmp_path)

    def _missing(*a):
        raise FileNotFoundError

    obj.checkpointer.load_on_dp_ranks = _missing
    monkeypatch.setattr(_spec_recipe_base, "find_latest_checkpoint", lambda root: ckpt)
    monkeypatch.setattr(_spec_recipe_base, "_is_checkpoint_model_config_compatible", lambda cfg, d: (True, ""))

    with caplog.at_level(logging.WARNING):
        obj.load_checkpoint()

    assert "RNG state not found" in caplog.text
    assert events[-1] == ("extra_load", str(ckpt))


def test_load_checkpoint_early_returns(tmp_path, monkeypatch):
    obj, events = _make_recipe(_Recipe, tmp_path)
    # LATEST with nothing saved yet.
    monkeypatch.setattr(_spec_recipe_base, "resolve_restore_from_to_checkpoint_dir", lambda root, r: None)
    obj.load_checkpoint("LATEST")
    # No auto-detected checkpoint.
    monkeypatch.setattr(_spec_recipe_base, "find_latest_checkpoint", lambda root: None)
    obj.load_checkpoint()
    # Checkpointing disabled.
    obj.checkpointer.config.enabled = False
    obj.load_checkpoint("anything")
    assert events == []

    obj.checkpointer.config.enabled = True
    monkeypatch.setattr(
        _spec_recipe_base, "resolve_restore_from_to_checkpoint_dir", lambda root, r: str(tmp_path / "missing")
    )
    with pytest.raises(FileNotFoundError):
        obj.load_checkpoint("missing")


# --------------------------------------------------------------------------- #
# Cadence and logging
# --------------------------------------------------------------------------- #
def test_log_saved_checkpoint_only_on_main_with_enabled_config(tmp_path, caplog):
    obj = _Recipe.__new__(_Recipe)
    obj.dist_env = SimpleNamespace(is_main=True)
    obj.checkpoint_config = SimpleNamespace(enabled=True, checkpoint_dir=str(tmp_path))
    with caplog.at_level(logging.INFO):
        obj._log_saved_checkpoint("step", 1, 5)
    assert "Saved step checkpoint" in caplog.text


def test_maybe_save_step_checkpoint_marks_final_step():
    obj = _Recipe.__new__(_Recipe)
    calls = []
    obj.save_checkpoint = lambda **kw: calls.append(kw)
    obj._log_saved_checkpoint = lambda *a: None
    obj.runtime = SimpleNamespace(global_step=4)
    obj.ckpt_every_steps = 2
    obj.total_optim_steps = 4
    assert obj._maybe_save_step_checkpoint(epoch=0) is True
    assert calls[0]["is_final_checkpoint"] is True
    obj.runtime.global_step = 3
    assert obj._maybe_save_step_checkpoint(epoch=0) is False


def test_maybe_save_final_checkpoint_skips_when_cadence_covered_final_step():
    obj = _Recipe.__new__(_Recipe)
    calls = []
    obj.save_checkpoint = lambda **kw: calls.append(kw)
    obj._log_saved_checkpoint = lambda *a: None
    obj.runtime = SimpleNamespace(global_step=0)
    assert obj._maybe_save_final_checkpoint(1) is False
    obj.runtime.global_step = 6
    obj.ckpt_every_steps = 3
    assert obj._maybe_save_final_checkpoint(1) is False
    obj.ckpt_every_steps = None
    assert obj._maybe_save_final_checkpoint(1) is True
    assert calls[0]["is_final_checkpoint"] is True


def test_wandb_log_forwards_to_active_run():
    logged = []
    obj = _Recipe.__new__(_Recipe)
    obj._wandb_log({"a": 1}, step=3)
    obj.wandb_run = SimpleNamespace(log=lambda data, step: logged.append((data, step)))
    obj._wandb_log({"a": 1}, step=3)
    assert logged == [({"a": 1}, 3)]


def _wandb_recipe(monkeypatch, *, is_main=True, wandb_block=None):
    calls = []
    monkeypatch.setattr(_spec_recipe_base, "suppress_wandb_log_messages", lambda: calls.append("suppress"))
    monkeypatch.setattr(
        _spec_recipe_base,
        "init_wandb_run",
        lambda wandb_kwargs, cfg_dict, default_name: calls.append((wandb_kwargs, cfg_dict, default_name)) or "RUN",
    )
    obj = _Recipe.__new__(_Recipe)
    obj.dist_env = SimpleNamespace(is_main=is_main)
    block = None if wandb_block is None else SimpleNamespace(to_dict=lambda: dict(wandb_block))
    obj.cfg = SimpleNamespace(
        get=lambda key, default=None: block if key == "wandb" else default,
        to_dict=lambda: {"lr": 1e-4},
    )
    return obj, calls


@pytest.mark.parametrize(
    "is_main,wandb_block",
    [(False, {"project": "p"}), (True, None), (True, {"enable": False, "project": "p"})],
)
def test_init_wandb_run_skipped(monkeypatch, is_main, wandb_block):
    obj, calls = _wandb_recipe(monkeypatch, is_main=is_main, wandb_block=wandb_block)
    obj._init_wandb_run("run")
    assert obj.wandb_run is None
    assert calls == []


def test_init_wandb_run_strips_enable_and_starts_on_main(monkeypatch):
    obj, calls = _wandb_recipe(monkeypatch, wandb_block={"enable": True, "project": "p", "group": "g"})
    obj._init_wandb_run("dspark_run")
    assert obj.wandb_run == "RUN"
    assert calls == ["suppress", ({"project": "p", "group": "g"}, {"lr": 1e-4}, "dspark_run")]


# --------------------------------------------------------------------------- #
# _save_meta / _load_meta
# --------------------------------------------------------------------------- #
def _meta_recipe():
    obj = _Recipe.__new__(_Recipe)
    obj.runtime = SimpleNamespace(global_step=7)
    obj._resume_epoch = 0
    obj.checkpoint_config = SimpleNamespace(allow_legacy_pickle_restore=False)
    return obj


def test_meta_round_trip(tmp_path):
    obj = _meta_recipe()
    obj._save_meta(str(tmp_path), "x_meta.pt", 3, block_size=4)
    obj.runtime.global_step = 0

    meta = obj._load_meta(str(tmp_path), "x_meta.pt")

    assert meta == {"global_step": 7, "epoch": 3, "block_size": 4}
    assert obj.runtime.global_step == 7
    assert obj._resume_epoch == 3


def test_load_meta_falls_back_to_legacy_name_and_missing_is_none(tmp_path):
    obj = _meta_recipe()
    assert obj._load_meta(str(tmp_path), "x_meta.pt", legacy_filename="old_meta.pt") is None
    assert obj.runtime.global_step == 7

    torch.save({"global_step": 2, "epoch": 1}, tmp_path / "old_meta.pt")
    assert obj._load_meta(str(tmp_path), "x_meta.pt", legacy_filename="old_meta.pt")["epoch"] == 1
    assert obj.runtime.global_step == 2
