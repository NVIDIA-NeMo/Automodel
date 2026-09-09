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

"""Final async checkpoint publication is owned by the checkpoint component."""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.checkpoint.lifecycle import CheckpointLifecycle
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


def _lifecycle(checkpoint_dir="/ckpt", process_group=None):
    config = CheckpointingConfig(
        checkpoint_dir=checkpoint_dir,
        save_consolidated=False,
    )
    return CheckpointLifecycle(config=config, process_group=process_group)


def test_complete_pending_publishes_latest_best_and_retention(monkeypatch):
    lifecycle = _lifecycle()
    events = []
    lifecycle.defer_publication(
        "/ckpt/epoch_1_step_10",
        best_val_metric=0.5,
        metric_key="val_loss",
    )
    monkeypatch.setattr(lifecycle, "_publish_checkpoint", lambda path: events.append(("latest", path)))
    monkeypatch.setattr(
        lifecycle,
        "_update_best_checkpoint",
        lambda path, value, metric_key: events.append(("best", path, value, metric_key)),
    )
    monkeypatch.setattr(lifecycle, "_prune_old_checkpoints", lambda: events.append(("prune",)))

    lifecycle.complete_pending()

    assert events == [
        ("latest", "/ckpt/epoch_1_step_10"),
        ("best", "/ckpt/epoch_1_step_10", 0.5, "val_loss"),
        ("prune",),
    ]
    assert lifecycle._pending_checkpoint_dir is None
    assert lifecycle._pending_best_checkpoint is None

    lifecycle.complete_pending()
    assert len(events) == 3


def test_complete_pending_uses_lifecycle_process_group(monkeypatch):
    process_group = object()
    lifecycle = _lifecycle(process_group=process_group)
    lifecycle.defer_publication(
        "/ckpt/epoch_1_step_10",
        best_val_metric=None,
        metric_key=None,
    )
    barriers = []
    rank_groups = []
    reduce_groups = []

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_rank",
        lambda group=None: rank_groups.append(group) or 0,
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: barriers.append(group))
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: reduce_groups.append(group),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(lifecycle, "_publish_checkpoint", lambda path: None)
    monkeypatch.setattr(lifecycle, "_prune_old_checkpoints", lambda: None)

    lifecycle.complete_pending()

    assert rank_groups == [process_group, process_group, process_group]
    assert barriers == [process_group, process_group, process_group]
    assert reduce_groups == [process_group, process_group, process_group]


def test_checkpointer_owns_lifecycle_with_the_same_process_group():
    process_group = object()
    config = CheckpointingConfig(
        enabled=False,
        checkpoint_dir="/ckpt",
        save_consolidated=False,
    )

    checkpointer = Checkpointer(
        config=config,
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        process_group=process_group,
    )

    assert checkpointer.lifecycle.config is config
    assert checkpointer.lifecycle.process_group is process_group


def test_checkpointer_finalize_waits_publishes_and_closes():
    events = []
    checkpointer = Checkpointer.__new__(Checkpointer)
    checkpointer.config = SimpleNamespace(enabled=True)
    checkpointer.lifecycle = SimpleNamespace(complete_pending=lambda: events.append("publish"))
    checkpointer.async_wait = lambda: events.append("wait")
    checkpointer.close = lambda: events.append("close")

    checkpointer.finalize()

    assert events == ["wait", "publish", "close"]


def test_checkpointer_finalize_closes_when_publication_fails():
    events = []
    checkpointer = Checkpointer.__new__(Checkpointer)
    checkpointer.config = SimpleNamespace(enabled=True)

    def fail_publication():
        events.append("publish")
        raise RuntimeError("publication failed")

    checkpointer.lifecycle = SimpleNamespace(complete_pending=fail_publication)
    checkpointer.async_wait = lambda: events.append("wait")
    checkpointer.close = lambda: events.append("close")

    with pytest.raises(RuntimeError, match="publication failed"):
        checkpointer.finalize()

    assert events == ["wait", "publish", "close"]


def test_checkpointer_finalize_only_closes_when_checkpointing_is_disabled():
    events = []
    checkpointer = Checkpointer.__new__(Checkpointer)
    checkpointer.config = SimpleNamespace(enabled=False)
    checkpointer.lifecycle = SimpleNamespace(complete_pending=lambda: events.append("publish"))
    checkpointer.async_wait = lambda: events.append("wait")
    checkpointer.close = lambda: events.append("close")

    checkpointer.finalize()

    assert events == ["close"]


def test_base_recipe_cleanup_delegates_to_checkpointer_finalize():
    events = []
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.checkpointer = SimpleNamespace(finalize=lambda: events.append("finalize"))

    recipe._finalize_and_close_checkpointer()

    assert events == ["finalize"]


def test_base_recipe_cleanup_without_checkpointer_is_a_noop():
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)

    recipe._finalize_and_close_checkpointer()
