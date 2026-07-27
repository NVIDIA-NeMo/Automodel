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

"""``BaseRecipe._finalize_pending_checkpoint`` flushes the last async checkpoint.

The async ``save_checkpoint`` path defers the latest/best symlink update to the
next save's preamble; the final checkpoint has no next save, so the training loop
calls this once at the end to wait the write and flush the deferred symlinks.
"""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe


def _recipe(enabled=True, pending=None, best_pending=None):
    # Any BaseRecipe subclass exercises the shared helper; TrainDFlashRecipe is one.
    r = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    r._waited = []
    r.checkpointer = SimpleNamespace(
        config=SimpleNamespace(enabled=enabled),
        async_wait=lambda: r._waited.append(1),
    )
    r._last_pending_checkpoint_dir = pending
    r._last_pending_best_checkpoint_info = best_pending
    r._latest = []
    r._best = []
    r._pruned = []
    r._update_latest_symlink = lambda d: r._latest.append(d)
    r._update_best_symlink = lambda d, v, metric_key=None: r._best.append((d, v, metric_key))
    r._prune_old_checkpoints = lambda: r._pruned.append(1)
    return r


def test_finalize_waits_and_flushes_pending_latest_and_best():
    r = _recipe(pending="/ckpt/epoch_1_step_10", best_pending={"path": "/ckpt/epoch_1_step_10", "val": 0.5})
    r._finalize_pending_checkpoint()
    assert r._waited == [1]
    assert r._latest == ["/ckpt/epoch_1_step_10"]
    assert r._best == [("/ckpt/epoch_1_step_10", 0.5, None)]
    assert r._pruned == [1]
    # pending cleared so a second finalize is a no-op
    assert r._last_pending_checkpoint_dir is None
    assert r._last_pending_best_checkpoint_info is None


def test_finalize_with_no_pending_only_waits():
    r = _recipe(pending=None, best_pending=None)
    r._finalize_pending_checkpoint()
    assert r._waited == [1]
    assert r._latest == []
    assert r._best == []


def test_finalize_uses_recipe_process_group(monkeypatch):
    process_group = object()
    barriers = []
    r = _recipe(pending="/ckpt/epoch_1_step_10", best_pending=None)
    r.mesh_context = SimpleNamespace(process_group=process_group)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: barriers.append(group))

    r._finalize_pending_checkpoint()

    assert barriers == [process_group, process_group]


def test_finalize_is_noop_when_checkpointing_disabled():
    r = _recipe(enabled=False, pending="/ckpt/epoch_1_step_10")
    r._finalize_pending_checkpoint()
    # disabled -> no async_wait, no symlink flush
    assert r._waited == []
    assert r._latest == []
    assert r._last_pending_checkpoint_dir == "/ckpt/epoch_1_step_10"


def test_finalize_is_noop_without_checkpointer():
    r = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    # No checkpointer attribute at all (e.g. checkpointing never set up).
    r._finalize_pending_checkpoint()  # must not raise


def test_finalize_and_close_closes_checkpointer_when_finalization_fails():
    events = []
    r = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    r.checkpointer = SimpleNamespace(close=lambda: events.append("close"))

    def fail_finalize():
        events.append("finalize")
        raise RuntimeError("publication failed")

    r._finalize_pending_checkpoint = fail_finalize

    with pytest.raises(RuntimeError, match="publication failed"):
        r._finalize_and_close_checkpointer()

    assert events == ["finalize", "close"]
