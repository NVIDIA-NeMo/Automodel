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

"""Unit tests for ``TrainDFlashRecipe`` helpers that don't require a real model.

The recipe is constructed via ``__new__`` (bypassing ``setup()``), so only the
attributes each helper reads are populated -- mirroring the EAGLE recipe tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_automodel.recipes.llm import train_dflash
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe


def _ckpt_self(ckpt_every_steps, save_every_epoch, global_step, total_optim_steps=None):
    calls = []
    return (
        SimpleNamespace(
            ckpt_every_steps=ckpt_every_steps,
            save_checkpoint_every_epoch=save_every_epoch,
            runtime=SimpleNamespace(global_step=global_step),
            total_optim_steps=total_optim_steps,
            dist_env=SimpleNamespace(is_main=True),
            checkpoint_config=None,
            save_checkpoint=lambda **kw: calls.append(kw),
            _log_saved_checkpoint=lambda *a, **k: None,
        ),
        calls,
    )


@pytest.mark.parametrize(
    "every,step,should_fire",
    [(2, 1, False), (2, 2, True), (2, 3, False), (3, 6, True), (None, 6, False), (0, 4, False)],
)
def test_maybe_save_step_checkpoint(every, step, should_fire):
    obj, calls = _ckpt_self(every, False, step)
    assert TrainDFlashRecipe._maybe_save_step_checkpoint(obj, epoch=0) is should_fire
    assert len(calls) == (1 if should_fire else 0)


def test_maybe_save_step_checkpoint_marks_cadence_save_final_at_last_step():
    obj, calls = _ckpt_self(ckpt_every_steps=2, save_every_epoch=False, global_step=8, total_optim_steps=8)

    assert TrainDFlashRecipe._maybe_save_step_checkpoint(obj, epoch=3) is True

    assert calls == [{"epoch": 3, "step": 8, "best_metric_key": "val_loss", "is_final_checkpoint": True}]


@pytest.mark.parametrize(
    "every,save_epoch,gs,should_fire",
    [
        (None, False, 7, True),  # no cadence -> final is the only checkpoint
        (2, False, 7, True),  # step cadence misses the last step -> safety net
        (2, False, 8, False),  # step cadence already saved the last step
        (None, True, 7, False),  # epoch cadence covers the last step
        (None, False, 0, False),  # nothing trained yet
    ],
)
def test_maybe_save_final_checkpoint(every, save_epoch, gs, should_fire):
    obj, calls = _ckpt_self(every, save_epoch, gs)
    assert TrainDFlashRecipe._maybe_save_final_checkpoint(obj, completed_epochs=3) is should_fire
    assert len(calls) == (1 if should_fire else 0)
    if should_fire:
        assert calls[0]["is_final_checkpoint"] is True


def test_resolve_mask_token_id_prefers_explicit():
    cfg = SimpleNamespace(get=lambda k, d=None: 99 if k == "mask_token_id" else d)
    assert TrainDFlashRecipe._resolve_mask_token_id(cfg, vocab_size=1000) == 99


def test_resolve_mask_token_id_requires_explicit():
    """Unset mask_token_id is a hard error (no silent fallback to pad, which often
    aliases eos and would quietly degrade the mask-slot semantics)."""
    cfg = SimpleNamespace(get=lambda k, d=None: d)  # nothing set
    with pytest.raises(ValueError, match="mask_token_id to be set explicitly"):
        TrainDFlashRecipe._resolve_mask_token_id(cfg, vocab_size=1000)


def test_resolve_mask_token_id_range_checks():
    """An id outside the vocab (a typo, or a stale token from another model) is
    rejected rather than indexing the embed table out of bounds -- both the upper
    (>= vocab_size) and the lower (< 0) bound."""
    for bad in (5000, -1):
        cfg = SimpleNamespace(get=lambda k, d=None, _v=bad: _v if k == "mask_token_id" else d)
        with pytest.raises(ValueError, match="out of range"):
            TrainDFlashRecipe._resolve_mask_token_id(cfg, vocab_size=1000)


def test_all_ranks_have_valid_single_process_passes_local_flag():
    # No DDP -> the local flag passes through unchanged (no collective).
    assert train_dflash._all_ranks_have_valid(1, is_ddp=False, device="cpu") is True
    assert train_dflash._all_ranks_have_valid(0, is_ddp=False, device="cpu") is False


def test_all_ranks_have_valid_ddp_min_reduces_across_ranks(monkeypatch):
    """Under DDP the flag is MIN-reduced: one rank without anchors makes all skip."""
    monkeypatch.setattr(train_dflash.dist, "is_available", lambda: True)
    monkeypatch.setattr(train_dflash.dist, "is_initialized", lambda: True)

    # Another rank skipped -> the collective drives the flag to 0 -> all skip.
    monkeypatch.setattr(train_dflash.dist, "all_reduce", lambda t, op=None: t.fill_(0))
    assert train_dflash._all_ranks_have_valid(1, is_ddp=True, device="cpu") is False

    # Every rank valid -> MIN leaves the 1 in place -> all run the backward.
    monkeypatch.setattr(train_dflash.dist, "all_reduce", lambda t, op=None: None)
    assert train_dflash._all_ranks_have_valid(1, is_ddp=True, device="cpu") is True
