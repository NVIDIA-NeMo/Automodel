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
    obj = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=5))
    cfg = SimpleNamespace(get=lambda k, d=None: 99 if k == "mask_token_id" else d)
    assert TrainDFlashRecipe._resolve_mask_token_id(obj, cfg) == 99


def test_resolve_mask_token_id_falls_back_to_pad():
    obj = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=5))
    cfg = SimpleNamespace(get=lambda k, d=None: d)  # nothing set
    assert TrainDFlashRecipe._resolve_mask_token_id(obj, cfg) == 5


def test_resolve_mask_token_id_raises_without_any():
    obj = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=None))
    cfg = SimpleNamespace(get=lambda k, d=None: d)
    with pytest.raises(ValueError, match="mask_token_id"):
        TrainDFlashRecipe._resolve_mask_token_id(obj, cfg)
