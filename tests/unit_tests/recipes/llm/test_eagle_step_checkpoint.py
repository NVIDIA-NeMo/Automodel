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

"""Unit tests for step-based checkpointing in the EAGLE-1/2 and EAGLE-3 recipes.

The EAGLE recipes run their own training loop (they do not use ``StepScheduler``)
and historically only saved a checkpoint at each epoch boundary. These tests guard
the added ``ckpt_every_steps`` (save every N optimizer steps) and
``save_checkpoint_every_epoch`` (toggle the end-of-epoch save) behavior:

1. ``_maybe_save_step_checkpoint`` fires exactly on ``global_step`` multiples of
   ``ckpt_every_steps`` and is a no-op when it is unset / non-positive.
2. Driven through the real EAGLE-1 training loop, mid-epoch step checkpoints are
   written at the right steps and the epoch save can be turned off independently.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.recipes.llm.train_eagle1 import TrainEagle1Recipe
from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe

RECIPE_CLASSES = [TrainEagle1Recipe, TrainEagle3Recipe]


# ---------------------------------------------------------------------------
# _maybe_save_step_checkpoint in isolation (both recipes share the same logic)
# ---------------------------------------------------------------------------


def _helper_self(ckpt_every_steps, global_step, total_optim_steps=None):
    """A minimal stand-in carrying just the attributes the helper reads."""
    calls = []
    return (
        SimpleNamespace(
            ckpt_every_steps=ckpt_every_steps,
            runtime=SimpleNamespace(global_step=global_step),
            total_optim_steps=total_optim_steps,
            dist_env=SimpleNamespace(is_main=True),
            checkpoint_config=None,
            save_checkpoint=lambda **kw: calls.append(kw),
            _log_saved_checkpoint=lambda *a, **k: None,
        ),
        calls,
    )


@pytest.mark.parametrize("recipe_cls", RECIPE_CLASSES)
@pytest.mark.parametrize(
    "every,step,should_fire",
    [
        (2, 1, False),
        (2, 2, True),
        (2, 3, False),
        (2, 4, True),
        (3, 6, True),
        (1, 7, True),
    ],
)
def test_maybe_save_step_checkpoint_fires_on_multiples(recipe_cls, every, step, should_fire):
    obj, calls = _helper_self(every, step)
    fired = recipe_cls._maybe_save_step_checkpoint(obj, epoch=0)
    assert fired is should_fire
    assert len(calls) == (1 if should_fire else 0)
    if should_fire:
        assert calls[0]["step"] == step
        assert calls[0]["epoch"] == 0
        assert calls[0]["val_loss"] is None


@pytest.mark.parametrize("recipe_cls", RECIPE_CLASSES)
@pytest.mark.parametrize("every", [None, 0, -1])
def test_maybe_save_step_checkpoint_disabled(recipe_cls, every):
    obj, calls = _helper_self(every, 10)
    assert recipe_cls._maybe_save_step_checkpoint(obj, epoch=0) is False
    assert calls == []


@pytest.mark.parametrize("recipe_cls", RECIPE_CLASSES)
def test_maybe_save_step_checkpoint_missing_attr_is_noop(recipe_cls):
    """A partially-constructed recipe (no ``ckpt_every_steps`` attr) must not crash."""
    obj = SimpleNamespace(runtime=SimpleNamespace(global_step=4))
    assert recipe_cls._maybe_save_step_checkpoint(obj, epoch=0) is False


@pytest.mark.parametrize("recipe_cls", RECIPE_CLASSES)
@pytest.mark.parametrize(
    "total,step,expect_final",
    [
        (None, 4, False),  # unknown total (e.g. unsized dataloader) -> never final
        (6, 4, False),  # mid-run step save
        (4, 4, True),  # save lands exactly on the last optimizer step
    ],
)
def test_step_checkpoint_final_flag_tracks_total_optim_steps(recipe_cls, total, step, expect_final):
    """The step save's final flag tracks ``total_optim_steps``."""
    obj, calls = _helper_self(2, step, total_optim_steps=total)
    assert recipe_cls._maybe_save_step_checkpoint(obj, epoch=0) is True
    assert calls[0]["is_final_checkpoint"] is expect_final


def _final_self(ckpt_every_steps, save_every_epoch, global_step):
    calls = []
    return (
        SimpleNamespace(
            ckpt_every_steps=ckpt_every_steps,
            save_checkpoint_every_epoch=save_every_epoch,
            runtime=SimpleNamespace(global_step=global_step),
            dist_env=SimpleNamespace(is_main=True),
            checkpoint_config=None,
            save_checkpoint=lambda **kw: calls.append(kw),
            _log_saved_checkpoint=lambda *a, **k: None,
        ),
        calls,
    )


@pytest.mark.parametrize("recipe_cls", RECIPE_CLASSES)
@pytest.mark.parametrize(
    "every,save_epoch,gs,should_fire",
    [
        (None, False, 7, True),  # no cadence -> final is the only checkpoint
        (2, False, 7, True),  # step cadence misses the final step (7 % 2 != 0) -> safety net
        (2, False, 8, False),  # step cadence already saved the final step (8 % 2 == 0)
        (None, True, 7, False),  # epoch cadence already saved the final step
        (2, True, 7, False),  # epoch cadence covers the final step
        (None, False, 0, False),  # nothing trained yet
    ],
)
def test_final_checkpoint_fires_unless_final_step_already_saved(recipe_cls, every, save_epoch, gs, should_fire):
    obj, calls = _final_self(ckpt_every_steps=every, save_every_epoch=save_epoch, global_step=gs)
    assert recipe_cls._maybe_save_final_checkpoint(obj, completed_epochs=3) is should_fire
    assert len(calls) == (1 if should_fire else 0)
    if should_fire:
        assert calls[0]["epoch"] == 3
        assert calls[0]["step"] == gs
        assert calls[0]["is_final_checkpoint"] is True


# ---------------------------------------------------------------------------
# End-to-end through the real EAGLE-1 training loop
# ---------------------------------------------------------------------------


class _ConstantGradModule(nn.Module):
    def __init__(self, n: int = 4):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n))

    def forward(self, **kwargs):
        # Mirror EagleStepMetrics: the recipe loop reads hidden_loss/token_loss
        # on every micro-batch (for W&B logging), so the fake must expose them.
        loss = self.w.sum()
        return SimpleNamespace(
            loss=loss,
            accuracy=torch.tensor(0.5),
            hidden_loss=loss.detach(),
            token_loss=torch.zeros((), device=self.w.device),
        )


class _FakeTargetWrapper:
    def generate_batch(self, input_ids, attention_mask, loss_mask, **packing_kwargs):
        return SimpleNamespace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            input_hidden_states=None,
            target_hidden_states=None,
            target_logits=None,
            position_ids=None,
            seq_lens=None,
            doc_remaining=None,
        )


class _ListLoader:
    def __init__(self, num_batches: int):
        one = torch.ones(1, 4, dtype=torch.long)
        self._batches = [
            {"input_ids": one.clone(), "attention_mask": one.clone(), "loss_mask": one.clone()}
            for _ in range(num_batches)
        ]

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _build_eagle1_recipe(num_batches, grad_accum, num_epochs, ckpt_every_steps, save_every_epoch):
    recipe = TrainEagle1Recipe.__new__(TrainEagle1Recipe)
    recipe.device = torch.device("cpu")
    recipe.dist_env = SimpleNamespace(is_main=True, world_size=1)
    recipe.trainer_module = _ConstantGradModule()
    recipe.target_wrapper = _FakeTargetWrapper()
    recipe.train_dataloader = _ListLoader(num_batches)
    recipe.val_dataloader = None
    recipe.runtime = SimpleNamespace(global_step=0)
    recipe.grad_accumulation_steps = grad_accum
    recipe.max_grad_norm = 1e9
    recipe.num_epochs = num_epochs
    recipe.log_every_steps = 1
    recipe.total_optim_steps = num_epochs * -(-num_batches // grad_accum)
    recipe.ckpt_every_steps = ckpt_every_steps
    recipe.save_checkpoint_every_epoch = save_every_epoch
    recipe.optimizer = torch.optim.SGD(recipe.trainer_module.parameters(), lr=0.0)
    recipe.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(recipe.optimizer, lambda s: 1.0)

    saves = []
    recipe.save_checkpoint = lambda **kw: saves.append((kw["epoch"], kw["step"], kw["is_final_checkpoint"]))
    return recipe, saves


def test_step_checkpoints_written_during_loop():
    # 6 batches, accum 2 -> 3 optimizer steps; ckpt every 2 steps -> saves at step 2.
    recipe, saves = _build_eagle1_recipe(
        num_batches=6, grad_accum=2, num_epochs=1, ckpt_every_steps=2, save_every_epoch=True
    )
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 3
    # mid-epoch step checkpoint at global_step 2, labeled with the current (0-based) epoch
    assert any(e == 0 and s == 2 for (e, s, _) in saves)
    # end-of-epoch checkpoint at the final step, labeled epoch + 1
    assert any(e == 1 and s == 3 for (e, s, _) in saves)


def test_save_checkpoint_every_epoch_false_skips_epoch_save():
    recipe, saves = _build_eagle1_recipe(
        num_batches=6, grad_accum=2, num_epochs=1, ckpt_every_steps=1, save_every_epoch=False
    )
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 3
    # Only per-step saves (epoch labels 0,0,0), none labeled epoch+1 (==1).
    assert [s for (e, s, _) in saves] == [1, 2, 3]
    assert all(e == 0 for (e, s, _) in saves)


def test_only_final_checkpoint_when_both_unset():
    # Neither cadence set -> exactly one save, at the end of the whole run,
    # labeled with num_epochs and the final global_step, flagged final.
    recipe, saves = _build_eagle1_recipe(
        num_batches=6, grad_accum=2, num_epochs=2, ckpt_every_steps=None, save_every_epoch=False
    )
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 6
    assert saves == [(2, 6, True)]


def test_step_only_saves_final_when_last_step_off_cadence():
    # 3 optimizer steps, ckpt every 2 -> step save at gs=2 only. gs=3 (the final
    # fully-trained step) is not a multiple of 2, so the safety net writes it,
    # labeled with num_epochs.
    recipe, saves = _build_eagle1_recipe(
        num_batches=6, grad_accum=2, num_epochs=1, ckpt_every_steps=2, save_every_epoch=False
    )
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 3
    # only the safety-net save is final; the mid-epoch step save is not
    assert saves == [(0, 2, False), (1, 3, True)]


def test_step_only_no_final_duplicate_when_last_step_on_cadence():
    # 3 optimizer steps, ckpt every 3 -> step save lands exactly on gs=3, so the
    # safety net must not add a duplicate final checkpoint.
    recipe, saves = _build_eagle1_recipe(
        num_batches=6, grad_accum=2, num_epochs=1, ckpt_every_steps=3, save_every_epoch=False
    )
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 3
    # the on-cadence save IS the run's last checkpoint, so it must be final
    assert saves == [(0, 3, True)]


def test_only_epoch_checkpoints_no_final_duplicate():
    # Epoch cadence on, step off -> per-epoch saves only; the final-checkpoint
    # fallback must NOT add a duplicate.
    recipe, saves = _build_eagle1_recipe(
        num_batches=6, grad_accum=2, num_epochs=2, ckpt_every_steps=None, save_every_epoch=True
    )
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 6
    # one save per epoch boundary, labeled epoch+1, never a num_epochs final
    # extra; only the last epoch's save is flagged final
    assert saves == [(1, 3, False), (2, 6, True)]


# ---------------------------------------------------------------------------
# save_checkpoint forwards is_final_checkpoint to the checkpointer
# ---------------------------------------------------------------------------


def _saving_recipe(recipe_cls, tmp_path):
    """A recipe stub with just enough state to run the real ``save_checkpoint``."""
    recipe = recipe_cls.__new__(recipe_cls)
    recipe.checkpointer = MagicMock()
    recipe.checkpointer.config = SimpleNamespace(enabled=True, is_async=False)
    recipe.checkpoint_config = SimpleNamespace(checkpoint_dir=str(tmp_path))
    recipe.tokenizer = object()
    recipe.optimizer = MagicMock()
    recipe.lr_scheduler = MagicMock()
    recipe.rng = MagicMock()
    recipe._module = lambda: SimpleNamespace(draft_model=nn.Linear(2, 2))
    recipe._save_extra_state = lambda path, epoch: None
    recipe._update_latest_symlink = lambda path: None
    recipe.cfg = SimpleNamespace()  # no raw_config -> the config snapshot is skipped
    return recipe


@pytest.mark.parametrize("recipe_cls", RECIPE_CLASSES)
@pytest.mark.parametrize("is_final", [True, False])
def test_save_checkpoint_forwards_final_flag_to_checkpointer(recipe_cls, is_final, tmp_path):
    """Regression: ``save_checkpoint`` used to derive the flag from a
    ``step_scheduler`` attribute these hand-rolled recipes never set, so the
    checkpointer never saw ``is_final_checkpoint=True`` and
    ``save_consolidated: final`` never exported the HF safetensors."""
    recipe = _saving_recipe(recipe_cls, tmp_path)
    recipe.save_checkpoint(epoch=1, step=5, is_final_checkpoint=is_final)
    assert recipe.checkpointer.save_model.call_args.kwargs["is_final_checkpoint"] is is_final
