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

"""Per-algorithm hooks of the shared draft training loop.

Each draft recipe reports its metric window and epoch summary through hooks of
``SpecDecodeRecipeBase``; these tests pin what each hook logs and returns, and
the parallelism-specific sync / clip hooks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from nemo_automodel.recipes.speculative_decoding import _spec_recipe_base
from nemo_automodel.recipes.speculative_decoding.algorithms import dspark
from nemo_automodel.recipes.speculative_decoding.algorithms.dflash import TrainDFlashRecipe, _DFlashTrainWindow
from nemo_automodel.recipes.speculative_decoding.algorithms.dspark import TrainDSparkRecipe, _DSparkMetricWindow
from nemo_automodel.recipes.speculative_decoding.algorithms.eagle import TrainEagle1Recipe
from nemo_automodel.recipes.speculative_decoding.algorithms.eagle3 import TrainEagle3Recipe


def _bare(recipe_cls, *, is_main: bool = True, **attrs):
    """Return a recipe built without ``setup()`` and with W&B captured."""
    recipe = recipe_cls.__new__(recipe_cls)
    recipe.dist_env = SimpleNamespace(is_main=is_main)
    recipe.runtime = SimpleNamespace(global_step=7)
    recipe.num_epochs = 2
    recipe.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [1e-4])
    recipe.wandb_run = mock.MagicMock()
    for key, value in attrs.items():
        setattr(recipe, key, value)
    return recipe


def _wandb_payload(recipe) -> dict:
    (payload,), kwargs = recipe.wandb_run.log.call_args
    assert kwargs == {"step": 7}
    return payload


# DSpark


def _dspark_window():
    avg = {"loss": 1.0, "ce_loss": 2.0, "l1_loss": 0.5, "confidence_loss": 0.1, "accept_rate": 0.4, "tau": 1.5}
    return SimpleNamespace(pack=lambda: torch.zeros(3), unpack=lambda reduced: dict(avg))


def test_dspark_window_log_mirrors_averages_under_train_prefix():
    recipe = _bare(TrainDSparkRecipe, metric_logger=mock.MagicMock())
    recipe._dp_allreduce = mock.MagicMock(side_effect=lambda t: t)
    pbar = mock.MagicMock()

    recipe._log_train_window(_dspark_window(), epoch_idx=1, grad_norm=None, pbar=pbar)

    recipe._dp_allreduce.assert_called_once()
    sample = recipe.metric_logger.log.call_args.args[0]
    assert sample.step == 7 and sample.epoch == 1
    assert sample.metrics["loss"] == 1.0 and sample.metrics["lr"] == 1e-4
    payload = _wandb_payload(recipe)
    assert payload["train/tv_loss"] == 0.5 and "train/l1_loss" not in payload
    assert payload["train/accept_rate"] == 0.4 and payload["train/epoch"] == 1
    pbar.set_postfix.assert_called_once()


def test_dspark_window_log_reduces_on_every_rank_but_logs_only_on_rank0():
    recipe = _bare(TrainDSparkRecipe, is_main=False, metric_logger=mock.MagicMock())
    recipe._dp_allreduce = mock.MagicMock(side_effect=lambda t: t)

    recipe._log_train_window(_dspark_window(), epoch_idx=0, grad_norm=None, pbar=None)

    recipe._dp_allreduce.assert_called_once()
    recipe.metric_logger.log.assert_not_called()
    recipe.wandb_run.log.assert_not_called()


def test_dspark_epoch_report_logs_validation_metrics():
    recipe = _bare(TrainDSparkRecipe)
    eval_metrics = {"val_loss": 1.25, "accept_rate": 0.5, "tau": 2.0}

    assert recipe._report_epoch(0, 3, eval_metrics) is eval_metrics

    assert _wandb_payload(recipe) == {"val/loss": 1.25, "val/accept_rate": 0.5, "val/tau": 2.0, "val/epoch": 0}


def test_dspark_clip_sync_and_fp8_hooks(monkeypatch):
    recipe = _bare(TrainDSparkRecipe, max_grad_norm=0.5, trainer_module=object(), block_size=4, device="cpu")
    clip = mock.MagicMock(return_value=torch.tensor(3.0))
    monkeypatch.setattr(dspark, "scale_grads_and_clip_grad_norm", clip)
    recipe._maybe_precompute_fp8_scales = mock.MagicMock()

    assert recipe._clip_grad_norm().item() == 3.0
    clip.assert_called_once_with(0.5, [recipe.trainer_module])
    assert recipe._sync_grads_before_clip() is None
    recipe._after_optimizer_step()
    recipe._maybe_precompute_fp8_scales.assert_called_once_with()
    window = recipe._new_train_window()
    assert isinstance(window, _DSparkMetricWindow) and window.block_size == 4


# DFlash


def test_dflash_window_log_includes_extra_metric_ratios():
    recipe = _bare(TrainDFlashRecipe, device="cpu")
    recipe._dp_allreduce = mock.MagicMock(side_effect=lambda t: t)
    window = _DFlashTrainWindow(
        loss=6.0,
        correct_tokens=3.0,
        valid_tokens=4.0,
        micro_batches=3,
        extra={"train/accept_len": [5.0, 2.0], "train/empty": [1.0, 0.0]},
    )

    recipe._log_train_window(window, epoch_idx=0, grad_norm=None, pbar=None)

    payload = _wandb_payload(recipe)
    assert payload["train/loss"] == pytest.approx(2.0)
    assert payload["train/accuracy"] == pytest.approx(0.75)
    assert payload["train/accept_len"] == pytest.approx(2.5)
    assert "train/empty" not in payload


def test_dflash_epoch_report_logs_validation_metrics():
    recipe = _bare(TrainDFlashRecipe, _skipped_micro_batches=1)

    assert recipe._report_epoch(0, 3, {"val_loss": 1.5, "val_accept_len": 2.0}) == {
        "val_loss": 1.5,
        "val_accept_len": 2.0,
    }
    assert _wandb_payload(recipe) == {"val/loss": 1.5, "val/accept_len": 2.0}


# EAGLE-1


def test_eagle1_epoch_report_logs_validation_metrics():
    recipe = _bare(TrainEagle1Recipe)

    recipe._report_epoch(1, 4, {"val_loss": 0.5, "val_accuracy": 0.25})

    assert _wandb_payload(recipe) == {"val/loss": 0.5, "val/accuracy": 0.25}


# EAGLE-3


@pytest.mark.parametrize("tau", [None, 1.75])
def test_eagle3_epoch_report_returns_checkpoint_metrics(tau):
    recipe = _bare(TrainEagle3Recipe)

    metrics = recipe._report_epoch(0, 3, (torch.tensor(0.5), torch.tensor(0.25), tau))

    assert metrics == {"val_loss": 0.5, "val_accuracy": 0.25}
    payload = _wandb_payload(recipe)
    assert payload["val/loss"] == 0.5
    assert ("val/tau_sim" in payload) == (tau is not None)
    assert recipe._report_epoch(0, 3, None) is None


def test_eagle3_train_start_logs_only_on_rank0():
    recipe = _bare(TrainEagle3Recipe, is_main=False, train_dataloader=mock.MagicMock())

    recipe._on_train_start(0)

    recipe.train_dataloader.__len__.assert_not_called()


def test_eagle3_finish_checkpointing_waits_for_pending_writes():
    checkpointer = mock.MagicMock()
    recipe = _bare(TrainEagle3Recipe, checkpointer=checkpointer)

    recipe._finish_checkpointing()

    checkpointer.async_wait.assert_called_once_with()
    checkpointer.lifecycle.complete_pending.assert_called_once_with()
    checkpointer.finalize.assert_not_called()


def test_eagle3_iterates_prefetched_pairs_when_prefetch_is_on():
    recipe = _bare(TrainEagle3Recipe, target_prefetch_depth=2, train_dataloader=["b0", "b1"])
    recipe._prefetched_batches = mock.MagicMock(return_value=iter([("b0", "t0")]))

    assert list(recipe._iter_train_batches()) == [("b0", "t0")]
    recipe._prefetched_batches.assert_called_once_with(["b0", "b1"])

    recipe.target_prefetch_depth = 0
    assert list(recipe._iter_train_batches()) == [("b0", None), ("b1", None)]


def test_eagle3_peagle_partitioned_step_bypasses_the_sync_context():
    recipe = _bare(TrainEagle3Recipe, _peagle_partitioned=True)
    recipe._peagle_partitioned_step = mock.MagicMock(return_value="metrics")
    sync_ctx = mock.MagicMock()

    assert recipe._train_micro_batch(("batch", None), sync_ctx) == "metrics"

    recipe._peagle_partitioned_step.assert_called_once_with("batch")
    sync_ctx.__enter__.assert_not_called()


def test_eagle3_sums_draft_grads_over_cp_before_the_tp_sync(monkeypatch):
    calls = []
    recipe = _bare(TrainEagle3Recipe, cp_group=object(), trainer_module=object())
    recipe._all_reduce_draft_grads_over_cp = lambda: calls.append("cp")
    monkeypatch.setattr(_spec_recipe_base, "synchronize_tp_replica_gradients", lambda parts, mesh: calls.append("tp"))

    recipe._sync_grads_before_clip()

    assert calls == ["cp", "tp"]


def test_dspark_teardown_closes_logger_wandb_and_remote_target():
    recipe = _bare(TrainDSparkRecipe, metric_logger=mock.MagicMock())
    recipe._finish_wandb = mock.MagicMock()
    recipe._close_remote_target = mock.MagicMock()

    recipe._on_train_end()

    recipe.metric_logger.close.assert_called_once_with()
    recipe._finish_wandb.assert_called_once_with()
    recipe._close_remote_target.assert_called_once_with()
