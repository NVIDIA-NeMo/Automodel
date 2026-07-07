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

"""Coverage for the simulated accept-length (tau) reporting in the EAGLE-3
recipe: ``_all_reduce_sum`` / ``_window_tau_sim``, the training-window
accumulation and progress-bar/log wiring, and the ``_run_eval`` tau column.

The trainer-side math (per-step counts, :func:`simulated_accept_length`) is
covered in ``tests/unit_tests/speculative/test_eagle3.py``; here the trainer
module is faked and only the recipe plumbing is exercised.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nemo_automodel.components.speculative.eagle.target import Eagle3TargetBatch
from nemo_automodel.recipes.llm._spec_train_utils import optim_steps_per_epoch as _optim_steps_per_epoch
from nemo_automodel.recipes.llm.train_eagle3 import (
    TrainEagle3Recipe,
    _all_reduce_sum,
    _window_tau_sim,
)

# ---------------------------------------------------------------------------
# Minimal stand-ins (mirrors test_train_eagle3_flush.py, plus step counts)
# ---------------------------------------------------------------------------


class _FakeDraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)
        self.config = SimpleNamespace(save_pretrained=lambda path: None)


class _FakeTrainerModule(nn.Module):
    """Fake trainer whose metrics carry fixed per-TTT-step counts.

    ``step_correct=[2,1]`` / ``step_valid=[4,2]`` give per-step accuracies
    ``[0.5, 0.5]`` and thus ``tau = 1 + 0.5 + 0.25 = 1.75`` for any window.
    With ``with_step_counts=False`` the metrics omit the fields entirely,
    modeling the P-EAGLE trainer.
    """

    def __init__(self, with_step_counts: bool = True):
        super().__init__()
        self.draft_model = _FakeDraftModel()
        self.dummy = nn.Linear(4, 4)
        self.with_step_counts = with_step_counts

    def forward(self, **kwargs):
        out = self.dummy(torch.randn(1, 4)).sum()
        metrics = SimpleNamespace(
            loss=out.abs() + 1.0,
            accuracy=torch.tensor(0.5),
            valid_tokens=torch.tensor(6),
        )
        if self.with_step_counts:
            metrics.step_correct = torch.tensor([2.0, 1.0])
            metrics.step_valid = torch.tensor([4.0, 2.0])
        return metrics


class _FakeTargetWrapper:
    def generate_batch(self, input_ids, attention_mask, loss_mask):
        bs, sl = input_ids.shape
        return Eagle3TargetBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states=torch.randn(bs, sl, 16),
            logits=torch.randn(bs, sl, 64),
        )

    def close(self):
        pass


class _KeyedLoader:
    def __init__(self, loader):
        self._loader = loader
        self.sampler = loader.sampler

    def __iter__(self):
        for ids, attn, lm in self._loader:
            yield {"input_ids": ids, "attention_mask": attn, "loss_mask": lm}

    def __len__(self):
        return len(self._loader)


def _make_loader(num_samples: int, sl: int = 6) -> _KeyedLoader:
    data = TensorDataset(
        torch.randint(0, 64, (num_samples, sl)),
        torch.ones(num_samples, sl, dtype=torch.long),
        torch.ones(num_samples, sl, dtype=torch.long),
    )
    return _KeyedLoader(DataLoader(data, batch_size=1))


def _build_recipe(tmp_path, num_samples=4, grad_accum=2, with_step_counts=True, with_val=False):
    trainer_module = _FakeTrainerModule(with_step_counts=with_step_counts)

    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe.device = torch.device("cpu")
    recipe.dist_env = SimpleNamespace(is_main=True, world_size=1)
    recipe.trainer_module = trainer_module
    recipe.target_wrapper = _FakeTargetWrapper()
    recipe.target_prefetch_depth = 0
    recipe.train_dataloader = _make_loader(num_samples)
    recipe.val_dataloader = _make_loader(2) if with_val else None
    recipe.output_dir = tmp_path
    recipe.runtime = SimpleNamespace(global_step=0)
    recipe.grad_accumulation_steps = grad_accum
    recipe.max_grad_norm = 1.0
    recipe.num_epochs = 1
    recipe.log_every_steps = 1
    recipe.peak_lr = 1e-4
    recipe.total_optim_steps = _optim_steps_per_epoch(num_samples, grad_accum)
    recipe.warmup_steps = 1
    recipe.min_lr_ratio = 0.1
    recipe.optimizer = torch.optim.AdamW([p for p in trainer_module.parameters() if p.requires_grad], lr=1e-4)
    recipe.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(recipe.optimizer, lambda s: 1.0)
    return recipe


class _FakeProgressBar:
    def __init__(self):
        self.n = 0
        self.postfix = None
        self.closed = False

    def update(self, count=1):
        self.n += count

    def set_postfix(self, **kwargs):
        self.postfix = kwargs

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# _all_reduce_sum / _window_tau_sim (non-distributed)
# ---------------------------------------------------------------------------


def test_all_reduce_sum_passthrough():
    t = torch.tensor([1.0, 2.0])
    assert _all_reduce_sum(t) is t


def test_window_tau_sim_none_inputs():
    assert _window_tau_sim(None, None) is None
    assert _window_tau_sim(torch.tensor([1.0]), None) is None


def test_window_tau_sim_empty_window():
    zero = torch.zeros(3)
    assert _window_tau_sim(zero, zero.clone()) is None


def test_window_tau_sim_value():
    tau = _window_tau_sim(torch.tensor([2.0, 1.0]), torch.tensor([4.0, 2.0]))
    assert tau == pytest.approx(1.75)


def test_window_tau_sim_does_not_mutate_buffers():
    correct = torch.tensor([2.0, 1.0])
    valid = torch.tensor([4.0, 2.0])
    _window_tau_sim(correct, valid)
    assert correct.tolist() == [2.0, 1.0]
    assert valid.tolist() == [4.0, 2.0]


# ---------------------------------------------------------------------------
# Training loop wiring
# ---------------------------------------------------------------------------


def test_train_loop_reports_tau_in_postfix_and_wandb(tmp_path, monkeypatch):
    fake_pbar = _FakeProgressBar()
    monkeypatch.setattr(TrainEagle3Recipe, "_make_progress_bar", lambda self, **kwargs: fake_pbar)
    recipe = _build_recipe(tmp_path, num_samples=4, grad_accum=2)
    logged = []
    recipe.wandb_run = SimpleNamespace(log=lambda data, step: logged.append(data), finish=lambda: None)
    recipe.run_train_validation_loop()
    assert set(fake_pbar.postfix) == {"loss", "acc", "lr", "tau"}
    assert fake_pbar.postfix["tau"] == "1.75"
    assert logged and all(entry["train/tau_sim"] == pytest.approx(1.75) for entry in logged)


def test_train_loop_without_step_counts_omits_tau(tmp_path, monkeypatch):
    """P-EAGLE-style metrics (no step counts) must leave the log line unchanged."""
    fake_pbar = _FakeProgressBar()
    monkeypatch.setattr(TrainEagle3Recipe, "_make_progress_bar", lambda self, **kwargs: fake_pbar)
    recipe = _build_recipe(tmp_path, num_samples=4, grad_accum=2, with_step_counts=False)
    logged = []
    recipe.wandb_run = SimpleNamespace(log=lambda data, step: logged.append(data), finish=lambda: None)
    recipe.run_train_validation_loop()
    assert set(fake_pbar.postfix) == {"loss", "acc", "lr"}
    assert logged and all("train/tau_sim" not in entry for entry in logged)


def test_trailing_flush_reports_tau(tmp_path, monkeypatch):
    """The trailing partial-window flush must carry the tau column too."""
    fake_pbar = _FakeProgressBar()
    monkeypatch.setattr(TrainEagle3Recipe, "_make_progress_bar", lambda self, **kwargs: fake_pbar)
    # 3 samples / accum 2 -> one full window + one trailing flush.
    recipe = _build_recipe(tmp_path, num_samples=3, grad_accum=2)
    logged = []
    recipe.wandb_run = SimpleNamespace(log=lambda data, step: logged.append(data), finish=lambda: None)
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 2
    assert len(logged) == 2
    assert all(entry["train/tau_sim"] == pytest.approx(1.75) for entry in logged)


# ---------------------------------------------------------------------------
# _run_eval
# ---------------------------------------------------------------------------


def test_run_eval_returns_tau(tmp_path):
    recipe = _build_recipe(tmp_path, with_val=True)
    val_loss, val_acc, val_tau = recipe._run_eval()
    assert torch.isfinite(val_loss)
    assert val_acc.item() == pytest.approx(0.5)
    assert val_tau == pytest.approx(1.75)


def test_run_eval_tau_none_without_step_counts(tmp_path):
    recipe = _build_recipe(tmp_path, with_step_counts=False, with_val=True)
    val_loss, val_acc, val_tau = recipe._run_eval()
    assert torch.isfinite(val_loss)
    assert val_tau is None


def test_run_eval_none_without_val_loader(tmp_path):
    recipe = _build_recipe(tmp_path, with_val=False)
    assert recipe._run_eval() is None
