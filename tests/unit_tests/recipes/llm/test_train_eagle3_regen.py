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

"""Recipe-side coverage for the EAGLE-3 on-policy regeneration integration:
``_setup_regen`` gating, ``_build_train_dataloader`` forwarding,
``_maybe_swap_regen_dataloader`` (rank-0 decide + rebind + LR-drift warning),
and the step-budget segment loop that swaps mid-epoch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nemo_automodel.components.speculative.eagle.target import Eagle3TargetBatch
from nemo_automodel.recipes.llm import train_eagle3 as te
from nemo_automodel.recipes.llm._spec_train_utils import optim_steps_per_epoch as _optim_steps_per_epoch
from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe


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


class _FakeTrainerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(4, 4)
        self.register_buffer("selected_token_ids", torch.arange(8))
        self.register_buffer("selected_token_mask", torch.ones(8, dtype=torch.bool))

    def forward(self, **kwargs):
        out = self.dummy(torch.randn(1, 4)).sum()
        return SimpleNamespace(loss=out.abs() + 1.0, accuracy=torch.tensor(0.5), valid_tokens=torch.tensor(4))


class _KeyedLoader:
    """DataLoader wrapper yielding the dict keys the training loop expects."""

    def __init__(self, num_samples, sl=6):
        data = TensorDataset(
            torch.randint(0, 64, (num_samples, sl)),
            torch.ones(num_samples, sl, dtype=torch.long),
            torch.ones(num_samples, sl, dtype=torch.long),
        )
        self._loader = DataLoader(data, batch_size=1)
        self.sampler = self._loader.sampler
        self._n = num_samples

    def __iter__(self):
        for ids, attn, lm in self._loader:
            yield {"input_ids": ids, "attention_mask": attn, "loss_mask": lm}

    def __len__(self):
        return self._n


def _bare_recipe(tmp_path, is_main=True):
    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe.output_dir = tmp_path
    recipe.dist_env = SimpleNamespace(is_main=is_main, world_size=1)
    return recipe


def _loop_recipe(tmp_path, num_samples=4, grad_accum=2, num_epochs=1):
    """A recipe wired to run the CPU training loop with regen hooks reachable."""
    recipe = _bare_recipe(tmp_path)
    tm = _FakeTrainerModule()
    recipe.device = torch.device("cpu")
    recipe.trainer_module = tm
    recipe.target_wrapper = _FakeTargetWrapper()
    recipe.cp_group = None
    recipe.target_prefetch_depth = 0
    recipe.train_dataloader = _KeyedLoader(num_samples)
    recipe.val_dataloader = None
    recipe.runtime = SimpleNamespace(global_step=0)
    recipe.grad_accumulation_steps = grad_accum
    recipe.max_grad_norm = 1.0
    recipe.num_epochs = num_epochs
    recipe.log_every_steps = 1
    recipe.peak_lr = 1e-4
    recipe.min_lr_ratio = 0.1
    recipe.warmup_steps = 1
    recipe.total_optim_steps = num_epochs * _optim_steps_per_epoch(num_samples, grad_accum)
    recipe._orig_batches_per_epoch = num_samples
    recipe._regen_enabled = False
    recipe.regen_runner = None
    recipe.optimizer = torch.optim.AdamW([p for p in tm.parameters() if p.requires_grad], lr=1e-4)
    recipe.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(recipe.optimizer, lambda s: 1.0)
    return recipe


# --------------------------------------------------------------------------- #
# _setup_regen gating
# --------------------------------------------------------------------------- #


def _regen_cfg(tmp_path, **regen_overrides):
    regen = {"every_steps": 10, "cuda_visible_devices": "7"}
    regen.update(regen_overrides)
    return {"regen": regen, "train_data_path": "/data/train", "output_dir": str(tmp_path)}


def test_setup_regen_disabled_leaves_runner_none(tmp_path):
    recipe = _bare_recipe(tmp_path)
    recipe._setup_regen({"output_dir": str(tmp_path)}, "/models/target")
    assert recipe.regen_runner is None
    assert recipe._regen_enabled is False


def test_setup_regen_does_not_read_unset_self_output_dir(tmp_path):
    # Regression: _setup_regen runs inside _setup_online_target, BEFORE the setup
    # body assigns self.output_dir, so it must derive output_dir from recipe_cfg.
    recipe = _bare_recipe(tmp_path)
    del recipe.output_dir  # _bare_recipe set it; prove _setup_regen doesn't need it
    recipe._setup_regen(_regen_cfg(tmp_path), "/models/target")
    assert recipe._regen_enabled is True
    assert recipe.regen_runner.config.output_dir == str(tmp_path / "regen")


def test_setup_regen_enabled_builds_runner_on_rank0(tmp_path):
    recipe = _bare_recipe(tmp_path)
    recipe._setup_regen(_regen_cfg(tmp_path), "/models/target")
    assert recipe._regen_enabled is True
    assert recipe.regen_runner is not None
    assert recipe.regen_runner.config.target_model == "/models/target"


def test_setup_regen_enabled_but_not_main_has_no_runner(tmp_path):
    recipe = _bare_recipe(tmp_path, is_main=False)
    recipe._setup_regen(_regen_cfg(tmp_path), "/models/target")
    # Every rank knows the feature is on (for the lockstep swap), only rank 0 owns the worker.
    assert recipe._regen_enabled is True
    assert recipe.regen_runner is None


def test_setup_regen_rejects_peft(tmp_path):
    recipe = _bare_recipe(tmp_path)
    recipe.peft_config = object()
    with pytest.raises(ValueError, match="not supported with peft"):
        recipe._setup_regen(_regen_cfg(tmp_path), "/models/target")


# --------------------------------------------------------------------------- #
# _build_train_dataloader forwards config to build_eagle3_dataloader
# --------------------------------------------------------------------------- #


def test_build_train_dataloader_forwards_config(tmp_path, monkeypatch):
    recipe = _bare_recipe(tmp_path)
    recipe.tokenizer = object()
    recipe.dp_mesh = object()
    recipe.dist_env = SimpleNamespace(is_main=True, world_size=4)
    recipe_args = SimpleNamespace(
        seq_length=512,
        micro_batch_size=2,
        get=lambda k, d=None: {
            "train_shuffle": False,
            "num_workers": 3,
            "shuffle_seed": 7,
            "mask_reasoning_content": True,
            "packed_sequence_size": 128,
        }.get(k, d),
    )
    recipe.cfg = SimpleNamespace(recipe_args=recipe_args)

    seen = {}
    monkeypatch.setattr(te, "build_eagle3_dataloader", lambda **kw: seen.update(kw) or "LOADER")

    out = recipe._build_train_dataloader("/regen/shards", split=None)
    assert out == "LOADER"
    assert seen["data_path"] == "/regen/shards"
    assert seen["split"] is None
    assert seen["seq_length"] == 512
    assert seen["batch_size"] == 2
    assert seen["shuffle"] is False
    assert seen["distributed"] is True  # world_size=4 > 1
    assert seen["packed_sequence_size"] == 128
    assert seen["tokenizer"] is recipe.tokenizer
    assert seen["dp_mesh"] is recipe.dp_mesh


# --------------------------------------------------------------------------- #
# _maybe_swap_regen_dataloader
# --------------------------------------------------------------------------- #


def test_swap_returns_false_when_disabled(tmp_path):
    recipe = _bare_recipe(tmp_path)
    recipe._regen_enabled = False
    assert recipe._maybe_swap_regen_dataloader() is False


def test_swap_returns_false_when_nothing_ready(tmp_path):
    recipe = _bare_recipe(tmp_path)
    recipe._regen_enabled = True
    recipe.regen_runner = SimpleNamespace(take_ready_shards=lambda: None)
    assert recipe._maybe_swap_regen_dataloader() is False


def test_swap_rebinds_and_warns_on_length_mismatch(tmp_path, monkeypatch, caplog):
    recipe = _bare_recipe(tmp_path)
    recipe._regen_enabled = True
    recipe.regen_runner = SimpleNamespace(take_ready_shards=lambda: "/regen/cycle_10/shards")
    recipe._orig_batches_per_epoch = 100
    new_loader = _KeyedLoader(40)  # different length than the original 100
    monkeypatch.setattr(recipe, "_build_train_dataloader", lambda data_path, split: new_loader)

    with caplog.at_level("WARNING"):
        swapped = recipe._maybe_swap_regen_dataloader()

    assert swapped is True
    assert recipe.train_dataloader is new_loader
    assert "40 batches vs the original 100" in caplog.text


def test_swap_no_warning_when_length_matches(tmp_path, monkeypatch, caplog):
    recipe = _bare_recipe(tmp_path)
    recipe._regen_enabled = True
    recipe.regen_runner = SimpleNamespace(take_ready_shards=lambda: "/regen/cycle_10/shards")
    recipe._orig_batches_per_epoch = 40
    monkeypatch.setattr(recipe, "_build_train_dataloader", lambda data_path, split: _KeyedLoader(40))

    with caplog.at_level("WARNING"):
        assert recipe._maybe_swap_regen_dataloader() is True
    assert "batches vs the original" not in caplog.text


# --------------------------------------------------------------------------- #
# Step-budget segment loop: mid-epoch swap trains on regen data and stops at budget
# --------------------------------------------------------------------------- #


def test_single_epoch_swaps_mid_run_and_stops_at_budget(tmp_path, monkeypatch):
    # num_samples=4, grad_accum=2 -> 2 optimizer steps per pass; num_epochs=1 -> budget 2.
    recipe = _loop_recipe(tmp_path, num_samples=4, grad_accum=2, num_epochs=1)
    recipe._regen_enabled = True

    swapped_dirs = []

    def fake_build(data_path, split):
        swapped_dirs.append(data_path)
        return _KeyedLoader(4)

    monkeypatch.setattr(recipe, "_build_train_dataloader", fake_build)
    # A cycle is ready exactly once, after the first optimizer step's swap check.
    ready = ["/regen/cycle_10/shards"]
    recipe.regen_runner = SimpleNamespace(
        maybe_launch=lambda step: False,
        take_ready_shards=lambda: ready.pop(0) if ready else None,
        shutdown=lambda: None,
    )

    recipe.run_train_validation_loop()

    # The budget (2 steps) is honored even though a mid-run swap started a fresh
    # segment, and the swap actually happened (so regen data was trained on) --
    # the single-epoch no-op is fixed.
    assert recipe.runtime.global_step == 2
    assert swapped_dirs == ["/regen/cycle_10/shards"]


def test_prefetch_drains_in_flight_requests_on_early_break(tmp_path):
    # A mid-epoch swap breaks the consumer loop; the prefetch generator must drain
    # (await) every dispatched-but-unconsumed request so the one-in-flight-per-server
    # recv ordering is not corrupted for the next segment.
    recipe = _bare_recipe(tmp_path)
    recipe.target_prefetch_depth = 2
    resolved = []
    dispatched = {"n": 0}

    class _Handle:
        def __init__(self, idx):
            self.idx = idx

        def result(self):
            resolved.append(self.idx)
            return f"target_{self.idx}"

    class _AsyncTargetWrapper:
        def generate_batch_async(self, input_ids, attention_mask, loss_mask):
            handle = _Handle(dispatched["n"])
            dispatched["n"] += 1
            return handle

    recipe.target_wrapper = _AsyncTargetWrapper()
    gen = recipe._prefetched_batches(_KeyedLoader(5))
    next(gen)  # consume one; the rest stay in flight up to the prefetch depth
    gen.close()  # the consumer stopped early (mid-epoch swap)

    # Every dispatched request was awaited exactly once -- none orphaned.
    assert sorted(resolved) == list(range(dispatched["n"]))
    assert dispatched["n"] >= 2  # depth-2 prefetch kept multiple in flight


def test_regen_disabled_run_is_unchanged(tmp_path):
    # Backward-compat: with regen off the step-budget guard is a no-op and the run
    # behaves exactly like the plain epoch loop (5 samples / accum 3 -> 2 steps).
    recipe = _loop_recipe(tmp_path, num_samples=5, grad_accum=3, num_epochs=1)
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 2
