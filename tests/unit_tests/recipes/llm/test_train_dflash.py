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

import logging
import pathlib
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.core import DFlashTrainerModule, NoValidAnchorsError
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.recipes.llm import train_dflash
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe
from nemo_automodel.recipes.llm.train_domino import TrainDominoRecipe

_VOCAB = 64
_HIDDEN = 32
_MASK_ID = _VOCAB - 1
_TARGET_LAYER_IDS = [1, 3, 5]


def _dflash_draft():
    cfg = Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    cfg.num_target_layers = 8
    cfg.block_size = 4
    cfg.dflash_config = {"mask_token_id": _MASK_ID, "target_layer_ids": _TARGET_LAYER_IDS}
    cfg._attn_implementation = "sdpa"
    return Qwen3DFlashDraftModel(cfg)


def _bare_dflash_recipe():
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.draft_model = _dflash_draft()
    recipe.mask_token_id = _MASK_ID
    recipe.block_size = 4
    recipe.target_model = SimpleNamespace(
        get_output_embeddings=lambda: torch.nn.Linear(_HIDDEN, _VOCAB, bias=False),
        get_input_embeddings=lambda: torch.nn.Embedding(_VOCAB, _HIDDEN),
    )
    return recipe


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


def test_save_checkpoint_applies_retention_after_sync_save(tmp_path):
    events = []
    obj = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    obj.checkpoint_config = SimpleNamespace(checkpoint_dir=str(tmp_path))
    obj.checkpointer = SimpleNamespace(
        config=SimpleNamespace(enabled=True, is_async=False),
        async_wait=lambda: events.append("wait"),
        save_model=lambda *args, **kwargs: events.append("save_model"),
        save_optimizer=lambda *args, **kwargs: events.append("save_optimizer"),
        save_on_dp_ranks=lambda *args, **kwargs: events.append("save_rng"),
    )
    obj._module = lambda: SimpleNamespace(draft_model=SimpleNamespace())
    obj.tokenizer = None
    obj.optimizer = SimpleNamespace()
    obj.lr_scheduler = SimpleNamespace()
    obj.rng = SimpleNamespace()
    obj.runtime = SimpleNamespace(global_step=1)
    obj.cfg = SimpleNamespace(raw_config={})
    obj.block_size = 4
    obj.mask_token_id = 99
    obj.target_wrapper = SimpleNamespace(target_layer_ids=[0, 1])
    obj._complete_pending_checkpoint = lambda: events.append("complete_pending")
    obj._update_latest_symlink = lambda path: events.append(("latest", path))
    obj._update_best_symlink = lambda path, val, metric_key=None: events.append(("best", path, val, metric_key))
    obj._prune_old_checkpoints = lambda: events.append("prune")

    TrainDFlashRecipe.save_checkpoint(obj, epoch=0, step=1, val_loss={"val_loss": 0.25})

    assert events[0:2] == ["wait", "complete_pending"]
    assert any(event[0] == "best" and event[3] == "val_loss" for event in events if isinstance(event, tuple))
    assert "prune" in events


def test_save_checkpoint_records_async_best_pending_info_without_metric(tmp_path):
    events = []
    obj = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    obj.checkpoint_config = SimpleNamespace(checkpoint_dir=str(tmp_path))
    obj.checkpointer = SimpleNamespace(
        config=SimpleNamespace(enabled=True, is_async=True),
        async_wait=lambda: events.append("wait"),
        save_model=lambda *args, **kwargs: events.append("save_model"),
        save_optimizer=lambda *args, **kwargs: events.append("save_optimizer"),
        save_on_dp_ranks=lambda *args, **kwargs: events.append("save_rng"),
    )
    obj._module = lambda: SimpleNamespace(draft_model=SimpleNamespace())
    obj.tokenizer = None
    obj.optimizer = SimpleNamespace()
    obj.lr_scheduler = SimpleNamespace()
    obj.rng = SimpleNamespace()
    obj.runtime = SimpleNamespace(global_step=1)
    obj.cfg = SimpleNamespace(raw_config={})
    obj.block_size = 4
    obj.mask_token_id = 99
    obj.target_wrapper = SimpleNamespace(target_layer_ids=[0, 1])
    obj._complete_pending_checkpoint = lambda: events.append("complete_pending")

    TrainDFlashRecipe.save_checkpoint(obj, epoch=0, step=1, best_metric_key="val_loss")

    expected_path = str(tmp_path / "epoch_0_step_1")
    assert events[0:2] == ["wait", "complete_pending"]
    assert obj._last_pending_checkpoint_dir == expected_path
    assert obj._last_pending_best_checkpoint_info == {
        "path": expected_path,
        "val": None,
        "metric_key": "val_loss",
    }


def test_build_checkpointer_logs_retention_policy(tmp_path, monkeypatch, caplog):
    built = []

    class FakeCheckpointer:
        def __init__(self, config, **kwargs):
            self.config = config
            built.append((config, kwargs))

    monkeypatch.setattr(train_dflash, "Checkpointer", FakeCheckpointer)
    obj = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    obj.cfg = SimpleNamespace(
        get=lambda key, default=None: {"checkpoint_dir": str(tmp_path), "max_recent_checkpoints": 1}
        if key == "checkpoint"
        else default
    )
    obj.output_dir = tmp_path
    obj.draft_model = SimpleNamespace(state_dict=lambda: {"weight": torch.zeros(1)})
    obj.dp_mesh = None

    with caplog.at_level(logging.INFO):
        TrainDFlashRecipe._build_checkpointer(obj, "target/repo")

    assert built
    assert "Checkpoint retention: keeping the most recent 1 checkpoint(s)" in caplog.text


def test_run_train_validation_loop_finalizes_before_close():
    events = []

    class FakePbar:
        def close(self):
            events.append("pbar_close")

    obj = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    obj.trainer_module = SimpleNamespace(train=lambda: None)
    obj.num_epochs = 1
    obj._resume_epoch = 0
    obj.dist_env = SimpleNamespace(is_main=False)
    obj.total_optim_steps = 1
    obj.runtime = SimpleNamespace(global_step=1)
    obj.train_dataloader = []
    obj._make_progress_bar = lambda **kwargs: FakePbar()
    obj._run_eval = lambda: None
    obj._maybe_save_final_checkpoint = lambda completed_epochs: events.append(("final", completed_epochs)) or True
    obj._finalize_pending_checkpoint = lambda: events.append("finalize")
    obj.checkpointer = SimpleNamespace(close=lambda: events.append("close"))

    TrainDFlashRecipe.run_train_validation_loop(obj)

    assert events == [("final", 1), "finalize", "close", "pbar_close"]


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


class _StubTrainerModule:
    """Callable trainer stub: yields one queued result (or raises it) per batch."""

    def __init__(self, results):
        self._results = list(results)
        self.mode_calls = []

    def eval(self):
        self.mode_calls.append("eval")

    def train(self):
        self.mode_calls.append("train")

    def __call__(self, **kwargs):
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _eval_self(trainer, num_batches):
    batch = {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "loss_mask": torch.ones(1, 4, dtype=torch.long),
    }
    obj = SimpleNamespace(
        val_dataloader=[dict(batch) for _ in range(num_batches)],
        device=torch.device("cpu"),
        trainer_module=trainer,
        target_wrapper=SimpleNamespace(
            generate_batch=lambda **kw: SimpleNamespace(
                input_ids=kw["input_ids"],
                hidden_states=kw["input_ids"],
                loss_mask=kw["loss_mask"],
                position_ids=None,
                seq_lens=None,
                doc_remaining=None,
            )
        ),
    )
    # _run_eval routes the trainer call through the same _run_trainer_step seam as
    # training (so subclasses inject their extra inputs); bind the base seam, which
    # forwards (input_ids, hidden_states, loss_mask) to the trainer module.
    obj._run_trainer_step = TrainDFlashRecipe._run_trainer_step.__get__(obj)
    obj._extra_eval_metric_sums = TrainDFlashRecipe._extra_eval_metric_sums.__get__(obj)
    obj._empty_extra_eval_metric_sums = TrainDFlashRecipe._empty_extra_eval_metric_sums.__get__(obj)
    return obj


def _distributed_domino_eval_worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    """Run uneven Domino validation on one CPU process.

    Rank 0 contributes no valid batch while rank 1 contributes scalar step
    statistics. Both ranks must enter the same Gloo collectives and produce the
    same globally reduced result.
    """
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        batch = {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "loss_mask": torch.ones(1, 4),
        }
        recipe = TrainDominoRecipe.__new__(TrainDominoRecipe)
        recipe.device = torch.device("cpu")
        recipe.val_dataloader = [batch]
        recipe.trainer_module = SimpleNamespace(eval=lambda: None, train=lambda: None)
        recipe.target_wrapper = SimpleNamespace(generate_batch=lambda **kwargs: SimpleNamespace(**kwargs))
        result = (
            NoValidAnchorsError("rank has no valid anchors")
            if rank == 0
            else SimpleNamespace(
                loss=torch.tensor(2.0),
                loss_weight=torch.tensor(4.0),
                accuracy=torch.tensor(0.8),
                valid_tokens=torch.tensor(5.0),
                correct_tokens=torch.tensor(4.0),
                accept_len_sum=torch.tensor(6.0),
                valid_blocks=torch.tensor(2.0),
                final_loss=torch.tensor(1.5),
                base_loss=torch.tensor(3.0),
                base_correct_tokens=torch.tensor(2.0),
                base_accept_len_sum=torch.tensor(4.0),
            )
        )

        def _run_step(_target_batch):
            if isinstance(result, Exception):
                raise result
            return result

        recipe._run_trainer_step = _run_step
        metrics = recipe._run_eval()
        torch.save(metrics, pathlib.Path(output_dir) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_run_eval_returns_none_without_val_dataloader():
    obj = SimpleNamespace(val_dataloader=None)
    assert TrainDFlashRecipe._run_eval(obj) is None


def test_run_eval_skips_short_micro_batches():
    trainer = _StubTrainerModule(
        [
            SimpleNamespace(
                loss=torch.tensor(1.0),
                loss_weight=torch.tensor(2.0),
                accuracy=torch.tensor(0.5),
                valid_tokens=torch.tensor(2.0),
                correct_tokens=torch.tensor(1.0),
                accept_len_sum=torch.tensor(4.0),
                valid_blocks=torch.tensor(2.0),
            ),
            NoValidAnchorsError("all samples too short"),
            SimpleNamespace(
                loss=torch.tensor(3.0),
                loss_weight=torch.tensor(6.0),
                accuracy=torch.tensor(1.0),
                valid_tokens=torch.tensor(6.0),
                correct_tokens=torch.tensor(6.0),
                accept_len_sum=torch.tensor(3.0),
                valid_blocks=torch.tensor(1.0),
            ),
        ]
    )
    obj = _eval_self(trainer, num_batches=3)

    metrics = TrainDFlashRecipe._run_eval(obj)

    # The short batch is skipped. The two valid batches are weighted by their
    # token/block counts instead of receiving equal batch weight.
    assert metrics == {
        "val_loss": 2.5,
        "val_accuracy": 0.875,
        "val_accept_len": pytest.approx(7.0 / 3.0),
    }
    assert trainer.mode_calls == ["eval", "train"]


def test_run_eval_all_batches_short_returns_zero_metrics():
    trainer = _StubTrainerModule([NoValidAnchorsError("short"), NoValidAnchorsError("short")])
    obj = _eval_self(trainer, num_batches=2)

    metrics = TrainDFlashRecipe._run_eval(obj)

    assert metrics == {"val_loss": 0.0, "val_accuracy": 0.0, "val_accept_len": 0.0}
    assert trainer.mode_calls == ["eval", "train"]


def test_distributed_domino_eval_handles_rank_with_no_valid_batches(tmp_path):
    init_file = tmp_path / "gloo_init"
    mp.spawn(
        _distributed_domino_eval_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    expected = {
        "val_loss": 2.0,
        "val_accuracy": pytest.approx(0.8),
        "val_accept_len": 3.0,
        "val_final_loss": 1.5,
        "val_base_loss": 3.0,
        "val_base_accuracy": pytest.approx(0.4),
        "val_base_accept_len": 2.0,
    }
    assert torch.load(tmp_path / "rank_0.pt", weights_only=True) == expected
    assert torch.load(tmp_path / "rank_1.pt", weights_only=True) == expected


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


def test_all_reduce_sum_uses_additive_distributed_statistics(monkeypatch):
    monkeypatch.setattr(train_dflash.dist, "is_available", lambda: True)
    monkeypatch.setattr(train_dflash.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_dflash.dist, "all_reduce", lambda value, op=None: value.add_(2.0))

    value = torch.tensor(3.0)
    assert train_dflash._all_reduce_sum(value).item() == 5.0


def test_wandb_log_forwards_metrics_when_run_exists():
    calls = []
    obj = SimpleNamespace(wandb_run=SimpleNamespace(log=lambda data, step: calls.append((data, step))))

    TrainDFlashRecipe._wandb_log(obj, {"val/accuracy": 0.75}, step=9)

    assert calls == [({"val/accuracy": 0.75}, 9)]


def test_wandb_log_is_noop_without_run():
    TrainDFlashRecipe._wandb_log(SimpleNamespace(), {"val/accuracy": 0.75}, step=9)


def _load_extra_state_self(mask_token_id=7):
    return SimpleNamespace(
        runtime=SimpleNamespace(global_step=0),
        _resume_epoch=0,
        mask_token_id=mask_token_id,
    )


def _write_meta(tmp_path, **fields):
    meta = {"global_step": 5, "epoch": 2, "block_size": 16, "target_layer_ids": [1, 2]}
    meta.update(fields)
    torch.save(meta, tmp_path / "dflash_meta.pt")
    return str(tmp_path)


def test_load_extra_state_restores_step_and_epoch(tmp_path):
    ckpt_dir = _write_meta(tmp_path, mask_token_id=7)
    obj = _load_extra_state_self(mask_token_id=7)
    TrainDFlashRecipe._load_extra_state(obj, ckpt_dir)
    assert obj.runtime.global_step == 5
    assert obj._resume_epoch == 2


def test_load_extra_state_raises_on_mask_token_id_mismatch(tmp_path):
    """A resume YAML whose mask_token_id disagrees with the checkpoint must fail loudly."""
    ckpt_dir = _write_meta(tmp_path, mask_token_id=7)
    obj = _load_extra_state_self(mask_token_id=99)
    with pytest.raises(ValueError, match="mask_token_id mismatch on resume"):
        TrainDFlashRecipe._load_extra_state(obj, ckpt_dir)


def test_load_extra_state_accepts_legacy_meta_without_mask_token_id(tmp_path):
    """Checkpoints saved before mask_token_id was persisted skip the check."""
    meta = {"global_step": 3, "epoch": 1}
    torch.save(meta, tmp_path / "dflash_meta.pt")
    obj = _load_extra_state_self(mask_token_id=99)
    TrainDFlashRecipe._load_extra_state(obj, str(tmp_path))
    assert obj.runtime.global_step == 3
    assert obj._resume_epoch == 1


def test_load_extra_state_noop_when_meta_missing(tmp_path):
    obj = _load_extra_state_self(mask_token_id=7)
    TrainDFlashRecipe._load_extra_state(obj, str(tmp_path))
    assert obj.runtime.global_step == 0
    assert obj._resume_epoch == 0


def test_build_trainer_module_defaults_loss_decay_gamma_to_paper_value():
    """Regression: an unset ``loss_decay_gamma`` used to fall back to ``None``
    (uniform weighting, decay silently disabled) instead of the paper default
    (Appendix A.1, matching ``DFlashDecayLoss``'s own default of 7.0)."""
    recipe = _bare_dflash_recipe()
    module = recipe._build_trainer_module("sdpa", {})
    assert isinstance(module, DFlashTrainerModule)
    assert module.loss_decay_gamma == 7.0


def test_build_trainer_module_respects_explicit_loss_decay_gamma():
    recipe = _bare_dflash_recipe()
    module = recipe._build_trainer_module("sdpa", {"loss_decay_gamma": None})
    assert module.loss_decay_gamma is None

    recipe = _bare_dflash_recipe()
    module = recipe._build_trainer_module("sdpa", {"loss_decay_gamma": 4.0})
    assert module.loss_decay_gamma == 4.0


def test_build_trainer_module_wires_loss_type_and_prefix_weight_base():
    recipe = _bare_dflash_recipe()
    module = recipe._build_trainer_module("sdpa", {})
    assert module.loss_type == "dflash"
    assert module.prefix_weight_base == 0.9

    recipe = _bare_dflash_recipe()
    module = recipe._build_trainer_module("sdpa", {"loss_type": "variable_prefix", "prefix_weight_base": 0.8})
    assert module.loss_type == "variable_prefix"
    assert module.prefix_weight_base == 0.8


def test_build_trainer_module_loss_type_null_falls_back_to_default():
    """An explicit ``loss_type: null`` in YAML must select the default objective
    (mirroring loss_decay_gamma's documented null convention), not crash on the
    string "None"."""
    recipe = _bare_dflash_recipe()
    module = recipe._build_trainer_module("sdpa", {"loss_type": None})
    assert module.loss_type == "dflash"
