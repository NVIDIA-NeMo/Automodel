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

"""Tests for the DFlash 2 recipe seams over the DFlash recipe.

The recipe is constructed via ``__new__`` (bypassing ``setup()``), so only the
overridden seams are exercised: the draft-class swap, the draft config extension,
the trainer-module swap, and the extra-metrics plumbing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.dflash2_core import (
    DFlash2StepMetrics,
    DFlash2TrainerModule,
)
from nemo_automodel.components.speculative.dflash.draft_qwen3_dflash2 import Qwen3DFlash2DraftModel
from nemo_automodel.components.speculative.dflash.registry import resolve_dflash_draft_spec
from nemo_automodel.recipes.llm.train_dflash2 import TrainDFlash2Recipe

VOCAB = 64
HIDDEN = 32
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1
TARGET_LAYER_IDS = [1, 3, 5]


def _dflash2_draft():
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
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
    cfg.block_size = BLOCK_SIZE
    cfg.dflash_config = {
        "mask_token_id": MASK_ID,
        "target_layer_ids": TARGET_LAYER_IDS,
        "conv_group_size": 8,
        "selector_rank": 16,
        "selector_top_k": 4,
    }
    cfg._attn_implementation = "sdpa"
    return Qwen3DFlash2DraftModel(cfg)


def _recipe():
    return TrainDFlash2Recipe.__new__(TrainDFlash2Recipe)


def _target_model():
    return SimpleNamespace(
        get_output_embeddings=lambda: torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        get_input_embeddings=lambda: torch.nn.Embedding(VOCAB, HIDDEN),
    )


def test_draft_cls_selects_the_dflash2_draft():
    """The recipe must build the DFlash 2 draft, and say so in ``architectures``.

    The DFlash recipe stamps the draft class name into the saved config, which is
    how a serving engine tells a DFlash checkpoint from a DFlash 2 one; picking
    the wrong class here would silently train a plain DFlash draft under a
    DFlash 2 config.
    """
    spec = resolve_dflash_draft_spec(["Qwen3ForCausalLM"])
    assert _recipe()._draft_cls(spec) is Qwen3DFlash2DraftModel
    assert Qwen3DFlash2DraftModel.__name__ == "Qwen3DFlash2DraftModel"


def test_build_dflash_config_adds_conv_and_selector_fields():
    recipe = _recipe()
    recipe.mask_token_id = MASK_ID
    cfg = {"conv_kernel_size": 3, "conv_group_size": 8, "selector_rank": 128, "selector_top_k": 8}
    out = recipe._build_dflash_config(cfg, TARGET_LAYER_IDS)
    assert out["mask_token_id"] == MASK_ID
    assert out["target_layer_ids"] == TARGET_LAYER_IDS
    assert out["conv_kernel_size"] == 3
    assert out["conv_group_size"] == 8
    assert out["selector_rank"] == 128
    assert out["selector_top_k"] == 8


def test_build_dflash_config_defaults_match_the_blog_shapes():
    recipe = _recipe()
    recipe.mask_token_id = MASK_ID
    out = recipe._build_dflash_config({}, TARGET_LAYER_IDS)
    assert out["conv_kernel_size"] == 2
    assert out["conv_group_size"] == 16
    assert out["selector_rank"] == 256
    assert out["selector_top_k"] == 16


def test_build_trainer_module_is_dflash2():
    recipe = _recipe()
    recipe.draft_model = _dflash2_draft()
    recipe.mask_token_id = MASK_ID
    recipe.block_size = BLOCK_SIZE
    recipe.target_model = _target_model()
    recipe.draft_sliding_window = 2048
    module = recipe._build_trainer_module(
        "sdpa", {"num_anchors": 7, "loss_decay_gamma": 5.0, "selector_loss_weight": 0.25}
    )
    assert isinstance(module, DFlash2TrainerModule)
    assert module.num_anchors == 7
    assert module.loss_decay_gamma == 5.0
    assert module.selector_loss_weight == 0.25
    # The window the recipe resolved must reach the trainer, or training and the
    # saved ``sliding_attention`` config would disagree.
    assert module.sliding_window == 2048


def test_build_trainer_module_defaults_loss_decay_gamma_to_paper_value():
    """An unset ``loss_decay_gamma`` must keep the DFlash paper default (7.0),
    not silently fall back to ``None`` (uniform weighting, decay disabled)."""
    recipe = _recipe()
    recipe.draft_model = _dflash2_draft()
    recipe.mask_token_id = MASK_ID
    recipe.block_size = BLOCK_SIZE
    recipe.target_model = _target_model()
    recipe.draft_sliding_window = None
    module = recipe._build_trainer_module("sdpa", {})
    assert module.loss_decay_gamma == 7.0
    assert module.selector_loss_weight == 1.0
    assert module.sliding_window is None


def test_build_trainer_module_rejects_loss_type():
    """The DFlash loss_type knob must fail loudly here instead of being silently
    ignored (DFlash 2 teacher-forces the selector from the fixed-anchor layout)."""
    recipe = _recipe()
    recipe.draft_model = _dflash2_draft()
    recipe.mask_token_id = MASK_ID
    with pytest.raises(ValueError, match="loss_type"):
        recipe._build_trainer_module("sdpa", {"loss_type": "variable_prefix"})


def _metrics(**overrides):
    fields = dict(
        loss=torch.tensor(1.0),
        loss_weight=torch.tensor(8.0),
        accuracy=torch.tensor(0.5),
        valid_tokens=torch.tensor(10.0),
        correct_tokens=torch.tensor(5.0),
        accept_len=torch.tensor(6.9),
        accept_len_sum=torch.tensor(13.8),
        valid_blocks=torch.tensor(2.0),
        base_loss=torch.tensor(2.3),
        selector_loss=torch.tensor(0.7),
        base_accuracy=torch.tensor(0.4),
        base_correct_tokens=torch.tensor(4.0),
        base_accept_len=torch.tensor(4.0),
        base_accept_len_sum=torch.tensor(8.0),
        candidate_recall=torch.tensor(0.9),
    )
    fields.update(overrides)
    return DFlash2StepMetrics(**fields)


def test_log_extra_train_metrics(caplog):
    recipe = _recipe()
    recipe._last_dflash2_metrics = _metrics()
    with caplog.at_level("INFO"):
        recipe._log_extra_train_metrics(epoch_idx=0)
    assert "dflash2:" in caplog.text
    assert "base_loss=2.3" in caplog.text
    assert "selector_loss=0.7" in caplog.text


def test_log_extra_train_metrics_noop_without_metrics():
    recipe = _recipe()
    recipe._last_dflash2_metrics = None
    # Must not raise when no step has run yet.
    recipe._log_extra_train_metrics(epoch_idx=0)


def test_train_sums_include_selector_diagnostics():
    recipe = _recipe()

    values = recipe._extra_train_metric_sums(_metrics())

    # Every entry is (numerator, denominator) so the caller can average it over
    # the same window as train/loss instead of reporting one micro-batch.
    assert values == {
        "train/accept_len": (pytest.approx(13.8), pytest.approx(2.0)),
        "train/base_loss": (pytest.approx(2.3), 1.0),
        "train/selector_loss": (pytest.approx(0.7), 1.0),
        "train/base_accuracy": (pytest.approx(4.0), pytest.approx(10.0)),
        "train/base_accept_len": (pytest.approx(8.0), pytest.approx(2.0)),
        "train/candidate_recall": (pytest.approx(9.0), pytest.approx(10.0)),
    }


def test_eval_sums_keep_additive_numerators():
    recipe = _recipe()

    sums = recipe._extra_eval_metric_sums(_metrics())

    assert sums["val_base_loss"][0].item() == pytest.approx(18.4)
    assert sums["val_base_loss"][1].item() == 8.0
    assert sums["val_selector_loss"][0].item() == pytest.approx(5.6)
    assert sums["val_base_accuracy"][0].item() == 4.0
    assert sums["val_base_accept_len"][0].item() == 8.0
    assert sums["val_base_accept_len"][1].item() == 2.0
    assert sums["val_candidate_recall"][0].item() == pytest.approx(9.0)


def test_empty_eval_sums_have_stable_collective_order():
    recipe = _recipe()
    recipe.device = torch.device("cpu")

    sums = recipe._empty_extra_eval_metric_sums()

    assert list(sums) == list(_recipe()._extra_eval_metric_sums(_metrics()))
    assert all(numerator.item() == 0.0 and denominator.item() == 0.0 for numerator, denominator in sums.values())


def test_run_trainer_step_caches_the_metrics(monkeypatch):
    recipe = _recipe()
    seen = {}

    def _fake_module(**kwargs):
        seen.update(kwargs)
        return _metrics()

    recipe.trainer_module = _fake_module
    target_batch = SimpleNamespace(
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        hidden_states=torch.zeros(1, 4, 8),
        loss_mask=torch.ones(1, 4),
        position_ids=None,
        seq_lens=None,
        doc_remaining=None,
    )
    out = recipe._run_trainer_step(target_batch)
    assert seen["input_ids"] is target_batch.input_ids
    assert recipe._last_dflash2_metrics is out


def test_setup_resets_the_metrics_cache(monkeypatch):
    recipe = _recipe()
    # Bypass the heavy DFlash setup (super().setup()); only the reset is under test.
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_dflash.TrainDFlashRecipe.setup", lambda self: None)
    recipe.setup()
    assert recipe._last_dflash2_metrics is None


def test_main_runs_setup_then_loop(monkeypatch):
    from nemo_automodel.recipes.llm import train_dflash2

    calls = []
    monkeypatch.setattr(train_dflash2, "parse_args_and_load_config", lambda p: SimpleNamespace())
    monkeypatch.setattr(TrainDFlash2Recipe, "setup", lambda self: calls.append("setup"))
    monkeypatch.setattr(TrainDFlash2Recipe, "run_train_validation_loop", lambda self: calls.append("loop"))
    train_dflash2.main("cfg.yaml")
    assert calls == ["setup", "loop"]
