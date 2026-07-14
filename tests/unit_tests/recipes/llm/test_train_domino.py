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

"""Tests for the Domino recipe seams over the DFlash recipe.

The recipe is constructed via ``__new__`` (bypassing ``setup()``), so only the
overridden seams are exercised: the draft config extension, the trainer-module
swap, the lambda_base injection, and the extra-metrics logging.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.domino_core import DominoStepMetrics, DominoTrainerModule
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.recipes.llm.train_domino import TrainDominoRecipe

VOCAB = 64
HIDDEN = 32
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1
TARGET_LAYER_IDS = [1, 3, 5]


def _domino_draft(shift_label=True, pure_prefix=1):
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
        "projector_type": "domino",
        "emb_dim": 16,
        "gru_hidden_dim": 24,
        "pure_draft_prefix_len": pure_prefix,
        "shift_label": shift_label,
    }
    cfg._attn_implementation = "sdpa"
    return Qwen3DFlashDraftModel(cfg)


def _recipe():
    return TrainDominoRecipe.__new__(TrainDominoRecipe)


def test_build_dflash_config_adds_domino_fields():
    recipe = _recipe()
    recipe.mask_token_id = MASK_ID
    cfg = {"emb_dim": 128, "gru_hidden_dim": 512, "pure_draft_prefix_len": 2, "shift_label": False}
    out = recipe._build_dflash_config(cfg, TARGET_LAYER_IDS)
    assert out["mask_token_id"] == MASK_ID
    assert out["target_layer_ids"] == TARGET_LAYER_IDS
    assert out["projector_type"] == "domino"
    assert out["emb_dim"] == 128
    assert out["gru_hidden_dim"] == 512
    assert out["pure_draft_prefix_len"] == 2
    assert out["shift_label"] is False


def test_build_dflash_config_defaults():
    recipe = _recipe()
    recipe.mask_token_id = MASK_ID
    out = recipe._build_dflash_config({}, TARGET_LAYER_IDS)
    assert out["emb_dim"] == 256
    assert out["gru_hidden_dim"] == 1024
    assert out["pure_draft_prefix_len"] == 1
    assert out["shift_label"] is True


def test_build_trainer_module_is_domino():
    recipe = _recipe()
    recipe.draft_model = _domino_draft(shift_label=True)
    recipe.mask_token_id = MASK_ID
    recipe.block_size = BLOCK_SIZE
    recipe.target_model = SimpleNamespace(
        get_output_embeddings=lambda: torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        get_input_embeddings=lambda: torch.nn.Embedding(VOCAB, HIDDEN),
    )
    module = recipe._build_trainer_module("sdpa", {"num_anchors": 7, "loss_decay_gamma": 5.0})
    assert isinstance(module, DominoTrainerModule)
    assert module.num_anchors == 7
    assert module.loss_decay_gamma == 5.0
    assert module.shift_label is True


def test_build_trainer_module_defaults_loss_decay_gamma_to_paper_value():
    """Regression: an unset ``loss_decay_gamma`` used to fall back to ``None``
    (uniform weighting, decay silently disabled) instead of the paper default
    (Appendix A.1, matching ``DFlashDecayLoss``'s own default of 7.0)."""
    recipe = _recipe()
    recipe.draft_model = _domino_draft(shift_label=True)
    recipe.mask_token_id = MASK_ID
    recipe.block_size = BLOCK_SIZE
    recipe.target_model = SimpleNamespace(
        get_output_embeddings=lambda: torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        get_input_embeddings=lambda: torch.nn.Embedding(VOCAB, HIDDEN),
    )
    module = recipe._build_trainer_module("sdpa", {})
    assert module.loss_decay_gamma == 7.0


def test_run_trainer_step_injects_lambda_base():
    recipe = _recipe()
    recipe.runtime = SimpleNamespace(global_step=50)
    recipe.total_optim_steps = 100
    recipe.lambda_base_start = 1.0
    recipe.lambda_base_decay_ratio = 1.0

    seen = {}

    def _fake_module(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(loss=torch.tensor(1.0), accuracy=torch.tensor(0.5))

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
    # global_step 50 / (100 * 1.0) -> lambda_base 0.5.
    assert seen["lambda_base"] == 0.5
    assert recipe._last_domino_metrics is out


def test_log_extra_train_metrics(caplog):
    recipe = _recipe()
    recipe._last_domino_metrics = DominoStepMetrics(
        loss=torch.tensor(1.0),
        accuracy=torch.tensor(0.5),
        valid_tokens=torch.tensor(10.0),
        final_loss=torch.tensor(0.9),
        base_loss=torch.tensor(2.3),
        base_accuracy=torch.tensor(0.4),
        accept_len=torch.tensor(6.9),
        base_accept_len=torch.tensor(4.0),
        lambda_base=torch.tensor(0.5),
    )
    with caplog.at_level("INFO"):
        recipe._log_extra_train_metrics(epoch_idx=0)
    assert "domino:" in caplog.text
    assert "base_loss=2.3" in caplog.text


def test_log_extra_train_metrics_noop_without_metrics():
    recipe = _recipe()
    recipe._last_domino_metrics = None
    # Must not raise when no step has run yet.
    recipe._log_extra_train_metrics(epoch_idx=0)


def test_setup_reads_lambda_base_schedule(monkeypatch):
    recipe = _recipe()
    recipe.cfg = SimpleNamespace(recipe_args={"lambda_base_start": 0.8, "lambda_base_decay_ratio": 0.25})
    # Bypass the heavy DFlash setup (super().setup()); only the schedule read is under test.
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_dflash.TrainDFlashRecipe.setup", lambda self: None)
    recipe.setup()
    assert recipe.lambda_base_start == 0.8
    assert recipe.lambda_base_decay_ratio == 0.25
    assert recipe._last_domino_metrics is None


def test_main_runs_setup_then_loop(monkeypatch):
    from nemo_automodel.recipes.llm import train_domino

    calls = []
    monkeypatch.setattr(train_domino, "parse_args_and_load_config", lambda p: SimpleNamespace())
    monkeypatch.setattr(TrainDominoRecipe, "setup", lambda self: calls.append("setup"))
    monkeypatch.setattr(TrainDominoRecipe, "run_train_validation_loop", lambda self: calls.append("loop"))
    train_domino.main("cfg.yaml")
    assert calls == ["setup", "loop"]


def test_build_trainer_module_rejects_loss_type():
    """The DFlash loss_type knob must fail loudly here instead of being
    silently ignored (Domino has its own dual-logit objective)."""
    recipe = _recipe()
    recipe.draft_model = _domino_draft(shift_label=True)
    recipe.mask_token_id = MASK_ID
    with pytest.raises(ValueError, match="loss_type"):
        recipe._build_trainer_module("sdpa", {"loss_type": "variable_prefix"})
