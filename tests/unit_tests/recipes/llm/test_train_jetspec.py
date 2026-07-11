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

"""Tests for the JetSpec recipe seams over the DFlash recipe.

The recipe is constructed via ``__new__`` (bypassing ``setup()``), so only the
overridden seams are exercised: the target-wrapper logit capture, the
trainer-module swap, and the target-logit injection into the trainer step.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.components.speculative.dflash.jetspec_core import JetSpecTrainerModule
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe
from nemo_automodel.recipes.llm.train_jetspec import TrainJetSpecRecipe

VOCAB = 64
HIDDEN = 32
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1
TARGET_LAYER_IDS = [1, 3, 5]


def _draft():
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
    cfg.dflash_config = {"mask_token_id": MASK_ID, "target_layer_ids": TARGET_LAYER_IDS}
    cfg._attn_implementation = "sdpa"
    return Qwen3DFlashDraftModel(cfg)


class _FakeTarget(nn.Module):
    """Minimal frozen-target stand-in for the target-wrapper / trainer-module seams."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=8)
        self._lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self._embed = nn.Embedding(VOCAB, HIDDEN)

    def get_output_embeddings(self):
        return self._lm_head

    def get_input_embeddings(self):
        return self._embed


def _jetspec_recipe():
    recipe = TrainJetSpecRecipe.__new__(TrainJetSpecRecipe)
    recipe.target_model = _FakeTarget()
    recipe.draft_model = _draft()
    recipe.mask_token_id = MASK_ID
    recipe.block_size = BLOCK_SIZE
    return recipe


def test_base_target_wrapper_does_not_capture_logits():
    """The DFlash base seam leaves logit capture off (hard-label CE needs no teacher dist)."""
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.target_model = _FakeTarget()
    wrapper = recipe._build_target_wrapper(TARGET_LAYER_IDS)
    assert wrapper.capture_logits is False


def test_jetspec_target_wrapper_captures_logits():
    recipe = _jetspec_recipe()
    wrapper = recipe._build_target_wrapper(TARGET_LAYER_IDS)
    assert wrapper.capture_logits is True
    assert wrapper.target_layer_ids == TARGET_LAYER_IDS


def test_build_dflash_config_stamps_causal():
    """JetSpec stamps causal=true so serving engines match its causal in-block attention."""
    recipe = _jetspec_recipe()
    recipe.mask_token_id = 151669
    cfg = recipe._build_dflash_config({}, TARGET_LAYER_IDS)
    assert cfg["causal"] is True
    assert cfg["mask_token_id"] == 151669
    assert cfg["target_layer_ids"] == TARGET_LAYER_IDS


def test_build_trainer_module_is_jetspec():
    recipe = _jetspec_recipe()
    module = recipe._build_trainer_module("sdpa", {"num_anchors": 7, "kd_temperature": 2.0, "kd_chunk_size": 64})
    assert isinstance(module, JetSpecTrainerModule)
    assert module.num_anchors == 7
    assert module.kd_temperature == 2.0
    assert module.kd_chunk_size == 64


def test_build_trainer_module_defaults():
    recipe = _jetspec_recipe()
    module = recipe._build_trainer_module("sdpa", {})
    assert module.num_anchors == 512
    assert module.kd_temperature == 1.0
    assert module.kd_chunk_size == 0


def test_run_trainer_step_passes_target_logits():
    recipe = _jetspec_recipe()
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
        logits=torch.randn(1, 4, VOCAB),
    )
    recipe._run_trainer_step(target_batch)
    assert seen["target_logits"] is target_batch.logits
    assert seen["input_ids"] is target_batch.input_ids
    assert seen["hidden_states"] is target_batch.hidden_states
    assert seen["loss_mask"] is target_batch.loss_mask


def test_main_runs_setup_then_loop(monkeypatch):
    from nemo_automodel.recipes.llm import train_jetspec

    calls = []
    monkeypatch.setattr(train_jetspec, "parse_args_and_load_config", lambda p: SimpleNamespace())
    monkeypatch.setattr(TrainJetSpecRecipe, "setup", lambda self: calls.append("setup"))
    monkeypatch.setattr(TrainJetSpecRecipe, "run_train_validation_loop", lambda self: calls.append("loop"))
    train_jetspec.main("cfg.yaml")
    assert calls == ["setup", "loop"]
