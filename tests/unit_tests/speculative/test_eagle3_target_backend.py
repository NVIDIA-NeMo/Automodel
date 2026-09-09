# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the EAGLE-3 target-model backend abstraction.

Covers the seam a remote target backend will plug into:

1. ``HFEagle3TargetModel`` satisfies the ``Eagle3TargetBackend`` contract and
   the base-class defaults (synchronous, no-op vocab mapping / close) behave.
2. ``Eagle3TargetBatch`` accepts exactly one supervision encoding and
   ``to_trainer_inputs`` dispatches it to the right trainer kwargs.
3. The two supervision encodings are numerically equivalent through the
   trainer: feeding full ``logits`` (co-located path) and feeding the
   precomputed ``target_probs`` / ``position_mask`` (the encoding a remote
   backend ships over the wire) produce a bit-identical loss. This is the
   guarantee that makes remote training behavior-preserving.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.modeling_outputs import CausalLMOutput

from nemo_automodel.components.speculative.eagle.backend import Eagle3TargetBackend
from nemo_automodel.components.speculative.eagle.core import Eagle3TrainerModule, _compute_target_distribution
from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.target import Eagle3TargetBatch, HFEagle3TargetModel


class _FakeHFCausalLM(nn.Module):
    """Minimal HF causal-LM stand-in returning ``CausalLMOutput`` with ``.logits``."""

    def __init__(self, num_layers: int = 4, hidden: int = 16, vocab: int = 32) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": num_layers})
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, input_ids, attention_mask=None, **kwargs):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return CausalLMOutput(logits=self.lm_head(h))


def _build_tiny_draft_model() -> LlamaEagle3DraftModel:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    config.torch_dtype = torch.float32
    config.draft_vocab_size = 16
    config.target_hidden_size = 32
    return LlamaEagle3DraftModel(config).to(torch.float32)


# --- 1. backend contract ---------------------------------------------------


def test_hf_eagle3_target_is_a_backend() -> None:
    target = HFEagle3TargetModel(_FakeHFCausalLM(num_layers=4), aux_layer_ids=[0, 1, 3])
    assert isinstance(target, Eagle3TargetBackend)
    # Co-located target is synchronous; prefetch is a remote-only capability.
    assert target.supports_async is False
    with pytest.raises(NotImplementedError):
        target.generate_batch_async(
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.long),
        )
    # Base-class defaults are no-ops for the co-located backend.
    assert target.set_vocab_mapping(torch.arange(4), torch.ones(4, dtype=torch.bool)) is None
    assert target.close() is None


def test_hf_eagle3_target_emits_logits_supervision() -> None:
    target = HFEagle3TargetModel(_FakeHFCausalLM(num_layers=4, vocab=32), aux_layer_ids=[0, 1, 3])
    input_ids = torch.randint(0, 32, (2, 8))
    attn = torch.ones(2, 8, dtype=torch.long)
    loss = torch.ones(2, 8, dtype=torch.long)
    batch = target.generate_batch(input_ids, attn, loss)
    # The co-located backend carries full logits, not precomputed probs.
    assert batch.logits is not None
    assert batch.target_probs is None and batch.position_mask is None
    assert "target_logits" in batch.to_trainer_inputs()


# --- 2. Eagle3TargetBatch supervision encodings ----------------------------


def _dummy_seq_tensors(batch: int = 2, seq: int = 8, hidden: int = 32):
    return {
        "aux_hidden_states": torch.randn(batch, seq, hidden * 3),
        "input_ids": torch.randint(0, 16, (batch, seq)),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        "loss_mask": torch.ones(batch, seq, dtype=torch.long),
    }


def test_target_batch_rejects_both_or_neither_supervision_source() -> None:
    base = _dummy_seq_tensors()
    # Neither.
    with pytest.raises(ValueError, match="exactly one supervision source"):
        Eagle3TargetBatch(**base)
    # Both.
    with pytest.raises(ValueError, match="exactly one supervision source"):
        Eagle3TargetBatch(
            **base,
            logits=torch.randn(2, 8, 128),
            target_probs=torch.rand(2, 8, 16),
            position_mask=torch.ones(2, 8, 1, dtype=torch.bool),
        )


def test_target_batch_to_trainer_inputs_dispatches_each_encoding() -> None:
    base = _dummy_seq_tensors()
    logits_batch = Eagle3TargetBatch(**base, logits=torch.randn(2, 8, 128))
    assert set(logits_batch.to_trainer_inputs()) == {
        "input_ids",
        "attention_mask",
        "loss_mask",
        "aux_hidden_states",
        "target_logits",
    }
    precomputed_batch = Eagle3TargetBatch(
        **base,
        target_probs=torch.rand(2, 8, 16),
        position_mask=torch.ones(2, 8, 1, dtype=torch.bool),
    )
    assert set(precomputed_batch.to_trainer_inputs()) == {
        "input_ids",
        "attention_mask",
        "loss_mask",
        "aux_hidden_states",
        "target_probs",
        "position_mask",
    }


# --- 3. logits vs precomputed equivalence through the trainer --------------


def test_logits_and_precomputed_paths_produce_identical_loss() -> None:
    """The encoding a remote backend ships (precomputed target_probs /
    position_mask) must drive the trainer identically to the co-located
    full-logits path. Otherwise disaggregation would silently change loss."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True

    trainer = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=3,
    )

    batch_size, seq_len = 2, 8
    common = dict(
        input_ids=torch.randint(0, config.draft_vocab_size, (batch_size, seq_len)),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        loss_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        aux_hidden_states=torch.randn(batch_size, seq_len, config.hidden_size * 3),
    )
    target_logits = torch.randn(batch_size, seq_len, config.vocab_size)

    # Co-located encoding: hand the trainer the full logits.
    logits_batch = Eagle3TargetBatch(**common, logits=target_logits)
    torch.manual_seed(1)
    live = trainer(**logits_batch.to_trainer_inputs())

    # Remote encoding: precompute the draft-vocab distribution the way a
    # server would (reusing the exact same projection), then hand that over.
    target_probs, position_mask = _compute_target_distribution(
        target_logits=target_logits,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        loss_mask=common["loss_mask"],
    )
    precomputed_batch = Eagle3TargetBatch(**common, target_probs=target_probs, position_mask=position_mask)
    torch.manual_seed(1)
    remote = trainer(**precomputed_batch.to_trainer_inputs())

    torch.testing.assert_close(live.loss, remote.loss, rtol=0, atol=0)
    torch.testing.assert_close(live.accuracy, remote.accuracy, rtol=0, atol=0)
