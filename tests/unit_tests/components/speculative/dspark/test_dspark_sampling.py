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

"""Tests for the sampling helpers and the Markov-head inference paths (CPU)."""

import torch

from nemo_automodel.components.speculative.dspark import _sampling
from nemo_automodel.components.speculative.dspark.markov_head import (
    GatedMarkovHead,
    RNNHead,
    VanillaMarkov,
)

VOCAB = 32
RANK = 8
HIDDEN = 16


def test_sampling_helpers():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, VOCAB)
    # Greedy (temperature 0) equals argmax.
    assert torch.equal(_sampling.sample_tokens(logits, temperature=0.0), logits.argmax(dim=-1))
    # Stochastic path returns in-range token ids.
    sampled = _sampling.sample_tokens(logits, temperature=1.0)
    assert sampled.shape == (2, 3) and int(sampled.min()) >= 0 and int(sampled.max()) < VOCAB

    probs = _sampling.logits_to_probs(logits, temperature=1.0)
    assert torch.allclose(probs.sum(-1), torch.ones(2, 3), atol=1e-5)
    # temperature 0 -> one-hot at the argmax.
    hard = _sampling.logits_to_probs(logits, temperature=0.0)
    assert torch.equal(hard.argmax(-1), logits.argmax(-1))

    flat = _sampling.sample_from_probs(probs)
    assert flat.shape == (2, 3)
    token_ids = torch.randint(0, VOCAB, (2, 3))
    gathered = _sampling.gather_token_probs(probs, token_ids)
    assert gathered.shape == (2, 3)
    residual = _sampling.sample_residual(probs[:, 0], probs[:, 0] * 0.5)
    assert residual.shape == (2,)


def test_vanilla_markov_inference():
    head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK)
    base_logits = torch.randn(2, 4, VOCAB)
    prev = torch.randint(0, VOCAB, (2,))
    tokens, corrected = head.sample_block_tokens(
        base_logits, first_prev_token_ids=prev, hidden_states=None, temperature=0.0
    )
    assert tokens.shape == (2, 4) and corrected.shape == (2, 4, VOCAB)
    step = head.apply_step_logits(base_logits[:, 0], token_ids=prev, hidden_states=None)
    assert step.shape == (2, VOCAB)


def test_gated_and_rnn_markov_block_inference():
    for cls in (GatedMarkovHead, RNNHead):
        head = cls(vocab_size=VOCAB, markov_rank=RANK, hidden_size=HIDDEN)
        base = torch.randn(2, 3, 4, VOCAB)  # [B, num_blocks, block_size, V]
        token_ids = torch.randint(0, VOCAB, (2, 3, 4))
        hidden = torch.randn(2, 3, 4, HIDDEN)
        biased = head.apply_block_logits(base, token_ids=token_ids, hidden_states=hidden)
        assert biased.shape == base.shape

        prev = torch.randint(0, VOCAB, (2,))
        seq_hidden = torch.randn(2, 4, HIDDEN)
        tokens, corrected = head.sample_block_tokens(
            torch.randn(2, 4, VOCAB), first_prev_token_ids=prev, hidden_states=seq_hidden, temperature=1.0
        )
        assert tokens.shape == (2, 4) and corrected.shape == (2, 4, VOCAB)
