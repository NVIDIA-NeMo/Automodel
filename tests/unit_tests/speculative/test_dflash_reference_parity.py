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

"""Parity with the reference DFlash decode path (github.com/z-lab/dflash).

The reference is not a dependency, so each test carries its own small
transcription of the behaviour being matched rather than importing it. Covered
here: the truncated sampling distribution, the target output transform, and the
accessors the decode path resolves the target through.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.draft_qwen3 import (
    Qwen3DFlashDraftModel,
    resolve_output_head,
    sample,
    sampling_probs,
    validate_sampling,
)

VOCAB = 64
HIDDEN = 32


def _reference_sampling_probs(logits, temperature, top_p=1.0, top_k=0):
    """Transcription of the reference ``_sampling_probs`` (dflash/model.py:46)."""
    scores = logits.float() / temperature
    vocab_size = scores.shape[-1]
    if 0 < top_k < vocab_size:
        scores, indices = torch.topk(scores, top_k, dim=-1)
    else:
        indices = None
    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if indices is not None:
        probs = torch.zeros_like(logits, dtype=probs.dtype).scatter(-1, indices, probs)
    return probs


@pytest.mark.parametrize(
    "temperature,top_p,top_k",
    [(1.0, 1.0, 0), (0.7, 1.0, 0), (1.0, 0.95, 0), (1.0, 1.0, 20), (1.0, 0.95, 20), (0.5, 0.8, 8)],
)
def test_sampling_probs_matches_the_reference(temperature, top_p, top_k):
    """Rejection sampling compares target and draft probabilities directly.

    Both sides must be built the same way as the reference builds them, so this
    demands exact equality rather than a tolerance. ``(1.0, 0.95, 20)`` is the
    combination the DFlash 2 blog evaluates with.
    """
    torch.manual_seed(42)
    logits = torch.randn(2, 5, VOCAB)

    produced = sampling_probs(logits, temperature, top_p, top_k)
    expected = _reference_sampling_probs(logits, temperature, top_p, top_k)

    assert torch.equal(produced, expected)
    torch.testing.assert_close(produced.sum(-1), torch.ones(2, 5))
    # Filtered tokens must be exactly zero: the rejection-sampling residual
    # ``clamp(p - q, 0)`` is only correct if excluded mass is 0, not merely small.
    assert torch.equal(produced > 0, expected > 0)


def test_truncation_actually_narrows_the_support():
    """A guard against the knobs being silently accepted and ignored."""
    torch.manual_seed(0)
    logits = torch.randn(1, 1, VOCAB)
    full = sampling_probs(logits, 1.0)
    assert int((full > 0).sum()) == VOCAB
    assert int((sampling_probs(logits, 1.0, top_k=8) > 0).sum()) == 8
    assert int((sampling_probs(logits, 1.0, top_p=0.5) > 0).sum()) < VOCAB


def test_top_p_always_keeps_the_most_likely_token():
    """Exclusive cumulative mass: a peaked distribution must not empty the support."""
    logits = torch.zeros(1, 1, VOCAB)
    logits[..., 7] = 50.0
    probs = sampling_probs(logits, 1.0, top_p=0.1)
    assert int(probs.argmax(-1)) == 7
    assert int((probs > 0).sum()) >= 1


def test_sample_is_greedy_at_zero_temperature_and_stays_in_support():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, VOCAB)
    assert torch.equal(sample(logits, 0.0), logits.argmax(-1))

    torch.manual_seed(0)
    drawn = sample(logits, 1.0, top_k=4)
    allowed = logits.topk(4, dim=-1).indices
    assert bool((allowed == drawn.unsqueeze(-1)).any(-1).all())


@pytest.mark.parametrize(
    "temperature,top_p,top_k", [(-0.1, 1.0, 0), (1.0, 0.0, 0), (1.0, 1.5, 0), (1.0, 1.0, -1)]
)
def test_invalid_sampling_parameters_are_rejected(temperature, top_p, top_k):
    with pytest.raises(ValueError, match="sampling parameters"):
        validate_sampling(temperature, top_p, top_k)


def _draft(**dflash_config):
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    cfg.num_target_layers = 4
    cfg.block_size = 4
    cfg.dflash_config = {"mask_token_id": VOCAB - 1, "target_layer_ids": [1], **dflash_config}
    cfg._attn_implementation = "sdpa"
    return Qwen3DFlashDraftModel(cfg)


def test_compute_logits_applies_the_targets_output_transform():
    """Muse Glimmer scales and softcaps its logits; the draft must do the same.

    Training supervises and decoding verifies against the *transformed* logits, so
    skipping them here would train one distribution and serve another.
    """
    head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    hidden = torch.randn(1, 3, HIDDEN)
    multiplier, softcap = 0.19611613513818404, 20.0

    draft = _draft(output_multiplier=multiplier, final_logit_softcapping=softcap)
    expected = torch.tanh(head(hidden) * multiplier / softcap) * softcap
    torch.testing.assert_close(draft.compute_logits(hidden, head), expected)


def test_compute_logits_is_the_identity_without_the_transform_fields():
    """Every Qwen3-family drafter omits both fields; that path must stay untouched."""
    head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    hidden = torch.randn(1, 3, HIDDEN)
    torch.testing.assert_close(_draft().compute_logits(hidden, head), head(hidden))


def test_non_positive_softcapping_is_rejected():
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        _draft(final_logit_softcapping=0.0)


def test_input_embedding_scale_is_applied_through_the_embedding_module():
    """The scale must reach the noise block, and the module call must be preserved.

    The reference indexes the raw weight table; we call the module because under
    tensor parallelism only the module carries the vocab-parallel DTensor plan.
    For a plain ``nn.Embedding`` the two agree.
    """

    class _Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(VOCAB, HIDDEN)

        def get_input_embeddings(self):
            return self.embedding

    target, ids = _Target(), torch.tensor([[1, 2, 3, 4]])
    unscaled = _draft().embed_noise_block(target, ids)
    torch.testing.assert_close(unscaled, target.embedding(ids))
    torch.testing.assert_close(_draft(input_embedding_scale=2.5).embed_noise_block(target, ids), unscaled * 2.5)


def test_output_head_falls_back_to_get_output_embeddings():
    """A ``*ForConditionalGeneration`` target exposes no ``lm_head`` attribute."""

    class _WithLmHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)

        def get_output_embeddings(self):
            return None

    class _WithoutLmHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)

        def get_output_embeddings(self):
            return self.head

    with_head = _WithLmHead()
    assert resolve_output_head(with_head) is with_head.lm_head
    without_head = _WithoutLmHead()
    assert resolve_output_head(without_head) is without_head.head
