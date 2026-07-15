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

"""Numerical parity between the streaming producer/loader and the colocated path.

This is the contract that PR 2 of issue #3062 is gated on: feeding the
same input to ``HFEagle3TargetModel.generate_batch`` (the existing
colocated path) and to ``FeatureProducer.produce`` +
``FeatureDataLoader.__next__`` (the streaming path) must yield a
bit-identical :class:`Eagle3TargetBatch`. The check covers every
supervision-relevant field so a producer-side transformation (a dropout
toggle, a dtype cast, a transpose, a stray ``clone``) trips here
instead of silently degrading a downstream draft.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel
from nemo_automodel.components.speculative.streaming import (
    LocalFeatureStore,
    SampleRefQueue,
)
from nemo_automodel.components.speculative.streaming.loader import FeatureDataLoader
from nemo_automodel.components.speculative.streaming.producer import FeatureProducer


class _FakeHFCausalLM(nn.Module):
    """Same deterministic HF stand-in used in ``tests/unit_tests/speculative/test_eagle3_target_backend.py``."""

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


@pytest.fixture
def colocated() -> HFEagle3TargetModel:
    return HFEagle3TargetModel(_FakeHFCausalLM(num_layers=4), aux_layer_ids=[0, 1, 3])


@pytest.fixture
def streaming_inputs():
    """Same input shape & values for both paths."""
    torch.manual_seed(0)
    return (
        torch.randint(0, 32, (2, 8)),
        torch.ones(2, 8, dtype=torch.long),
        torch.ones(2, 8, dtype=torch.long),
    )


def test_streaming_path_matches_colocated_path(colocated, streaming_inputs) -> None:
    input_ids, attention_mask, loss_mask = streaming_inputs

    baseline = colocated.generate_batch(input_ids, attention_mask, loss_mask)

    store = LocalFeatureStore(max_samples=8, max_bytes=4 * 1024 * 1024)
    queue = SampleRefQueue(store)
    producer = FeatureProducer(colocated, store, run_id="parity")
    ref = producer.produce(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask)
    queue.put(ref)
    loader = FeatureDataLoader(queue, store)

    streamed = next(iter(loader))

    assert torch.equal(baseline.aux_hidden_states, streamed.aux_hidden_states)
    assert torch.equal(baseline.input_ids, streamed.input_ids)
    assert torch.equal(baseline.attention_mask, streamed.attention_mask)
    assert torch.equal(baseline.loss_mask, streamed.loss_mask)
    assert baseline.logits is not None and streamed.logits is not None
    assert torch.equal(baseline.logits, streamed.logits)
    assert streamed.target_probs is None
    assert streamed.position_mask is None


def test_streaming_path_over_multiple_inputs_matches_colocated(colocated) -> None:
    """Re-running through the streaming pipeline must stay stable.

    A drift in the producer (a hidden cache, an unintended ``clone``
    alias) would surface as soon as a second sample lands and the first
    one's tensors mutate. Pin it down by repeating the round-trip.
    """
    store = LocalFeatureStore(max_samples=8, max_bytes=4 * 1024 * 1024)
    queue = SampleRefQueue(store)
    producer = FeatureProducer(colocated, store, run_id="parity")
    loader = FeatureDataLoader(queue, store)

    torch.manual_seed(1)
    batches_streamed = []
    for _ in range(3):
        ids = torch.randint(0, 32, (2, 8))
        attn = torch.ones(2, 8, dtype=torch.long)
        loss = torch.ones(2, 8, dtype=torch.long)
        baseline = colocated.generate_batch(ids, attn, loss)
        ref = producer.produce(input_ids=ids, attention_mask=attn, loss_mask=loss)
        queue.put(ref)
        streamed = next(iter(loader))
        batches_streamed.append((baseline, streamed, ids, attn, loss))

    for baseline, streamed, ids, attn, loss in batches_streamed:
        assert torch.equal(baseline.aux_hidden_states, streamed.aux_hidden_states)
        assert torch.equal(baseline.input_ids, streamed.input_ids)
        assert torch.equal(baseline.logits, streamed.logits)
