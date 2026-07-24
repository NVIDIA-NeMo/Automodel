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

"""Unit tests for :class:`FeatureProducer`.

The producer's contract under test:

1. ``produce`` forwards the inputs to the wrapped backend's
   ``generate_batch``, packs the resulting tensors under the EAGLE-3
   schema key set, and returns a :class:`SampleRef` whose
   ``feature_specs`` mirror the measured shapes / dtypes exactly.
2. The producer's algorithm inference picks EAGLE-3 for an
   ``HFEagle3TargetModel`` and rejects backend classes it doesn't
   recognize.
3. ``produce`` raises if the wrapped backend returns a
   ``Eagle3TargetBatch`` with the precomputed draft-vocab encoding
   (the colocated producer only supports the colocated encoding).
4. ``set_vocab_mapping`` and ``get_input_embeddings`` are pass-throughs
   to the wrapped backend.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

from nemo_automodel.components.speculative.eagle.target import Eagle3TargetBatch, HFEagle3TargetModel
from nemo_automodel.components.speculative.streaming import (
    FeatureAlgorithm,
    LocalFeatureStore,
)
from nemo_automodel.components.speculative.streaming.eagle3 import EAGLE3_SUPERVISION_ENCODINGS
from nemo_automodel.components.speculative.streaming.producer import FeatureProducer


class _FakeHFCausalLM(nn.Module):
    """Tiny HF causal-LM stand-in returning ``CausalLMOutput`` with ``.logits``."""

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
def target() -> HFEagle3TargetModel:
    return HFEagle3TargetModel(_FakeHFCausalLM(num_layers=4), aux_layer_ids=[0, 1, 3])


@pytest.fixture
def store() -> LocalFeatureStore:
    return LocalFeatureStore(max_samples=8, max_bytes=4 * 1024 * 1024)


def _inputs(*, batch: int = 2, seq: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randint(0, 32, (batch, seq)),
        torch.ones(batch, seq, dtype=torch.long),
        torch.ones(batch, seq, dtype=torch.long),
    )


# --- 1. Round-trip via the wrapped backend --------------------------------


def test_producer_passes_input_through_to_wrapped_backend(target, store) -> None:
    producer = FeatureProducer(target, store, run_id="r1")
    input_ids, attn, loss = _inputs()
    ref = producer.produce(input_ids=input_ids, attention_mask=attn, loss_mask=loss)
    assert ref.algorithm is FeatureAlgorithm.EAGLE3
    assert set(ref.feature_specs) == set(EAGLE3_SUPERVISION_ENCODINGS)


def test_producer_ref_feature_specs_match_measured_tensors(target, store) -> None:
    producer = FeatureProducer(target, store, run_id="r1")
    input_ids, attn, loss = _inputs()
    ref = producer.produce(input_ids=input_ids, attention_mask=attn, loss_mask=loss)
    # The store stashed detached copies; comparing ref's reported
    # ``feature_specs`` against the inputs' shapes / dtypes is what the
    # producer contract promises.
    assert ref.feature_specs["input_ids"].shape == tuple(input_ids.shape)
    assert ref.feature_specs["input_ids"].dtype is torch.long
    assert ref.feature_specs["logits"].shape[0] == input_ids.shape[0]
    assert ref.feature_specs["logits"].shape[1] == input_ids.shape[1]


def test_producer_assigns_unique_sample_ids(target, store) -> None:
    producer = FeatureProducer(target, store, run_id="r1")
    ids = set()
    for _ in range(3):
        input_ids, attn, loss = _inputs()
        ref = producer.produce(input_ids=input_ids, attention_mask=attn, loss_mask=loss)
        ids.add(ref.sample_id)
    assert len(ids) == 3


def test_producer_sample_id_factory_is_used(target, store) -> None:
    seen = []

    def factory(n):
        seen.append(n)
        return f"id-{n}"

    producer = FeatureProducer(target, store, run_id="r1", sample_id_factory=factory)
    input_ids, attn, loss = _inputs()
    ref = producer.produce(input_ids=input_ids, attention_mask=attn, loss_mask=loss)
    assert ref.sample_id == "id-1"
    assert seen == [1]


# --- 2. Algorithm inference -----------------------------------------------


def test_producer_infers_eagle3_algorithm_from_runtime_type(target, store) -> None:
    producer = FeatureProducer(target, store, run_id="r1")
    assert producer._algorithm is FeatureAlgorithm.EAGLE3


def test_producer_algorithm_override_wins(target, store) -> None:
    producer = FeatureProducer(target, store, run_id="r1", algorithm=FeatureAlgorithm.EAGLE3)
    assert producer._algorithm is FeatureAlgorithm.EAGLE3


def test_producer_rejects_unknown_backend_type(store) -> None:
    class RandomBackend:
        def generate_batch(self, **kwargs):
            raise NotImplementedError

        def get_input_embeddings(self):
            raise NotImplementedError

    with pytest.raises(ValueError, match="cannot infer FeatureAlgorithm"):
        FeatureProducer(RandomBackend(), store, run_id="r1")


# --- 3. Precomputed-vocab encoding rejected -------------------------------


def test_producer_rejects_precomputed_draft_vocab_encoding(store) -> None:
    class PrecomputedBackend:
        def get_input_embeddings(self) -> nn.Embedding:
            return nn.Embedding(1, 1)

        def generate_batch(self, *, input_ids, attention_mask, loss_mask, **kwargs):
            return Eagle3TargetBatch(
                aux_hidden_states=torch.zeros(2, 8, 96),
                input_ids=torch.zeros(2, 8, dtype=torch.long),
                attention_mask=torch.ones(2, 8, dtype=torch.long),
                loss_mask=torch.ones(2, 8, dtype=torch.long),
                target_probs=torch.zeros(2, 8, 16),
                position_mask=torch.ones(2, 8, dtype=torch.bool),
            )

    producer = FeatureProducer(PrecomputedBackend(), store, run_id="r1", algorithm=FeatureAlgorithm.EAGLE3)
    input_ids, attn, loss = _inputs()
    with pytest.raises(ValueError, match=r"requires .*\.logits"):
        producer.produce(input_ids=input_ids, attention_mask=attn, loss_mask=loss)


# --- 4. Pass-throughs ------------------------------------------------------


def test_producer_set_vocab_mapping_forwards(target, store) -> None:
    seen = []

    def _set_vocab(selected_token_ids, selected_token_mask):
        seen.append((selected_token_ids, selected_token_mask))

    target.set_vocab_mapping = _set_vocab  # type: ignore[method-assign]
    producer = FeatureProducer(target, store, run_id="r1")
    ids = torch.arange(4)
    mask = torch.ones(4, dtype=torch.bool)
    producer.set_vocab_mapping(ids, mask)
    assert seen and torch.equal(seen[0][0], ids)


def test_producer_get_input_embeddings_forwards(target, store) -> None:
    producer = FeatureProducer(target, store, run_id="r1")
    embeddings = producer.get_input_embeddings()
    assert embeddings is target.get_input_embeddings()


def test_producer_run_id_required(target, store) -> None:
    with pytest.raises(ValueError, match="run_id"):
        FeatureProducer(target, store, run_id="")
