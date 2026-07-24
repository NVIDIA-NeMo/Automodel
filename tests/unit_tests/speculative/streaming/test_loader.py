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

"""Unit tests for :class:`FeatureDataLoader`.

The loader's contract under test:

1. Each ``__next__`` yields an :class:`Eagle3TargetBatch` materialized
   from the next leased :class:`SampleRef`.
2. The previous batch's lease is ack'd and its store handle released on
   the NEXT pull -- so a trainer can hold one batch across one forward
   without it being freed.
3. ``consume_now`` releases the most recently yielded batch eagerly
   (e.g. immediately after backward) and is idempotent.
4. Iteration ends on ``close`` (or when the queue drains); ``close``
   releases pending resources and shuts the queue.
5. Refs whose ``algorithm`` does not match the loader's are failed back
   to the queue (so a producer emitting a different algorithm does not
   silently corrupt the trainer's batches).
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
    SampleRefQueue,
)
from nemo_automodel.components.speculative.streaming.loader import FeatureDataLoader
from nemo_automodel.components.speculative.streaming.refs import SampleRef
from nemo_automodel.components.speculative.streaming.stores.local import LocalFeatureStore as _Store


class _FakeHFCausalLM(nn.Module):
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
def store() -> LocalFeatureStore:
    return _Store(max_samples=16, max_bytes=4 * 1024 * 1024)


@pytest.fixture
def queue(store) -> SampleRefQueue:
    return SampleRefQueue(store)


@pytest.fixture
def backend() -> HFEagle3TargetModel:
    return HFEagle3TargetModel(_FakeHFCausalLM(num_layers=4), aux_layer_ids=[0, 1, 3])


def _produce_one(
    target,
    store,
    *,
    sample_id="s1",
    position_ids=None,
    seq_lens=None,
    doc_remaining=None,
) -> SampleRef:
    input_ids = torch.randint(0, 32, (2, 8))
    attn = torch.ones(2, 8, dtype=torch.long)
    loss = torch.ones(2, 8, dtype=torch.long)
    batch: Eagle3TargetBatch = target.generate_batch(
        input_ids,
        attn,
        loss,
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    tensors = {
        "aux_hidden_states": batch.aux_hidden_states,
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "loss_mask": batch.loss_mask,
        "logits": batch.logits,
    }
    if position_ids is not None:
        tensors["position_ids"] = position_ids
    if seq_lens is not None:
        tensors["seq_lens"] = seq_lens
    if doc_remaining is not None:
        tensors["doc_remaining"] = doc_remaining
    return store.put(
        sample_id,
        tensors,
        run_id="r1",
        algorithm=FeatureAlgorithm.EAGLE3,
        schema_version=1,
        target_model_version="0",
        draft_weight_version="0",
        num_tokens=16,
    )


# --- 1. lease + materialize -------------------------------------------------


def test_loader_round_trips_packing_metadata(queue, store, backend) -> None:
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, -1)
    seq_lens = torch.tensor([[4, 4], [8, 0]], dtype=torch.long)
    doc_remaining = torch.ones(2, 8, dtype=torch.long)
    input_ids = torch.randint(0, 32, (2, 8))
    attn = torch.ones(2, 8, dtype=torch.long)
    loss = torch.ones(2, 8, dtype=torch.long)
    batch: Eagle3TargetBatch = backend.generate_batch(input_ids, attn, loss)
    ref = store.put(
        "packed",
        {
            "aux_hidden_states": batch.aux_hidden_states,
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "loss_mask": batch.loss_mask,
            "logits": batch.logits,
            "position_ids": position_ids,
            "seq_lens": seq_lens,
            "doc_remaining": doc_remaining,
        },
        run_id="r1",
        algorithm=FeatureAlgorithm.EAGLE3,
        schema_version=1,
        target_model_version="0",
        draft_weight_version="0",
        num_tokens=16,
    )
    queue.put(ref)
    loader = FeatureDataLoader(queue, store)
    batch = next(iter(loader))
    assert batch.seq_lens is not None
    assert torch.equal(batch.position_ids, position_ids)
    assert torch.equal(batch.seq_lens, seq_lens)
    assert torch.equal(batch.doc_remaining, doc_remaining)


def test_loader_yields_eagle3_target_batch_with_expected_fields(queue, store, backend) -> None:
    ref = _produce_one(backend, store, sample_id="s1")
    queue.put(ref)
    loader = FeatureDataLoader(queue, store)

    batch = next(iter(loader))

    assert isinstance(batch, Eagle3TargetBatch)
    assert batch.aux_hidden_states.shape[0] == 2
    assert batch.logits is not None
    assert batch.target_probs is None
    assert batch.position_mask is None
    assert store.health().sample_count == 1


def test_loader_release_previous_on_next_pull(queue, store, backend) -> None:
    # Two samples -> pull twice. The first pull holds sample 1 alive; the
    # second pull releases it AND stores sample 2 alive.
    ref1 = _produce_one(backend, store, sample_id="s1")
    ref2 = _produce_one(backend, store, sample_id="s2")
    queue.put(ref1)
    queue.put(ref2)

    loader = FeatureDataLoader(queue, store)
    batch1 = next(iter(loader))
    assert store.health().sample_count == 2  # both still resident
    next(iter(loader))
    # Pulling the second batch releases the first.
    assert store.health().sample_count == 1
    # batch1 still carries a copy of sample 1's tensors (the LocalFeatureStore
    # hands out detached copies), so it's safe to use even though the
    # store has dropped it.
    assert batch1.aux_hidden_states.shape == (2, 8, 48)


def test_loader_stops_when_queue_drains(queue, store) -> None:
    loader = FeatureDataLoader(queue, store)
    with pytest.raises(StopIteration):
        next(iter(loader))


# --- 2. consume_now + close ------------------------------------------------


def test_consume_now_releases_previous_lease(queue, store, backend) -> None:
    ref1 = _produce_one(backend, store, sample_id="s1")
    queue.put(ref1)
    loader = FeatureDataLoader(queue, store)
    next(iter(loader))
    assert store.health().sample_count == 1
    loader.consume_now()
    assert store.health().sample_count == 0


def test_consume_now_is_idempotent(queue, store, backend) -> None:
    ref1 = _produce_one(backend, store, sample_id="s1")
    queue.put(ref1)
    loader = FeatureDataLoader(queue, store)
    next(iter(loader))
    loader.consume_now()
    loader.consume_now()  # second call is a no-op, not an error
    assert store.health().sample_count == 0


def test_close_releases_pending_and_stops_iteration(queue, store, backend) -> None:
    ref1 = _produce_one(backend, store, sample_id="s1")
    queue.put(ref1)
    loader = FeatureDataLoader(queue, store)
    next(iter(loader))
    assert store.health().sample_count == 1
    loader.close()
    assert store.health().sample_count == 0
    with pytest.raises(StopIteration):
        next(iter(loader))


def test_close_is_idempotent(queue, store) -> None:
    loader = FeatureDataLoader(queue, store)
    loader.close()
    loader.close()  # second call must not raise


def test_context_manager_releases_on_exit(queue, store, backend) -> None:
    ref1 = _produce_one(backend, store, sample_id="s1")
    queue.put(ref1)
    with FeatureDataLoader(queue, store) as loader:
        next(iter(loader))
        assert store.health().sample_count == 1
    assert store.health().sample_count == 0


# --- 3. algorithm mismatch -------------------------------------------------


def test_loader_fails_lease_on_algorithm_mismatch(queue, store) -> None:
    from nemo_automodel.components.speculative.streaming.refs import FeatureSpec

    bad_ref = SampleRef(
        sample_id="bad",
        run_id="r1",
        store_uri=store.store_uri,
        feature_keys={
            "hidden_states": "s/hidden_states",
            "input_ids": "s/input_ids",
            "attention_mask": "s/attention_mask",
            "loss_mask": "s/loss_mask",
        },
        feature_specs={
            k: FeatureSpec(shape=(2, 8), dtype=torch.float32)
            for k in ("hidden_states", "input_ids", "attention_mask", "loss_mask")
        },
        algorithm=FeatureAlgorithm.DFLASH,
        schema_version=1,
        num_tokens=16,
        estimated_bytes=64,
        target_model_version="0",
        draft_weight_version="0",
    )
    store.put(
        bad_ref.sample_id,
        {
            "hidden_states": torch.zeros(2, 8),
            "input_ids": torch.zeros(2, 8, dtype=torch.long),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "loss_mask": torch.ones(2, 8, dtype=torch.long),
        },
        run_id="r1",
        algorithm=FeatureAlgorithm.DFLASH,
        schema_version=1,
        target_model_version="0",
        draft_weight_version="0",
        num_tokens=16,
    )
    queue.put(bad_ref)
    loader = FeatureDataLoader(queue, store)
    with pytest.raises(ValueError, match=r"bound to algorithm=.*EAGLE3"):
        next(iter(loader))
    # The mismatched lease is failed back, not silently dropped: the
    # queue pending count covers it for redelivery.
    assert queue.outstanding_count() == 0
    assert queue.pending_count() == 1
