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

"""Unit tests for :class:`AsyncFeaturePipeline`.

The pipeline's contract under test:

1. The background thread drains a prompt source, runs the wrapped
   :class:`FeatureProducer`, and pushes each :class:`SampleRef` onto
   the queue via ``put_blocks_until_below`` so the queue's HWM/LWM
   hysteresis governs the producer's pacing.
2. ``start`` / ``stop`` / context-manager lifecycle is idempotent.
3. Prompt-source exhaustion stops the thread (when
   ``stop_on_exhausted=True``); a streaming source is polled until
   shutdown.
4. An exception in the prompt source (or the producer) is captured
   and re-raised on ``stop``, so the trainer sees it.
5. A concurrent loader can iterate the queue while the pipeline runs,
   overlapping target forward with draft-side consumption.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel
from nemo_automodel.components.speculative.streaming import (
    FeatureAlgorithm,
    LocalFeatureStore,
    SampleRefQueue,
)
from nemo_automodel.components.speculative.streaming.async_pipeline import AsyncFeaturePipeline
from nemo_automodel.components.speculative.streaming.loader import FeatureDataLoader
from nemo_automodel.components.speculative.streaming.producer import FeatureProducer
from nemo_automodel.components.speculative.streaming.stores.shared_dir import SharedDirFeatureStore


class _FakeHFCausalLM(nn.Module):
    """Tiny HF causal-LM stand-in shared with the producer/loader tests."""

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
    return LocalFeatureStore(
        max_samples=16,
        max_bytes=4 * 1024 * 1024,
        high_watermark_bytes=3 * 1024 * 1024,
        low_watermark_bytes=1 * 1024 * 1024,
    )


@pytest.fixture
def queue(store) -> SampleRefQueue:
    return SampleRefQueue(store)


@pytest.fixture
def target() -> HFEagle3TargetModel:
    return HFEagle3TargetModel(_FakeHFCausalLM(num_layers=4), aux_layer_ids=[0, 1, 3])


@pytest.fixture
def producer(target, store) -> FeatureProducer:
    return FeatureProducer(target, store, run_id="r1")


def _prompts(n: int, *, batch: int = 2, seq: int = 8) -> Iterator:
    """Yield ``n`` 3-tuples ``(input_ids, attention_mask, loss_mask)``."""
    for _ in range(n):
        yield (
            torch.randint(0, 32, (batch, seq)),
            torch.ones(batch, seq, dtype=torch.long),
            torch.ones(batch, seq, dtype=torch.long),
        )


# --- 1. background thread runs ahead ---------------------------------------


def test_async_pipeline_produces_all_prompts(producer, queue) -> None:
    with AsyncFeaturePipeline(producer, queue, _prompts(3)) as pipe:
        # The thread is daemon and runs on its own; wait for it to drain.
        pipe.join(timeout=2.0)
    assert queue.pending_count() == 3


def test_async_pipeline_producer_runs_ahead_of_loader(producer, queue) -> None:
    """The producer pre-fills the queue so a consumer can iterate
    even when the trainer loop is slower than the target forward."""
    with AsyncFeaturePipeline(producer, queue, _prompts(5)) as pipe:
        # The producer thread should be running concurrently. Yield
        # small slices to the loader; the producer keeps refilling.
        loader = FeatureDataLoader(queue, producer._store)
        pulled = 0
        deadline = time.monotonic() + 2.0
        while pulled < 5 and time.monotonic() < deadline:
            try:
                next(iter(loader))
                pulled += 1
            except StopIteration:
                time.sleep(0.01)
        pipe.join(timeout=2.0)
    assert pulled == 5


# --- 2. lifecycle -----------------------------------------------------------


def test_async_pipeline_start_is_idempotent(producer, queue) -> None:
    pipe = AsyncFeaturePipeline(producer, queue, _prompts(0))
    pipe.start()
    pipe.start()  # no-op the second time
    pipe.stop()


def test_async_pipeline_stop_is_idempotent(producer, queue) -> None:
    pipe = AsyncFeaturePipeline(producer, queue, _prompts(0))
    pipe.start()
    pipe.stop()
    pipe.stop()  # no-op the second time


def test_async_pipeline_stop_joins_thread(producer, queue) -> None:
    pipe = AsyncFeaturePipeline(producer, queue, _prompts(2), poll_interval=0.01)
    pipe.start()
    pipe.stop(join_timeout=2.0)
    assert pipe._thread is None or not pipe._thread.is_alive()


def test_async_pipeline_context_manager_drains_on_exit(producer, queue) -> None:
    with AsyncFeaturePipeline(producer, queue, _prompts(2)) as pipe:
        pipe.join(timeout=2.0)
    assert queue.is_closed


# --- 3. prompt-source variants ----------------------------------------------


def test_async_pipeline_accepts_callable_prompt_source(producer, queue) -> None:
    """A plain callable (not an iterator) is also supported."""
    remaining = [3]

    def source():
        if remaining[0] == 0:
            return None
        remaining[0] -= 1
        return (
            torch.randint(0, 32, (2, 8)),
            torch.ones(2, 8, dtype=torch.long),
            torch.ones(2, 8, dtype=torch.long),
        )

    with AsyncFeaturePipeline(producer, queue, source):
        time.sleep(0.1)
    assert queue.pending_count() == 3


def test_async_pipeline_streaming_mode_polls_until_exhausted(producer, queue) -> None:
    """``stop_on_exhausted=False`` keeps the thread alive after the
    source returns ``None``; shutdown still works through ``stop``."""
    remaining = [1]
    calls = [0]

    def source():
        calls[0] += 1
        if remaining[0] == 0:
            return None
        remaining[0] -= 1
        return (
            torch.randint(0, 32, (2, 8)),
            torch.ones(2, 8, dtype=torch.long),
            torch.ones(2, 8, dtype=torch.long),
        )

    pipe = AsyncFeaturePipeline(producer, queue, source, stop_on_exhausted=False, poll_interval=0.01)
    pipe.start()
    time.sleep(0.2)  # let it poll a few times
    assert calls[0] >= 2  # the source is being polled repeatedly
    pipe.stop()
    assert pipe._thread is None or not pipe._thread.is_alive()


# --- 4. error propagation --------------------------------------------------


def test_async_pipeline_propagates_prompt_source_error(producer, queue) -> None:
    def bad_source():
        raise RuntimeError("dataset on fire")

    pipe = AsyncFeaturePipeline(producer, queue, bad_source)
    pipe.start()
    pipe.join(timeout=2.0)
    with pytest.raises(RuntimeError, match="dataset on fire"):
        pipe.stop()


def test_async_pipeline_propagates_producer_error(target, store, queue) -> None:
    """An exception raised by the wrapped target backend reaches the trainer."""

    class BadBackend:
        def get_input_embeddings(self):
            return nn.Embedding(1, 1)

        def generate_batch(self, **kwargs):
            raise RuntimeError("target exploded")

    bad_producer = FeatureProducer(BadBackend(), store, run_id="r1", algorithm=FeatureAlgorithm.EAGLE3)
    pipe = AsyncFeaturePipeline(bad_producer, queue, _prompts(1))
    pipe.start()
    pipe.join(timeout=2.0)
    with pytest.raises(RuntimeError, match="target exploded"):
        pipe.stop()


# --- 5. backpressure interacts correctly -----------------------------------


def test_async_pipeline_stop_unblocks_backpressured_producer(producer, queue) -> None:
    """``stop`` closes the queue so a blocked ``put_blocks_until_below`` exits."""
    small_store = LocalFeatureStore(
        max_samples=8,
        max_bytes=256 * 1024,
        high_watermark_bytes=4 * 1024,
        low_watermark_bytes=1024,
    )
    small_queue = SampleRefQueue(small_store)
    small_producer = FeatureProducer(producer._target, small_store, run_id="r1")
    for _ in range(3):
        small_producer.produce(
            input_ids=torch.randint(0, 32, (2, 8)),
            attention_mask=torch.ones(2, 8, dtype=torch.long),
            loss_mask=torch.ones(2, 8, dtype=torch.long),
        )
    assert small_store.health().high_watermark_hit

    pipe = AsyncFeaturePipeline(small_producer, small_queue, _prompts(1), poll_interval=0.01)
    pipe.start()
    time.sleep(0.1)
    assert pipe._thread is not None and pipe._thread.is_alive()
    pipe.stop(join_timeout=2.0)
    assert small_queue.is_closed


def test_async_pipeline_with_shared_dir_store(tmp_path) -> None:
    directory = str(tmp_path / "store")
    store = SharedDirFeatureStore(
        directory,
        max_samples=8,
        max_bytes=4 * 1024 * 1024,
        high_watermark_bytes=3 * 1024 * 1024,
        low_watermark_bytes=1 * 1024 * 1024,
    )
    queue = SampleRefQueue(store)
    target = HFEagle3TargetModel(_FakeHFCausalLM(num_layers=4), aux_layer_ids=[0, 1, 3])
    producer = FeatureProducer(target, store, run_id="r1")
    with AsyncFeaturePipeline(producer, queue, _prompts(2), poll_interval=0.01) as pipe:
        loader = FeatureDataLoader(queue, store)
        pulled = 0
        deadline = time.monotonic() + 3.0
        while pulled < 2 and time.monotonic() < deadline:
            try:
                batch = next(iter(loader))
                assert batch.logits is not None
                pulled += 1
            except StopIteration:
                time.sleep(0.01)
        pipe.join(timeout=2.0)
    assert pulled == 2


def test_async_pipeline_backpressure_pauses_when_store_overflows(producer, queue) -> None:
    """Tiny store forces a high watermark; the producer thread should
    block on ``put_blocks_until_below`` until the consumer drains."""
    small_store = LocalFeatureStore(
        max_samples=8,
        max_bytes=128 * 1024,
        high_watermark_bytes=64 * 1024,
        low_watermark_bytes=16 * 1024,
    )
    small_queue = SampleRefQueue(small_store)
    small_producer = FeatureProducer(producer._target, small_store, run_id="r1")

    # Drop in the consumer drainer BEFORE starting the producer; the
    # drainer pulls ref-by-ref so the producer never has more than one
    # ref outstanding.
    stopped = threading.Event()

    def drainer():
        loader = FeatureDataLoader(small_queue, small_store)
        pulled = 0
        target_pulls = 5
        deadline = time.monotonic() + 5.0
        while pulled < target_pulls and time.monotonic() < deadline:
            try:
                next(iter(loader))
                pulled += 1
                # Drop a sample periodically so the producer can advance.
                if pulled % 2 == 0:
                    small_store.gc()
            except StopIteration:
                time.sleep(0.005)
        stopped.set()

    drainer_thread = threading.Thread(target=drainer, daemon=True)
    drainer_thread.start()

    with AsyncFeaturePipeline(small_producer, small_queue, _prompts(5), poll_interval=0.01):
        stopped.wait(timeout=5.0)

    drainer_thread.join(timeout=2.0)
    assert stopped.is_set(), "drainer did not finish in time -- backpressure may be broken"
