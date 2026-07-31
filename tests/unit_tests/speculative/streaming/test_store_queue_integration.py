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

"""Integration tests for :class:`LocalFeatureStore` + :class:`SampleRefQueue`."""

from __future__ import annotations

import torch

from nemo_automodel.components.speculative.streaming import (
    FeatureAlgorithm,
    LocalFeatureStore,
    SampleRefQueue,
)


def _eagle3_features() -> dict[str, torch.Tensor]:
    return {
        "aux_hidden_states": torch.zeros(64, dtype=torch.float32),
        "input_ids": torch.zeros(8, dtype=torch.long),
        "attention_mask": torch.ones(8, dtype=torch.long),
        "loss_mask": torch.ones(8, dtype=torch.long),
        "logits": torch.zeros(8, 32, dtype=torch.float32),
    }


def test_store_queue_put_acquire_get_release_ack_cycle() -> None:
    store = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    queue = SampleRefQueue(store)
    ref = store.put(
        "s1",
        _eagle3_features(),
        run_id="r1",
        algorithm=FeatureAlgorithm.EAGLE3,
        schema_version=1,
        target_model_version="0",
        draft_weight_version="0",
        num_tokens=8,
    )
    queue.put(ref)
    assert queue.pending_count() == 1
    assert store.health().sample_count == 1

    lease = queue.acquire()
    assert lease is not None
    assert lease.ref.sample_id == "s1"
    assert queue.outstanding_count() == 1

    tensors, handle = store.get(lease.ref)
    assert set(tensors) == set(_eagle3_features())
    store.release(handle)
    queue.ack(lease)

    assert queue.outstanding_count() == 0
    assert queue.pending_count() == 0
    assert store.health().sample_count == 0


def test_store_queue_backpressure_unblocks_after_consumer_release() -> None:
    store = LocalFeatureStore(
        max_samples=2,
        max_bytes=32 * 1024,
        high_watermark_bytes=16 * 1024,
        low_watermark_bytes=4 * 1024,
    )
    queue = SampleRefQueue(store)
    ref1 = store.put(
        "s1",
        _eagle3_features(),
        run_id="r1",
        algorithm=FeatureAlgorithm.EAGLE3,
        schema_version=1,
        target_model_version="0",
        draft_weight_version="0",
        num_tokens=8,
    )
    queue.put_blocks_until_below(ref1)

    lease = queue.acquire()
    assert lease is not None
    _, handle = store.get(lease.ref)
    store.release(handle)
    queue.ack(lease)

    ref2 = store.put(
        "s2",
        _eagle3_features(),
        run_id="r1",
        algorithm=FeatureAlgorithm.EAGLE3,
        schema_version=1,
        target_model_version="0",
        draft_weight_version="0",
        num_tokens=8,
    )
    queue.put_blocks_until_below(ref2)
    assert queue.pending_count() == 1
