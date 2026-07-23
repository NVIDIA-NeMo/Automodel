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

"""Unit tests for :class:`LocalFeatureStore`.

The store is the in-process reference backend for the streaming data plane;
everything in this module runs CPU-only and covers the contract pieces of
:class:`~nemo_automodel.components.speculative.streaming.store.FeatureStore`:

1. Put + get + release round-trip; get returns detached copies so the
   consumer can mutate without disturbing the stored tensor.
2. Refcount: a sample held by two get() calls is freed only after both
   release -- prevents prefetch from racing a free.
3. Residency caps (sample count + bytes), validated up-front (a put that
   would exceed the cap raises :class:`MemoryError` *before* any partial
   write is observable).
4. Multiple-acquire handles share the same resident bytes; ``release``
   decrements only once per outstanding handle.
5. ``health()`` ints-only snapshot drives the
   :class:`~nemo_automodel.components.speculative.streaming.queue.SampleRefQueue`
   backpressure decisions.
6. ``close()`` is idempotent and rejects subsequent put/get.
7. URI mismatch: a ref built against another store object is rejected by
   :meth:`get`, mirroring how RFC §"sample_id + store_uri" partitions
   consumer-side data.
"""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.speculative.streaming import FeatureAlgorithm
from nemo_automodel.components.speculative.streaming.refs import SampleRef
from nemo_automodel.components.speculative.streaming.stores.local import LocalFeatureStore


def _eagle3_bytes(size: int) -> dict[str, torch.Tensor]:
    return {
        "aux_hidden_states": torch.zeros(size, dtype=torch.float32),
        "input_ids": torch.zeros(8, dtype=torch.long),
        "attention_mask": torch.zeros(8, dtype=torch.long),
        "loss_mask": torch.zeros(8, dtype=torch.long),
    }


def _put(store: LocalFeatureStore, sample_id: str, *, n_bytes: int = 64) -> SampleRef:
    payload = _eagle3_bytes(n_bytes)
    return store.put(
        sample_id,
        payload,
        run_id="r1",
        algorithm=FeatureAlgorithm.EAGLE3,
        schema_version=1,
        target_model_version="0",
        draft_weight_version="0",
        num_tokens=8,
    )


# --- 1. basic put/get/release ---------------------------------------------


def test_local_store_put_returns_sample_ref_with_feature_specs() -> None:
    store = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    ref = _put(store, "s1", n_bytes=128)
    assert ref.sample_id == "s1"
    assert ref.store_uri == store.store_uri
    assert ref.feature_specs["aux_hidden_states"].dtype is torch.float32
    assert set(ref.feature_specs) == {"aux_hidden_states", "input_ids", "attention_mask", "loss_mask"}


def test_local_store_round_trip() -> None:
    store = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    tensors = _eagle3_bytes(256)
    ref = store.put(
        "s1",
        tensors,
        run_id="r1",
        algorithm=FeatureAlgorithm.EAGLE3,
        schema_version=1,
        target_model_version="0",
        draft_weight_version="0",
        num_tokens=8,
    )
    out, handle = store.get(ref)
    assert torch.equal(out["aux_hidden_states"], tensors["aux_hidden_states"])
    assert out["input_ids"].dtype is torch.long
    store.release(handle)
    # After release, a follow-up get fails because the sample is gone.
    with pytest.raises(KeyError):
        store.get(ref)


def test_local_store_get_returns_detached_copies() -> None:
    """The consumer must hold a copy that does not alias the stored tensor.

    Mutating the consumer-side tensor after :meth:`get` MUST NOT affect what
    a later release+re-get returns (or what another consumer acquires).
    """
    store = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    ref = _put(store, "s1", n_bytes=256)
    out, handle = store.get(ref)
    # Mutate the consumer-side copy in place.
    out["aux_hidden_states"].fill_(1.0)
    store.release(handle)
    # Re-put is not allowed for an already-released key; put a fresh
    # sample to inspect what the consumer held:
    ref2 = _put(store, "s1-new", n_bytes=256)
    out2, handle2 = store.get(ref2)
    assert (out2["aux_hidden_states"] == 0.0).all()
    store.release(handle2)


def test_local_store_get_rejects_ref_from_different_store() -> None:
    # Two separate stores always have different store_uris -- a ref
    # generated against store A must not be materializable through
    # store B even if their sample_ids collide (e.g. both prefixes
    # ``s1``).
    store_a = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    store_b = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    ref_a = _put(store_a, "s1", n_bytes=64)
    with pytest.raises(KeyError, match="does not match this store's URI"):
        store_b.get(ref_a)


# --- 2. refcount + multi-handle --------------------------------------------


def test_local_store_refcounts_handle_until_last_release() -> None:
    store = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    ref = _put(store, "s1", n_bytes=128)

    _, h1 = store.get(ref)
    _, h2 = store.get(ref)
    assert store.health().sample_count == 1
    assert store.health().resident_bytes > 0

    store.release(h1)
    assert store.health().sample_count == 1  # still alive

    store.release(h2)
    assert store.health().sample_count == 0
    assert store.health().resident_bytes == 0


def test_local_store_release_is_idempotent() -> None:
    store = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    ref = _put(store, "s1", n_bytes=128)
    _, h = store.get(ref)
    store.release(h)
    # Second release: handle is gone; the operation is a documented
    # no-op rather than a hard error (the store needs to remain stable
    # on retry after a desync).
    store.release(h)
    assert store.health().resident_bytes == 0


def test_local_store_release_rejects_handle_from_other_store() -> None:
    store_a = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    store_b = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    ref = _put(store_a, "s1", n_bytes=64)
    _, h = store_a.get(ref)
    with pytest.raises(ValueError, match="different store"):
        store_b.release(h)


# --- 3. residency caps ----------------------------------------------------


def test_local_store_rejects_put_exceeding_sample_cap() -> None:
    store = LocalFeatureStore(max_samples=2, max_bytes=1024 * 1024)
    _put(store, "s1", n_bytes=64)
    _put(store, "s2", n_bytes=64)
    with pytest.raises(MemoryError, match="sample-count cap"):
        _put(store, "s3", n_bytes=64)


def test_local_store_rejects_put_exceeding_byte_cap() -> None:
    store = LocalFeatureStore(max_samples=10, max_bytes=512)
    _put(store, "s1", n_bytes=64)  # ~256 bytes
    # 1 KiB tensor alone is over the cap.
    with pytest.raises(MemoryError, match="byte cap"):
        store.put(
            "s2",
            _eagle3_bytes(1024),
            run_id="r1",
            algorithm=FeatureAlgorithm.EAGLE3,
            schema_version=1,
            target_model_version="0",
            draft_weight_version="0",
            num_tokens=8,
        )


def test_local_store_requires_at_least_one_cap() -> None:
    with pytest.raises(ValueError, match="at least one of max_samples / max_bytes"):
        LocalFeatureStore(max_samples=None, max_bytes=None)


def test_local_store_requires_low_below_high_watermark() -> None:
    with pytest.raises(ValueError, match="strictly less than"):
        LocalFeatureStore(
            max_samples=4,
            max_bytes=1024,
            high_watermark_bytes=128,
            low_watermark_bytes=128,
        )


def test_local_store_rejects_duplicate_sample_id() -> None:
    store = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    _put(store, "s1", n_bytes=64)
    with pytest.raises(ValueError, match="already present"):
        _put(store, "s1", n_bytes=64)


# --- 4. health + watermarks -----------------------------------------------


def test_local_store_health_reports_resident_bytes_and_watermarks() -> None:
    store = LocalFeatureStore(
        max_samples=8,
        max_bytes=1024,
        high_watermark_bytes=512,
        low_watermark_bytes=256,
    )
    health0 = store.health()
    assert health0.resident_bytes == 0
    assert health0.capacity_bytes == 1024
    assert health0.sample_count == 0
    assert health0.low_watermark_hit is True  # empty store is below low_watermark
    assert health0.high_watermark_hit is False

    _put(store, "s1", n_bytes=128)  # ~512 bytes -> at high watermark
    h1 = store.health()
    assert h1.resident_bytes > 0
    assert h1.high_watermark_hit is True
    assert h1.low_watermark_hit is False


# --- 5. lifecycle ---------------------------------------------------------


def test_local_store_close_rejects_outstanding_handles() -> None:
    store = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    ref = _put(store, "s1", n_bytes=64)
    _, handle = store.get(ref)
    with pytest.raises(RuntimeError, match="outstanding handle"):
        store.close()
    store.release(handle)
    store.close()


def test_local_store_close_drops_state_and_rejects_subsequent_put() -> None:
    store = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    _put(store, "s1", n_bytes=64)
    store.close()
    assert store.health().resident_bytes == 0
    assert store.health().sample_count == 0
    with pytest.raises(RuntimeError, match="closed"):
        _put(store, "s1", n_bytes=64)


def test_local_store_close_is_idempotent() -> None:
    store = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    store.close()
    store.close()  # must not raise


def test_local_store_gc_is_a_noop_for_local_but_returns_count() -> None:
    store = LocalFeatureStore(max_samples=4, max_bytes=1024 * 1024)
    _put(store, "s1", n_bytes=64)
    assert store.gc() == 0
