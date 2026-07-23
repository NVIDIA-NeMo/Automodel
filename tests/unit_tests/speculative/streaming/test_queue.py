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

"""Unit tests for :class:`SampleRefQueue`.

The queue is the only control-plane coupler between producers and consumers;
its guarantees are:

1. FIFO lease ordering across multiple producers / consumers within a
   single process.
2. :class:`VisibilityTimeout` reclaim: a lease whose deadline has passed
   is re-deliverable, with the redelivery count incremented.
3. Backpressure via :meth:`FeatureStore.health` (ints only).
4. :meth:`fail` re-enqueues a lease for retry without dropping its
   tensors; :meth:`ack` is the only sanctioned way to free the lease.
5. :meth:`close` drains the queue (subsequent acquires return ``None``).

These tests run CPU-only and exercise the API directly rather than through
the trainer. PR 2's :class:`FeatureDataLoader` consumer integration is the
end-to-end coverage for the recipe-side path.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from nemo_automodel.components.speculative.streaming import (
    FeatureAlgorithm,
    LocalFeatureStore,
    SampleRef,
    SampleRefQueue,
    VisibilityTimeout,
)


def _eagle3_bytes(n_floats: int) -> dict[str, torch.Tensor]:
    return {
        "aux_hidden_states": torch.zeros(n_floats, dtype=torch.float32),
        "input_ids": torch.zeros(8, dtype=torch.long),
        "attention_mask": torch.zeros(8, dtype=torch.long),
        "loss_mask": torch.zeros(8, dtype=torch.long),
    }


def _put(store: LocalFeatureStore, sample_id: str, *, n_floats: int = 64) -> SampleRef:
    return store.put(
        sample_id,
        _eagle3_bytes(n_floats),
        run_id="r1",
        algorithm=FeatureAlgorithm.EAGLE3,
        schema_version=1,
        target_model_version="0",
        draft_weight_version="0",
        num_tokens=8,
    )


@pytest.fixture
def store() -> LocalFeatureStore:
    return LocalFeatureStore(
        max_samples=16,
        max_bytes=4 * 1024 * 1024,
        high_watermark_bytes=3 * 1024 * 1024,
        low_watermark_bytes=1 * 1024 * 1024,
    )


# --- 1. ordering + acquire/ack --------------------------------------------


def test_queue_put_then_acquire_yields_first_put_ref(store: LocalFeatureStore) -> None:
    ref_a = _put(store, "a")
    ref_b = _put(store, "b")
    q = SampleRefQueue(store)
    q.put(ref_a)
    q.put(ref_b)
    lease1 = q.acquire()
    lease2 = q.acquire()
    assert lease1 is not None and lease1.ref.sample_id == "a"
    assert lease2 is not None and lease2.ref.sample_id == "b"
    q.ack(lease1)
    q.ack(lease2)


def test_queue_acquire_returns_none_when_empty_and_drained(store: LocalFeatureStore) -> None:
    q = SampleRefQueue(store)
    q.close()
    assert q.acquire() is None


def test_queue_ack_is_idempotent(store: LocalFeatureStore) -> None:
    ref = _put(store, "s1")
    q = SampleRefQueue(store)
    q.put(ref)
    lease = q.acquire()
    assert lease is not None
    q.ack(lease)
    # No-op, not an error.
    q.ack(lease)
    # Lease is no longer in outstanding.
    assert q.outstanding_count() == 0


# --- 2. fail / redelivery -------------------------------------------------


def test_queue_fail_re_enqueues_for_retry_without_dropping_tensors(store: LocalFeatureStore) -> None:
    ref = _put(store, "s1")
    q = SampleRefQueue(store)
    q.put(ref)
    lease1 = q.acquire()
    assert lease1 is not None
    assert lease1.redelivery_count == 0
    q.fail(lease1)
    # Lease is back in pending; a fresh acquire hands out the same ref
    # with a bumped redelivery_count.
    lease2 = q.acquire()
    assert lease2 is not None
    assert lease2.ref.sample_id == "s1"
    assert lease2.redelivery_count == 1
    q.ack(lease2)


# --- 3. visibility-timeout reclaim ---------------------------------------


def test_queue_reclaims_expired_leases_and_increments_redelivery_count(
    store: LocalFeatureStore,
) -> None:
    ref = _put(store, "s1")
    q = SampleRefQueue(store, visibility_timeout=VisibilityTimeout(seconds=0.05))
    q.put(ref)
    lease1 = q.acquire()
    assert lease1 is not None and lease1.redelivery_count == 0

    # Wait past the visibility timeout without acking.
    time.sleep(0.10)
    reclaimed = q.reclaim_expired()
    assert reclaimed == 1
    assert q.outstanding_count() == 0
    assert q.pending_count() == 1

    # A future acquire returns the same ref with a bumped count.
    lease2 = q.acquire()
    assert lease2 is not None
    assert lease2.redelivery_count == 1
    q.ack(lease2)


def test_queue_does_not_reclaim_unexpired_leases(store: LocalFeatureStore) -> None:
    ref = _put(store, "s1")
    q = SampleRefQueue(store, visibility_timeout=VisibilityTimeout(seconds=10.0))
    q.put(ref)
    lease = q.acquire()
    assert lease is not None
    assert q.reclaim_expired() == 0
    assert q.outstanding_count() == 1
    q.ack(lease)


# --- 4. duplicate-id enforcement -----------------------------------------


def test_queue_rejects_duplicate_pending_sample_id(store: LocalFeatureStore) -> None:
    ref = _put(store, "s1")
    q = SampleRefQueue(store)
    q.put(ref)
    with pytest.raises(ValueError, match="already present"):
        q.put(ref)


def test_queue_rejects_duplicate_while_outstanding(store: LocalFeatureStore) -> None:
    ref_a = _put(store, "s1")
    ref_b = _put(store, "s2")
    q = SampleRefQueue(store)
    q.put(ref_a)
    lease = q.acquire()
    assert lease is not None
    q.put(ref_b)  # fine -- different sample_id
    with pytest.raises(ValueError, match="already present"):
        q.put(ref_a)
    q.ack(lease)


# --- 5. backpressure -----------------------------------------------------


def test_queue_put_blocks_until_store_drops_below_low_watermark() -> None:
    """A put that crosses the high watermark blocks the producer until the consumer drains."""
    big_store = LocalFeatureStore(
        max_samples=16,
        max_bytes=512 * 1024,
        high_watermark_bytes=256 * 1024,
        low_watermark_bytes=64 * 1024,
    )
    # Warm the store so it is over the low watermark after the producer
    # puts a second sample.
    _put(big_store, "warm", n_floats=4 * 1024)  # ~16 KB
    q = SampleRefQueue(big_store)

    ref_pause = _put(big_store, "p1", n_floats=64 * 1024 - 4 * 1024)  # ~252 KB
    # The store is now well over the high watermark.
    assert big_store.health().high_watermark_hit

    paused = threading.Event()

    def producer() -> None:
        ref_extra = _put(big_store, "p2", n_floats=8 * 1024)
        # put_blocks_until_below should pause: resident is currently
        # ~268 KB > high_watermark of 256 KB. Background drainer (below)
        # frees space and unblocks the producer.
        paused.set()
        q.put_blocks_until_below(ref_extra, poll_interval=0.01)

    def drainer() -> None:
        paused.wait()
        time.sleep(0.05)
        # Drain "warm" + "p1" out via get/release.
        out, h = big_store.get(ref_pause)
        # Also drop "warm" so the cap calculation gets a fresh post-state.
        ref_warm = _put(big_store, "_discard", n_floats=1)
        out2, h2 = big_store.get(ref_warm)
        big_store.release(h2)
        big_store.release(h)
        # Resident bytes are now ~16 KB < low_watermark of 64 KB.

    t_prod = threading.Thread(target=producer)
    t_drain = threading.Thread(target=drainer)
    t_prod.start()
    t_drain.start()
    t_prod.join(timeout=5)
    t_drain.join(timeout=5)
    assert not t_prod.is_alive(), "producer thread did not finish in time"
    assert q.pending_count() == 1


def test_queue_pause_and_resume_callbacks_fire_on_watermark_transitions() -> None:
    big_store = LocalFeatureStore(
        max_samples=16,
        max_bytes=512 * 1024,
        high_watermark_bytes=128 * 1024,
        low_watermark_bytes=32 * 1024,
    )
    ref_warm = _put(big_store, "warm", n_floats=4 * 1024)  # ~16 KB
    pause_log = []
    resume_log = []

    q = SampleRefQueue(
        big_store,
        on_pause=lambda h: pause_log.append(h.resident_bytes),
        on_resume=lambda h: resume_log.append(h.resident_bytes),
    )
    ref_pause = _put(big_store, "p1", n_floats=32 * 1024 - 4 * 1024)  # push over high
    assert big_store.health().high_watermark_hit
    ref_extra = _put(big_store, "extra", n_floats=4 * 1024)

    def drainer() -> None:
        time.sleep(0.05)
        # Drop both warm and p1 so the post-state resident bytes drop
        # strictly below the low watermark -- that is what the
        # hysteresis-aware resume condition needs. With only p1
        # released, resident lands at ~32 KiB which equals the low
        # watermark and the producer correctly stays paused.
        _, h_warm = big_store.get(ref_warm)
        _, h_p1 = big_store.get(ref_pause)
        big_store.release(h_p1)
        big_store.release(h_warm)

    t = threading.Thread(target=drainer)
    t.start()
    q.put_blocks_until_below(ref_extra, poll_interval=0.01)
    t.join(timeout=5)

    assert pause_log, "on_pause did not fire while store was at high watermark"
    assert resume_log, "on_resume did not fire after the store drained"
    # The producer paused with the store above its high watermark and
    # resumed only after the consumer drained enough to cross the low
    # watermark. The two callbacks report monotonically-shifted resident
    # bytes: the resume reading MUST be strictly less than the pause
    # reading, otherwise the hysteresis is broken.
    assert pause_log[0] > resume_log[-1]
    assert pause_log[0] >= 128 * 1024  # paused for the configured high watermark
    # After releasing both warm and p1, only ``extra`` is left (~16 KiB);
    # the resume callback fires at resident_bytes <= low_watermark_bytes.
    assert resume_log[-1] <= 32 * 1024


# --- 6. close ------------------------------------------------------------


def test_queue_close_makes_acquire_return_none(store: LocalFeatureStore) -> None:
    q = SampleRefQueue(store)
    q.close()
    assert q.acquire() is None


def test_queue_is_closed_property_distinguishes_lifecycle_from_empty_poll(
    store: LocalFeatureStore,
) -> None:
    """Consumers reading ``acquire() == None`` need to know which case they hit.

    ``None`` on a still-open queue means "transient empty, retry";
    ``None`` on a closed queue means "drained, stop". :attr:`is_closed`
    is the canonical signal -- the property returns ``False`` until
    :meth:`close` runs and ``True`` after.
    """
    q = SampleRefQueue(store)
    assert q.is_closed is False
    lease = q.acquire()
    assert lease is None
    assert q.is_closed is False  # empty poll, NOT shutdown

    q.close()
    assert q.is_closed is True
    lease = q.acquire()
    assert lease is None  # shutdown, drain done


# --- 7. ack counter cleanup -----------------------------------------------


def test_queue_ack_drops_sample_counter_entry(store: LocalFeatureStore) -> None:
    """``_sample_counters`` must not grow unbounded across long runs.

    Every successful ack (no pending redelivery) should remove the
    sample-id key from the counter dict. Putting + acking the same
    sample repeatedly must leave the counter empty.
    """
    q = SampleRefQueue(store)
    ref = _put(store, "s1", n_floats=64)
    q.put(ref)
    lease = q.acquire()
    assert lease is not None
    assert lease.redelivery_count == 0
    q.ack(lease)
    # The counter entry must be popped on ack; if it survived, a long
    # run with many unique sample_ids would leak unboundedly.
    assert "s1" not in q._sample_counters  # noqa: SLF001 -- inspecting internals on purpose


def test_queue_ack_after_fail_preserves_counter_until_terminal_ack(
    store: LocalFeatureStore,
) -> None:
    """``fail`` re-enqueues with an incremented counter; the entry stays
    until the final ack that drops the sample for good."""
    q = SampleRefQueue(store)
    ref = _put(store, "s1", n_floats=64)
    q.put(ref)
    lease1 = q.acquire()
    assert lease1 is not None
    q.fail(lease1)
    lease2 = q.acquire()
    assert lease2 is not None
    assert lease2.redelivery_count == 1
    q.ack(lease2)
    assert "s1" not in q._sample_counters  # noqa: SLF001


# --- 8. explicit high/low_watermark_bytes ctor args ------------------------


def test_queue_with_explicit_watermark_bytes_overrides_store_defaults() -> None:
    """The queue's ctor thresholds must take effect, not silently defer
    to the store's defaults."""
    store = LocalFeatureStore(
        max_samples=16,
        max_bytes=512 * 1024,
        # store defaults: pause at 384 KiB, resume at 64 KiB
        high_watermark_bytes=384 * 1024,
        low_watermark_bytes=64 * 1024,
    )
    # queue overrides: pause at 256 KiB, resume at 128 KiB
    SampleRefQueue(
        store,
        high_watermark_bytes=256 * 1024,
        low_watermark_bytes=128 * 1024,
    )
    # Fill to ~192 KiB: below the queue's high (256), above its low (128).
    # With deferred-to-store behavior, this would be PAUSED (192 > 64 = store's low is hit).
    # With explicit queue thresholds, this stays UNPAUSED.
    _put(store, "warm", n_floats=24 * 1024)  # ~96 KiB
    _put(store, "p1", n_floats=24 * 1024)  # total ~192 KiB
    pause_log = []
    q2 = SampleRefQueue(store, on_pause=lambda h: pause_log.append(h.resident_bytes))
    ref_extra = _put(store, "extra", n_floats=4 * 1024)
    # Producer should NOT pause: resident ~208 KiB, below queue high=256 KiB.
    q2.put(ref_extra)
    assert pause_log == []  # queue's explicit threshold was honored, not the store's


def test_queue_rejects_misconfigured_watermark_bytes() -> None:
    store = LocalFeatureStore()
    with pytest.raises(ValueError, match="strictly less than"):
        SampleRefQueue(
            store,
            high_watermark_bytes=128 * 1024,
            low_watermark_bytes=128 * 1024,
        )
    with pytest.raises(ValueError, match="strictly less than"):
        SampleRefQueue(
            store,
            high_watermark_bytes=128 * 1024,
            low_watermark_bytes=256 * 1024,
        )
    with pytest.raises(ValueError, match="positive"):
        SampleRefQueue(store, high_watermark_bytes=0)
    with pytest.raises(ValueError, match="non-negative"):
        SampleRefQueue(store, low_watermark_bytes=-1)


def test_queue_hysteresis_preserves_state_in_band(store: LocalFeatureStore) -> None:
    """A producer at resident just below high stays UNPAUSED; one at
    resident just above low stays PAUSED. The band itself preserves
    the producer's existing state, which is what prevents flapping."""
    _put(store, "warm", n_floats=4 * 1024)  # ~16 KiB; resident=16 KiB
    pause_log = []
    q2 = SampleRefQueue(
        store,
        high_watermark_bytes=128 * 1024,
        low_watermark_bytes=32 * 1024,
        on_pause=lambda h: pause_log.append(h.resident_bytes),
    )
    # Resident 16 KiB is below high AND above low: in the band.
    # Producer is not currently paused, so it stays unpaused.
    ref_extra = _put(store, "extra", n_floats=4 * 1024)
    q2.put(ref_extra)
    assert pause_log == []
    # Producer state stays unpaused; the next put should also succeed.
    ref_extra2 = _put(store, "extra2", n_floats=4 * 1024)
    q2.put(ref_extra2)
    assert pause_log == []


def test_queue_close_after_close_is_safe(store: LocalFeatureStore) -> None:
    q = SampleRefQueue(store)
    q.close()
    q.close()
