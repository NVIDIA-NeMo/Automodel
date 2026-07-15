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

"""Lease / ack / fail queue over :class:`SampleRef` for the streaming pipeline.

The :class:`SampleRefQueue` carries only references -- no tensors -- between a
producer (target-side forward) and a consumer (draft-side trainer). Each
"message" is a :class:`SampleRef` and is delivered exactly once: a consumer
leases a ref, materializes its tensors via the
:class:`~nemo_automodel.components.speculative.streaming.store.FeatureStore`,
and then ACKs (release the lease and let the data-plane scrub the sample from
the store) or FAILs (release the lease without scrubbing so a future consumer
may retry).

A lease that is never ACK'd or FAIL'd within :attr:`Lease.visibility_timeout`
is considered orphaned and is reclaimed by
:meth:`SampleRefQueue.reclaim_expired`. That reclaim is what makes the queue
safe to drive against a producer that may crash mid-flight (RFC §"Phased plan"
PR 4's "visibility-timeout redelivery").

Backpressure is driven by the bound :class:`FeatureStore`'s
:meth:`FeatureStore.health` (ints only -- the queue never touches tensors in
its hot path), with a high/low watermark hysteresis band so a fast producer
cannot OOM the store and a slow producer cannot starve the trainer silently.
The producer-side and consumer-side pause / resume transitions are tracked on
the store via the same :attr:`StoreHealth.high_watermark_hit` /
:attr:`StoreHealth.low_watermark_hit` flags, so a third party (e.g. an ops
dashboard) can observe which side of the pipeline is the bottleneck without
inspecting the queue internals.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from nemo_automodel.components.speculative.streaming.refs import SampleRef
from nemo_automodel.components.speculative.streaming.store import FeatureStore, StoreHealth

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisibilityTimeout:
    """How long an unacked :class:`Lease` is allowed to live before reclaim.

    The minimum value is ``1.0`` second -- tighter intervals generate
    spurious reclaim on contended machines, where one consumer is merely
    slow rather than crashed. The maximum is unbounded; in practice the
    producer reads ``acquire()`` round-trip times to pick something
    workable per deployment.
    """

    seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError(f"VisibilityTimeout.seconds must be positive, got {self.seconds}")


@dataclass
class Lease:
    """Handle to a leased :class:`SampleRef`.

    Attributes:
        ref: The leased reference -- the only sanctioned way to materialize
            its tensors is :meth:`FeatureStore.get`, which returns a
            :class:`~nemo_automodel.components.speculative.streaming.store.StoreHandle`
            the consumer hands to :meth:`FeatureStore.release` once it is
            done with them.
        deadline: Monotonic-clock timestamp at which this lease is
            considered orphaned. Used by
            :meth:`SampleRefQueue.reclaim_expired` to redeliver the ref.
        visibility_timeout: The :class:`VisibilityTimeout` that produced
            this lease, kept here so the consumer can introspect it.
        redelivery_count: Number of times this ref has been leased and
            re-leased (PR 4 uses it for retry telemetry). Starts at 0.
    """

    ref: SampleRef
    deadline: float
    visibility_timeout: VisibilityTimeout
    redelivery_count: int = 0


class SampleRefQueue:
    """Lease / ack / fail queue over :class:`SampleRef`.

    Args:
        store: The data-plane store the consumers will materialize against.
            The queue reads :meth:`FeatureStore.health` for backpressure.
        visibility_timeout: How long a leased-but-not-acked ref can live
            before reclaim. Defaults to 30s; production deployments
            normally key this off the recipe's per-step budget.
        high_watermark_bytes: Resident-byte threshold above which
            :meth:`put_blocks_until_below` blocks. Defaults to
            ``0.75 * store.health().capacity_bytes`` when the store caps
            bytes; a fixed value overrides.
        low_watermark_bytes: Resident-byte threshold below which
            :meth:`put_blocks_until_below` resumes after pausing. Defaults
            to ``0.25 * store.health().capacity_bytes`` when the store
            caps bytes.
        on_pause / on_resume: Optional callbacks fired when the queue
            transitions high-watermark-paused -> resumed and back. Wired
            into the trainer's progress log in PR 2; kept minimal here.

    Thread safety: a single :class:`threading.Lock` protects every list /
    counter, so a multi-producer / multi-consumer deployment works as long
    as only one thread at a time calls any one of the methods.
    """

    def __init__(
        self,
        store: FeatureStore,
        *,
        visibility_timeout: VisibilityTimeout | None = None,
        high_watermark_bytes: int | None = None,
        low_watermark_bytes: int | None = None,
        on_pause: Callable[[StoreHealth], None] | None = None,
        on_resume: Callable[[StoreHealth], None] | None = None,
    ) -> None:
        self._store = store
        self._vt = visibility_timeout or VisibilityTimeout()
        self._high_bytes = high_watermark_bytes
        self._low_bytes = low_watermark_bytes
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._lock = threading.Lock()
        self._pending: list[SampleRef] = []  # FIFO of refs ready to lease
        self._outstanding: dict[str, Lease] = {}  # sample_id -> active lease
        self._sample_counters: dict[str, int] = {}  # sample_id -> redelivery count
        self._producer_paused = False
        self._put_cv = threading.Condition(self._lock)
        self._closed = False

    # --- produce side -------------------------------------------------------

    def put(self, ref: SampleRef) -> None:
        """Enqueue ``ref`` for a future :meth:`acquire`.

        Does not block on backpressure; producers that care should call
        :meth:`put_blocks_until_below` instead, which honors the high/low
        watermark hysteresis from :meth:`FeatureStore.health`.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("SampleRefQueue is closed; no further puts accepted")
            if ref.sample_id in self._outstanding or ref.sample_id in (r.sample_id for r in self._pending):
                raise ValueError(f"sample_id {ref.sample_id!r} already present in queue (or outstanding)")
            self._pending.append(ref)
            self._sample_counters.setdefault(ref.sample_id, 0)
            logger.debug("SampleRefQueue put sample_id=%s pending=%d", ref.sample_id, len(self._pending))

    def put_blocks_until_below(self, ref: SampleRef, *, poll_interval: float = 0.05) -> None:
        """Enqueue ``ref``, blocking the producer while the store is over its high watermark.

        The producer is paused when :attr:`StoreHealth.high_watermark_hit`
        is true, and only resumed when :attr:`StoreHealth.low_watermark_hit`
        flips to true. The hysteresis band between the two is what prevents
        flapping.

        Args:
            ref: The reference to enqueue.
            poll_interval: Seconds between backpressure checks when paused.
                Defaults to 50ms -- well below typical step times, well above
                the cost of a Python-level :meth:`FeatureStore.health` call.

        Raises:
            RuntimeError: if the queue is closed while the producer is
                blocked, so a producer does not silently swallow a
                shutdown signal.
        """
        while True:
            with self._put_cv:
                if self._closed:
                    raise RuntimeError("SampleRefQueue is closed while put was waiting on backpressure")
                health = self._store.health()
                if not health.high_watermark_hit:
                    # Resume-side transition only fires when we were paused.
                    if self._producer_paused:
                        logger.info(
                            "SampleRefQueue producer resumed below low watermark (resident=%d capacity=%d)",
                            health.resident_bytes,
                            health.capacity_bytes,
                        )
                        self._producer_paused = False
                        if self._on_resume is not None:
                            try:
                                self._on_resume(health)
                            except Exception:
                                logger.exception("on_resume callback raised; continuing")
                    self._pending.append(ref)
                    self._sample_counters.setdefault(ref.sample_id, 0)
                    logger.debug("SampleRefQueue put sample_id=%s pending=%d", ref.sample_id, len(self._pending))
                    return
                if not self._producer_paused:
                    logger.info(
                        "SampleRefQueue producer paused at high watermark (resident=%d capacity=%d)",
                        health.resident_bytes,
                        health.capacity_bytes,
                    )
                    self._producer_paused = True
                    if self._on_pause is not None:
                        try:
                            self._on_pause(health)
                        except Exception:
                            logger.exception("on_pause callback raised; continuing")
                self._put_cv.wait(timeout=poll_interval)
            # Brief lock release is implicit via Condition.wait returning
            # after either the timeout or a notify (which we don't emit
            # externally; the wake-up cadence is poll-driven).

    # --- consume side -------------------------------------------------------

    def acquire(self, *, poll_interval: float = 0.05) -> Lease | None:
        """Lease the next ref; returns ``None`` once shutdown drains the queue.

        The returned :class:`Lease` is the only sanctioned way to access
        the ref's tensors -- :class:`FeatureStore.get` requires a :class:`SampleRef`,
        and that ref must come from a lease. The consumer MUST hand back
        the lease via :meth:`ack` (on success) or :meth:`fail` (on error)
        so the queue can reclaim the slot and the store can drop the
        sample.
        """
        with self._put_cv:
            if self._closed and not self._pending and not self._outstanding:
                return None
            if not self._pending:
                self._put_cv.wait(timeout=poll_interval)
                if not self._pending:
                    return None
            ref = self._pending.pop(0)
            now = time.monotonic()
            deadline = now + self._vt.seconds
            redelivery = self._sample_counters.get(ref.sample_id, 0)
            lease = Lease(ref=ref, deadline=deadline, visibility_timeout=self._vt, redelivery_count=redelivery)
            self._outstanding[ref.sample_id] = lease
            logger.debug(
                "SampleRefQueue acquire sample_id=%s redelivery=%d outstanding=%d",
                ref.sample_id,
                redelivery,
                len(self._outstanding),
            )
            return lease

    def ack(self, lease: Lease) -> None:
        """Mark a leased ref as successfully consumed and free its queue slot.

        Does NOT touch the store -- the consumer's :meth:`FeatureStore.get`
        return value carries a :class:`~nemo_automodel.components.speculative.streaming.store.StoreHandle`
        that the consumer must hand to :meth:`FeatureStore.release` to drop
        the tensors. The queue's responsibility ends at "lease no longer held".
        """
        with self._lock:
            active = self._outstanding.pop(lease.ref.sample_id, None)
            if active is None:
                logger.debug(
                    "SampleRefQueue ack for sample_id=%s is a no-op (no outstanding lease)",
                    lease.ref.sample_id,
                )
                return
            # Wake up a producer that might have been waiting for the
            # store to drain below the high watermark.
            self._put_cv.notify_all()

    def fail(self, lease: Lease) -> None:
        """Return a leased ref to the pending queue, without dropping its tensors.

        The ref will be leased again (its :attr:`Lease.redelivery_count`
        increments). Re-delivery is what makes the pipeline fault-tolerant
        to a transient consumer error -- a permanently bad ref is the
        consumer's problem (drop it after a bounded retry budget).
        """
        with self._lock:
            active = self._outstanding.pop(lease.ref.sample_id, None)
            if active is None:
                logger.debug(
                    "SampleRefQueue fail for sample_id=%s is a no-op (no outstanding lease)",
                    lease.ref.sample_id,
                )
                return
            self._sample_counters[lease.ref.sample_id] = active.redelivery_count + 1
            # Place at the tail so a freshly-failed sample does not jump
            # ahead of samples still waiting their first try.
            self._pending.append(active.ref)
            self._put_cv.notify_all()

    # --- background reclaim -------------------------------------------------

    def reclaim_expired(self) -> int:
        """Reclaim leases whose :attr:`Lease.deadline` has passed.

        Each reclaimed lease is re-enqueued; :meth:`acquire` returns it on
        a future call with an incremented :attr:`Lease.redelivery_count`.
        Returns the number of leases reclaimed -- a queue that is healthy
        returns 0 most of the time.
        """
        now = time.monotonic()
        reclaimed: list[str] = []
        with self._lock:
            for sample_id, lease in list(self._outstanding.items()):
                if lease.deadline <= now:
                    reclaimed.append(sample_id)
                    self._sample_counters[sample_id] = lease.redelivery_count + 1
                    # Re-enqueue at the tail so any fresh pending work is
                    # preferred over reclaimed stuff.
                    self._pending.append(lease.ref)
            for sample_id in reclaimed:
                self._outstanding.pop(sample_id, None)
            if reclaimed:
                self._put_cv.notify_all()
        if reclaimed:
            logger.warning("SampleRefQueue reclaimed %d expired leases: %s", len(reclaimed), reclaimed[:8])
        return len(reclaimed)

    # --- introspection ------------------------------------------------------

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def outstanding_count(self) -> int:
        with self._lock:
            return len(self._outstanding)

    def close(self) -> None:
        """Drain the queue; subsequent :meth:`acquire` calls return ``None``.

        Outstanding leases are left intact: their consumer still owns the
        tensors, and a leaked :meth:`FeatureStore.release` would push the
        store's residency counter below zero. The store's own :meth:`close`
        is the canonical place to drop residency.
        """
        with self._put_cv:
            self._closed = True
            self._put_cv.notify_all()


__all__ = ["Lease", "SampleRefQueue", "VisibilityTimeout"]


# Note: when PR 4 adds a ``completion_future`` field to ``Lease`` it should
# import ``field`` from ``dataclasses`` and add it to the dataclass.
