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

"""CUDA graph for fixed-address TE-ops MXFP8 cache refreshes.

The optimizer remains eager. This manager captures only the deterministic
post-step copy/quantization kernels that refresh preallocated MXFP8 expert
weight caches from their ordinary BF16 parameter owners.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from nemo_automodel.recipes.llm.full_iteration_cuda_graph import (
    _distributed_capture_barrier,
    _get_graph_pool,
    _get_shared_capture_stream,
)


class MXFP8CacheRefreshCudaGraphError(RuntimeError):
    """Raised when the cache-refresh graph cannot be replayed safely."""


class MXFP8CacheRefreshTarget(Protocol):
    """Fixed-storage cache refresh operations consumed by the graph manager."""

    @property
    def managed_owner_ids(self) -> frozenset[int]:
        """Return parameter identities whose post-step refresh this target owns."""

    def graph_signature(self) -> tuple[Any, ...]:
        """Return every owner and destination identity retained by capture."""

    def eager_refresh(self) -> int:
        """Refresh all caches eagerly and return the number refreshed."""

    def capture_refresh(self) -> None:
        """Launch only graph-capturable refresh kernels."""

    def mark_replayed(self) -> int:
        """Advance Python generation state after one actual graph launch."""


class MXFP8CacheRefreshCudaGraphManager:
    """Capture and replay a fixed-address MXFP8 cache refresh target.

    The target signature is immutable for the manager lifetime. A parameter,
    cache wrapper, or destination-buffer replacement destroys any installed
    graph and poisons the manager before stale addresses can be launched.
    """

    def __init__(self, target: MXFP8CacheRefreshTarget, *, use_single_mempool: bool = True) -> None:
        if not isinstance(use_single_mempool, bool):
            raise TypeError("use_single_mempool must be a boolean")
        self.target = target
        self.use_single_mempool = use_single_mempool
        self._signature = target.graph_signature()
        if not self._signature:
            raise ValueError("MXFP8 cache-refresh graph requires a non-empty target signature")
        self._graph: Any = None
        self._failed_reason: str | None = None
        self._closed = False
        self.capture_count = 0
        self.replay_count = 0
        self.eager_refresh_count = 0

    @property
    def is_captured(self) -> bool:
        """Whether a valid cache-refresh graph is installed."""
        return self._graph is not None and self._failed_reason is None and not self._closed

    @property
    def managed_owner_ids(self) -> frozenset[int]:
        """Return parameters whose ordinary optimizer hooks must be skipped."""
        return self.target.managed_owner_ids

    def _destroy_graph(self) -> None:
        graph = self._graph
        if graph is not None:
            reset = getattr(graph, "reset", None)
            if callable(reset):
                reset()
        # Retain the live graph when reset() fails. Releasing it after a failed
        # CUDA teardown could free cache storage still referenced by capture.
        self._graph = None

    def _poison(self, reason: str) -> None:
        if self._graph is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._destroy_graph()
        self._failed_reason = reason

    def _validate_signature(self) -> None:
        current = self.target.graph_signature()
        if current != self._signature:
            raise MXFP8CacheRefreshCudaGraphError(
                "TE-ops MXFP8 cache owner or fixed-address destination changed after graph setup"
            )

    def _capture_and_replay(self) -> int:
        graph = torch.cuda.CUDAGraph()
        self._graph = graph
        try:
            current_stream = torch.cuda.current_stream()
            capture_stream = _get_shared_capture_stream()
            capture_stream.wait_stream(current_stream)
            torch.cuda.synchronize()
            _distributed_capture_barrier()
            with torch.cuda.graph(
                graph,
                stream=capture_stream,
                pool=_get_graph_pool(self.use_single_mempool),
                capture_error_mode="thread_local",
            ):
                self.target.capture_refresh()
            torch.cuda.synchronize()
            _distributed_capture_barrier()
            graph.replay()
            current_stream.wait_stream(capture_stream)
            refreshed = self.target.mark_replayed()
        except Exception:
            self._destroy_graph()
            raise
        self.capture_count += 1
        return refreshed

    def _replay(self) -> int:
        current_stream = torch.cuda.current_stream()
        capture_stream = _get_shared_capture_stream()
        capture_stream.wait_stream(current_stream)
        self._graph.replay()
        current_stream.wait_stream(capture_stream)
        refreshed = self.target.mark_replayed()
        self.replay_count += 1
        return refreshed

    def run(self, *, capture_allowed: bool) -> int:
        """Refresh after one optimizer step, capturing only when F/B is captured."""
        if self._closed:
            raise MXFP8CacheRefreshCudaGraphError("MXFP8 cache-refresh CUDA graph manager is closed")
        if self._failed_reason is not None:
            raise MXFP8CacheRefreshCudaGraphError(
                f"MXFP8 cache-refresh CUDA graph manager is failed closed; reason: {self._failed_reason}"
            )
        if not isinstance(capture_allowed, bool):
            raise TypeError("capture_allowed must be a boolean")

        try:
            self._validate_signature()
            if not capture_allowed:
                if self._graph is not None:
                    self.reset()
                    self._validate_signature()
                refreshed = self.target.eager_refresh()
                self.eager_refresh_count += 1
                self._validate_signature()
                return refreshed
            if not torch.cuda.is_available():
                raise MXFP8CacheRefreshCudaGraphError("MXFP8 cache-refresh CUDA graphs require CUDA")
            if self._graph is None:
                return self._capture_and_replay()
            return self._replay()
        except Exception as error:
            reason = str(error) or type(error).__name__
            self._poison(reason)
            if isinstance(error, MXFP8CacheRefreshCudaGraphError):
                raise
            raise MXFP8CacheRefreshCudaGraphError(f"MXFP8 cache-refresh CUDA graph failed: {reason}") from error

    def reset(self) -> None:
        """Destroy capture while retaining the immutable cache signature."""
        if self._closed:
            return
        if torch.cuda.is_available() and self._graph is not None:
            torch.cuda.synchronize()
        self._destroy_graph()
        self._failed_reason = None
        self.capture_count = 0
        self.replay_count = 0

    def close(self) -> None:
        """Idempotently destroy the graph and reject future refreshes."""
        if self._closed:
            return
        self.reset()
        self._closed = True
