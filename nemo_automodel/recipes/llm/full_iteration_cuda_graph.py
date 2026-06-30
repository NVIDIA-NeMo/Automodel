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

"""Whole-forward/backward CUDA graph capture with fixed batch storage.

This module deliberately stops at the forward/backward boundary. Optimizer,
gradient-clipping, scheduler, checkpoint, and logging work remain eager so a
caller can validate the completed backward before updating parameters.
Dynamic values produced inside the model, including MoE routing splits, are
not recorded or canonicalized here. A backend must independently make every
operation inside the supplied callable CUDA-graph safe.
"""

from __future__ import annotations

import enum
import gc
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

_SHARED_CAPTURE_STREAM: Any = None
_SHARED_GRAPH_POOL: Any = None
logger = logging.getLogger(__name__)


class FullIterationCudaGraphError(RuntimeError):
    """Raised when a full-iteration graph cannot be captured or safely replayed."""


def _get_transformer_engine_rng_states() -> Any:
    """Return graph-aware TE RNG states without requiring TE at import time."""
    try:
        from transformer_engine.pytorch.distributed import get_all_rng_states
    except (ImportError, AttributeError):
        return ()
    return get_all_rng_states()


@dataclass(frozen=True)
class _TensorMetadata:
    """Source tensor properties that must not change between graph replays."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    layout: torch.layout
    stride: tuple[int, ...]
    requires_grad: bool
    device: torch.device

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> _TensorMetadata:
        return cls(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            layout=tensor.layout,
            stride=tuple(tensor.stride()),
            requires_grad=tensor.requires_grad,
            device=tensor.device,
        )


def _tensor_object_alias_pattern(tensors: Sequence[torch.Tensor]) -> tuple[int, ...]:
    """Encode repeated tensor objects without retaining their identities."""
    indices: dict[int, int] = {}
    pattern = []
    for tensor in tensors:
        tensor_id = id(tensor)
        if tensor_id not in indices:
            indices[tensor_id] = len(indices)
        pattern.append(indices[tensor_id])
    return tuple(pattern)


def _tensor_storage_alias_pattern(tensors: Sequence[torch.Tensor]) -> tuple[int, ...]:
    """Encode storage sharing between otherwise distinct tensor objects."""
    indices: dict[tuple[Any, ...], int] = {}
    pattern = []
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storage_key = (tensor.device, getattr(storage, "_cdata", id(storage)))
        if storage_key not in indices:
            indices[storage_key] = len(indices)
        pattern.append(indices[storage_key])
    return tuple(pattern)


def _same_control_value(expected: Any, actual: Any) -> bool:
    """Compare a non-tensor pytree leaf without invoking tensor-like equality."""
    if type(expected) is not type(actual):
        return False
    if expected is None or isinstance(expected, (bool, int, float, str, bytes, enum.Enum, torch.dtype, torch.device)):
        return bool(expected == actual)
    return expected is actual


def _get_shared_capture_stream() -> Any:
    """Return the process-wide stream shared by full-iteration captures."""
    global _SHARED_CAPTURE_STREAM
    if _SHARED_CAPTURE_STREAM is None:
        _SHARED_CAPTURE_STREAM = torch.cuda.Stream()
    return _SHARED_CAPTURE_STREAM


def _get_shared_graph_pool() -> Any:
    """Return the process-wide CUDA graph memory-pool handle."""
    global _SHARED_GRAPH_POOL
    if _SHARED_GRAPH_POOL is None:
        _SHARED_GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _SHARED_GRAPH_POOL


def _get_graph_pool(use_single_mempool: bool) -> Any:
    """Return a shared or capture-local CUDA graph memory pool."""
    if use_single_mempool:
        return _get_shared_graph_pool()
    return torch.cuda.graph_pool_handle()


def _distributed_capture_barrier() -> None:
    """Synchronize ranks at capture boundaries when distributed is active."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


@contextmanager
def _stale_capture_stream_override():
    """Temporarily let autograd replace stale streams during graph capture.

    PyTorch added this guard for AccumulateGrad nodes created by eager warmup
    on a stream that differs from the later capture stream. Older supported
    builds do not expose it, in which case capture retains its native checks.
    """
    setter = getattr(torch.autograd.graph, "set_override_stale_capture_stream", None)
    if not callable(setter):
        setter = getattr(torch._C, "_set_override_stale_capture_stream", None)
    if not callable(setter):
        yield
        return

    getter = getattr(torch._C, "_get_override_stale_capture_stream", None)
    if not callable(getter):
        getter = getattr(torch._C, "_override_stale_capture_stream", None)
    previous = bool(getter()) if callable(getter) else False
    setter(True)
    try:
        yield
    finally:
        setter(previous)


class _StaticCudaBatchBuffer:
    """Own a fixed-pytree set of strided tensors on one target device."""

    def __init__(self, batches: Sequence[Any], *, device: torch.device) -> None:
        if not isinstance(batches, (list, tuple)) or not batches:
            raise FullIterationCudaGraphError("Full-iteration CUDA graphs require a non-empty list or tuple of batches")

        leaves, self._tree_spec = tree_flatten(batches)
        self._tensor_positions = tuple(index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor))
        tensor_occurrences = tuple(leaves[index] for index in self._tensor_positions)
        self._tensor_alias_pattern = _tensor_object_alias_pattern(tensor_occurrences)

        unique_tensors = []
        for tensor, alias_index in zip(tensor_occurrences, self._tensor_alias_pattern):
            if alias_index == len(unique_tensors):
                unique_tensors.append(tensor)

        storage_alias_pattern = _tensor_storage_alias_pattern(unique_tensors)
        if len(storage_alias_pattern) != len(set(storage_alias_pattern)):
            raise FullIterationCudaGraphError(
                "Full-iteration CUDA graph inputs do not support distinct tensor objects that share storage; "
                "pass the same tensor object for intentional aliases"
            )
        self._storage_alias_pattern = storage_alias_pattern
        self._tensor_metadata = tuple(_TensorMetadata.from_tensor(tensor) for tensor in unique_tensors)

        template_leaves = list(leaves)
        for position in self._tensor_positions:
            template_leaves[position] = None
        self._template_leaves = tuple(template_leaves)

        static_tensors = []
        for tensor in unique_tensors:
            if tensor.layout is not torch.strided:
                raise FullIterationCudaGraphError(
                    f"Full-iteration CUDA graph inputs must use torch.strided layout, got {tensor.layout}"
                )
            overlap_checker = getattr(torch, "_debug_has_internal_overlap", None)
            has_internal_overlap = (
                int(overlap_checker(tensor)) != 0
                if callable(overlap_checker)
                else any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride()))
            )
            if has_internal_overlap:
                # Expanded/as_strided inputs may be readable but cannot be used
                # as copy destinations. The source stride remains part of the
                # replay signature; only the graph-owned storage is materialized.
                static_tensor = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device=device)
            else:
                static_tensor = torch.empty_strided(
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    dtype=tensor.dtype,
                    device=device,
                )
            with torch.no_grad():
                static_tensor.copy_(tensor, non_blocking=True)
            static_tensor.requires_grad_(tensor.requires_grad)
            static_tensors.append(static_tensor)
        self._static_tensors = tuple(static_tensors)
        self.batches = self._rebuild(self._static_tensors)

    def _rebuild(self, tensors: Sequence[torch.Tensor]) -> Sequence[Any]:
        leaves = list(self._template_leaves)
        for position, alias_index in zip(self._tensor_positions, self._tensor_alias_pattern):
            leaves[position] = tensors[alias_index]
        return tree_unflatten(leaves, self._tree_spec)

    def _validate(self, batches: Sequence[Any]) -> tuple[torch.Tensor, ...]:
        if not isinstance(batches, (list, tuple)) or not batches:
            raise FullIterationCudaGraphError("Full-iteration CUDA graphs require a non-empty list or tuple of batches")

        leaves, tree_spec = tree_flatten(batches)
        if tree_spec != self._tree_spec:
            raise FullIterationCudaGraphError("Full-iteration CUDA graph batch pytree or mapping keys changed")

        tensor_positions = tuple(index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor))
        if tensor_positions != self._tensor_positions:
            raise FullIterationCudaGraphError("Full-iteration CUDA graph tensor/control leaf positions changed")

        tensor_occurrences = tuple(leaves[index] for index in tensor_positions)
        if _tensor_object_alias_pattern(tensor_occurrences) != self._tensor_alias_pattern:
            raise FullIterationCudaGraphError("Full-iteration CUDA graph tensor object aliasing changed")

        unique_tensors = []
        for tensor, alias_index in zip(tensor_occurrences, self._tensor_alias_pattern):
            if alias_index == len(unique_tensors):
                unique_tensors.append(tensor)

        if _tensor_storage_alias_pattern(unique_tensors) != self._storage_alias_pattern:
            raise FullIterationCudaGraphError("Full-iteration CUDA graph tensor storage aliasing changed")

        for index, (expected, tensor) in enumerate(zip(self._tensor_metadata, unique_tensors)):
            actual = _TensorMetadata.from_tensor(tensor)
            if actual != expected:
                raise FullIterationCudaGraphError(
                    "Full-iteration CUDA graph tensor metadata changed at unique tensor "
                    f"{index}: expected {expected}, got {actual}"
                )

        tensor_position_set = set(tensor_positions)
        for index, (expected, actual) in enumerate(zip(self._template_leaves, leaves)):
            if index in tensor_position_set:
                continue
            if not _same_control_value(expected, actual):
                raise FullIterationCudaGraphError(
                    f"Full-iteration CUDA graph non-tensor control changed at pytree leaf {index}"
                )
        return tuple(unique_tensors)

    def copy_from(self, batches: Sequence[Any]) -> None:
        """Validate and copy a new fixed-geometry gradient-accumulation batch."""
        tensors = self._validate(batches)
        with torch.no_grad():
            for static_tensor, tensor in zip(self._static_tensors, tensors):
                static_tensor.copy_(tensor, non_blocking=True)


class FullIterationCudaGraphManager:
    """Warm up, capture, and replay one fixed-geometry forward/backward callable.

    Args:
        forward_backward: Callable that receives the fixed list or tuple of
            gradient-accumulation batches and performs only forward/backward.
        warmup_iterations: Number of real eager calls to run before capture.
        use_single_mempool: Reuse one process-wide graph memory pool when
            ``True``. A new pool handle is used for this capture when ``False``.
        rng_state_providers: Optional additional callables queried immediately
            before capture. Each returns graph-safe ``torch.Generator`` states
            as a mapping, iterable, single generator, or ``None``.

    The capture uses one process-wide stream and graph pool so later optimizer
    graph work can share the same allocation arena. Input structure, mapping
    keys, tensor metadata, and alias relationships are validated before any
    static input is overwritten. A mismatch poisons the manager until an
    explicit :meth:`reset`, preventing accidental replay with stale inputs.
    """

    def __init__(
        self,
        forward_backward: Callable[[Sequence[Any]], Any],
        *,
        warmup_iterations: int = 1,
        use_single_mempool: bool = False,
        rng_state_providers: Sequence[Callable[[], Any]] | None = None,
    ) -> None:
        if not callable(forward_backward):
            raise TypeError("forward_backward must be callable")
        if isinstance(warmup_iterations, bool) or not isinstance(warmup_iterations, int) or warmup_iterations < 0:
            raise ValueError("warmup_iterations must be a non-negative integer")
        if not isinstance(use_single_mempool, bool):
            raise TypeError("use_single_mempool must be a boolean")
        if rng_state_providers is None:
            rng_state_providers = ()
        if not isinstance(rng_state_providers, Sequence) or isinstance(rng_state_providers, (str, bytes)):
            raise TypeError("rng_state_providers must be a sequence of callables")
        if any(not callable(provider) for provider in rng_state_providers):
            raise TypeError("rng_state_providers must contain only callables")
        self._forward_backward = forward_backward
        self.warmup_iterations = warmup_iterations
        self.use_single_mempool = use_single_mempool
        self._rng_state_providers = (_get_transformer_engine_rng_states, *tuple(rng_state_providers))
        self._completed_warmups = 0
        self._static_inputs: _StaticCudaBatchBuffer | None = None
        self._copy_stream: Any = None
        self._graph: Any = None
        self._result: Any = None
        self._failed_reason: str | None = None
        self._closed = False
        self.capture_count = 0
        self.replay_count = 0

    @property
    def is_captured(self) -> bool:
        """Whether a valid CUDA graph is installed for replay."""
        return self._graph is not None and self._failed_reason is None and not self._closed

    @property
    def completed_warmups(self) -> int:
        """Number of eager warmup iterations completed since the last reset."""
        return self._completed_warmups

    def _destroy_graph(self) -> None:
        graph = self._graph
        self._result = None
        self._graph = None
        if graph is not None:
            reset = getattr(graph, "reset", None)
            if callable(reset):
                reset()

    def _poison(self, reason: str) -> None:
        if self._graph is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._destroy_graph()
        self._failed_reason = reason

    def _copy_batches(self, batches: Sequence[Any]) -> Sequence[Any]:
        current_stream = torch.cuda.current_stream()
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        self._copy_stream.wait_stream(current_stream)
        try:
            with torch.cuda.stream(self._copy_stream):
                if self._static_inputs is None:
                    device = torch.device("cuda", torch.cuda.current_device())
                    self._static_inputs = _StaticCudaBatchBuffer(batches, device=device)
                else:
                    self._static_inputs.copy_from(batches)
        finally:
            current_stream.wait_stream(self._copy_stream)
        return self._static_inputs.batches

    def _collect_graph_rng_states(self) -> tuple[torch.Generator, ...]:
        """Collect and identity-deduplicate graph-safe generator states."""
        states: list[torch.Generator] = []
        seen: set[int] = set()
        for provider_index, provider in enumerate(self._rng_state_providers):
            provided = provider()
            if provided is None:
                continue
            if isinstance(provided, Mapping):
                provided_states = provided.values()
            elif isinstance(provided, torch.Generator):
                provided_states = (provided,)
            else:
                try:
                    provided_states = iter(provided)
                except TypeError as error:
                    raise FullIterationCudaGraphError(
                        f"CUDA graph RNG-state provider {provider_index} must return a mapping or iterable"
                    ) from error
            for state in provided_states:
                if not isinstance(state, torch.Generator):
                    raise FullIterationCudaGraphError(
                        "CUDA graph RNG-state providers must return graph-safe torch.Generator objects; "
                        f"provider {provider_index} returned {type(state).__name__}"
                    )
                state_id = id(state)
                if state_id in seen:
                    continue
                seen.add(state_id)
                states.append(state)
        return tuple(states)

    def _capture(self, static_batches: Sequence[Any]) -> Any:
        graph = torch.cuda.CUDAGraph()
        self._graph = graph
        try:
            register_generator_state = getattr(graph, "register_generator_state", None)
            rng_states = self._collect_graph_rng_states()
            if rng_states and not callable(register_generator_state):
                raise FullIterationCudaGraphError(
                    "This PyTorch CUDAGraph does not support register_generator_state(), but a CUDA RNG tracker "
                    "is active"
                )
            for state in rng_states:
                register_generator_state(state)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.info("Capturing full-iteration CUDA graph after %d eager warmups", self._completed_warmups)
            capture_stream = _get_shared_capture_stream()
            # The caller's current stream already waits for the static-input
            # copy stream. Make the shared capture stream consume that complete
            # dependency chain explicitly instead of relying on graph-context
            # implementation details.
            capture_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.synchronize()
            _distributed_capture_barrier()
            gc.collect()
            torch.cuda.empty_cache()
            with _stale_capture_stream_override(), torch.autograd.set_multithreading_enabled(False):
                with torch.cuda.graph(
                    graph,
                    stream=capture_stream,
                    pool=_get_graph_pool(self.use_single_mempool),
                    capture_error_mode="thread_local",
                ):
                    self._result = self._forward_backward(static_batches)
            torch.cuda.synchronize()
            _distributed_capture_barrier()
            # Ending stream capture only instantiates the graph. Launch it once
            # so this optimizer iteration receives a real forward/backward,
            # matching subsequent replay iterations and Megatron's wrapper.
            graph.replay()
            torch.cuda.current_stream().wait_stream(capture_stream)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.info("Full-iteration CUDA graph capture complete")
        except Exception:
            self._destroy_graph()
            raise
        self.capture_count = 1
        return self._result

    def run(self, batches: Sequence[Any]) -> Any:
        """Execute eager warmup, graph capture, or graph replay for one step."""
        if self._closed:
            raise FullIterationCudaGraphError("Full-iteration CUDA graph manager is closed")
        if self._failed_reason is not None:
            raise FullIterationCudaGraphError(
                f"Full-iteration CUDA graph manager is failed closed; call reset() before reuse: {self._failed_reason}"
            )
        if not torch.cuda.is_available():
            raise FullIterationCudaGraphError("Full-iteration CUDA graphs require CUDA")

        try:
            static_batches = self._copy_batches(batches)
            if self._graph is not None:
                self._graph.replay()
                torch.cuda.current_stream().wait_stream(_get_shared_capture_stream())
                self.replay_count += 1
                return self._result
            if self._completed_warmups < self.warmup_iterations:
                result = self._forward_backward(static_batches)
                self._completed_warmups += 1
                return result
            return self._capture(static_batches)
        except Exception as error:
            reason = str(error) or type(error).__name__
            self._poison(reason)
            if isinstance(error, FullIterationCudaGraphError):
                raise
            raise FullIterationCudaGraphError(f"Full-iteration CUDA graph execution failed: {reason}") from error

    def reset(self) -> None:
        """Destroy capture state and permit warmup/capture with a new signature."""
        if self._closed:
            return
        if torch.cuda.is_available() and self._graph is not None:
            torch.cuda.synchronize()
        self._destroy_graph()
        self._static_inputs = None
        self._copy_stream = None
        self._completed_warmups = 0
        self._failed_reason = None
        self.capture_count = 0
        self.replay_count = 0

    def close(self) -> None:
        """Idempotently destroy graph and static input state."""
        if self._closed:
            return
        self.reset()
        self._closed = True

    def __call__(self, batches: Sequence[Any]) -> Any:
        """Delegate to :meth:`run`."""
        return self.run(batches)
