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

"""Paged activation stashing for fixed-capacity Transformer Engine MoE experts.

The actual routed-token count remains a CUDA scalar. Triton kernels consume that
scalar directly, pack only live rows into fixed page buffers, and restore those
rows immediately before expert backward. This avoids a device-to-host sync and
does not freeze expert splits.

The PP1 full-iteration runner may opt into a dedicated transfer stream. Packing
then overlaps the token-unpermutation and intervening transformer work, and is
joined before the next expert starts. Reload uses the same stream (the circular
page allocator is not safe across independent pack/unpack streams). After each
expert backward, the next LIFO expert group is prefetched while the main stream
continues through intervening backward work; its post-expert boundary rejoins
before saved tensors are exposed to autograd. The current PP1 immediate-F/B
recipe validates that LIFO order. General PP/VP or interleaved schedules require
recorded schedule metadata like Megatron's and are deliberately not inferred
from group IDs.

This module deliberately owns only activation storage. A full-iteration CUDA
graph still needs a separately fixed HybridEP dispatch shape and a global
overflow rerun policy. The first eager forward/backward records peak page usage;
call :meth:`PagedStashManager.prepare` before CUDA graph capture. A distributed
runner must combine :meth:`PagedStashManager.check_overflow` with its dispatcher
flag in one all-reduce, then discard/rerun every rank when either flag is set.

Nested ``saved_tensors_hooks`` (including PyTorch activation checkpointing) are
not claimed to be supported yet. Callers can use :meth:`PagedStashManager.disabled`
around such regions until their exact composition has been validated.
"""

from __future__ import annotations

import contextlib
import enum
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from nemo_automodel.components.moe.paged_stash_ops import (
    GLOBAL_BLOCK_SIZE,
    HAVE_TRITON,
    paged_stash_copy_kernel,
    paged_stash_pop_kernel,
)

_SCALE_INV_BLOCK_SIZE = 32
_MXFP8_EXPERT_TOKEN_ALIGNMENT = 128


class PagedStashOverflowError(RuntimeError):
    """Raised after an iteration whose fixed CUDA page capacity was exceeded."""


class _PagedStashState(enum.Enum):
    DISABLED = "disabled"
    RECORDING = "recording"
    ACTIVE = "active"


def _storage_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return a bit-preserving dtype accepted by a plain Triton copy kernel."""
    return torch.uint8 if torch.empty((), dtype=dtype).element_size() == 1 else dtype


class _PagedStashBuffer:
    """One circular GPU page allocator for a dtype and per-token width."""

    def __init__(
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype,
        overflow: torch.Tensor,
    ) -> None:
        if num_tokens <= 0:
            raise ValueError(f"Paged stash buffer capacity must be positive, got {num_tokens}")
        self.hidden_size = hidden_size
        self.page_size = page_size
        self.device = device
        self.dtype = dtype
        self.overflow = overflow
        self.num_pages = math.ceil(num_tokens / page_size)
        self.total_tokens = self.num_pages * page_size
        self.storage = torch.empty((self.total_tokens, hidden_size), dtype=dtype, device=device)
        self.free_list = torch.arange(self.num_pages, dtype=torch.int64, device=device)
        self.head = torch.zeros(1, dtype=torch.int64, device=device)
        self.tail = torch.full((1,), self.num_pages, dtype=torch.int64, device=device)
        self.capacity = torch.full((1,), self.num_pages, dtype=torch.int64, device=device)
        self._reset_free_list = torch.arange(self.num_pages, dtype=torch.int64, device=device)
        self._reset_tail = torch.full((1,), self.num_pages, dtype=torch.int64, device=device)

    def reset(self) -> None:
        """Restore the allocator without creating graph-unsafe temporary tensors."""
        self.free_list.copy_(self._reset_free_list)
        self.head.zero_()
        self.tail.copy_(self._reset_tail)


@dataclass
class _RecordedTensor:
    tensor: torch.Tensor
    key: tuple[torch.dtype, int]
    charged_tokens: int
    released: bool = False


class _PagedTensor:
    """Saved tensor represented by device-counted rows in a page buffer."""

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        num_tokens_tensor: torch.Tensor,
        max_num_tokens: int,
        hidden_size: int,
        page_size: int,
        columnwise_scale_inv: bool,
    ) -> None:
        if not tensor.is_contiguous():
            raise RuntimeError("Paged stash only supports contiguous TE grouped saved tensors")
        self._tensor: torch.Tensor | None = tensor
        self._original_tensor: torch.Tensor | None = None
        self.num_tokens_tensor = num_tokens_tensor
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.page_size = page_size
        self.columnwise_scale_inv = columnwise_scale_inv
        self.original_shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.device = tensor.device
        max_rows = self.effective_max_num_tokens
        self.page_record = torch.empty(math.ceil(max_rows / page_size), dtype=torch.int64, device=tensor.device)
        self._new_head = torch.empty(1, dtype=torch.int64, device=tensor.device)
        self._new_tail = torch.empty(1, dtype=torch.int64, device=tensor.device)

    @property
    def effective_max_num_tokens(self) -> int:
        """Return the physical row bound for this saved tensor."""
        if self.columnwise_scale_inv:
            return self.max_num_tokens // _SCALE_INV_BLOCK_SIZE
        return self.max_num_tokens

    @property
    def effective_num_tokens_tensor(self) -> torch.Tensor:
        """Return the live row count without copying it to the host."""
        if self.columnwise_scale_inv:
            return self.num_tokens_tensor // _SCALE_INV_BLOCK_SIZE
        return self.num_tokens_tensor

    def offload(self, buffer: _PagedStashBuffer) -> None:
        """Pack live rows and release the original activation reference."""
        if self._tensor is None:
            raise RuntimeError("Paged tensor was stashed more than once")
        storage_dtype = _storage_dtype(self.dtype)
        source = self._tensor.view(storage_dtype).reshape(self.effective_max_num_tokens, self.hidden_size)
        grid = (max(1, min(self.effective_max_num_tokens, 2048)),)
        paged_stash_copy_kernel[grid](
            source,
            buffer.storage,
            self.effective_num_tokens_tensor,
            buffer.free_list,
            buffer.head,
            buffer.tail,
            buffer.capacity,
            self.page_record,
            buffer.overflow,
            self._new_head,
            PAGE_SIZE=self.page_size,
            HIDDEN_SIZE=self.hidden_size,
            MAX_NUM_TOKENS=self.effective_max_num_tokens,
            BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
        )
        buffer.head.copy_(self._new_head)
        self._original_tensor = self._tensor
        self._tensor = None

    def release_original(self) -> None:
        """Drop the source after its same-stream stash kernel has been enqueued."""
        self._original_tensor = None

    def allocate_for_reload(self) -> None:
        """Allocate the fixed destination on the main graph stream."""
        if self._tensor is not None:
            raise RuntimeError("Paged tensor was restored more than once")
        self._tensor = torch.empty(self.original_shape, dtype=self.dtype, device=self.device)

    def reload(self, buffer: _PagedStashBuffer) -> None:
        """Restore live rows into a destination allocated by the caller."""
        if self._tensor is None:
            raise RuntimeError("Paged tensor reload requires a preallocated destination")
        storage_dtype = _storage_dtype(self.dtype)
        destination = self._tensor.view(storage_dtype).reshape(self.effective_max_num_tokens, self.hidden_size)
        grid = (max(1, min(self.effective_max_num_tokens, 2048)),)
        paged_stash_pop_kernel[grid](
            buffer.storage,
            destination,
            self.effective_num_tokens_tensor,
            self.page_record,
            buffer.overflow,
            buffer.free_list,
            buffer.tail,
            buffer.capacity,
            self._new_tail,
            PAGE_SIZE=self.page_size,
            HIDDEN_SIZE=self.hidden_size,
            MAX_NUM_TOKENS=self.effective_max_num_tokens,
            BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
        )
        buffer.tail.copy_(self._new_tail)

    def unpack(self) -> torch.Tensor:
        """Return the restored tensor to autograd's unpack hook."""
        if self._tensor is None:
            raise RuntimeError("Paged tensor reached backward before its group boundary restored it")
        return self._tensor


@dataclass
class _GroupState:
    group_id: int
    name: str
    max_num_tokens: int
    num_tokens_tensor: torch.Tensor


class _PagedStashBoundary(torch.autograd.Function):
    """Stash after expert forward and restore immediately before expert backward."""

    @staticmethod
    def forward(
        ctx: Any,
        output: torch.Tensor,
        manager: PagedStashManager,
        group_id: int,
    ) -> torch.Tensor:
        ctx.manager = manager
        ctx.group_id = group_id
        manager._stash_group(group_id)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        ctx.manager._begin_group_backward(ctx.group_id)
        return grad_output, None, None


class _PagedStashPreBoundary(torch.autograd.Function):
    """Start the next PP1 LIFO reload after this expert backward completes."""

    @staticmethod
    def forward(
        ctx: Any,
        input_: torch.Tensor,
        backward_anchor: torch.Tensor,
        manager: PagedStashManager,
        group_id: int,
    ) -> torch.Tensor:
        del backward_anchor
        ctx.manager = manager
        ctx.group_id = group_id
        ctx.input_requires_grad = input_.requires_grad
        ctx.active = manager._state is _PagedStashState.ACTIVE
        return input_

    @staticmethod
    def backward(ctx: Any, grad_input: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        if ctx.active:
            ctx.manager._finish_group_backward(ctx.group_id)
        return grad_input if ctx.input_requires_grad else None, None, None, None


class _PagedStashGroup:
    """One non-nestable saved-tensor-hook scope around a TE grouped MLP."""

    def __init__(self, manager: PagedStashManager, state: _GroupState | None) -> None:
        self.manager = manager
        self.state = state
        self._hooks: Any = None
        self._exited = False
        self._started = False

    def start(self, input_: torch.Tensor) -> torch.Tensor:
        """Attach the post-expert-backward prefetch boundary when requested."""
        if self.state is None:
            return input_
        if (
            self.manager._state in (_PagedStashState.RECORDING, _PagedStashState.ACTIVE)
            and self.manager._full_iteration_stream_overlap
        ):
            self._started = True
            return _PagedStashPreBoundary.apply(
                input_,
                self.manager._get_backward_anchor(input_),
                self.manager,
                self.state.group_id,
            )
        return input_

    def __enter__(self) -> _PagedStashGroup:
        if self.state is not None:
            self._hooks = torch.autograd.graph.saved_tensors_hooks(
                self.manager._pack_saved_tensor,
                self.manager._unpack_saved_tensor,
            )
            self._hooks.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        try:
            if self._hooks is not None:
                self._hooks.__exit__(exc_type, exc_value, traceback)
        finally:
            self._exited = True
            if exc_type is not None and self.state is not None:
                self.manager._abort_group(self.state.group_id, f"group raised {exc_type.__name__}")
        return False

    def commit(self, output: torch.Tensor) -> torch.Tensor:
        """Close the forward group and attach its pre-backward restore boundary."""
        if not self._exited:
            raise RuntimeError("Paged stash group must exit its context before commit()")
        if self.state is None:
            return output
        if (
            self.manager._state is _PagedStashState.ACTIVE
            and self.manager._full_iteration_stream_overlap
            and not self._started
        ):
            raise RuntimeError("Full-iteration paged stash requires start() before expert compute")
        return self.manager._commit_group(self.state.group_id, output)


class PagedStashManager:
    """Process-local page buffers and lifecycle for TE grouped expert activations."""

    def __init__(self) -> None:
        self._state = _PagedStashState.DISABLED
        self._disable_depth = 0
        self._page_size = 64
        self._buffer_size_factor = 1.1
        self._next_group_id = 0
        self._current_group: _GroupState | None = None
        self._group_tensors: dict[int, list[_PagedTensor]] = {}
        self._recorded_peak_tokens: dict[tuple[torch.dtype, int], int] = {}
        self._recorded_current_tokens: dict[tuple[torch.dtype, int], int] = {}
        self._record_device: torch.device | None = None
        self._recording_invalid_reason: str | None = None
        self._buffers: dict[tuple[torch.dtype, int], _PagedStashBuffer] = {}
        self._overflow: torch.Tensor | None = None
        self._full_iteration_stream_overlap = False
        self._transfer_stream: Any = None
        self._transfer_stream_status = "idle"
        self._pending_original_releases: list[_PagedTensor] = []
        self._backward_group_stack: list[int] = []
        self._prefetched_group_id: int | None = None
        self._backward_in_progress_group_id: int | None = None
        self._groups_without_stash: set[int] = set()
        self._backward_anchor: torch.Tensor | None = None

    @property
    def is_enabled(self) -> bool:
        """Return whether recording or stashing is configured."""
        return self._state is not _PagedStashState.DISABLED

    @property
    def is_active(self) -> bool:
        """Return whether fixed page buffers have been allocated."""
        return self._state is _PagedStashState.ACTIVE

    def configure(
        self,
        *,
        enabled: bool,
        page_size: int = 64,
        buffer_size_factor: float = 1.1,
    ) -> None:
        """Enable a recording warmup or disable a quiescent manager.

        Repeated identical enable calls are harmless because every TE expert
        layer observes the same process-local manager.
        """
        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        if isinstance(buffer_size_factor, bool) or not isinstance(buffer_size_factor, (int, float)):
            raise ValueError(f"buffer_size_factor must be a positive number, got {buffer_size_factor}")
        if buffer_size_factor <= 0:
            raise ValueError(f"buffer_size_factor must be positive, got {buffer_size_factor}")
        if self._current_group is not None or self._group_tensors:
            raise RuntimeError("Cannot reconfigure paged stash while groups are live")
        if not enabled:
            self.close()
            return
        if self._state is not _PagedStashState.DISABLED:
            if page_size != self._page_size or not math.isclose(buffer_size_factor, self._buffer_size_factor):
                raise RuntimeError("All paged-stash experts must use the same page_size and buffer_size_factor")
            return
        self._page_size = page_size
        self._buffer_size_factor = buffer_size_factor
        self._state = _PagedStashState.RECORDING

    def configure_full_iteration_stream_overlap(self, *, enabled: bool) -> None:
        """Opt into the PP1 full-iteration transfer-stream contract.

        This is intentionally configured by the recipe only after its PP1/CP1
        immediate-forward/backward eligibility checks. The stream itself is
        created by :meth:`prepare`, before active warmup and graph capture.
        """
        if self._current_group is not None or self._group_tensors:
            raise RuntimeError("Cannot change paged-stash stream overlap while groups are live")
        if self._transfer_stream_status != "idle" or self._pending_original_releases:
            raise RuntimeError("Cannot change paged-stash stream overlap while a transfer is pending")
        if (
            self._backward_group_stack
            or self._prefetched_group_id is not None
            or self._backward_in_progress_group_id is not None
        ):
            raise RuntimeError("Cannot change paged-stash stream overlap while a backward schedule is live")
        if self._groups_without_stash:
            raise RuntimeError("Cannot change paged-stash stream overlap while expert boundaries are live")
        if enabled and self._state is _PagedStashState.DISABLED:
            raise RuntimeError("Paged-stash stream overlap requires an enabled manager")
        self._full_iteration_stream_overlap = bool(enabled)
        if not enabled:
            self._transfer_stream = None

    def _require_transfer_stream(self) -> Any:
        if self._transfer_stream is None:
            raise RuntimeError(
                "Paged-stash transfer stream was not created before active execution; "
                "prepare the stash before CUDA graph capture"
            )
        return self._transfer_stream

    def _get_backward_anchor(self, input_: torch.Tensor) -> torch.Tensor:
        """Return a persistent grad edge even when an expert input is frozen.

        The pre-expert boundary must run after expert backward to clear the
        current group and prefetch the next one. A custom Function whose only
        tensor input does not require gradients has no backward node, even
        though trainable expert weights can make the expert output require
        gradients. This scalar supplies that edge without accumulating a grad.
        """
        anchor = self._backward_anchor
        if anchor is None:
            anchor = torch.empty((), dtype=torch.float32, device=input_.device, requires_grad=True)
            self._backward_anchor = anchor
        elif anchor.device != input_.device:
            raise RuntimeError(
                f"Paged-stash backward anchor is on {anchor.device}, but the expert input is on {input_.device}"
            )
        return anchor

    def _release_pending_originals(self) -> None:
        while self._pending_original_releases:
            self._pending_original_releases.pop(0).release_original()

    def _wait_for_stash_to_complete(self) -> None:
        """Join an asynchronous pack before another expert can reuse storage."""
        if not self._full_iteration_stream_overlap:
            return
        if self._transfer_stream_status == "packing":
            torch.cuda.current_stream().wait_stream(self._require_transfer_stream())
            self._transfer_stream_status = "idle"
            self._release_pending_originals()
        elif self._transfer_stream_status == "reloading":
            raise RuntimeError(
                "PP1 paged-stash reload prefetch reached a new forward expert; "
                "this schedule requires explicit PP/VP metadata"
            )
        elif self._pending_original_releases:
            raise RuntimeError("Paged stash has pending source tensors without an active pack")

    @contextlib.contextmanager
    def disabled(self) -> Iterator[None]:
        """Temporarily bypass hooks, including around unvalidated checkpoint scopes."""
        self._disable_depth += 1
        try:
            yield
        finally:
            self._disable_depth -= 1

    def group(
        self,
        *,
        name: str,
        max_num_tokens: int,
        num_tokens_tensor: torch.Tensor,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> _PagedStashGroup:
        """Create one saved-tensor scope around a grouped expert invocation."""
        if not self.is_enabled or self._disable_depth or not torch.is_grad_enabled():
            return _PagedStashGroup(self, None)
        # A previous expert's pack may overlap token unpermutation, attention,
        # routing, and dispatch, but its source storage must be retired before
        # the next expert starts producing saved activations.
        self._wait_for_stash_to_complete()
        if self._current_group is not None:
            raise RuntimeError("Paged stash groups cannot nest; disable paged stash around outer saved_tensors_hooks")
        if max_num_tokens <= 0:
            raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}")
        if not isinstance(num_tokens_tensor, torch.Tensor) or num_tokens_tensor.numel() != 1:
            raise TypeError("num_tokens_tensor must be a one-element Tensor")
        integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        if num_tokens_tensor.dtype not in integer_dtypes:
            raise TypeError(f"num_tokens_tensor must have an integer dtype, got {num_tokens_tensor.dtype}")
        if tokens_per_expert is not None:
            if not isinstance(tokens_per_expert, torch.Tensor) or tokens_per_expert.dim() != 1:
                raise TypeError("tokens_per_expert must be a one-dimensional Tensor")
            if tokens_per_expert.dtype not in integer_dtypes:
                raise TypeError(f"tokens_per_expert must have an integer dtype, got {tokens_per_expert.dtype}")
            if tokens_per_expert.device != num_tokens_tensor.device:
                raise ValueError("tokens_per_expert and num_tokens_tensor must be on the same device")
            misaligned = tokens_per_expert.remainder(_MXFP8_EXPERT_TOKEN_ALIGNMENT).ne(0).any()
            if self._state is _PagedStashState.RECORDING:
                if bool(misaligned.item()):
                    raise RuntimeError(
                        "Paged MXFP8 stash requires every expert split to be aligned to "
                        f"{_MXFP8_EXPERT_TOKEN_ALIGNMENT} rows"
                    )
            elif self._state is _PagedStashState.ACTIVE:
                # Keep this check entirely on device during capture/replay. A
                # violation poisons the fast attempt and is globally rerun
                # without stashing before the optimizer can update parameters.
                if self._overflow is None:
                    raise RuntimeError("Active paged stash is missing its overflow flag")
                self._overflow.logical_or_(misaligned)
        state = _GroupState(
            group_id=self._next_group_id,
            name=name,
            max_num_tokens=max_num_tokens,
            num_tokens_tensor=num_tokens_tensor.reshape(1).clone(),
        )
        self._next_group_id += 1
        self._current_group = state
        self._group_tensors[state.group_id] = []
        return _PagedStashGroup(self, state)

    def _tensor_layout(
        self,
        tensor: torch.Tensor,
        group: _GroupState,
        columnwise_scale_inv: bool,
    ) -> tuple[tuple[torch.dtype, int], int]:
        max_rows = group.max_num_tokens
        if columnwise_scale_inv:
            if max_rows % _SCALE_INV_BLOCK_SIZE:
                raise RuntimeError(
                    f"Columnwise scale-inverse max token count {max_rows} is not divisible by {_SCALE_INV_BLOCK_SIZE}"
                )
            max_rows //= _SCALE_INV_BLOCK_SIZE
        if max_rows <= 0 or tensor.numel() % max_rows:
            raise RuntimeError(
                f"TE grouped saved tensor with {tensor.numel()} elements cannot be represented as {max_rows} rows"
            )
        hidden_size = tensor.numel() // max_rows
        return (tensor.dtype, hidden_size), max_rows

    def _pack_saved_tensor(self, tensor: torch.Tensor) -> Any:
        group = self._current_group
        if group is None:
            raise RuntimeError("Paged stash pack hook ran outside a group")
        marker = getattr(tensor, "grouped_tensor_scale_inv", None)
        if marker is None or tensor.dim() == 0:
            return tensor
        if group.num_tokens_tensor.device != tensor.device:
            raise RuntimeError(
                "Paged stash token count and saved tensor must be on the same device, got "
                f"{group.num_tokens_tensor.device} and {tensor.device}"
            )
        columnwise_scale_inv = bool(marker)
        key, max_rows = self._tensor_layout(tensor, group, columnwise_scale_inv)

        if self._state is _PagedStashState.RECORDING:
            actual_tokens = int(group.num_tokens_tensor.item())
            if columnwise_scale_inv:
                actual_tokens //= _SCALE_INV_BLOCK_SIZE
            if actual_tokens < 0 or actual_tokens > max_rows:
                raise RuntimeError(f"Recorded TE grouped token count {actual_tokens} is outside [0, {max_rows}]")
            if self._record_device is None:
                self._record_device = tensor.device
            elif tensor.device != self._record_device:
                raise RuntimeError(
                    f"Paged stash recording changed device from {self._record_device} to {tensor.device}"
                )
            # The active allocator cannot share a partial page between saved
            # tensors. Profile the exact page-rounded charge, rather than raw
            # live rows, or many small scale tensors understate peak capacity.
            charged_tokens = math.ceil(actual_tokens / self._page_size) * self._page_size
            current = self._recorded_current_tokens.get(key, 0) + charged_tokens
            self._recorded_current_tokens[key] = current
            self._recorded_peak_tokens[key] = max(self._recorded_peak_tokens.get(key, 0), current)
            return _RecordedTensor(tensor=tensor, key=key, charged_tokens=charged_tokens)

        if self._state is not _PagedStashState.ACTIVE:
            return tensor
        if tensor.device.type != "cuda":
            raise RuntimeError(f"Active paged stash requires CUDA tensors, got {tensor.device}")
        if key not in self._buffers:
            raise RuntimeError(f"TE saved-tensor layout {key} was not observed during paged-stash recording warmup")
        paged_tensor = _PagedTensor(
            tensor,
            num_tokens_tensor=group.num_tokens_tensor,
            max_num_tokens=group.max_num_tokens,
            hidden_size=key[1],
            page_size=self._page_size,
            columnwise_scale_inv=columnwise_scale_inv,
        )
        self._group_tensors[group.group_id].append(paged_tensor)
        return paged_tensor

    def _unpack_saved_tensor(self, saved: Any) -> torch.Tensor:
        if isinstance(saved, _RecordedTensor):
            if not saved.released:
                current = self._recorded_current_tokens[saved.key] - saved.charged_tokens
                if current < 0:
                    raise RuntimeError(f"Paged stash recording underflow for layout {saved.key}")
                self._recorded_current_tokens[saved.key] = current
                saved.released = True
            return saved.tensor
        if isinstance(saved, _PagedTensor):
            return saved.unpack()
        return saved

    def _commit_group(self, group_id: int, output: torch.Tensor) -> torch.Tensor:
        if self._current_group is None or self._current_group.group_id != group_id:
            raise RuntimeError(f"Paged stash group {group_id} committed out of order")
        self._current_group = None
        if self._state is _PagedStashState.RECORDING:
            self._group_tensors.pop(group_id, None)
            return output
        if not self._group_tensors[group_id]:
            self._group_tensors.pop(group_id)
            if self._full_iteration_stream_overlap:
                # start() already attached a pre-boundary. Remember that no
                # matching post-boundary exists so its backward is a no-op.
                self._groups_without_stash.add(group_id)
            return output
        return _PagedStashBoundary.apply(output, self, group_id)

    def _abort_group(self, group_id: int, reason: str) -> None:
        if self._current_group is not None and self._current_group.group_id == group_id:
            self._current_group = None
        self._group_tensors.pop(group_id, None)
        if self._state is _PagedStashState.RECORDING:
            self._recording_invalid_reason = reason

    def _stash_group(self, group_id: int) -> None:
        tensors = self._group_tensors.get(group_id)
        if tensors is None:
            raise RuntimeError(f"Unknown paged stash group {group_id}")
        if not self._full_iteration_stream_overlap:
            for paged_tensor in tensors:
                key = (paged_tensor.dtype, paged_tensor.hidden_size)
                paged_tensor.offload(self._buffers[key])
            for paged_tensor in tensors:
                paged_tensor.release_original()
            return

        if self._transfer_stream_status != "idle" or self._pending_original_releases:
            raise RuntimeError("Paged stash started a pack before the previous transfer was joined")
        current_stream = torch.cuda.current_stream()
        transfer_stream = self._require_transfer_stream()
        transfer_stream.wait_stream(current_stream)
        with torch.cuda.stream(transfer_stream):
            for paged_tensor in tensors:
                key = (paged_tensor.dtype, paged_tensor.hidden_size)
                paged_tensor.offload(self._buffers[key])
        self._pending_original_releases.extend(tensors)
        self._transfer_stream_status = "packing"
        self._backward_group_stack.append(group_id)

    def _start_group_reload(self, group_id: int) -> None:
        """Start one reload on the transfer stream without joining it."""
        if self._prefetched_group_id is not None:
            raise RuntimeError(
                f"Paged stash already prefetched group {self._prefetched_group_id}, cannot start {group_id}"
            )
        tensors = self._group_tensors.pop(group_id, None)
        if tensors is None:
            raise RuntimeError(f"Unknown paged stash group {group_id} during backward")
        current_stream = torch.cuda.current_stream()
        transfer_stream = self._require_transfer_stream()
        for paged_tensor in reversed(tensors):
            paged_tensor.allocate_for_reload()
        transfer_stream.wait_stream(current_stream)
        self._transfer_stream_status = "reloading"
        with torch.cuda.stream(transfer_stream):
            for paged_tensor in reversed(tensors):
                key = (paged_tensor.dtype, paged_tensor.hidden_size)
                paged_tensor.reload(self._buffers[key])
        self._prefetched_group_id = group_id

    def _wait_for_group_reload(self, group_id: int) -> None:
        """Join the matching prefetched reload before saved-tensor unpack."""
        if self._prefetched_group_id != group_id:
            raise RuntimeError(f"Paged stash expected prefetched group {group_id}, got {self._prefetched_group_id}")
        current_stream = torch.cuda.current_stream()
        transfer_stream = self._require_transfer_stream()
        current_stream.wait_stream(transfer_stream)
        self._transfer_stream_status = "idle"
        self._prefetched_group_id = None
        self._release_pending_originals()

    def _reload_group(self, group_id: int) -> None:
        """Restore one group, synchronously unless full-iteration overlap is armed."""
        if not self._full_iteration_stream_overlap:
            tensors = self._group_tensors.pop(group_id, None)
            if tensors is None:
                raise RuntimeError(f"Unknown paged stash group {group_id} during backward")
            for paged_tensor in reversed(tensors):
                paged_tensor.allocate_for_reload()
                key = (paged_tensor.dtype, paged_tensor.hidden_size)
                paged_tensor.reload(self._buffers[key])
            return
        self._start_group_reload(group_id)
        self._wait_for_group_reload(group_id)

    def _begin_group_backward(self, group_id: int) -> None:
        """Validate PP1 LIFO order and expose this group's restored tensors."""
        if not self._full_iteration_stream_overlap:
            self._reload_group(group_id)
            return
        if not self._backward_group_stack or self._backward_group_stack[-1] != group_id:
            expected = self._backward_group_stack[-1] if self._backward_group_stack else None
            raise RuntimeError(
                f"PP1 paged-stash backward order is not LIFO: expected group {expected}, got {group_id}; "
                "this schedule requires explicit PP/VP metadata"
            )
        if self._backward_in_progress_group_id is not None:
            raise RuntimeError(
                f"Paged stash began group {group_id} backward while group "
                f"{self._backward_in_progress_group_id} is still active"
            )
        if self._prefetched_group_id is None:
            self._start_group_reload(group_id)
        self._wait_for_group_reload(group_id)
        self._backward_group_stack.pop()
        self._backward_in_progress_group_id = group_id

    def _finish_group_backward(self, group_id: int) -> None:
        """Prefetch the next LIFO group after the current expert backward."""
        if not self._full_iteration_stream_overlap:
            return
        if group_id in self._groups_without_stash:
            self._groups_without_stash.remove(group_id)
            return
        if self._backward_in_progress_group_id != group_id:
            raise RuntimeError(f"Paged stash finished group {group_id}, expected {self._backward_in_progress_group_id}")
        self._backward_in_progress_group_id = None
        if self._backward_group_stack:
            self._start_group_reload(self._backward_group_stack[-1])

    def prepare(self) -> None:
        """Allocate fixed CUDA page buffers from one completed eager warmup."""
        if self._state is not _PagedStashState.RECORDING:
            raise RuntimeError(f"prepare() requires recording state, got {self._state.value}")
        if self._recording_invalid_reason is not None:
            raise RuntimeError(f"Paged stash recording was invalid: {self._recording_invalid_reason}")
        if self._current_group is not None or self._group_tensors:
            raise RuntimeError("Paged stash warmup still has live groups; run a complete backward before prepare()")
        nonzero_usage = {key: value for key, value in self._recorded_current_tokens.items() if value}
        if nonzero_usage:
            raise RuntimeError(f"Paged stash warmup still has live saved tensors: {nonzero_usage}")
        if not self._recorded_peak_tokens:
            raise RuntimeError("Paged stash warmup observed no TE grouped saved tensors")
        if self._record_device is None or self._record_device.type != "cuda":
            raise RuntimeError(f"Paged stash page buffers require a CUDA warmup, got {self._record_device}")
        if not HAVE_TRITON:
            raise RuntimeError("Paged stash requires Triton")

        self._overflow = torch.zeros(1, dtype=torch.int64, device=self._record_device)
        for (dtype, hidden_size), peak_tokens in self._recorded_peak_tokens.items():
            capacity = max(1, math.ceil(peak_tokens * self._buffer_size_factor))
            storage_dtype = _storage_dtype(dtype)
            self._buffers[dtype, hidden_size] = _PagedStashBuffer(
                num_tokens=capacity,
                hidden_size=hidden_size,
                page_size=self._page_size,
                device=self._record_device,
                dtype=storage_dtype,
                overflow=self._overflow,
            )
        if self._full_iteration_stream_overlap:
            self._transfer_stream = torch.cuda.Stream(device=self._record_device)
            if self._backward_anchor is None:
                self._backward_anchor = torch.empty(
                    (), dtype=torch.float32, device=self._record_device, requires_grad=True
                )
        self._state = _PagedStashState.ACTIVE

    def finish_iteration(self) -> None:
        """Synchronously validate overflow after a single-rank forward/backward.

        Distributed runners must instead stack :meth:`check_overflow` with their
        other device flags, perform one all-reduce, and only then inspect the
        result on the host. Calling this rank-local diagnostic before a collective
        can deadlock when only some ranks overflow.
        """
        if self._current_group is not None or self._group_tensors:
            raise RuntimeError("Paged stash iteration finished with live expert groups")
        if (
            self._transfer_stream_status != "idle"
            or self._pending_original_releases
            or self._backward_group_stack
            or self._prefetched_group_id is not None
            or self._backward_in_progress_group_id is not None
            or self._groups_without_stash
        ):
            raise RuntimeError("Paged stash iteration finished with live transfer or backward schedule state")
        if self._overflow is not None and bool(self._overflow.item()):
            raise PagedStashOverflowError(
                "MoE paged stash capacity overflowed; discard this attempt and rerun without paged stash"
            )

    def check_overflow(self) -> torch.Tensor | None:
        """Return the device-resident overflow flag without synchronizing the host."""
        return self._overflow

    def reset_after_overflow(self) -> None:
        """Clear overflow metadata after the graph using these buffers is destroyed."""
        if self._current_group is not None or self._group_tensors:
            raise RuntimeError("Cannot reset paged stash while groups are live")
        if self._transfer_stream_status != "idle" or self._pending_original_releases:
            raise RuntimeError("Cannot reset paged stash while a transfer is pending")
        if (
            self._backward_group_stack
            or self._prefetched_group_id is not None
            or self._backward_in_progress_group_id is not None
            or self._groups_without_stash
        ):
            raise RuntimeError("Cannot reset paged stash while a backward schedule is live")
        if self._overflow is not None:
            self._overflow.zero_()
        for buffer in self._buffers.values():
            buffer.reset()

    def restart_recording(self) -> None:
        """Drop fixed pages and start a fresh eager profile after graph teardown."""
        page_size = self._page_size
        buffer_size_factor = self._buffer_size_factor
        stream_overlap = self._full_iteration_stream_overlap
        self.close()
        self.configure(
            enabled=True,
            page_size=page_size,
            buffer_size_factor=buffer_size_factor,
        )
        self.configure_full_iteration_stream_overlap(enabled=stream_overlap)

    def diagnostics(self) -> dict[str, Any]:
        """Return bounded lifecycle and allocation diagnostics."""
        return {
            "state": self._state.value,
            "page_size": self._page_size,
            "buffer_size_factor": self._buffer_size_factor,
            "recorded_peak_tokens": dict(self._recorded_peak_tokens),
            "buffer_tokens": {key: buffer.total_tokens for key, buffer in self._buffers.items()},
            "live_groups": len(self._group_tensors),
            "full_iteration_stream_overlap": self._full_iteration_stream_overlap,
            "transfer_stream_status": self._transfer_stream_status,
            "backward_schedule_depth": len(self._backward_group_stack),
            "prefetched_group_id": self._prefetched_group_id,
        }

    def _clear_state(self) -> None:
        """Release all manager-owned references after synchronization/validation."""
        self._current_group = None
        self._group_tensors.clear()
        self._buffers.clear()
        self._overflow = None
        self._recorded_peak_tokens.clear()
        self._recorded_current_tokens.clear()
        self._record_device = None
        self._recording_invalid_reason = None
        self._next_group_id = 0
        self._disable_depth = 0
        self._full_iteration_stream_overlap = False
        self._transfer_stream = None
        self._transfer_stream_status = "idle"
        self._pending_original_releases.clear()
        self._backward_group_stack.clear()
        self._prefetched_group_id = None
        self._backward_in_progress_group_id = None
        self._groups_without_stash.clear()
        self._backward_anchor = None
        self._state = _PagedStashState.DISABLED

    def force_abort_after_error(self) -> None:
        """Best-effort no-throw cleanup after exceptional graph teardown.

        The caller must destroy any CUDA graph that references the page buffers
        first. Normal teardown uses :meth:`close` so lifecycle bugs still fail
        loudly; this path exists only to keep cleanup from masking the original
        execution error.
        """
        device = self._record_device
        with contextlib.suppress(Exception):
            if torch.cuda.is_available() and device is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
        with contextlib.suppress(Exception):
            self._release_pending_originals()
        self._clear_state()

    def close(self) -> None:
        """Release page buffers after every CUDA graph that references them is gone."""
        if self._current_group is not None or self._group_tensors:
            raise RuntimeError("Cannot close paged stash while groups are live")
        if self._transfer_stream_status != "idle" or self._pending_original_releases:
            raise RuntimeError("Cannot close paged stash while a transfer is pending")
        if (
            self._backward_group_stack
            or self._prefetched_group_id is not None
            or self._backward_in_progress_group_id is not None
            or self._groups_without_stash
        ):
            raise RuntimeError("Cannot close paged stash while a backward schedule is live")
        self._clear_state()


_PAGED_STASH_MANAGER = PagedStashManager()


def get_paged_stash_manager() -> PagedStashManager:
    """Return the process-local TE expert paged-stash manager."""
    return _PAGED_STASH_MANAGER


__all__ = ["PagedStashManager", "PagedStashOverflowError", "get_paged_stash_manager"]
