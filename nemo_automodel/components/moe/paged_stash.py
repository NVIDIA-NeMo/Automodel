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

This module deliberately owns only explicitly marked TE expert activation
storage. The first eager forward/backward records peak page usage; call
:meth:`PagedStashManager.prepare` before partial CUDA graph capture. A
distributed runner must reduce :meth:`PagedStashManager.check_overflow` across
ranks, then discard gradients and rerun the whole step eagerly on every rank.

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
        live_token_mask: torch.Tensor,
        live_token_offsets: torch.Tensor,
        max_num_tokens: int,
        hidden_size: int,
        page_size: int,
    ) -> None:
        if not tensor.is_contiguous():
            raise RuntimeError("Paged stash only supports contiguous TE grouped saved tensors")
        self._tensor: torch.Tensor | None = tensor
        self._original_tensor: torch.Tensor | None = None
        self.num_tokens_tensor = num_tokens_tensor
        self.live_token_mask = live_token_mask
        self.live_token_offsets = live_token_offsets
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.page_size = page_size
        self.original_shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.device = tensor.device
        self.page_record = torch.empty(math.ceil(max_num_tokens / page_size), dtype=torch.int64, device=tensor.device)
        self._new_head = torch.empty(1, dtype=torch.int64, device=tensor.device)
        self._new_tail = torch.empty(1, dtype=torch.int64, device=tensor.device)

    def offload(self, buffer: _PagedStashBuffer) -> None:
        """Pack live rows and release the original activation reference."""
        if self._tensor is None:
            raise RuntimeError("Paged tensor was stashed more than once")
        storage_dtype = _storage_dtype(self.dtype)
        source = self._tensor.view(storage_dtype).reshape(self.max_num_tokens, self.hidden_size)
        grid = (max(1, min(self.max_num_tokens, 2048)),)
        paged_stash_copy_kernel[grid](
            source,
            buffer.storage,
            self.num_tokens_tensor,
            self.live_token_mask,
            self.live_token_offsets,
            buffer.free_list,
            buffer.head,
            buffer.tail,
            buffer.capacity,
            self.page_record,
            buffer.overflow,
            self._new_head,
            PAGE_SIZE=self.page_size,
            HIDDEN_SIZE=self.hidden_size,
            MAX_NUM_TOKENS=self.max_num_tokens,
            BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
        )
        buffer.head.copy_(self._new_head)
        self._original_tensor = self._tensor
        self._tensor = None

    def release_original(self) -> None:
        """Drop the source after its same-stream stash kernel has been enqueued."""
        self._original_tensor = None

    def reload(self, buffer: _PagedStashBuffer) -> None:
        """Allocate the original fixed shape and restore its live rows."""
        if self._tensor is not None:
            raise RuntimeError("Paged tensor was restored more than once")
        # Padded rows participate in the fixed-capacity GroupedLinear backward.
        # They must be zero rather than uninitialized when only live rows are restored.
        self._tensor = torch.zeros(self.original_shape, dtype=self.dtype, device=self.device)
        storage_dtype = _storage_dtype(self.dtype)
        destination = self._tensor.view(storage_dtype).reshape(self.max_num_tokens, self.hidden_size)
        grid = (max(1, min(self.max_num_tokens, 2048)),)
        paged_stash_pop_kernel[grid](
            buffer.storage,
            destination,
            self.num_tokens_tensor,
            self.live_token_mask,
            self.live_token_offsets,
            self.page_record,
            buffer.overflow,
            buffer.free_list,
            buffer.tail,
            buffer.capacity,
            self._new_tail,
            PAGE_SIZE=self.page_size,
            HIDDEN_SIZE=self.hidden_size,
            MAX_NUM_TOKENS=self.max_num_tokens,
            BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
        )
        buffer.tail.copy_(self._new_tail)

    def unpack(self) -> torch.Tensor:
        """Return the restored tensor to autograd's unpack hook."""
        if self._tensor is None:
            raise RuntimeError("Paged tensor reached backward before its group boundary restored it")
        return self._tensor

    def release_reloaded(self) -> None:
        """Drop the restored tensor after every expert backward consumer has run."""
        if self._tensor is None:
            raise RuntimeError("Paged tensor reload storage was released before expert backward completed")
        self._tensor = None


@dataclass
class _GroupState:
    group_id: int
    name: str
    max_num_tokens: int
    num_tokens_tensor: torch.Tensor
    live_token_mask: torch.Tensor
    live_token_offsets: torch.Tensor
    activation_surfaces: list[_ActivationSurface]
    observed_saved_metadata: list[tuple[Any, ...]]


@dataclass(frozen=True)
class _ActivationSurface:
    """One explicitly registered contiguous activation storage range."""

    device: torch.device
    dtype: torch.dtype
    storage_ptr: int
    element_start: int
    numel: int
    num_rows: int
    row_width: int
    live_token_mask: torch.Tensor

    def match(self, tensor: torch.Tensor) -> tuple[int, torch.Tensor] | None:
        """Return row count and mask for a contained row-aligned alias."""
        if tensor.device != self.device or tensor.dtype != self.dtype:
            return None
        if (
            tensor.dim() < 2
            or tensor.shape[0] <= 0
            or tensor.untyped_storage().data_ptr() != self.storage_ptr
            or not tensor.is_contiguous()
        ):
            return None
        tensor_row_width = tensor.numel() // tensor.shape[0]
        if tensor_row_width != self.row_width:
            return None
        relative_start = tensor.storage_offset() - self.element_start
        if relative_start < 0 or relative_start % self.row_width or tensor.numel() % self.row_width:
            return None
        relative_end = relative_start + tensor.numel()
        if relative_end > self.numel:
            return None
        row_start = relative_start // self.row_width
        num_rows = tensor.numel() // self.row_width
        return num_rows, self.live_token_mask.narrow(0, row_start, num_rows)


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
        ctx.manager._reload_group(ctx.group_id)
        return grad_output, None, None


class _PagedStashPreBoundary(torch.autograd.Function):
    """Release restored tensors after the complete expert backward finishes."""

    @staticmethod
    def forward(
        ctx: Any,
        input_: torch.Tensor,
        manager: PagedStashManager,
        group_id: int,
    ) -> torch.Tensor:
        ctx.manager = manager
        ctx.group_id = group_id
        ctx.active = manager.is_active
        return input_

    @staticmethod
    def backward(ctx: Any, grad_input: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        if ctx.active:
            ctx.manager._finish_group_backward(ctx.group_id)
        return grad_input, None, None


class _PagedStashGroup:
    """One non-nestable saved-tensor-hook scope around a TE grouped MLP."""

    def __init__(self, manager: PagedStashManager, state: _GroupState | None) -> None:
        self.manager = manager
        self.state = state
        self._hooks: Any = None
        self._exited = False
        self._started = False

    def start(self, input_: torch.Tensor) -> torch.Tensor:
        """Attach the boundary that releases reload storage after expert backward."""
        if self.state is None:
            return input_
        if self._started:
            raise RuntimeError("Paged stash group start() may only be called once")
        if not input_.requires_grad:
            self.manager._abort_group(self.state.group_id, "expert input does not require gradients")
            raise RuntimeError("Paged stash requires the expert input to require gradients")
        self._started = True
        return _PagedStashPreBoundary.apply(
            input_,
            self.manager,
            self.state.group_id,
        )

    def mark_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mark one exact GroupedLinear activation input as pageable."""
        if self.state is None:
            return tensor
        if not tensor.is_contiguous() or tensor.dim() < 2 or tensor.shape[0] != self.state.max_num_tokens:
            raise RuntimeError(
                "Paged stash activation must be contiguous with the fixed expert-token dimension first; "
                f"got shape={tuple(tensor.shape)} contiguous={tensor.is_contiguous()}"
            )
        row_width = tensor.numel() // tensor.shape[0]
        self.state.activation_surfaces.append(
            _ActivationSurface(
                device=tensor.device,
                dtype=tensor.dtype,
                storage_ptr=tensor.untyped_storage().data_ptr(),
                element_start=tensor.storage_offset(),
                numel=tensor.numel(),
                num_rows=tensor.shape[0],
                row_width=row_width,
                live_token_mask=self.state.live_token_mask,
            )
        )
        return tensor

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
        if not self._started:
            raise RuntimeError("Paged stash group requires start() before expert compute")
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
        self._group_saved_tensor_counts: dict[int, int] = {}
        self._recorded_peak_tokens: dict[tuple[torch.dtype, int], int] = {}
        self._recorded_current_tokens: dict[tuple[torch.dtype, int], int] = {}
        self._record_device: torch.device | None = None
        self._recording_invalid_reason: str | None = None
        self._buffers: dict[tuple[torch.dtype, int], _PagedStashBuffer] = {}
        self._overflow: torch.Tensor | None = None
        self._backward_group_stack: list[int] = []
        self._backward_in_progress_group_id: int | None = None

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
        live_token_mask: torch.Tensor,
    ) -> _PagedStashGroup:
        """Create one saved-tensor scope around a grouped expert invocation."""
        if not self.is_enabled or self._disable_depth or not torch.is_grad_enabled():
            return _PagedStashGroup(self, None)
        if self._current_group is not None:
            raise RuntimeError("Paged stash groups cannot nest; disable paged stash around outer saved_tensors_hooks")
        if max_num_tokens <= 0:
            raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}")
        if not isinstance(live_token_mask, torch.Tensor) or live_token_mask.dim() != 1:
            raise TypeError("live_token_mask must be a one-dimensional Tensor")
        if live_token_mask.dtype != torch.bool:
            raise TypeError(f"live_token_mask must have boolean dtype, got {live_token_mask.dtype}")
        if live_token_mask.numel() != max_num_tokens:
            raise ValueError(
                f"live_token_mask has {live_token_mask.numel()} rows, expected max_num_tokens={max_num_tokens}"
            )
        live_token_offsets = live_token_mask.to(torch.int64).cumsum(dim=0).sub(1)
        state = _GroupState(
            group_id=self._next_group_id,
            name=name,
            max_num_tokens=max_num_tokens,
            num_tokens_tensor=live_token_mask.sum().reshape(1),
            live_token_mask=live_token_mask,
            live_token_offsets=live_token_offsets,
            activation_surfaces=[],
            observed_saved_metadata=[],
        )
        self._next_group_id += 1
        self._current_group = state
        self._group_tensors[state.group_id] = []
        self._group_saved_tensor_counts[state.group_id] = 0
        return _PagedStashGroup(self, state)

    def _pack_saved_tensor(self, tensor: torch.Tensor) -> Any:
        group = self._current_group
        if group is None:
            raise RuntimeError("Paged stash pack hook ran outside a group")
        if len(group.observed_saved_metadata) < 16:
            group.observed_saved_metadata.append(
                (tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())
            )
        match = None
        for surface in group.activation_surfaces:
            match = surface.match(tensor)
            if match is not None:
                break
        if match is None:
            return tensor
        max_rows, live_token_mask = match
        num_tokens_tensor = live_token_mask.sum().reshape(1)
        live_token_offsets = live_token_mask.to(torch.int64).cumsum(dim=0).sub(1)
        if num_tokens_tensor.device != tensor.device:
            raise RuntimeError(
                "Paged stash token count and saved tensor must be on the same device, got "
                f"{num_tokens_tensor.device} and {tensor.device}"
            )
        if max_rows <= 0 or tensor.numel() % max_rows:
            raise RuntimeError(
                f"TE grouped saved tensor with {tensor.numel()} elements cannot be represented as {max_rows} rows"
            )
        key = (tensor.dtype, tensor.numel() // max_rows)
        self._group_saved_tensor_counts[group.group_id] += 1

        if self._state is _PagedStashState.RECORDING:
            actual_tokens = int(num_tokens_tensor.item())
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
            num_tokens_tensor=num_tokens_tensor,
            live_token_mask=live_token_mask,
            live_token_offsets=live_token_offsets,
            max_num_tokens=max_rows,
            hidden_size=key[1],
            page_size=self._page_size,
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
        group_state = self._current_group
        self._current_group = None
        saved_tensor_count = self._group_saved_tensor_counts.pop(group_id)
        if saved_tensor_count == 0:
            self._group_tensors.pop(group_id, None)
            raise RuntimeError(
                "Paged stash observed no registered TE expert activation storage; "
                "the selected GroupedLinear path does not save the registered activation surface; "
                f"registered surfaces={group_state.activation_surfaces}, "
                f"observed saved metadata={group_state.observed_saved_metadata}"
            )
        if self._state is _PagedStashState.RECORDING:
            self._group_tensors.pop(group_id, None)
            return output
        if not self._group_tensors[group_id]:
            self._group_tensors.pop(group_id)
            return output
        return _PagedStashBoundary.apply(output, self, group_id)

    def _abort_group(self, group_id: int, reason: str) -> None:
        if self._current_group is not None and self._current_group.group_id == group_id:
            self._current_group = None
        self._group_tensors.pop(group_id, None)
        self._group_saved_tensor_counts.pop(group_id, None)
        if self._state is _PagedStashState.RECORDING:
            self._recording_invalid_reason = reason

    def _stash_group(self, group_id: int) -> None:
        tensors = self._group_tensors.get(group_id)
        if tensors is None:
            raise RuntimeError(f"Unknown paged stash group {group_id}")
        for paged_tensor in tensors:
            key = (paged_tensor.dtype, paged_tensor.hidden_size)
            paged_tensor.offload(self._buffers[key])
        for paged_tensor in tensors:
            paged_tensor.release_original()
        self._backward_group_stack.append(group_id)

    def _reload_group(self, group_id: int) -> None:
        if not self._backward_group_stack or self._backward_group_stack[-1] != group_id:
            expected = self._backward_group_stack[-1] if self._backward_group_stack else None
            raise RuntimeError(f"Paged stash backward order must be LIFO: expected group {expected}, got {group_id}")
        if self._backward_in_progress_group_id is not None:
            raise RuntimeError(
                f"Paged stash began group {group_id} backward while group "
                f"{self._backward_in_progress_group_id} is still active"
            )
        tensors = self._group_tensors.get(group_id)
        if tensors is None:
            raise RuntimeError(f"Unknown paged stash group {group_id} during backward")
        for paged_tensor in reversed(tensors):
            key = (paged_tensor.dtype, paged_tensor.hidden_size)
            paged_tensor.reload(self._buffers[key])
        self._backward_group_stack.pop()
        self._backward_in_progress_group_id = group_id

    def _finish_group_backward(self, group_id: int) -> None:
        """Release restored tensors once the complete expert backward has consumed them."""
        if self._backward_in_progress_group_id != group_id:
            raise RuntimeError(f"Paged stash finished group {group_id}, expected {self._backward_in_progress_group_id}")
        tensors = self._group_tensors.pop(group_id, None)
        if tensors is None:
            raise RuntimeError(f"Unknown paged stash group {group_id} after backward")
        for paged_tensor in tensors:
            paged_tensor.release_reloaded()
        self._backward_in_progress_group_id = None

    def prepare(self) -> None:
        """Allocate fixed CUDA page buffers from one completed eager warmup."""
        if self._state is not _PagedStashState.RECORDING:
            raise RuntimeError(f"prepare() requires recording state, got {self._state.value}")
        if self._recording_invalid_reason is not None:
            raise RuntimeError(f"Paged stash recording was invalid: {self._recording_invalid_reason}")
        if self._current_group is not None or self._group_tensors or self._group_saved_tensor_counts:
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
        self._state = _PagedStashState.ACTIVE

    def finish_iteration(self) -> None:
        """Synchronously validate overflow after a single-rank forward/backward.

        Distributed runners must instead stack :meth:`check_overflow` with their
        other device flags, perform one all-reduce, and only then inspect the
        result on the host. Calling this rank-local diagnostic before a collective
        can deadlock when only some ranks overflow.
        """
        if self._current_group is not None or self._group_tensors or self._group_saved_tensor_counts:
            raise RuntimeError("Paged stash iteration finished with live expert groups")
        if self._backward_group_stack:
            raise RuntimeError(
                f"Paged stash iteration finished with pending backward groups {self._backward_group_stack}"
            )
        if self._backward_in_progress_group_id is not None:
            raise RuntimeError(
                "Paged stash iteration finished while expert backward is still active for group "
                f"{self._backward_in_progress_group_id}"
            )
        if self._overflow is not None and bool(self._overflow.item()):
            raise PagedStashOverflowError(
                "MoE paged stash capacity overflowed; discard this attempt and rerun without paged stash"
            )

    def check_overflow(self) -> torch.Tensor | None:
        """Return the device-resident overflow flag without synchronizing the host."""
        return self._overflow

    def reset_after_overflow(self) -> None:
        """Clear overflow metadata after the graph using these buffers is destroyed."""
        if self._current_group is not None or self._group_tensors or self._group_saved_tensor_counts:
            raise RuntimeError("Cannot reset paged stash while groups are live")
        if self._backward_group_stack or self._backward_in_progress_group_id is not None:
            raise RuntimeError("Cannot reset paged stash while expert backward is live")
        if self._overflow is not None:
            self._overflow.zero_()
        for buffer in self._buffers.values():
            buffer.reset()

    def restart_recording(self) -> None:
        """Drop fixed pages and start a fresh eager profile after graph teardown."""
        page_size = self._page_size
        buffer_size_factor = self._buffer_size_factor
        self.close()
        self.configure(
            enabled=True,
            page_size=page_size,
            buffer_size_factor=buffer_size_factor,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return bounded lifecycle and allocation diagnostics."""
        return {
            "state": self._state.value,
            "page_size": self._page_size,
            "buffer_size_factor": self._buffer_size_factor,
            "recorded_peak_tokens": dict(self._recorded_peak_tokens),
            "buffer_tokens": {key: buffer.total_tokens for key, buffer in self._buffers.items()},
            "buffer_bytes": {key: buffer.storage.nbytes for key, buffer in self._buffers.items()},
            "live_groups": len(self._group_tensors),
            "backward_schedule_depth": len(self._backward_group_stack),
            "backward_in_progress_group_id": self._backward_in_progress_group_id,
        }

    def close(self) -> None:
        """Release page buffers after every CUDA graph that references them is gone."""
        if self._current_group is not None or self._group_tensors or self._group_saved_tensor_counts:
            raise RuntimeError("Cannot close paged stash while groups are live")
        if self._backward_group_stack:
            raise RuntimeError("Cannot close paged stash while backward groups are live")
        if self._backward_in_progress_group_id is not None:
            raise RuntimeError("Cannot close paged stash while expert backward is live")
        self._buffers.clear()
        self._overflow = None
        self._recorded_peak_tokens.clear()
        self._recorded_current_tokens.clear()
        self._record_device = None
        self._recording_invalid_reason = None
        self._group_saved_tensor_counts.clear()
        self._backward_group_stack.clear()
        self._backward_in_progress_group_id = None
        self._next_group_id = 0
        self._disable_depth = 0
        self._state = _PagedStashState.DISABLED


_PAGED_STASH_MANAGER = PagedStashManager()


def get_paged_stash_manager() -> PagedStashManager:
    """Return the process-local TE expert paged-stash manager."""
    return _PAGED_STASH_MANAGER


__all__ = ["PagedStashManager", "PagedStashOverflowError", "get_paged_stash_manager"]
