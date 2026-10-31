# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Dropless NCCL-EP token transport through Transformer Engine."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType

import torch
import torch.distributed as dist

from nemo_automodel.components.models.common.utils import NcclEpConfig
from nemo_automodel.shared.import_utils import safe_import, safe_import_from

# NCCL-EP reads group options from the environment once, when the group is created.
# The scan layout fills each expert's receive slots in (source rank, token) order.
# The default single-NVLink-domain layout assigns slots atomically, so a
# non-reentrant activation-checkpoint recompute could pair recomputed expert
# activations with a layout other than the one the forward's routing handle holds.
_ORDERED_LAYOUT_ENV = "NCCL_EP_HT_EM_AG_SCAN_MODE"


@dataclass
class _Runtime:
    api: ModuleType | None = None
    group: dist.ProcessGroup | None = None
    settings: tuple[int | bool | None, ...] | None = None
    # Zero-copy symmetric buffers of shape [recv_capacity, hidden], shared by every layer:
    # each layer's combine reads the expert output before the next layer writes it, and
    # each layer's dispatch backward reads the input gradient before the next combine
    # backward overwrites it.
    expert_output: torch.Tensor | None = None
    expert_input_grad: torch.Tensor | None = None


_runtime = _Runtime()


def finalize_nccl_ep() -> None:
    """Release the owned NCCL-EP context before destroying its process group."""
    if _runtime.api is not None:
        # Drop the symmetric buffers while the group that registered them is still valid.
        _runtime.expert_output = None
        _runtime.expert_input_grad = None
        _runtime.api.ep_finalize()
        _runtime.api = None
        _runtime.group = None
        _runtime.settings = None


@contextmanager
def _ordered_layout_environment() -> Iterator[None]:
    """Select NCCL-EP's ordered expert layout for groups created inside the block."""
    previous = os.environ.get(_ORDERED_LAYOUT_ENV)
    os.environ[_ORDERED_LAYOUT_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_ORDERED_LAYOUT_ENV, None)
        else:
            os.environ[_ORDERED_LAYOUT_ENV] = previous


def _join_transport(
    *,
    group: dist.ProcessGroup,
    capacity: int,
    recv_capacity: int | None,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    zero_copy: bool,
) -> ModuleType:
    """Bootstrap the process-wide transport once, or check that it matches these settings.

    Args:
        group: NCCL expert-parallel group with contiguous expert ownership.
        capacity: Group-uniform input extent, a multiple of 64.
        recv_capacity: Fixed receive rows per rank, or None to size them per dispatch.
        hidden_dim: Width of each dispatched token.
        num_experts: Global expert count across the EP group.
        top_k: Distinct expert selections per token.
        zero_copy: Whether dispatch and combine use symmetric-memory windows; requires
            ``recv_capacity``. Bootstrap then also allocates the shared expert buffers.

    Returns:
        Transformer Engine's NCCL-EP API module.
    """
    available, api = safe_import("transformer_engine.pytorch.ep")
    if available:
        available, _ = safe_import_from("transformer_engine_torch", "ep_initialize")
    if not available:
        raise ImportError(
            "dispatcher='nccl_ep' requires Transformer Engine with NCCL-EP support "
            "and its transformer_engine.pytorch.ep API; the default public-wheel "
            "installation disables NCCL-EP."
        )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("NCCL-EP requires Hopper or newer CUDA GPUs (SM90+)")
    if dist.get_backend(group) != "nccl":
        raise ValueError("NCCL-EP requires an NCCL process group")
    settings = (capacity, recv_capacity, hidden_dim, num_experts, top_k, zero_copy, torch.cuda.current_device())
    if _runtime.api is None:
        if api.is_ep_bootstrapped():
            raise RuntimeError("NCCL-EP is already initialized outside AutoModel; finalize that context first")
        with _ordered_layout_environment():
            api.ep_bootstrap(
                group,
                num_experts=num_experts,
                max_tokens_per_rank=capacity,
                hidden_dim=hidden_dim,
                num_topk=top_k,
                # Zero lets NCCL-EP size its kernels. An explicit budget would also cap its
                # token-reordering kernels, which otherwise use every SM.
                max_num_sms=0,
                # Overflowing a fixed capacity traps on the device instead of dropping tokens.
                recv_capacity_per_rank=recv_capacity,
                zero_copy=zero_copy,
            )
        _runtime.api, _runtime.group, _runtime.settings = api, group, settings
        if zero_copy:
            # Collective: every EP rank registers each buffer as one window.
            _runtime.expert_output = api.symm_mem_alloc((recv_capacity, hidden_dim), torch.bfloat16, group)
            _runtime.expert_input_grad = api.symm_mem_alloc((recv_capacity, hidden_dim), torch.bfloat16, group)
    elif _runtime.group is not group or _runtime.settings != settings:
        raise ValueError("NCCL-EP supports one EP group and one capacity/shape configuration per process")
    return api


class NcclEpManager:
    """Layer-local routing state for the flex dispatcher's NCCL-EP path."""

    def __init__(
        self,
        config: NcclEpConfig,
        *,
        group: dist.ProcessGroup,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        expert_alignment: int = 1,
    ) -> None:
        """Join the process-wide transport and set up layer-local routing state.

        The first manager in a process bootstraps NCCL-EP collectively on ``group``, so
        every EP rank must construct it with identical arguments, outside forward and
        activation-checkpoint recomputation. Later managers must use the same settings.

        Args:
            config: NCCL-EP settings shared by every layer and EP rank.
            group: NCCL expert-parallel group with contiguous expert ownership.
            hidden_dim: Width of each dispatched token.
            num_experts: Global expert count, divisible by the EP group size.
            top_k: Distinct expert selections per token.
            expert_alignment: Power-of-two row multiple of each local expert's dispatched
                segment, as the expert GEMMs require; 1 packs segments without padding.
        """
        if config.max_tokens_per_rank is None:
            raise ValueError("dispatcher='nccl_ep' requires nccl_ep.max_tokens_per_rank")
        if group.size() < 2 or num_experts % group.size():
            raise ValueError("NCCL-EP requires EP size >= 2 dividing the global expert count")
        if not 0 < top_k <= num_experts or hidden_dim <= 0:
            raise ValueError("Invalid NCCL-EP top-k or hidden dimension")
        if expert_alignment < 1 or expert_alignment & (expert_alignment - 1):
            raise ValueError(f"NCCL-EP expert alignment must be a power of two, got {expert_alignment}")
        self.max_tokens = config.max_tokens_per_rank
        self.capacity = (config.max_tokens_per_rank + 63) // 64 * 64
        self.num_local_experts = num_experts // group.size()
        self.expert_alignment = expert_alignment
        self.recv_capacity = None
        if config.recv_capacity_factor is not None:
            # Each local expert's segment may grow by up to expert_alignment - 1 padding rows.
            routed = math.ceil(self.capacity * top_k * config.recv_capacity_factor)
            padded = routed + self.num_local_experts * (expert_alignment - 1)
            self.recv_capacity = -(-padded // expert_alignment) * expert_alignment
        self.zero_copy = config.zero_copy
        self.api = _join_transport(
            group=group,
            capacity=self.capacity,
            recv_capacity=self.recv_capacity,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            zero_copy=self.zero_copy,
        )
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.token_indices: torch.Tensor | None = None
        self.token_probs: torch.Tensor | None = None
        self.dispatched_probs: torch.Tensor | None = None
        self.tokens_per_expert: torch.Tensor | None = None
        self._buffer = None
        self._num_tokens = 0

    def setup_metadata(self, num_local_tokens: int, probs: torch.Tensor) -> None:
        """Select top-k from dense per-rank routing weights.

        Args:
            num_local_tokens: Local input token count.
            probs: Tensor of shape [tokens, ep_size, local_experts], with zero weights
                for inactive routes. The expert axes enumerate contiguous global expert
                IDs; ep_size is the number of ranks in the EP group.
        """
        self.token_probs, indices = probs.reshape(num_local_tokens, self.num_experts).topk(self.top_k, dim=-1)
        self.token_indices = indices.masked_fill(self.token_probs == 0, -1)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        """Dispatch into a fresh expert-packed buffer with differentiable weights.

        Args:
            hidden_states: BF16 CUDA tensor of shape [local_tokens, hidden], with at
                most ``max_tokens_per_rank`` rows.
            async_finish: Must be false; TE owns stream ordering.
            allocate_on_comm_stream: Must be false; TE owns allocations.

        Returns:
            BF16 tensor of shape [received_rows, hidden], packed by local expert and,
            within an expert, ordered by source rank and token. Each expert's segment is
            zero-padded to a multiple of ``expert_alignment`` rows. ``received_rows`` is the
            padded row count, or ``recv_capacity`` with ``recv_capacity_factor``, in which
            case rows past the padded count are uninitialized and never combined. Storage
            is unique to this call and retained by TE autograd through backward; with
            ``zero_copy`` it is a symmetric-memory window.
        """
        if async_finish or allocate_on_comm_stream:
            raise ValueError("NCCL-EP uses Transformer Engine stream ordering")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("The NCCL-EP dispatcher does not support CUDA Graph capture")
        if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
            raise ValueError("NCCL-EP dispatch requires BF16 [local_tokens, hidden] inputs")
        num_tokens, hidden = hidden_states.shape
        if hidden != self.hidden_dim or num_tokens > self.max_tokens:
            raise ValueError(
                "NCCL-EP input exceeds nccl_ep.max_tokens_per_rank or has an incompatible hidden dimension"
            )
        if self.token_indices is None or self.token_probs is None:
            raise RuntimeError("NCCL-EP routing metadata must be set before dispatch")
        if self.token_indices.shape != (num_tokens, self.top_k) or self.token_probs.shape != (num_tokens, self.top_k):
            raise ValueError("NCCL-EP indices and probabilities must have shape [local_tokens, top_k]")
        # Recompute may abandon a forward before combine. Never reuse its handle:
        # backward for another layer or microbatch can still own the old routing.
        buffer = self.api.EpBuffer(
            top_k=self.top_k,
            max_tokens_per_rank=self.capacity,
            hidden_dim=self.hidden_dim,
            num_local_experts=self.num_local_experts,
            recv_capacity_per_rank=self.recv_capacity,
            alignment=self.expert_alignment if self.expert_alignment > 1 else 0,
        )
        received, self.dispatched_probs, self.tokens_per_expert = self.api.ep_dispatch(
            buffer,
            hidden_states.contiguous(),
            self.token_indices.to(torch.int64).contiguous(),
            self.token_probs.float().contiguous(),
        )
        self._buffer = buffer
        self._num_tokens = num_tokens
        return received

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the already packed dispatch result without a second permutation.

        Args:
            hidden_states: Tensor of shape [received_rows, hidden], expert-packed.

        Returns:
            The input activation tensor of shape [received_rows, hidden], returned by
            reference, and FP32 routing weights of shape [received_rows], aligned with
            its expert-packed rows.
        """
        if self.dispatched_probs is None:
            raise RuntimeError("NCCL-EP dispatch has not completed")
        return hidden_states, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """Return device counts of shape [local_experts] for the packed rows, padding included."""
        if self.tokens_per_expert is None:
            raise RuntimeError("NCCL-EP dispatch has not completed")
        return self.tokens_per_expert

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Keep expert-packed outputs in the layout consumed by combine.

        Args:
            hidden_states: Tensor of shape [received_rows, hidden], expert-packed.

        Returns:
            The input tensor of shape [received_rows, hidden], returned by reference
            without copying or weighting it again.
        """
        return hidden_states

    def expert_buffers(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the shared symmetric buffers that zero-copy experts write into.

        Returns:
            None without ``zero_copy``. Otherwise detached BF16 views of shape
            [recv_capacity, hidden]: the expert output that combine reads, and the
            gradient of the dispatched tokens that dispatch backward reads. Both are
            shared by every layer; combine backward also writes the expert-output
            gradient into the second buffer before the experts consume it.
        """
        if not self.zero_copy:
            return None
        # Detached so autograd metadata attached to one layer's outputs never reaches the shared buffers.
        return _runtime.expert_output.detach(), _runtime.expert_input_grad.detach()

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        """Combine expert outputs back into the dispatched token order.

        Args:
            hidden_states: BF16 tensor of shape [received_rows, hidden],
                expert-packed and already multiplied by routing weights. Rows past
                the routed count are ignored. With ``zero_copy`` it must be the
                expert-output buffer from :meth:`expert_buffers`.
            async_finish: Must be false; TE owns stream ordering.
            allocate_on_comm_stream: Must be false; TE owns allocations.

        Returns:
            BF16 tensor of shape [local_tokens, hidden] in original token order.
        """
        if async_finish or allocate_on_comm_stream:
            raise ValueError("NCCL-EP uses Transformer Engine stream ordering")
        if self._buffer is None:
            raise RuntimeError("NCCL-EP combine requires a matching dispatch")
        grad_out = None
        if self.zero_copy:
            if hidden_states.data_ptr() != _runtime.expert_output.data_ptr():
                raise RuntimeError(
                    "nccl_ep.zero_copy requires the experts to write their output into the shared "
                    "symmetric buffer from expert_buffers()"
                )
            grad_out = _runtime.expert_input_grad.detach()
        result = self.api.ep_combine(
            self._buffer, hidden_states.contiguous(), num_local_tokens=self._num_tokens, grad_out=grad_out
        )
        self._buffer = None
        self.dispatched_probs = None
        self.tokens_per_expert = None
        self.token_indices = None
        self.token_probs = None
        return result
