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

"""Shared eager, dropless NCCL-EP transport through Transformer Engine."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nemo_automodel.shared.import_utils import safe_import, safe_import_from


@dataclass
class _Runtime:
    api: ModuleType | None = None
    group: dist.ProcessGroup | None = None
    settings: tuple[int, ...] | None = None


_runtime = _Runtime()


def finalize_nccl_ep() -> None:
    """Release the owned NCCL-EP context before destroying its process group."""
    if _runtime.api is not None:
        _runtime.api.ep_finalize()
        _runtime.api = None
        _runtime.group = None
        _runtime.settings = None


@dataclass(kw_only=True)
class NcclEpConfig:
    """NCCL-EP capacity, identical on every EP rank and layer in a process.

    Args:
        max_tokens_per_rank: Upper bound on tokens entering one MoE layer per
            microbatch, including masked tokens. Required when selecting NCCL-EP.
            Rounded up to a multiple of 64 for the transport. Shorter inputs are
            padded with invalid routes; receive buffers remain dropless and dynamic.
    """

    max_tokens_per_rank: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens_per_rank is not None and self.max_tokens_per_rank <= 0:
            raise ValueError("nccl_ep.max_tokens_per_rank must be positive")

    def build(
        self,
        *,
        group: dist.ProcessGroup,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        num_sms: int,
    ) -> NcclEpManager:
        """Initialize the transport collectively, outside forward/recomputation.

        Args:
            group: NCCL expert-parallel group with contiguous expert ownership.
            hidden_dim: Width of each dispatched token.
            num_experts: Global expert count, divisible by the EP group size.
            top_k: Distinct expert selections per token.
            num_sms: Transport SM budget; zero lets NCCL-EP choose.

        Returns:
            A layer-local manager sharing the process-wide communication context.
        """
        if self.max_tokens_per_rank is None:
            raise ValueError("dispatcher='nccl_ep' requires nccl_ep.max_tokens_per_rank")
        if group.size() < 2 or num_experts % group.size():
            raise ValueError("NCCL-EP requires EP size >= 2 dividing the global expert count")
        if not 0 < top_k <= num_experts or hidden_dim <= 0 or num_sms < 0:
            raise ValueError("Invalid NCCL-EP top-k, hidden dimension, or SM budget")
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
        capacity = (self.max_tokens_per_rank + 63) // 64 * 64
        settings = (capacity, hidden_dim, num_experts, top_k, num_sms, torch.cuda.current_device())
        if _runtime.api is None:
            if api.is_ep_bootstrapped():
                raise RuntimeError("NCCL-EP is already initialized outside AutoModel; finalize that context first")
            api.ep_bootstrap(
                group,
                num_experts=num_experts,
                max_tokens_per_rank=capacity,
                hidden_dim=hidden_dim,
                num_topk=top_k,
                max_num_sms=num_sms,
                recv_capacity_per_rank=None,
            )
            _runtime.api, _runtime.group, _runtime.settings = api, group, settings
        elif _runtime.group is not group or _runtime.settings != settings:
            raise ValueError("NCCL-EP supports one EP group and one capacity/shape/SM configuration per process")
        return NcclEpManager(
            api=api,
            max_tokens=self.max_tokens_per_rank,
            capacity=capacity,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_local_experts=num_experts // group.size(),
            top_k=top_k,
        )


class NcclEpManager:
    """Layer-local routing state for the flex dispatcher's eager NCCL-EP path."""

    def __init__(
        self,
        *,
        api: ModuleType,
        max_tokens: int,
        capacity: int,
        hidden_dim: int,
        num_experts: int,
        num_local_experts: int,
        top_k: int,
    ) -> None:
        self.api = api
        self.max_tokens = max_tokens
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
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
            probs: Tensor of shape [tokens, EP ranks, local_experts], with
                zero weights for inactive routes.
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
            hidden_states: BF16 CUDA tensor of shape [local_tokens, hidden].
            async_finish: Must be false; TE owns stream ordering.
            allocate_on_comm_stream: Must be false; TE owns allocations.

        Returns:
            BF16 tensor of shape [received_tokens, hidden], packed by local expert.
            Storage is unique to this call and retained by TE autograd through backward.
        """
        if async_finish or allocate_on_comm_stream:
            raise ValueError("NCCL-EP uses Transformer Engine stream ordering")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("The eager NCCL-EP dispatcher does not support CUDA Graph capture")
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
            recv_capacity_per_rank=None,
        )
        padding = self.capacity - num_tokens
        tokens = F.pad(hidden_states, (0, 0, 0, padding)).contiguous()
        indices = F.pad(self.token_indices.to(torch.int64), (0, 0, 0, padding), value=-1).contiguous()
        weights = F.pad(self.token_probs.float(), (0, 0, 0, padding)).contiguous()
        received, self.dispatched_probs, self.tokens_per_expert = self.api.ep_dispatch(buffer, tokens, indices, weights)
        self._buffer = buffer
        self._num_tokens = num_tokens
        return received

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the already packed dispatch result without a second permutation.

        Args:
            hidden_states: Tensor of shape [received_tokens, hidden], expert-packed.

        Returns:
            The same activations and FP32 weights of shape [received_tokens].
        """
        if self.dispatched_probs is None:
            raise RuntimeError("NCCL-EP dispatch has not completed")
        return hidden_states, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """Return device counts of shape [local_experts] for the packed rows."""
        if self.tokens_per_expert is None:
            raise RuntimeError("NCCL-EP dispatch has not completed")
        return self.tokens_per_expert

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Keep expert-packed outputs in the layout consumed by combine.

        Args:
            hidden_states: Tensor of shape [received_tokens, hidden], expert-packed.

        Returns:
            The same tensor, without copying or weighting it again.
        """
        return hidden_states

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        """Combine expert outputs and remove invalid padding tokens.

        Args:
            hidden_states: BF16 tensor of shape [received_tokens, hidden],
                expert-packed and already multiplied by routing weights.
            async_finish: Must be false; TE owns stream ordering.
            allocate_on_comm_stream: Must be false; TE owns allocations.

        Returns:
            BF16 tensor of shape [local_tokens, hidden] in original token order.
        """
        if async_finish or allocate_on_comm_stream:
            raise ValueError("NCCL-EP uses Transformer Engine stream ordering")
        if self._buffer is None:
            raise RuntimeError("NCCL-EP combine requires a matching dispatch")
        result = self.api.ep_combine(self._buffer, hidden_states.contiguous(), num_local_tokens=self.capacity)
        self._buffer = None
        self.dispatched_probs = None
        self.tokens_per_expert = None
        self.token_indices = None
        self.token_probs = None
        return result[: self._num_tokens]
