# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .fused_a2a import (
    fused_combine,
    fused_dispatch,
)
from .fused_indices_converter import (
    fused_indices_to_multihot,
)
from .moe_utils import (
    permute,
    unpermute,
)

SHARING_DEEPEP_MANAGER = True

""" We use the following notation throughout this file:
     H: hidden size
     B: micro batch size
     S: sequence length
     TP: tensor model parallel size
     EP: expert model parallel size
     num_local_tokens: S/TP*B
     num_global_tokens: num_local_tokens*TP*EP
"""


class _DispatchManager(ABC):
    """
    A manager class to handle dispatch and combine processes for MoE models.

    DispatcherManager handles token dispatching according to the routing_map of format
    [num_local_tokens, world_size, num_instances]. The routing_map is a 3D tensor where each
    element indicates whether a token should be sent to a specific rank.

    num_instances is the maximum number of tokens instances dispatched into a target rank, it
    can be the number of local experts, or the size of sub_group.
    """

    @abstractmethod
    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        """Set up metadata of routing_map and probs."""
        pass

    @abstractmethod
    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Dispatch the hidden_states according to the routing_map."""
        pass

    @abstractmethod
    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine the hidden_states after expert processing."""
        pass

    @abstractmethod
    def get_dispached_metadata(self) -> torch.Tensor:
        """Get the metadata of the dispatched hidden_states."""
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the permuted hidden states by instances."""
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the restored hidden states by instances."""
        pass


class _DeepepManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        capacity_factor: Optional[float] = None,
        num_experts: Optional[int] = None,
        num_local_experts: Optional[int] = None,
        router_dtype: Optional[str] = None,
        moe_router_expert_pad_multiple: Optional[int] = None,
    ):
        self.group = group
        self.router_topk = router_topk
        self.capacity_factor = capacity_factor
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.router_dtype = router_dtype
        self.moe_router_expert_pad_multiple = moe_router_expert_pad_multiple

        # Metadata
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from https://github.com/deepseek-ai/deepep."
            )

    def setup_metadata(self, num_local_tokens: int, probs: torch.Tensor):
        """
        Process routing map and probabilities to prepare dispatch metadata
        """
        probs = probs.reshape(num_local_tokens, self.num_experts)
        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = torch.topk(probs, self.router_topk, dim=-1)
        # Mask the indices of dropped tokens with -1
        if self.capacity_factor is not None:
            mask = self.token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        """
        Dispatch the hidden_states
        """
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                # print("DeepEP only supports float32 probs, please set --moe-router-dtype=fp32")
                # TODO: remove this
                pass
            self.token_probs = self.token_probs.float()  # downcast or upcast
        (
            hidden_states,
            dispatched_indices,
            dispatched_probs,
            num_tokens_per_expert,
            handle,
        ) = fused_dispatch(
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            self.group,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

        return hidden_states

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (torch.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - routing_map: Multihot vector.
                - probs: Multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.long,
            device=indices.device,
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.float,
            device=indices.device,
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(mask.sum(dim=1))
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_dispached_metadata(self) -> torch.Tensor:
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        """
        Reverse process using fused kernel to unpermute and perform all-to-all in single step
        """
        hidden_states, _ = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Release the handle after combine operation
        self.handle = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
        """
        if self.permute_fusion:
            self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(
                self.dispatched_indices,
                self.dispatched_probs,
                self.num_local_experts,
            )
        else:
            self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs
            )
        if self.moe_router_expert_pad_multiple:
            with torch.cuda.nvtx.range("pad_routing_map"):
                from megatron.core.transformer.moe.moe_utils import pad_routing_map

                self.dispatched_routing_map = pad_routing_map(
                    self.dispatched_routing_map, self.moe_router_expert_pad_multiple
                )
            # self.tokens_per_expert = self.dispatched_routing_map.sum(dim=0)
            self.tokens_per_expert = (
                torch.ceil(self.tokens_per_expert / self.moe_router_expert_pad_multiple)
                * self.moe_router_expert_pad_multiple
            )
            self.tokens_per_expert = self.tokens_per_expert.long()

        self.hidden_shape_before_permute = hidden_states.shape
        assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
        hidden_states, permuted_probs, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            probs=self.dispatched_probs,
            num_out_tokens=self.tokens_per_expert.sum().item(),
            fused=self.permute_fusion,
        )
        if self.router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return hidden_states, permuted_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Restore the hidden states to their original ordering before expert processing
        """
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            fused=self.permute_fusion,
        )
        return hidden_states


@dataclass
class MoEConfig:
    moe_enable_deepep: bool = True
    """Enable DeepEP for efficient token dispatching and combine in MoE models."""

    moe_permute_fusion: bool = False
    """Fuse token rearrangement ops during token dispatching."""

    moe_expert_capacity_factor: Optional[float] = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token
    will be dropped. The default is None."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_router_expert_pad_multiple: Optional[int] = None
    """Number of tokens to pad to a multiple of for each expert."""

    num_moe_experts: int = 64
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    moe_router_dtype: str = "fp32"
    """Data type for routing and expert output weighted averaging. Using fp32 or fp64 can
    improve stability especially when the number of experts is large (e.g. finegrained-moe).
    None means no changes for dtype."""


class MoEFlexTokenDispatcher:
    """
    Flex token dispatcher using DeepEP.
    """

    shared_comm_manager: _DeepepManager = None  # shared by all instances of MoEFlexTokenDispatcher

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: MoEConfig,
        ep_group: torch.distributed.ProcessGroup,
    ):
        """
        Initialize the Flex token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (MoEConfig): Configuration for the transformer model.
            group (torch.distributed.ProcessGroup): Process group for MoE operations.
        """
        self.config = config
        self.shared_experts = None

        self.group = ep_group
        self.ep_size = ep_group.size()
        # use model_comm_pgs.expt_tp_group as tensor parallel group in this module.
        # self.tp_group = tp_group
        # self.tp_group = tp_ep_group

        self.tp_size = 1  # TP is not used
        # self.tp_rank = self.tp_group.rank()

        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        assert self.tp_size * self.ep_size > 1, "Flex token dispatcher requires TPxEP > 1"
        assert self.config.moe_enable_deepep, (
            "DeepEP is not enabled. Please set --moe-enable-deepep to use DeepEP backend."
        )
        if SHARING_DEEPEP_MANAGER:
            if MoEFlexTokenDispatcher.shared_comm_manager is None:
                MoEFlexTokenDispatcher.shared_comm_manager = _DeepepManager(
                    group=ep_group,
                    router_topk=self.tp_size * self.config.moe_router_topk,
                    permute_fusion=self.config.moe_permute_fusion,
                    capacity_factor=self.config.moe_expert_capacity_factor,
                    num_experts=self.tp_size * self.config.num_moe_experts,
                    num_local_experts=self.num_local_experts,
                    router_dtype=self.config.moe_router_dtype,
                    moe_router_expert_pad_multiple=self.config.moe_router_expert_pad_multiple,
                )
            self._comm_manager = MoEFlexTokenDispatcher.shared_comm_manager
        else:
            self._comm_manager = _DeepepManager(
                group=ep_group,
                router_topk=self.tp_size * self.config.moe_router_topk,
                permute_fusion=self.config.moe_permute_fusion,
                capacity_factor=self.config.moe_expert_capacity_factor,
                num_experts=self.tp_size * self.config.num_moe_experts,
                num_local_experts=self.num_local_experts,
                router_dtype=self.config.moe_router_dtype,
                moe_router_expert_pad_multiple=self.config.moe_router_expert_pad_multiple,
            )

    def _initialize_metadata(self, num_local_tokens: int, probs: torch.Tensor) -> torch.Tensor:
        """
        Initialize the routing map and probs to a unified format covering the TPxEP group.
        This design decouples the communication group from underlying model parallelism groups,
        such that the communication strategy of tokens can be agnostic of TP size and EP size.
        """
        world_size = self.tp_size * self.ep_size
        probs = (
            probs.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        return probs

    def dispatch_preprocess2(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        """
        Preprocesses the hidden states and routing information before dispatching tokens to experts.
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self._comm_manager.token_probs = token_probs
        self._comm_manager.token_indices = token_indices
        return hidden_states, self._comm_manager.token_probs

    def dispatch_preprocess(self, hidden_states: torch.Tensor, num_local_tokens: int, probs: torch.Tensor):
        """
        Preprocesses the hidden states and routing information before dispatching tokens to experts.
        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed
            num_local_tokens (int): Number of tokens to be processed
            probs (torch.Tensor): Routing probabilities for each token-expert pair

        Returns:
            Tuple containing:
            - torch.Tensor: Reshaped hidden states
            - torch.Tensor: Token probabilities from the communication manager
            - None: Placeholder for compatibility
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Initialize metadata
        probs = self._initialize_metadata(num_local_tokens=num_local_tokens, probs=probs)

        self._comm_manager.setup_metadata(num_local_tokens=num_local_tokens, probs=probs)
        return hidden_states, self._comm_manager.token_probs

    def dispatch_all_to_all(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Performs all-to-all communication to dispatch tokens across expert parallel ranks.
        """
        return (
            self._comm_manager.dispatch(hidden_states, async_finish, allocate_on_comm_stream),
            self._comm_manager.dispatched_probs,
        )

    def dispatch_postprocess(self, hidden_states: torch.Tensor):
        """
        Post-processes the dispatched hidden states after all-to-all communication.

        This method retrieves the permuted hidden states by experts, calculates the number of tokens
        per expert, and returns the processed data ready for expert processing.
        """
        global_input_tokens, permuted_probs = self._comm_manager.get_permuted_hidden_states_by_experts(hidden_states)
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()
        return global_input_tokens, tokens_per_expert, permuted_probs

    def token_permutation(
        self, hidden_states: torch.Tensor, num_local_tokens: int, probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permutes tokens according to probs and dispatches them to experts.

        This method implements the token permutation process in three steps:
        1. Preprocess the hidden states
        2. Perform all-to-all communication to dispatch tokens
        3. Post-process the dispatched tokens for expert processing
        """
        hidden_states, _ = self.dispatch_preprocess(
            hidden_states=hidden_states, num_local_tokens=num_local_tokens, probs=probs
        )
        hidden_states, _ = self.dispatch_all_to_all(hidden_states, async_finish=False, allocate_on_comm_stream=False)
        global_input_tokens, tokens_per_expert, permuted_probs = self.dispatch_postprocess(hidden_states)

        return global_input_tokens, tokens_per_expert, permuted_probs

    def token_permutation2(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permutes tokens according to probs and dispatches them to experts.

        This method implements the token permutation process in three steps:
        1. Preprocess the hidden states
        2. Perform all-to-all communication to dispatch tokens
        3. Post-process the dispatched tokens for expert processing
        """
        hidden_states, _ = self.dispatch_preprocess2(
            hidden_states=hidden_states,
            num_local_tokens=num_local_tokens,
            token_probs=token_probs,
            token_indices=token_indices,
        )
        hidden_states, _ = self.dispatch_all_to_all(hidden_states, async_finish=False, allocate_on_comm_stream=False)
        global_input_tokens, tokens_per_expert, permuted_probs = self.dispatch_postprocess(hidden_states)

        return global_input_tokens, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states: torch.Tensor):
        """
        Pre-processes the hidden states before combining them after expert processing.

        This method restores the hidden states to their original ordering before expert processing
        by using the communication manager's restoration function.
        """
        hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)
        return hidden_states

    def combine_all_to_all(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Performs all-to-all communication to combine tokens after expert processing.
        """
        return self._comm_manager.combine(hidden_states, async_finish, allocate_on_comm_stream)

    def combine_postprocess(self, hidden_states: torch.Tensor):
        """
        Post-processes the combined hidden states after all-to-all communication.

        This method reshapes the combined hidden states to match the original input shape.
        """
        return hidden_states.view(self.hidden_shape)

    def token_unpermutation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Reverses the token permutation process to restore the original token order.

        This method implements the token unpermutation process in three steps:
        1. Pre-process the hidden states to restore their original ordering
        2. Perform all-to-all communication to combine tokens
        3. Post-process the combined tokens to match the original input shape
        """
        hidden_states = self.combine_preprocess(hidden_states)
        hidden_states = self.combine_all_to_all(hidden_states, False, False)
        hidden_states = self.combine_postprocess(hidden_states)

        return hidden_states
