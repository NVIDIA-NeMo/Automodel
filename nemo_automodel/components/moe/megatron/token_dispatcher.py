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
from typing import List, Literal, Optional, Tuple

import torch

from .fused_a2a import (
    fused_combine,
    fused_dispatch,
    hybrid_ep_combine,
    hybrid_ep_dispatch,
    set_deepep_num_sms,
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
        num_experts: Optional[int] = None,
        num_local_experts: Optional[int] = None,
        router_dtype: Optional[str] = None,
    ):
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.router_dtype = router_dtype

        # Metadata
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from https://github.com/deepseek-ai/deepep."
            )

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        """
        Process routing map and probabilities to prepare dispatch metadata.

        Args:
            routing_map (torch.Tensor): Token to expert routing map [num_tokens, num_experts].
            probs (torch.Tensor): Routing probabilities [num_tokens, num_experts].
        """
        num_tokens = routing_map.shape[0]
        routing_map = routing_map.reshape(num_tokens, self.num_experts)
        probs = probs.reshape(num_tokens, self.num_experts)
        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = torch.topk(probs, self.router_topk, dim=-1)

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


class _HybridEPManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    HybridEP backend. See https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep for more details.

    The workflow of the HybridEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Permute tokens for communication, perform all-to-all communication,
        and permute tokens for experts in single step
    (3) combine():
        - Unpermute tokens for communication, perform all-to-all communication,
        and unpermute tokens for attention in single step
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        num_experts: int,
        router_topk: int,
        permute_fusion: bool = False,
        moe_hybridep_num_sms: int = 24,
    ):
        """
        Initialize the HybridEP dispatcher.

        Args:
            group (torch.distributed.ProcessGroup): The process group to use for communication.
                This should be the ETPxEP group.
            num_local_experts (int): The number of local experts.
            num_experts (int): The total number of experts in the group.
            router_topk (int): The number of experts each token is routed to.
            permute_fusion (bool): Whether to use fused permutation.
            moe_hybridep_num_sms (int): Number of SMs used by HybridEP APIs.
        """
        self.group = group
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.moe_hybridep_num_sms = moe_hybridep_num_sms
        # Actually the the up-bound for the number of tokens
        # after permute op, None means no up-bound, will cause a CPU sync
        self.num_permuted_tokens = None

        # Metadata
        self.token_probs: Optional[torch.Tensor] = None
        self.routing_map: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None
        # Used for padding the output for each expert
        self.pad_multiple = None

        if hybrid_ep_dispatch is None:
            raise ImportError(
                "HybridEP is not installed. Please install HybridEP package from "
                "https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep."
            )

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        """
        Process routing map and probabilities to prepare dispatch metadata.

        Args:
            routing_map (torch.Tensor): Token to expert routing map [num_tokens, num_experts].
            probs (torch.Tensor): Routing probabilities [num_tokens, num_experts].
        """
        num_tokens = routing_map.shape[0]
        self.routing_map = routing_map.reshape(num_tokens, self.num_experts)
        self.token_probs = probs.reshape(num_tokens, self.num_experts)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> torch.Tensor:
        # HybridEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            self.token_probs = self.token_probs.float()  # downcast or upcast
        dispatched_hidden, self.dispatched_probs, _, tokens_per_expert, self.handle = hybrid_ep_dispatch(
            x=hidden_states,
            routing_map=self.routing_map,
            probs=self.token_probs,
            group=self.group,
            num_local_experts=self.num_local_experts,
            num_sms_dispatch_api=self.moe_hybridep_num_sms,
            num_sms_combine_api=self.moe_hybridep_num_sms,
            num_permuted_tokens=self.num_permuted_tokens,
            pad_multiple=self.pad_multiple,
        )

        self.tokens_per_expert = tokens_per_expert
        # self.num_permuted_tokens is necessary to allocate the output tensor for permute
        self.num_permuted_tokens = self.tokens_per_expert.sum()

        return dispatched_hidden

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> torch.Tensor:
        hidden_states = hybrid_ep_combine(
            x=hidden_states,
            handle=self.handle,
            num_permuted_tokens=self.num_permuted_tokens,
            pad_multiple=self.pad_multiple,
        )
        # Release the used handle/num_permuted_tokens which could change in each iteration.
        self.handle = None
        self.num_permuted_tokens = None
        return hidden_states

    def _indices_to_multihot(self, indices: torch.Tensor, probs: torch.Tensor):
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
            (batch_size, self.num_experts),
            dtype=torch.bool,
            device=indices.device,
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_experts),
            dtype=torch.float,
            device=indices.device,
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(mask.sum(dim=1))
        multihot_routing_map[row_indices, valid_indices] = True
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map, multihot_probs

    def setup_metadata_from_indices(self, token_indices: torch.Tensor, token_probs: torch.Tensor):
        """
        Process token indices and probabilities to prepare dispatch metadata.
        Converts from topk indices format to multihot routing_map format.

        Args:
            token_indices (torch.Tensor): Token indices [num_tokens, topk], -1 means masked out.
            token_probs (torch.Tensor): Token probabilities [num_tokens, topk].
        """
        if self.permute_fusion:
            self.routing_map, self.token_probs = fused_indices_to_multihot(token_indices, token_probs, self.num_experts)
        else:
            self.routing_map, self.token_probs = self._indices_to_multihot(token_indices, token_probs)

    def get_dispached_metadata(self) -> torch.Tensor:
        return None, self.dispatched_probs

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states, self.dispatched_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert


@dataclass
class MoEConfig:
    moe_permute_fusion: bool = False
    """Fuse token rearrangement ops during token dispatching."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    num_moe_experts: int = 64
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    moe_router_dtype: str = "fp32"
    """Data type for routing and expert output weighted averaging. Using fp32 or fp64 can
    improve stability especially when the number of experts is large (e.g. finegrained-moe).
    None means no changes for dtype."""

    moe_flex_dispatcher_backend: Literal["deepep", "hybridep"] = "deepep"
    """Backend for the flex token dispatcher. Options: 'deepep' or 'hybridep'.
    The hybridep backend uses HybridEP for fused all-to-all communication which can provide
    performance improvements especially for intra-node and multi-node NVLink scenarios."""

    moe_deepep_num_sms: int = 20
    """Number of SMs to use for DeepEP backend."""

    moe_hybridep_num_sms: int = 24
    """Number of SMs to use for HybridEP dispatch and combine APIs."""


class MoEFlexTokenDispatcher:
    """
    Flex token dispatcher supporting both DeepEP and HybridEP backends.
    """

    shared_deepep_manager: _DeepepManager = None  # shared by all instances using DeepEP
    shared_hybridep_manager: _HybridEPManager = None  # shared by all instances using HybridEP

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
            ep_group (torch.distributed.ProcessGroup): Process group for MoE operations.
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

        backend = self.config.moe_flex_dispatcher_backend

        if backend == "deepep":
            # Set DeepEP SM count
            if set_deepep_num_sms is not None:
                set_deepep_num_sms(self.config.moe_deepep_num_sms)
            if SHARING_DEEPEP_MANAGER:
                if MoEFlexTokenDispatcher.shared_deepep_manager is None:
                    MoEFlexTokenDispatcher.shared_deepep_manager = _DeepepManager(
                        group=ep_group,
                        router_topk=self.tp_size * self.config.moe_router_topk,
                        permute_fusion=self.config.moe_permute_fusion,
                        num_experts=self.tp_size * self.config.num_moe_experts,
                        num_local_experts=self.num_local_experts,
                        router_dtype=self.config.moe_router_dtype,
                    )
                self._comm_manager = MoEFlexTokenDispatcher.shared_deepep_manager
            else:
                self._comm_manager = _DeepepManager(
                    group=ep_group,
                    router_topk=self.tp_size * self.config.moe_router_topk,
                    permute_fusion=self.config.moe_permute_fusion,
                    num_experts=self.tp_size * self.config.num_moe_experts,
                    num_local_experts=self.num_local_experts,
                    router_dtype=self.config.moe_router_dtype,
                )
        elif backend == "hybridep":
            if SHARING_DEEPEP_MANAGER:
                if MoEFlexTokenDispatcher.shared_hybridep_manager is None:
                    MoEFlexTokenDispatcher.shared_hybridep_manager = _HybridEPManager(
                        group=ep_group,
                        num_local_experts=self.num_local_experts,
                        num_experts=self.tp_size * self.config.num_moe_experts,
                        router_topk=self.tp_size * self.config.moe_router_topk,
                        permute_fusion=self.config.moe_permute_fusion,
                        moe_hybridep_num_sms=self.config.moe_hybridep_num_sms,
                    )
                self._comm_manager = MoEFlexTokenDispatcher.shared_hybridep_manager
            else:
                self._comm_manager = _HybridEPManager(
                    group=ep_group,
                    num_local_experts=self.num_local_experts,
                    num_experts=self.tp_size * self.config.num_moe_experts,
                    router_topk=self.tp_size * self.config.moe_router_topk,
                    permute_fusion=self.config.moe_permute_fusion,
                    moe_hybridep_num_sms=self.config.moe_hybridep_num_sms,
                )
        else:
            raise ValueError(
                f"Invalid backend: {backend}. "
                "Please set moe_flex_dispatcher_backend='deepep' or "
                "moe_flex_dispatcher_backend='hybridep'"
            )

    def _initialize_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the routing map and probs to a unified format covering the TPxEP group.
        This design decouples the communication group from underlying model parallelism groups,
        such that the communication strategy of tokens can be agnostic of TP size and EP size.

        This function expands the routing_map from shape [num_local_tokens, num_experts] to
        [num_local_tokens, world_size, num_local_experts]. Each element in the routing_map
        indicates whether a token should be sent to a specific rank. Specifically, the
        routing_map is replicated across TP group since each TP ranks in a TP group should
        receive the same tokens.
        """
        num_local_tokens = routing_map.shape[0]
        world_size = self.tp_size * self.ep_size
        # Organize routing map and probs to [num_local_tokens, world_size, num_local_experts]
        routing_map = (
            routing_map.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        probs = (
            probs.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        return routing_map, probs

    def dispatch_preprocess2(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        """
        Preprocesses the hidden states and routing information before dispatching tokens to experts.
        This method accepts pre-computed token_probs and token_indices in topk format.

        For DeepEP backend: uses token_indices and token_probs directly.
        For HybridEP backend: converts token_indices to routing_map (multihot format) using
            fused_indices_to_multihot when permute_fusion is enabled.
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        if isinstance(self._comm_manager, _HybridEPManager):
            # HybridEP needs routing_map in multihot format, convert from indices
            self._comm_manager.setup_metadata_from_indices(token_indices, token_probs)
        else:
            # DeepEP uses token_indices and token_probs directly
            self._comm_manager.token_probs = token_probs
            self._comm_manager.token_indices = token_indices

        return hidden_states, self._comm_manager.token_probs

    def dispatch_preprocess(self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor):
        """
        Preprocesses the hidden states and routing information before dispatching tokens to experts.

        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed.
            routing_map (torch.Tensor): Map indicating which expert each token should be routed to.
            probs (torch.Tensor): Routing probabilities for each token-expert pair.

        Returns:
            Tuple containing:
            - torch.Tensor: Reshaped hidden states
            - torch.Tensor: Token probabilities from the communication manager
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Initialize metadata
        routing_map, probs = self._initialize_metadata(routing_map, probs)

        self._comm_manager.setup_metadata(routing_map, probs)
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
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permutes tokens according to routing_map and probs, and dispatches them to experts.

        This method implements the token permutation process in three steps:
        1. Preprocess the hidden states
        2. Perform all-to-all communication to dispatch tokens
        3. Post-process the dispatched tokens for expert processing
        """
        hidden_states, _ = self.dispatch_preprocess(hidden_states=hidden_states, routing_map=routing_map, probs=probs)
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
