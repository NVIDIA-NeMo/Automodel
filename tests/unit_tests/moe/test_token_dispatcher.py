# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
    _DeepepManager,
    _HybridEPManager,
    _HybridEPMetadataProcessor,
)


@pytest.fixture
def hybrid_ep_manager():
    """Create a _HybridEPManager with mocked hybrid_ep_dispatch import."""
    with patch(
        "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
        new=lambda *a, **kw: None,
    ):
        manager = _HybridEPManager(
            group=None,
            num_local_experts=2,
            num_experts=8,
            router_topk=2,
        )
    return manager


class TestIndicesToMultihot:
    """Tests for _HybridEPManager._indices_to_multihot."""

    def test_basic(self, hybrid_ep_manager):
        """Basic topk=2 case with valid indices."""
        indices = torch.tensor([[0, 3], [1, 5]])
        probs = torch.tensor([[0.6, 0.4], [0.7, 0.3]])

        routing_map, multihot_probs = hybrid_ep_manager._indices_to_multihot(indices, probs)

        assert routing_map.shape == (2, 8)
        assert routing_map[0, 0] and routing_map[0, 3]
        assert routing_map[1, 1] and routing_map[1, 5]
        assert routing_map.sum() == 4

        assert multihot_probs[0, 0] == pytest.approx(0.6)
        assert multihot_probs[0, 3] == pytest.approx(0.4)
        assert multihot_probs[1, 1] == pytest.approx(0.7)
        assert multihot_probs[1, 5] == pytest.approx(0.3)

    def test_scoped_processor_matches_existing_probabilities(self, hybrid_ep_manager):
        indices = torch.tensor([[0, 3], [1, -1]])
        probs = torch.tensor([[0.6, 0.4], [0.7, 0.0]])
        processor = _HybridEPMetadataProcessor(num_experts=8, permute_fusion=False)

        expected_routing, expected_probs = hybrid_ep_manager._indices_to_multihot(indices, probs)
        actual_routing, actual_probs = processor(indices, probs)

        torch.testing.assert_close(actual_routing, expected_routing)
        torch.testing.assert_close(actual_probs, expected_probs)

    def test_scoped_processor_preserves_dense_probability_gradients(self):
        indices = torch.tensor([[0, 3], [1, 5]])
        probs = torch.tensor([[0.6, 0.4], [0.7, 0.3]], requires_grad=True)
        processor = _HybridEPMetadataProcessor(num_experts=8, permute_fusion=False)

        _, dense_probs = processor(indices, probs)
        dense_weights = torch.arange(16, dtype=dense_probs.dtype).reshape(2, 8)
        (dense_probs * dense_weights).sum().backward()

        expected_grad = dense_weights.gather(1, indices)
        torch.testing.assert_close(probs.grad, expected_grad)

    def test_dispatch_passes_dense_routing_and_probs(self, hybrid_ep_manager):
        routing_map = torch.tensor([[True, False], [False, True]])
        dense_probs = torch.zeros(2, 8)
        hidden = torch.randn(2, 4)
        hybrid_ep_manager.routing_map = routing_map
        hybrid_ep_manager.token_probs = dense_probs
        dispatch = Mock(return_value=(hidden, dense_probs, None, torch.tensor([1, 1]), object()))

        with patch("nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch", dispatch):
            actual = hybrid_ep_manager.dispatch(hidden)

        assert actual is hidden
        dispatch.assert_called_once()
        kwargs = dispatch.call_args.kwargs
        assert kwargs["routing_map"] is routing_map
        assert kwargs["probs"] is dense_probs
        assert "topk_idx" not in kwargs
        assert "num_experts" not in kwargs

    def test_permute_fusion_is_forwarded_to_dispatch_and_combine(self, hybrid_ep_manager):
        manager = hybrid_ep_manager
        manager.moe_hybridep_permute_fusion = True
        hidden = torch.randn(2, 4)
        dense_probs = torch.zeros(2, 8)
        manager.routing_map = torch.tensor([[True, False], [False, True]])
        manager.token_probs = dense_probs
        handle = object()
        dispatch = Mock(return_value=(hidden, dense_probs, None, torch.tensor([1, 1]), handle))
        combine = Mock(return_value=hidden)

        with (
            patch("nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch", dispatch),
            patch("nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_combine", combine),
        ):
            manager.dispatch(hidden)
            manager.combine(hidden)

        assert dispatch.call_args.kwargs["fuse_permute"] is True
        assert combine.call_args.kwargs["fuse_permute"] is True

    def test_hybridep_constructor_tuning_is_forwarded(self, hybrid_ep_manager):
        manager = hybrid_ep_manager
        manager.moe_hybridep_num_sms_preprocessing = 132
        manager.moe_hybridep_num_blocks_permute = 112
        manager.moe_hybridep_num_blocks_unpermute = 111
        manager.routing_map = torch.tensor([[True, False], [False, True]])
        manager.token_probs = torch.zeros(2, 8)
        dispatch = Mock(return_value=(torch.randn(2, 4), None, None, torch.tensor([1, 1]), object()))

        with patch("nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch", dispatch):
            manager.dispatch(torch.randn(2, 4))

        kwargs = dispatch.call_args.kwargs
        assert kwargs["num_sms_preprocessing_api"] == 132
        assert kwargs["num_blocks_permute"] == 112
        assert kwargs["num_blocks_unpermute"] == 111

    def test_dispatch_preprocess_expands_dense_routing(self, hybrid_ep_manager):
        dispatcher = object.__new__(MoEFlexTokenDispatcher)
        dispatcher._comm_manager = hybrid_ep_manager
        dispatcher.hybridep_metadata_processor = _HybridEPMetadataProcessor(num_experts=8, permute_fusion=False)
        hidden = torch.randn(2, 4)
        token_indices = torch.tensor([[0, 3], [1, 5]])
        token_probs = torch.tensor([[0.6, 0.4], [0.7, 0.3]])

        actual_hidden, actual_probs = dispatcher.dispatch_preprocess2(hidden, 2, token_probs, token_indices)

        assert actual_hidden.data_ptr() == hidden.data_ptr()
        assert hybrid_ep_manager.routing_map.shape == (2, 8)
        assert hybrid_ep_manager.routing_map.sum() == 4
        assert actual_probs.shape == (2, 8)

    def test_topk_1(self, hybrid_ep_manager):
        """Each token routed to exactly one expert."""
        indices = torch.tensor([[2], [7]])
        probs = torch.tensor([[1.0], [1.0]])

        routing_map, multihot_probs = hybrid_ep_manager._indices_to_multihot(indices, probs)

        assert routing_map.sum() == 2
        assert routing_map[0, 2] and routing_map[1, 7]

    def test_all_minus_one(self, hybrid_ep_manager):
        """All indices are -1 (no valid routing)."""
        indices = torch.tensor([[-1, -1], [-1, -1]])
        probs = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        routing_map, multihot_probs = hybrid_ep_manager._indices_to_multihot(indices, probs)

        assert routing_map.sum() == 0
        assert multihot_probs.sum() == 0

    def test_partial_minus_one(self, hybrid_ep_manager):
        """Some indices are -1 (partial routing)."""
        indices = torch.tensor([[3, -1], [-1, 6]])
        probs = torch.tensor([[0.8, 0.0], [0.0, 0.5]])

        routing_map, multihot_probs = hybrid_ep_manager._indices_to_multihot(indices, probs)

        assert routing_map.sum() == 2
        assert routing_map[0, 3] and routing_map[1, 6]
        assert multihot_probs[0, 3] == pytest.approx(0.8)
        assert multihot_probs[1, 6] == pytest.approx(0.5)

    def test_single_token(self, hybrid_ep_manager):
        """Single token with multiple expert assignments."""
        indices = torch.tensor([[0, 7]])
        probs = torch.tensor([[0.5, 0.5]])

        routing_map, multihot_probs = hybrid_ep_manager._indices_to_multihot(indices, probs)

        assert routing_map.shape == (1, 8)
        assert routing_map.sum() == 2
        assert routing_map[0, 0] and routing_map[0, 7]


@pytest.mark.parametrize("enabled", [False, True])
def test_token_unpermutation_applies_async_setting_to_deepep_combine(enabled):
    dispatcher = object.__new__(MoEFlexTokenDispatcher)
    manager = object.__new__(_DeepepManager)
    manager.get_restored_hidden_states_by_experts = Mock(side_effect=lambda tensor: tensor)
    manager.combine = Mock(side_effect=lambda tensor, async_finish, allocate_on_comm_stream: tensor)
    dispatcher._comm_manager = manager
    dispatcher.config = SimpleNamespace(moe_deepep_async_dispatch=enabled)
    dispatcher.hidden_shape = (2, 4)
    hidden_states = torch.randn(2, 4)

    actual = dispatcher.token_unpermutation(hidden_states)

    torch.testing.assert_close(actual, hidden_states)
    manager.combine.assert_called_once_with(hidden_states, enabled, enabled)
