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

from unittest.mock import Mock, patch

import pytest
import torch

from nemo_automodel.components.moe.megatron.token_dispatcher import _HybridEPManager


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


class TestHybridEPPartitionedCombine:
    def test_reuses_handle_and_clears_manager_state(self, hybrid_ep_manager):
        hidden = torch.arange(2 * 3 * 4, dtype=torch.bfloat16).view(2, 3, 4)
        handle = object()
        hybrid_ep_manager.handle = handle
        hybrid_ep_manager.num_permuted_tokens = torch.tensor(6)
        combine = Mock(side_effect=lambda *, x, **kwargs: x + 1)

        with patch(
            "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_combine",
            combine,
        ):
            output = hybrid_ep_manager.combine_partitions(hidden)

        torch.testing.assert_close(output, hidden + 1)
        assert combine.call_count == 3
        assert all(call.kwargs["handle"] is handle for call in combine.call_args_list)
        assert all(call.kwargs["x"].is_contiguous() for call in combine.call_args_list)
        assert hybrid_ep_manager.handle is None
        assert hybrid_ep_manager.num_permuted_tokens is None

    def test_rejects_non_partitioned_input(self, hybrid_ep_manager):
        with pytest.raises(ValueError, match="tokens, partitions, hidden"):
            hybrid_ep_manager.combine_partitions(torch.zeros(2, 4))


class TestHybridEPVariableTokenCounts:
    def test_dispatch_pads_private_inputs_to_group_maximum(self, hybrid_ep_manager):
        indices = torch.tensor([[0, 3], [1, 5], [2, 7]])
        probs = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.8, 0.2]])
        hybrid_ep_manager.setup_metadata_from_indices(indices, probs)
        hidden = torch.arange(12, dtype=torch.bfloat16).view(3, 4)
        captured = {}

        def _all_reduce_max(length, *, op, group):
            assert op == torch.distributed.ReduceOp.MAX
            assert group is None
            length.fill_(5)

        def _dispatch(**kwargs):
            captured.update(kwargs)
            return kwargs["x"], kwargs["probs"], None, torch.tensor([1, 2]), object()

        with (
            patch.object(torch.distributed, "all_reduce", _all_reduce_max),
            patch(
                "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
                _dispatch,
            ),
        ):
            output = hybrid_ep_manager.dispatch(hidden)

        assert output.shape == (5, 4)
        torch.testing.assert_close(output[:3], hidden)
        torch.testing.assert_close(output[3:], torch.zeros(2, 4, dtype=torch.bfloat16))
        assert captured["routing_map"].shape == (5, 8)
        assert not captured["routing_map"][3:].any()
        torch.testing.assert_close(captured["probs"][3:], torch.zeros(2, 8))
        assert hybrid_ep_manager.num_unpadded_tokens == 3

    def test_combine_removes_group_padding(self, hybrid_ep_manager):
        hybrid_ep_manager.handle = object()
        hybrid_ep_manager.num_permuted_tokens = torch.tensor(6)
        hybrid_ep_manager.num_unpadded_tokens = 3
        hidden = torch.arange(20, dtype=torch.bfloat16).view(5, 4)

        with patch(
            "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_combine",
            side_effect=lambda *, x, **kwargs: x + 1,
        ):
            output = hybrid_ep_manager.combine(hidden)

        torch.testing.assert_close(output, hidden[:3] + 1)
        assert hybrid_ep_manager.handle is None
        assert hybrid_ep_manager.num_permuted_tokens is None
        assert hybrid_ep_manager.num_unpadded_tokens is None

    def test_partitioned_combine_removes_group_padding(self, hybrid_ep_manager):
        hybrid_ep_manager.handle = object()
        hybrid_ep_manager.num_permuted_tokens = torch.tensor(6)
        hybrid_ep_manager.num_unpadded_tokens = 3
        hidden = torch.arange(5 * 2 * 4, dtype=torch.bfloat16).view(5, 2, 4)

        with patch(
            "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_combine",
            side_effect=lambda *, x, **kwargs: x + 1,
        ):
            output = hybrid_ep_manager.combine_partitions(hidden)

        torch.testing.assert_close(output, hidden[:3] + 1)
        assert hybrid_ep_manager.handle is None
        assert hybrid_ep_manager.num_permuted_tokens is None
        assert hybrid_ep_manager.num_unpadded_tokens is None
