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

    def test_scoped_processor_matches_existing_conversion(self, hybrid_ep_manager):
        indices = torch.tensor([[0, 3], [1, -1]])
        probs = torch.tensor([[0.6, 0.4], [0.7, 0.0]])
        processor = _HybridEPMetadataProcessor(num_experts=8, permute_fusion=False)

        expected = hybrid_ep_manager._indices_to_multihot(indices, probs)
        actual = processor(indices, probs)

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

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


class TestHybridEPTokenSlots:
    """Real rows survive dispatch; common slots need no activation padding."""

    def _run(self, manager, monkeypatch, num_tokens, group_max):
        import nemo_automodel.components.moe.megatron.token_dispatcher as td

        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
        reduce = Mock(side_effect=lambda tensor, **kwargs: tensor.fill_(group_max))
        monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
        dispatch = Mock(side_effect=lambda **kw: (kw["x"], kw["probs"], None, kw["routing_map"].sum(0), "handle"))
        monkeypatch.setattr(td, "hybrid_ep_dispatch", dispatch)
        monkeypatch.setattr(td, "hybrid_ep_combine", lambda x, **kwargs: x)
        hidden = torch.randn(num_tokens, 4)
        manager.routing_map = torch.ones(num_tokens, 8, dtype=torch.bool)
        manager.token_probs = torch.full((num_tokens, 8), 0.125)
        combined = manager.combine(manager.dispatch(hidden))
        assert combined is hidden
        assert dispatch.call_args.kwargs["x"] is hidden
        assert dispatch.call_args.kwargs["routing_map"].shape == (num_tokens, 8)
        assert dispatch.call_args.kwargs["probs"].shape == (num_tokens, 8)
        return dispatch.call_args.kwargs["num_of_tokens_per_rank"], reduce

    @pytest.mark.parametrize("num_tokens,group_max", [(3, 5), (6, 6), (4, 4), (0, 5), (0, 0)])
    def test_dynamic_counts_negotiate_without_padding(self, hybrid_ep_manager, monkeypatch, num_tokens, group_max):
        slots, reduce = self._run(hybrid_ep_manager, monkeypatch, num_tokens, group_max)
        assert slots == group_max
        reduce.assert_called_once()

    def test_capacity_handles_changing_local_counts_without_collectives(self, hybrid_ep_manager, monkeypatch):
        from nemo_automodel.components.moe.megatron.token_dispatcher import hybrid_ep_token_capacity

        with hybrid_ep_token_capacity(32):
            for count in (8, 3, 0, 17, 32):
                slots, reduce = self._run(hybrid_ep_manager, monkeypatch, count, 99)
                assert slots == 32
                reduce.assert_not_called()
        slots, reduce = self._run(hybrid_ep_manager, monkeypatch, 3, 5)
        assert slots == 5
        reduce.assert_called_once()

    def test_overflow_raises_without_rank_conditional_collective(self, hybrid_ep_manager, monkeypatch):
        from nemo_automodel.components.moe.megatron.token_dispatcher import hybrid_ep_token_capacity

        with hybrid_ep_token_capacity(4), pytest.raises(ValueError, match="exceeding proven capacity"):
            self._run(hybrid_ep_manager, monkeypatch, 5, 5)
        torch.distributed.all_reduce.assert_not_called()
        slots, reduce = self._run(hybrid_ep_manager, monkeypatch, 5, 6)
        assert slots == 6
        reduce.assert_called_once()

    def test_nested_capacity_restores_outer_scope(self, hybrid_ep_manager, monkeypatch):
        from nemo_automodel.components.moe.megatron.token_dispatcher import hybrid_ep_token_capacity

        with hybrid_ep_token_capacity(32):
            with hybrid_ep_token_capacity(16):
                assert self._run(hybrid_ep_manager, monkeypatch, 3, 99)[0] == 16
            assert self._run(hybrid_ep_manager, monkeypatch, 3, 99)[0] == 32

    def test_capacity_is_visible_on_autograd_worker_thread(self, hybrid_ep_manager, monkeypatch):
        from concurrent.futures import ThreadPoolExecutor

        from nemo_automodel.components.moe.megatron.token_dispatcher import hybrid_ep_token_capacity

        with hybrid_ep_token_capacity(32), ThreadPoolExecutor(max_workers=1) as executor:
            slots, reduce = executor.submit(self._run, hybrid_ep_manager, monkeypatch, 3, 99).result()
            assert slots == 32
            reduce.assert_not_called()
