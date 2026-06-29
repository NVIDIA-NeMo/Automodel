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

from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
    TokenDispatcherConfig,
    _HybridEPManager,
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


class _FakeGroup:
    def size(self):
        return 2


class _FakeDeepepManager:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def reset_deepep_dispatcher_state(monkeypatch):
    from nemo_automodel.components.moe.megatron import token_dispatcher

    v2_dispatch = object()
    v2_combine = object()
    monkeypatch.setattr(MoEFlexTokenDispatcher, "shared_deepep_manager", None)
    monkeypatch.setattr(MoEFlexTokenDispatcher, "shared_deepep_v2_manager", None)
    monkeypatch.setattr(token_dispatcher, "_DeepepManager", _FakeDeepepManager)
    monkeypatch.setattr(token_dispatcher, "deepep_v2_fused_dispatch", v2_dispatch)
    monkeypatch.setattr(token_dispatcher, "deepep_v2_fused_combine", v2_combine)
    monkeypatch.setattr(token_dispatcher, "set_deepep_num_sms", lambda _num_sms: None)
    monkeypatch.setattr(token_dispatcher, "set_deepep_v2_num_sms", lambda _num_sms: None)
    return v2_dispatch, v2_combine


def _build_dispatcher(backend: str) -> MoEFlexTokenDispatcher:
    config = TokenDispatcherConfig(
        moe_flex_dispatcher_backend=backend,
        moe_router_topk=2,
        num_moe_experts=8,
        moe_share_token_dispatcher=False,
    )
    return MoEFlexTokenDispatcher(
        num_local_experts=4,
        local_expert_indices=[0, 1, 2, 3],
        config=config,
        ep_group=_FakeGroup(),
    )


class TestMoEFlexTokenDispatcherDeepEPSelection:
    def test_deepep_uses_legacy_manager(self, reset_deepep_dispatcher_state):
        dispatcher = _build_dispatcher("deepep")

        assert isinstance(dispatcher._comm_manager, _FakeDeepepManager)
        assert "_dispatch_fn" not in dispatcher._comm_manager.kwargs

    def test_deepep_v2_uses_elastic_manager(self, reset_deepep_dispatcher_state):
        v2_dispatch, v2_combine = reset_deepep_dispatcher_state
        dispatcher = _build_dispatcher("deepep_v2")

        assert isinstance(dispatcher._comm_manager, _FakeDeepepManager)
        assert dispatcher._comm_manager.kwargs["_dispatch_fn"] is v2_dispatch
        assert dispatcher._comm_manager.kwargs["_combine_fn"] is v2_combine
