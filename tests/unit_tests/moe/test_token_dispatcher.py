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

from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
    TokenDispatcherConfig,
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


def test_hybridep_manager_accepts_expert_padding_multiple():
    """HybridEP keeps the alignment requested by the TE grouped-MLP backend."""
    with patch(
        "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
        new=lambda *args, **kwargs: None,
    ):
        manager = _HybridEPManager(
            group=None,
            num_local_experts=2,
            num_experts=8,
            router_topk=2,
            moe_router_expert_pad_multiple=256,
        )

    assert manager.pad_multiple == 256


def test_hybridep_manager_forwards_preprocessing_sms_with_dynamic_receive_sizing():
    """Preprocessing tuning must not introduce a cached or static receive-token count."""
    captured_kwargs = {}

    def fake_hybrid_ep_dispatch(**kwargs):
        captured_kwargs.update(kwargs)
        tokens_per_expert = torch.tensor([1, 3])
        return kwargs["x"], kwargs["probs"], None, tokens_per_expert, object()

    with patch(
        "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
        new=fake_hybrid_ep_dispatch,
    ):
        manager = _HybridEPManager(
            group=None,
            num_local_experts=2,
            num_experts=8,
            router_topk=2,
            moe_hybridep_num_sms_preprocessing=32,
        )
        manager.setup_metadata(
            torch.tensor([[True, False, True, False, False, False, False, False]]),
            torch.tensor([[0.6, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        )
        manager.dispatch(torch.randn(1, 8))

    assert captured_kwargs["num_sms_preprocessing_api"] == 32
    assert captured_kwargs["num_permuted_tokens"] is None
    assert manager.num_permuted_tokens == 4


def test_hybridep_shared_manager_identity_includes_preprocessing_sms_for_ep():
    """EP layers share managers only when their HybridEP preprocessing configuration matches."""
    saved_alias = MoEFlexTokenDispatcher.shared_hybridep_manager
    saved_cache = MoEFlexTokenDispatcher._shared_hybridep_managers
    MoEFlexTokenDispatcher.shared_hybridep_manager = None
    MoEFlexTokenDispatcher._shared_hybridep_managers = {}
    group = Mock()
    group.size.return_value = 2
    config_kwargs = {
        "moe_flex_dispatcher_backend": "hybridep",
        "num_moe_experts": 8,
        "moe_router_topk": 2,
        "moe_share_token_dispatcher": True,
    }

    try:
        with patch(
            "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
            new=lambda *args, **kwargs: None,
        ):
            default_dispatcher = MoEFlexTokenDispatcher(
                num_local_experts=4,
                local_expert_indices=list(range(4)),
                config=TokenDispatcherConfig(**config_kwargs),
                ep_group=group,
            )
            tuned_dispatcher = MoEFlexTokenDispatcher(
                num_local_experts=4,
                local_expert_indices=list(range(4)),
                config=TokenDispatcherConfig(**config_kwargs, moe_hybridep_num_sms_preprocessing=32),
                ep_group=group,
            )
            tuned_dispatcher_again = MoEFlexTokenDispatcher(
                num_local_experts=4,
                local_expert_indices=list(range(4)),
                config=TokenDispatcherConfig(**config_kwargs, moe_hybridep_num_sms_preprocessing=32),
                ep_group=group,
            )
    finally:
        MoEFlexTokenDispatcher.shared_hybridep_manager = saved_alias
        MoEFlexTokenDispatcher._shared_hybridep_managers = saved_cache

    assert default_dispatcher._comm_manager is not tuned_dispatcher._comm_manager
    assert tuned_dispatcher._comm_manager is tuned_dispatcher_again._comm_manager


def test_hybridep_metadata_processor_matches_manager_and_preserves_prob_grads():
    """The graphable preprocess boundary is pure and differentiable in probabilities."""
    processor = _HybridEPMetadataProcessor(num_experts=8, permute_fusion=False)
    indices = torch.tensor([[0, 3], [1, 5]])
    probs = torch.tensor([[0.6, 0.4], [0.7, 0.3]], requires_grad=True)

    routing_map, multihot_probs = processor(indices, probs)

    assert routing_map.shape == (2, 8)
    assert routing_map.sum() == 4
    torch.testing.assert_close(
        multihot_probs,
        torch.tensor([[0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0], [0.0, 0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]]),
    )

    multihot_probs.sum().backward()
    torch.testing.assert_close(probs.grad, torch.ones_like(probs))


def test_hybridep_metadata_processor_accepts_te_dense_router_output():
    """TE's dense map/probs pass straight to HybridEP without losing gradients."""
    processor = _HybridEPMetadataProcessor(num_experts=8, permute_fusion=True)
    routing_map = torch.tensor(
        [
            [True, False, False, True, False, False, False, False],
            [False, True, False, False, False, True, False, False],
        ]
    )
    probs = torch.tensor(
        [
            [0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
        ],
        requires_grad=True,
    )

    processed_map, processed_probs = processor(routing_map, probs)

    assert processed_map is routing_map
    assert processed_probs is probs
    processed_probs.sum().backward()
    torch.testing.assert_close(probs.grad, torch.ones_like(probs))


def test_hybridep_metadata_processor_rejects_invalid_dense_shape():
    processor = _HybridEPMetadataProcessor(num_experts=8, permute_fusion=True)
    with pytest.raises(ValueError, match="Dense HybridEP routing metadata must have shape"):
        processor(torch.ones(2, 7, dtype=torch.bool), torch.ones(2, 7))
