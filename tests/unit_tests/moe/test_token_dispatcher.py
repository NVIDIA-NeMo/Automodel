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


def test_hybridep_static_rank_budget_is_aligned_without_freezing_expert_counts():
    """Static rank sizing keeps each dispatch's live device expert counts."""
    captured_kwargs = []
    returned_counts = [torch.tensor([1, 4]), torch.tensor([3, 2])]

    def fake_hybrid_ep_dispatch(**kwargs):
        call_index = len(captured_kwargs)
        captured_kwargs.append(kwargs)
        handle = (object(), torch.tensor(0))
        return kwargs["x"], kwargs["probs"], None, returned_counts[call_index], handle

    with patch(
        "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
        new=fake_hybrid_ep_dispatch,
    ):
        manager = _HybridEPManager(
            group=None,
            num_local_experts=2,
            num_experts=8,
            router_topk=2,
            moe_router_expert_pad_multiple=8,
            moe_expert_rank_capacity_factor=1.1,
        )
        manager.setup_metadata(torch.zeros(5, 8, dtype=torch.bool), torch.zeros(5, 8))

        manager.dispatch(torch.randn(5, 8))
        first_counts = manager.get_number_of_tokens_per_expert()
        manager.dispatch(torch.randn(5, 8))
        second_counts = manager.get_number_of_tokens_per_expert()

    # int(5 tokens * top-2 * 1.1) == 11, aligned up to 8.
    assert [call["num_permuted_tokens"] for call in captured_kwargs] == [16, 16]
    assert first_counts is returned_counts[0]
    assert second_counts is returned_counts[1]
    assert not torch.equal(first_counts, second_counts)


def test_hybridep_static_rank_budget_accumulates_and_resets_device_overflow():
    """Overflow remains a tensor and accumulates until the iteration resets it."""
    overflow_flags = iter((torch.tensor(0), torch.tensor(1)))

    def fake_hybrid_ep_dispatch(**kwargs):
        handle = (object(), next(overflow_flags))
        return kwargs["x"], kwargs["probs"], None, torch.tensor([2, 2]), handle

    with patch(
        "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
        new=fake_hybrid_ep_dispatch,
    ):
        manager = _HybridEPManager(
            group=None,
            num_local_experts=2,
            num_experts=8,
            router_topk=2,
            moe_expert_rank_capacity_factor=1.25,
        )
        manager.setup_metadata(torch.zeros(2, 8, dtype=torch.bool), torch.zeros(2, 8))

        manager.dispatch(torch.randn(2, 8))
        over_budget = manager.check_over_budget()
        assert isinstance(over_budget, torch.Tensor)
        assert not over_budget.item()

        manager.dispatch(torch.randn(2, 8))
        assert manager.check_over_budget().item()
        assert manager.check_over_budget() is over_budget

        manager.reset_over_budget()
        assert not manager.check_over_budget().item()


def test_hybridep_disabling_static_rank_budget_restores_exact_dynamic_path():
    """Fallback can restore HybridEP's original dropless dynamic receive sizing."""
    captured_sizes = []

    def fake_hybrid_ep_dispatch(**kwargs):
        captured_sizes.append(kwargs["num_permuted_tokens"])
        handle = (object(), torch.tensor(1)) if kwargs["num_permuted_tokens"] is not None else object()
        return kwargs["x"], kwargs["probs"], None, torch.tensor([2, 3]), handle

    with patch(
        "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
        new=fake_hybrid_ep_dispatch,
    ):
        manager = _HybridEPManager(
            group=None,
            num_local_experts=2,
            num_experts=8,
            router_topk=2,
            moe_router_expert_pad_multiple=8,
            moe_expert_rank_capacity_factor=1.25,
        )
        manager.setup_metadata(torch.zeros(4, 8, dtype=torch.bool), torch.zeros(4, 8))

        manager.dispatch(torch.randn(4, 8))
        assert manager.check_over_budget().item()

        manager.set_static_rank_budget(None)
        manager.dispatch(torch.randn(4, 8))

    assert captured_sizes == [16, None]
    assert manager.moe_expert_rank_capacity_factor is None
    assert manager.num_permuted_tokens == 5
    assert not manager.check_over_budget().item()


def test_hybridep_dispatch_trims_only_extra_probability_rows():
    """Static receive padding cannot leave probability rows misaligned with hidden rows."""

    def fake_hybrid_ep_dispatch(**kwargs):
        dispatched_hidden = torch.randn(4, kwargs["x"].shape[-1])
        dispatched_probs = torch.randn(6, kwargs["probs"].shape[-1])
        return dispatched_hidden, dispatched_probs, None, torch.tensor([2, 2]), object()

    with patch(
        "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
        new=fake_hybrid_ep_dispatch,
    ):
        manager = _HybridEPManager(
            group=None,
            num_local_experts=2,
            num_experts=8,
            router_topk=2,
        )
        manager.setup_metadata(torch.zeros(3, 8, dtype=torch.bool), torch.zeros(3, 8))
        dispatched_hidden = manager.dispatch(torch.randn(3, 8))

    assert dispatched_hidden.shape[0] == 4
    assert manager.dispatched_probs.shape[0] == dispatched_hidden.shape[0]


def test_hybridep_combine_retains_handle_captured_by_cuda_graph():
    """Opaque HybridEP resources remain alive for later CUDA graph replay."""
    handle = object()

    with (
        patch(
            "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
            new=lambda *args, **kwargs: None,
        ),
        patch(
            "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_combine",
            new=lambda **kwargs: kwargs["x"],
        ),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.is_current_stream_capturing", return_value=True),
    ):
        manager = _HybridEPManager(
            group=None,
            num_local_experts=2,
            num_experts=8,
            router_topk=2,
        )
        manager.handle = handle
        manager.num_permuted_tokens = torch.tensor(4)
        result = manager.combine(torch.randn(4, 8))

    assert result.shape == (4, 8)
    assert manager.handle is None
    assert manager._cuda_graph_handles == [handle]


def test_hybridep_reset_runtime_state_releases_completed_call_state():
    """Shape-mode transitions must not retain checkpoint-recompute handles or metadata."""
    with patch(
        "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
        new=lambda *args, **kwargs: None,
    ):
        manager = _HybridEPManager(
            group=None,
            num_local_experts=2,
            num_experts=8,
            router_topk=2,
        )
    manager.handle = object()
    manager.num_permuted_tokens = torch.tensor(4)
    manager.token_probs = torch.ones(2, 8)
    manager.routing_map = torch.ones(2, 8, dtype=torch.bool)
    manager.dispatched_probs = torch.ones(4, 8)
    manager.tokens_per_expert = torch.ones(2, dtype=torch.int32)

    manager.reset_runtime_state()

    assert manager.handle is None
    assert manager.num_permuted_tokens is None
    assert manager.token_probs is None
    assert manager.routing_map is None
    assert manager.dispatched_probs is None
    assert manager.tokens_per_expert is None


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
