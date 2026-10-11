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

"""Host-extent contracts for the existing static balanced HybridEP path."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from nemo_automodel.components.moe.megatron import fused_a2a, token_dispatcher
from nemo_automodel.components.moe.parallelizer import _replay_hybridep_dispatch_on_recompute


def _manager(monkeypatch: pytest.MonkeyPatch, *, static: bool = True) -> token_dispatcher._HybridEPManager:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    return token_dispatcher._HybridEPManager(
        group=None, num_local_experts=2, num_experts=2, router_topk=1, benchmark_static_routing=static
    )


def test_static_dispatch_and_combine_never_read_count_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _manager(monkeypatch)
    counts = Mock(spec=torch.Tensor)
    counts.sum.side_effect = AssertionError("Static extent must come from the output shape")
    seen_dispatch: list[int | None] = []
    seen_combine: list[int] = []

    def dispatch(
        *, x: torch.Tensor, probs: torch.Tensor, num_permuted_tokens: int | None, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor, None, Mock, str]:
        seen_dispatch.append(num_permuted_tokens)
        return x, probs, None, counts, "layout"

    def combine(*, x: torch.Tensor, num_permuted_tokens: int, **kwargs: object) -> torch.Tensor:
        seen_combine.append(num_permuted_tokens)
        return x

    monkeypatch.setattr(token_dispatcher, "hybrid_ep_dispatch", dispatch)
    monkeypatch.setattr(token_dispatcher, "hybrid_ep_combine", combine)
    for _ in range(3):
        manager.routing_map = torch.ones(4, 2, dtype=torch.bool)
        manager.token_probs = torch.ones(4, 2)
        manager.combine(manager.dispatch(torch.randn(4, 3)))
    assert seen_dispatch == [None, 4, 4]
    assert seen_combine == [4, 4, 4]
    assert all(type(extent) is int for extent in seen_combine)
    counts.sum.assert_not_called()


def test_recorder_host_extent_does_not_reduce_counts() -> None:
    recorder = fused_a2a.HybridEPDispatchReplayRecorder()
    counts = Mock(spec=torch.Tensor)
    counts.sum.side_effect = AssertionError("Host extent must bypass GPU count reads")
    recorder.record("layout", counts, 4)
    recorder.finalize()
    assert recorder.take() == ["layout", counts, 4]
    counts.sum.assert_not_called()


@pytest.mark.parametrize("extent", [None, torch.tensor(99)])
def test_dynamic_recorder_still_derives_actual_extent(extent: torch.Tensor | None) -> None:
    recorder = fused_a2a.HybridEPDispatchReplayRecorder()
    if extent is None:
        recorder.record("layout", torch.tensor([2, 3]))
    else:
        recorder.record("layout", torch.tensor([2, 3]), extent)
    recorder.finalize()
    assert recorder.take()[2] == 5


class _PermutationBuffer:
    """CPU communication stand-in; only actual distributed tests certify NCCL."""

    def __init__(self) -> None:
        self.order = torch.tensor([0, 2, 1, 3])
        self.expert_ids = torch.tensor([0, 1, 0, 1])
        self.extents: list[int | None] = []
        self.counts = torch.tensor([2, 2])

    def dispatch_with_permute(
        self,
        *,
        hidden: torch.Tensor,
        probs: torch.Tensor | None = None,
        num_permuted_tokens: int | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None, torch.Tensor, str]:
        """Permute [4, 3] hidden rows and optional [4, 2] routing weights.

        Returns:
            Hidden [4, 3], selected weights [4] or None, no scaling,
            expert counts [2], and an opaque layout handle.
        """
        extent = num_permuted_tokens
        self.extents.append(extent)
        assert extent is None or (type(extent) is int and extent == 4)
        selected_probs = None if probs is None else probs[torch.arange(4), self.expert_ids][self.order]
        return hidden[self.order], selected_probs, None, self.counts, "layout"

    def combine_with_unpermute(
        self,
        *,
        hidden: torch.Tensor,
        probs: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Restore [4, 3] hidden rows and optional [4] probability gradients.

        Returns:
            Hidden [4, 3] and dense probability gradients [4, 2] or None.
        """
        restored_probs = None
        if probs is not None:
            restored_probs = torch.zeros(4, 2, dtype=probs.dtype)
            restored_probs[torch.arange(4), self.expert_ids] = probs[self.order]
        return hidden[self.order], restored_probs


@pytest.mark.parametrize("ac", ["none", "checkpoint", "selective"])
def test_static_manager_backward_and_recompute_match_independent_math(monkeypatch: pytest.MonkeyPatch, ac: str) -> None:
    """Check two updates, no capacity rows, exact gradients, and host-int replay."""
    manager = _manager(monkeypatch)
    buffer = _PermutationBuffer()
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", buffer)
    monkeypatch.setattr(token_dispatcher, "hybrid_ep_dispatch", fused_a2a.hybrid_ep_dispatch)
    monkeypatch.setattr(token_dispatcher, "hybrid_ep_combine", fused_a2a.hybrid_ep_combine)
    weight = torch.tensor([1.25, -0.75], requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()

    def body(hidden: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Apply the two experts to [4, 3] hidden and [4, 2] routing weights."""
        routing = torch.nn.functional.one_hot(buffer.expert_ids, 2).bool()
        manager.setup_metadata(routing, probs)
        dispatched = manager.dispatch(hidden)
        scales = weight.repeat_interleave(2)
        output = dispatched.sin() * scales[:, None] * manager.dispatched_probs[:, None]
        return manager.combine(output)

    for step in range(2):
        hidden = (torch.arange(12).reshape(4, 3).float() / 10 + step).requires_grad_()
        probs = torch.full((4, 2), 0.5, requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_()
        reference_probs = probs.detach().clone().requires_grad_()
        if ac == "none":
            output = body(hidden, probs)
        else:
            if ac == "selective":

                def policy(ctx, op, *args, **kwargs):
                    return (
                        CheckpointPolicy.MUST_SAVE
                        if op in (torch.ops.aten.sum.default, torch.ops.aten._local_scalar_dense.default)
                        else CheckpointPolicy.PREFER_RECOMPUTE
                    )

                context = lambda: create_selective_checkpoint_contexts(policy)
            else:
                context = lambda: (nullcontext(), nullcontext())
            output = checkpoint(
                body,
                hidden,
                probs,
                use_reentrant=False,
                context_fn=_replay_hybridep_dispatch_on_recompute(context),
            )
        selected_probs = reference_probs[torch.arange(4), buffer.expert_ids]
        reference = reference_hidden.sin() * reference_weight[buffer.expert_ids, None] * selected_probs[:, None]
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        output.square().sum().backward()
        reference.square().sum().backward()
        for actual, expected in (
            (hidden.grad, reference_hidden.grad),
            (probs.grad, reference_probs.grad),
            (weight.grad, reference_weight.grad),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with torch.no_grad():
            weight.add_(weight.grad, alpha=-0.01)
            reference_weight.add_(reference_weight.grad, alpha=-0.01)
        weight.grad = reference_weight.grad = None
    assert buffer.extents[0] is None
    assert all(type(extent) is int and extent == 4 for extent in buffer.extents[1:])
