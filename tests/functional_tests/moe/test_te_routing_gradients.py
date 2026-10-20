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

"""TE dispatcher routing-gradient parity with legacy grouped GEMMs, on one GPU."""

from dataclasses import dataclass

import pytest
import torch
from torch.autograd.function import FunctionCtx
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsTE


@dataclass
class _GroupedInputDispatcher:
    """Keep already expert-grouped inputs in place to isolate the expert computation."""

    counts: list[int]

    def token_permutation2(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        """Return inputs without communication or permutation.

        Args:
            hidden_states: Tensor of shape [tokens, hidden], grouped by expert.
            num_local_tokens: Number of input rows.
            token_probs: Tensor of shape [tokens, 1] with one route per token.
            token_indices: Integer tensor of shape [tokens, 1] with the route's expert ID.

        Returns:
            Hidden states [tokens, hidden], row counts per expert, and probabilities [tokens].
        """
        assert num_local_tokens == sum(self.counts)
        return hidden_states, self.counts, token_probs[:, 0]

    def token_unpermutation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return expert-grouped outputs of shape [tokens, hidden] unchanged.

        Args:
            hidden_states: Tensor of shape [tokens, hidden], grouped by expert.

        Returns:
            The same tensor, aliasing the input.
        """
        return hidden_states


def _experts(*, counts: list[int], bias: bool = True) -> GroupedExpertsTE:
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=256,
        inter_dim=512,
        moe_inter_dim=512,
        norm_topk_prob=False,
        expert_bias=bias,
        dtype=torch.bfloat16,
    )
    backend = BackendConfig(experts="te", dispatcher="deepep")
    module = GroupedExpertsTE(config, backend=backend)
    module.gate_up_linear.reset_parameters()
    module.down_linear.reset_parameters()
    module.ep_size = 1
    module.token_dispatcher = _GroupedInputDispatcher(counts)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("train_router", [False, True])
def test_te_dispatch_preserves_router_gradients(fp8: bool, checkpointed: bool, train_router: bool) -> None:
    """Coupled dispatcher outputs preserve frozen and trainable router semantics.

    Args:
        fp8: Whether the expert GEMMs use FP8 current scaling.
        checkpointed: Whether to recompute the expert forward during backward.
        train_router: Whether the original routing probabilities need gradients.
    """
    te = pytest.importorskip("transformer_engine.pytorch", reason="Optional TE backend is required")
    from transformer_engine.common.recipe import Float8CurrentScaling

    class CoupledDispatch(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: FunctionCtx, hidden: torch.Tensor, probability: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Clone grouped hidden states [tokens, hidden] and probabilities [tokens].

            Args:
                ctx: Autograd context; no values need saving for identity derivatives.
                hidden: Already grouped hidden states of shape [tokens, hidden].
                probability: One probability per grouped token, shape [tokens].

            Returns:
                Independent output tensors with the same shapes and a shared grad_fn.
            """
            return hidden.clone(), probability.clone()

        @staticmethod
        def backward(
            ctx: FunctionCtx, hidden_grad: torch.Tensor, probability_grad: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return identity derivatives, preserving PyTorch's materialized zero gradients.

            Args:
                ctx: Autograd context, unused by the identity derivative.
                hidden_grad: Gradient of shape [tokens, hidden].
                probability_grad: Gradient of shape [tokens], zero when unused.

            Returns:
                Input gradients with the same shapes as the two forward inputs.
            """
            return hidden_grad, probability_grad

    class CoupledDispatcher(_GroupedInputDispatcher):
        def token_permutation2(
            self,
            hidden_states: torch.Tensor,
            num_local_tokens: int,
            token_probs: torch.Tensor,
            token_indices: torch.Tensor,
        ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
            """Attach one autograd node to both dispatcher outputs.

            Args:
                hidden_states: Already grouped hidden states [tokens, hidden].
                num_local_tokens: Number of rows in the hidden states.
                token_probs: Routing probabilities [tokens, 1].
                token_indices: Expert IDs [tokens, 1].

            Returns:
                Grouped hidden states [tokens, hidden], expert counts, probabilities [tokens].
            """
            hidden, counts, probability = super().token_permutation2(
                hidden_states, num_local_tokens, token_probs, token_indices
            )
            hidden, probability = CoupledDispatch.apply(hidden, probability)
            # A multi-output dispatcher attaches a grad_fn to frozen probabilities.
            assert probability.requires_grad
            return hidden, counts, probability

    torch.manual_seed(17)
    counts = [17, 0, 63, 32]
    reference = _experts(counts=counts, bias=False)
    candidate = _experts(counts=counts, bias=False)
    candidate.token_dispatcher = CoupledDispatcher(counts)
    with torch.no_grad():
        for expected, actual in zip(reference.parameters(), candidate.parameters(), strict=True):
            actual.copy_(expected)
    activation = candidate.expert_activation

    def checked_activation(hidden: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
        """Check routing gradients at the activation boundary, including FP8 padding.

        Args:
            hidden: Gate/up projection of shape [padded_tokens, 2 * intermediate].
            probability: Routing probabilities of shape [padded_tokens, 1].

        Returns:
            Weighted activation of shape [padded_tokens, intermediate].
        """
        assert probability.requires_grad == train_router
        return activation(hidden, probability)

    candidate.expert_activation = checked_activation
    hidden = torch.randn(sum(counts), 256, device="cuda", dtype=torch.bfloat16)
    probabilities = torch.rand(sum(counts), 1, device="cuda", dtype=torch.float32)
    upstream = torch.randn_like(hidden)
    mask = torch.ones(sum(counts), device="cuda", dtype=torch.bool)
    indices = torch.cat([torch.full((count, 1), expert) for expert, count in enumerate(counts)]).cuda()
    values = []
    for module in (reference, candidate):
        x = hidden.detach().clone().requires_grad_()
        probs = probabilities.detach().clone().requires_grad_(train_router)

        def run(x: torch.Tensor, probs: torch.Tensor, *, module: GroupedExpertsTE = module) -> torch.Tensor:
            """Evaluate one expert module with the same precision during recomputation.

            Args:
                x: Grouped hidden states of shape [tokens, hidden].
                probs: Routing probabilities of shape [tokens, 1].
                module: Expert implementation pinned for checkpoint recomputation.

            Returns:
                Expert output of shape [tokens, hidden].
            """
            with te.autocast(enabled=fp8, recipe=Float8CurrentScaling()):
                return module(x, mask, probs, indices)

        output = checkpoint(run, x, probs, use_reentrant=False) if checkpointed else run(x, probs)
        output.backward(upstream)
        values.append((output, x.grad, probs.grad))
    for expected, actual in zip(*values, strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected, actual in zip(reference.parameters(), candidate.parameters(), strict=True):
        assert expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
