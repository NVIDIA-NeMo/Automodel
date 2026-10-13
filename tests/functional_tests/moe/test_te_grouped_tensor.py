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

"""Numerical coverage of the native TE grouped GEMM integration, on one GPU."""

from dataclasses import dataclass

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import set_is_first_microbatch
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


def _experts(*, native: bool, counts: list[int]) -> GroupedExpertsTE:
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
        expert_bias=True,
        dtype=torch.bfloat16,
    )
    backend = BackendConfig(experts="te", dispatcher="deepep", te_grouped_tensor=native)
    module = GroupedExpertsTE(config, backend=backend)
    module.gate_up_linear.reset_parameters()
    module.down_linear.reset_parameters()
    module.ep_size = 1
    module.token_dispatcher = _GroupedInputDispatcher(counts)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_native_te_grouped_tensor_numerics(fp8: bool, checkpointed: bool) -> None:
    """Preserve outputs, all gradients, updates, padding and empty-expert behavior."""
    te = pytest.importorskip("transformer_engine.pytorch", reason="Optional TE backend is required")
    from transformer_engine.common.recipe import Float8CurrentScaling

    torch.manual_seed(13)
    # Imbalanced counts exercise FP8 padding and an empty expert in both GEMMs.
    counts = [17, 0, 63, 32]
    reference = _experts(native=False, counts=counts)
    candidate = _experts(native=True, counts=counts)
    with torch.no_grad():
        for reference_parameter, parameter in zip(reference.parameters(), candidate.parameters(), strict=True):
            parameter.copy_(reference_parameter)
    optimizers = [torch.optim.SGD(module.parameters(), lr=0.001) for module in (reference, candidate)]
    mask = torch.ones(sum(counts), device="cuda", dtype=torch.bool)
    indices = torch.cat([torch.full((count, 1), expert) for expert, count in enumerate(counts)]).cuda()
    try:
        for _step in range(2):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            for microbatch in range(3):
                set_is_first_microbatch(microbatch == 0)
                x = torch.randn(sum(counts), 256, dtype=torch.bfloat16, device="cuda")
                probs = torch.rand(sum(counts), 1, dtype=torch.bfloat16, device="cuda")
                upstream = torch.randn_like(x)
                outputs, input_grads, probability_grads = [], [], []
                for module in (reference, candidate):
                    hidden = x.detach().clone().requires_grad_()
                    probability = probs.detach().clone().requires_grad_()

                    def run(hidden: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
                        """Evaluate the same precision recipe during checkpoint recomputation.

                        Args:
                            hidden: Tensor of shape [tokens, hidden], grouped by expert.
                            probability: Tensor of shape [tokens, 1] with routing probabilities.

                        Returns:
                            Tensor of shape [tokens, hidden].
                        """
                        with te.autocast(enabled=fp8, recipe=Float8CurrentScaling()):
                            return module(hidden, mask, probability, indices)

                    if checkpointed:
                        output = checkpoint(run, hidden, probability, use_reentrant=False)
                    else:
                        output = run(hidden, probability)
                    output.backward(upstream)
                    outputs.append(output.detach())
                    input_grads.append(hidden.grad)
                    probability_grads.append(probability.grad)
                for expected, actual in (outputs, input_grads, probability_grads):
                    # BF16 grouped GEMM reductions may choose different tilings; FP8
                    # operands are quantized independently by each TE module.
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
            for (name, expected), (_, actual) in zip(
                reference.named_parameters(), candidate.named_parameters(), strict=True
            ):
                assert expected.grad is not None and actual.grad is not None, name
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0.03, atol=0.003)
                if name.endswith(("weight1", "bias1")):
                    assert torch.count_nonzero(actual.grad).item() == 0, name
            for optimizer in optimizers:
                optimizer.step()
        # The integration keeps the existing per-expert parameter/checkpoint layout.
        for (name, expected), (actual_name, actual) in zip(
            reference.named_parameters(), candidate.named_parameters(), strict=True
        ):
            assert name == actual_name
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    finally:
        set_is_first_microbatch(None)
