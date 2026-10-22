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


def _experts(
    *, counts: list[int], bias: bool = True, after_down: bool = False, dtype: torch.dtype = torch.bfloat16
) -> GroupedExpertsTE:
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
        apply_router_weight_after_down=after_down,
        dtype=dtype,
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


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return ||actual - expected|| / ||expected|| in fp32 for tensors of the same shape.

    Args:
        actual: Tensor of any shape.
        expected: Tensor of the same shape as ``actual``.

    Returns:
        Relative Frobenius error.
    """
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_te_router_weight_after_down_matches_reference(fp8: bool, bias: bool) -> None:
    """After-down routing scales the biased, post-processed expert output, like the torch experts.

    Args:
        fp8: Whether the expert GEMMs use FP8 current scaling, which pads each expert's rows.
        bias: Whether the experts carry gate/up and down biases.
    """
    te = pytest.importorskip("transformer_engine.pytorch", reason="Optional TE backend is required")
    from transformer_engine.common.recipe import Float8CurrentScaling

    torch.manual_seed(23)
    counts = [17, 0, 63, 32]
    module = _experts(counts=counts, bias=bias, after_down=True)
    if bias:
        with torch.no_grad():
            for parameter in (*module.gate_up_linear.parameters(), *module.down_linear.parameters()):
                if parameter.ndim == 1:
                    parameter.normal_(std=0.5)

    def post(values: torch.Tensor) -> torch.Tensor:
        """Normalize each row of [tokens, hidden] expert outputs in fp32, a nonlinear post-expert step.

        Args:
            values: Expert outputs of shape [tokens, hidden].

        Returns:
            Row-normalized outputs of shape [tokens, hidden] in ``values``' dtype.
        """
        return (values.float() * torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(values.dtype)

    hidden = torch.randn(sum(counts), 256, device="cuda", dtype=torch.bfloat16)
    probabilities = torch.rand(sum(counts), 1, device="cuda", dtype=torch.float32)
    upstream = torch.randn_like(hidden)
    mask = torch.ones(sum(counts), device="cuda", dtype=torch.bool)
    indices = torch.cat([torch.full((count, 1), expert) for expert, count in enumerate(counts)]).cuda()
    x = hidden.clone().requires_grad_()
    probs = probabilities.clone().requires_grad_()
    with te.autocast(enabled=fp8, recipe=Float8CurrentScaling()):
        output = module(x, mask, probs, indices, post_expert_output=post)
    output.backward(upstream)

    def reference(weight_router_after_down: bool) -> tuple[torch.Tensor, ...]:
        """Evaluate the experts with fp32 matmuls on copies of the module's weights.

        Args:
            weight_router_after_down: Scale the post-processed output (True) or the activation (False).

        Returns:
            Output [tokens, hidden], then the gradients of x, probs, and each expert's weights.
        """
        x_ref = hidden.float().requires_grad_()
        probs_ref = probabilities.clone().requires_grad_()
        weights = [
            [getattr(linear, f"{kind}{expert}").detach().float().requires_grad_() for kind in ("weight", "bias")]
            if bias
            else [getattr(linear, f"weight{expert}").detach().float().requires_grad_()]
            for linear in (module.gate_up_linear, module.down_linear)
            for expert in range(len(counts))
        ]
        outputs = []
        for expert, (rows, p) in enumerate(zip(x_ref.split(counts), probs_ref.split(counts), strict=True)):
            gate_up, down = weights[expert], weights[len(counts) + expert]
            projected = torch.nn.functional.linear(rows, *gate_up)
            gate, up = projected.chunk(2, -1)
            activated = torch.nn.functional.silu(gate) * up
            if weight_router_after_down:
                outputs.append(post(torch.nn.functional.linear(activated, *down)) * p)
            else:
                weighted_bias = down[1] * p if bias else 0.0
                outputs.append(post(torch.nn.functional.linear(activated * p, down[0]) + weighted_bias))
        expected = torch.cat(outputs)
        expected.backward(upstream.float())
        return expected, x_ref.grad, probs_ref.grad, *(w.grad for pair in weights for w in pair)

    actual_grads = [
        getattr(linear, f"{kind}{expert}").grad
        for linear in (module.gate_up_linear, module.down_linear)
        for expert in range(len(counts))
        for kind in (("weight", "bias") if bias else ("weight",))
    ]
    expected = reference(weight_router_after_down=True)
    activation_weighted = reference(weight_router_after_down=False)
    tolerance = 0.08 if fp8 else 0.01
    # The post-expert norm makes the two weighting orders differ by far more than the tolerance.
    assert _relative_error(activation_weighted[0], expected[0]) > 4 * tolerance
    assert _relative_error(output, expected[0]) < tolerance
    for actual, wanted in zip((x.grad, probs.grad), expected[1:3], strict=True):
        assert _relative_error(actual, wanted) < 2 * tolerance
    for actual, wanted in zip(actual_grads, expected[3:], strict=True):
        if wanted.norm() == 0:
            # The empty expert still receives a zero gradient from the grouped GEMM.
            assert actual is not None and actual.abs().max() == 0
        else:
            assert _relative_error(actual, wanted) < 2 * tolerance


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("after_down", [False, True])
def test_te_fp32_master_weights_match_bf16_weights(fp8: bool, after_down: bool) -> None:
    """FP32 expert weights compute BF16 tokens like BF16 weights and keep FP32 gradients.

    No FSDP mixed-precision wrapper casts EP-local experts when ep_shard is 1, so FP32
    master weights reach the experts with BF16 dispatched tokens.

    Args:
        fp8: Whether the expert GEMMs use FP8 current scaling.
        after_down: Whether the router weight scales the expert output after the down projection.
    """
    te = pytest.importorskip("transformer_engine.pytorch", reason="Optional TE backend is required")
    from transformer_engine.common.recipe import Float8CurrentScaling

    torch.manual_seed(29)
    counts = [17, 0, 63, 32]
    master = _experts(counts=counts, after_down=after_down, dtype=torch.float32)
    compute = _experts(counts=counts, after_down=after_down)
    with torch.no_grad():
        for source, target in zip(master.parameters(), compute.parameters(), strict=True):
            if source.ndim == 1:
                source.normal_(std=0.5)
            # BF16-representable FP32 values make the BF16 module an exact reference.
            source.copy_(source.to(torch.bfloat16))
            target.copy_(source)
    hidden = torch.randn(sum(counts), 256, device="cuda", dtype=torch.bfloat16)
    probabilities = torch.rand(sum(counts), 1, device="cuda", dtype=torch.float32)
    upstream = torch.randn_like(hidden)
    mask = torch.ones(sum(counts), device="cuda", dtype=torch.bool)
    indices = torch.cat([torch.full((count, 1), expert) for expert, count in enumerate(counts)]).cuda()
    values = []
    for module in (master, compute):
        x = hidden.clone().requires_grad_()
        probs = probabilities.clone().requires_grad_()
        with te.autocast(enabled=fp8, recipe=Float8CurrentScaling()):
            output = module(x, mask, probs, indices)
        output.backward(upstream)
        values.append((output, x.grad, probs.grad))
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for (name, actual), expected in zip(master.named_parameters(), compute.parameters(), strict=True):
        assert actual.dtype == torch.float32 and actual.grad.dtype == torch.float32, name
        if name.startswith("down_linear.bias") and not after_down and expected.grad.norm() > 0:
            # The router-weighted down-bias correction accumulates in the bias dtype, so the BF16
            # reference rounds this gradient to BF16.
            assert _relative_error(actual.grad, expected.grad) < 1e-2, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad.float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
def test_te_fp32_master_weights_without_routed_tokens() -> None:
    """A rank that receives no routed tokens still gives FP32 master weights FP32 zero gradients."""
    pytest.importorskip("transformer_engine.pytorch", reason="Optional TE backend is required")

    class EmptyDispatcher(_GroupedInputDispatcher):
        def token_permutation2(
            self,
            hidden_states: torch.Tensor,
            num_local_tokens: int,
            token_probs: torch.Tensor,
            token_indices: torch.Tensor,
        ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
            """Route none of the local tokens to this rank's experts.

            Args:
                hidden_states: Local hidden states of shape [tokens, hidden].
                num_local_tokens: Number of local rows.
                token_probs: Routing probabilities [tokens, 1].
                token_indices: Expert IDs [tokens, 1].

            Returns:
                Empty hidden states [0, hidden], zero row counts, and empty probabilities [0].
            """
            return hidden_states[:0], [0] * len(self.counts), token_probs[:0, 0]

    counts = [0, 0, 0, 0]
    module = _experts(counts=counts, dtype=torch.float32)
    module.token_dispatcher = EmptyDispatcher(counts)
    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(8, 1, device="cuda", dtype=torch.float32, requires_grad=True)
    mask = torch.ones(8, device="cuda", dtype=torch.bool)
    indices = torch.zeros(8, 1, device="cuda", dtype=torch.long)
    module(x, mask, probs, indices).float().sum().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32, name
        assert parameter.grad.abs().max() == 0, name
