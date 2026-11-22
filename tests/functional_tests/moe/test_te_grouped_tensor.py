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
import torch.nn.functional as F
from torch.autograd.function import FunctionCtx
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import set_is_first_microbatch
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsTE
from nemo_automodel.shared.import_utils import is_te_min_version


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


def _experts(*, native: bool, counts: list[int], bias: bool = True) -> GroupedExpertsTE:
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
    backend = BackendConfig(experts="te", dispatcher="deepep", te_grouped_tensor=native)
    module = GroupedExpertsTE(config, backend=backend)
    assert module.gate_up_linear.use_grouped_tensor is native
    assert module.down_linear.use_grouped_tensor is native
    module.gate_up_linear.reset_parameters()
    module.down_linear.reset_parameters()
    module.ep_size = 1
    module.token_dispatcher = _GroupedInputDispatcher(counts)
    return module


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Measure tensor error without dividing by individual near-zero elements.

    Args:
        actual: Observed tensor, with the same shape as ``expected``.
        expected: Reference tensor of any shape.

    Returns:
        Ratio of error L2 norm to reference L2 norm; zero references require exact equality.
    """
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    actual, expected = actual.float(), expected.float()
    norm = expected.norm().item()
    if norm == 0:
        assert torch.count_nonzero(actual).item() == 0
        return 0.0
    return (actual - expected).norm().item() / norm


def _fp32_reference(
    module: GroupedExpertsTE,
    hidden: torch.Tensor,
    probability: torch.Tensor,
    upstream: torch.Tensor,
    counts: list[int],
) -> dict[str, torch.Tensor]:
    """Evaluate an independent, unquantized expert MLP and its gradients.

    Args:
        module: Expert module whose parameters supply the reference weights.
        hidden: Tensor of shape [tokens, hidden], grouped by expert.
        probability: Routing probabilities of shape [tokens, 1].
        upstream: Output gradients of shape [tokens, hidden].
        counts: Token count for each expert, including empty experts.

    Returns:
        FP32 output, input and routing gradients, and gradients keyed by parameter name.
    """
    parameters = {name: value.detach().float().requires_grad_() for name, value in module.named_parameters()}
    hidden = hidden.detach().float().requires_grad_()
    probability = probability.detach().float().requires_grad_()
    outputs = []
    offset = 0
    for expert, count in enumerate(counts):
        rows = slice(offset, offset + count)
        gate_up = F.linear(
            hidden[rows],
            parameters[f"gate_up_linear.weight{expert}"],
            parameters[f"gate_up_linear.bias{expert}"] if module.expert_bias else None,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up * probability[rows]
        output = F.linear(activated, parameters[f"down_linear.weight{expert}"])
        if module.expert_bias:
            output = output + parameters[f"down_linear.bias{expert}"] * probability[rows]
        outputs.append(output)
        offset += count
    output = torch.cat(outputs)
    gradients = torch.autograd.grad(output, (hidden, probability, *parameters.values()), grad_outputs=upstream.float())
    return {
        "output": output,
        "input_grad": gradients[0],
        "probability_grad": gradients[1],
        **dict(zip(parameters, gradients[2:], strict=True)),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
def test_native_te_rejects_delayed_scaling() -> None:
    te = pytest.importorskip("transformer_engine.pytorch")
    if not is_te_min_version("2.19.0"):
        pytest.skip("Native grouped-tensor GEMMs require TE 2.19.0 or newer")
    from transformer_engine.common.recipe import DelayedScaling

    module = _experts(native=True, counts=[17, 0, 63, 32])
    x = torch.randn(112, 256, device="cuda", dtype=torch.bfloat16)
    probabilities = torch.ones(112, 1, device="cuda", dtype=x.dtype)
    indices = torch.zeros(112, 1, device="cuda", dtype=torch.long)
    with te.autocast(enabled=True, recipe=DelayedScaling()):
        with pytest.raises(ValueError, match="Set te_grouped_tensor=False"):
            module(x, torch.ones(112, device="cuda", dtype=torch.bool), probabilities, indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
@pytest.mark.parametrize("precision", ["bf16", "fp8", "block", "mxfp8"])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_native_te_grouped_tensor_numerics(precision: str, checkpointed: bool, bias: bool) -> None:
    """Preserve outputs, all gradients, updates, padding and empty-expert behavior.

    Args:
        precision: BF16, FP8 current scaling, FP8 block scaling, or MXFP8.
        checkpointed: Whether to recompute the expert forward during backward.
        bias: Whether the grouped projections include bias parameters.
    """
    te = pytest.importorskip("transformer_engine.pytorch", reason="Optional TE backend is required")
    if not is_te_min_version("2.19.0"):
        pytest.skip("Native grouped-tensor GEMMs require TE 2.19.0 or newer")
    from transformer_engine.common.recipe import Float8BlockScaling, Float8CurrentScaling, MXFP8BlockScaling
    from transformer_engine.pytorch.module.grouped_linear import is_module_grouped_tensor_path_supported

    fp8 = precision != "bf16"
    recipe = {"block": Float8BlockScaling, "mxfp8": MXFP8BlockScaling}.get(precision, Float8CurrentScaling)()
    if not is_module_grouped_tensor_path_supported(recipe if fp8 else None, torch.bfloat16):
        pytest.skip("The native grouped-tensor path does not support this device/recipe combination")

    torch.manual_seed(13)
    # Imbalanced counts exercise FP8 padding and an empty expert in both GEMMs.
    counts = [17, 0, 63, 32]
    reference = _experts(native=False, counts=counts, bias=bias)
    candidate = _experts(native=True, counts=counts, bias=bias)
    with torch.no_grad():
        for reference_parameter, parameter in zip(reference.parameters(), candidate.parameters(), strict=True):
            parameter.copy_(reference_parameter)
    optimizers = [torch.optim.SGD(module.parameters(), lr=0.001) for module in (reference, candidate)]
    mask = torch.ones(sum(counts), device="cuda", dtype=torch.bool)
    indices = torch.cat([torch.full((count, 1), expert) for expert, count in enumerate(counts)]).cuda()
    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
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

                    def run(
                        hidden: torch.Tensor, probability: torch.Tensor, *, module: GroupedExpertsTE = module
                    ) -> torch.Tensor:
                        """Evaluate the same precision recipe during checkpoint recomputation.

                        Args:
                            hidden: Tensor of shape [tokens, hidden], grouped by expert.
                            probability: Tensor of shape [tokens, 1] with routing probabilities.
                            module: Expert module bound for this checkpoint evaluation.

                        Returns:
                            Tensor of shape [tokens, hidden].
                        """
                        with te.autocast(enabled=fp8, recipe=recipe):
                            return module(hidden, mask, probability, indices)

                    if checkpointed:
                        output = checkpoint(run, hidden, probability, use_reentrant=False)
                    else:
                        output = run(hidden, probability)
                    output.backward(upstream)
                    assert hidden.grad is not None and probability.grad is not None
                    outputs.append(output.detach())
                    input_grads.append(hidden.grad)
                    probability_grads.append(probability.grad)
                for name, (expected, actual) in zip(
                    ("output", "input_grad", "probability_grad"), (outputs, input_grads, probability_grads), strict=True
                ):
                    # Different GEMM tilings and BF16 reductions produce relative
                    # outliers near zero. Bound the whole-tensor error instead.
                    assert _relative_l2(actual, expected) < (0.03 if fp8 else 0.01), name
                if microbatch == 0:
                    oracle = _fp32_reference(reference, x, probs, upstream, counts)
                    oracle_errors = []
                    for index, module in enumerate((reference, candidate)):
                        values = {
                            "output": outputs[index],
                            "input_grad": input_grads[index],
                            "probability_grad": probability_grads[index],
                        }
                        for name, parameter in module.named_parameters():
                            assert parameter.grad is not None, name
                            values[name] = parameter.grad
                        oracle_errors.append(
                            {name: _relative_l2(value, oracle[name]) for name, value in values.items()}
                        )
                    for name in oracle:
                        legacy_error, native_error = (errors[name] for errors in oracle_errors)
                        # Both implementations must approximate the independent
                        # FP32 computation, including the router and every expert.
                        assert max(legacy_error, native_error) < (0.12 if fp8 else 0.01), name
                        assert native_error <= legacy_error + 0.005, name
            for (name, expected), (_, actual) in zip(
                reference.named_parameters(), candidate.named_parameters(), strict=True
            ):
                assert expected.grad is not None and actual.grad is not None, name
                assert _relative_l2(actual.grad, expected.grad) < (0.04 if fp8 else 0.01), name
                if name.endswith(("weight1", "bias1")):
                    assert torch.count_nonzero(actual.grad).item() == 0, name
            for optimizer in optimizers:
                optimizer.step()
        # Keep per-expert parameters and the exported stacked checkpoint layout.
        for (name, expected), (actual_name, actual) in zip(
            reference.named_parameters(), candidate.named_parameters(), strict=True
        ):
            assert name == actual_name
            assert _relative_l2(actual, expected) < 0.02, name
        expected_state, actual_state = reference.state_dict(), candidate.state_dict()
        assert expected_state.keys() == actual_state.keys()
        for name, expected in expected_state.items():
            assert _relative_l2(actual_state[name], expected) < 0.02, name
    finally:
        set_is_first_microbatch(None)
        torch.set_float32_matmul_precision(original_precision)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP8 recipe switching requires CUDA")
def test_native_te_grouped_tensor_recipe_switch() -> None:
    """Changing external autocast recipes preserves expert outputs and all gradients."""
    te = pytest.importorskip("transformer_engine.pytorch", reason="Optional TE backend is required")
    if not is_te_min_version("2.19.0"):
        pytest.skip("Native grouped-tensor GEMMs require TE 2.19.0 or newer")
    from transformer_engine.common.recipe import Float8CurrentScaling, MXFP8BlockScaling
    from transformer_engine.pytorch.module.grouped_linear import is_module_grouped_tensor_path_supported

    recipes = [Float8CurrentScaling(), MXFP8BlockScaling(), Float8CurrentScaling()]
    if not all(is_module_grouped_tensor_path_supported(recipe, torch.bfloat16) for recipe in recipes):
        pytest.skip("Recipe switching requires native MXFP8 support on Blackwell")
    torch.manual_seed(41)
    counts = [17, 0, 63, 32]
    reference = _experts(native=False, counts=counts, bias=False)
    candidate = _experts(native=True, counts=counts, bias=False)
    with torch.no_grad():
        for expected, actual in zip(reference.parameters(), candidate.parameters(), strict=True):
            actual.copy_(expected)
    mask = torch.ones(sum(counts), dtype=torch.bool, device="cuda")
    indices = torch.cat([torch.full((count, 1), expert) for expert, count in enumerate(counts)]).cuda()
    try:
        for recipe in recipes:
            set_is_first_microbatch(True)
            hidden = torch.randn(sum(counts), 256, dtype=torch.bfloat16, device="cuda")
            probability = torch.rand(sum(counts), 1, dtype=torch.bfloat16, device="cuda")
            upstream = torch.randn_like(hidden)
            results = []
            for module in (reference, candidate):
                x = hidden.detach().clone().requires_grad_()
                probs = probability.detach().clone().requires_grad_()
                with te.autocast(enabled=True, recipe=recipe):
                    output = module(x, mask, probs, indices)
                gradients = torch.autograd.grad(output, (x, probs, *module.parameters()), upstream)
                results.append((output, *gradients))
            for expected, actual in zip(*results, strict=True):
                assert _relative_l2(actual, expected) < 0.04
    finally:
        set_is_first_microbatch(None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("train_router", [False, True])
def test_native_te_dispatch_gradient_parity(fp8: bool, checkpointed: bool, train_router: bool) -> None:
    """Native GEMMs preserve gradients with coupled dispatcher outputs.

    This checks public outputs/gradients, independently of the optional optimization
    that detaches unused dispatcher probability gradients.

    Args:
        fp8: Whether the expert GEMMs use FP8 current scaling.
        checkpointed: Whether to recompute the expert forward during backward.
        train_router: Whether the original routing probabilities need gradients.
    """
    te = pytest.importorskip("transformer_engine.pytorch", reason="Optional TE backend is required")
    if not is_te_min_version("2.19.0"):
        pytest.skip("Native grouped-tensor GEMMs require TE 2.19.0 or newer")
    from transformer_engine.common.recipe import Float8CurrentScaling
    from transformer_engine.pytorch.module.grouped_linear import is_module_grouped_tensor_path_supported

    if not is_module_grouped_tensor_path_supported(Float8CurrentScaling() if fp8 else None, torch.bfloat16):
        pytest.skip("The native grouped-tensor path does not support this device/recipe combination")

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
    reference = _experts(native=True, counts=counts, bias=False)
    candidate = _experts(native=True, counts=counts, bias=False)
    candidate.token_dispatcher = CoupledDispatcher(counts)
    with torch.no_grad():
        for expected, actual in zip(reference.parameters(), candidate.parameters(), strict=True):
            actual.copy_(expected)
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
