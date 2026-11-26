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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F
from torch.autograd.function import FunctionCtx
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import TENvfp4Config, set_is_first_microbatch
from nemo_automodel.components.moe import experts as experts_module
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsTE
from nemo_automodel.shared.import_utils import is_te_min_version

if TYPE_CHECKING:
    from transformer_engine.common.recipe import Recipe


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


# Relative L2 error of NVFP4 GEMMs against FP32 on the Gaussian test data below; 0.24 to 0.26 on GB300.
NVFP4_ORACLE_TOLERANCE = 0.4


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


def _nvfp4_recipe_or_skip(*, stochastic_rounding: bool) -> "Recipe":
    """The NVFP4 recipe of ``TENvfp4Config``, or a skip where the grouped-tensor path cannot run it."""
    pytest.importorskip("transformer_engine.pytorch", reason="Optional TE backend is required")
    if not is_te_min_version("2.19.0"):
        pytest.skip("Native grouped-tensor GEMMs require TE 2.19.0 or newer")
    from transformer_engine.pytorch.module.grouped_linear import is_module_grouped_tensor_path_supported

    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("NVFP4 GEMMs require a Blackwell GPU")
    recipe = TENvfp4Config(stochastic_rounding=stochastic_rounding).build_recipe()
    if not is_module_grouped_tensor_path_supported(recipe, torch.bfloat16):
        pytest.skip("The native grouped-tensor path does not support NVFP4 on this device")
    return recipe


def _nvfp4_step(
    module: GroupedExpertsTE,
    recipe: "Recipe | None",
    x: torch.Tensor,
    probs: torch.Tensor,
    upstream: torch.Tensor,
    *,
    checkpointed: bool,
) -> dict[str, torch.Tensor]:
    """Forward and backward of an expert module under a recipe, with the gradient of every input and parameter."""
    import transformer_engine.pytorch as te

    module.zero_grad(set_to_none=True)
    hidden = x.detach().clone().requires_grad_()
    probability = probs.detach().clone().requires_grad_()
    mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    indices = torch.zeros(x.shape[0], 1, device=x.device, dtype=torch.long)

    def run(hidden: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
        with te.autocast(enabled=recipe is not None, recipe=recipe):
            return module(hidden, mask, probability, indices)

    with te.autocast(enabled=recipe is not None, recipe=recipe):
        if checkpointed:
            output = checkpoint(run, hidden, probability, use_reentrant=False)
        else:
            output = run(hidden, probability)
        output.backward(upstream)
    values = {"output": output.detach(), "input_grad": hidden.grad, "probability_grad": probability.grad}
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        values[name] = parameter.grad
    return values


def _record_gemm_calls(module: GroupedExpertsTE) -> list[tuple[int, bool]]:
    """``(rows, use_grouped_tensor)`` of every call of the module's gate/up projection, in call order."""
    calls: list[tuple[int, bool]] = []

    def record(linear: torch.nn.Module, args: tuple[torch.Tensor, torch.Tensor | list[int]]) -> None:
        assert isinstance(args[1], torch.Tensor) == linear.use_grouped_tensor
        calls.append((args[0].shape[0], linear.use_grouped_tensor))

    module.gate_up_linear.register_forward_pre_hook(record)
    return calls


def _record_weight_quantizations(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Per expert weight handed to TE's weight quantization: ``True`` if quantized, ``False`` if a stored copy is reused."""
    import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear

    quantized: list[bool] = []
    quantize_weight = te_grouped_linear.quantize_weight

    def counting_quantize_weight(**kwargs: torch.Tensor | bool | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        quantized.append(kwargs["workspace"] is None or bool(kwargs["update_workspace"]))
        return quantize_weight(**kwargs)

    monkeypatch.setattr(te_grouped_linear, "quantize_weight", counting_quantize_weight)
    return quantized


def _renew_weights(reference: GroupedExpertsTE, candidate: GroupedExpertsTE) -> None:
    """New weights for an optimizer step, the same in both modules."""
    with torch.no_grad():
        for reference_parameter, parameter in zip(reference.parameters(), candidate.parameters(), strict=True):
            reference_parameter.normal_(std=0.05)
            parameter.copy_(reference_parameter)


def _random_batch(rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(rows, 256, dtype=torch.bfloat16, device="cuda")
    return x, torch.rand(rows, 1, dtype=torch.bfloat16, device="cuda"), torch.randn_like(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
@pytest.mark.parametrize("checkpointed", [False, True])
def test_native_te_grouped_tensor_matches_the_split_path_under_nvfp4(checkpointed: bool) -> None:
    """NVFP4 experts compute the same through the grouped-tensor GEMMs as through one GEMM per expert.

    Stochastic rounding is off: the two paths draw different random numbers for it, so only round-to-nearest
    gradients can be compared value by value.
    """
    recipe = _nvfp4_recipe_or_skip(stochastic_rounding=False)
    torch.manual_seed(13)
    # Padded to the NVFP4 alignment of 128 rows: one group, an empty expert, one group, three groups.
    counts = [17, 0, 63, 300]
    reference = _experts(native=False, counts=counts, bias=False)
    candidate = _experts(native=True, counts=counts, bias=False)
    with torch.no_grad():
        for reference_parameter, parameter in zip(reference.parameters(), candidate.parameters(), strict=True):
            parameter.copy_(reference_parameter)
    split_calls, grouped_calls = _record_gemm_calls(reference), _record_gemm_calls(candidate)
    try:
        for microbatch in range(2):
            set_is_first_microbatch(microbatch == 0)
            x, probs, upstream = _random_batch(sum(counts))
            split, grouped = (
                _nvfp4_step(module, recipe, x, probs, upstream, checkpointed=checkpointed)
                for module in (reference, candidate)
            )
            # Same quantized operands in both paths: the forward and the input gradient agree bit for bit.
            assert torch.equal(grouped["output"], split["output"])
            assert torch.equal(grouped["input_grad"], split["input_grad"])
            for name, expected in split.items():
                # One grouped GEMM sums each weight gradient in another order than one GEMM per expert.
                assert _relative_l2(grouped[name], expected) < 1e-4, name
                if name.endswith("weight1"):
                    assert torch.count_nonzero(grouped[name]).item() == 0, name
            # Both must approximate the unquantized computation equally well.
            oracle = _fp32_reference(reference, x, probs, upstream, counts)
            for name, expected in oracle.items():
                split_error = _relative_l2(split[name], expected)
                grouped_error = _relative_l2(grouped[name], expected)
                assert max(split_error, grouped_error) < NVFP4_ORACLE_TOLERANCE, name
                assert abs(grouped_error - split_error) < 1e-3, name
    finally:
        set_is_first_microbatch(None)
    # The dispatch with an empty expert ran on the path each module is built for, in every forward.
    assert len(split_calls) == len(grouped_calls) == (4 if checkpointed else 2)
    assert not any(path for _, path in split_calls) and all(path for _, path in grouped_calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
@pytest.mark.parametrize("checkpointed", [False, True])
def test_nvfp4_dispatch_above_the_grouped_limit_runs_the_per_expert_path(
    monkeypatch: pytest.MonkeyPatch, checkpointed: bool
) -> None:
    """A dispatch above the grouped quantization limit computes what ``te_grouped_tensor=False`` computes, bit for bit.

    The limit is lowered so that this small dispatch exceeds it. The weight quantizations match the split module over
    optimizer steps and all three values of the micro-batch flag, and the module returns to grouped-tensor GEMMs
    once the limit allows it.
    """
    recipe = _nvfp4_recipe_or_skip(stochastic_rounding=False)
    torch.manual_seed(29)
    # Padded to 384, 128, 128 and 128 rows of a 1024-wide gate/up output: 786,432 elements.
    counts = [300, 17, 40, 63]
    monkeypatch.setattr(experts_module, "_NVFP4_GROUPED_QUANTIZE_MAX_ELEMENTS", 768 * 1024 - 1)
    reference = _experts(native=False, counts=counts, bias=False)
    candidate = _experts(native=True, counts=counts, bias=False)
    reference_calls, candidate_calls = _record_gemm_calls(reference), _record_gemm_calls(candidate)
    weight_quantized = _record_weight_quantizations(monkeypatch)
    forwards = 0
    try:
        for flags in ((True, False), (True, False), (None,)):
            _renew_weights(reference, candidate)
            for flag in flags:
                set_is_first_microbatch(flag)
                x, probs, upstream = _random_batch(sum(counts))
                results, quantizations = [], []
                for module in (reference, candidate):
                    weight_quantized.clear()
                    results.append(_nvfp4_step(module, recipe, x, probs, upstream, checkpointed=checkpointed))
                    quantizations.append(sum(weight_quantized))
                forwards += 2 if checkpointed else 1
                # Two projections of four experts, quantized in the forward and again in its recompute.
                assert quantizations == [{True: 8, False: 0, None: 8}[flag] * (2 if checkpointed else 1)] * 2
                per_expert, fallback = results
                assert per_expert.keys() == fallback.keys()
                for name, expected in per_expert.items():
                    assert torch.equal(fallback[name], expected), name
        assert [path for _, path in reference_calls] == [False] * forwards
        assert [path for _, path in candidate_calls] == [False] * forwards
        assert candidate.gate_up_linear.use_grouped_tensor and candidate.down_linear.use_grouped_tensor

        # Within the limit the same module runs grouped-tensor GEMMs again, with the same result.
        monkeypatch.setattr(experts_module, "_NVFP4_GROUPED_QUANTIZE_MAX_ELEMENTS", 768 * 1024)
        set_is_first_microbatch(None)
        per_expert, grouped = (
            _nvfp4_step(module, recipe, x, probs, upstream, checkpointed=checkpointed)
            for module in (reference, candidate)
        )
        assert [path for _, path in candidate_calls[forwards:]] == [True] * (2 if checkpointed else 1)
        assert torch.equal(grouped["output"], per_expert["output"])
        assert torch.equal(grouped["input_grad"], per_expert["input_grad"])
        for name, expected in per_expert.items():
            assert _relative_l2(grouped[name], expected) < 1e-4, name
    finally:
        set_is_first_microbatch(None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_nvfp4_dispatch_above_the_per_expert_limit_runs_in_chunks(
    monkeypatch: pytest.MonkeyPatch, native: bool, checkpointed: bool
) -> None:
    """A dispatch above the per-expert quantization limit is quantized in chunks of rows and stays right.

    The limit is lowered to 512 rows of the 1024-wide gate/up output. Chunks that end at expert boundaries give the
    results of one call bit for bit; an expert larger than a chunk is quantized in pieces, so it is compared with FP32
    and the experts that were not cut with the single call.
    """
    recipe = _nvfp4_recipe_or_skip(stochastic_rounding=False)
    torch.manual_seed(37)
    limit = 512 * 1024
    weight_quantized = _record_weight_quantizations(monkeypatch)
    forwards = 2 if checkpointed else 1
    try:
        # Padded to 384, 128, 0, 128 rows: chunks of 512 and 128 rows that end at expert boundaries. Then
        # 1024, 128, 0, 128 rows: the first expert fills two chunks, the others share the third.
        cases = [
            (counts, chunk_rows, flags)
            for counts, chunk_rows in (([300, 17, 0, 63], [512, 128]), ([900, 17, 0, 63], [512, 512, 256]))
            for flags in ((True, False), (None,))
        ]
        for counts, chunk_rows, flags in cases:
            reference = _experts(native=False, counts=counts, bias=False)
            candidate = _experts(native=native, counts=counts, bias=False)
            calls = _record_gemm_calls(candidate)
            _renew_weights(reference, candidate)
            for flag in flags:
                set_is_first_microbatch(flag)
                x, probs, upstream = _random_batch(sum(counts))

                weight_quantized.clear()
                single = _nvfp4_step(reference, recipe, x, probs, upstream, checkpointed=checkpointed)
                single_quantizations = sum(weight_quantized)
                calls.clear()
                weight_quantized.clear()
                with monkeypatch.context() as lowered:
                    lowered.setattr(experts_module, "_NVFP4_SPLIT_QUANTIZE_MAX_ELEMENTS", limit)
                    chunked = _nvfp4_step(candidate, recipe, x, probs, upstream, checkpointed=checkpointed)
                chunked_quantizations = sum(weight_quantized)

                # Every call ran one GEMM per expert on at most 512 rows; the chunks cover the padded rows.
                assert calls == [(rows, False) for rows in chunk_rows] * forwards
                assert candidate.gate_up_linear.use_grouped_tensor is native
                # Two projections of four experts. In a first micro-batch the first call of the forward quantizes
                # the cached weights and the first call of its recompute quantizes them again; without the cache
                # every call quantizes them.
                assert single_quantizations == {True: 8 * forwards, False: 0, None: 8 * forwards}[flag]
                expected_chunked = {True: 8 * forwards, False: 0, None: 8 * forwards * len(chunk_rows)}
                assert chunked_quantizations == expected_chunked[flag]
                assert single.keys() == chunked.keys()
                if len(chunk_rows) == 2:
                    for name, expected in single.items():
                        assert torch.equal(chunked[name], expected), name
                    continue
                # The first expert was quantized in two pieces: its values differ from the single call by
                # NVFP4 rounding, the other experts' values do not differ.
                first = slice(0, counts[0])
                rest = slice(counts[0], sum(counts))
                oracle = _fp32_reference(candidate, x, probs, upstream, counts)
                for name in ("output", "input_grad"):
                    assert torch.equal(chunked[name][rest], single[name][rest]), name
                    assert _relative_l2(chunked[name][first], oracle[name][first]) < NVFP4_ORACLE_TOLERANCE, name
                for linear in ("gate_up_linear", "down_linear"):
                    for expert in (1, 2, 3):
                        name = f"{linear}.weight{expert}"
                        assert torch.equal(chunked[name], single[name]), name
                    name = f"{linear}.weight0"
                    assert _relative_l2(chunked[name], oracle[name]) < NVFP4_ORACLE_TOLERANCE, name
    finally:
        set_is_first_microbatch(None)


class _NanFilledEmpty:
    """``torch`` as Transformer Engine's ``grouped_linear`` module sees it, with ``empty`` returning NaN.

    Transformer Engine allocates the outputs of its GEMMs with ``torch.empty``. With this in its place an
    element that a GEMM does not write is NaN instead of whatever the allocator's block held.
    """

    def __getattr__(self, name: str) -> object:
        return getattr(torch, name)

    def empty(self, *args: int, **kwargs: torch.dtype | torch.device | str) -> torch.Tensor:
        """Allocate like ``torch.empty`` and fill floating-point tensors with NaN."""
        tensor = torch.empty(*args, **kwargs)
        return tensor.fill_(float("nan")) if tensor.is_floating_point() else tensor


class _SharedExpert(torch.nn.Module):
    """The shared expert of the model, 4096 -> 5376 -> 4096 with ``relu(x)**2``, on Transformer Engine linears."""

    def __init__(self) -> None:
        super().__init__()
        import transformer_engine.pytorch as te

        self.up = te.Linear(4096, 5376, bias=False, params_dtype=torch.bfloat16, device="cuda")
        self.down = te.Linear(5376, 4096, bias=False, params_dtype=torch.bfloat16, device="cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)).square())


# Rows per expert of one rank of 32 experts of 1024 -> 2688 -> 1024, multiples of the 128 rows NVFP4 pads to.
REAL_LAYOUTS: dict[str, list[int]] = {
    # 22 routes for each of 8 x 8192 tokens on one of 16 ranks of 512 experts: 3.9e9 elements in the
    # intermediate tensor, above the limit of the grouped quantization.
    "balanced_3.9e9": [45056] * 32,
    # About the same total spread unevenly, with two experts without rows.
    "uneven_4.1e9": [96000] * 8 + [24064] * 16 + [0] * 2 + [64000] * 6,
    # One expert holds 1,600,000 rows, 4.3e9 elements by itself; two experts have no rows.
    "one_expert_4.3e9": [1280] * 3 + [1600000] + [1280] * 3 + [0] + [1280] * 12 + [0] + [1280] * 11,
    # Half the experts without rows; the others stay within the limit of the grouped quantization (1.9e9).
    "half_empty_1.9e9": [0] * 16 + [45056] * 16,
}
REAL_WIDTH = 2688
# 15/16 of the 2**31 and 2**32 elements at which the two NVFP4 quantizations start to fail.
REAL_GROUPED_LIMIT = 2013265920
REAL_PER_EXPERT_LIMIT = 4026531840


def _rank_config(num_experts: int, *, native: bool, side_stream: bool) -> tuple[MoEConfig, BackendConfig]:
    """Config of one rank's experts of 1024 -> 2688 -> 1024; the test runs the shared expert itself."""
    config = MoEConfig(
        n_routed_experts=num_experts,
        n_shared_experts=int(side_stream),
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=1024,
        inter_dim=REAL_WIDTH,
        moe_inter_dim=REAL_WIDTH,
        norm_topk_prob=False,
        expert_bias=False,
        expert_activation="relu2",
        dtype=torch.bfloat16,
    )
    backend = BackendConfig(
        experts="te", dispatcher="deepep", te_grouped_tensor=native, shared_expert_overlap=side_stream
    )
    return config, backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped GEMMs require CUDA")
@pytest.mark.parametrize(
    ("layout", "native", "side_stream"),
    [(layout, native, False) for layout in REAL_LAYOUTS for native in (True, False)]
    + [("half_empty_1.9e9", False, True)],
)
def test_nvfp4_experts_at_real_sizes_match_fp32_row_by_row(
    monkeypatch: pytest.MonkeyPatch, layout: str, native: bool, side_stream: bool
) -> None:
    """NVFP4 experts are right in every row and expert at the sizes one rank of the model receives.

    TE's NVFP4 quantization is wrong past 2**31 elements on the grouped-tensor path (rows from that offset on get the
    last expert's tensor scale) and wraps past 2**32 with one GEMM per expert. Expert inputs differ in scale by
    factors of 3 and 9, so a row quantized with another expert's scale is off by more than 0.5 of its norm
    (NVFP4 rounding leaves 0.2 to 0.35), and TE's output buffers are pre-filled with NaN so an unwritten element
    shows. Output, input gradient and both weight gradients are compared with FP32 values computed here, expert by
    expert. The side stream is exercised with one GEMM per expert only: on cuBLASLt
    before 13.8.1 a grouped GEMM can leave output unwritten while a GEMM runs on a second stream."""
    recipe = _nvfp4_recipe_or_skip(stochastic_rounding=True)
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear

    if torch.cuda.mem_get_info()[0] < 160 * 2**30:
        pytest.skip("Needs 160 GiB of free GPU memory")
    monkeypatch.setattr(te_grouped_linear, "torch", _NanFilledEmpty())

    counts = REAL_LAYOUTS[layout]
    num_experts, dim, inter = len(counts), 1024, REAL_WIDTH
    total = sum(counts)
    elements = total * inter
    config, backend = _rank_config(num_experts, native=native, side_stream=side_stream)
    module = GroupedExpertsTE(config, backend=backend)
    module.gate_up_linear.reset_parameters()
    module.down_linear.reset_parameters()
    module.ep_size = 1
    module.token_dispatcher = _GroupedInputDispatcher(counts)
    calls = _record_gemm_calls(module)
    shared = _SharedExpert() if side_stream else None
    stream = torch.cuda.Stream() if side_stream else None

    generator = torch.Generator(device="cuda").manual_seed(77)
    scale = torch.tensor([3.0 if expert % 2 == 0 else 1 / 3 for expert in range(num_experts - 1)] + [1.0])
    x = torch.randn(total, dim, generator=generator, device="cuda")
    x = (x * scale.cuda().repeat_interleave(torch.tensor(counts).cuda()).unsqueeze(1)).to(torch.bfloat16)
    probs = 0.05 + 0.4 * torch.rand(total, 1, generator=generator, device="cuda")
    upstream = (torch.randn(total, dim, generator=generator, device="cuda") * 1e-3).to(torch.bfloat16)
    shared_input = torch.randn(65536, 4096, generator=generator, device="cuda").to(torch.bfloat16)
    mask = torch.ones(total, device="cuda", dtype=torch.bool)
    indices = torch.zeros(total, 1, device="cuda", dtype=torch.long)

    def block(hidden: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The routed experts, with the shared expert on the side stream when the case asks for it."""
        with te.autocast(enabled=True, recipe=recipe):
            if stream is None:
                return module(hidden, mask, probs, indices), tokens
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                side = shared(tokens)
            tokens.record_stream(stream)
            routed = module(hidden, mask, probs, indices)
            current = torch.cuda.current_stream()
            current.wait_stream(stream)
            side.record_stream(current)
            return routed, side

    def row_errors(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
        """Relative L2 error of every row."""
        assert torch.isfinite(actual).all()
        return (actual.float() - expected).norm(dim=1) / expected.norm(dim=1)

    # A dispatch with empty experts within the grouped limit is repeated: its exact zeros hold in every round.
    rounds = 24 if 0 in counts and elements <= REAL_GROUPED_LIMIT else 1
    set_is_first_microbatch(True)
    try:
        for _ in range(rounds):
            module.zero_grad(set_to_none=True)
            hidden = x.clone().requires_grad_()
            tokens = shared_input.clone().requires_grad_()
            with te.autocast(enabled=True, recipe=recipe):
                output, side = checkpoint(block, hidden, tokens, use_reentrant=False)
                torch.autograd.backward([output, side], [upstream, torch.ones_like(side) * 1e-3])
            assert hidden.grad is not None

            # The expert MLP and its gradients in FP32, expert by expert in blocks of rows, written out
            # without autograd.
            worst = {"output": 0.0, "input_grad": 0.0, "gate_up_weight_grad": 0.0, "down_weight_grad": 0.0}
            offset = 0
            with torch.no_grad():
                for expert, rows in enumerate(counts):
                    gate_up = getattr(module.gate_up_linear, f"weight{expert}")
                    down = getattr(module.down_linear, f"weight{expert}")
                    if rows == 0:
                        # No rows, no gradient: exact zeros, not what the output buffer held.
                        assert torch.count_nonzero(gate_up.grad) == 0, expert
                        assert torch.count_nonzero(down.grad) == 0, expert
                        continue
                    gate_up_grad = torch.zeros_like(gate_up, dtype=torch.float32)
                    down_grad = torch.zeros_like(down, dtype=torch.float32)
                    for start in range(offset, offset + rows, 131072):
                        span = slice(start, min(start + 131072, offset + rows))
                        inputs, probability, output_grad = x[span].float(), probs[span], upstream[span].float()
                        pre_activation = inputs @ gate_up.float().T
                        activated = pre_activation.relu().square() * probability
                        activated_grad = output_grad @ down.float()
                        pre_activation_grad = activated_grad * 2 * pre_activation.relu() * probability
                        worst["output"] = max(
                            worst["output"], row_errors(output[span], activated @ down.float().T).max().item()
                        )
                        worst["input_grad"] = max(
                            worst["input_grad"],
                            row_errors(hidden.grad[span], pre_activation_grad @ gate_up.float()).max().item(),
                        )
                        gate_up_grad += pre_activation_grad.T @ inputs
                        down_grad += output_grad.T @ activated
                    worst["gate_up_weight_grad"] = max(
                        worst["gate_up_weight_grad"], _relative_l2(gate_up.grad, gate_up_grad)
                    )
                    worst["down_weight_grad"] = max(worst["down_weight_grad"], _relative_l2(down.grad, down_grad))
                    offset += rows
            # On GB300: 0.26 to 0.28 and 0.34 to 0.36 for the worst row, 0.29 and 0.25 for the worst expert.
            assert worst["output"] < 0.45, worst
            assert worst["input_grad"] < 0.45, worst
            assert worst["gate_up_weight_grad"] < NVFP4_ORACLE_TOLERANCE, worst
            assert worst["down_weight_grad"] < NVFP4_ORACLE_TOLERANCE, worst

    finally:
        set_is_first_microbatch(None)

    # No quantization covered more than its path handles: every forward and every recompute ran the same
    # calls, each within the per-expert limit, on grouped-tensor GEMMs only for a module built for them and a
    # dispatch within their limit.
    per_pass = len(calls) // (2 * rounds)
    forward_calls = calls[:per_pass]
    assert calls == forward_calls * (2 * rounds)
    assert sum(rows for rows, _ in forward_calls) == total
    assert all(rows * inter <= REAL_PER_EXPERT_LIMIT for rows, _ in forward_calls)
    # The layouts above need no extra call for ending their chunks at expert boundaries.
    assert len(forward_calls) == math.ceil(elements / REAL_PER_EXPERT_LIMIT)
    grouped = native and elements <= REAL_GROUPED_LIMIT
    assert all(path is grouped for _, path in forward_calls)
    assert module.gate_up_linear.use_grouped_tensor is native


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
