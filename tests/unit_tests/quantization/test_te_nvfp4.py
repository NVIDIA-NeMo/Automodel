# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""NVFP4 forward with MXFP8 backward through real Transformer Engine modules (Blackwell GPUs only)."""

from collections import OrderedDict

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.common.utils import HAVE_TE, TENvfp4Config
from nemo_automodel.components.quantization.stochastic_rounding import stochastic_round_to_e4m3
from nemo_automodel.shared.import_utils import is_te_min_version


def _nvfp4_capable() -> bool:
    return HAVE_TE and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


pytestmark = pytest.mark.skipif(
    not (_nvfp4_capable() and is_te_min_version("2.19.0")),
    reason="requires Transformer Engine 2.19+ and an NVFP4 GPU",
)

RECIPE_OPTIONS = {
    "random_hadamard_transform": False,
    "weight_blocks_2d": False,
    "row_scaled_activation": True,
    "four_over_six": "activations",
    "backward": "mxfp8",
}


def _relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value.double() - reference.double()).norm() / reference.double().norm())


def _mxfp8_columnwise(stochastic: bool):
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    from nemo_automodel.components.quantization.te_nvfp4 import StochasticMXFP8Quantizer

    if stochastic:
        return StochasticMXFP8Quantizer(rowwise=False, columnwise=True)
    return MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3, rowwise=False, columnwise=True)


def test_stochastic_cast_stores_the_stochastically_rounded_values():
    # Magnitudes within a factor of four: every value is in the exactly representable range of its block.
    torch.manual_seed(0)
    magnitude = 0.5 + 1.5 * torch.rand(256, 512, device="cuda")
    values = (magnitude * torch.where(torch.rand_like(magnitude) < 0.5, -1.0, 1.0)).to(torch.bfloat16)
    torch.manual_seed(5)
    expected = stochastic_round_to_e4m3(values)
    torch.manual_seed(5)
    stored = _mxfp8_columnwise(stochastic=True)(values).dequantize(dtype=torch.float32)
    assert torch.equal(stored, expected.float())
    # Control: the plain MXFP8 cast of the same values is a different tensor.
    nearest = _mxfp8_columnwise(stochastic=False)(values).dequantize(dtype=torch.float32)
    assert not torch.equal(nearest, expected.float())


def test_stochastic_cast_has_no_bias_and_the_nearest_cast_has():
    torch.manual_seed(0)
    values = (torch.randn(512, 512, device="cuda") * torch.exp(torch.randn(512, 1, device="cuda"))).to(torch.bfloat16)
    draws = 300
    quantizer = _mxfp8_columnwise(stochastic=True)
    total = torch.zeros_like(values, dtype=torch.float64)
    for _ in range(draws):
        total += quantizer(values).dequantize(dtype=torch.float32).double()
    one_draw = _relative_error(quantizer(values).dequantize(dtype=torch.float32), values)
    nearest = _relative_error(_mxfp8_columnwise(stochastic=False)(values).dequantize(dtype=torch.float32), values)
    assert one_draw > 0.02 and nearest > 0.02
    # The error of the mean shrinks with the square root of the number of draws.
    assert _relative_error(total / draws, values) < 3 * one_draw / draws**0.5


def test_factory_builds_the_quantizer_of_every_tensor():
    from transformer_engine.pytorch.quantization import QuantizerRole
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    from nemo_automodel.components.quantization.te_nvfp4 import (
        NVFP4ForwardMXFP8BackwardFactory,
        StochasticMXFP8Quantizer,
    )

    factory = NVFP4ForwardMXFP8BackwardFactory(
        weight_blocks_2d=False, row_scaled_activation=True, four_over_six="activations", stochastic_rounding=True
    )
    for module_type in ("linear", "grouped_linear"):
        activation = factory(QuantizerRole(module_type=module_type, tensor_type="input"))
        weight = factory(QuantizerRole(module_type=module_type, tensor_type="weight"))
        for operand in (activation, weight):
            assert isinstance(operand, HybridQuantizer) and operand.columnwise_source == "rowwise_dequantized"
            assert isinstance(operand.rowwise_quantizer, NVFP4Quantizer)
            assert isinstance(operand.columnwise_quantizer, StochasticMXFP8Quantizer)
            assert not operand.rowwise_quantizer.with_rht and not operand.rowwise_quantizer.stochastic_rounding
            assert not operand.rowwise_quantizer.with_2d_quantization
        assert activation.rowwise_quantizer.row_scaled_nvfp4 and activation.rowwise_quantizer.nvfp4_use_4over6
        assert activation.rowwise_quantizer.nvfp4_e4m3_max == 256
        assert not weight.rowwise_quantizer.row_scaled_nvfp4 and not weight.rowwise_quantizer.nvfp4_use_4over6
        assert type(factory(QuantizerRole(module_type=module_type, tensor_type="grad_output"))) is MXFP8Quantizer
    assert type(factory(None)) is MXFP8Quantizer

    other = NVFP4ForwardMXFP8BackwardFactory(
        weight_blocks_2d=True, row_scaled_activation=False, four_over_six="weights", stochastic_rounding=False
    )
    activation = other(QuantizerRole(module_type="grouped_linear", tensor_type="input"))
    weight = other(QuantizerRole(module_type="grouped_linear", tensor_type="weight"))
    assert (
        type(activation.columnwise_quantizer) is MXFP8Quantizer and type(weight.columnwise_quantizer) is MXFP8Quantizer
    )
    assert not activation.rowwise_quantizer.row_scaled_nvfp4 and not activation.rowwise_quantizer.nvfp4_use_4over6
    assert weight.rowwise_quantizer.with_2d_quantization and weight.rowwise_quantizer.nvfp4_use_4over6


def test_config_builds_one_recipe_per_set_of_options():
    from transformer_engine.common.recipe import CustomRecipe

    from nemo_automodel.components.quantization.te_nvfp4 import NVFP4ForwardMXFP8BackwardFactory

    config = TENvfp4Config(**RECIPE_OPTIONS)
    recipe = config.build_recipe()
    assert isinstance(recipe, CustomRecipe)
    assert recipe.qfactory == NVFP4ForwardMXFP8BackwardFactory(
        weight_blocks_2d=False, row_scaled_activation=True, four_over_six="activations", stochastic_rounding=True
    )
    # Transformer Engine modules rebuild their quantizers when the recipe changes; rebuilding it must not.
    assert config.build_recipe() == recipe
    assert TENvfp4Config(**RECIPE_OPTIONS, stochastic_rounding=False).build_recipe() != recipe


def _forward_operands(inputs: torch.Tensor, weights: list[torch.Tensor], splits: list[int]):
    """Dequantized NVFP4 operands of the forward GEMMs, as FP64: per-expert inputs and weights."""
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    def quantizer(*, row_scaled: bool, four_over_six: bool) -> NVFP4Quantizer:
        return NVFP4Quantizer(
            fp4_dtype=DType.kFloat4E2M1,
            rowwise=True,
            columnwise=False,
            row_scaled_nvfp4=row_scaled,
            nvfp4_use_4over6=four_over_six,
            nvfp4_e4m3_max=256 if four_over_six else 448,
        )

    activations = [
        quantizer(row_scaled=True, four_over_six=True)(part.contiguous()).dequantize(dtype=torch.bfloat16).double()
        for part in inputs.split(splits)
    ]
    dequantized = [
        quantizer(row_scaled=False, four_over_six=False)(weight).dequantize(dtype=torch.bfloat16).double()
        for weight in weights
    ]
    return activations, dequantized


@pytest.mark.parametrize("stochastic", [True, False])
def test_backward_gemms_see_the_forward_operands_in_expectation(stochastic: bool):
    """Mean gradients over many passes equal the gradients of the dequantized forward operands; not with nearest."""
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    experts, rows, in_features, out_features = 2, 256, 512, 256
    splits = [rows] * experts
    torch.manual_seed(0)
    weights = [torch.randn(out_features, in_features, device="cuda").mul_(0.02).bfloat16() for _ in range(experts)]
    inputs = torch.randn(experts * rows, in_features, device="cuda")
    inputs = (inputs * torch.exp(torch.randn(experts * rows, 1, device="cuda"))).bfloat16()
    output_grad = torch.randn(experts * rows, out_features, device="cuda").bfloat16()

    activations, dequantized_weights = _forward_operands(inputs, weights, splits)
    gradient = [
        MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3)(part.contiguous()).dequantize(dtype=torch.float32).double()
        for part in output_grad.split(splits)
    ]
    expected_output = torch.cat([x @ w.T for x, w in zip(activations, dequantized_weights)])
    expected_input_grad = torch.cat([g @ w for g, w in zip(gradient, dequantized_weights)])
    expected_weight_grad = torch.stack([g.T @ x for g, x in zip(gradient, activations)])

    linear = te.GroupedLinear(
        num_gemms=experts,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
    )
    with torch.no_grad():
        for index, weight in enumerate(weights):
            getattr(linear, f"weight{index}").copy_(weight)
    config = TENvfp4Config(**RECIPE_OPTIONS, stochastic_rounding=stochastic)
    passes = 100
    input_grad_sum = torch.zeros_like(expected_input_grad)
    weight_grad_sum = torch.zeros_like(expected_weight_grad)
    for index in range(passes):
        leaf = inputs.clone().requires_grad_()
        with config.maybe_te_autocast():
            output = linear(leaf, splits)
        output.backward(output_grad)
        input_grad = leaf.grad.double()
        weight_grad = torch.stack([getattr(linear, f"weight{i}").grad.double() for i in range(experts)])
        linear.zero_grad()
        input_grad_sum += input_grad
        weight_grad_sum += weight_grad
        if index == 0:
            assert _relative_error(output.detach(), expected_output) < 0.01
            one_pass = (
                _relative_error(input_grad, expected_input_grad),
                _relative_error(weight_grad, expected_weight_grad),
            )
    mean = (
        _relative_error(input_grad_sum / passes, expected_input_grad),
        _relative_error(weight_grad_sum / passes, expected_weight_grad),
    )
    # One pass carries the MXFP8 rounding error of the backward operands either way.
    assert min(one_pass) > 0.015
    if stochastic:
        assert max(mean) < 0.01
    else:
        assert min(mean) > 0.015  # rounding to nearest repeats the same error in every pass


def test_mxfp8_fqns_modules_run_the_mxfp8_recipe_inside_the_nvfp4_autocast():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.quantization import autocast

    def build() -> nn.ModuleDict:
        # Transformer Engine modules keep the quantizers of their first forward, so every recipe gets its own copy.
        torch.manual_seed(0)
        layers = OrderedDict(
            (str(index), te.Linear(512, 512, bias=False, params_dtype=torch.bfloat16)) for index in range(2)
        )
        return nn.ModuleDict({"layers": nn.ModuleDict(layers)}).to("cuda")

    model, mxfp8_model = build(), build()
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    config = TENvfp4Config(**RECIPE_OPTIONS, mxfp8_fqns=["layers.1"])
    config.prepare_modules(model)
    with torch.no_grad():
        with config.maybe_te_autocast():
            outputs = [model.layers[index](x) for index in ("0", "1")]
        with autocast(enabled=True, recipe=MXFP8BlockScaling()):
            mxfp8_outputs = [mxfp8_model.layers[index](x) for index in ("0", "1")]
        exact = [x.float() @ model.layers[index].weight.float().T for index in ("0", "1")]
    assert torch.equal(outputs[1], mxfp8_outputs[1])
    # The other module ran NVFP4: a different result with a larger error than MXFP8.
    assert not torch.equal(outputs[0], mxfp8_outputs[0])
    assert _relative_error(outputs[0], exact[0]) > 2 * _relative_error(mxfp8_outputs[0], exact[0])
