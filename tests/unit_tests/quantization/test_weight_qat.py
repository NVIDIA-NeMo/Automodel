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

"""CPU numerical and storage-contract tests using independent scalar oracles."""

import itertools
import math
import struct
from dataclasses import FrozenInstanceError, asdict, fields, replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate

from nemo_automodel.components.quantization.weight_qat import (
    QuantizedWeight,
    WeightFakeQuantizer,
    WeightQuantizationConfig,
)

_FP8 = WeightQuantizationConfig(format="fp8", block_size=(32, 32))
_FP8_FLOAT = WeightQuantizationConfig(format="fp8", block_size=(32, 32), scale_format="float32")
_FP4 = WeightQuantizationConfig(format="mxfp4", block_size=(1, 32))
_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _fp8_value(code: int) -> float:
    magnitude = code & 127
    exponent, fraction = divmod(magnitude, 8)
    value = math.ldexp(fraction, -9) if exponent == 0 else math.ldexp(1 + fraction / 8, exponent - 7)
    return math.copysign(value, -1 if code & 128 else 1)


def _fp4_value(code: int) -> float:
    return math.copysign((0, 0.5, 1, 1.5, 2, 3, 4, 6)[code & 7], -1 if code & 8 else 1)


def _scalar_code(value: float, *, fp8: bool) -> int:
    """Exhaustively compare scalar distances; the secondary key implements ties-even."""
    decode = _fp8_value if fp8 else _fp4_value
    code = min(range(127 if fp8 else 8), key=lambda candidate: (abs(abs(value) - decode(candidate)), candidate % 2))
    sign = (128 if fp8 else 8) if math.copysign(1, value) < 0 else 0
    return code + sign


def _scalar_scale(amax: float, maximum: float) -> int:
    """Search representable powers directly, independently of frexp/logarithms."""
    if amax == 0:
        return 127
    for exponent in range(-127, 128):
        if amax <= math.ldexp(maximum, exponent):
            return exponent + 127
    raise ValueError("unrepresentable oracle scale")


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Check values including signed zeros through their storage bits.

    Args:
        actual: Strided CPU tensor of arbitrary shape and numeric dtype.
        expected: Tensor with the same shape/dtype/device as actual. Arbitrary
            strides are accepted; neither input is mutated.
    """
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _assert_scalar_oracle(weight: torch.Tensor, quantizer: WeightFakeQuantizer) -> None:
    """Check grouping, codes, scale bytes, and QDQ against a scalar reference.

    Args:
        weight: Finite strided CPU tensor of shape [..., out, in], in float32,
            bfloat16, or float16, satisfying quantizer's block divisibility.
            Arbitrary leading axes and strides are supported without mutation.
        quantizer: Quantizer with resolved block_size and no tensor state.
    """
    config = quantizer.config
    assert config.block_size is not None
    rows, cols = config.block_size
    shape = tuple(weight.shape)
    fp8 = config.format == "fp8"
    decode = _fp8_value if fp8 else _fp4_value
    expected_codes = torch.empty(shape, dtype=torch.uint8)
    expected_values = torch.empty(shape, dtype=torch.float32)
    scale_dtype = torch.float32 if config.scale_format == "float32" else torch.uint8
    expected_scales = torch.empty((*shape[:-2], shape[-2] // rows, shape[-1] // cols), dtype=scale_dtype)
    for leading in itertools.product(*(range(size) for size in shape[:-2])):
        for row in range(0, shape[-2], rows):
            for col in range(0, shape[-1], cols):
                index = (*leading, slice(row, row + rows), slice(col, col + cols))
                source = weight[index].float().reshape(-1).tolist()
                amax = max(abs(value) for value in source)
                if config.scale_format == "float32":
                    stored_scale = scale = _f32(max(amax, _f32(1e-12)) / 448)
                else:
                    stored_scale = _scalar_scale(amax, 448 if fp8 else 6)
                    scale = math.ldexp(1, stored_scale - 127)
                codes = [_scalar_code(_f32(value / scale), fp8=fp8) for value in source]
                expected_codes[index] = torch.tensor(codes, dtype=torch.uint8).reshape(rows, cols)
                expected_values[index] = torch.tensor([decode(code) * scale for code in codes]).reshape(rows, cols)
                expected_scales[(*leading, row // rows, col // cols)] = stored_scale
    if fp8:
        expected_payload = expected_codes
    else:
        # Python packing, rather than the vectorized bitwise implementation.
        codes = expected_codes.reshape(-1).tolist()
        expected_payload = torch.tensor([lo + 16 * hi for lo, hi in zip(codes[::2], codes[1::2])], dtype=torch.uint8)
        expected_payload = expected_payload.reshape(*shape[:-1], shape[-1] // 2)
    encoded = quantizer.quantize(weight)
    _assert_bits(encoded.payload, expected_payload)
    _assert_bits(encoded.scales, expected_scales)
    _assert_bits(encoded.dequantize(dtype=weight.dtype), expected_values.to(weight.dtype))
    _assert_bits(quantizer(weight), expected_values.to(weight.dtype))


@pytest.mark.parametrize(
    "config, expected",
    [(WeightQuantizationConfig(), (128, 128)), (_FP8, (32, 32)), (_FP4, (1, 32)), (_FP8_FLOAT, (32, 32))],
)
def test_config_build_is_resolved_frozen_and_stateless(
    config: WeightQuantizationConfig, expected: tuple[int, int]
) -> None:
    original = asdict(config)
    quantizer = config.build()
    other = config.build()
    assert isinstance(quantizer, nn.Module)
    assert quantizer.config.block_size == expected
    assert quantizer.config is not config
    assert quantizer.config is not other.config
    assert asdict(config) == original
    with pytest.raises(FrozenInstanceError):
        assert hasattr(quantizer.config, "format")
        setattr(quantizer.config, "format", "mxfp4")
    assert not list(quantizer.parameters())
    assert not list(quantizer.buffers())
    assert not quantizer.state_dict()
    assert [field.name for field in fields(WeightQuantizationConfig)] == ["format", "block_size", "scale_format"]
    assert [field.name for field in fields(QuantizedWeight)] == ["payload", "scales", "config", "shape"]


def test_mxfp4_default_does_not_mutate_declarative_config() -> None:
    config = WeightQuantizationConfig(format="mxfp4")
    assert config.build().config == _FP4
    assert config.block_size is None


@pytest.mark.parametrize("format_name", ["FP8", "fp4", "e4m3", "nvfp4", "", None, 8])
def test_invalid_format(format_name: object) -> None:
    with pytest.raises(ValueError, match="format"):
        replace(WeightQuantizationConfig(), format=format_name)


@pytest.mark.parametrize("block", [(32,), (32, 32, 32), (32.0, 32), (True, 32), "32,32"])
def test_noncanonical_block_type(block: object) -> None:
    with pytest.raises(TypeError, match="block_size"):
        replace(_FP8, block_size=block)


@pytest.mark.parametrize("config", [_FP8, _FP4])
@pytest.mark.parametrize("block", [(0, 32), (-1, 32), (1, 16), (64, 64), (32, 128)])
def test_unsupported_block_size(config: WeightQuantizationConfig, block: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="block_size"):
        replace(config, block_size=block)


def test_format_specific_blocks_fail_closed() -> None:
    with pytest.raises(ValueError, match="block_size"):
        WeightQuantizationConfig(format="fp8", block_size=(1, 32))
    with pytest.raises(ValueError, match="block_size"):
        WeightQuantizationConfig(format="mxfp4", block_size=(32, 32))
    with pytest.raises(TypeError, match="config"):
        WeightFakeQuantizer(None)


def test_fp4_all_midpoints_and_neighbors_against_scalar_oracle() -> None:
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    values = torch.stack(
        (
            torch.nextafter(thresholds, torch.zeros_like(thresholds)),
            thresholds,
            torch.nextafter(thresholds, torch.full_like(thresholds, math.inf)),
        ),
        dim=-1,
    ).reshape(-1)
    # Include 6 in each group to anchor scale=1, and exercise both zero signs.
    positive = torch.cat((values, torch.tensor([0.0, -0.0]), torch.full((9,), 6.0)))
    weight = torch.stack((positive, -positive))
    _assert_scalar_oracle(weight, _FP4.build())
    tie_row = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 4).reshape(1, 32)
    expected = torch.tensor([[0x20, 0x42, 0x64, 0x76] * 4], dtype=torch.uint8)
    _assert_bits(_FP4.build().quantize(tie_row).payload, expected)


def test_fp8_known_bytes_and_midpoints() -> None:
    values = [0.0, -0.0, 1.0, -1.0, 448.0, -448.0, 2**-9, -(2**-9), 1.0625, 1.1875, 2**-10, 3 * 2**-10]
    weight = torch.tensor(values + [448.0] * (32 * 32 - len(values))).reshape(32, 32)
    encoded = _FP8.build().quantize(weight)
    assert encoded.payload.flatten()[:12].tolist() == [0x00, 0x80, 0x38, 0xB8, 0x7E, 0xFE, 1, 129, 56, 58, 0, 2]
    assert encoded.scales.item() == 127
    _assert_scalar_oracle(weight, _FP8.build())


@pytest.mark.parametrize("config", [_FP8, _FP4, _FP8_FLOAT])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_multi_expert_axes_and_strides_against_scalar_oracle(
    config: WeightQuantizationConfig, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(151)
    rows = 32 if config.format == "fp8" else 2
    source = torch.randn((2, 2, 64, rows), generator=generator, dtype=dtype)
    weight = source.transpose(-1, -2)
    # Distinct experts, row blocks, and column groups must never share an amax.
    weight[0, 0].mul_(0.03125)
    weight[1, 0].mul_(8)
    weight[..., :rows // 2, 32:].mul_(4)
    assert not weight.is_contiguous()
    before = weight.clone()
    _assert_scalar_oracle(weight, config.build())
    _assert_bits(weight, before)


def test_default_fp8_128_blocks_and_leading_axes() -> None:
    weight = torch.empty(2, 256, 256)
    expected_scales = torch.tensor([[[127, 128], [129, 130]], [[131, 132], [133, 134]]], dtype=torch.uint8)
    for expert, row, col in itertools.product(range(2), range(2), range(2)):
        exponent = int(expected_scales[expert, row, col]) - 127
        weight[expert, row * 128 : (row + 1) * 128, col * 128 : (col + 1) * 128] = math.ldexp(448, exponent)
    encoded = WeightQuantizationConfig().build().quantize(weight)
    _assert_bits(encoded.scales, expected_scales)
    assert bool((encoded.payload == 0x7E).all())
    _assert_bits(encoded.dequantize(dtype=torch.float32), weight)


@pytest.mark.parametrize("config", [_FP8, _FP4])
def test_exact_scale_boundaries_and_nextafter(config: WeightQuantizationConfig) -> None:
    maximum = 448 if config.format == "fp8" else 6
    exponents = [-127, -126, -100, -10, 0, 20, 110, 119]
    if config.format == "mxfp4":
        exponents.append(125)
    boundaries = torch.tensor([math.ldexp(maximum, exponent) for exponent in exponents])
    maxima = torch.stack(
        (
            torch.nextafter(boundaries, torch.zeros_like(boundaries)),
            boundaries,
            torch.nextafter(boundaries, torch.full_like(boundaries, math.inf)),
        ),
        dim=-1,
    ).flatten()
    assert config.block_size is not None
    weight = torch.zeros(len(maxima), *config.block_size)
    weight[:, 0, 0] = maxima
    encoded = config.build().quantize(weight)
    expected = torch.tensor([_scalar_scale(value, maximum) for value in maxima.tolist()], dtype=torch.uint8)
    _assert_bits(encoded.scales.flatten(), expected)
    assert encoded.scales.flatten()[1::3].tolist() == [exponent + 127 for exponent in exponents]
    assert encoded.scales.flatten()[2::3].tolist() == [exponent + 128 for exponent in exponents]


@pytest.mark.parametrize("config", [_FP8, _FP4])
def test_smallest_fp32_subnormals_saturate_scale(config: WeightQuantizationConfig) -> None:
    assert config.block_size is not None
    weight = torch.zeros(config.block_size)
    smallest = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    weight[0, 0], weight[0, 1] = smallest, -smallest
    encoded = config.build().quantize(weight)
    assert encoded.scales.item() == 0
    output = encoded.dequantize(dtype=torch.float32)
    assert bool((output == 0).all())
    assert not bool(torch.signbit(output[0, 0]))
    assert bool(torch.signbit(output[0, 1]))


@pytest.mark.parametrize("config", [_FP8, _FP4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_subnormal_inputs_compute_in_fp32(
    config: WeightQuantizationConfig, dtype: torch.dtype
) -> None:
    assert config.block_size is not None
    fraction_bits = 7 if dtype == torch.bfloat16 else 10
    smallest = math.ldexp(torch.finfo(dtype).tiny, -fraction_bits)
    weight = torch.full(config.block_size, smallest, dtype=dtype)
    weight[:, 1::2].neg_()
    _assert_scalar_oracle(weight, config.build())


@pytest.mark.parametrize("config", [_FP8, _FP4])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_zero_block_scale_and_signed_zero_ste(config: WeightQuantizationConfig, dtype: torch.dtype) -> None:
    assert config.block_size is not None
    weight = torch.zeros(config.block_size, dtype=dtype)
    weight[:, 1::2] = -0.0
    weight.requires_grad_()
    quantizer = config.build()
    encoded = quantizer.quantize(weight)
    assert encoded.scales.item() == 127
    if config.format == "mxfp4":
        assert bool((encoded.payload == 0x80).all())
    else:
        assert bool((encoded.payload[:, 0::2] == 0).all())
        assert bool((encoded.payload[:, 1::2] == 128).all())
    _assert_bits(quantizer(weight), weight)
    _assert_bits(encoded.dequantize(dtype=dtype), weight)


@pytest.mark.parametrize("scale_byte", [0, 1, 64, 127, 128, 246])
@pytest.mark.parametrize("config", [_FP8, _FP4])
def test_all_finite_payload_bytes_roundtrip(config: WeightQuantizationConfig, scale_byte: int) -> None:
    if config.format == "fp8":
        codes = [byte for byte in range(256) if byte not in (127, 255)]
        payload = torch.tensor((codes * 5)[:1024], dtype=torch.uint8).reshape(32, 32)
        shape = (32, 32)
        decoded = [_fp8_value(byte) for byte in payload.flatten().tolist()]
        scale_shape = (1, 1)
    else:
        payload = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(16, 16)
        shape = (16, 32)
        decoded = [value for byte in range(256) for value in (_fp4_value(byte % 16), _fp4_value(byte // 16))]
        scale_shape = (16, 1)
    scales = torch.full(scale_shape, scale_byte, dtype=torch.uint8)
    encoded = QuantizedWeight(payload=payload, scales=scales, config=config, shape=shape)
    expected = torch.tensor(
        [value * math.ldexp(1, scale_byte - 127) for value in decoded], dtype=torch.float32
    ).reshape(shape)
    output = encoded.dequantize(dtype=torch.float32)
    _assert_bits(output, expected)
    restored = QuantizedWeight(payload=payload.clone(), scales=scales.clone(), config=replace(config), shape=shape)
    _assert_bits(restored.dequantize(dtype=torch.float32), expected)
    requantized = config.build().quantize(output)
    _assert_bits(requantized.payload, payload)
    _assert_bits(requantized.scales, scales)


@pytest.mark.parametrize("config", [_FP8, _FP4, _FP8_FLOAT])
def test_noncontiguous_encoded_storage(config: WeightQuantizationConfig) -> None:
    generator = torch.Generator().manual_seed(31)
    encoded = config.build().quantize(torch.randn((2, 64, 64), generator=generator))
    payload_storage = torch.zeros((*encoded.payload.shape[:-1], encoded.payload.shape[-1] * 2), dtype=torch.uint8)
    payload_storage[..., ::2] = encoded.payload
    scale_storage = torch.zeros(
        (*encoded.scales.shape[:-1], encoded.scales.shape[-1] * 2), dtype=encoded.scales.dtype
    )
    scale_storage[..., ::2] = encoded.scales
    strided = replace(encoded, payload=payload_storage[..., ::2], scales=scale_storage[..., ::2])
    assert not strided.payload.is_contiguous()
    _assert_bits(strided.dequantize(dtype=torch.float32), encoded.dequantize(dtype=torch.float32))


@pytest.mark.parametrize("config", [_FP8, _FP4, _FP8_FLOAT])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_ste_exact_forward_and_random_identity_gradient(config: WeightQuantizationConfig, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn((2, 64, 32), dtype=dtype, generator=generator).transpose(-1, -2).detach().requires_grad_()
    upstream = torch.randn((2, 64, 32), dtype=dtype, generator=generator).transpose(-1, -2)
    quantizer = config.build()
    expected = quantizer.quantize(weight).dequantize(dtype=dtype)
    for _ in range(2):
        output = quantizer(weight)
        _assert_bits(output, expected)
        (gradient,) = torch.autograd.grad(output, weight, upstream)
        _assert_bits(gradient, upstream)
    assert not quantizer.quantize(weight).payload.requires_grad
    assert not quantizer.quantize(weight).scales.requires_grad


def test_ste_backpropagates_through_merged_weight() -> None:
    generator = torch.Generator().manual_seed(29)
    frozen = torch.randn((32, 32), generator=generator)
    update = torch.randn((32, 32), generator=generator, requires_grad=True)
    merged = frozen + 2 * update
    upstream = torch.randn((32, 32), generator=generator)
    output = _FP4.build()(merged)
    (gradient,) = torch.autograd.grad(output, update, upstream)
    _assert_bits(gradient, 2 * upstream)
    assert frozen.grad is None


@pytest.mark.parametrize("config", [_FP8, _FP4, _FP8_FLOAT])
def test_no_aliases_or_persistent_quantizer_state(config: WeightQuantizationConfig) -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn((32, 32), generator=generator)
    quantizer = config.build()
    encoded = quantizer.quantize(weight)
    before = weight.clone()
    payload_before, scales_before = encoded.payload.clone(), encoded.scales.clone()
    decoded = encoded.dequantize(dtype=torch.float32)
    decoded.zero_()
    quantizer(weight).zero_()
    _assert_bits(weight, before)
    _assert_bits(encoded.payload, payload_before)
    _assert_bits(encoded.scales, scales_before)
    weight.fill_(100)
    _assert_bits(encoded.payload, payload_before)
    _assert_bits(encoded.scales, scales_before)
    encoded.payload.zero_()
    _assert_bits(weight, torch.full_like(weight, 100))
    assert not quantizer.state_dict()
    quantizer.to(dtype=torch.bfloat16)
    _assert_bits(encoded.scales, scales_before)
    assert encoded.payload.dtype == torch.uint8
    assert encoded.scales.dtype == (torch.float32 if config.scale_format == "float32" else torch.uint8)
    assert not list(quantizer.parameters())
    assert not list(quantizer.buffers())


def test_byte_buffers_survive_module_dtype_casts() -> None:
    encoded = _FP4.build().quantize(torch.arange(32, dtype=torch.float32).reshape(1, 32))
    owner = nn.Module()
    owner.register_buffer("payload", encoded.payload)
    owner.register_buffer("scales", encoded.scales)
    before = {name: value.clone() for name, value in owner.state_dict().items()}
    for dtype in _DTYPES:
        owner.to(dtype=dtype)
        for name, value in owner.state_dict().items():
            _assert_bits(value, before[name])


@pytest.mark.parametrize("config", [_FP8, _FP4])
@pytest.mark.parametrize("nonfinite", [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_reject_nonfinite(config: WeightQuantizationConfig, nonfinite: float, dtype: torch.dtype) -> None:
    weight = torch.zeros((32, 32), dtype=dtype)
    weight[0, 0] = nonfinite
    quantizer = config.build()
    with pytest.raises(ValueError, match="finite"):
        quantizer.quantize(weight)
    with pytest.raises(ValueError, match="finite"):
        quantizer(weight)


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.int32, torch.uint8, torch.bool, torch.complex64, torch.float8_e4m3fn]
)
def test_reject_weight_dtype(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match="dtype"):
        _FP8.build().quantize(torch.zeros((32, 32), dtype=dtype))


@pytest.mark.parametrize("shape", [(), (32,), (32, 31), (31, 32), (0, 32), (32, 0), (0, 32, 32)])
def test_reject_weight_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises((TypeError, ValueError), match="shape"):
        _FP8.build().quantize(torch.zeros(shape))


def test_reject_unsupported_tensor_storage() -> None:
    with pytest.raises(TypeError, match="strided"):
        _FP8.build().quantize(torch.zeros(32, 32).to_sparse())
    with pytest.raises(ValueError, match="CPU or CUDA"):
        _FP8.build().quantize(torch.empty(32, 32, device="meta"))
    with pytest.raises(TypeError, match="torch.Tensor"):
        _FP8.build().quantize(None)


@pytest.mark.parametrize("field", ["payload", "scales"])
def test_reject_encoded_storage_dtype(field: str) -> None:
    encoded = _FP8.build().quantize(torch.zeros(32, 32))
    with pytest.raises(TypeError, match=field):
        replace(encoded, **{field: getattr(encoded, field).float()})


@pytest.mark.parametrize("field", ["payload", "scales"])
def test_reject_encoded_storage_shape(field: str) -> None:
    encoded = _FP8.build().quantize(torch.zeros(32, 32))
    with pytest.raises(ValueError, match=f"{field} shape"):
        replace(encoded, **{field: getattr(encoded, field).flatten()})


@pytest.mark.parametrize("shape", [[32, 32], (32,), (32, True), (0, 32), (31, 32)])
def test_reject_encoded_logical_shape(shape: object) -> None:
    encoded = _FP8.build().quantize(torch.zeros(32, 32))
    with pytest.raises((TypeError, ValueError), match="shape"):
        replace(encoded, shape=shape)


def test_reject_unresolved_encoded_config() -> None:
    encoded = _FP8.build().quantize(torch.zeros(32, 32))
    with pytest.raises(TypeError, match="resolved"):
        replace(encoded, config=WeightQuantizationConfig())
    with pytest.raises(TypeError, match="config"):
        replace(encoded, config=None)


@pytest.mark.parametrize("byte", [127, 255])
def test_reject_fp8_nan_codes_and_revalidate_mutation(byte: int) -> None:
    encoded = _FP8.build().quantize(torch.zeros(32, 32))
    payload = encoded.payload.clone()
    payload[0, 0] = byte
    with pytest.raises(ValueError, match="E4M3FN"):
        replace(encoded, payload=payload)
    encoded.payload[0, 0] = byte
    with pytest.raises(ValueError, match="E4M3FN"):
        encoded.dequantize(dtype=torch.float32)


@pytest.mark.parametrize("config", [_FP8, _FP4])
def test_reject_e8m0_nan_byte_and_revalidate_mutation(config: WeightQuantizationConfig) -> None:
    encoded = config.build().quantize(torch.zeros(32, 32))
    with pytest.raises(ValueError, match="E8M0"):
        replace(encoded, scales=torch.full_like(encoded.scales, 255))
    encoded.scales.fill_(255)
    with pytest.raises(ValueError, match="E8M0"):
        encoded.dequantize(dtype=torch.float32)


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.int32, torch.uint8, torch.bool, torch.complex64, None, "float32"]
)
def test_reject_output_dtype(dtype: object) -> None:
    encoded = _FP8.build().quantize(torch.zeros(32, 32))
    with pytest.raises(TypeError, match="dtype"):
        encoded.dequantize(dtype=dtype)


def test_dequantize_dtype_is_keyword_only() -> None:
    encoded = _FP8.build().quantize(torch.zeros(32, 32))
    with pytest.raises(TypeError):
        encoded.dequantize(torch.float32)


@pytest.mark.parametrize("config", [_FP8, _FP4])
def test_highest_valid_e8m0_exponent(config: WeightQuantizationConfig) -> None:
    encoded = config.build().quantize(torch.zeros(32, 32))
    payload = torch.full_like(encoded.payload, 0x38 if config.format == "fp8" else 0x22)
    encoded = replace(encoded, payload=payload, scales=torch.full_like(encoded.scales, 254))
    _assert_bits(encoded.dequantize(dtype=torch.float32), torch.full((32, 32), 2.0**127, dtype=torch.float32))


@pytest.mark.parametrize("config", [_FP8, _FP4])
def test_decode_rejects_overflow_but_not_underflow(config: WeightQuantizationConfig) -> None:
    encoded = config.build().quantize(torch.zeros(32, 32))
    payload = torch.full_like(encoded.payload, 0x7E if config.format == "fp8" else 0x77)
    overflow = replace(encoded, payload=payload, scales=torch.full_like(encoded.scales, 254))
    with pytest.raises(ValueError, match="overflow"):
        overflow.dequantize(dtype=torch.float32)
    fp16_overflow = replace(encoded, payload=payload, scales=torch.full_like(encoded.scales, 143))
    with pytest.raises(ValueError, match="overflow"):
        fp16_overflow.dequantize(dtype=torch.float16)
    tiny = replace(encoded, payload=payload, scales=torch.zeros_like(encoded.scales))
    assert bool((tiny.dequantize(dtype=torch.float16) == 0).all())


@pytest.mark.parametrize("config", [_FP8, _FP4])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_finite_input_with_unrepresentable_qdq_fails_closed(
    config: WeightQuantizationConfig, dtype: torch.dtype
) -> None:
    weight = torch.full((32, 32), torch.finfo(dtype).max, dtype=dtype)
    encoded = config.build().quantize(weight)
    assert bool((encoded.scales < 255).all())
    with pytest.raises(ValueError, match="overflow"):
        encoded.dequantize(dtype=dtype)
    with pytest.raises(ValueError, match="overflow"):
        config.build()(weight)


def test_expanded_input_is_not_mutated() -> None:
    weight = torch.tensor([[1.25]], requires_grad=True).expand(32, 32)
    _assert_scalar_oracle(weight, _FP8.build())
    _assert_bits(weight, torch.full((32, 32), 1.25))


@pytest.mark.parametrize("scale_format", ["FLOAT32", "fp32", "ue8m0", "", None, 32, True, {}])
def test_invalid_scale_format(scale_format: object) -> None:
    with pytest.raises(ValueError, match="scale_format"):
        replace(_FP8, scale_format=scale_format)


@pytest.mark.parametrize("block_size", [None, (1, 32)])
def test_mxfp4_rejects_float32_scales_even_with_unresolved_block(block_size) -> None:
    with pytest.raises(ValueError, match="mxfp4 requires"):
        WeightQuantizationConfig("mxfp4", block_size, scale_format="float32")


def test_float32_scales_do_not_round_to_powers_of_two() -> None:
    weight = torch.full((32, 32), 3.0)
    quantizer = _FP8_FLOAT.build()
    encoded = quantizer.quantize(weight)
    assert encoded.scales.dtype == torch.float32
    assert encoded.scales.item() == _f32(3.0 / 448)
    assert math.frexp(encoded.scales.item())[0] != 0.5
    assert bool((encoded.payload == 0x7E).all())
    _assert_bits(encoded.dequantize(dtype=torch.float32), weight)
    assert not torch.equal(encoded.payload, _FP8.build().quantize(weight).payload)
    _assert_scalar_oracle(weight, quantizer)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_float32_floor_zero_subnormal_and_boundary_scalar_oracle(dtype: torch.dtype) -> None:
    # This locks the requested 1e-12 floor, not every upstream re-quantizer.
    # Verified Transformers 5.15.1 Fp8Quantize uses 1/(448/amax) with no floor
    # and scale=1 for zero blocks: at amax=1e-14 its scale is ~2.232e-17,
    # versus ~2.232e-15 here. Reciprocal rounding can also differ by one ULP
    # for ordinary blocks (e.g. amax=3). The GLM decoder accepts both; the
    # released checkpoint's FP32-scale layout does not specify its quantizer.
    floor = torch.tensor(1e-12)
    maxima = torch.tensor([
        0.0, torch.finfo(torch.float32).tiny, 1e-14,
        torch.nextafter(floor, torch.tensor(0.0)).item(), floor.item(),
        torch.nextafter(floor, torch.tensor(math.inf)).item(), 1e-10,
    ], dtype=dtype)
    weight = maxima[:, None, None].expand(-1, 32, 32).clone()
    weight[:, :, 1::2].neg_()
    weight.requires_grad_()
    quantizer = _FP8_FLOAT.build()
    _assert_scalar_oracle(weight, quantizer)
    encoded = quantizer.quantize(weight)
    assert encoded.scales[0].item() == _f32(_f32(1e-12) / 448)
    assert bool(torch.signbit(quantizer(weight)[0, :, 1::2]).all())
    quantizer(weight).sum().backward()
    _assert_bits(weight.grad, torch.ones_like(weight))


@pytest.mark.parametrize("value", [0.0, -0.0, -1.0, math.inf, -math.inf, math.nan])
def test_float32_scales_reject_invalid_values_and_revalidate_mutation(value: float) -> None:
    encoded = _FP8_FLOAT.build().quantize(torch.ones(32, 32))
    with pytest.raises(ValueError, match="finite and strictly positive"):
        replace(encoded, scales=torch.full_like(encoded.scales, value))
    encoded.scales.fill_(value)
    with pytest.raises(ValueError, match="finite and strictly positive"):
        encoded.dequantize(dtype=torch.float32)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16, torch.float64])
def test_float32_scale_storage_requires_exact_dtype(dtype: torch.dtype) -> None:
    encoded = _FP8_FLOAT.build().quantize(torch.ones(32, 32))
    with pytest.raises(TypeError, match="scales.*dtype"):
        replace(encoded, scales=encoded.scales.to(dtype))


def test_float32_scales_accept_255_and_decode_before_low_precision_cast() -> None:
    encoded = _FP8_FLOAT.build().quantize(torch.ones(32, 32))
    payload = torch.full_like(encoded.payload, 0x38)  # E4M3 1.0
    valid = replace(encoded, payload=payload, scales=torch.full_like(encoded.scales, 255.0))
    _assert_bits(valid.dequantize(dtype=torch.float32), torch.full((32, 32), 255.0))
    scale = _f32(0.001234567)
    valid = replace(encoded, payload=torch.full_like(payload, 0x7E), scales=torch.full_like(encoded.scales, scale))
    expected = torch.full((32, 32), _f32(448 * scale))
    for dtype in _DTYPES:
        _assert_bits(valid.dequantize(dtype=dtype), expected.to(dtype))
    overflow = replace(valid, scales=torch.full_like(encoded.scales, torch.finfo(torch.float32).max))
    with pytest.raises(ValueError, match="overflow"):
        overflow.dequantize(dtype=torch.float32)


@pytest.mark.parametrize("shape", [(2048, 4096), (4096, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_glm_block_fp8_decoder_matches_packed_actual_expert_shapes(shape, dtype) -> None:
    from nemo_automodel.components.models.glm5_next.state_dict_adapter import dequantize_block_fp8

    # Actual released gate/up and down matrix dimensions, not a square crop.
    generator = torch.Generator().manual_seed(53)
    weight = torch.randn(shape, generator=generator, dtype=dtype)
    weight[:128, :128].zero_()
    quantizer = WeightQuantizationConfig(scale_format="float32").build()
    encoded = quantizer.quantize(weight)
    assert encoded.payload.shape == shape
    assert encoded.scales.shape == (shape[0] // 128, shape[1] // 128)
    assert encoded.payload.dtype == torch.uint8 and encoded.scales.dtype == torch.float32
    # GLM consumes E4M3 values and FP32 weight_scale_inv directly.
    actual = dequantize_block_fp8(encoded.payload.view(torch.float8_e4m3fn), encoded.scales, dtype=dtype)
    _assert_bits(actual, encoded.dequantize(dtype=dtype))
    _assert_bits(actual, quantizer(weight))


def test_real_dtensor_is_explicitly_rejected(tmp_path: Path) -> None:
    # One CPU rank is enough to exercise actual DTensor dispatch rejection;
    # no mocked subclass or distributed quantization behavior is involved.
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0])
        weight = DTensor.from_local(torch.zeros(32, 32), mesh, [Replicate()], run_check=False)
        with pytest.raises(TypeError, match="DTensor"):
            _FP8.build().quantize(weight)
        with pytest.raises(TypeError, match="DTensor"):
            _FP8.build()(weight)
        encoded = _FP8.build().quantize(torch.zeros(32, 32))
        payload = DTensor.from_local(encoded.payload, mesh, [Replicate()], run_check=False)
        with pytest.raises(TypeError, match="DTensor"):
            replace(encoded, payload=payload)
        scales = DTensor.from_local(encoded.scales, mesh, [Replicate()], run_check=False)
        with pytest.raises(TypeError, match="DTensor"):
            replace(encoded, scales=scales)
    finally:
        dist.destroy_process_group()