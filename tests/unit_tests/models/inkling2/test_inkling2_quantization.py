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

"""Inkling 2 NVFP4 / MXFP8 expert dequantization against hand-computed values.

Expected values are written out from the format description (E2M1 code table, E4M3 and UE8M0 bit layouts), not
from the implementation, so a nibble-order, block-index, scale-format or rounding-order error fails here.
"""

import pytest
import torch

import nemo_automodel.components.models.inkling2.quantization as quantization
from nemo_automodel.components.models.inkling2.quantization import (
    dequantize_mxfp8_experts,
    dequantize_nvfp4_experts,
)

pytestmark = pytest.mark.cpu_only

# E2M1 value of each 4-bit code: bits 0-2 index the magnitude, bit 3 is the sign.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
E4M3_ONE = 0x38  # exponent 7 (bias 7), mantissa 0
DENOMINATOR = 2688.0  # 6 * 448, the released checkpoint's outer-scale denominator


def _u8(values: list[int], shape: tuple[int, ...]) -> torch.Tensor:
    """Build a uint8 tensor of ``shape`` from row-major byte values."""
    return torch.tensor(values, dtype=torch.uint8).reshape(shape)


def _e4m3(values: list[int], shape: tuple[int, ...]) -> torch.Tensor:
    """Build a float8_e4m3fn tensor of ``shape`` from row-major E4M3 bit patterns."""
    return _u8(values, shape).view(torch.float8_e4m3fn)


def _nvfp4(weight: torch.Tensor, scale: torch.Tensor, amax: list[float], denominator: float = DENOMINATOR, **kwargs):
    """Dequantize with float32 amax / denominator tensors built from Python floats.

    Args:
        weight: uint8 tensor of shape [experts, out, in / 2], two E2M1 codes per byte (low nibble first).
        scale: float8_e4m3fn tensor of shape [experts, out, in / 16].
        amax: One value per expert.
        denominator: Outer-scale denominator shared by the experts.
        **kwargs: ``device`` / ``dtype`` passed through.

    Returns:
        Tensor of shape [experts, out, in].
    """
    return dequantize_nvfp4_experts(
        weight,
        scale,
        torch.tensor(amax, dtype=torch.float32),
        torch.tensor([denominator], dtype=torch.float32),
        **kwargs,
    )


def test_nvfp4_decodes_every_code_with_low_nibble_first():
    # Byte k holds code k in the low nibble (column 2k) and code 15 - k in the high nibble (column 2k + 1).
    weight = _u8([k | ((15 - k) << 4) for k in range(16)], (1, 1, 16))
    scale = _e4m3([E4M3_ONE, E4M3_ONE], (1, 1, 2))
    got = _nvfp4(weight, scale, [DENOMINATOR])  # outer scale exactly 1
    expected = [
        0.0, -6.0, 0.5, -4.0, 1.0, -3.0, 1.5, -2.0, 2.0, -1.5, 3.0, -1.0, 4.0, -0.5, 6.0, -0.0,
        -0.0, 6.0, -0.5, 4.0, -1.0, 3.0, -1.5, 2.0, -2.0, 1.5, -3.0, 1.0, -4.0, 0.5, -6.0, 0.0,
    ]  # fmt: skip
    assert got.dtype == torch.float32 and got.shape == (1, 1, 32)
    assert got.flatten().tolist() == expected


def test_nvfp4_e4m3_block_scales_including_subnormals():
    # Six 16-column blocks per row; row 0 holds code 7 (+6) everywhere, row 1 code 15 (-6).
    weight = _u8([0x77] * 48 + [0xFF] * 48, (1, 2, 48))
    # zero, smallest subnormal 2^-9, largest subnormal 7 * 2^-9, smallest normal 2^-6, 1.5, largest finite 448
    scale_bytes = [0x00, 0x01, 0x07, 0x08, 0x3C, 0x7E]
    scale = _e4m3(scale_bytes * 2, (1, 2, 6))
    got = _nvfp4(weight, scale, [DENOMINATOR])
    block_values = [0.0, 0.01171875, 0.08203125, 0.09375, 9.0, 2688.0]  # 6 * scale
    assert got[0, 0].view(6, 16).tolist() == [[v] * 16 for v in block_values]
    assert got[0, 1].view(6, 16).tolist() == [[-v] * 16 for v in block_values]


def test_nvfp4_outer_scale_is_per_expert_amax_over_denominator():
    weight = _u8([0x22] * 8 * 3, (3, 1, 8))  # code 2 (+1.0) everywhere
    scale = _e4m3([E4M3_ONE] * 3, (3, 1, 1))
    assert _nvfp4(weight, scale, [2688.0, 1344.0, 0.0]).flatten(1).tolist() == [[1.0] * 16, [0.5] * 16, [0.0] * 16]
    halved = _nvfp4(weight, scale, [2688.0, 1344.0, 0.0], denominator=1344.0)
    assert halved.flatten(1).tolist() == [[2.0] * 16, [1.0] * 16, [0.0] * 16]


def test_nvfp4_rounds_the_exact_code_scale_product_once():
    # Every code against a spread of E4M3 scales, with amax values whose amax / 2688 is not a power of two.
    codes = torch.arange(16, dtype=torch.uint8).repeat(8)  # 128 columns
    weight = (codes[0::2] | (codes[1::2] << 4)).reshape(1, 1, 64).expand(4, 7, 64).contiguous()
    scale_bytes = torch.tensor([0x01, 0x05, 0x0B, 0x29, 0x3B, 0x4D, 0x63], dtype=torch.uint8)
    scale = scale_bytes.view(1, 7, 1).expand(4, 7, 8).contiguous().view(torch.float8_e4m3fn)
    amax = [0.1, 0.0812446, 0.7217177, 3.3]
    got = _nvfp4(weight, scale, amax)

    # Reference in float64: code * scale * fp32(amax / denominator) is exact there, rounded to float32 once.
    outer = torch.tensor(amax, dtype=torch.float32) / torch.tensor(DENOMINATOR, dtype=torch.float32)
    code_values = torch.tensor([E2M1[int(c)] for c in codes], dtype=torch.float64)
    scale_values = scale_bytes.view(torch.float8_e4m3fn).to(torch.float64)
    exact = code_values.view(1, 1, 128) * scale_values.view(1, 7, 1) * outer.to(torch.float64).view(4, 1, 1)
    assert torch.equal(got, exact.to(torch.float32))

    # The order matters: folding the outer scale into the block scale first rounds twice and differs somewhere.
    folded = code_values.float().view(1, 1, 128) * (scale_values.float().view(1, 7, 1) * outer.view(4, 1, 1))
    assert not torch.equal(folded, got)


def test_nvfp4_indexes_scales_by_expert_row_and_block():
    experts, rows, columns = 2, 3, 32
    weight = _u8([0x22] * (experts * rows * columns // 2), (experts, rows, columns // 2))  # +1.0 everywhere
    scale_bytes = [0x30 + i for i in range(experts * rows * 2)]  # distinct normal E4M3 per (expert, row, block)
    scale = _e4m3(scale_bytes, (experts, rows, 2))
    got = _nvfp4(weight, scale, [2688.0, 5376.0])  # outer scales 1 and 2
    e4m3_value = {b: (1 + (b & 7) / 8) * 2.0 ** ((b >> 3) - 7) for b in scale_bytes}
    for e in range(experts):
        for o in range(rows):
            for i in range(columns):
                expected = e4m3_value[scale_bytes[(e * rows + o) * 2 + i // 16]] * (1.0, 2.0)[e]
                assert got[e, o, i].item() == expected


def test_nvfp4_chunked_decode_matches_one_pass(monkeypatch):
    generator = torch.Generator().manual_seed(0)
    weight = torch.randint(0, 256, (3, 5, 24), generator=generator, dtype=torch.uint8)
    scale = torch.randint(0, 0x7F, (3, 5, 3), generator=generator, dtype=torch.uint8).view(torch.float8_e4m3fn)
    amax = [0.25, 0.5, 0.75]
    whole = _nvfp4(weight, scale, amax)
    for chunk in (16, 48, 100):  # one row per step, rows straddling experts, uneven tail
        monkeypatch.setattr(quantization, "_DEQUANT_CHUNK_ELEMS", chunk)
        assert torch.equal(_nvfp4(weight, scale, amax), whole)


def test_nvfp4_bfloat16_output_is_the_float32_value_rounded_once():
    generator = torch.Generator().manual_seed(1)
    weight = torch.randint(0, 256, (2, 4, 32), generator=generator, dtype=torch.uint8)
    scale = torch.randint(0x08, 0x7F, (2, 4, 4), generator=generator, dtype=torch.uint8).view(torch.float8_e4m3fn)
    fp32 = _nvfp4(weight, scale, [0.3, 0.07])
    bf16 = _nvfp4(weight, scale, [0.3, 0.07], dtype=torch.bfloat16)
    assert bf16.dtype == torch.bfloat16 and torch.equal(bf16, fp32.to(torch.bfloat16))


def test_nvfp4_leaves_inputs_unchanged():
    weight = torch.arange(64, dtype=torch.uint8).reshape(2, 2, 16)
    scale = _e4m3([0x38, 0x40] * 4, (2, 2, 2))
    amax = torch.tensor([1.0, 2.0])
    denominator = torch.tensor([DENOMINATOR])
    snapshots = [t.view(torch.uint8).clone() if t.dtype != torch.float32 else t.clone() for t in (weight, scale)]
    dequantize_nvfp4_experts(weight, scale, amax, denominator)
    assert torch.equal(weight, snapshots[0]) and torch.equal(scale.view(torch.uint8), snapshots[1])
    assert amax.tolist() == [1.0, 2.0] and denominator.tolist() == [DENOMINATOR]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"weight": torch.zeros(1, 2, 8, dtype=torch.int8)}, "uint8 weight"),
        ({"scale": torch.ones(1, 2, 1, dtype=torch.bfloat16)}, "float8_e4m3fn scale"),
        ({"amax": torch.ones(1, dtype=torch.float64)}, "float32"),
        ({"weight": torch.zeros(1, 2, 7, dtype=torch.uint8)}, "divisible by 16"),
        ({"scale": torch.ones(1, 2, 2).to(torch.float8_e4m3fn)}, "scale shape"),
        ({"amax": torch.ones(2)}, "amax must be"),
        ({"denominator": torch.ones(())}, "denominator"),
        ({"amax": torch.tensor([-1.0])}, "non-negative"),
        ({"amax": torch.tensor([float("inf")])}, "finite"),
        ({"denominator": torch.tensor([0.0])}, "positive"),
        ({"scale": _e4m3([0x38, 0x7F], (1, 2, 1))}, "NaN"),
        ({"dtype": torch.int32}, "floating-point"),
    ],
)
def test_nvfp4_rejects_invalid_inputs(change, message):
    """Each override replaces one valid input: weight uint8 [1, 2, 8], scale float8_e4m3fn [1, 2, 1], amax float32
    [1], denominator float32 [1]; overrides use the same axis order ([experts, out, in / 2], [experts, out, in / 16],
    [experts], [1]) with a wrong dtype, shape or value."""
    inputs = {
        "weight": torch.zeros(1, 2, 8, dtype=torch.uint8),
        "scale": _e4m3([E4M3_ONE, E4M3_ONE], (1, 2, 1)),
        "amax": torch.ones(1),
        "denominator": torch.tensor([DENOMINATOR]),
        "dtype": torch.float32,
    }
    inputs.update(change)
    with pytest.raises(ValueError, match=message):
        dequantize_nvfp4_experts(
            inputs["weight"], inputs["scale"], inputs["amax"], inputs["denominator"], dtype=inputs["dtype"]
        )


def test_mxfp8_ue8m0_exponents_and_e4m3_values():
    # Row 0: weight 1.0 everywhere, one UE8M0 byte per 32-column block, from the smallest (2^-127) to 2^127.
    # Row 1: signed, subnormal and largest E4M3 values under exponent 128 (scale 2).
    exponents = [0, 1, 127, 128, 254]
    row1_values = [0x01, 0x07, 0x08, 0x7E, 0x81, 0xFE, 0x00, 0x80]  # 2^-9, 7*2^-9, 2^-6, 448, negatives, +-0
    weight = _e4m3([E4M3_ONE] * 160 + row1_values * 20, (1, 2, 160))
    scale = _u8(exponents + [128] * 5, (1, 2, 5))
    got = dequantize_mxfp8_experts(weight, scale)
    assert got.dtype == torch.float32
    assert got[0, 0].view(5, 32).tolist() == [[v] * 32 for v in (2.0**-127, 2.0**-126, 1.0, 2.0, 2.0**127)]
    row1 = [2.0**-8, 7 * 2.0**-8, 2.0**-5, 896.0, -(2.0**-8), -896.0, 0.0, -0.0]
    assert got[0, 1].tolist() == row1 * 20
    # The smallest E4M3 subnormal under the smallest exponent stays exact in float32 (2^-136).
    tiny = dequantize_mxfp8_experts(_e4m3([0x01] * 32, (1, 1, 32)), _u8([0], (1, 1, 1)))
    assert tiny.flatten().tolist() == [2.0**-136] * 32


def test_mxfp8_decodes_each_row_with_its_own_exponent():
    # 64 rows: exponents vary inside each 32-row block, so a [32, 32]-block reading of the scale would fail.
    rows = 64
    weight = _e4m3([0x3C] * (rows * 64), (1, rows, 64))  # 1.5
    exponents = [120 + (r % 7) for r in range(rows) for _ in range(2)]
    got = dequantize_mxfp8_experts(weight, _u8(exponents, (1, rows, 2)))
    for r in range(rows):
        assert got[0, r].tolist() == [1.5 * 2.0 ** (120 + (r % 7) - 127)] * 64


def test_mxfp8_chunked_and_bfloat16_outputs(monkeypatch):
    generator = torch.Generator().manual_seed(2)
    weight = torch.randint(0, 256, (3, 4, 64), generator=generator, dtype=torch.uint8)
    weight[(weight & 0x7F) == 0x7F] = 0  # drop the E4M3 NaN patterns
    weight = weight.view(torch.float8_e4m3fn)
    scale = torch.randint(100, 150, (3, 4, 2), generator=generator, dtype=torch.uint8)
    whole = dequantize_mxfp8_experts(weight, scale)
    torch.testing.assert_close(whole, weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, -1))
    assert torch.equal(dequantize_mxfp8_experts(weight, scale, dtype=torch.bfloat16), whole.to(torch.bfloat16))
    for chunk in (64, 192, 300):
        monkeypatch.setattr(quantization, "_DEQUANT_CHUNK_ELEMS", chunk)
        assert torch.equal(dequantize_mxfp8_experts(weight, scale), whole)


@pytest.mark.parametrize(
    ("weight", "scale", "message"),
    [
        (torch.zeros(1, 32, 64, dtype=torch.uint8), torch.zeros(1, 32, 2, dtype=torch.uint8), "float8_e4m3fn weight"),
        (torch.zeros(1, 32, 64).to(torch.float8_e4m3fn), torch.zeros(1, 32, 2).to(torch.float8_e4m3fn), "uint8 scale"),
        (torch.zeros(1, 32, 48).to(torch.float8_e4m3fn), torch.zeros(1, 32, 2, dtype=torch.uint8), "divisible by 32"),
        # a compact [experts, out / 32, in / 32] grid is not the released layout
        (torch.zeros(1, 32, 64).to(torch.float8_e4m3fn), torch.zeros(1, 1, 2, dtype=torch.uint8), "one per row"),
        (torch.zeros(1, 1, 32).to(torch.float8_e4m3fn), torch.full((1, 1, 1), 255, dtype=torch.uint8), "NaN"),
    ],
)
def test_mxfp8_rejects_invalid_inputs(weight, scale, message):
    """weight: tensor of shape [experts, out, in]; scale: tensor of shape [experts, out, in / 32] when valid. Each
    case breaks one of the dtype, divisibility, per-row scale shape or finite-exponent requirements."""
    with pytest.raises(ValueError, match=message):
        dequantize_mxfp8_experts(weight, scale)
