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

import pytest
import torch

import nemo_automodel.components.quantization.nvfp4 as nvfp4_module
from nemo_automodel.components.quantization.mxfp4 import _FP4_E2M1_TABLE
from nemo_automodel.components.quantization.nvfp4 import (
    E2M1_MAX,
    E4M3_MAX,
    NVFP4_BLOCK_SIZE,
    NVFP4GroupedMM,
    dequantize_nvfp4,
    quantize_nvfp4,
    quantize_nvfp4_modelopt,
)

E, N, K = 3, 16, 128  # [experts, out, in]; K blockable by 16 and 16-byte aligned for the grouped GEMM


def _grouped_mm_available(device: torch.device) -> bool:
    x = torch.randn(4, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(2, K, N, dtype=torch.bfloat16, device=device)
    try:
        torch._grouped_mm(x, w, offs=torch.tensor([2, 4], dtype=torch.int32, device=device))
    except (RuntimeError, NotImplementedError):
        return False
    return True


def test_storage_layout_is_half_byte_per_param_plus_block_and_row_scales():
    w = torch.randn(E, N, K, dtype=torch.bfloat16)
    packed, scales, global_scale = quantize_nvfp4(w)
    assert packed.dtype == torch.uint8 and packed.shape == (E, N, K // 2)
    assert scales.dtype == torch.float8_e4m3fn and scales.shape == (E, N, K // NVFP4_BLOCK_SIZE)
    assert global_scale.dtype == torch.float32 and global_scale.shape == (E, N)
    assert packed.numel() + scales.numel() == w.numel() * 0.5625


def test_last_dim_must_be_blockable():
    with pytest.raises(ValueError, match="divisible"):
        quantize_nvfp4(torch.randn(2, 8, 40))


def test_modelopt_layout_decodes_even_column_from_the_low_nibble():
    """The ModelOpt / TensorRT-LLM packing keeps column 2i in the low nibble of byte i."""
    codes = torch.tensor([[1, 7, 9, 0]], dtype=torch.uint8)  # 0.5, 6.0, -0.5, 0.0
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = torch.ones(1, 1, dtype=torch.float32).to(torch.float8_e4m3fn)
    packed = torch.cat([packed, torch.zeros(1, NVFP4_BLOCK_SIZE // 2 - 2, dtype=torch.uint8)], dim=-1)
    out = dequantize_nvfp4(packed, scales, torch.ones(1), torch.float32)
    assert out[0, :4].tolist() == [0.5, 6.0, -0.5, 0.0]


def test_values_on_the_e2m1_grid_round_trip_exactly():
    torch.manual_seed(0)
    n_blocks = K // NVFP4_BLOCK_SIZE
    codes = torch.randint(0, 16, (E, N, n_blocks, NVFP4_BLOCK_SIZE))
    codes[..., 0] = 7  # a +6.0 code in every block, so amax(block) / 6 recovers the block scale exactly
    # Power-of-two block scales with the row maximum fixed at 4: the row scale becomes 4 / 448 and every
    # recovered block scale is 2^e * 112 = 7 * 2^(e+4), representable in e4m3 (3 mantissa bits).
    exponents = torch.randint(-3, 3, (E, N, n_blocks))
    exponents[..., 0] = 2
    scales = 2.0 ** exponents.float()
    w = (_FP4_E2M1_TABLE[codes] * scales.unsqueeze(-1)).reshape(E, N, K)
    packed, q_scales, global_scale = quantize_nvfp4(w)
    torch.testing.assert_close(global_scale, torch.full((E, N), 4.0 / 448.0), rtol=0, atol=0)
    torch.testing.assert_close(q_scales.float(), scales * 112.0, rtol=0, atol=0)
    torch.testing.assert_close(dequantize_nvfp4(packed, q_scales, global_scale, torch.float32), w, rtol=1e-6, atol=1e-6)


def test_modelopt_quantization_uses_one_tensor_scale():
    torch.manual_seed(0)
    w = torch.randn(N, K, dtype=torch.bfloat16) * 3
    packed, scales, tensor_scale = quantize_nvfp4_modelopt(w)
    assert tensor_scale.shape == () and tensor_scale.dtype == torch.float32
    torch.testing.assert_close(tensor_scale, w.float().abs().amax() / (E2M1_MAX * E4M3_MAX), rtol=0, atol=0)
    assert packed.shape == (N, K // 2) and scales.shape == (N, K // NVFP4_BLOCK_SIZE)
    dq = dequantize_nvfp4(packed, scales, tensor_scale.expand(N), torch.float32)
    err = (dq - w.float()).abs()
    scale = scales.float() * tensor_scale
    assert torch.all(err.view(N, -1, NVFP4_BLOCK_SIZE) <= 1.0 * scale.unsqueeze(-1) + 1e-6)
    # The same codes and block scales come out of the row-wise API when the tensor scale is passed in.
    packed2, scales2, rows = quantize_nvfp4(w, global_scale=tensor_scale)
    assert torch.equal(packed2, packed) and torch.equal(scales2.view(torch.uint8), scales.view(torch.uint8))
    assert torch.equal(rows, tensor_scale.expand(N))


def test_chunked_dequantization_matches_single_chunk(monkeypatch):
    torch.manual_seed(0)
    packed, scales, global_scale = quantize_nvfp4(torch.randn(E, N, K, dtype=torch.bfloat16))
    whole = dequantize_nvfp4(packed, scales, global_scale, torch.bfloat16)
    monkeypatch.setattr(nvfp4_module, "_DEQUANT_CHUNK_ELEMS", 5 * K)  # five rows per chunk, rows not divisible by it
    chunked = dequantize_nvfp4(packed, scales, global_scale, torch.bfloat16)
    assert torch.equal(whole, chunked)


def test_dequantize_rejects_mismatched_scale_shapes():
    packed, scales, global_scale = quantize_nvfp4(torch.randn(E, N, K))
    with pytest.raises(ValueError, match="scales shape"):
        dequantize_nvfp4(packed, scales[:, :, :-1], global_scale, torch.float32)
    with pytest.raises(ValueError, match="global_scale shape"):
        dequantize_nvfp4(packed, scales, global_scale[0], torch.float32)


def test_dequantize_error_is_bounded_by_the_e2m1_step():
    torch.manual_seed(0)
    w = torch.randn(E, N, K, dtype=torch.float32)
    packed, scales, global_scale = quantize_nvfp4(w)
    dq = dequantize_nvfp4(packed, scales, global_scale, torch.float32)
    scale = scales.to(torch.float32) * global_scale.unsqueeze(-1)  # [E, N, K // 16]
    err = (dq - w).abs().view(E, N, -1, NVFP4_BLOCK_SIZE)
    # Largest gap between consecutive e2m1 magnitudes is 2 (4 -> 6), so rounding costs at most one decode scale;
    # a block whose e4m3 scale rounded down clamps its amax to 6 at a cost below 0.4 scales.
    bound = 1.0 * scale.unsqueeze(-1) + 1e-6
    assert torch.all(err <= bound)
    assert torch.all(scales.to(torch.float32) <= E4M3_MAX)


def test_quantize_dequantize_idempotent():
    torch.manual_seed(0)
    w = torch.randn(E, N, K, dtype=torch.bfloat16)
    dq1 = dequantize_nvfp4(*quantize_nvfp4(w), torch.bfloat16)
    dq2 = dequantize_nvfp4(*quantize_nvfp4(dq1), torch.bfloat16)
    assert torch.equal(dq1, dq2)


def test_zero_row_decodes_to_zeros():
    w = torch.zeros(2, K, dtype=torch.bfloat16)
    w[1, NVFP4_BLOCK_SIZE:] = 1.5
    packed, scales, global_scale = quantize_nvfp4(w)
    dq = dequantize_nvfp4(packed, scales, global_scale, torch.bfloat16)
    assert torch.equal(dq[0], torch.zeros(K, dtype=torch.bfloat16))
    assert torch.equal(dq[1, :NVFP4_BLOCK_SIZE], torch.zeros(NVFP4_BLOCK_SIZE, dtype=torch.bfloat16))
    assert torch.equal(dq[1, NVFP4_BLOCK_SIZE:], torch.full((K - NVFP4_BLOCK_SIZE,), 1.5, dtype=torch.bfloat16))


def test_int8_packed_input_is_accepted():
    w = torch.randn(2, K, dtype=torch.bfloat16)
    packed, scales, global_scale = quantize_nvfp4(w)
    a = dequantize_nvfp4(packed, scales, global_scale, torch.float32)
    b = dequantize_nvfp4(packed.view(torch.int8), scales, global_scale, torch.float32)
    assert torch.equal(a, b)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_grouped_mm_dequantizes_and_backpropagates_to_the_input(device):
    device = torch.device(device)
    if not _grouped_mm_available(device):
        pytest.skip("torch._grouped_mm is not available on this device")
    torch.manual_seed(0)
    packed, scales, global_scale = quantize_nvfp4(torch.randn(E, N, K, dtype=torch.bfloat16, device=device))
    x = torch.randn(8, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    offs = torch.tensor([3, 3, 8], dtype=torch.int32, device=device)  # expert 1 receives no tokens
    out = NVFP4GroupedMM.apply(x, packed, scales, global_scale, offs)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    x_ref = x.detach().clone().requires_grad_(True)
    w_t = dequantize_nvfp4(packed, scales, global_scale, torch.bfloat16)  # [E, N, K]
    ref = torch._grouped_mm(x_ref, w_t.transpose(-2, -1), offs=offs)
    ref.backward(grad_out)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)


def test_quantize_keeps_subnormal_block_scales():
    """e4m3 block scales reach 2^-9 (subnormal); clamping at the normal minimum would zero small blocks."""
    w = torch.zeros(1, 2 * NVFP4_BLOCK_SIZE)
    w[0, 0], w[0, 1] = 6.0 / 512, 1.0 / 512  # block 0: max code 6 at scale 2^-9, second value code 1
    w[0, NVFP4_BLOCK_SIZE] = 6.0  # block 1 keeps the row amax at 6
    packed, scales, global_scale = quantize_nvfp4(w, global_scale=torch.tensor(1.0))
    assert float(scales[0, 0]) == 2.0**-9 and float(scales[0, 1]) == 1.0
    dq = dequantize_nvfp4(packed, scales, global_scale, torch.float32)
    assert dq[0, 0].item() == 6.0 / 512 and dq[0, 1].item() == 1.0 / 512 and dq[0, NVFP4_BLOCK_SIZE].item() == 6.0
    assert torch.equal(dq[0, 2:NVFP4_BLOCK_SIZE], torch.zeros(NVFP4_BLOCK_SIZE - 2))
