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

"""CUDA forward/backward coverage for DSV4 fused RoPE and residual mixing."""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4HyperConnection,
    _apply_partial_rope,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("length", [4096, 8192])
@pytest.mark.parametrize("inverse", [False, True])
def test_triton_rope_preserves_outputs_and_gradients(length: int, inverse: bool) -> None:
    torch.manual_seed(17)
    # Strided BHSD matches the projection output and exercises nonzero positions.
    x = torch.randn(1, length, 64, 512, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    frequency = 10000 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
    angles = (torch.arange(length, device="cuda").float() + 13)[:, None] * frequency
    cos = angles.cos().repeat(1, 2).unsqueeze(0)
    sin = angles.sin().repeat(1, 2).unsqueeze(0) * (-1 if inverse else 1)
    grad = torch.randn(1, length, 64, 512, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    expected = _apply_partial_rope(x, cos, sin, 64)
    expected_grad = torch.autograd.grad(expected, x, grad)[0]
    actual = _apply_partial_rope(x, cos, sin, 64, backend="triton")
    actual_grad = torch.autograd.grad(actual, x, grad)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert actual.transpose(1, 2).is_contiguous()
    # Independent complex-number oracle distinguishes adjacent-pair from half-split RoPE.
    reference_pairs = torch.view_as_complex(x[..., -64:].float().contiguous().reshape(1, 64, length, 32, 2))
    reference_frequency = torch.complex(cos[..., :32], sin[..., :32]).unsqueeze(1)
    reference = torch.view_as_real(reference_pairs * reference_frequency).flatten(-2).to(x.dtype)
    torch.testing.assert_close(actual[..., -64:], reference, rtol=0.008, atol=0.015625)
    torch.testing.assert_close(actual[..., :-64], x[..., :-64], rtol=0, atol=0)


def test_compiled_hc_preserves_forward_and_parameter_gradients() -> None:
    torch.manual_seed(21)
    eager = DeepseekV4HyperConnection(4, 128, 5, 1e-6, 1e-6, "auto").cuda()
    eager.init_weights(0.02)
    compiled = DeepseekV4HyperConnection(4, 128, 5, 1e-6, 1e-6, "auto", compile_hc=True).cuda()
    compiled.load_state_dict(eager.state_dict())
    x = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn_like(x)
    results = []
    for module in (eager, compiled):
        pre, post, comb = module(x)
        collapsed = module.collapse(x, pre)
        output = module.expand(collapsed, x, post, comb)
        grads = torch.autograd.grad(output, (x, *module.parameters()), upstream)
        results.append((output, grads))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=0.016, atol=0.03125)
    for actual, expected in zip(results[1][1], results[0][1], strict=True):
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float()
        )
        assert torch.isfinite(actual).all()
        assert relative_l2 < 0.01


def test_rope_frequency_gradients_keep_the_eager_path() -> None:
    torch.manual_seed(4)
    x = torch.randn(1, 2, 8, 16, device="cuda", requires_grad=True)
    cos = torch.randn(1, 8, 8, device="cuda", requires_grad=True)
    sin = torch.randn(1, 8, 8, device="cuda", requires_grad=True)
    upstream = torch.randn_like(x)
    expected = _apply_partial_rope(x, cos, sin, 8)
    actual = _apply_partial_rope(x, cos, sin, 8, backend="triton")
    expected_grads = torch.autograd.grad(expected, (x, cos, sin), upstream)
    actual_grads = torch.autograd.grad(actual, (x, cos, sin), upstream)
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [1.0, 1e-4])
def test_fused_qnorm_rope_matches_independent_fp64_oracle(scale: float) -> None:
    from nemo_automodel.components.models.deepseek_v4.kernels.qnorm_rope import qnorm_rope_triton

    torch.manual_seed(1729)
    x = (torch.randn(2, 17, 3, 64, device="cuda") * scale).transpose(1, 2).requires_grad_()
    angles = torch.randn(2, 17, 16, device="cuda")
    cos, sin = angles.cos().repeat(1, 1, 2), angles.sin().repeat(1, 1, 2)
    upstream = torch.randn_like(x)
    normalized = (x.double() * (x.double().square().mean(-1, keepdim=True) + 1e-6).rsqrt()).float()
    pairs = torch.view_as_complex(normalized[..., -32:].double().contiguous().reshape(2, 3, 17, 16, 2))
    frequency = torch.complex(cos[..., :16].double(), sin[..., :16].double()).unsqueeze(1)
    rotated = torch.view_as_real(pairs * frequency).flatten(-2).float()
    expected = torch.cat((normalized[..., :-32], rotated), -1)
    actual = qnorm_rope_triton(x, cos, sin, 32, 1e-6)
    expected_grad = torch.autograd.grad(expected, x, upstream)[0]
    actual_grad = torch.autograd.grad(actual, x, upstream)[0]
    for value, reference in ((actual, expected), (actual_grad, expected_grad)):
        error = (value - reference).norm() / reference.norm()
        assert error < 2e-6
    # This oracle must detect a conjugate-sign error at nonzero positions.
    wrong = qnorm_rope_triton(x, cos, -sin, 32, 1e-6)
    assert (wrong - expected).norm() / expected.norm() > 0.1


@pytest.mark.parametrize("length", [4096, 8192])
@pytest.mark.parametrize("frequency_dtype", [torch.float32, torch.bfloat16])
def test_fused_qnorm_rope_preserves_bf16_rounding_and_gradients(
    length: int,
    frequency_dtype: torch.dtype,
) -> None:
    import torch.nn.functional as F

    from nemo_automodel.components.models.deepseek_v4.kernels.qnorm_rope import qnorm_rope_triton

    torch.manual_seed(1731)
    x = torch.randn(1, length, 64, 512, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    angle = torch.randn(1, length, 32, device="cuda")
    cos, sin = angle.cos().repeat(1, 1, 2), angle.sin().repeat(1, 1, 2)
    cos, sin = cos.to(frequency_dtype), sin.to(frequency_dtype)
    upstream = torch.randn_like(x)
    expected = _apply_partial_rope(F.rms_norm(x, (512,), eps=1e-6), cos, sin, 64)
    actual = qnorm_rope_triton(x, cos, sin, 64, 1e-6)
    expected_grad = torch.autograd.grad(expected, x, upstream)[0]
    actual_grad = torch.autograd.grad(actual, x, upstream)[0]
    assert actual.transpose(1, 2).is_contiguous()
    # FP32 reduction ordering can cross a BF16 rounding midpoint. Both the
    # normalization output and rotated backward gradient retain BF16 rounding.
    for value, reference in ((actual, expected), (actual_grad, expected_grad)):
        assert torch.isfinite(value).all()
        error = value.float() - reference.float()
        assert error.norm() / reference.float().norm() < 2e-4
        assert error.abs().max() <= 0.0625


# Each shape holds just over 2**31 elements: the first exceeds int32 element
# offsets through the sequence, the second through its batch stride.
_LARGE_SHAPES = [(2, 32769), (1, 65537)]


def _large_rope_inputs(batch: int, length: int) -> tuple[torch.Tensor, ...]:
    """Build token-major BF16 Q activations whose element count exceeds 2**31.

    Args:
        batch: Batch size.
        length: Sequence length per sample.

    Returns:
        Tuple of x and upstream gradient, both [batch, 64, length, 512] views of
        [batch, length, 64, 512] storage, followed by FP32 cosine and sine
        tables of shape [batch, length, 64].
    """
    # x, upstream, output and input gradient: four ~4.3 GB tensors plus allocator slack.
    if torch.cuda.mem_get_info()[0] < 24 * 2**30:
        pytest.skip("requires 24 GiB of free CUDA memory")
    torch.manual_seed(31)
    x = torch.randn(batch, length, 64, 512, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    upstream = torch.randn(batch, length, 64, 512, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    assert x.numel() > 2**31
    angles = torch.randn(batch, length, 32, device="cuda")
    return x.requires_grad_(), upstream, angles.cos().repeat(1, 1, 2), angles.sin().repeat(1, 1, 2)


def _sample_positions(length: int) -> torch.Tensor:
    """Select first, middle and final sequence positions.

    Args:
        length: Sequence length per sample.

    Returns:
        CUDA int64 tensor of shape [5]. The final positions of the last sample
        lie beyond 2**31 elements in the large test tensors.
    """
    return torch.tensor([0, 1, length // 2, length - 2, length - 1], device="cuda")


@pytest.mark.parametrize(("batch", "length"), _LARGE_SHAPES)
def test_triton_rope_addresses_tensors_beyond_int32_offsets(batch: int, length: int) -> None:
    x, upstream, cos, sin = _large_rope_inputs(batch, length)
    actual = _apply_partial_rope(x, cos, sin, 64, backend="triton")
    actual_grad = torch.autograd.grad(actual, x, upstream)[0]
    positions = _sample_positions(length)
    # Eager RoPE is position-local, so sampled positions form an exact oracle.
    sampled = x.detach()[:, :, positions].requires_grad_()
    expected = _apply_partial_rope(sampled, cos[:, positions], sin[:, positions], 64)
    expected_grad = torch.autograd.grad(expected, sampled, upstream[:, :, positions])[0]
    torch.testing.assert_close(actual[:, :, positions], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad[:, :, positions], expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize(("batch", "length"), _LARGE_SHAPES)
def test_fused_qnorm_rope_addresses_tensors_beyond_int32_offsets(batch: int, length: int) -> None:
    import torch.nn.functional as F

    from nemo_automodel.components.models.deepseek_v4.kernels.qnorm_rope import qnorm_rope_triton

    x, upstream, cos, sin = _large_rope_inputs(batch, length)
    actual = qnorm_rope_triton(x, cos, sin, 64, 1e-6)
    actual_grad = torch.autograd.grad(actual, x, upstream)[0]
    positions = _sample_positions(length)
    sampled = x.detach()[:, :, positions].requires_grad_()
    expected = _apply_partial_rope(F.rms_norm(sampled, (512,), eps=1e-6), cos[:, positions], sin[:, positions], 64)
    expected_grad = torch.autograd.grad(expected, sampled, upstream[:, :, positions])[0]
    # Same BF16 rounding-boundary bounds as the 4K/8K parity test above.
    for value, reference in ((actual[:, :, positions], expected), (actual_grad[:, :, positions], expected_grad)):
        error = value.float() - reference.float()
        assert torch.isfinite(value).all()
        assert error.norm() / reference.float().norm() < 2e-4
        assert error.abs().max() <= 0.0625
