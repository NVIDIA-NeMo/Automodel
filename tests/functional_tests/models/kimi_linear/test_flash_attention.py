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

"""Real CUDA-kernel output and backward parity for packed Kimi MLA."""

import pytest
import torch

from nemo_automodel.components.models.kimi_linear.config import KimiLinear48BConfig
from nemo_automodel.components.models.kimi_linear.cp import (
    KimiPackedContext,
    build_document_causal_mask,
)
from nemo_automodel.components.models.kimi_linear.flash_attention import (
    build_flash_layout,
    packed_flash_attention,
)
from nemo_automodel.components.models.kimi_linear.model import KimiMLAAttention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA FlashAttention forward/backward",
)


def _assert_parity(actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> None:
    """Compare nonzero tensors using global relative error and maximum magnitude.

    Args:
        actual: Tensor of arbitrary shape produced by the FlashAttention path.
        expected: Reference tensor with the same arbitrary shape as actual.
        tolerance: Maximum relative L2 and relative maximum absolute error.
    """
    delta = actual.float() - expected.float()
    reference = expected.float()
    relative_l2 = delta.norm() / reference.norm().clamp_min(1e-12)
    assert relative_l2.item() < tolerance
    assert delta.abs().max().item() <= tolerance * reference.abs().max().item() + 1e-6


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("qk_dim,value_dim", [(192, 128), (48, 64)])
def test_real_kernel_matches_dense_outputs_and_qkv_backward(dtype, qk_dim, value_dim) -> None:
    pytest.importorskip("flash_attn.cute", reason="requires the optional FlashAttention-4 CUDA kernel")
    torch.manual_seed(74)
    ids = torch.tensor(
        [
            [1, 1, 1, 0, 2, 2, 2, 2, 0, 3, 3, 3, 3],
            [0, 1, 0, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    q = torch.randn(2, 4, 13, qk_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn(2, 4, 13, value_dim, device="cuda", dtype=dtype, requires_grad=True)
    mask = build_document_causal_mask(ids, ids, q_global_start=0, dtype=dtype)
    scores = (q @ k.transpose(-1, -2)) * qk_dim**-0.5 + mask
    reference = (scores.softmax(-1, dtype=torch.float32).to(dtype) @ v) * (ids > 0)[:, None, :, None]
    actual = packed_flash_attention(q, k, v, layout=build_flash_layout(ids), scale=qk_dim**-0.5)
    tolerance = 0.01 if dtype == torch.bfloat16 else 0.003
    _assert_parity(actual, reference, tolerance)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(reference, (q, k, v), grad, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    for a, b in zip(actual_grads, expected_grads):
        _assert_parity(a, b, tolerance)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mla_projection_parameter_and_input_gradients(dtype) -> None:
    pytest.importorskip("flash_attn.cute", reason="requires the optional FlashAttention-4 CUDA kernel")
    torch.manual_seed(19)
    config = KimiLinear48BConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=32,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        torch_dtype=dtype,
    )
    eager = KimiMLAAttention(config, 0).cuda()
    fast_config = KimiLinear48BConfig.from_dict({**config.to_dict(), "mla_attn_implementation": "fa4"})
    fast = KimiMLAAttention(fast_config, 0).cuda()
    fast.load_state_dict(eager.state_dict(), strict=True)
    ids = torch.tensor(
        [[1] * 7 + [0] + [2] * 5 + [0] * 3, [0] * 3 + [1] * 5 + [2] * 8],
        device="cuda",
        dtype=torch.int32,
    )
    context = KimiPackedContext(ids)
    x = torch.randn(2, 16, 64, device="cuda", dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    reference = eager(
        x,
        attention_mask=build_document_causal_mask(ids, ids, q_global_start=0, dtype=dtype),
        packed_context=context,
    )
    actual = fast(y, packed_context=context, flash_layout=build_flash_layout(ids))
    valid = ids > 0
    tolerance = 0.015 if dtype == torch.bfloat16 else 0.004
    _assert_parity(actual[valid], reference[valid], tolerance)
    grad = torch.randn_like(actual) * valid[..., None]
    reference.backward(grad)
    actual.backward(grad)
    _assert_parity(y.grad, x.grad, tolerance)
    for (name, a), (other, b) in zip(fast.named_parameters(), eager.named_parameters()):
        assert name == other
        assert a.grad is not None and b.grad is not None, name
        _assert_parity(a.grad, b.grad, tolerance)
