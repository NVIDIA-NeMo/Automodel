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

"""Real CUDA-kernel output and backward parity for packed Kimi MLA.

These tests constrain replacement of dtype-matched eager attention, rather than
claiming FP32 accuracy. The CPU dense oracle separately checks layout arithmetic
in FP32. Both relative L2 and normalized maximum error must satisfy the fixed
bounds; BF16 has a wider budget than FP16 because its mantissa is shorter.
"""

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_linear.config import KimiLinear48BConfig
from nemo_automodel.components.models.kimi_linear.cp import (
    KimiPackedContext,
    build_document_causal_mask,
)
from nemo_automodel.components.models.kimi_linear.flash_attention import (
    build_flash_layout,
    packed_flash_attention,
)
from nemo_automodel.components.models.kimi_linear.model import KimiLinear48BForCausalLM, KimiMLAAttention

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention forward/backward"),
    pytest.mark.timeout(600),
]


@pytest.fixture(scope="module", autouse=True)
def _require_fa4_gpu_architecture() -> None:
    """Require at least SM90 for the FA4 CUDA tests."""
    capability = torch.cuda.get_device_capability()
    if capability < (9, 0):
        pytest.fail(f"Kimi Linear FA4 CUDA tests require SM90 or newer; got SM{capability[0]}{capability[1]}")


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
    fast = KimiMLAAttention(config, 0, backend=BackendConfig(attn="fa4")).cuda()
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


def test_checkpointed_backbone_logits_and_parameter_gradients() -> None:
    """Compare a real BF16 FA4 backbone under non-reentrant checkpoint recomputation."""
    torch.manual_seed(41)
    config = KimiLinear48BConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=32,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=0,
        moe_intermediate_size=32,
        first_k_dense_replace=1,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1]},
        torch_dtype=torch.bfloat16,
    )
    eager_backend = BackendConfig(
        attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch"
    )
    fast_backend = BackendConfig(attn="fa4", linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch")
    eager = KimiLinear48BForCausalLM(config, backend=eager_backend).cuda()
    eager.initialize_weights(buffer_device=torch.device("cuda"), dtype=torch.bfloat16)
    fast = KimiLinear48BForCausalLM(config, backend=fast_backend).cuda()
    fast.load_state_dict(eager.state_dict(), strict=True)
    for model in (eager, fast):
        for name, layer in list(model.model.layers.items()):
            model.model.layers[name] = ptd_checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    ids = torch.tensor([[1, 1, 0, 2, 2], [0, 1, 1, 0, 0]], device="cuda", dtype=torch.int32)
    tokens = torch.tensor([[3, 4, 0, 5, 6], [0, 7, 8, 0, 0]], device="cuda")
    expected = eager(input_ids=tokens, attention_mask=ids, padding_mask=ids == 0).logits
    actual = fast(input_ids=tokens, attention_mask=ids, padding_mask=ids == 0).logits
    valid = ids > 0
    _assert_parity(actual[valid], expected[valid], 0.015)
    grad = torch.randn_like(actual) * valid[..., None]
    expected.backward(grad)
    actual.backward(grad)
    for (name, parameter), (other, reference) in zip(fast.named_parameters(), eager.named_parameters()):
        assert name == other
        assert parameter.grad is not None and reference.grad is not None, name
        _assert_parity(parameter.grad, reference.grad, 0.015)
