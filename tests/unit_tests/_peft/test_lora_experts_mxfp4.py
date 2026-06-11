# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_automodel.components._peft.lora import patch_moe_module
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA, GroupedExpertsLoRAMXFP4
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fp4_utils import dequantize_mxfp4, quantize_mxfp4
from nemo_automodel.components.moe.layers import GroupedExperts


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def moe_config():
    # Dims divisible by 32 so both contraction dims are mxfp4-blockable.
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        norm_topk_prob=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
    )


def test_quantize_dequantize_idempotent():
    """A second quantize/dequantize round-trip must reproduce the first exactly."""
    torch.manual_seed(0)
    w = torch.randn(3, 8, 64, dtype=torch.bfloat16)
    packed, scales = quantize_mxfp4(w)
    assert packed.dtype == torch.int8
    assert packed.shape == (3, 8, 32)
    assert scales.dtype == torch.float8_e8m0fnu
    assert scales.shape == (3, 8, 2)

    dq1 = dequantize_mxfp4(packed, scales, torch.bfloat16)
    packed2, scales2 = quantize_mxfp4(dq1)
    dq2 = dequantize_mxfp4(packed2, scales2, torch.bfloat16)
    assert torch.equal(dq1, dq2)


def test_quantize_zero_block():
    """All-zero blocks must encode to scale byte 0 and decode back to exact zeros."""
    w = torch.zeros(2, 64, dtype=torch.bfloat16)
    w[1, 32:] = 1.5  # one nonzero block to confirm mixed handling
    packed, scales = quantize_mxfp4(w)
    dq = dequantize_mxfp4(packed, scales, torch.bfloat16)
    assert torch.equal(dq[0], torch.zeros(64, dtype=torch.bfloat16))
    assert torch.equal(dq[1, :32], torch.zeros(32, dtype=torch.bfloat16))
    assert torch.equal(dq[1, 32:], torch.full((32,), 1.5, dtype=torch.bfloat16))


def test_dequantize_matches_dsv4_adapter():
    """fp4_utils dequant must agree exactly with the DeepSeek V4 state dict adapter."""
    from nemo_automodel.components.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter

    torch.manual_seed(1)
    w = torch.randn(4, 16, 96, dtype=torch.bfloat16)
    packed, scales = quantize_mxfp4(w)
    dq = dequantize_mxfp4(packed, scales, torch.bfloat16)
    ref = DeepSeekV4StateDictAdapter._dequantize_expert_fp4(
        packed.view(-1, packed.shape[-1]), scales.view(-1, scales.shape[-1]), torch.bfloat16
    )
    assert torch.equal(ref.view_as(dq), dq)


def _make_representable_(experts: GroupedExperts) -> None:
    """Replace expert weights in-place with their mxfp4 round-trip so storage is exact."""
    with torch.no_grad():
        for name in ("gate_and_up_projs", "down_projs"):
            param = getattr(experts, name)
            w_t = param.data.transpose(-2, -1).contiguous()
            packed, scales = quantize_mxfp4(w_t)
            param.data.copy_(dequantize_mxfp4(packed, scales, param.dtype).transpose(-2, -1))


def _routing_inputs(moe_config, num_tokens, device, dtype):
    torch.manual_seed(7)
    x = torch.randn(num_tokens, moe_config.dim, dtype=dtype, device=device, requires_grad=True)
    indices = torch.stack(
        [
            torch.randperm(moe_config.n_routed_experts, device=device)[: moe_config.n_activated_experts]
            for _ in range(num_tokens)
        ]
    )
    weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=dtype, device=device)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    return x, token_mask, weights, indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mxfp4_module_packs_base_weights(moe_config, device):
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)

    lora = GroupedExpertsLoRAMXFP4(orig, lora_dim=8, alpha=16)

    assert lora._mxfp4_resident
    assert not hasattr(lora, "gate_and_up_projs")
    assert not hasattr(lora, "down_projs")
    assert lora.gate_and_up_projs_packed.dtype == torch.int8
    assert lora.gate_and_up_projs_scales.dtype == torch.float8_e8m0fnu
    assert not lora.gate_and_up_projs_packed.requires_grad
    # Checkpoint orientation: [E, out, in/2] for gate+up ([E, 2*inter, dim/2]).
    assert lora.gate_and_up_projs_packed.shape == (4, 2 * moe_config.moe_inter_dim, moe_config.dim // 2)
    assert lora.down_projs_packed.shape == (4, moe_config.dim, moe_config.moe_inter_dim // 2)
    # Only LoRA params are trainable.
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mxfp4_forward_backward_matches_bf16(moe_config, device):
    """With fp4-representable base weights, the packed module must match the bf16 module."""
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)
    _make_representable_(orig)

    ref = GroupedExpertsLoRA(orig, lora_dim=8, alpha=16).to(device)
    mx = GroupedExpertsLoRAMXFP4(orig, lora_dim=8, alpha=16).to(device)
    with torch.no_grad():
        for name in ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"):
            getattr(mx, name).data.copy_(getattr(ref, name).data)

    x_ref, token_mask, weights, indices = _routing_inputs(moe_config, 32, device, torch.bfloat16)
    x_mx = x_ref.detach().clone().requires_grad_(True)

    y_ref = ref(x_ref, token_mask, weights, indices)
    y_mx = mx(x_mx, token_mask, weights, indices)
    torch.testing.assert_close(y_mx, y_ref, atol=1e-6, rtol=1e-6)

    y_ref.float().pow(2).sum().backward()
    y_mx.float().pow(2).sum().backward()
    torch.testing.assert_close(x_mx.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for name in ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"):
        torch.testing.assert_close(getattr(mx, name).grad, getattr(ref, name).grad, atol=1e-5, rtol=1e-5)
    # Base weights are frozen; no grads anywhere else.
    assert mx.gate_and_up_projs_packed.grad is None
    assert mx.down_projs_packed.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mxfp4_no_routed_tokens(moe_config, device):
    """The all-masked dummy path must run and keep LoRA gradients flowing."""
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)

    mx = GroupedExpertsLoRAMXFP4(orig, lora_dim=8, alpha=16).to(device)
    x, _, weights, indices = _routing_inputs(moe_config, 8, device, torch.bfloat16)
    token_mask = torch.zeros(8, dtype=torch.bool, device=device)

    y = mx(x, token_mask, weights, indices)
    assert y.shape == x.shape
    y.float().sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mxfp4_requires_torch_mm_backend(moe_config, device):
    orig = GroupedExperts(moe_config).to(device)  # default per-expert loop backend
    with torch.no_grad():
        orig.init_weights(buffer_device=device)
    with pytest.raises(NotImplementedError, match="torch_mm"):
        GroupedExpertsLoRAMXFP4(orig, lora_dim=8, alpha=16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_patch_moe_module_mxfp4(moe_config, device):
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)

    patched = patch_moe_module(orig, dim=4, alpha=8, expert_weight_format="mxfp4")
    assert isinstance(patched, GroupedExpertsLoRAMXFP4)

    patched_bf16 = patch_moe_module(orig, dim=4, alpha=8)
    assert isinstance(patched_bf16, GroupedExpertsLoRA)
    assert not isinstance(patched_bf16, GroupedExpertsLoRAMXFP4)
