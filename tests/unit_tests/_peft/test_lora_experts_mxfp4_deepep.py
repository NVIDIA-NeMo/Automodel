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

"""mxfp4-resident experts under DeepEP dispatch.

The DeepEP token all-to-all is mocked (``MockDeepEPDispatcher``) so the forward path is
exercised without a real DeepEP backend or process group — only the post-dispatch
grouped GEMM differs between bf16 and mxfp4, and that is what these tests pin down.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.components._peft.lora import convert_frozen_experts_to_mxfp4, patch_moe_module
from nemo_automodel.components._peft.lora_experts import (
    GroupedExpertsDeepEPLoRA,
    GroupedExpertsDeepEPLoRAMXFP4,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP
from nemo_automodel.components.moe.quantized_experts import GroupedExpertsDeepEPMXFP4, GroupedExpertsMXFP4
from nemo_automodel.components.quantization.mxfp4 import dequantize_mxfp4, quantize_mxfp4


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


class MockDeepEPDispatcher:
    """Mock dispatcher that returns pre-set permuted tensors (no comms)."""

    def __init__(self, permuted_x, tokens_per_expert, permuted_probs):
        self.token_permutation2 = MagicMock(return_value=(permuted_x, tokens_per_expert, permuted_probs))

    def token_unpermutation(self, hidden_states):
        return hidden_states


def _make_representable_(experts) -> None:
    """Replace expert weights in-place with their mxfp4 round-trip so storage is exact."""
    with torch.no_grad():
        for name in ("gate_and_up_projs", "down_projs"):
            param = getattr(experts, name)
            w_t = param.data.transpose(-2, -1).contiguous()
            packed, scales = quantize_mxfp4(w_t)
            param.data.copy_(dequantize_mxfp4(packed, scales, param.dtype).transpose(-2, -1))


def _make_deepep(moe_config, device, *, use_torch_mm=True, representable=False):
    """Build a materialized GroupedExpertsDeepEP with single-rank dispatcher state injected."""
    orig = GroupedExpertsDeepEP(moe_config).to(device).to(torch.bfloat16)
    with torch.no_grad():
        orig.init_weights(device)
    if representable:
        _make_representable_(orig)
    orig.n_routed_experts = moe_config.n_routed_experts
    orig.ep_size = 1
    orig.ep_rank = 0
    orig.use_torch_mm = use_torch_mm
    return orig


def _inject_dispatcher(module, num_tokens, dim, device):
    """Attach single-rank dispatcher state + a mock that routes all tokens to expert 0."""
    module.n_routed_experts = 4
    module.ep_size = 1
    module.ep_rank = 0
    tokens_per_expert = torch.tensor([num_tokens, 0, 0, 0], dtype=torch.long, device="cpu")
    permuted_x = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)
    permuted_probs = torch.ones(num_tokens, device=device, dtype=torch.bfloat16)
    module.token_dispatcher = MockDeepEPDispatcher(permuted_x, tokens_per_expert, permuted_probs)
    return permuted_x, tokens_per_expert, permuted_probs


# --- wiring / construction ----------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_frozen_deepep_mxfp4_packs_and_freezes(moe_config, device):
    orig = _make_deepep(moe_config, device)
    mx = GroupedExpertsDeepEPMXFP4(orig)

    assert mx._mxfp4_resident
    assert not hasattr(mx, "gate_and_up_projs")
    assert not hasattr(mx, "down_projs")
    assert mx.gate_and_up_projs_packed.dtype == torch.int8
    assert mx.down_projs_scales.dtype == torch.float8_e8m0fnu
    # Checkpoint orientation [E, out, in/2].
    assert mx.gate_and_up_projs_packed.shape == (4, 2 * moe_config.moe_inter_dim, moe_config.dim // 2)
    # DeepEP dispatcher knobs are carried over.
    assert mx.dispatcher_backend == orig.dispatcher_backend
    assert mx.use_torch_mm is True
    # Fully frozen.
    assert [n for n, p in mx.named_parameters() if p.requires_grad] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_deepep_mxfp4_only_lora_trainable(moe_config, device):
    orig = _make_deepep(moe_config, device)
    mx = GroupedExpertsDeepEPLoRAMXFP4(orig, lora_dim=8, alpha=16)

    assert mx._mxfp4_resident
    assert not hasattr(mx, "gate_and_up_projs")
    assert mx.gate_and_up_projs_packed.dtype == torch.int8
    trainable = {n for n, p in mx.named_parameters() if p.requires_grad}
    assert trainable == {"lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_patch_moe_module_deepep_mxfp4(moe_config, device):
    orig = _make_deepep(moe_config, device)
    patched = patch_moe_module(orig, dim=8, alpha=16, expert_weight_format="mxfp4")
    assert isinstance(patched, GroupedExpertsDeepEPLoRAMXFP4)

    orig2 = _make_deepep(moe_config, device)
    patched_bf16 = patch_moe_module(orig2, dim=8, alpha=16)
    assert isinstance(patched_bf16, GroupedExpertsDeepEPLoRA)
    assert not isinstance(patched_bf16, GroupedExpertsDeepEPLoRAMXFP4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deepep_mxfp4_requires_torch_mm_backend(moe_config, device):
    orig = _make_deepep(moe_config, device, use_torch_mm=False)  # gmm path -> unsupported
    with pytest.raises(NotImplementedError, match="torch_mm"):
        GroupedExpertsDeepEPMXFP4(orig)
    with pytest.raises(NotImplementedError, match="torch_mm"):
        GroupedExpertsDeepEPLoRAMXFP4(orig, lora_dim=8, alpha=16)


def test_passthrough_deepep_registers_packed_params_on_meta(moe_config):
    with torch.device("meta"):
        orig = GroupedExpertsDeepEP(moe_config)
    orig.use_torch_mm = True

    mx = GroupedExpertsDeepEPMXFP4(orig, passthrough=True)
    assert mx._mxfp4_resident
    assert not hasattr(mx, "gate_and_up_projs")  # never created bf16 storage
    up_proj_dim = 2 * moe_config.moe_inter_dim  # gated
    assert tuple(mx.gate_and_up_projs_packed.shape) == (moe_config.n_routed_experts, up_proj_dim, moe_config.dim // 2)
    assert mx.gate_and_up_projs_packed.is_meta
    assert mx.gate_and_up_projs_packed.dtype == torch.int8
    assert [n for n, p in mx.named_parameters() if p.requires_grad] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_convert_frozen_experts_to_mxfp4_handles_deepep(moe_config, device):
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = _make_deepep(moe_config, device)

    model = TinyModel()
    n = convert_frozen_experts_to_mxfp4(model)
    assert n == 1
    assert isinstance(model.experts, GroupedExpertsDeepEPMXFP4)
    # Not the torch-path class.
    assert not isinstance(model.experts, GroupedExpertsMXFP4)
    # Idempotent.
    assert convert_frozen_experts_to_mxfp4(model) == 0


# --- forward numerics (mock dispatcher) ---------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_frozen_deepep_mxfp4_forward_matches_bf16(moe_config, device):
    """Frozen mxfp4 DeepEP forward must match the bf16 DeepEP forward on representable weights."""
    orig = _make_deepep(moe_config, device, representable=True)
    mx = GroupedExpertsDeepEPMXFP4(orig)

    num_tokens = 8
    permuted_x, tokens_per_expert, permuted_probs = _inject_dispatcher(mx, num_tokens, moe_config.dim, device)
    # Share the exact same mock return on the bf16 reference.
    orig.token_dispatcher = MockDeepEPDispatcher(permuted_x, tokens_per_expert, permuted_probs)

    x = torch.randn(num_tokens, moe_config.dim, device=device, dtype=torch.bfloat16)
    weights = torch.ones(num_tokens, 1, device=device, dtype=torch.bfloat16)
    indices = torch.zeros(num_tokens, 1, dtype=torch.long, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    y_mx = mx(x, token_mask, weights, indices)
    # GroupedExpertsDeepEP.forward calls .to_local() on plain Parameters; patch for the test.
    with torch.no_grad(), patch.object(torch.Tensor, "to_local", new=lambda self: self, create=True):
        y_ref = orig(x, token_mask, weights, indices)

    assert y_mx.shape == (num_tokens, moe_config.dim)
    torch.testing.assert_close(y_mx, y_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_deepep_mxfp4_forward_matches_bf16(moe_config, device):
    """LoRA mxfp4 DeepEP forward must match the bf16 DeepEP LoRA forward with the same adapters."""
    orig = _make_deepep(moe_config, device, representable=True)

    ref = GroupedExpertsDeepEPLoRA(orig, lora_dim=8, alpha=16).to(device).to(torch.bfloat16)
    ref.use_torch_mm = True
    # Pack on CPU-allocated base params, then move packed storage + adapters to device.
    mx = GroupedExpertsDeepEPLoRAMXFP4(orig, lora_dim=8, alpha=16).to(device)
    # Give LoRA non-trivial values (B is zero-init by default) and sync both modules. The
    # DeepEP base allocates fp32 (it relies on a later .to(bf16)); assign the bf16 ref tensor
    # outright so the mxfp4 adapters are bf16 — .to(bf16) on the module would corrupt the
    # int8 packed base.
    with torch.no_grad():
        for name in ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"):
            getattr(ref, name).data.normal_(0, 0.02)
            getattr(mx, name).data = getattr(ref, name).data.clone().to(device)

    num_tokens = 8
    permuted_x, tokens_per_expert, permuted_probs = _inject_dispatcher(mx, num_tokens, moe_config.dim, device)
    ref.token_dispatcher = MockDeepEPDispatcher(permuted_x, tokens_per_expert, permuted_probs)
    ref.n_routed_experts = 4
    ref.ep_size = 1

    x = torch.randn(num_tokens, moe_config.dim, device=device, dtype=torch.bfloat16)
    weights = torch.ones(num_tokens, 1, device=device, dtype=torch.bfloat16)
    indices = torch.zeros(num_tokens, 1, dtype=torch.long, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    y_mx = mx(x, token_mask, weights, indices)
    with torch.no_grad(), patch.object(torch.Tensor, "to_local", new=lambda self: self, create=True):
        y_ref = ref(x, token_mask, weights, indices)

    assert y_mx.shape == (num_tokens, moe_config.dim)
    torch.testing.assert_close(y_mx, y_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_deepep_mxfp4_no_routed_tokens(moe_config, device):
    """The all-zero-tokens dummy path must run and keep LoRA gradients flowing."""
    orig = _make_deepep(moe_config, device)
    mx = GroupedExpertsDeepEPLoRAMXFP4(orig, lora_dim=8, alpha=16).to(device)
    # DeepEP base allocates fp32; make adapters bf16 to match activations (see note above).
    with torch.no_grad():
        for name in ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"):
            p = getattr(mx, name)
            p.data = p.data.to(torch.bfloat16)

    num_tokens = 8
    # tokens_per_expert all zeros -> dummy path.
    permuted_x = torch.randn(num_tokens, moe_config.dim, device=device, dtype=torch.bfloat16)
    permuted_probs = torch.ones(num_tokens, device=device, dtype=torch.bfloat16)
    tokens_per_expert = torch.zeros(4, dtype=torch.long, device="cpu")
    mx.n_routed_experts = 4
    mx.ep_size = 1
    mx.token_dispatcher = MockDeepEPDispatcher(permuted_x, tokens_per_expert, permuted_probs)

    x = torch.randn(num_tokens, moe_config.dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.ones(num_tokens, 1, device=device, dtype=torch.bfloat16)
    indices = torch.zeros(num_tokens, 1, dtype=torch.long, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    y = mx(x, token_mask, weights, indices)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.float().sum().backward()
    assert mx.lora_gate_and_up_A.grad is not None
