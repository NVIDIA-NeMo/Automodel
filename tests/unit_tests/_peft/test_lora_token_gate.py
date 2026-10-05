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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components._peft.lora import (
    LinearLoRA,
    PeftConfig,
    apply_lora_to_linear_modules,
    lora_token_gate,
    patch_linear_module,
)

B, S, H, INTER = 2, 6, 16, 32


class SwiGLUMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, INTER, bias=False)
        self.up_proj = nn.Linear(H, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, H, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        """Args: x: Tensor of shape [batch, sequence, hidden]."""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, H)
        self.mlp = SwiGLUMLP()

    def forward(self, x):
        """Args: x: Tensor of shape [batch, sequence, hidden]."""
        return self.mlp(self.q_proj(x))


def _lora_model(seed=0, **cfg):
    torch.manual_seed(seed)
    model = Block()
    peft = PeftConfig(target_modules=["*_proj"], dim=4, alpha=8, **cfg)
    apply_lora_to_linear_modules(model, peft)
    # Non-zero lora_B so the adapter actually changes the output.
    for m in model.modules():
        if isinstance(m, LinearLoRA):
            nn.init.normal_(m.lora_B.weight, std=0.1)
    return model


def _lora_grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if "lora_" in n}


def _gate():
    gate = torch.zeros(B, S, dtype=torch.bool)
    gate[0, 2:] = True
    gate[1, :3] = True
    return gate


@torch.no_grad()
def _base_forward(model, x):
    """Frozen-base reference computed directly from the base weights (no LoRA, no fused paths).

    Args:
        x: Tensor of shape [batch, sequence, hidden].

    Returns:
        Tensor of shape [batch, sequence, hidden].
    """
    q = F.linear(x, model.q_proj.weight, model.q_proj.bias)
    mlp = model.mlp
    return F.linear(F.silu(F.linear(q, mlp.gate_proj.weight)) * F.linear(q, mlp.up_proj.weight), mlp.down_proj.weight)


@pytest.mark.parametrize("use_memory_efficient_lora", [True, False])
def test_all_ones_gate_matches_ungated_forward_and_backward(use_memory_efficient_lora):
    model = _lora_model(use_memory_efficient_lora=use_memory_efficient_lora, use_triton=False)
    x = torch.randn(B, S, H)
    upstream = torch.randn(B, S, H)

    ref = model(x)
    ref.backward(upstream)
    ref_grads = _lora_grads(model)
    model.zero_grad()

    with lora_token_gate(model, torch.ones(B, S, dtype=torch.bool)):
        out = model(x)
        out.backward(upstream)
    torch.testing.assert_close(out, ref)
    for name, grad in _lora_grads(model).items():
        torch.testing.assert_close(grad, ref_grads[name], msg=name)


def test_all_zeros_gate_matches_frozen_base_and_gives_zero_adapter_grads():
    model = _lora_model(use_triton=False)
    x = torch.randn(B, S, H)
    with lora_token_gate(model, torch.zeros(B, S, dtype=torch.bool)):
        out = model(x)
        out.backward(torch.randn_like(out))
    torch.testing.assert_close(out, _base_forward(model, x))
    for name, grad in _lora_grads(model).items():
        assert torch.count_nonzero(grad) == 0, name


def test_mixed_gate_routes_tokens_and_isolates_gradients():
    model = _lora_model(use_triton=False)
    gate = _gate()
    # Tokens are independent through q_proj + MLP, so each token's output is either the adapted or base model's.
    x = torch.randn(B, S, H)
    with torch.no_grad():
        adapted = model(x)
        base = _base_forward(model, x)
    with lora_token_gate(model, gate):
        out = model(x)
    torch.testing.assert_close(out[gate], adapted[gate])
    torch.testing.assert_close(out[~gate], base[~gate])

    # A loss on gated-off tokens only must leave the adapter untouched.
    model.zero_grad()
    with lora_token_gate(model, gate):
        out = model(x)
        (out * (~gate).unsqueeze(-1)).pow(2).sum().backward()
    for name, grad in _lora_grads(model).items():
        assert torch.count_nonzero(grad) == 0, name

    # A loss on gated-on tokens matches the ungated adapter gradient for those tokens.
    model.zero_grad()
    upstream = torch.randn(B, S, H) * gate.unsqueeze(-1)
    model(x).backward(upstream)
    ref_grads = _lora_grads(model)
    model.zero_grad()
    with lora_token_gate(model, gate):
        model(x).backward(upstream)
    for name, grad in _lora_grads(model).items():
        torch.testing.assert_close(grad, ref_grads[name], msg=name)


def test_gate_bypasses_fused_mlp_and_triton_paths():
    model = _lora_model(use_memory_efficient_lora=True, use_triton=True)
    assert getattr(model.mlp, "_lora_mlp_fused", False)
    x = torch.randn(B, S, H)
    with torch.no_grad():
        base = _base_forward(model, x)
        with lora_token_gate(model, torch.zeros(B, S, dtype=torch.bool)):
            out = model(x)
    torch.testing.assert_close(out, base)


def test_gate_with_activation_checkpointing_matches_plain_backward():
    model = _lora_model(use_triton=False)
    gate = _gate()
    x = torch.randn(B, S, H, requires_grad=True)
    upstream = torch.randn(B, S, H)

    with lora_token_gate(model, gate):
        model(x).backward(upstream)
    ref_grads = _lora_grads(model)
    model.zero_grad()

    with lora_token_gate(model, gate):
        checkpoint(model, x, use_reentrant=False).backward(upstream)
    for name, grad in _lora_grads(model).items():
        torch.testing.assert_close(grad, ref_grads[name], msg=name)


def test_flattened_tokens_accept_batch_sequence_gate():
    torch.manual_seed(0)
    linear = patch_linear_module(nn.Linear(H, H), dim=4, alpha=8, use_triton=False)
    nn.init.normal_(linear.lora_B.weight, std=0.1)
    gate = _gate()
    x = torch.randn(B, S, H)
    with lora_token_gate(linear, gate):
        flat = linear(x.reshape(B * S, H)).reshape(B, S, H)
        batched = linear(x)
    torch.testing.assert_close(flat, batched)
    torch.testing.assert_close(flat[~gate], F.linear(x, linear.weight, linear.bias)[~gate])


def test_gate_is_removed_on_exit_and_on_error():
    model = _lora_model(use_triton=False)
    with lora_token_gate(model, _gate()):
        assert model.q_proj._lora_token_gate is not None
    assert model.q_proj._lora_token_gate is None
    with pytest.raises(RuntimeError, match="boom"):
        with lora_token_gate(model, _gate()):
            raise RuntimeError("boom")
    assert all(m._lora_token_gate is None for m in model.modules() if isinstance(m, LinearLoRA))


def test_gate_shape_mismatch_raises():
    model = _lora_model(use_triton=False)
    with lora_token_gate(model, torch.ones(B, S + 1, dtype=torch.bool)):
        with pytest.raises(ValueError, match="lora_token_gate has"):
            model(torch.randn(B, S, H))


def test_gate_rejects_model_without_lora_and_dora():
    with pytest.raises(ValueError, match="no LoRA-patched linear layers"):
        with lora_token_gate(nn.Linear(H, H), torch.ones(B, S)):
            pass
    dora = patch_linear_module(nn.Linear(H, H), dim=4, alpha=8, use_dora=True, use_triton=False)
    with lora_token_gate(dora, torch.ones(B, S)):
        with pytest.raises(NotImplementedError, match="DoRA"):
            dora(torch.randn(B, S, H))
