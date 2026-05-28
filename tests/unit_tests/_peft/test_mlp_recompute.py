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

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components._peft.mlp_recompute import apply_mlp_activation_recompute


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden=16, inter=32):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = SwiGLUMLP()
        self.other = nn.Linear(16, 16)  # should not be wrapped


def _run(model, x):
    out = model.mlp(x) if isinstance(model, Block) else model(x)
    out.sum().backward()
    return out, x.grad, {n: p.grad for n, p in model.named_parameters()}


def test_recompute_is_numerically_transparent():
    torch.manual_seed(0)
    ref = SwiGLUMLP().double()
    wrapped = SwiGLUMLP().double()
    wrapped.load_state_dict(ref.state_dict())

    x = torch.randn(2, 5, 16, dtype=torch.float64)

    x_ref = x.clone().requires_grad_(True)
    ref_out = ref(x_ref)
    ref_out.sum().backward()

    assert apply_mlp_activation_recompute(wrapped) == 1

    x_w = x.clone().requires_grad_(True)
    w_out = wrapped(x_w)
    w_out.sum().backward()

    assert torch.allclose(ref_out, w_out, atol=1e-10, rtol=1e-8)
    assert torch.allclose(x_ref.grad, x_w.grad, atol=1e-10, rtol=1e-8)
    ref_grads = {n: p.grad for n, p in ref.named_parameters()}
    for n, p in wrapped.named_parameters():
        assert torch.allclose(ref_grads[n], p.grad, atol=1e-10, rtol=1e-8), n


def test_only_swiglu_mlps_are_wrapped():
    model = Block()
    assert apply_mlp_activation_recompute(model) == 1  # only the MLP, not `other`


def test_count_across_multiple_mlps():
    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([SwiGLUMLP(), SwiGLUMLP(), SwiGLUMLP()])

    assert apply_mlp_activation_recompute(Stack()) == 3
