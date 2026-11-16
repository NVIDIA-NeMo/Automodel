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

"""Check remote-model FP32 boundaries without changing neighboring BF16 math."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel._transformers.kernel_patches import apply_model_runtime_patches

pytestmark = pytest.mark.cpu_only


class _ExplicitFP32Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 64))

    def forward(self, hidden_states):
        """Project [tokens, hidden] inputs to FP32 [tokens, experts] logits."""
        logits = F.linear(hidden_states.float(), self.weight.float())
        weights, indices = logits.softmax(-1).topk(2)
        return indices, weights, logits


class _ExplicitFP32RoPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", 1 / (10000 ** (torch.arange(0, 64, 2).float() / 64)))

    def forward(self, x, position_ids):
        """Return [batch, sequence, head_dim] cos/sin for [batch, sequence] positions.

        Args:
            x: Tensor of shape [batch, sequence, hidden], providing the output dtype.
            position_ids: Integer tensor of shape [batch, sequence].

        Returns:
            Cosine and sine tensors of shape [batch, sequence, head_dim].
        """
        angles = (self.inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
        angles = torch.cat((angles, angles), -1)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)


def test_llada2_guard_preserves_fp32_routing_and_gradients():
    torch.manual_seed(4494)
    model = nn.Module()
    model.config = SimpleNamespace(
        architectures=["LLaDA2MoeModelLM"], num_experts=8, first_k_dense_replace=0, num_hidden_layers=1
    )
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.gate = _ExplicitFP32Gate()
    model.model.layers = nn.ModuleList([layer])
    gate = layer.mlp.gate
    x = torch.randn(128, 64, dtype=torch.bfloat16, requires_grad=True)
    expected = gate(x)
    upstream = torch.randn_like(expected[2])
    expected_grad = torch.autograd.grad(expected[2], (x, gate.weight), upstream)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert gate(x)[2].dtype == torch.bfloat16  # Negative control.
    apply_model_runtime_patches(model, mesh=None)
    for candidate in (model, copy.deepcopy(model)):
        guarded = candidate.model.layers[0].mlp.gate
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = guarded(x)
            neighbor = F.linear(x, torch.randn(4, 64))
        assert neighbor.dtype == torch.bfloat16
        assert actual[2].dtype == torch.float32
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        gradients = torch.autograd.grad(actual[2], (x, guarded.weight), upstream)
        for result, reference in zip(gradients, expected_grad):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_nemotron_diffusion_guard_preserves_long_position_rope():
    model = nn.Module()
    model.config = SimpleNamespace(architectures=["NemotronLabsDiffusionModel"])
    model.encoder = nn.Module()
    model.encoder.rotary_emb = _ExplicitFP32RoPE()
    x = torch.zeros(1, 8, 64, dtype=torch.bfloat16)
    positions = torch.arange(8192, 8200).view(1, -1)
    expected = model.encoder.rotary_emb(x, positions)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        unguarded = model.encoder.rotary_emb(x, positions)
    assert not torch.equal(unguarded[0], expected[0])  # Negative control.
    apply_model_runtime_patches(model, mesh=None)
    for candidate in (model, copy.deepcopy(model)):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = candidate.encoder.rotary_emb(x, positions)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
