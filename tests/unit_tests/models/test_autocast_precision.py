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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4HashGate
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import indexer_scores_torch
from nemo_automodel.components.models.deepseek_v41.layers import _hc_project_core
from nemo_automodel.components.models.gemma4_moe.model import Gemma4Gate
from nemo_automodel.components.models.glm5_next.layers import Glm5NextHyperConnection
from nemo_automodel.components.models.inkling.layers import InklingGate
from nemo_automodel.components.models.kimi_k3.model import KimiK3Gate
from nemo_automodel.components.moe.layers import Gate
from tests.utils.lora_storage import tiny_moe_config


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return request.param


def _router(kind):
    config = tiny_moe_config(dtype=torch.float32)
    config.n_activated_experts = 2
    if kind == "common":
        return Gate(config, gate_precision=torch.float32)
    if kind == "kimi":
        config.score_func = "sigmoid_with_bias"
        return KimiK3Gate(config)
    if kind == "hash":
        return DeepseekV4HashGate(SimpleNamespace(hidden_size=16, vocab_size=32), config)
    if kind == "gemma":
        return Gemma4Gate(
            SimpleNamespace(
                hidden_size=16, num_experts=4, top_k_experts=2, torch_dtype=torch.float32, rms_norm_eps=1e-6
            )
        )
    return InklingGate(
        SimpleNamespace(
            n_routed_experts=4,
            n_shared_experts=1,
            hidden_size=16,
            route_scale=1.0,
            num_experts_per_tok=2,
            torch_dtype=torch.float32,
        )
    )


@pytest.mark.parametrize("kind", ["common", "kimi", "hash", "gemma", "inkling"])
def test_router_fp32_math_survives_recipe_autocast(kind, device):
    """Routing probabilities, expert choices and input gradients retain FP32 math."""
    torch.manual_seed(17)
    module = _router(kind).to(device)
    for p in module.parameters():
        nn.init.normal_(p, std=0.2)
    x = torch.randn(8, 16, device=device, requires_grad=True)
    args = () if kind == "inkling" else (torch.ones(8, device=device, dtype=torch.bool), None)
    expected = module(x, *args)
    ref_grad = torch.autograd.grad(expected[0].square().sum(), x)[0]
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = module(x, *args)
    grad = torch.autograd.grad(actual[0].square().sum(), x)[0]
    for out, ref in zip(actual, expected):
        if ref is not None:
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)


def test_indexer_fp32_scores_survive_recipe_autocast(device):
    """Indexer scores [batch, sequence, pooled] retain FP32 matmul precision."""
    q = torch.randn(2, 7, 3, 8, device=device, requires_grad=True)
    kv = torch.randn(2, 5, 8, device=device, requires_grad=True)
    weights = torch.randn(2, 7, 3, device=device, requires_grad=True)
    expected = indexer_scores_torch(q, kv, weights, 8**-0.5)
    ref_grads = torch.autograd.grad(expected.sum(), (q, kv, weights))
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = indexer_scores_torch(q, kv, weights, 8**-0.5)
    grads = torch.autograd.grad(actual.sum(), (q, kv, weights))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for grad, ref in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["deepseek", "glm"])
def test_hyperconnection_fp32_projection_survives_recipe_autocast(family, device):
    """Mixing coefficients and collapsed residual streams retain explicit precision."""
    x = torch.randn(2, 4, 2, 8, device=device, requires_grad=True)
    if family == "deepseek":
        args = (
            torch.randn(8, 16, device=device),
            torch.randn(8, device=device),
            torch.ones(3, device=device),
            2,
            1e-6,
            1e-6,
        )
        run = lambda x: _hc_project_core(x, *args)
    else:
        run = Glm5NextHyperConnection(
            SimpleNamespace(
                hc_mult=2, hc_sinkhorn_iters=3, hc_eps=1e-6, rms_norm_eps=1e-6, hidden_size=8, torch_dtype=torch.float32
            )
        ).to(device)
        nn.init.normal_(run.fn, std=0.2)
    expected = run(x)
    ref_grad = torch.autograd.grad(sum(t.square().sum() for t in expected), x)[0]
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = run(x)
    grad = torch.autograd.grad(sum(t.square().sum() for t in actual), x)[0]
    for out, ref in zip(actual, expected):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
