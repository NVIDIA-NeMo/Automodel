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

"""Explicit FP32 model arithmetic must survive an enclosing autocast region."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture(
    params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))]
)
def device(request):
    return torch.device(request.param)


@pytest.fixture(params=["shared", "gemma4", "kimi_k3", "muse_spark", "dsv4_hash"])
def router(request, device):
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.layers import Gate

    config = MoEConfig(
        dim=128,
        inter_dim=256,
        moe_inter_dim=64,
        n_routed_experts=16,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0,
        aux_loss_coeff=0,
        score_func="sigmoid_with_bias",
        route_scale=1.0,
        norm_topk_prob=True,
        dtype=torch.float32,
    )
    torch.manual_seed(42)
    if request.param == "shared":
        gate = Gate(config, gate_precision=torch.float32)
    elif request.param == "gemma4":
        from nemo_automodel.components.models.gemma4_moe.model import Gemma4Gate

        gate = Gemma4Gate(
            SimpleNamespace(
                hidden_size=128,
                num_experts=16,
                top_k_experts=2,
                rms_norm_eps=1e-6,
                torch_dtype=torch.float32,
            )
        )
    elif request.param == "kimi_k3":
        from nemo_automodel.components.models.kimi_k3.model import KimiK3Gate

        gate = KimiK3Gate(config, gate_precision=torch.float32)
    elif request.param == "muse_spark":
        from nemo_automodel.components.models.muse_spark.layers import MuseSparkGate

        gate = MuseSparkGate(config, router_output_multiplier=0.7, division_eps=1e-6)
    else:
        from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4HashGate

        gate = DeepseekV4HashGate(SimpleNamespace(hidden_size=128, vocab_size=32), config)
    gate.to(device)
    with torch.no_grad():
        for parameter in gate.parameters():
            if parameter.ndim == 2:
                parameter.normal_(0, 0.1)
    return gate


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
def test_fp32_router_selection_weights_and_gradients_survive_autocast(router, device, storage_dtype):
    router.to(storage_dtype)
    x = torch.randn(4096, 128, device=device, dtype=storage_dtype, requires_grad=True)
    mask = torch.ones(4096, device=device, dtype=torch.bool)
    expected, indices, _ = router(x, mask, None)
    upstream = torch.randn_like(expected)
    parameters = (x, *router.parameters())
    gradients = torch.autograd.grad((expected * upstream).sum(), parameters)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual, actual_indices, _ = router(x, mask, None)
    actual_gradients = torch.autograd.grad((actual * upstream).sum(), parameters)
    torch.testing.assert_close(actual_indices, indices, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_gradients, gradients):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("autocast_dtype", [torch.bfloat16, torch.float16])
def test_inkling_fp32_router_selection_weights_and_gradients_survive_autocast(device, storage_dtype, autocast_dtype):
    from nemo_automodel.components.models.inkling.layers import InklingGate

    torch.manual_seed(2714489)
    router = InklingGate(
        SimpleNamespace(
            hidden_size=64,
            n_routed_experts=8,
            n_shared_experts=2,
            num_experts_per_tok=2,
            route_scale=1.0,
            torch_dtype=storage_dtype,
        )
    ).to(device)
    with torch.no_grad():
        router.weight.normal_(0, 0.1)
        router.e_score_correction_bias.uniform_(-0.01, 0.01)
    correction_bias = router.e_score_correction_bias.detach().clone()
    x = torch.randn(4096, 64, device=device, dtype=storage_dtype, requires_grad=True)
    with torch.autocast(device.type, enabled=False):
        expected, indices, shared = router(x)
    upstream = torch.randn_like(expected)
    shared_upstream = torch.randn_like(shared)
    parameters = (x, router.weight, router.global_scale)
    gradients = torch.autograd.grad((expected * upstream).sum() + (shared * shared_upstream).sum(), parameters)
    with torch.autocast(device.type, dtype=autocast_dtype):
        actual, actual_indices, actual_shared = router(x)
    actual_gradients = torch.autograd.grad(
        (actual * upstream).sum() + (actual_shared * shared_upstream).sum(), parameters
    )
    torch.testing.assert_close(actual_indices, indices, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_shared, shared, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_gradients, gradients):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert router.e_score_correction_bias.dtype == torch.float32
    torch.testing.assert_close(router.e_score_correction_bias, correction_bias, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["deepseek", "gpt_oss", "ernie", "llama"])
@pytest.mark.parametrize("fused", [False, True])
def test_long_position_rope_matches_fp32_under_autocast(device, family, fused):
    positions = torch.tensor([[0, 257, 8191, 65535]], device=device)
    frequency = 1.0 / (10000 ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64))
    angles = positions.float().unsqueeze(-1) * frequency
    x = torch.zeros(1, 4, 1, 64, device=device, dtype=torch.bfloat16)
    if family == "deepseek":
        from nemo_automodel.components.models.deepseek_v3.rope_utils import freqs_cis_from_position_ids

        with torch.autocast(device.type, dtype=torch.bfloat16):
            actual = freqs_cis_from_position_ids(positions, frequency, for_fused_rope=fused)
        expected = (
            angles[0].repeat_interleave(2, -1).view(4, 1, 1, 64)
            if fused
            else torch.polar(torch.ones_like(angles), angles)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif family == "gpt_oss":
        from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis

        rope = RotaryEmbedding(64, 10000, torch.bfloat16, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            actual = position_ids_to_freqs_cis(rope, positions, for_fused_rope=fused)
            cos, sin = rope._compute_cos_sin(8192)
        expected = torch.cat((angles, angles), -1) if fused else torch.cat((angles.cos(), angles.sin()), -1)
        torch.testing.assert_close(actual, expected[0].view(4, 1, 1, 64) if fused else expected, rtol=0, atol=0)
        reference = torch.arange(8192, device=device, dtype=torch.float32)[:, None] * frequency
        torch.testing.assert_close(cos, reference.cos(), rtol=0, atol=0)
        torch.testing.assert_close(sin, reference.sin(), rtol=0, atol=0)
    elif family == "ernie":
        from nemo_automodel.components.models.ernie4_5.rope_utils import Ernie4_5RotaryEmbedding

        rope = Ernie4_5RotaryEmbedding(SimpleNamespace(hidden_size=64, num_attention_heads=1, rope_theta=10000), device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            cos, sin = rope(x, positions)
        torch.testing.assert_close(cos, angles.cos().repeat_interleave(2, -1), rtol=0, atol=0)
        torch.testing.assert_close(sin, angles.sin().repeat_interleave(2, -1), rtol=0, atol=0)
    else:
        from transformers import LlamaConfig

        from nemo_automodel.components.models.llama.rope_utils import LlamaRotaryEmbedding

        rope = LlamaRotaryEmbedding(
            LlamaConfig(
                hidden_size=64,
                num_attention_heads=1,
                head_dim=64,
                rope_parameters={"rope_type": "default", "rope_theta": 10000},
            ),
            rope_fusion=False,
        ).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            cos, sin = rope(x, positions)
        torch.testing.assert_close(cos, torch.cat((angles, angles), -1).cos(), rtol=0, atol=0)
        torch.testing.assert_close(sin, torch.cat((angles, angles), -1).sin(), rtol=0, atol=0)


def test_fp32_lm_head_output_and_gradients_survive_autocast(device):
    from nemo_automodel.components.models.common.utils import compute_lm_head_logits

    torch.manual_seed(42)
    head = nn.Linear(64, 32, bias=False, device=device)
    x = torch.randn(2, 4, 64, device=device, requires_grad=True)
    expected = compute_lm_head_logits(head, x, fp32_lm_head=True).logits
    gradients = torch.autograd.grad(expected.square().sum(), (x, head.weight))
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = compute_lm_head_logits(head, x, fp32_lm_head=True).logits
    actual_gradients = torch.autograd.grad(actual.square().sum(), (x, head.weight))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_gradients, gradients):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_fp32_hyperconnection_projection_survives_autocast(device):
    from nemo_automodel.components.models.deepseek_v41.layers import _hc_project_core

    torch.manual_seed(42)
    x = torch.randn(2, 4, 3, 32, device=device, requires_grad=True)
    fn = torch.randn(15, 96, device=device, requires_grad=True)
    base = torch.randn(15, device=device, requires_grad=True)
    scale = torch.randn(3, device=device, requires_grad=True)
    expected = _hc_project_core(x, fn, base, scale, 3, 1e-6, 1e-6)
    gradients = torch.autograd.grad(sum(t.square().sum() for t in expected), (x, fn, base, scale))
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = _hc_project_core(x, fn, base, scale, 3, 1e-6, 1e-6)
    actual_gradients = torch.autograd.grad(sum(t.square().sum() for t in actual), (x, fn, base, scale))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    for a, e in zip(actual_gradients, gradients):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_fp32_indexer_scores_survive_autocast(device):
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import indexer_scores_torch

    torch.manual_seed(42)
    q = torch.randn(2, 5, 3, 32, device=device)
    kv = torch.randn(2, 7, 32, device=device)
    weights = torch.randn(2, 5, 3, device=device)
    expected = indexer_scores_torch(q, kv, weights, 0.125)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = indexer_scores_torch(q, kv, weights, 0.125)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.topk(3).indices, expected.topk(3).indices, rtol=0, atol=0)


@pytest.mark.parametrize("kernel", ["sparse", "dense"])
def test_fp32_sparse_attention_oracle_survives_autocast(device, kernel):
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import (
        dense_attention_topk_torch,
        sparse_attention_torch,
    )

    torch.manual_seed(42)
    q = torch.randn(2, 5, 3, 32, device=device, requires_grad=True)
    kv = torch.randn(2, 7, 32, device=device, requires_grad=True)
    sinks = torch.randn(3, device=device, requires_grad=True)
    indices = torch.tensor([0, 2, 5, -1], device=device).expand(2, 5, -1)
    operation = sparse_attention_torch if kernel == "sparse" else dense_attention_topk_torch
    expected = operation(q, kv, sinks, indices, 0.125)
    gradients = torch.autograd.grad(expected.square().sum(), (q, kv, sinks))
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = operation(q, kv, sinks, indices, 0.125)
    actual_gradients = torch.autograd.grad(actual.square().sum(), (q, kv, sinks))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, e in zip(actual_gradients, gradients):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["dsv4", "glm5"])
def test_hyperconnection_coefficients_survive_autocast(device, family):
    if family == "dsv4":
        from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4HyperConnection

        layer = DeepseekV4HyperConnection(3, 32, 4, 1e-6, 1e-6).to(device)
        operation = layer.compute_weights
    else:
        from nemo_automodel.components.models.glm5_next.layers import Glm5NextHyperConnection

        layer = Glm5NextHyperConnection(
            SimpleNamespace(
                hc_mult=3,
                hc_sinkhorn_iters=4,
                hc_eps=1e-6,
                rms_norm_eps=1e-6,
                hidden_size=32,
                torch_dtype=torch.float32,
            )
        ).to(device)
        operation = layer
    torch.manual_seed(42)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(0, 0.1)
    x = torch.randn(2, 4, 3, 32, device=device, requires_grad=True)
    expected = operation(x)
    parameters = (x, *layer.parameters())
    gradients = torch.autograd.grad(sum(t.square().sum() for t in expected), parameters)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = operation(x)
    actual_gradients = torch.autograd.grad(sum(t.square().sum() for t in actual), parameters)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    for a, e in zip(actual_gradients, gradients):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["hyperhead", "compressor"])
def test_fp32_stream_collapse_and_compression_survive_autocast(device, operation):
    from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4HyperHead
    from nemo_automodel.components.models.deepseek_v41.attention import _CompressorLinear

    torch.manual_seed(42)
    if operation == "hyperhead":
        layer = DeepseekV4HyperHead(3, 32, 1e-6, 1e-6).to(device)
        layer.init_weights(0.1)
        x = torch.randn(2, 4, 3, 32, device=device)
    else:
        layer = _CompressorLinear(32, 16, bias=False, device=device)
        x = torch.randn(2, 4, 32, device=device)
    expected = layer(x)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_recurrent_fp32_state_survives_autocast(device):
    from nemo_automodel.components.models.glm5_next.layers import _torch_recurrent_kda

    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 6, 2, 8, device=device, requires_grad=True) for _ in range(3)]
    g = torch.randn_like(q).sigmoid().log().requires_grad_()
    beta = torch.randn(1, 6, 2, device=device).sigmoid().requires_grad_()
    boundaries = torch.tensor([0, 3, 6], device=device, dtype=torch.int32)
    expected = _torch_recurrent_kda(q, k, v, g, beta, boundaries)
    gradients = torch.autograd.grad(expected.square().sum(), (q, k, v, g, beta))
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = _torch_recurrent_kda(q, k, v, g, beta, boundaries)
    actual_gradients = torch.autograd.grad(actual.square().sum(), (q, k, v, g, beta))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, e in zip(actual_gradients, gradients):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_step5_region_scores_survive_autocast(device):
    from nemo_automodel.components.models.step5.sparse import indexer_score_tile

    torch.manual_seed(42)
    q = torch.randn(1, 8, 2, 3, 32, device=device, dtype=torch.bfloat16)
    summary = torch.randn(1, 5, 32, device=device, dtype=torch.bfloat16)
    weights = torch.randn(1, 8, 2, 3, device=device)
    expected = indexer_score_tile(q, weights, summary, 0, 8)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = indexer_score_tile(q, weights, summary, 0, 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float16])
def test_qwen35_rotary_values_and_buffers_survive_autocast(device, compute_dtype):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from nemo_automodel.components.models.qwen3_5.model import Fp32SafeQwen3_5TextRotaryEmbedding

    config = Qwen3_5TextConfig(hidden_size=128, num_attention_heads=2, num_key_value_heads=2, head_dim=64)
    rope = Fp32SafeQwen3_5TextRotaryEmbedding(config).to(device)
    x = torch.zeros(1, 4, 128, device=device, dtype=compute_dtype)
    positions = torch.tensor([[0, 257, 8191, 65535]], device=device).expand(3, -1, -1)
    buffers = {name: value.clone() for name, value in rope.named_buffers()}
    expected = rope(x, positions)
    with torch.autocast(device.type, dtype=compute_dtype):
        actual = rope(x, positions)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    for name, value in rope.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        assert value.dtype == torch.float32


def test_glm_dsa_key_selection_survives_autocast(device):
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm_moe_dsa.layers import GlmMoeDsaIndexer

    config = SimpleNamespace(
        index_n_heads=8,
        index_head_dim=32,
        qk_rope_head_dim=16,
        index_topk=8,
        q_lora_rank=32,
        hidden_size=128,
        torch_dtype=torch.bfloat16,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    torch.manual_seed(42)
    indexer = GlmMoeDsaIndexer(config, backend).to(device)
    x = torch.randn(2, 256, 128, device=device, dtype=torch.bfloat16)
    residual = torch.randn(2, 256, 32, device=device, dtype=torch.bfloat16)
    angles = torch.zeros(2, 256, 8, device=device)
    frequencies = torch.polar(torch.ones_like(angles), angles)
    # CUDA autocast promotes LayerNorm to FP32. Hold its output constant so
    # this comparison isolates score precision from the norm's precision policy.
    normalized = []
    handle = indexer.k_norm.register_forward_hook(lambda module, inputs, output: normalized.append(output.detach()))
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = indexer(x, residual, frequencies)
    handle.remove()
    handle = indexer.k_norm.register_forward_hook(lambda module, inputs, output: normalized[0])
    expected = indexer(x, residual, frequencies)
    handle.remove()
    # Projections have BF16 operands in both runs. FP32 score formation and
    # head reduction must preserve the selected key sets under root autocast.
    torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values, rtol=0, atol=0)


def test_muse_vision_rotary_coefficients_survive_autocast(device):
    from nemo_automodel.components.models.muse_spark.vision import _rope_frequencies

    expected = _rope_frequencies(64, 40, 40, 10000, device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = _rope_frequencies(64, 40, 40, 10000, device)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Each adjacent pair is a unit complex phase.
    torch.testing.assert_close(actual.square().sum(-1), torch.ones_like(actual[..., 0]))


def test_compressor_fp32_pooling_survives_autocast(device):
    from nemo_automodel.components.models.deepseek_v41.attention import _Compressor

    torch.manual_seed(42)
    layer = _Compressor(
        SimpleNamespace(hidden_size=128, head_dim=32, rms_norm_eps=1e-6), ratio=4, dtype=torch.bfloat16
    ).to(device)
    x = torch.randn(2, 256, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    expected = layer(x)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_draft_confidence_logits_survive_autocast(device):
    from nemo_automodel.components.models.deepseek_v41.dspark import _DeepseekV41DSparkConfidenceHead

    torch.manual_seed(42)
    layer = _DeepseekV41DSparkConfidenceHead(128, 32).to(device)
    x = torch.randn(2, 256, 128, device=device, dtype=torch.bfloat16)
    markov = torch.randn(2, 256, 32, device=device, dtype=torch.bfloat16)
    expected = layer(x, markov)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = layer(x, markov)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qsa_route_selection_survives_autocast(device):
    from nemo_automodel.components.models.qwen3_8_flash_next.qsa import select_qsa_token_ids

    torch.manual_seed(42)
    q = torch.randn(2, 256, 8, 32, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 64, 1, 32, device=device, dtype=torch.bfloat16)
    lengths = torch.tensor([256, 256], device=device)
    expected = select_qsa_token_ids(q, k, lengths, token_budget=16, compress_ratio=4)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = select_qsa_token_ids(q, k, lengths, token_budget=16, compress_ratio=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qsa_attention_outputs_and_gradients_survive_autocast(device):
    from nemo_automodel.components.models.qwen3_8_flash_next.qsa import gathered_qsa_gqa_attention

    torch.manual_seed(42)
    q = torch.randn(2, 8, 4, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    k, v = [torch.randn(2, 16, 2, 32, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
    ids = torch.tensor([0, 2, 5, 9], device=device).expand(2, 8, -1)
    expected = gathered_qsa_gqa_attention(q, k, v, ids)
    gradients = torch.autograd.grad(expected.float().square().sum(), (q, k, v))
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = gathered_qsa_gqa_attention(q, k, v, ids)
    actual_gradients = torch.autograd.grad(actual.float().square().sum(), (q, k, v))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, e in zip(actual_gradients, gradients):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("score_type", ["max", "lse"])
def test_minimax_sparse_block_selection_survives_autocast(device, score_type):
    from nemo_automodel.components.models.minimax_m3_vl.layers import select_sparse_blocks

    torch.manual_seed(42)
    q = torch.randn(2, 256, 8, 32, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 256, 1, 32, device=device, dtype=torch.bfloat16)
    options = dict(block_size=4, topk_blocks=4, init_blocks=1, local_blocks=1, score_type=score_type)
    expected = select_sparse_blocks(q, k, **options)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = select_sparse_blocks(q, k, **options)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gemma_fp32_sdpa_outputs_and_gradients_survive_autocast(device):
    from nemo_automodel.components.models.gemma4_moe.sdpa_fp32 import sdpa_fp32_attention_forward

    module = SimpleNamespace(num_key_value_groups=1, is_causal=False, training=False)
    torch.manual_seed(42)
    q, k, v = [torch.randn(2, 4, 8, 32, device=device, requires_grad=True) for _ in range(3)]
    expected = sdpa_fp32_attention_forward(module, q, k, v)[0]
    gradients = torch.autograd.grad(expected.square().sum(), (q, k, v))
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = sdpa_fp32_attention_forward(module, q, k, v)[0]
    actual_gradients = torch.autograd.grad(actual.square().sum(), (q, k, v))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, e in zip(actual_gradients, gradients):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
