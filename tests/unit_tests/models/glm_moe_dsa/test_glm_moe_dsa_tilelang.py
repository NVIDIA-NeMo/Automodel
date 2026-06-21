# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Tests for the optional GLM-5.2 DSA TileLang kernels."""

import pytest
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa import optimized_kernels as ok
from nemo_automodel.components.models.glm_moe_dsa.layers import GlmMoeDsaMLA

# GLM-5.2 DSA kernel dims (kv_lora_rank + qk_rope_head_dim == 576 is hard-coded in the kernel).
KV_LORA = 512
ROPE = 64
QK_NOPE = 128
V_HEAD = 128
QK_HEAD = QK_NOPE + ROPE
N_HEADS = 16
IDX_HEADS = 16
IDX_DIM = 128
TOPK = 64
T = 256

_run_gpu = (
    ok.is_dsa_kernel_available("sparse_attn") and ok.is_dsa_kernel_available("indexer") and torch.cuda.is_available()
)
requires_kernels = pytest.mark.skipif(not _run_gpu, reason="requires tilelang kernels (CUDA + tilelang installed)")


def test_is_dsa_kernel_available_returns_bool():
    assert isinstance(ok.is_dsa_kernel_available("indexer"), bool)
    assert isinstance(ok.is_dsa_kernel_available("sparse_attn"), bool)
    with pytest.raises(ValueError):
        ok.is_dsa_kernel_available("nope")


def test_should_use_tilelang_torch_is_false():
    tensors = (torch.zeros(2),)
    assert ok.should_use_tilelang("torch", available=True, kernel_name="x", tensors=tensors) is False


def test_should_use_tilelang_forced_but_unavailable_raises():
    tensors = (torch.zeros(2),)
    with pytest.raises(RuntimeError, match="TileLang backend was requested"):
        ok.should_use_tilelang("tilelang", available=False, kernel_name="indexer", tensors=tensors)
    with pytest.raises(RuntimeError, match="TileLang backend was requested"):
        ok.should_use_tilelang("tilelang", available=True, kernel_name="indexer", tensors=tensors, require_bf16=True)


def _dense_indexer_logits(index_q, index_k, weights_raw, scale):
    """Dense reference for the lighting indexer: relu(q dot k * scale), then head-weighted sum."""
    scores = torch.relu(torch.einsum("thd,sd->ths", index_q.float(), index_k.float()) * scale)
    logits = torch.einsum("th,ths->ts", weights_raw.float(), scores)
    causal = torch.ones(logits.shape[-2], logits.shape[-1], dtype=torch.bool, device=logits.device).triu(1)
    return logits.masked_fill(causal, float("-inf"))


@requires_kernels
def test_indexer_topk_matches_dense():
    torch.manual_seed(0)
    dev = "cuda"
    scale = IDX_DIM**-0.5
    index_q = torch.randn(T, IDX_HEADS, IDX_DIM, device=dev, dtype=torch.bfloat16)
    index_k = torch.randn(T, IDX_DIM, device=dev, dtype=torch.bfloat16)
    weights_proj = torch.randn(T, IDX_HEADS, device=dev, dtype=torch.float32)

    head_weights = (weights_proj * (IDX_HEADS**-0.5) * scale).contiguous()
    cu_seqlens = torch.tensor([0, T], device=dev, dtype=torch.int32)
    topk_tl = ok.tilelang_indexer_topk(index_q.contiguous(), index_k.contiguous(), head_weights, cu_seqlens, TOPK)
    assert topk_tl.shape == (T, 1, TOPK)
    assert topk_tl.dtype == torch.int32

    logits = _dense_indexer_logits(index_q, index_k, weights_proj * (IDX_HEADS**-0.5), scale)
    topk_ref = logits.topk(TOPK, dim=-1).indices

    tilelang_topk = topk_tl.squeeze(1)
    overlaps = []
    for t in range(TOPK, T):
        actual = set(tilelang_topk[t].tolist()) - {-1}
        expected = set(topk_ref[t].tolist())
        overlaps.append(len(actual & expected) / len(expected))
    mean_overlap = sum(overlaps) / len(overlaps)
    assert mean_overlap > 0.97, f"indexer top-k set overlap too low: {mean_overlap:.4f}"


def _causal_topk_indices(num_tokens, topk, device):
    idx = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)
    for t in range(num_tokens):
        n = min(t + 1, topk)
        idx[t, :n] = torch.randperm(t + 1, device=device)[:n].to(torch.int32)
    return idx


def _dense_sparse_mla(q_nope, q_pe, kv_c, k_pe, w_kc, w_vc, topk_idx, scale):
    """Dense MLA attention restricted to the per-query top-k key list."""
    k_nope = torch.einsum("tc,hjc->thj", kv_c.float(), w_kc.float())
    v = torch.einsum("tc,hjc->thj", kv_c.float(), w_vc.float())
    k = torch.cat([k_nope, k_pe.float().unsqueeze(1).expand(-1, q_nope.shape[1], -1)], dim=-1)
    q = torch.cat([q_nope.float(), q_pe.float()], dim=-1)
    scores = torch.einsum("thd,shd->ths", q, k) * scale

    keep = torch.zeros(scores.shape[0], scores.shape[-1], dtype=torch.bool, device=scores.device)
    valid_mask = topk_idx >= 0
    rows = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1).expand_as(topk_idx)[valid_mask]
    cols = topk_idx.long()[valid_mask]
    keep[rows, cols] = True
    scores = scores.masked_fill(~keep.unsqueeze(1), float("-inf"))
    probs = scores.softmax(dim=-1)
    return torch.einsum("ths,shj->thj", probs, v)


@requires_kernels
def test_sparse_mla_absorbed_matches_dense():
    torch.manual_seed(0)
    dev = "cuda"
    scale = QK_HEAD**-0.5
    q_nope = torch.randn(T, N_HEADS, QK_NOPE, device=dev, dtype=torch.bfloat16)
    q_pe = torch.randn(T, N_HEADS, ROPE, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(T, KV_LORA, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(T, ROPE, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N_HEADS, QK_NOPE + V_HEAD, KV_LORA, device=dev, dtype=torch.bfloat16) * (KV_LORA**-0.5)
    w_kc = w[:, :QK_NOPE, :]
    w_vc = w[:, QK_NOPE:, :]
    topk_idx = _causal_topk_indices(T, TOPK, dev)

    q_absorbed = torch.einsum("thd,hdc->thc", q_nope, w_kc)
    q_tl = torch.cat([q_absorbed, q_pe], dim=-1).to(torch.bfloat16)
    kv_latent = torch.cat([kv_c, k_pe], dim=-1).unsqueeze(1).to(torch.bfloat16)
    out_tl = ok.tilelang_sparse_attention(q_tl, kv_latent, topk_idx.view(T, 1, TOPK).contiguous(), w_vc, scale)

    out_ref = _dense_sparse_mla(q_nope, q_pe, kv_c, k_pe, w_kc, w_vc, topk_idx, scale)
    cosine = torch.nn.functional.cosine_similarity(out_tl.float().flatten(), out_ref.float().flatten(), dim=0).item()
    assert cosine > 0.99, f"sparse MLA cosine vs dense oracle too low: {cosine:.5f}"


def _small_dsa_config():
    return GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=512,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        attention_bias=False,
        kv_lora_rank=KV_LORA,
        q_lora_rank=256,
        qk_head_dim=QK_HEAD,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_HEAD,
        index_n_heads=IDX_HEADS,
        index_head_dim=IDX_DIM,
        index_topk=TOPK,
        mlp_layer_types=["dense"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        torch_dtype="bfloat16",
    )


@requires_kernels
def test_mla_forward_backward_tilelang_thd():
    torch.manual_seed(0)
    dev = "cuda"
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend).to(dev).to(torch.bfloat16)
    assert mla.attn_func is None and mla.attn_module is None

    angles = torch.randn(T, config.qk_rope_head_dim // 2, device=dev, dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    x = torch.randn(T, config.hidden_size, device=dev, dtype=torch.bfloat16, requires_grad=True)
    cu_seqlens = torch.tensor([0, T], device=dev, dtype=torch.int32)

    out = mla(x, freqs_cis, attention_mask=None, cu_seqlens=cu_seqlens, qkv_format="thd")
    assert out.shape == (T, config.hidden_size)
    assert torch.isfinite(out).all()

    out.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert mla.kv_b_proj.weight.grad is not None and torch.isfinite(mla.kv_b_proj.weight.grad).all()
    assert mla.q_b_proj.weight.grad is not None and torch.isfinite(mla.q_b_proj.weight.grad).all()


@requires_kernels
def test_mla_tilelang_rejects_bshd():
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend).to("cuda").to(torch.bfloat16)

    angles = torch.randn(2, T, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    x = torch.randn(2, T, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="THD"):
        mla(x, freqs_cis, attention_mask=None, qkv_format="bshd")
