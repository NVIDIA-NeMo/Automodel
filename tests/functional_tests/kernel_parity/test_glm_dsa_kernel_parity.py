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

"""GLM DSA TileLang kernel parity, including LoRA and input gradients."""

import pytest
import torch

from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa import optimized_kernels as ok
from nemo_automodel.components.models.glm_moe_dsa.layers import GlmMoeDsaMLA
from tests.unit_tests.models.glm_moe_dsa.test_glm_moe_dsa_tilelang import (
    IDX_DIM,
    IDX_HEADS,
    KV_LORA,
    N_HEADS,
    QK_HEAD,
    QK_NOPE,
    ROPE,
    TOPK,
    V_HEAD,
    T,
    _causal_topk_indices,
    _dense_indexer_logits,
    _dense_sparse_mla,
    _small_dsa_config,
    requires_kernels,
)

pytestmark = pytest.mark.timeout(300)


@requires_kernels
def test_indexer_topk_matches_dense() -> None:
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


@requires_kernels
def test_indexer_topk_short_sequence_below_index_topk() -> None:
    """When the packed sequence has fewer keys than index_topk, the indexer must not

    crash on torch.topk ("selected index k out of range"); it should select the
    available keys and pad back to the fixed index_topk width with the -1 sentinel so
    the sparse-MLA kernel still gets a width that is a multiple of block_I (64).
    """
    torch.manual_seed(0)
    dev = "cuda"
    short_t = TOPK // 2  # e.g. 32 < TOPK (64): triggers the short-sequence path
    assert short_t < TOPK
    scale = IDX_DIM**-0.5
    index_q = torch.randn(short_t, IDX_HEADS, IDX_DIM, device=dev, dtype=torch.bfloat16)
    index_k = torch.randn(short_t, IDX_DIM, device=dev, dtype=torch.bfloat16)
    weights_proj = torch.randn(short_t, IDX_HEADS, device=dev, dtype=torch.float32)

    head_weights = (weights_proj * (IDX_HEADS**-0.5) * scale).contiguous()
    cu_seqlens = torch.tensor([0, short_t], device=dev, dtype=torch.int32)
    topk_tl = ok.tilelang_indexer_topk(index_q.contiguous(), index_k.contiguous(), head_weights, cu_seqlens, TOPK)

    # Width stays at the fixed index_topk (multiple of block_I) even though only short_t keys exist.
    assert topk_tl.shape == (short_t, 1, TOPK)
    assert TOPK % 64 == 0
    tilelang_topk = topk_tl.squeeze(1)
    # Every selected (non-sentinel) index must be a valid causal key position (< short_t).
    valid = tilelang_topk[tilelang_topk != -1]
    assert int(valid.min()) >= 0 and int(valid.max()) < short_t
    # Each query keeps at most (its causal window) real keys; the rest are -1 padding.
    for t in range(short_t):
        n_real = int((tilelang_topk[t] != -1).sum())
        assert n_real <= t + 1


@requires_kernels
def test_sparse_mla_absorbed_matches_dense() -> None:
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


@requires_kernels
def test_sparse_mla_lora_effective_weight_matches_dense_forward_and_backward() -> None:
    """BF16 TileLang must match the dense oracle for materialized LoRA K/V weights."""
    torch.manual_seed(0)
    dev = "cuda"
    base = torch.nn.Linear(
        KV_LORA,
        N_HEADS * (QK_NOPE + V_HEAD),
        bias=False,
        device=dev,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        base.weight.normal_(std=KV_LORA**-0.5)
    tilelang_lora = LinearLoRA(base, dim=8, alpha=16).train()
    reference_lora = LinearLoRA(base, dim=8, alpha=16).train()
    with torch.no_grad():
        tilelang_lora.lora_A.weight.normal_(std=0.02)
        tilelang_lora.lora_B.weight.normal_(std=0.02)
        reference_lora.lora_A.weight.copy_(tilelang_lora.lora_A.weight)
        reference_lora.lora_B.weight.copy_(tilelang_lora.lora_B.weight)

    q_nope = torch.randn(T, N_HEADS, QK_NOPE, device=dev, dtype=torch.bfloat16, requires_grad=True)
    q_pe = torch.randn(T, N_HEADS, ROPE, device=dev, dtype=torch.bfloat16, requires_grad=True)
    kv_c = torch.randn(T, KV_LORA, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k_pe = torch.randn(T, ROPE, device=dev, dtype=torch.bfloat16, requires_grad=True)
    q_nope_ref = q_nope.detach().clone().requires_grad_(True)
    q_pe_ref = q_pe.detach().clone().requires_grad_(True)
    kv_c_ref = kv_c.detach().clone().requires_grad_(True)
    k_pe_ref = k_pe.detach().clone().requires_grad_(True)
    topk_idx = _causal_topk_indices(T, TOPK, dev)
    scale = QK_HEAD**-0.5

    weight = tilelang_lora.materialize_effective_weight().view(N_HEADS, QK_NOPE + V_HEAD, KV_LORA)
    w_kc = weight[:, :QK_NOPE, :]
    w_vc = weight[:, QK_NOPE:, :]
    q_absorbed = torch.einsum("thd,hdc->thc", q_nope, w_kc)
    q_tl = torch.cat([q_absorbed, q_pe], dim=-1)
    kv_latent = torch.cat([kv_c, k_pe], dim=-1).unsqueeze(1)
    out_tl = ok.tilelang_sparse_attention(q_tl, kv_latent, topk_idx.view(T, 1, TOPK).contiguous(), w_vc, scale)

    reference_weight = reference_lora.materialize_effective_weight().view(
        N_HEADS,
        QK_NOPE + V_HEAD,
        KV_LORA,
    )
    out_ref = _dense_sparse_mla(
        q_nope_ref,
        q_pe_ref,
        kv_c_ref,
        k_pe_ref,
        reference_weight[:, :QK_NOPE, :],
        reference_weight[:, QK_NOPE:, :],
        topk_idx,
        scale,
    )
    output_cosine = torch.nn.functional.cosine_similarity(
        out_tl.float().flatten(),
        out_ref.float().flatten(),
        dim=0,
    ).item()
    assert output_cosine > 0.99, f"LoRA sparse MLA output cosine too low: {output_cosine:.5f}"

    output_grad = torch.randn_like(out_tl)
    out_tl.backward(output_grad)
    out_ref.backward(output_grad.float())
    gradient_pairs = {
        "q_nope": (q_nope.grad, q_nope_ref.grad),
        "q_pe": (q_pe.grad, q_pe_ref.grad),
        "kv_c": (kv_c.grad, kv_c_ref.grad),
        "k_pe": (k_pe.grad, k_pe_ref.grad),
        "lora_A": (tilelang_lora.lora_A.weight.grad, reference_lora.lora_A.weight.grad),
        "lora_B": (tilelang_lora.lora_B.weight.grad, reference_lora.lora_B.weight.grad),
    }
    for name, (actual_grad, reference_grad) in gradient_pairs.items():
        assert actual_grad is not None and reference_grad is not None
        assert torch.isfinite(actual_grad).all() and torch.isfinite(reference_grad).all()
        gradient_cosine = torch.nn.functional.cosine_similarity(
            actual_grad.float().flatten(),
            reference_grad.float().flatten(),
            dim=0,
        ).item()
        assert gradient_cosine > 0.99, f"LoRA sparse MLA {name} gradient cosine too low: {gradient_cosine:.5f}"


@requires_kernels
def test_mla_forward_backward_tilelang_thd() -> None:
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
def test_mla_tilelang_rejects_bshd() -> None:
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend).to("cuda").to(torch.bfloat16)

    angles = torch.randn(2, T, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    x = torch.randn(2, T, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="THD"):
        mla(x, freqs_cis, attention_mask=None, qkv_format="bshd")
