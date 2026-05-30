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

"""Stage-2 tests: block-sparse DSA indexer for MiniMax M3 sparse-attention layers."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.minimax_m3_vl.layers import (
    MiniMaxM3Indexer,
    build_block_sparse_attn_bias,
)

from .test_minimax_m3_parity import _gemma_rmsnorm, _partial_neox_rope, _swiglu_oai


def _brute_block_select(idx_q, idx_k, block_size, topk, init_blocks, local_blocks, num_q_heads):
    """Independent O(T^2) reference: returns additive bias [B, num_q_heads, T, T]."""
    bsz, seqlen, h_idx, dim = idx_q.shape
    scale = dim**-0.5
    rep = num_q_heads // h_idx
    bias = torch.full((bsz, num_q_heads, seqlen, seqlen), float("-inf"))
    for b in range(bsz):
        for h in range(h_idx):
            for i in range(seqlen):
                nb = i // block_size + 1
                bscore = {}
                for blk in range(nb):
                    keys = [j for j in range(blk * block_size, min((blk + 1) * block_size, seqlen)) if j <= i]
                    if keys:
                        bscore[blk] = max(
                            (idx_q[b, i, h].float() @ idx_k[b, j, 0].float()).item() * scale for j in keys
                        )
                forced = {i // block_size} if local_blocks > 0 else set()
                forced |= set(range(min(init_blocks, nb)))
                rest = sorted((blk for blk in bscore if blk not in forced), key=lambda x: bscore[x], reverse=True)
                selected = set(forced)
                for blk in rest:
                    if len(selected) >= min(topk, nb):
                        break
                    selected.add(blk)
                for blk in selected:
                    for j in range(blk * block_size, min((blk + 1) * block_size, seqlen)):
                        if j <= i:
                            bias[b, h * rep : (h + 1) * rep, i, j] = 0.0
    return bias


@pytest.mark.parametrize("block_size,topk", [(4, 2), (4, 3), (8, 1), (4, 1)])
def test_block_sparse_bias_matches_brute_force(block_size, topk):
    torch.manual_seed(0)
    idx_q = torch.randn(2, 16, 2, 8)
    idx_k = torch.randn(2, 16, 1, 8)
    mine = build_block_sparse_attn_bias(
        idx_q,
        idx_k,
        block_size=block_size,
        topk_blocks=topk,
        init_blocks=0,
        local_blocks=1,
        num_q_heads=4,
        score_type="max",
    )
    ref = _brute_block_select(idx_q, idx_k, block_size, topk, 0, 1, 4)
    # Compare the boolean attend/ignore pattern (selection is what matters).
    assert torch.equal(mine == 0, ref == 0)


def test_dense_layers_have_no_indexer_sparse_layers_do(sparse_model):
    layers = sparse_model.model.layers
    assert layers["0"].self_attn.indexer is None  # sparse_attention_freq[0] == 0
    for li in ("1", "2"):
        assert isinstance(layers[li].self_attn.indexer, MiniMaxM3Indexer)


def test_sparse_model_forward_finite(sparse_model):
    cfg = sparse_model.config
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits = sparse_model(ids)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_adapter_index_key_naming_and_roundtrip(sparse_model):
    adapter = sparse_model.state_dict_adapter
    native = {k: v.clone() for k, v in sparse_model.state_dict().items()}
    assert any("self_attn.indexer.index_q_proj" in k for k in native)

    hf = adapter.to_hf(native)
    # HF layout has a flat self_attn.index_q_proj (no 'indexer' segment).
    assert any(k.endswith("self_attn.index_q_proj.weight") and "indexer" not in k for k in hf)
    assert any(k.endswith("self_attn.index_k_norm.weight") and "indexer" not in k for k in hf)

    back = adapter.from_hf(hf)
    assert set(back.keys()) == set(native.keys())
    for key in native:
        assert torch.allclose(native[key].float(), back[key].float(), atol=1e-6), key


def _ref_forward_sparse(hf, ids, cfg):
    """Reference forward with brute-force block-sparse selection on sparse layers."""
    sa = cfg.sparse_attention_config
    block_size, topk = sa["sparse_block_size"], sa["sparse_topk_blocks"]
    init_b, local_b = sa["sparse_init_block"], sa["sparse_local_block"]
    n_idx, idx_dim = sa["sparse_num_index_heads"], sa["sparse_index_dim"]
    sparse_freq = sa["sparse_attention_freq"]

    H, Hkv, Dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    eps, rdim, base = cfg.rms_norm_eps, cfg.rotary_dim, cfg.rope_theta
    alpha, limit, scale = cfg.swiglu_alpha, cfg.swiglu_limit, cfg.routed_scaling_factor
    topk_e = cfg.num_experts_per_tok
    bsz, seq = ids.shape
    pos = torch.arange(seq)[None, :].expand(bsz, -1)
    x = F.embedding(ids, hf["model.embed_tokens.weight"])

    for li in range(cfg.num_hidden_layers):
        p = f"model.layers.{li}."
        h = _gemma_rmsnorm(x, hf[p + "input_layernorm.weight"], eps)
        q = (h @ hf[p + "self_attn.q_proj.weight"].T).view(bsz, seq, H, Dh)
        k = (h @ hf[p + "self_attn.k_proj.weight"].T).view(bsz, seq, Hkv, Dh)
        v = (h @ hf[p + "self_attn.v_proj.weight"].T).view(bsz, seq, Hkv, Dh)
        q = _gemma_rmsnorm(q, hf[p + "self_attn.q_norm.weight"], eps)
        k = _gemma_rmsnorm(k, hf[p + "self_attn.k_norm.weight"], eps)
        q, k = _partial_neox_rope(q, k, pos, rdim, base)

        if sparse_freq[li] != 0:
            iq = (h @ hf[p + "self_attn.index_q_proj.weight"].T).view(bsz, seq, n_idx, idx_dim)
            ik = (h @ hf[p + "self_attn.index_k_proj.weight"].T).view(bsz, seq, 1, idx_dim)
            iq = _gemma_rmsnorm(iq, hf[p + "self_attn.index_q_norm.weight"], eps)
            ik = _gemma_rmsnorm(ik, hf[p + "self_attn.index_k_norm.weight"], eps)
            iq, ik = _partial_neox_rope(iq, ik, pos, rdim, base)
            attn_bias = _brute_block_select(iq, ik, block_size, topk, init_b, local_b, H)
            is_causal = False
        else:
            attn_bias = None
            is_causal = True

        qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
        kh = kh.repeat_interleave(H // Hkv, dim=1)
        vh = vh.repeat_interleave(H // Hkv, dim=1)
        ao = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_bias, is_causal=is_causal, scale=Dh**-0.5)
        ao = ao.transpose(1, 2).reshape(bsz, seq, H * Dh)
        x = x + ao @ hf[p + "self_attn.o_proj.weight"].T

        h = _gemma_rmsnorm(x, hf[p + "post_attention_layernorm.weight"], eps)
        if cfg.moe_layer_freq[li] == 0:
            g = h @ hf[p + "mlp.gate_proj.weight"].T
            u = h @ hf[p + "mlp.up_proj.weight"].T
            x = x + _swiglu_oai(g, u, alpha, limit) @ hf[p + "mlp.down_proj.weight"].T
        else:
            flat = h.reshape(-1, h.shape[-1])
            logits = flat.float() @ hf[p + "block_sparse_moe.gate.weight"].T.float()
            probs = torch.sigmoid(logits)
            idx = (probs + hf[p + "block_sparse_moe.e_score_correction_bias"].float()).topk(topk_e, dim=-1).indices
            w = probs.gather(-1, idx)
            w = (w / w.sum(-1, keepdim=True)) * scale
            out = torch.zeros_like(flat)
            for tok in range(flat.shape[0]):
                for j in range(topk_e):
                    e = idx[tok, j].item()
                    ep = f"{p}block_sparse_moe.experts.{e}."
                    g = flat[tok] @ hf[ep + "w1.weight"].T
                    u = flat[tok] @ hf[ep + "w3.weight"].T
                    out[tok] += w[tok, j] * (_swiglu_oai(g, u, alpha, limit) @ hf[ep + "w2.weight"].T)
            sp = f"{p}block_sparse_moe.shared_experts."
            g = flat @ hf[sp + "gate_proj.weight"].T
            u = flat @ hf[sp + "up_proj.weight"].T
            out = out + _swiglu_oai(g, u, alpha, limit) @ hf[sp + "down_proj.weight"].T
            x = x + out.reshape(bsz, seq, -1)

    x = _gemma_rmsnorm(x, hf["model.norm.weight"], eps)
    return x @ hf["lm_head.weight"].T


def test_sparse_e2e_parity_vs_reference(sparse_model):
    torch.manual_seed(0)
    cfg = sparse_model.config
    hf = {k: v.float() for k, v in sparse_model.state_dict_adapter.to_hf(sparse_model.state_dict()).items()}
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        mine = sparse_model(ids).float()
        ref = _ref_forward_sparse(hf, ids, cfg)
    max_diff = (mine - ref).abs().max().item()
    cos = F.cosine_similarity(mine.flatten(), ref.flatten(), dim=0).item()
    assert max_diff < 1e-4, f"max_diff={max_diff}"
    assert cos > 0.9999, f"cos_sim={cos}"
