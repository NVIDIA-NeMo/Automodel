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

"""Stage-1 parity: the NeMo M3 text backbone vs a faithful CPU transcription of
the sglang reference forward (``sglang.srt.models.minimax_m3``).

Both consume identical weights (exported to HF layout via the state-dict
adapter), CPU/float32, strict tolerance.
"""

import torch
import torch.nn.functional as F


def _gemma_rmsnorm(x, w, eps):
    d = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * (1.0 + w.float())).to(d)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _partial_neox_rope(q, k, pos, rotary_dim, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = pos.float()[:, :, None] * inv_freq[None, None, :]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[:, :, None, :]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[:, :, None, :]

    def app(x):
        xr, xp = x[..., :rotary_dim], x[..., rotary_dim:]
        xr = xr * cos + _rotate_half(xr) * sin
        return torch.cat([xr, xp], dim=-1)

    return app(q), app(k)


def _swiglu_oai(gate, up, alpha, limit):
    gate = gate.float().clamp(max=limit)
    up = up.float().clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + 1.0)


def _ref_forward(hf, ids, cfg):
    H, Hkv, Dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    eps, rdim, base = cfg.rms_norm_eps, cfg.rotary_dim, cfg.rope_theta
    alpha, limit, scale = cfg.swiglu_alpha, cfg.swiglu_limit, cfg.routed_scaling_factor
    topk = cfg.num_experts_per_tok
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
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        k = k.repeat_interleave(H // Hkv, dim=1)
        v = v.repeat_interleave(H // Hkv, dim=1)
        ao = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=Dh**-0.5)
        ao = ao.transpose(1, 2).reshape(bsz, seq, H * Dh)
        x = x + ao @ hf[p + "self_attn.o_proj.weight"].T

        h = _gemma_rmsnorm(x, hf[p + "post_attention_layernorm.weight"], eps)
        if cfg.moe_layer_freq[li] == 0:
            g = h @ hf[p + "mlp.gate_proj.weight"].T
            u = h @ hf[p + "mlp.up_proj.weight"].T
            x = x + _swiglu_oai(g, u, alpha, limit) @ hf[p + "mlp.down_proj.weight"].T
        else:
            hf_flat = h.reshape(-1, h.shape[-1])
            logits = hf_flat.float() @ hf[p + "block_sparse_moe.gate.weight"].T.float()
            probs = torch.sigmoid(logits)
            sel = probs + hf[p + "block_sparse_moe.e_score_correction_bias"].float()
            idx = sel.topk(topk, dim=-1).indices
            w = probs.gather(-1, idx)
            w = (w / w.sum(-1, keepdim=True)) * scale
            out = torch.zeros_like(hf_flat)
            for tok in range(hf_flat.shape[0]):
                for j in range(topk):
                    e = idx[tok, j].item()
                    ep = f"{p}block_sparse_moe.experts.{e}."
                    g = hf_flat[tok] @ hf[ep + "w1.weight"].T
                    u = hf_flat[tok] @ hf[ep + "w3.weight"].T
                    out[tok] += w[tok, j] * (_swiglu_oai(g, u, alpha, limit) @ hf[ep + "w2.weight"].T)
            sp = f"{p}block_sparse_moe.shared_experts."
            g = hf_flat @ hf[sp + "gate_proj.weight"].T
            u = hf_flat @ hf[sp + "up_proj.weight"].T
            out = out + _swiglu_oai(g, u, alpha, limit) @ hf[sp + "down_proj.weight"].T
            x = x + out.reshape(bsz, seq, -1)

    x = _gemma_rmsnorm(x, hf["model.norm.weight"], eps)
    return x @ hf["lm_head.weight"].T


def test_stage1_parity_vs_sglang_reference(model):
    torch.manual_seed(0)
    cfg = model.config
    hf = {k: v.float() for k, v in model.state_dict_adapter.to_hf(model.state_dict()).items()}
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        mine = model(ids).float()
        ref = _ref_forward(hf, ids, cfg)

    max_diff = (mine - ref).abs().max().item()
    cos = F.cosine_similarity(mine.flatten(), ref.flatten(), dim=0).item()
    assert mine.shape == ref.shape
    assert max_diff < 1e-4, f"max_diff={max_diff}"
    assert cos > 0.9999, f"cos_sim={cos}"
