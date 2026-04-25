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

"""DeepSeek V4 Attention Layer.

Architecture (from official inference/model.py):

Q path:
  x  -> wq_a [hidden -> q_lora_rank]
     -> q_norm (RMSNorm)
     -> wq_b  [q_lora_rank -> n_heads * head_dim]
     -> reshape [n_heads, head_dim]
     -> per-head RMSNorm  (q_norm applied per-head in official code)
     -> apply_rotary_emb on last rope_head_dim dims

KV path (K = V, single latent):
  x  -> wkv   [hidden -> head_dim]        # single KV head, K = V = kv
     -> kv_norm (RMSNorm on head_dim)
     -> apply_rotary_emb on last rope_head_dim dims
  K = V = kv  (one latent vector serves both key and value)

Output path (grouped):
  o [bsz, seq, n_heads, head_dim]
    -> reshape [bsz, seq, n_groups, n_heads_per_group * head_dim]
    -> wo_a einsum per group: [n_heads_per_group * head_dim] -> [o_lora_rank]
    -> reshape [bsz, seq, n_groups * o_lora_rank]
    -> wo_b [n_groups * o_lora_rank -> hidden]

attn_sink: learnable per-head scalar bias added to attention-sink position score.

HC (Hyper-Connections):
  Each Block maintains hc_mult=4 copies of the hidden state.
  hc_pre  reduces [bsz, seq, hc_mult, dim] -> [bsz, seq, dim] via Sinkhorn mixing.
  hc_post expands [bsz, seq, dim] -> [bsz, seq, hc_mult, dim].
  See ``_hc_split_sinkhorn`` for the pure-torch port of the reference mixer
  (ported from miles PR 1045's ``kernel/sinkhorn.py``).

Sliding-window / compress-ratio attention is NOT yet implemented.
All layers use full causal attention regardless of compress_ratios.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.deepseek_v3.rope_utils import (
    apply_rotary_emb_qk,
)
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

# ---------------------------------------------------------------------------
# Grouped output projection (wo_a)
# ---------------------------------------------------------------------------


class GroupedOutputProjection(nn.Module):
    """Block-diagonal output projection for DeepSeek V4.

    Splits the n_heads attention outputs into n_groups groups, projects each
    group's n_heads_per_group * head_dim dimensions to o_lora_rank, then
    concatenates.

    Weight shape: [n_groups * o_lora_rank, n_heads_per_group * head_dim]
                = [8 * 1024, 8 * 512] = [8192, 4096]

    This matches the official model.py:
        wo_a = Linear(n_heads * head_dim // n_groups, n_groups * o_lora_rank)
        wo_a = wo_a.weight.view(n_groups, o_lora_rank, -1)
        o = einsum("bsgd,grd->bsgr", o_grouped, wo_a)
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        o_lora_rank: int,
        n_groups: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        assert n_heads % n_groups == 0
        self.n_groups = n_groups
        self.n_heads_per_group = n_heads // n_groups
        self.head_dim = head_dim
        self.o_lora_rank = o_lora_rank
        in_per_group = self.n_heads_per_group * head_dim  # 8 * 512 = 4096
        out_total = n_groups * o_lora_rank  # 8 * 1024 = 8192
        self.weight = nn.Parameter(torch.zeros(out_total, in_per_group, dtype=dtype))

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        # o: [..., n_heads, head_dim]
        shape = o.shape[:-2]
        # -> [..., n_groups, n_heads_per_group * head_dim]
        o_g = o.reshape(*shape, self.n_groups, self.n_heads_per_group * self.head_dim)
        # weight: [n_groups * o_lora_rank, n_heads_per_group * head_dim]
        #      -> [n_groups, o_lora_rank, n_heads_per_group * head_dim]
        w = self.weight.reshape(self.n_groups, self.o_lora_rank, self.n_heads_per_group * self.head_dim)
        # einsum: [..., g, d] x [g, r, d]^T -> [..., g, r]
        out = torch.einsum("...gd,grd->...gr", o_g, w)
        # -> [..., n_groups * o_lora_rank]
        return out.reshape(*shape, self.n_groups * self.o_lora_rank)

    def init_weights(self, init_std: float = 0.02) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=init_std)


# ---------------------------------------------------------------------------
# HC (Hyper-Connections) — ported verbatim from HuggingFace transformers
# PR 45616 (Arthur Zucker, "Add DeepSeek V4").  Source-of-truth reference:
# ``transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py``
# classes ``DeepseekV4HyperConnection`` (lines 613-670) and
# ``DeepseekV4HyperHead`` (lines 673-690).
#
# The previous pure-torch port (mean-pool / softmax-comb) had three silent
# divergences vs HF: (a) ``comb`` used row-softmax where HF uses ``sigmoid``,
# (b) ``post`` had a ``2 *`` prefactor + missing ``+eps``, (c) the mixer was
# wrapped in ``torch.no_grad()``.  Rather than patch those line-by-line,
# swap wholesale to the HF classes so future HF updates flow through cleanly
# via the adapter's state-dict rename rules.
# ---------------------------------------------------------------------------


class DeepseekV4HyperConnection(nn.Module):
    """Per-site HyperConnection mixer (attention or FFN).  Ported from
    ``transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py``
    class ``DeepseekV4HyperConnection``.

    Owns ``fn`` (packed linear), ``base`` (bias), and ``scale`` (scalar
    per-head gains).  ``compute_weights`` produces three mixer tensors:

      - ``pre``   [B, S, H]       : sigmoid-gated collapse weights
      - ``post``  [B, S, H]       : sigmoid-gated expand weights
      - ``comb``  [B, S, H, H]    : doubly-stochastic combination matrix
                                    from Sinkhorn-normalising sigmoid gates

    All math runs in fp32 regardless of the outer cast policy; parameters
    cast themselves via ``.float()`` on each forward.  HF lists these params
    in ``_keep_in_fp32_modules_strict`` — the KAutomodel adapter does the
    same via submodule-name matching.
    """

    def __init__(
        self,
        hc_mult: int,
        hidden_size: int,
        hc_sinkhorn_iters: int,
        hc_eps: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.norm_eps = rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def compute_weights(
        self, hidden_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_streams.flatten(start_dim=2).float()  # [B, S, H*D]
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        # HC mixer params are kept in fp32 for Sinkhorn stability — cast defensively.
        mix = torch.nn.functional.linear(flat, self.fn.float()) * rsqrt  # [B, S, (2+H)*H]
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)
        hc = self.hc_mult
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc].float()) + self.hc_eps
        post = torch.sigmoid(mix[..., hc : 2 * hc] * post_scale + self.base[hc : 2 * hc].float()) + self.hc_eps
        comb = (
            torch.sigmoid(
                mix[..., 2 * hc :].view(*mix.shape[:-1], hc, hc) * comb_scale
                + self.base[2 * hc :].view(hc, hc).float()
            )
            + self.hc_eps
        )
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb


class DeepseekV4HyperHead(nn.Module):
    """Final HC-stream collapse before the shared RMSNorm + ``lm_head``.
    Ported from ``modular_deepseek_v4.py`` class ``DeepseekV4HyperHead``.

    Sigmoid-weighted sum over the ``hc_mult`` streams (no Sinkhorn).  Used
    once at the end of ``DeepseekV4Model.forward`` to go from
    ``[B, S, H, D]`` back to ``[B, S, D]``.
    """

    def __init__(self, hc_mult: int, hidden_size: int, hc_eps: float, rms_norm_eps: float):
        super().__init__()
        self.hc_mult = hc_mult
        self.norm_eps = rms_norm_eps
        self.eps = hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(flat, self.hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


# ---------------------------------------------------------------------------
# Main attention layer
# ---------------------------------------------------------------------------


class DeepseekV4Attention(nn.Module):
    """GQA attention with Q/O LoRA for DeepSeek V4.

    K = V = wkv(x):  a single head_dim latent vector used as both key and value.
    attn_sink      :  per-head scalar bias (shape [n_heads]) applied to the
                      "sink" position in sparse_attn.  Stored as float32.
    wo_a           :  grouped projection, weight [n_groups*o_lora_rank,
                                                   n_heads_per_group*head_dim].
    """

    def __init__(self, config: DeepseekV4Config, backend: BackendConfig):
        super().__init__()

        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups

        self.backend = backend
        self.rope_fusion = backend.rope_fusion
        linear_impl = backend.linear
        rms_norm_impl = backend.rms_norm
        hidden_size = config.hidden_size
        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)

        # Q LoRA
        self.wq_a = initialize_linear_module(linear_impl, hidden_size, self.q_lora_rank, bias=False, dtype=model_dtype)
        self.q_norm = initialize_rms_norm_module(
            rms_norm_impl, self.q_lora_rank, eps=config.rms_norm_eps, dtype=model_dtype
        )
        self.wq_b = initialize_linear_module(
            linear_impl, self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=model_dtype
        )

        # Combined KV projection: K = V = wkv(x), single latent of dim head_dim
        self.wkv = initialize_linear_module(linear_impl, hidden_size, self.head_dim, bias=False, dtype=model_dtype)
        self.kv_norm = initialize_rms_norm_module(
            rms_norm_impl, self.head_dim, eps=config.rms_norm_eps, dtype=model_dtype
        )

        # Grouped output projection
        self.wo_a = GroupedOutputProjection(
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            o_lora_rank=self.o_lora_rank,
            n_groups=self.o_groups,
            dtype=model_dtype,
        )
        self.wo_b = initialize_linear_module(
            linear_impl, self.o_groups * self.o_lora_rank, hidden_size, bias=False, dtype=model_dtype
        )

        # Attention sink: per-head learnable scalar (stored in float32 per official impl).
        # Shape: [n_heads] — acts as an extra "sink" logit appended to the softmax
        # denominator of every query row.  The sink column does not correspond to any
        # token, so it contributes nothing to the output but lets probability mass
        # drain away from real tokens for heads that should attend to "nothing".
        self.register_parameter(
            "attn_sink",
            nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32), requires_grad=True),
        )

        # V4 uses the plain head_dim^-0.5 scale — unlike V3, the reference
        # model.py line 464 does NOT fold in a YaRN mscale correction.  Applying
        # mscale*mscale here (as V3 does) makes attention scores ~1.6x sharper
        # than the trained model expects and shifts every layer's output
        # distribution, which at step 0 shows up as logits std=3.5 and
        # argmax_match=0% despite correctly loaded weights.
        self.softmax_scale = self.head_dim**-0.5

        # V4 uses 1 KV head (K=V=kv), so num_gqa_groups = n_heads for standard GQA path
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=self.n_heads,
            num_qk_channels=self.head_dim,
            num_v_channels=self.head_dim,
            softmax_scale=self.softmax_scale,
            # GQA with n_kv_heads=1: num_gqa_groups=n_heads
            num_gqa_groups=self.n_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if x.dim() == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
            bsz, seq_len = 1, num_tokens
        else:
            qkv_format = "bshd"
            bsz, seq_len, _ = x.shape

        # --- Q path ---
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        if qkv_format == "thd":
            q = q.view(num_tokens, self.n_heads, self.head_dim)
        else:
            q = q.view(bsz, seq_len, self.n_heads, self.head_dim)

        # Per-head RMSNorm on Q (as in official model: q *= rsqrt(q.sq.mean(-1, keepdim=True)))
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-6).to(q.dtype)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # --- KV path: K = V = single latent ---
        kv = self.kv_norm(self.wkv(x))  # [..., head_dim]
        kv_nope, kv_pe = torch.split(kv, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # RoPE — V4 applies rotation to q/k here and an INVERSE rotation to the
        # attention output later, before wo_a (see the official model.py).  The
        # TE fused-rope path does not give us a complex freqs_cis to conjugate,
        # so force the non-fused (complex) path for V4.
        cu_seqlens = attn_kwargs.get("cu_seqlens", None)
        # kv_pe needs a head dimension for apply_rotary_emb_qk
        head_unsqueeze_dim = 2 if qkv_format == "bshd" else 1
        kv_pe = kv_pe.unsqueeze(head_unsqueeze_dim)
        q_pe, kv_pe = apply_rotary_emb_qk(
            q_pe,
            kv_pe,
            freqs_cis,
            format=qkv_format,
            rope_fusion=False,
            cu_seqlens=cu_seqlens,
            cp_size=attn_kwargs.get("cp_size", 1),
            cp_rank=attn_kwargs.get("cp_rank", 0),
        )
        kv_pe = kv_pe.squeeze(head_unsqueeze_dim)

        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = torch.cat([kv_nope, kv_pe], dim=-1)

        # Expand kv to n_heads (K=V shared across all heads) for standard attention path
        if qkv_format == "thd":
            k = kv.unsqueeze(1).expand(num_tokens, self.n_heads, self.head_dim)
            v = k  # K = V
        else:
            k = kv.unsqueeze(2).expand(bsz, seq_len, self.n_heads, self.head_dim)
            v = k  # K = V

        # Manual scaled dot-product attention with per-head attn_sink.
        # The default preprocess/attn_func path uses is_causal=True which skips the
        # bias we need for the sink. Keep the attention math in fp32 for stability.
        if qkv_format == "bshd":
            # [B, S, H, D] -> [B, H, S, D]
            q_a = q.transpose(1, 2)
            k_a = k.transpose(1, 2)
            v_a = v.transpose(1, 2)
        else:
            # thd (single packed sequence) -> [1, H, T, D]
            q_a = q.unsqueeze(0).transpose(1, 2)
            k_a = k.unsqueeze(0).transpose(1, 2)
            v_a = v.unsqueeze(0).transpose(1, 2)

        s_len = q_a.size(-2)
        scores = torch.matmul(q_a.float(), k_a.float().transpose(-2, -1)) * self.softmax_scale
        causal_mask = torch.triu(
            torch.ones(s_len, s_len, dtype=torch.bool, device=scores.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        # Append per-head sink logit: attn_sink[h] acts as an extra softmax column
        # whose value slot is zero (contributes nothing to the output).
        sink = self.attn_sink.float().view(1, -1, 1, 1).expand(scores.size(0), -1, s_len, 1)
        scores_ext = torch.cat([scores, sink], dim=-1)
        probs = torch.softmax(scores_ext, dim=-1)[..., :-1]
        o_a = torch.matmul(probs.to(v_a.dtype), v_a)  # [B, H, S, D]

        if qkv_format == "bshd":
            o = o_a.transpose(1, 2).contiguous()  # [B, S, H, D]
        else:
            o = o_a.squeeze(0).transpose(0, 1).contiguous()  # [T, H, D]

        # --- O path (grouped) ---
        # o: [bsz, seq, n_heads, head_dim] or [num_tokens, n_heads, head_dim]
        # Apply INVERSE RoPE to the pe part of o before the output projection,
        # matching the official model.py:
        #   apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
        # Without this, wo_a sees a rotated representation that its trained
        # weights do not expect — empirically this produces ~N(0, 3.5) logits
        # that gravitate to a few attractor tokens (argmax_match = 0%).
        o_nope, o_pe = torch.split(o, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # Build a complex conjugate freqs_cis view and multiply.  freqs_cis here
        # is the non-fused complex form produced by freqs_cis_from_position_ids
        # (see the rope_fusion=False change above).
        o_pe_c = torch.view_as_complex(o_pe.float().view(*o_pe.shape[:-1], -1, 2))
        fc = freqs_cis.conj()
        # For bshd tensors fc shape is [B, S, rope_head_dim/2]; for thd it is
        # [T, rope_head_dim/2]. Broadcast over the n_heads axis.
        if qkv_format == "bshd":
            fc = fc.view(fc.size(0), fc.size(1), 1, fc.size(-1))
        else:
            fc = fc.view(fc.size(0), 1, fc.size(-1))
        o_pe_c = o_pe_c * fc
        o_pe = torch.view_as_real(o_pe_c).flatten(-2).to(o_pe.dtype)
        o = torch.cat([o_nope, o_pe], dim=-1)
        x_out = self.wo_b(self.wo_a(o))  # wo_a handles the reshape internally
        return x_out

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for linear in (self.wq_a, self.wq_b, self.wkv, self.wo_b):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        self.wo_a.init_weights(init_std)
        for norm in (self.q_norm, self.kv_norm):
            norm.reset_parameters()
        nn.init.zeros_(self.attn_sink)
