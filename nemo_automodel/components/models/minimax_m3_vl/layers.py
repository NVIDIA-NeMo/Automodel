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

"""MiniMax M3 VL text-backbone layers.

Stage 1 covers the dense + MoE text path (no sparse-attention index branch and
no MTP).  Mirrors the canonical sglang reference
``sglang.srt.models.minimax_m3`` (``MiniMaxM3Attention`` / ``MiniMaxM3MLP`` /
``MiniMaxM3MoE`` / ``MiniMaxM3DecoderLayer``):

* per-head **Gemma** RMSNorm on Q/K (``qk_norm_type='per_head'``,
  ``use_gemma_norm=True``),
* partial RoPE (``rotary_dim=64`` of ``head_dim=128``) reusing the gpt_oss
  rotary utilities (as the existing ``minimax_m2`` backbone does),
* SwiGLU-OAI activation ``gate * sigmoid(alpha * gate) * (up + 1)`` with gate
  clamped ``max=limit`` and up clamped ``+/-limit`` for dense and shared experts,
* per-layer dense-vs-MoE selection from ``moe_layer_freq``.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk
from nemo_automodel.components.moe.layers import MoE, MoEConfig


class MiniMaxM3RMSNorm(nn.Module):
    """RMSNorm with optional Gemma-style zero-centered gamma (``x_normed * (1 + w)``).

    When ``gemma=True`` the learnable weight is centered at 0 and the effective
    scale is ``1 + weight`` (matching HF ``GemmaRMSNorm`` and the sglang M3
    reference). Used both for hidden-size norms and, with ``dim=head_dim``, for
    per-head Q/K normalization (the input is normalized over its last dim, so a
    ``[..., num_heads, head_dim]`` tensor is normalized independently per head).
    """

    def __init__(self, dim: int, eps: float = 1e-6, gemma: bool = True):
        super().__init__()
        self.eps = eps
        self.gemma = gemma
        self.weight = nn.Parameter(torch.zeros(dim) if gemma else torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        weight = self.weight.float()
        if self.gemma:
            weight = weight + 1.0
        return (x * weight).to(dtype)

    def reset_parameters(self) -> None:
        if self.gemma:
            nn.init.zeros_(self.weight)
        else:
            nn.init.ones_(self.weight)


def swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """GPT-OSS / MiniMax-M3 SwiGLU-OAI: ``gate * sigmoid(alpha * gate) * (up + 1)``.

    Gate is clamped ``max=limit`` and up is clamped ``+/-limit`` (when
    ``limit > 0``), computed in fp32 and cast back. Equivalent to sglang's
    ``swiglu_no_interleaved_with_alpha_and_limit``.
    """
    dtype = gate.dtype
    gate = gate.float()
    up = up.float()
    if limit > 0.0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    out = gate * torch.sigmoid(alpha * gate) * (up + 1.0)
    return out.to(dtype)


class MiniMaxM3MLP(nn.Module):
    """Dense / shared-expert MLP with SwiGLU-OAI activation (separate gate/up/down)."""

    def __init__(self, config: Any, intermediate_size: int, backend: BackendConfig):
        super().__init__()
        self.alpha = float(getattr(config, "swiglu_alpha", 1.702))
        self.limit = float(getattr(config, "swiglu_limit", 7.0))
        self.gate_proj = initialize_linear_module(backend.linear, config.hidden_size, intermediate_size, bias=False)
        self.up_proj = initialize_linear_module(backend.linear, config.hidden_size, intermediate_size, bias=False)
        self.down_proj = initialize_linear_module(backend.linear, intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu_oai(self.gate_proj(x), self.up_proj(x), self.alpha, self.limit))

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for linear in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


@torch.no_grad()
def build_block_sparse_attn_bias(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    *,
    block_size: int,
    topk_blocks: int,
    init_blocks: int,
    local_blocks: int,
    num_q_heads: int,
    score_type: str = "max",
) -> torch.Tensor:
    """Build the additive block-sparse causal attention bias from index q/k.

    Mirrors the sglang ``minimax_sparse`` selection (``block_size_q=1`` ->
    per-query-position): the index score for (query ``i``, key ``j``) is
    ``(idx_q[i] . idx_k[j]) * idx_dim**-0.5`` with causal masking; keys are
    grouped into blocks of ``block_size`` and reduced per block (``max`` or
    ``lse``). For each query, the current block (``local_blocks``) and the first
    ``init_blocks`` are always kept and the remaining budget is filled with the
    highest-scoring causal blocks, up to ``min(topk_blocks, valid_blocks)``.

    Args:
        idx_q: ``[B, T, H_idx, D]`` index queries (post norm + RoPE).
        idx_k: ``[B, T, 1, D]`` shared index key (post norm + RoPE).
        num_q_heads: number of main attention heads; the per-idx-head bias is
            expanded ``num_q_heads // H_idx`` times (GQA, repeat-interleave).

    Returns:
        ``[B, num_q_heads, T, T]`` float bias (``0`` where attended, ``-inf``
        otherwise). Non-differentiable (hard selection).
    """
    bsz, seqlen, h_idx, dim = idx_q.shape
    device = idx_q.device
    scale = dim**-0.5

    q = idx_q.permute(0, 2, 1, 3).float()  # [B, H_idx, Tq, D]
    k = idx_k.permute(0, 2, 1, 3).float()  # [B, 1, Tk, D]
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, H_idx, Tq, Tk]

    causal = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=device))
    neg_inf = float("-inf")
    scores = scores.masked_fill(~causal, neg_inf)

    num_blocks = (seqlen + block_size - 1) // block_size
    pad = num_blocks * block_size - seqlen
    if pad:
        scores = F.pad(scores, (0, pad), value=neg_inf)
    scores = scores.view(bsz, h_idx, seqlen, num_blocks, block_size)
    if score_type == "lse":
        block_score = torch.logsumexp(scores, dim=-1)
    else:  # "max"
        block_score = scores.amax(dim=-1)  # [B, H_idx, Tq, num_blocks]

    qpos = torch.arange(seqlen, device=device)
    blk = torch.arange(num_blocks, device=device)
    cur_block = qpos // block_size
    valid_blocks = cur_block + 1  # causal: blocks [0, valid_blocks)
    causal_block = blk[None, :] < valid_blocks[:, None]  # [Tq, num_blocks]
    forced = ((blk[None, :] == cur_block[:, None]) & (local_blocks > 0)) | (blk[None, :] < init_blocks)
    forced = forced & causal_block

    sel = block_score.masked_fill(~causal_block[None, None], neg_inf)
    sel = sel.masked_fill(forced[None, None], float("inf"))  # force-include init/local blocks

    k_eff = min(topk_blocks, num_blocks)
    topk_idx = sel.topk(k_eff, dim=-1).indices  # [B, H_idx, Tq, k_eff]
    block_sel = torch.zeros_like(block_score, dtype=torch.bool).scatter_(-1, topk_idx, True)
    block_sel = block_sel & causal_block[None, None]  # drop non-causal padding picks

    key_sel = block_sel.repeat_interleave(block_size, dim=-1)[..., :seqlen]  # [B, H_idx, Tq, Tk]
    key_sel = key_sel & causal[None, None]
    bias = torch.where(key_sel, 0.0, neg_inf).to(torch.float32)

    rep = num_q_heads // h_idx
    return bias.repeat_interleave(rep, dim=1)  # [B, num_q_heads, Tq, Tk]


def _padding_mask_to_additive_bias(attention_mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Convert an incoming attention mask to an additive key bias broadcastable to ``ref``.

    Accepts a 2-D ``[B, T]`` keep-mask (1/True = attend) or an already-additive
    float mask; returns ``0`` where attended and ``-inf`` where masked.
    """
    if attention_mask.is_floating_point() and attention_mask.dim() >= 3:
        return attention_mask.to(ref.dtype)
    mask = attention_mask
    if mask.dim() == 2:
        mask = mask[:, None, None, :]  # [B, 1, 1, T] -> masks padded *keys*
    return torch.where(mask.bool(), 0.0, float("-inf")).to(dtype=ref.dtype, device=ref.device)


class MiniMaxM3Indexer(nn.Module):
    """Lightning indexer (selection-only) for MiniMax M3 sparse-attention layers.

    Projects hidden states to ``num_index_heads`` index queries and a single
    shared index key (``disable_index_value=True`` for M3, so there is no index
    value/output projection). Per-head Gemma RMSNorm + partial RoPE mirror the
    main attention. The produced ``idx_q``/``idx_k`` feed
    :func:`build_block_sparse_attn_bias` to select which key blocks each query
    attends to.
    """

    def __init__(self, config: Any, sparse_cfg: dict, backend: BackendConfig):
        super().__init__()
        self.backend = backend
        self.num_index_heads = sparse_cfg["sparse_num_index_heads"]
        self.index_head_dim = sparse_cfg["sparse_index_dim"]
        self.block_size = sparse_cfg["sparse_block_size"]
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]
        self.init_blocks = sparse_cfg.get("sparse_init_block", 0)
        self.local_blocks = sparse_cfg.get("sparse_local_block", 1)
        self.score_type = sparse_cfg.get("sparse_score_type", "max")
        gemma = getattr(config, "use_gemma_norm", False)

        self.index_q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_index_heads * self.index_head_dim, bias=False
        )
        self.index_k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.index_head_dim, bias=False
        )
        self.index_q_norm = MiniMaxM3RMSNorm(self.index_head_dim, eps=config.rms_norm_eps, gemma=gemma)
        self.index_k_norm = MiniMaxM3RMSNorm(self.index_head_dim, eps=config.rms_norm_eps, gemma=gemma)

    def forward(
        self, x: torch.Tensor, *, freqs_cis: torch.Tensor, num_q_heads: int, **attn_kwargs: Any
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        idx_q = self.index_q_norm(self.index_q_proj(x).view(bsz, seqlen, self.num_index_heads, self.index_head_dim))
        idx_k = self.index_k_norm(self.index_k_proj(x).view(bsz, seqlen, 1, self.index_head_dim))
        idx_q, idx_k = apply_rotary_emb_qk(
            idx_q,
            idx_k,
            freqs_cis,
            format="bshd",
            rope_fusion=self.backend.rope_fusion,
            cp_size=attn_kwargs.get("cp_size", 1),
            cp_rank=attn_kwargs.get("cp_rank", 0),
        )
        return build_block_sparse_attn_bias(
            idx_q,
            idx_k,
            block_size=self.block_size,
            topk_blocks=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            num_q_heads=num_q_heads,
            score_type=self.score_type,
        )

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        nn.init.trunc_normal_(self.index_q_proj.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.index_k_proj.weight, mean=0.0, std=init_std)
        self.index_q_norm.reset_parameters()
        self.index_k_norm.reset_parameters()


class MiniMaxM3Attention(nn.Module):
    """MiniMax M3 GQA attention with per-head Gemma Q/K norm and partial RoPE.

    When ``is_sparse_attention_layer`` is set, an additional lightning indexer
    (``index_q/k_proj`` + per-head Gemma norm) selects, per query, the top-k key
    *blocks* to attend to (block-level DeepSeek-style sparse attention). M3 sets
    ``disable_index_value=True`` so the index branch is selection-only.
    """

    def __init__(
        self,
        config: Any,
        backend: BackendConfig,
        *,
        is_sparse_attention_layer: bool = False,
    ):
        super().__init__()
        self.backend = backend
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.is_sparse_attention_layer = is_sparse_attention_layer
        gemma = getattr(config, "use_gemma_norm", False)

        # Fail loudly on unsupported configs: M3 does not implement the attention
        # output gate, and only per-head QK norm is supported (the only mode the
        # sparse index branch is valid for).
        assert not getattr(config, "attention_output_gate", False), (
            "MiniMax M3 attention_output_gate is not implemented"
        )
        qk_norm_type = getattr(config, "qk_norm_type", "per_head")
        if self.use_qk_norm or is_sparse_attention_layer:
            assert qk_norm_type == "per_head", f"MiniMax M3 only supports qk_norm_type='per_head', got {qk_norm_type!r}"

        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        if self.use_qk_norm:
            self.q_norm = MiniMaxM3RMSNorm(self.head_dim, eps=config.rms_norm_eps, gemma=gemma)
            self.k_norm = MiniMaxM3RMSNorm(self.head_dim, eps=config.rms_norm_eps, gemma=gemma)
        else:
            self.q_norm = None
            self.k_norm = None

        self.indexer = (
            MiniMaxM3Indexer(config, config.sparse_attention_config, backend) if is_sparse_attention_layer else None
        )

        softmax_scale = self.head_dim**-0.5
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=self.num_heads,
            num_qk_channels=self.head_dim,
            num_v_channels=self.head_dim,
            softmax_scale=softmax_scale,
            num_gqa_groups=self.num_kv_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
            q = self.q_proj(x).view(num_tokens, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(num_tokens, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            qkv_format = "bshd"
            bsz, seqlen, _ = x.size()
            q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # Per-head QK norm (over head_dim) is applied before RoPE, matching the
        # sglang reference (``_qk_norm`` then ``rotary_emb``).
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.indexer is not None:
            if qkv_format != "bshd":
                raise NotImplementedError("MiniMax M3 sparse attention currently supports bshd format only.")
            sparse_bias = self.indexer(x, freqs_cis=freqs_cis, num_q_heads=self.num_heads, **attn_kwargs)
            # Preserve the caller's padding mask: padded keys must stay masked
            # rather than becoming eligible for top-k block selection.
            if attention_mask is not None:
                sparse_bias = sparse_bias + _padding_mask_to_additive_bias(attention_mask, sparse_bias)
            attention_mask = sparse_bias

        q, k = apply_rotary_emb_qk(
            q,
            k,
            freqs_cis,
            format=qkv_format,
            rope_fusion=self.backend.rope_fusion,
            cu_seqlens=attn_kwargs.get("cu_seqlens", None),
            cp_size=attn_kwargs.get("cp_size", 1),
            cp_rank=attn_kwargs.get("cp_rank", 0),
        )

        # SDPA's additive (float) mask must share the query dtype. M3 builds its
        # block-sparse / padding attention bias in float32, so align it to the
        # compute dtype here. Kept local to M3 rather than in the shared attention
        # util so other models' attention paths are unaffected. Boolean key-padding
        # masks (dim <= 2) are left untouched and handled by the shared util.
        if attention_mask is not None and attention_mask.is_floating_point():
            attention_mask = attention_mask.to(q.dtype)
        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, **attn_kwargs
        )
        out = self.attn_func(q, k, v, **_attn_kwargs)
        out = postprocess_output_for_attn(out, self.backend.attn)

        flatten_dim = 2 if qkv_format == "bshd" else 1
        return self.o_proj(out.flatten(flatten_dim))

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
            self.k_norm.reset_parameters()
        if self.indexer is not None:
            self.indexer.init_weights(buffer_device, init_std)


class Block(nn.Module):
    """MiniMax M3 decoder block: attention + (dense MLP or MoE) with Gemma norms.

    ``moe_layer_freq[layer_idx] == 0`` -> dense ``MiniMaxM3MLP`` (with
    ``dense_intermediate_size``); otherwise a routed ``MoE`` plus a separate
    SwiGLU-OAI shared expert (kept M3-local rather than using ``MoE``'s built-in
    shared expert, whose generic ``MLP`` does not implement SwiGLU-OAI).
    """

    def __init__(self, layer_idx: int, config: Any, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        self.layer_idx = layer_idx

        # Sparse-attention layers are selected by sparse_attention_config's
        # ``sparse_attention_freq`` (layers 0-2 are dense, 3-59 sparse for M3).
        sparse_cfg = getattr(config, "sparse_attention_config", None)
        if sparse_cfg is not None and sparse_cfg.get("use_sparse_attention", True):
            sparse_freq = sparse_cfg.get("sparse_attention_freq")
            is_sparse_attention_layer = sparse_freq is None or sparse_freq[layer_idx] != 0
        else:
            is_sparse_attention_layer = False

        if is_sparse_attention_layer:
            # MiniMaxM3Indexer only implements the selection-only branch
            # (disable_index_value=True; no index value/output projections).
            disable_flags = sparse_cfg.get("sparse_disable_index_value")
            assert disable_flags is None or disable_flags[layer_idx] != 0, (
                f"MiniMax M3 sparse layer {layer_idx} has disable_index_value=0 (index value/output "
                "projections), which is not supported (only the selection-only indexer is implemented)."
            )
        self.self_attn = MiniMaxM3Attention(config, backend, is_sparse_attention_layer=is_sparse_attention_layer)

        moe_layer_freq = getattr(config, "moe_layer_freq", None)
        self.is_moe_layer = True if moe_layer_freq is None else moe_layer_freq[layer_idx] != 0

        if self.is_moe_layer:
            self.mlp = MoE(moe_config, backend)
            n_shared = getattr(config, "n_shared_experts", 0) or 0
            if n_shared > 0:
                shared_inter = getattr(config, "shared_intermediate_size", config.intermediate_size) * n_shared
                self.shared_experts = MiniMaxM3MLP(config, shared_inter, backend)
            else:
                self.shared_experts = None
        else:
            self.mlp = MiniMaxM3MLP(config, config.dense_intermediate_size, backend)
            self.shared_experts = None

        gemma = getattr(config, "use_gemma_norm", False)
        self.input_layernorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)
        self.post_attention_layernorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        attn_out = self.self_attn(
            x=self.input_layernorm(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
        x = x + attn_out

        normed = self.post_attention_layernorm(x)
        if self.is_moe_layer:
            mlp_out = self.mlp(normed, padding_mask)
            if self.shared_experts is not None:
                mlp_out = mlp_out + self.shared_experts(normed)
        else:
            mlp_out = self.mlp(normed)
        x = x + mlp_out
        return x

    def init_weights(self, buffer_device: torch.device):
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(buffer_device)
