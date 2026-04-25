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

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

# ---------------------------------------------------------------------------
# DeepSeek V4 attention + compressor + indexer + rotary embedding, ported
# verbatim from HuggingFace transformers PR 45616 (Arthur Zucker, "Add
# DeepSeek V4") at
#   transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py
# with two adjustments:
#   1) Rotary helper ``apply_rotary_pos_emb`` and ``repeat_kv`` are inlined
#      so we do not depend on HF's transformers-version-specific rotary API.
#   2) The ``DeepseekV4Cache`` integration is replaced with a minimal
#      training-only shim — KAutomodel training never carries a KV cache,
#      so ``accumulate_windows`` / ``update_pool`` are pass-throughs on a
#      per-forward scratch dict.  The KV-cache path is left for a future
#      inference port.
# The compressor + indexer modules are only constructed when a layer's
# ``compress_ratio`` is non-zero; the ``validate`` YAML sets all ratios to 0
# so those modules are dormant in the current smoke run but fully wired.
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (Llama / GPT-NeoX style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_partial_rope_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int
) -> torch.Tensor:
    """Interleaved RoPE on the last ``rope_head_dim`` dims of ``x`` (pairs are
    ``(2k, 2k+1)``).  Matches the DeepSeek inference reference's complex-mul
    formulation in ``dsv4flash/inference/model.py:apply_rotary_emb``: the
    released DSV4-Flash weights were trained with this layout, NOT the
    Llama-style ``rotate_half`` layout HF transformers PR 45616/45643 still
    uses (pairs ``(d, d+rd/2)``).

    Args:
        x: ``[..., rope_head_dim]`` (or larger trailing dim with rope on the
            last ``rope_head_dim`` slice).  Typical attention-layout shapes:
            ``[B, H, S, D]`` for q/k or ``[B, 1, S, D]`` for shared-KV.
        cos, sin: shape ``[B, S, rope_head_dim]`` produced by the Llama-style
            ``cat([freqs, freqs], -1)`` rotary; we take the first half which
            contains the unique per-pair frequencies (the second half is a
            duplicate that the Llama-style helper needs and we don't).
        rope_head_dim: Must be even.

    Inverse rotation: pass ``-sin`` instead of ``sin`` (caller's
    responsibility — same as our existing inverse-rope call site).
    """
    rd = rope_head_dim
    half = rd // 2
    nope, rope = x[..., :-rd], x[..., -rd:]
    # Pair-reshape last dim: [..., rd] -> [..., rd/2, 2]
    rope_pairs = rope.unflatten(-1, (-1, 2))
    a, b = rope_pairs[..., 0], rope_pairs[..., 1]  # [..., rd/2]
    c = cos[..., :half]
    s = sin[..., :half]
    # Broadcast c/s up to ``a``'s rank by inserting a head dim before S.
    while c.ndim < a.ndim:
        c = c.unsqueeze(1)
        s = s.unsqueeze(1)
    new_a = a * c - b * s
    new_b = a * s + b * c
    new_rope = torch.stack([new_a, new_b], dim=-1).flatten(-2)
    return torch.cat([nope, new_rope], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of transformers.models.llama.modeling_llama.apply_rotary_pos_emb."""
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Port of transformers.models.llama.modeling_llama.repeat_kv."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class DeepseekV4RotaryEmbedding(nn.Module):
    """V4 rotary embedding.  Produces ``(cos, sin)`` sized to ``qk_rope_head_dim``
    (via ``partial_rotary_factor = qk_rope_head_dim / head_dim``), matching HF.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        rope_theta: float,
        head_dim: int,
        partial_rotary_factor: float,
        attention_scaling: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.attention_scaling = attention_scaling
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force fp32 for numerical stability.
        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear (HF PR 45616 port).

    ``weight`` parameter has the standard ``nn.Linear`` shape
    ``[out_features, in_features_per_group]`` so quantizers keyed on
    ``nn.Linear.weight`` still find it; ``forward`` does per-group bmm.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_groups, in_features_per_group]
        batch_shape = x.shape[:-2]
        d_in = x.shape[-1]
        out_per_group = self.out_features // self.n_groups
        w = self.weight.view(self.n_groups, out_per_group, d_in)
        x = x.reshape(-1, self.n_groups, d_in).permute(1, 0, 2)
        y = torch.bmm(x, w.transpose(-1, -2)).permute(1, 0, 2)
        return y.reshape(*batch_shape, self.n_groups, out_per_group)


class DeepseekV4TrainCache:
    """Training-only cache shim mirroring the three methods ``DeepseekV4Compressor``
    / ``DeepseekV4Indexer`` call on ``DeepseekV4Cache``.

    KAutomodel training forward is stateless — we never persist KV or compressor
    windows across calls.  Each ``DeepseekV4Attention.forward`` creates a fresh
    cache instance, which holds per-layer scratch dicts for the duration of the
    call.  When a full window hasn't accumulated yet we return an empty tensor
    and let the downstream code handle it.
    """

    def __init__(self):
        self.compressor_state: list[dict] = []
        self.indexer_state: list[dict] = []

    def _branch_state(self, state_key: str, layer_idx: int) -> dict:
        store = getattr(self, state_key, None)
        if store is None:
            store = []
            setattr(self, state_key, store)
        while len(store) <= layer_idx:
            store.append({"buffer_kv": None, "buffer_gate": None, "pooled": None})
        return store[layer_idx]

    def accumulate_windows(
        self,
        kv: torch.Tensor,
        gate: torch.Tensor,
        layer_idx: int,
        state_key: str,
        ratio: int,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        state = self._branch_state(state_key, layer_idx)
        buf_kv, buf_gate = state["buffer_kv"], state["buffer_gate"]
        if buf_kv is not None and buf_kv.shape[1]:
            kv = torch.cat([buf_kv, kv], dim=1)
            gate = torch.cat([buf_gate, gate], dim=1)
        usable = (kv.shape[1] // ratio) * ratio
        state["buffer_kv"] = kv[:, usable:]
        state["buffer_gate"] = gate[:, usable:]
        pool_base = max(0, start_pos) - (buf_kv.shape[1] if buf_kv is not None else 0)
        return kv[:, :usable], gate[:, :usable], pool_base

    def update_pool(self, new_pooled: torch.Tensor, layer_idx: int, state_key: str) -> torch.Tensor:
        state = self._branch_state(state_key, layer_idx)
        pool = state["pooled"]
        if new_pooled.shape[1] > 0:
            pool = new_pooled if pool is None else torch.cat([pool, new_pooled], dim=1)
            state["pooled"] = pool
        if pool is None:
            pool = new_pooled.new_zeros((new_pooled.shape[0], 0, new_pooled.shape[-1]))
        return pool


def _apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int) -> torch.Tensor:
    """Split ``x`` along its last dim into nope (first) and rope (last
    ``rope_head_dim``) slices, rotate only the rope slice with INTERLEAVED
    pair-RoPE (pairs ``(2k, 2k+1)``), concat back.

    The DSV4-Flash released checkpoint uses interleaved RoPE end-to-end
    (see ``dsv4flash/inference/model.py:apply_rotary_emb`` — complex
    multiplication on ``view_as_complex`` of pairs).  HF transformers PR
    45616 / PR 45643 ship a Llama-style ``rotate_half`` here instead, which
    pairs ``(d, d+rd/2)``.  Same algebra but a different dim-to-frequency
    mapping — the released weights expect the interleaved layout, so the
    Llama-style helper produces wrong activations on the released checkpoint
    (verified empirically: kv_post_rope cosine drops from 0.9999 to 0.866
    after one block under Llama-style; matches at >0.999 under interleaved).
    """
    return _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim)


def _overlap_transform(tensor: torch.Tensor, head_dim: int, fill_value: float) -> torch.Tensor:
    """Reshape ``[B, S, ratio, 2*head_dim]`` -> ``[B, S, 2*ratio, head_dim]`` with the
    cross-window overlap from the DeepSeek inference reference (``Compressor.overlap_transform``
    in ``dsv4flash/inference/model.py:307-314``).

    Window N consumes:
      * positions ``[ratio:]`` of the new tensor: the **second half** of the feature dim
        of window N (current block).
      * positions ``[:ratio]`` of the new tensor: the **first half** of the feature dim
        of window N-1 (previous block, i.e. the overlap into the past).

    Window 0 has no previous block, so its ``[:ratio]`` slice is left at ``fill_value``
    (``0`` for the kv tensor, ``-inf`` for the score tensor so softmax masks it out).
    """
    b, s, ratio, _ = tensor.shape
    new = tensor.new_full((b, s, 2 * ratio, head_dim), fill_value)
    new[:, :, ratio:] = tensor[:, :, :, head_dim:]
    new[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
    return new


def _pool_windows(
    kv: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
    head_dim: int,
    overlap: bool = False,
) -> torch.Tensor:
    """Softmax-gated sum-pool over ``ratio`` consecutive tokens.

    Non-overlap mode (HF PR 45616 layout, ratio==128 in V4-Flash):
      Input  ``kv``/``gate`` of shape ``[B, length, head_dim]``.
      Reshape to ``[B, length/ratio, ratio, head_dim]`` and pool over the ``ratio`` axis.

    Overlap mode (DeepSeek inference reference layout, ratio==4 in V4-Flash):
      Input  ``kv``/``gate`` of shape ``[B, length, 2*head_dim]`` (``wkv``/``wgate``
      project to ``2*head_dim`` so each window can carry both its own kv and a
      half-overlap into the next window).
      Reshape to ``[B, length/ratio, ratio, 2*head_dim]``, apply :func:`_overlap_transform`
      to remap to ``[B, length/ratio, 2*ratio, head_dim]``, then pool over the ``2*ratio``
      axis.  Each compressed token thus aggregates ``2*ratio = 8`` raw tokens — the
      ``ratio`` tokens of the current window plus the ``ratio`` tokens of the previous
      window — giving smoother compression boundaries that the released checkpoint
      was trained under.

    HF PR 45616 omits the overlap path entirely; the released DSV4-Flash safetensors
    have ``ape``/``wkv``/``wgate`` shapes that only match the overlap layout (``[ratio,
    2*head_dim]`` and ``[2*head_dim, hidden]``), so we must support it here to load
    the released weights.
    """
    coff = 2 if overlap else 1
    feat = coff * head_dim
    batch, length, _ = kv.shape
    n_windows = length // ratio
    kv_w = kv.view(batch, n_windows, ratio, feat)
    gate_w = gate.view(batch, n_windows, ratio, feat) + ape
    if overlap:
        kv_w = _overlap_transform(kv_w, head_dim, fill_value=0.0)
        gate_w = _overlap_transform(gate_w, head_dim, fill_value=float("-inf"))
    return (kv_w * gate_w.softmax(dim=2)).sum(dim=2)


def _rope_pool_positions(
    pool_length: int, pool_base: int, ratio: int, device: torch.device, batch: int
) -> torch.Tensor:
    return (torch.arange(pool_length, device=device) * ratio + pool_base).unsqueeze(0).expand(batch, -1)


def build_causal_padding_mask(
    attention_mask: torch.Tensor | None,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int = 1,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    """Build a 4D additive causal+padding (+optional sliding-window) mask
    compatible with ``eager_attention_with_sink``.

    Mirrors HF's ``create_sliding_window_causal_mask`` (used in
    ``DeepseekV4Model.forward``): each query at position ``i`` attends only to
    keys at positions ``[max(0, i - sliding_window + 1), i]``.  The DSV4-Flash
    weights were trained with this banding on every layer, so dropping it makes
    the softmax see a different distribution than training and degrades loss.

    Inputs:
        attention_mask: 2D ``[B, S]`` tensor with 1=valid, 0=padding (HF convention),
            or already-4D additive mask, or ``None``.
        sliding_window: if not None, mask out keys further back than this many
            positions from the query (in addition to causal masking).
    Returns:
        ``[B, 1, S, S]`` additive mask of ``dtype`` (0 where keep, large negative
        where mask).
    """
    min_value = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    causal = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device)
    causal = torch.triu(causal, diagonal=1)
    if sliding_window is not None and sliding_window > 0:
        # Mask k_pos < q_pos - (sliding_window - 1)  →  diagonal = -(window - 1)
        # tril at that diagonal keeps the lower-band; we need to MASK the lower
        # tail (older keys).  Build a "too old" mask: positions where (q - k) >= window.
        idx = torch.arange(seq_len, device=device)
        too_old = (idx.unsqueeze(0) - idx.unsqueeze(1)) >= sliding_window  # [k_pos, q_pos] ?
        # We want shape [q_pos, k_pos], so use [q_pos=row, k_pos=col]:
        too_old = (idx.unsqueeze(1) - idx.unsqueeze(0)) >= sliding_window
        causal = causal.masked_fill(too_old, min_value)
    causal = causal.unsqueeze(0).unsqueeze(0)  # [1,1,S,S]
    if attention_mask is None:
        return causal.expand(batch_size, 1, seq_len, seq_len).contiguous()
    if attention_mask.dim() == 4:
        return attention_mask.to(dtype)
    if attention_mask.dim() == 2:
        # 1=valid, 0=padding -> 0 keep, min_value mask, broadcast over query rows
        pad_add = (1.0 - attention_mask.to(dtype)) * min_value  # [B, S]
        pad_add = pad_add.unsqueeze(1).unsqueeze(2)  # [B,1,1,S]
        return (causal + pad_add).to(dtype)
    raise ValueError(f"Unsupported attention_mask rank: {attention_mask.dim()}")


def eager_attention_with_sink(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager attention with per-head sink: appends an extra softmax column
    whose logit is ``module.sinks[h]`` and whose value-slot is zero.  Ported
    verbatim from HF PR 45616.
    """
    del kwargs
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : attn_weights.shape[-1]]
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1]
    probs = F.dropout(probs, p=dropout, training=module.training).to(value_states.dtype)
    return torch.matmul(probs, value_states).transpose(1, 2).contiguous(), probs


class DeepseekV4Indexer(nn.Module):
    """HF PR 45616 port.  Picks the top-k compressed positions per query when
    ``compress_ratio == 4``.  Owns its own pool at ``index_head_dim`` plus a
    query projection + weights_proj head-mixer.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_ratio = 4
        # Indexer's pool is always at compress_ratio==4, which means overlap mode
        # (matching the released checkpoint's ``indexer.compressor.{ape,wkv,wgate}``
        # shapes of ``[ratio, 2*index_head_dim]`` / ``[2*index_head_dim, hidden_size]``).
        self.overlap = True
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        proj_dim = 2 * self.head_dim  # overlap mode
        self.wkv = nn.Linear(config.hidden_size, proj_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, proj_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(self.compress_ratio, proj_dim))
        self.kv_norm = initialize_rms_norm_module("torch_fp32", self.head_dim, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        rotary: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: DeepseekV4TrainCache,
        layer_idx: int,
        start_pos: int,
    ) -> torch.LongTensor:
        batch, seq_len, _ = hidden_states.shape
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        ready_kv, ready_gate, pool_base = cache.accumulate_windows(
            kv, gate, layer_idx, "indexer_state", self.compress_ratio, start_pos
        )
        new_pooled = self.kv_norm(
            _pool_windows(
                ready_kv,
                ready_gate,
                self.ape,
                self.compress_ratio,
                self.head_dim,
                overlap=self.overlap,
            )
        )
        if new_pooled.shape[1] > 0:
            positions = _rope_pool_positions(
                new_pooled.shape[1], pool_base, self.compress_ratio, new_pooled.device, new_pooled.shape[0]
            )
            cos, sin = rotary(new_pooled, positions)
            new_pooled = _apply_partial_rope(new_pooled.unsqueeze(1), cos, sin, self.rope_head_dim).squeeze(1)
        pooled_kv = cache.update_pool(new_pooled, layer_idx, "indexer_state")

        cos, sin = position_embeddings
        q = self.wq_b(q_residual).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim).transpose(1, 2)
        scores = torch.matmul(q.float(), pooled_kv.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
        topk = min(self.index_topk, pooled_kv.shape[1])
        return index_scores.topk(topk, dim=-1).indices


class DeepseekV4Compressor(nn.Module):
    """HF PR 45616 port.  Long-range KV branch.  Pools ``compress_ratio`` tokens
    into one compressed KV; when ``ratio == 4`` the Indexer narrows the pool.
    """

    def __init__(self, config: DeepseekV4Config, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        # Overlap mode (compress_ratio==4) doubles the feature dim of wkv / wgate /
        # ape — the released DSV4-Flash checkpoint was trained that way to give each
        # compressed token cross-window context. Non-overlap mode (compress_ratio==128)
        # keeps a flat head_dim. ``kv_norm`` always normalizes over ``head_dim`` because
        # the overlap_transform inside ``_pool_windows`` collapses 2*head_dim → head_dim
        # before the norm runs.
        self.overlap = compress_ratio == 4
        coff = 2 if self.overlap else 1
        proj_dim = coff * head_dim
        self.wkv = nn.Linear(config.hidden_size, proj_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, proj_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(compress_ratio, proj_dim))
        self.kv_norm = initialize_rms_norm_module("torch_fp32", head_dim, eps=config.rms_norm_eps)
        self.indexer: DeepseekV4Indexer | None = (
            DeepseekV4Indexer(config) if compress_ratio == 4 else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor | None,
        rotary: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: DeepseekV4TrainCache,
        layer_idx: int,
        start_pos: int,
    ) -> torch.Tensor:
        # Debug dumps for compressor (paired with reference compressor at
        # ``dsv4flash/inference/model.py:Compressor.forward``).  Files:
        # ``$DSV4_DEBUG_DUMP/rank{R}/C{layer:02d}_<stage>.pt``.
        import os as _os
        try:
            import torch.distributed as _dist

            _rank = _dist.get_rank() if _dist.is_initialized() else 0
        except Exception:
            _rank = 0
        _root = _os.environ.get("DSV4_DEBUG_DUMP")
        _cdir = f"{_root}/rank{_rank:02d}" if _root else None

        def _cdump(stage: str, t: torch.Tensor) -> None:
            if _cdir:
                _os.makedirs(_cdir, exist_ok=True)
                torch.save(t.detach().to(torch.bfloat16).cpu(), f"{_cdir}/C{layer_idx:02d}_{stage}.pt")

        batch, seq_len, _ = hidden_states.shape
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        _cdump("kv_pre_pool", kv)
        _cdump("gate_pre_pool", gate)
        ready_kv, ready_gate, pool_base = cache.accumulate_windows(
            kv, gate, layer_idx, "compressor_state", self.compress_ratio, start_pos
        )
        new_pooled = self.kv_norm(
            _pool_windows(
                ready_kv,
                ready_gate,
                self.ape,
                self.compress_ratio,
                self.head_dim,
                overlap=self.overlap,
            )
        )
        _cdump("pooled_pre_rope", new_pooled)
        positions = _rope_pool_positions(new_pooled.shape[1], pool_base, self.compress_ratio, new_pooled.device, batch)
        cos, sin = rotary(new_pooled, positions)
        new_pooled = _apply_partial_rope(new_pooled.unsqueeze(1), cos, sin, self.rope_head_dim).squeeze(1)
        _cdump("pooled_post_rope", new_pooled)
        pooled = cache.update_pool(new_pooled, layer_idx, "compressor_state").unsqueeze(1)

        # Indexer narrows the attended compressed positions per query.  The
        # caller (DSV4Attention) is responsible for turning ``indexer_topk``
        # into an additive attention mask; we do NOT pre-gather here.  The
        # earlier per-query ``torch.gather`` produced an
        # ``[B, 1, S*topk, D]`` tensor that, when concatenated to ``full_kv``
        # and run through dense attention with ``F.pad(value=0.0)``, let
        # every query attend to every other query's gathered slice — a
        # silent non-causal leak (verified empirically: layer 2 attention
        # output cosine-vs-reference jumps from 0.81 to 0.99+ once we move
        # to mask-driven sparse semantics).
        #
        # ``indexer_topk`` follows the reference contract from
        # ``dsv4flash/inference/model.py:472-475``: shape ``[B, S, K]`` with
        # entries that are either valid pool indices in ``[0, P_total)``
        # or ``-1`` for "do not attend" (masked by causality).
        indexer_topk: torch.LongTensor | None = None
        if self.indexer is not None:
            raw_topk = self.indexer(
                hidden_states, q_residual, rotary, position_embeddings, cache, layer_idx, start_pos
            )
            threshold = (
                torch.arange(1, seq_len + 1, device=raw_topk.device) // self.compress_ratio
            ).unsqueeze(1)
            causal_invalid = raw_topk >= threshold
            indexer_topk = torch.where(
                causal_invalid, torch.full_like(raw_topk, -1), raw_topk
            )
            _cdump("indexer_topk", indexer_topk)
        return pooled, indexer_topk


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

        # ``pre`` and ``post`` have DIFFERENT formulas in the released DSV4-Flash
        # (see ``dsv4flash/inference/kernel.py:hc_split_sinkhorn_kernel`` 391-394):
        #   pre  = sigmoid(...) + eps     range (eps, 1+eps]
        #   post = 2 * sigmoid(...)       range (0, 2)  — NO +eps, AND a 2x prefactor
        # HF transformers PR 45616 / 45643 treats post identically to pre (sigmoid
        # + eps), which makes ``post`` half the magnitude the released weights
        # were trained against — verified empirically on the parity test
        # (auto post std = 0.5x ref post std before this fix).
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc].float()) + self.hc_eps
        post = 2.0 * torch.sigmoid(mix[..., hc : 2 * hc] * post_scale + self.base[hc : 2 * hc].float())

        # ``comb`` uses softmax(dim=-1) on raw logits + eps, then sinkhorn.  HF
        # uses sigmoid + eps + sinkhorn — also a divergence from the reference
        # kernel.  Reference (kernel.py:395-413):
        #   1. comb_logit = mix * scale + base
        #   2. row_softmax(dim=-1) + eps   (numerically stable, NOT sigmoid)
        #   3. col-norm / sum(dim=-2)
        #   4. for sinkhorn_iters - 1: row-norm / sum(dim=-1) ; col-norm / sum(dim=-2)
        comb_logit = (
            mix[..., 2 * hc :].view(*mix.shape[:-1], hc, hc) * comb_scale
            + self.base[2 * hc :].view(hc, hc).float()
        )
        comb = torch.softmax(comb_logit, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
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
# DeepseekV4Attention — port of HF PR 45616's DeepseekV4Attention with:
#   - no inheritance from DeepseekV3Attention (we don't want the HF
#     PreTrainedModel scaffolding);
#   - ``position_embeddings`` passed in from DeepseekV4Model as a ``(cos, sin)``
#     tuple produced by a matching ``DeepseekV4RotaryEmbedding`` (plus a
#     separate rotary / position_embeddings pair for the compressor path);
#   - ``past_key_values`` always ``None`` on the training path; the compressor
#     / indexer use a per-forward ``DeepseekV4TrainCache`` shim that behaves
#     the same as HF's ``DeepseekV4Cache`` within a single call;
#   - a fixed dispatch to ``eager_attention_with_sink`` (KAutomodel does not
#     register attention backends in HF's ``ALL_ATTENTION_FUNCTIONS`` table,
#     and the sink fold-in needs this path specifically).
# ---------------------------------------------------------------------------


class DeepseekV4Attention(nn.Module):
    """Sliding-window attention + Compressor + Indexer + attention sink.

    Single-head KV (``num_key_value_heads=1``), grouped low-rank output via
    :class:`DeepseekV4GroupedLinear`.  ``compress_ratio == 0`` layers skip
    the compressor / indexer and run pure SWA.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int, backend: BackendConfig | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = int(config.compress_ratios[layer_idx]) if config.compress_ratios else 0
        self.num_heads = config.num_attention_heads
        # Single KV head broadcast to all attention heads (``num_key_value_groups == num_heads``).
        self.num_key_value_groups = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.sliding_window = int(getattr(config, "sliding_window", 128) or 128)
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0) or 0.0)
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = initialize_rms_norm_module("torch_fp32", config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = initialize_rms_norm_module("torch_fp32", self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))

        self.compressor = (
            DeepseekV4Compressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_embeddings_compress: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_compress: nn.Module | None = None,
        start_pos: int = 0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del kwargs
        batch, seq_len = hidden_states.shape[:2]
        # IMPORTANT: for compress_ratio>0 layers the released DSV4-Flash uses
        # the compress-rope (theta=160000 + YaRN) for the MAIN attention Q/KV
        # too, NOT just for the compressor sub-module.  Reference at
        # ``dsv4flash/inference/model.py:476-501`` builds ``self.freqs_cis``
        # with ``compress_rope_theta`` whenever ``compress_ratio != 0``.  The
        # caller passes both ``position_embeddings`` (theta=10000, no YaRN)
        # and ``position_embeddings_compress`` (theta=160000, YaRN); we pick
        # the right one here based on compress_ratio.
        if self.compress_ratio and position_embeddings_compress is not None:
            cos, sin = position_embeddings_compress
        else:
            cos, sin = position_embeddings

        # Intra-attention debug dumps (paired with reference Attention.forward).
        # Files written to ``$DSV4_DEBUG_DUMP/rank{R}/A{layer:02d}_<stage>.pt``.
        import os as _os
        try:
            import torch.distributed as _dist

            _rank = _dist.get_rank() if _dist.is_initialized() else 0
        except Exception:
            _rank = 0
        _root = _os.environ.get("DSV4_DEBUG_DUMP")
        _adir = f"{_root}/rank{_rank:02d}" if _root else None
        _li = self.layer_idx

        def _adump(stage: str, t: torch.Tensor) -> None:
            if _adir:
                _os.makedirs(_adir, exist_ok=True)
                torch.save(t.detach().to(torch.bfloat16).cpu(), f"{_adir}/A{_li:02d}_{stage}.pt")

        q_residual = self.q_norm(self.wq_a(hidden_states))
        _adump("q_residual", q_residual)
        q_perhead = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        _adump("q_perhead_pre_rsqrt", q_perhead)  # [B, S, H, D] — ref-aligned
        q = q_perhead.transpose(1, 2)  # [B, H, S, D]
        kv = self.kv_norm(self.wkv(hidden_states)).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        _adump("kv_pre_rope", kv.squeeze(1))  # [B, S, D] — ref-aligned

        # Per-head, non-learnable rsqrt on Q before RoPE (matches reference
        # ``dsv4flash/inference/model.py:498``; missing from HF PR 45616).
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
        _adump("q_post_rsqrt", q.transpose(1, 2))  # [B, S, H, D]

        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim)
        kv = _apply_partial_rope(kv, cos, sin, self.rope_head_dim)
        _adump("q_post_rope", q.transpose(1, 2))  # [B, S, H, D]
        _adump("kv_post_rope", kv.squeeze(1))  # [B, S, D]

        full_kv = kv

        if self.compressor is not None:
            assert rotary_compress is not None and position_embeddings_compress is not None, (
                "DeepseekV4Attention: compressor enabled but no rotary_compress / "
                "position_embeddings_compress supplied by the Block/Model."
            )
            cache = DeepseekV4TrainCache()
            pooled, indexer_topk = self.compressor(
                hidden_states,
                q_residual=q_residual,
                rotary=rotary_compress,
                position_embeddings=position_embeddings_compress,
                cache=cache,
                layer_idx=self.layer_idx,
                start_pos=start_pos,
            )
            n_pooled = pooled.shape[2]
            full_kv = torch.cat([full_kv, pooled], dim=2)

            # Extend the additive 4D attention mask with a per-query
            # compressed-position mask so dense attention reproduces the
            # reference's ``sparse_attn`` semantics (per-query topk_idxs +
            # causality on the compressed pool).
            #
            # * compress_ratio == 4 (Indexer present): mask=0 only at the
            #   pool positions selected by ``indexer_topk`` for that query,
            #   -inf elsewhere.  ``-1`` entries in ``indexer_topk`` are
            #   already causally-masked by Compressor.
            # * compress_ratio > 4 (no Indexer, e.g. 128): every query q can
            #   attend to compressed position p iff ``p < (q+1) // ratio``
            #   (matches ``get_compress_topk_idxs`` in
            #   ``dsv4flash/inference/model.py:289-296``).
            if attention_mask is not None and n_pooled > 0:
                min_val = torch.finfo(attention_mask.dtype).min
                if indexer_topk is not None:
                    valid = indexer_topk != -1  # [B, S, K]
                    safe_idx = indexer_topk.clamp(min=0)
                    indicator = torch.zeros(
                        (batch, seq_len, n_pooled),
                        dtype=torch.bool,
                        device=full_kv.device,
                    )
                    indicator.scatter_(-1, safe_idx, valid)
                    compressed_mask = torch.where(
                        indicator,
                        torch.zeros((), dtype=attention_mask.dtype, device=full_kv.device),
                        torch.full((), min_val, dtype=attention_mask.dtype, device=full_kv.device),
                    )  # [B, S, P]
                else:
                    q_pos = torch.arange(seq_len, device=full_kv.device)
                    p_pos = torch.arange(n_pooled, device=full_kv.device)
                    threshold = (q_pos + 1) // self.compress_ratio
                    allowed = p_pos.unsqueeze(0) < threshold.unsqueeze(1)  # [S, P]
                    compressed_mask = torch.where(
                        allowed,
                        torch.zeros((), dtype=attention_mask.dtype, device=full_kv.device),
                        torch.full((), min_val, dtype=attention_mask.dtype, device=full_kv.device),
                    ).expand(batch, seq_len, n_pooled)
                compressed_mask = compressed_mask.unsqueeze(1)  # [B, 1, S, P]
                attention_mask = torch.cat([attention_mask, compressed_mask], dim=-1)

        # If a caller supplied a 4D mask shorter than full_kv but no compressor
        # ran (shouldn't happen, but kept for defense), fall back to neutral pad.
        if attention_mask is not None and full_kv.shape[2] > attention_mask.shape[-1]:
            attention_mask = F.pad(
                attention_mask, (0, full_kv.shape[2] - attention_mask.shape[-1]), value=0.0
            )

        attn_output, attn_weights = eager_attention_with_sink(
            self,
            q,
            full_kv,
            full_kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        # eager_attention_with_sink returns [B, S, H, D] (already transposed).
        _adump("attn_o_pre_inv_rope", attn_output)  # [B, S, H, D]

        # Inverse RoPE on the attention output (same (cos, -sin) conjugate pattern
        # HF uses).  Reference: modular_deepseek_v4.py:607.
        attn_output = _apply_partial_rope(
            attn_output.transpose(1, 2), cos, -sin, self.rope_head_dim
        ).transpose(1, 2)
        _adump("attn_o_post_inv_rope", attn_output)  # [B, S, H, D]

        grouped = attn_output.reshape(batch, seq_len, -1).view(batch, seq_len, self.config.o_groups, -1)
        wo_a_out = self.wo_a(grouped)
        _adump("wo_a_out", wo_a_out)
        out = self.wo_b(wo_a_out.flatten(2))
        _adump("wo_b_out", out)
        return out, attn_weights

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for linear in (self.wq_a, self.wq_b, self.wkv, self.wo_b, self.wo_a):
            if hasattr(linear, "weight"):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        for norm in (self.q_norm, self.kv_norm):
            norm.reset_parameters()
        nn.init.zeros_(self.sinks)
        if self.compressor is not None:
            for mod in self.compressor.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.trunc_normal_(mod.weight, mean=0.0, std=init_std)
            nn.init.zeros_(self.compressor.ape)
            if self.compressor.indexer is not None:
                nn.init.zeros_(self.compressor.indexer.ape)
