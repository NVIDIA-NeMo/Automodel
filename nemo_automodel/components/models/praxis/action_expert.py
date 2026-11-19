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

"""Action expert: a small transformer that denoises the action tokens.

The expert reads the encoder's input sequence (its global tokens) and the encoder block
outputs at the tap blocks; it never writes back into the video stream. Its layers
alternate cross attention (even layers, one tap each) and self attention (odd layers).

Expert query sequence::

    [ proprio (1 per chunk) | future tokens (F per chunk) | action tokens (1 per noised frame) ]

Cross-attention key/value sequence::

    [ global tokens (register, anchor, text) | clean frame tokens | noised frame tokens ]

Queries and keys turn by the trunk rotary positions before their RMS norm; the rotary
position of a frame is the position of its first token.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nemo_automodel.components.models.praxis.backbone import EncoderFeatures
from nemo_automodel.components.models.praxis.config import ActionExpertConfig
from nemo_automodel.components.models.praxis.layers import IntegerRotaryEmbedding, apply_rotary
from nemo_automodel.components.models.praxis.masks import (
    KIND_CLEAN,
    KIND_GLOBAL,
    KIND_NOISED,
    MaskCache,
    Slot,
    TokenTags,
    dense_mask,
    expert_cross_rule,
    expert_self_rule,
    window_tags,
)


def rms_norm(x: Tensor, eps: float = 1e-4) -> Tensor:
    """Weightless RMS norm over the last axis; the mean and scale run in float32.

    Args:
        x: Tensor of shape [..., hidden].

    Returns:
        Tensor of shape [..., hidden] in the dtype of ``x``.
    """
    mean_sq = torch.mean(x**2, dim=-1, keepdim=True, dtype=torch.float32)
    return (x.to(torch.float32) * torch.rsqrt(mean_sq + eps)).to(x.dtype)


def geglu(x: Tensor) -> Tensor:
    """Gated GELU: ``gelu_tanh(first half) * second half`` over the last axis."""
    value, gate = torch.chunk(x, 2, dim=-1)
    return F.gelu(value, approximate="tanh") * gate


class SinusoidalTimestep(nn.Module):
    """``[cos(t f), sin(t f)]`` features of the flow time followed by a two-layer MLP."""

    def __init__(self, dim: int, frequency_dim: int = 256, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps.

        Args:
            t: Float tensor of shape [...].

        Returns:
            Tensor of shape [..., dim] in the MLP weight dtype.
        """
        half = self.frequency_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        freqs = torch.exp(exponent).to(t.dtype)
        args = t.unsqueeze(-1) * freqs
        weight = self.mlp[0].weight
        features = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(weight.dtype)
        return self.mlp(features)


class ScaleAdaLN(nn.Module):
    """Scale-only adaptive RMS norm: ``(1 + proj(silu(cond))) * rms_norm(x)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, tokens: Tensor, cond: Tensor) -> Tensor:
        """Normalize and scale.

        Args:
            tokens: Tensor of shape [batch, queries, hidden].
            cond: Tensor of shape [batch, queries, hidden].

        Returns:
            Tensor of shape [batch, queries, hidden] in the dtype of ``tokens``.
        """
        scale = self.proj(F.silu(cond))
        return ((1 + scale.float()) * rms_norm(tokens)).to(tokens.dtype)


class ActionTauEncoder(nn.Module):
    """Action token embedding with the flow time at the input: ``w3(silu(w2([w1(a); tau(t)])))``."""

    def __init__(self, action_patch_dim: int, dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(action_patch_dim, dim)
        self.tau = SinusoidalTimestep(dim)
        self.w2 = nn.Linear(2 * dim, dim)
        self.w3 = nn.Linear(dim, dim)

    def forward(self, actions: Tensor, tau: Tensor) -> Tensor:
        """Embed noised action tokens.

        Args:
            actions: Tensor of shape [batch, action_tokens, action_patch_dim].
            tau: Flow time per token, shape [batch, action_tokens].

        Returns:
            Tensor of shape [batch, action_tokens, hidden].
        """
        a = self.w1(actions)
        t = self.tau(tau).to(a.dtype)
        return self.w3(F.silu(self.w2(torch.cat([a, t], dim=-1))))


class ActionExpertBlock(nn.Module):
    """One expert layer: cross or self attention, then a GEGLU MLP, each behind a scale-only AdaLN."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, is_cross: bool) -> None:
        super().__init__()
        self.is_cross = is_cross
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.adaln_attn = ScaleAdaLN(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, 2 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)
        self.adaln_mlp = ScaleAdaLN(dim)
        self.mlp_in = nn.Linear(dim, 8 * dim)
        self.mlp_out = nn.Linear(4 * dim, dim)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        kv_source: Tensor | None,
        mask: Tensor,
        q_table: Tensor,
        k_table: Tensor,
    ) -> Tensor:
        """Run the layer.

        Args:
            x: Expert tokens of shape [batch, queries, hidden].
            cond: Timestep conditioning of shape [batch, queries, hidden].
            kv_source: Cross-attention source of shape [batch, keys, hidden], or ``None`` for self attention.
            mask: Boolean tensor of shape [queries, keys], true where attention is allowed.
            q_table: Rotary table of the queries, shape [batch, queries, head_dim // 2, 2].
            k_table: Rotary table of the keys, shape [batch, keys, head_dim // 2, 2].

        Returns:
            Tensor of shape [batch, queries, hidden].
        """
        b, q_len, _ = x.shape
        h = self.adaln_attn(x, cond)
        q = self.to_q(h)
        kv_in = kv_source if self.is_cross else h
        if kv_in is None:
            raise ValueError("a cross-attention block needs kv_source")
        k, v = self.to_kv(kv_in).chunk(2, dim=-1)

        def heads(t: Tensor) -> Tensor:
            return t.view(b, t.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        q = rms_norm(apply_rotary(heads(q), q_table))
        k = rms_norm(apply_rotary(heads(k), k_table))
        attn = F.scaled_dot_product_attention(q, k, heads(v), attn_mask=mask[None, None])
        x = x + self.to_out(attn.transpose(1, 2).reshape(b, q_len, -1))
        return x + self.mlp_out(geglu(self.mlp_in(self.adaln_mlp(x, cond))))


def _query_chunks(num_action_tokens: int, num_future: int, chunk_size: int, device: torch.device) -> Tensor:
    """Chunk index of every expert query in the ``[proprio | future | actions]`` layout."""
    num_chunks = max(1, num_action_tokens // chunk_size)
    chunks = torch.arange(num_chunks, device=device)
    actions = torch.arange(num_action_tokens, device=device) // chunk_size
    return torch.cat([chunks, chunks.repeat_interleave(num_future), actions])


class ActionExpert(nn.Module):
    """The action transformer (see the module docstring)."""

    def __init__(self, cfg: ActionExpertConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.num_future = cfg.num_future_per_chunk
        self.rope = IntegerRotaryEmbedding(cfg.rope.dim, cfg.rope.effective_dim, cfg.rope.base, cfg.rope.table_size)
        self.action_encoder = ActionTauEncoder(cfg.action_patch_dim, cfg.dim)
        self.timestep_embedder = SinusoidalTimestep(cfg.dim)
        self.future_tokens = nn.Parameter(torch.zeros(cfg.num_future_per_chunk, cfg.dim))
        self.future_chunk_embed = nn.Embedding(cfg.max_chunks, cfg.dim)
        self.proprio_token = nn.Parameter(torch.zeros(1, cfg.dim))
        self.embed_proprio = nn.Linear(cfg.proprio_patch_dim, cfg.dim)
        self.tap_proj = nn.Linear(cfg.trunk_dim, cfg.dim, bias=False)
        self.text_proj = nn.Linear(cfg.trunk_dim, cfg.dim, bias=False)
        self.tap_norm_scale = nn.Parameter(torch.ones(cfg.trunk_dim)) if cfg.tap_norm else None
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(cfg.dim, cfg.num_heads, cfg.head_dim, is_cross=(i % 2 == 0))
                for i in range(cfg.num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(cfg.dim, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(cfg.dim, cfg.action_patch_dim)
        self._masks = MaskCache()

    def _masks_for(
        self, q_tags: TokenTags, kv_tags: TokenTags, window: int, key: tuple[int, int, int, int, str]
    ) -> tuple[Tensor, Tensor]:
        device = q_tags.kind.device

        def build_cross() -> Tensor:
            return dense_mask(expert_cross_rule(q_tags, kv_tags, window), q_tags.length, kv_tags.length, device)

        def build_self() -> Tensor:
            return dense_mask(expert_self_rule(q_tags), q_tags.length, q_tags.length, device)

        cross = self._masks.get(("cross", *key), build_cross)
        self_mask = self._masks.get(("self", *key), build_self)
        assert isinstance(cross, Tensor) and isinstance(self_mask, Tensor)
        return cross, self_mask

    def forward(
        self,
        features: EncoderFeatures,
        noised_actions: Tensor,
        action_timesteps: Tensor,
        proprio: Tensor,
    ) -> Tensor:
        """Predict the action velocity from the encoder features of one window.

        Args:
            features: Encoder taps and sequence layout of the window (batch size 1).
            noised_actions: Noised action tokens of shape [1, K * F, action_patch_dim], chunk-major.
            action_timesteps: Flow time of each chunk, shape [1, K].
            proprio: Normalized boundary pose of each chunk, shape [1, K, proprio_patch_dim].

        Returns:
            Action velocity of shape [1, K * F, action_patch_dim] in the taps dtype.
        """
        kk, cs, tpf = features.n_chunks, features.frames_per_chunk, features.tokens_per_frame
        n_noised, n_clean = cs * kk, cs * (kk - 1)
        device = features.taps[0].device
        dtype = features.embeddings.dtype
        if features.embeddings.shape[0] != 1:
            raise ValueError("the action expert runs one window at a time")
        if noised_actions.shape[:2] != (1, n_noised):
            raise ValueError(f"noised_actions {tuple(noised_actions.shape)} for {n_noised} action tokens")

        def span_indices(span: tuple[int, int] | None) -> list[int]:
            if span is None:
                return []
            return list(range(span[0], span[0] + span[1]))

        g_pos = [p for kind in self.cfg.global_order for p in span_indices(features.global_spans[kind])]
        s_pos = span_indices(features.clean_span) + span_indices(features.noised_span)
        g_idx = torch.tensor(g_pos, dtype=torch.long, device=device)
        s_idx = torch.tensor(s_pos, dtype=torch.long, device=device)
        n_global = len(g_pos)
        ids = features.positions.to(device)
        kv_ids = torch.cat([ids.index_select(1, g_idx), ids.index_select(1, s_idx)], dim=1)

        # Query sequence [proprio (K) | future (K * F) | actions (T)].
        a_t = action_timesteps.to(device).repeat_interleave(cs, dim=1)
        act = self.action_encoder(noised_actions.to(device, dtype), a_t)
        fut = (
            self.future_tokens[None, None].expand(1, kk, -1, -1) + self.future_chunk_embed.weight[:kk][None, :, None, :]
        )
        fut = fut.reshape(1, kk * self.num_future, self.dim).to(act.dtype)
        prop = self.proprio_token[None].expand(1, kk, -1).to(act.dtype) + self.embed_proprio(
            proprio.to(device, act.dtype)
        )
        x = torch.cat([prop, fut, act], dim=1)

        q_chunk = _query_chunks(n_noised, self.num_future, cs, device)
        tok_frame = (q_chunk * cs).clamp(min=0, max=n_noised - 1)
        cond = self.timestep_embedder(a_t.gather(1, tok_frame[None])).to(x.dtype)

        # Rotary positions: a frame is identified by its first token; action token i takes
        # noised frame i, the proprio and future tokens of chunk k take noised frame k * F.
        frame_ids = kv_ids[:, n_global:].view(1, n_clean + n_noised, tpf).amin(dim=-1)
        noised_ids = frame_ids[:, n_clean:]
        n_prefix = q_chunk.numel() - n_noised
        q_frame = torch.cat([(q_chunk[:n_prefix] * cs).clamp(min=0), torch.arange(n_noised, device=device)])
        q_ids = noised_ids.gather(1, q_frame[None])
        q_table = self.rope(q_ids)
        k_table = self.rope(kv_ids)

        q_tags = TokenTags(
            kind=torch.full_like(q_chunk, KIND_NOISED, dtype=torch.int32),
            chunk=q_chunk.to(torch.int32),
            frame=(q_chunk * cs).to(torch.int32),
        )
        kv_tags = window_tags(
            (Slot(KIND_GLOBAL, 1, n_global), Slot(KIND_CLEAN, n_clean, tpf), Slot(KIND_NOISED, n_noised, tpf)),
            cs,
            device,
        )
        cross_mask, self_mask = self._masks_for(
            q_tags, kv_tags, n_clean + n_noised, (kk, cs, tpf, n_global, str(device))
        )

        text_kv = self.text_proj(features.embeddings.index_select(1, g_idx))
        cross_index = 0
        for block in self.blocks:
            if block.is_cross:
                feat = features.taps[cross_index].index_select(1, s_idx)
                if self.tap_norm_scale is not None:
                    feat = (rms_norm(feat) * self.tap_norm_scale).to(feat.dtype)
                kv = torch.cat([text_kv.to(feat.dtype), self.tap_proj(feat)], dim=1)
                x = block(x, cond, kv, cross_mask, q_table, k_table)
                cross_index += 1
            else:
                x = block(x, cond, None, self_mask, q_table, q_table)
        out = self.head(self.norm_out(x[:, -n_noised:]))
        return out.view(1, n_noised, self.cfg.action_patch_dim)
