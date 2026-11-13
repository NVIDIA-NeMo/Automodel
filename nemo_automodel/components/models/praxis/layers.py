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

"""Transformer layers shared by the two towers of the Praxis video backbone.

Every block is pre-norm with a zero-centred RMSNorm (scale ``1 + weight``), gated
grouped-query attention over integer rotary positions, and either a dense SwiGLU or an
Automodel :class:`~nemo_automodel.components.moe.layers.MoE` feed-forward. The decoder
tower modulates each block input with a per-token shift and scale from the encoder.

Norms, rotary rotation, modulation and the router run in float32; everything else in
the parameter dtype (bf16 in the released checkpoint).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MLP, MoE

ModulationDtype = Literal["bfloat16", "float32", "mixed"]

# Automatic dynamic shapes: the sequence length changes with the prompt length, and a static
# compile would hit the recompile limit and fall back to eager flex attention, which
# materializes the dense attention matrix.
_compiled_flex_attention = torch.compile(flex_attention)


def modulate(x: Tensor, shift: Tensor, scale: Tensor, dtype: ModulationDtype) -> Tensor:
    """Return ``x * scale + shift`` in the dtype of ``x``.

    Args:
        x: Tensor of shape [batch, tokens, hidden].
        shift: Tensor broadcastable to ``x``.
        scale: Tensor broadcastable to ``x``.
        dtype: ``"mixed"`` and ``"float32"`` compute in float32, ``"bfloat16"`` in bf16.

    Returns:
        Tensor of shape [batch, tokens, hidden] in the dtype of ``x``.
    """
    compute = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    return (x.to(compute) * scale.to(compute) + shift.to(compute)).to(x.dtype)


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm whose scale is ``1 + weight``; ``cast_dtype`` selects the compute dtype."""

    def __init__(self, dim: int, eps: float = 1e-6, cast_dtype: torch.dtype | None = torch.float32) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.cast_dtype = cast_dtype
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize over the last axis.

        Args:
            x: Tensor of shape [..., hidden].

        Returns:
            Tensor of shape [..., hidden] in the dtype of ``x``.
        """
        weight = self.weight
        if self.cast_dtype is not None:
            x_c, weight = x.to(self.cast_dtype), weight.to(self.cast_dtype)
        else:
            x_c = x
        out = F.rms_norm(x_c, (self.dim,), weight=1 + weight, eps=self.eps)
        return out.to(x.dtype)


class IntegerRotaryEmbedding(nn.Module):
    """Rotary (cos, sin) table for large integer positions.

    Only the first ``effective_dim // 2`` frequency pairs rotate; the remaining pairs have
    zero frequency. A position ``p`` is split as ``p = high * table_size + low`` and the
    angle of each part is formed in float64 so that positions in the millions keep their
    precision; the two parts are combined with the angle-sum identity.
    """

    def __init__(self, dim: int, effective_dim: int, base: int, table_size: int) -> None:
        super().__init__()
        if effective_dim > dim or effective_dim % 2 or dim % 2:
            raise ValueError(f"rotary dims must be even with effective_dim <= dim, got {effective_dim}, {dim}")
        self.dim = dim
        self.effective_dim = effective_dim
        self.base = base
        self.table_size = table_size
        # A plain attribute: module casts and FSDP must not touch the frequencies.
        self.theta = self.make_theta(torch.device("cpu"))

    def make_theta(self, device: torch.device) -> Tensor:
        """Per-pair frequencies of shape [dim // 2] in float32 on ``device`` (zero beyond ``effective_dim``)."""
        exponent = torch.arange(0, self.effective_dim, 2, dtype=torch.float32, device=device) / self.effective_dim
        pad = torch.zeros((self.dim - self.effective_dim) // 2, dtype=torch.float32, device=device)
        return torch.cat([1.0 / (self.base**exponent), pad])

    def forward(self, positions: Tensor) -> Tensor:
        """Build the rotary table.

        Args:
            positions: Integer tensor of shape [batch, tokens].

        Returns:
            Float32 tensor of shape [batch, tokens, dim // 2, 2] holding ``(cos, sin)`` per pair.
        """
        theta = self.theta.to(positions.device).double()
        high = positions // self.table_size
        low = positions % self.table_size
        high_pos = (high.to(torch.float32) * self.table_size).double()
        low_pos = low.to(torch.float32).double()

        def cos_sin(angle: Tensor) -> tuple[Tensor, Tensor]:
            return angle.cos().float().double(), angle.sin().float().double()

        cos_h, sin_h = cos_sin(high_pos[..., None] * theta)
        cos_l, sin_l = cos_sin(low_pos[..., None] * theta)
        cos = cos_h * cos_l - sin_h * sin_l
        sin = sin_h * cos_l + cos_h * sin_l
        return torch.stack([cos, sin], dim=-1).float()


def apply_rotary(x: Tensor, table: Tensor) -> Tensor:
    """Rotate ``x`` with a rotary table in float32 (rotate-half layout).

    Args:
        x: Tensor of shape [batch, heads, tokens, head_dim]; channel ``j`` pairs with ``j + head_dim // 2``.
        table: Float32 tensor of shape [batch, tokens, head_dim // 2, 2] from :class:`IntegerRotaryEmbedding`.

    Returns:
        Tensor of shape [batch, heads, tokens, head_dim] in the dtype of ``x``.
    """
    x0, x1 = x.float().chunk(2, dim=-1)
    cos = table[..., 0].unsqueeze(1)
    sin = table[..., 1].unsqueeze(1)
    out = torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)
    return out.to(x.dtype)


class GatedAttention(nn.Module):
    """Grouped-query attention with a sigmoid output gate per head.

    Order: project, rotate q and k, RMS-normalize q and k per head, attend, gate, project out.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qk_norm_eps: float,
        attn_backend: str,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.attn_backend = attn_backend
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.gate_proj = nn.Linear(dim, num_heads, bias=False)
        self.q_norm = ZeroCenteredRMSNorm(head_dim, eps=qk_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=True)

    def forward(self, x: Tensor, rope: Tensor, mask: BlockMask | Tensor | None) -> Tensor:
        """Attend within one sequence.

        Args:
            x: Tensor of shape [batch, tokens, hidden].
            rope: Float32 rotary table of shape [batch, tokens, head_dim // 2, 2].
            mask: A flex ``BlockMask`` over [tokens, tokens], a boolean tensor of shape
                [tokens, tokens] (true where attention is allowed), or ``None`` for full attention.

        Returns:
            Tensor of shape [batch, tokens, hidden].
        """
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        gate = self.gate_proj(x).sigmoid().transpose(1, 2).unsqueeze(-1)
        q = self.q_norm(apply_rotary(q, rope))
        k = self.k_norm(apply_rotary(k, rope))
        if isinstance(mask, BlockMask):
            out = _compiled_flex_attention(q, k, v, block_mask=mask, enable_gqa=self.num_kv_heads != self.num_heads)
        else:
            attn_mask = None if mask is None else mask[None, None]
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=self.num_kv_heads != self.num_heads
            )
        out = (out * gate).transpose(1, 2).reshape(b, s, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm block ``x + attn(norm(x) * s)``, ``x + ffn(norm(x) * s)`` with ``s = 1 / sqrt(layer_index)``.

    ``s`` is a persistent buffer so the checkpoint value (rounded to the parameter dtype) is used.
    """

    s: Tensor

    def __init__(
        self,
        dim: int,
        attention: GatedAttention,
        ffn: nn.Module,
        layer_index: int,
        norm_eps: float,
        modulation_dim: int | None,
        modulation_dtype: ModulationDtype,
    ) -> None:
        super().__init__()
        self.register_buffer("s", torch.tensor(1.0 / math.sqrt(layer_index)))
        self.ln_in = ZeroCenteredRMSNorm(dim, eps=norm_eps)
        self.attn = attention
        self.ln_out = ZeroCenteredRMSNorm(dim, eps=norm_eps)
        self.ffn = ffn
        self.modulator = None if modulation_dim is None else nn.Linear(modulation_dim, 2 * dim)
        self.modulation_dtype: ModulationDtype = modulation_dtype

    @property
    def mlp(self) -> MLP | MoE:
        """Expose the feed-forward sub-block (``ffn``) under the name the shared MoE parallelizer looks up."""
        return self.ffn

    def forward(self, x: Tensor, rope: Tensor, mask: BlockMask | Tensor | None, x_mod: Tensor | None) -> Tensor:
        """Run the block.

        Args:
            x: Tensor of shape [batch, tokens, hidden].
            rope: Float32 rotary table of shape [batch, tokens, head_dim // 2, 2].
            mask: Attention mask (see :meth:`GatedAttention.forward`).
            x_mod: Per-token conditioning of shape [batch, tokens, modulation_dim] when the block
                has a modulator, else ``None``.

        Returns:
            Tensor of shape [batch, tokens, hidden].
        """
        if self.modulator is not None:
            if x_mod is None:
                raise ValueError("a modulated block needs x_mod")
            shift, scale = self.modulator(x_mod).chunk(2, dim=-1)
            x = modulate(x, shift, 1 + scale, self.modulation_dtype)
        x = x + self.attn(self.ln_in(x) * self.s, rope, mask)
        h = self.ffn(self.ln_out(x) * self.s)
        return x + h.to(x.dtype)


def build_moe_config(
    dim: int,
    expert_hidden: int,
    num_routed_experts: int,
    top_k_routed: int,
    routed_scaling_factor: float,
    dtype: torch.dtype,
    gate_bias_update_factor: float,
    aux_loss_coeff: float = 0.0,
) -> MoEConfig:
    """MoE settings of the Praxis trunk: fp32 sigmoid router with a selection bias, one shared expert.

    The routed sum is scaled by ``routed_scaling_factor`` and the top-k scores are
    renormalized to sum to one, as in DeepSeek-V3.
    """
    return MoEConfig(
        n_routed_experts=num_routed_experts,
        n_shared_experts=1,
        n_activated_experts=top_k_routed,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=gate_bias_update_factor,
        aux_loss_coeff=aux_loss_coeff,
        score_func="sigmoid",
        route_scale=routed_scaling_factor,
        dim=dim,
        inter_dim=expert_hidden,
        moe_inter_dim=expert_hidden,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        router_weights_fp32=True,
        force_e_score_correction_bias=True,
        dtype=dtype,
    )


class TransformerStack(nn.Module):
    """A stack of :class:`TransformerBlock` sharing one rotary table.

    Blocks listed in ``dense_blocks`` use a dense SwiGLU; the others an Automodel ``MoE``.
    """

    def __init__(
        self,
        depth: int,
        dense_blocks: tuple[int, ...],
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qk_norm_eps: float,
        norm_eps: float,
        dense_hidden: int,
        moe_config: MoEConfig,
        rope: IntegerRotaryEmbedding,
        modulation_dim: int | None,
        modulation_dtype: ModulationDtype,
        backend: BackendConfig,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rope = rope
        blocks = []
        for i in range(depth):
            ffn: nn.Module
            if i in dense_blocks:
                ffn = MLP(dim, dense_hidden, backend.linear, dtype=dtype, activation="swiglu", bias=False)
            else:
                ffn = MoE(moe_config, backend)
            attention = GatedAttention(dim, num_heads, num_kv_heads, head_dim, qk_norm_eps, backend.attn)
            blocks.append(TransformerBlock(dim, attention, ffn, i + 1, norm_eps, modulation_dim, modulation_dtype))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        mask: BlockMask | Tensor | None,
        x_mod: Tensor | None,
        tap_blocks: tuple[int, ...],
    ) -> tuple[Tensor, list[Tensor]]:
        """Run every block and collect the tap outputs.

        Args:
            x: Tensor of shape [batch, tokens, hidden].
            positions: Integer tensor of shape [batch, tokens] with the rotary positions.
            mask: Attention mask (see :meth:`GatedAttention.forward`).
            x_mod: Per-token conditioning of shape [batch, tokens, modulation_dim], or ``None``.
            tap_blocks: Indices of the blocks whose outputs are returned.

        Returns:
            The output of shape [batch, tokens, hidden] and the tap outputs, each of shape
            [batch, tokens, hidden], in the order of ``tap_blocks``.
        """
        rope = self.rope(positions)
        taps: dict[int, Tensor] = {}
        for i, block in enumerate(self.blocks):
            x = block(x, rope, mask, x_mod)
            if i in tap_blocks:
                taps[i] = x
        return x, [taps[i] for i in tap_blocks]
