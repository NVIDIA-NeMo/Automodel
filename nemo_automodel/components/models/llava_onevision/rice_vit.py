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
"""Rice Vision Transformer for LLaVA-OneVision-1.5.

Ported from lmms-lab/LLaVA-OneVision-1.5's modeling_llavaonevision1_5.py.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers.activations import ACT2FN


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def _block_diagonal_attention_mask(
    cu_seqlens: torch.Tensor,
    sequence_length: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a block-diagonal mask without host-side segment iteration.

    Args:
        cu_seqlens: Packed segment offsets with shape [num_segments + 1].
        sequence_length: Total number of packed tokens.
        dtype: When omitted, return a boolean visibility mask. Otherwise return
            an additive mask in this dtype with zero for visible positions.
        device: Device on which to construct the mask. Defaults to the offsets'
            device.

    Returns:
        Attention mask with shape [1, sequence_length, sequence_length]. Inputs
        are not mutated.
    """
    mask_device = device if device is not None else cu_seqlens.device
    segment_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(mask_device)
    segment_ids = torch.repeat_interleave(
        torch.arange(segment_lengths.numel(), device=mask_device),
        segment_lengths,
        output_size=sequence_length,
    )
    visible = segment_ids[:, None] == segment_ids[None, :]
    if dtype is None:
        return visible.unsqueeze(0)
    additive_mask = torch.full(
        (sequence_length, sequence_length),
        torch.finfo(dtype).min,
        device=mask_device,
        dtype=dtype,
    )
    return additive_mask.masked_fill(visible, 0).unsqueeze(0)


class RiceRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class RicePatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        # HF reference hard-codes temporal_patch_size=1; temporal stacking is done
        # externally via grid_thw[:, 0] + cu_seqlens repeat_interleave.
        self.temporal_patch_size = 1
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=[patch_size, patch_size], stride=[patch_size, patch_size], bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class RicePatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-05,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class RiceMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RiceAttention(nn.Module):
    """Eager block-diagonal attention over variable-length image segments."""

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = _block_diagonal_attention_mask(cu_seqlens, seq_length, q.dtype, q.device)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v).transpose(0, 1).reshape(seq_length, -1)
        return self.proj(attn_output)


class RiceFlashAttention2(nn.Module):
    """Flash-attention-2 variant using flash_attn_varlen_func (requires flash_attn)."""

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func

        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        return self.proj(attn_output)


class RiceSdpaAttention(nn.Module):
    """SDPA variant with an additive block-diagonal mask."""

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = _block_diagonal_attention_mask(cu_seqlens, seq_length, device=q.device)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
        return self.proj(attn_output)


_ATTENTION_CLASSES = {
    "eager": RiceAttention,
    "sdpa": RiceSdpaAttention,
    "flash_attention_2": RiceFlashAttention2,
}


class RiceBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "eager") -> None:
        super().__init__()
        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        attn_cls = _ATTENTION_CLASSES.get(attn_implementation, RiceAttention)
        self.attn = attn_cls(config.hidden_size, num_heads=config.num_heads)
        self.mlp = RiceMlp(
            dim=config.hidden_size,
            hidden_dim=int(config.intermediate_size),
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class RiceTransformer(nn.Module):
    """Rice ViT with per-image class-token insertion and block-diagonal attention.

    Matches the HF reference: one CLS token is prepended at the start of each
    image segment inside the flat packed sequence, and the attention mask is
    built from a cu_seqlens that accounts for the extra CLS per segment.
    """

    def __init__(self, config, attn_implementation: str = "eager"):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.patch_embed = RicePatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = RiceRotaryEmbedding(head_dim // 2)

        scale = config.hidden_size**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(config.hidden_size))
        self.class_pos_emb = nn.Parameter(torch.randn(1, head_dim // 2))

        self.blocks = nn.ModuleList([RiceBlock(config, attn_implementation) for _ in range(config.depth)])
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.merger = RicePatchMerger(
            dim=config.text_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(pixel_values)
        # rot_pos_emb depends on the (non-persistent) ``inv_freq`` buffer — which
        # can miss a to(device) call after LoRA/FSDP transforms — so pin it to
        # hidden_states' device explicitly.
        rotary_pos_emb = self.rot_pos_emb(grid_thw).to(hidden_states.device)
        img_feats = hidden_states.shape[0]

        # Per-segment sequence lengths: H*W repeated T times per image
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        cu = cu_seqlens.to(device=hidden_states.device, dtype=torch.long)
        num_segments = cu.numel() - 1

        cls_token = self.class_embedding.to(hidden_states.dtype).unsqueeze(0)
        new_total = img_feats + num_segments
        D = hidden_states.size(-1)

        new_hidden = hidden_states.new_empty((new_total, D))
        new_rotary_pos_emb = rotary_pos_emb.new_empty((new_total, rotary_pos_emb.shape[-1]))

        segment_lengths = cu[1:] - cu[:-1]
        segment_ids = torch.repeat_interleave(
            torch.arange(num_segments, device=hidden_states.device),
            segment_lengths,
            output_size=img_feats,
        )
        patch_positions = torch.arange(img_feats, device=hidden_states.device) + segment_ids + 1
        cls_positions = cu[:-1] + torch.arange(num_segments, device=hidden_states.device)
        new_hidden.index_copy_(0, patch_positions, hidden_states)
        new_hidden.index_copy_(0, cls_positions, cls_token.expand(num_segments, -1))
        new_rotary_pos_emb.index_copy_(0, patch_positions, rotary_pos_emb)
        new_rotary_pos_emb.index_copy_(
            0,
            cls_positions,
            self.class_pos_emb.to(new_rotary_pos_emb).expand(num_segments, -1),
        )

        hidden_states = new_hidden
        cu_seqlens = (cu + torch.arange(num_segments + 1, device=hidden_states.device)).to(torch.int32)
        rotary_pos_emb = new_rotary_pos_emb

        hidden_states = self.pre_layernorm(hidden_states)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        # ``patch_positions`` points at every non-CLS token in original patch
        # order, so one indexed read strips all per-segment CLS tokens.
        stripped = hidden_states[patch_positions]

        return self.merger(stripped)
