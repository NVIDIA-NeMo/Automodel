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

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_dtype, key_dtype = query.dtype, key.dtype
    query, key = query.float(), key.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)
    return query.to(query_dtype), key.to(key_dtype)


class MiMoVisionRotaryEmbedding(nn.Module):
    """Two-dimensional rotary frequencies used by the MiMo vision tower."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = self._build_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._inv_freq_initialized = not inv_freq.is_meta

    def _build_inv_freq(self, device: torch.device | None = None) -> torch.Tensor:
        positions = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        return 1.0 / (self.theta ** (positions / self.dim))

    def forward(self, seqlen: int, device: torch.device | None = None) -> torch.Tensor:
        device = device or self.inv_freq.device
        if not self._inv_freq_initialized or self.inv_freq.is_meta or self.inv_freq.device != device:
            self.inv_freq = self._build_inv_freq(device)
            self._inv_freq_initialized = True
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class MiMoVisionPatchEmbed(nn.Module):
    """Convert flattened spatiotemporal image patches to vision tokens."""

    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        embed_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(dtype=self.proj.weight.dtype)).view(-1, self.embed_dim)


class MiMoVisionSwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network used by each vision block."""

    def __init__(self, dim: int, intermediate_dim: int, hidden_act: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=True, dtype=dtype)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=True, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=True, dtype=dtype)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MiMoVisionAttention(nn.Module):
    """Grouped-query vision attention with optional local windows and sinks."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        use_sinks: bool,
        window_size: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.window_size = window_size

        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        self.qkv = nn.Linear(dim, qkv_dim, bias=True, dtype=dtype)
        self.proj = nn.Linear(num_heads * head_dim, dim, bias=True, dtype=dtype)
        self.sinks = nn.Parameter(torch.zeros(num_heads, dtype=dtype)) if use_sinks else None

    def _build_window_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.window_size <= 0:
            return None
        row = torch.arange(seq_len, device=device).unsqueeze(1)
        col = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        return mask.masked_fill((row - col).abs() > self.window_size, float("-inf"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        full_attn: bool = False,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        query_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        query = qkv[:, :query_dim].view(seq_len, self.num_heads, self.head_dim)
        key = qkv[:, query_dim : query_dim + kv_dim].view(seq_len, self.num_kv_heads, self.head_dim)
        value = qkv[:, query_dim + kv_dim :].view(seq_len, self.num_kv_heads, self.head_dim)
        query, key = _apply_rotary_pos_emb(query, key, *position_embeddings)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = []
        for query_chunk, key_chunk, value_chunk in zip(
            torch.split(query, lengths, dim=0),
            torch.split(key, lengths, dim=0),
            torch.split(value, lengths, dim=0),
        ):
            query_chunk = query_chunk.unsqueeze(0).transpose(1, 2)
            key_chunk = key_chunk.unsqueeze(0).transpose(1, 2)
            value_chunk = value_chunk.unsqueeze(0).transpose(1, 2)
            if self.num_kv_groups > 1:
                key_chunk = key_chunk.repeat_interleave(self.num_kv_groups, dim=1)
                value_chunk = value_chunk.repeat_interleave(self.num_kv_groups, dim=1)

            attention_mask = None
            if not full_attn:
                attention_mask = self._build_window_mask(query_chunk.shape[2], query_chunk.device, query_chunk.dtype)
            if self.sinks is not None:
                sink_bias = torch.zeros(
                    1,
                    self.num_heads,
                    query_chunk.shape[2],
                    key_chunk.shape[2],
                    device=query_chunk.device,
                    dtype=query_chunk.dtype,
                )
                sink_bias[..., 0] = self.sinks.view(1, self.num_heads, 1)
                attention_mask = sink_bias if attention_mask is None else attention_mask + sink_bias

            output = F.scaled_dot_product_attention(
                query_chunk,
                key_chunk,
                value_chunk,
                attn_mask=attention_mask,
                scale=self.scaling,
            )
            outputs.append(output.squeeze(0).transpose(0, 1))

        return self.proj(torch.cat(outputs, dim=0).reshape(seq_len, -1))


class MiMoVisionBlock(nn.Module):
    """Pre-normalized MiMo vision transformer block."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_act: str,
        rms_norm_eps: float,
        *,
        use_sinks: bool,
        window_size: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=rms_norm_eps, dtype=dtype)
        self.norm2 = nn.RMSNorm(dim, eps=rms_norm_eps, dtype=dtype)
        self.attn = MiMoVisionAttention(
            dim,
            num_heads,
            num_kv_heads,
            head_dim,
            use_sinks=use_sinks,
            window_size=window_size,
            dtype=dtype,
        )
        self.mlp = MiMoVisionSwiGLUMLP(dim, intermediate_dim, hidden_act, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        full_attn: bool = False,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens,
            position_embeddings,
            full_attn=full_attn,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiMoVisionPatchMerger(nn.Module):
    """Merge each spatial 2x2 group and project it to the text width."""

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.hidden_size = context_dim * spatial_merge_size**2
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6, bias=False, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim, bias=False, dtype=dtype),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(hidden_states).view(-1, self.hidden_size))


class MiMoVisionTransformer(nn.Module):
    """MiMo-V2.6 vision encoder with checkpoint-compatible parameter names."""

    def __init__(self, config: dict[str, Any] | Any, *, dtype: torch.dtype) -> None:
        super().__init__()
        config = SimpleNamespace(**config) if isinstance(config, dict) else config
        self.config = config
        hidden_size = int(config.hidden_size)
        depth = int(config.depth)
        num_heads = int(config.num_heads)
        num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
        head_dim = int(getattr(config, "qk_channels", 64))
        spatial_merge_size = int(getattr(config, "spatial_merge_size", 2))
        self.fullatt_block_indexes = set(getattr(config, "fullatt_block_indexes", []))
        self.vit_window_attn_types = list(getattr(config, "vit_window_attn_types", None) or [-1] * depth)
        if len(self.vit_window_attn_types) != depth:
            raise ValueError(f"vit_window_attn_types has {len(self.vit_window_attn_types)} entries for depth={depth}")

        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit = spatial_merge_size**2
        self.patch_embed = MiMoVisionPatchEmbed(
            patch_size=int(config.patch_size),
            temporal_patch_size=int(config.temporal_patch_size),
            in_channels=int(getattr(config, "in_channels", None) or getattr(config, "in_chans", 3)),
            embed_dim=hidden_size,
            dtype=dtype,
        )
        self.rotary_pos_emb = MiMoVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                MiMoVisionBlock(
                    hidden_size,
                    int(config.intermediate_size),
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    config.hidden_act,
                    float(getattr(config, "rms_norm_eps", 1e-6)),
                    use_sinks=bool(getattr(config, "use_sink", False)) and i not in self.fullatt_block_indexes,
                    window_size=int(getattr(config, "visual_token_window_size", -1)),
                    dtype=dtype,
                )
                for i in range(depth)
            ]
        )
        self.merger = MiMoVisionPatchMerger(
            int(config.out_hidden_size),
            hidden_size,
            spatial_merge_size,
            dtype,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize a materialized vision tower for scratch training."""
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for block in self.blocks:
            if block.attn.sinks is not None:
                nn.init.zeros_(block.attn.sinks)

    def _apply_merge_index(self, tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unflatten(0, (-1, self.spatial_merge_unit))
        return tensor[index].flatten(0, 1)

    def _get_window_index(self, grid_thw: torch.Tensor, *, column_major: bool) -> torch.Tensor:
        indices = []
        offset = 0
        for grid_t, grid_h, grid_w in grid_thw.tolist():
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            if column_major:
                index = index.transpose(1, 2)
            indices.append(index.reshape(-1) + offset)
            offset += grid_t * llm_grid_h * llm_grid_w
        return torch.cat(indices)

    def _rotary_positions(self, grid_thw: torch.Tensor, device: torch.device) -> torch.Tensor:
        position_ids = []
        for grid_t, grid_h, grid_w in grid_thw.tolist():
            hpos = torch.arange(grid_h).unsqueeze(1).expand(-1, grid_w)
            hpos = hpos.reshape(
                grid_h // self.spatial_merge_size,
                self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3)
            wpos = torch.arange(grid_w).unsqueeze(0).expand(grid_h, -1)
            wpos = wpos.reshape(
                grid_h // self.spatial_merge_size,
                self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3)
            position_ids.append(torch.stack((hpos.flatten(), wpos.flatten()), dim=-1).repeat(grid_t, 1))
        position_ids = torch.cat(position_ids)
        max_grid_size = int(grid_thw[:, 1:].max().item())
        return self.rotary_pos_emb(max_grid_size, device=device)[position_ids.to(device)].flatten(1)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        if grid_thw is None:
            raise ValueError("grid_thw must be provided with pixel_values")
        hidden_states = self.patch_embed(pixel_values.to(device=self.patch_embed.proj.weight.device, dtype=self.dtype))

        rotary = self._rotary_positions(grid_thw.cpu(), hidden_states.device)
        rotary = torch.cat((rotary, rotary), dim=-1)
        row_embeddings = (rotary.cos(), rotary.sin())

        column_index = self._get_window_index(grid_thw.cpu(), column_major=True).to(hidden_states.device)
        reverse_column_index = torch.argsort(column_index)
        column_rotary = self._apply_merge_index(rotary, column_index)
        column_embeddings = (column_rotary.cos(), column_rotary.sin())

        lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = F.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0).to(hidden_states.device)

        for index, block in enumerate(self.blocks):
            window_type = self.vit_window_attn_types[index]
            if window_type == 1 and (index == 0 or self.vit_window_attn_types[index - 1] != 1):
                hidden_states = self._apply_merge_index(hidden_states, column_index)
            if index > 0 and window_type != 1 and self.vit_window_attn_types[index - 1] == 1:
                hidden_states = self._apply_merge_index(hidden_states, reverse_column_index)
            position_embeddings = column_embeddings if window_type == 1 else row_embeddings
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                position_embeddings,
                full_attn=index in self.fullatt_block_indexes,
            )

        return self.merger(hidden_states)
