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

"""MiniMax M3 VL vision tower (CLIP-style, Conv3d patch embed + 3D RoPE).

Mirrors the canonical sglang reference ``sglang.srt.models.minimax_vl_common``:
a Conv3d patch embedding over pre-patchified pixel values, ``pre_layrnorm``, a
stack of bidirectional CLIP encoder layers with axis-split 3D RoPE, then a
2-layer GELU multimodal projector (vision -> text hidden) and a spatial
patch-merger (``spatial_merge_size**2`` tokens -> 1).

Vision weights are stored unquantized (head_dim is not MXFP8-aligned), and the
checkpoint keeps separate ``q/k/v/out_proj`` (no QKV fusion).
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """NEOX-style half rotation: ``cat([-x2, x1])`` (matches the duplicated cos/sin)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_vision_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply 3D RoPE to the first ``rope_dim`` channels of q/k ([S, H, D])."""
    rope_dim = cos.shape[-1]
    cos = cos[:, None, :]  # [S, 1, rope_dim]
    sin = sin[:, None, :]
    q_rot, q_pass = q[..., :rope_dim].float(), q[..., rope_dim:]
    k_rot, k_pass = k[..., :rope_dim].float(), k[..., rope_dim:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    q = torch.cat((q_rot.to(q_pass.dtype), q_pass), dim=-1)
    k = torch.cat((k_rot.to(k_pass.dtype), k_pass), dim=-1)
    return q, k


class MiniMaxM3VisionEmbeddings(nn.Module):
    """Conv3d patch embedding over pre-patchified pixel values ([N, C*T*P*P])."""

    def __init__(self, config: Any):
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.img_token_compression_config.get("temporal_patch_size", 2)
        self.patch_embedding = nn.Conv3d(
            in_channels=self.num_channels,
            out_channels=config.hidden_size,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.patch_embedding.weight.dtype)
        pixel_values = pixel_values.reshape(
            -1, self.num_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.patch_embedding(pixel_values).reshape(pixel_values.shape[0], -1)


class MiniMaxM3VisionAttention(nn.Module):
    """Bidirectional multi-head attention with separate q/k/v/out projections + 3D RoPE."""

    def __init__(self, config: Any):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None):
        seq = x.shape[0]
        q = self.q_proj(x).view(seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(seq, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(seq, self.num_heads, self.head_dim)
        q, k = _apply_vision_rope(q, k, cos, sin)
        # [S, H, D] -> [1, H, S, D] for SDPA.
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        out = out.squeeze(0).transpose(0, 1).reshape(seq, self.num_heads * self.head_dim)
        return self.out_proj(out)


class MiniMaxM3VisionEncoderLayer(nn.Module):
    """CLIP-style encoder block: pre-norm attention + pre-norm GELU MLP (fc1/fc2)."""

    def __init__(self, config: Any):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = MiniMaxM3VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.mlp.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.self_attn(self.layer_norm1(x), cos, sin, attn_mask)
        h = self.layer_norm2(x)
        x = x + self.mlp.fc2(self.act(self.mlp.fc1(h)))
        return x


class MiniMaxM3VisionTransformer(nn.Module):
    """Conv3d embeddings + pre_layrnorm + bidirectional CLIP encoder with 3D RoPE."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.img_token_compression_config.get("spatial_merge_size", 2)
        self.embeddings = MiniMaxM3VisionEmbeddings(config)
        # "layrnorm" typo matches the published checkpoint weight name.
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [MiniMaxM3VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        head_dim = config.hidden_size // config.num_attention_heads
        rope_dims = 2 * (head_dim // 2)
        axis_dim = int(2 * ((rope_dims // 3) // 2))
        self.axis_dim = axis_dim
        theta = config.rope_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_position_freqs(self, grid_thw: list[list[int]], device) -> torch.Tensor:
        """Per-token [seq, 3*axis_dim/2] frequencies (t/h/w), spatial-merge-aware."""
        m = self.spatial_merge_size
        all_freqs = []
        for grid_t, grid_h, grid_w in grid_thw:
            tokens_per_frame = grid_h * grid_w
            tpos = torch.arange(grid_t, device=device).unsqueeze(1).expand(-1, tokens_per_frame).flatten()

            hpos = torch.arange(grid_h, device=device).unsqueeze(1).expand(-1, grid_w)
            hpos = hpos.reshape(grid_h // m, m, grid_w // m, m).permute(0, 2, 1, 3)
            hpos = hpos.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).flatten()

            wpos = torch.arange(grid_w, device=device).unsqueeze(0).expand(grid_h, -1)
            wpos = wpos.reshape(grid_h // m, m, grid_w // m, m).permute(0, 2, 1, 3)
            wpos = wpos.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).flatten()

            seq_t = torch.arange(max(grid_t, 1), device=device, dtype=self.inv_freq.dtype)
            seq_hw = torch.arange(max(grid_h, grid_w), device=device, dtype=self.inv_freq.dtype)
            freqs_t = torch.outer(seq_t, self.inv_freq)
            freqs_hw = torch.outer(seq_hw, self.inv_freq)
            all_freqs.append(torch.cat([freqs_t[tpos], freqs_hw[hpos], freqs_hw[wpos]], dim=-1))
        return torch.cat(all_freqs, dim=0)

    @staticmethod
    def _block_diag_mask(grid_thw: list[list[int]], device) -> torch.Tensor:
        """Bidirectional within each image, no cross-image attention."""
        seqlens = [int(t * h * w) for t, h, w in grid_thw]
        total = sum(seqlens)
        mask = torch.zeros(1, 1, total, total, dtype=torch.bool, device=device)
        start = 0
        for s in seqlens:
            mask[..., start : start + s, start : start + s] = True
            start += s
        return mask

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        device = pixel_values.device
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)

        freqs = self._rope_position_freqs(grid_thw, device)  # [seq, 3*axis_dim/2]
        cos = freqs.cos().repeat(1, 2)  # NEOX duplicate -> [seq, rope_dim]
        sin = freqs.sin().repeat(1, 2)
        attn_mask = self._block_diag_mask(grid_thw, device) if len(grid_thw) > 1 else None

        for layer in self.encoder.layers:
            x = layer(x, cos, sin, attn_mask)
        return x


class MiniMaxVLMultiModalProjector(nn.Module):
    """2-layer GELU projector: vision_hidden -> projector_hidden -> text_hidden."""

    def __init__(self, vision_hidden: int, text_hidden: int, projector_hidden: int, act: str, bias: bool):
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden, projector_hidden, bias=bias)
        self.act = ACT2FN[act]
        self.linear_2 = nn.Linear(projector_hidden, text_hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class MiniMaxVLPatchMerger(nn.Module):
    """Merge ``spatial_merge_size**2`` projected tokens then GELU-MLP back to text_hidden."""

    def __init__(self, spatial_merge_size: int, text_hidden: int, projector_hidden: int, act: str, bias: bool):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.linear_1 = nn.Linear(text_hidden * spatial_merge_size**2, projector_hidden, bias=bias)
        self.act = ACT2FN[act]
        self.linear_2 = nn.Linear(projector_hidden, text_hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0] // (self.spatial_merge_size**2), -1)
        return self.linear_2(self.act(self.linear_1(x)))


class MiniMaxM3VisionModel(nn.Module):
    """Vision tower: ViT + multimodal projector + patch merger (returns text-dim image tokens)."""

    def __init__(
        self,
        config: Any,
        text_hidden_size: int,
        projector_hidden_size: int,
        *,
        projector_hidden_act: str = "gelu",
        multimodal_projector_bias: bool = True,
    ):
        super().__init__()
        self.config = config
        self.vision_model = MiniMaxM3VisionTransformer(config)
        # projector_hidden_act / multimodal_projector_bias are top-level VL config
        # fields (not on vision_config), so they are passed in by the wrapper.
        self.multi_modal_projector = MiniMaxVLMultiModalProjector(
            config.hidden_size, text_hidden_size, projector_hidden_size, projector_hidden_act, multimodal_projector_bias
        )
        spatial_merge_size = config.img_token_compression_config.get("spatial_merge_size", 2)
        self.patch_merge_mlp = MiniMaxVLPatchMerger(
            spatial_merge_size, text_hidden_size, projector_hidden_size, projector_hidden_act, multimodal_projector_bias
        )

    @property
    def dtype(self):
        return self.vision_model.embeddings.patch_embedding.weight.dtype

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        hidden_states = self.vision_model(pixel_values, grid_thw)
        hidden_states = self.multi_modal_projector(hidden_states)
        hidden_states = self.patch_merge_mlp(hidden_states)
        return hidden_states
