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

"""Hy4-flash vision tower: patch embedding, ViT blocks and the spatial-merge projector.

Semantics follow ``HYV4VLVisionModel`` of the transformers reference (PR 50903): a learned position grid
interpolated with an explicit ``scale_factor`` of ``(size + 0.1) / grid`` (this shifts the sampling grid relative to
``size=``), full bidirectional attention inside each image, an RMSNorm + 2x2 conv merge + 1x1 conv projector, and one
``image_newline`` embedding appended to every merged row. Module names follow the released ``vit.*`` keys, which
the state-dict adapter therefore leaves unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.models.hy_v4_flash.config import HY4FlashVisionConfig


class HY4FlashVisionEmbeddings(nn.Module):
    """Conv patch embedding plus an interpolated learned position grid."""

    def __init__(self, config: HY4FlashVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(config.num_channels, config.hidden_size, config.patch_size, config.patch_size)
        self.edge = config.max_image_size // config.patch_size
        # Row 0 is a never-used cls slot; rows 1.. form the ``edge x edge`` learned patch grid.
        self.position_embedding = nn.Embedding(self.edge**2 + 1, config.hidden_size)

    def interpolate_positions(self, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        """Learned grid resampled to ``height x width`` patches.

        Returns:
            Tensor of shape [height * width, hidden].
        """
        grid = self.position_embedding(torch.arange(1, self.edge**2 + 1, device=self.position_embedding.weight.device))
        grid = grid.reshape(1, self.edge, self.edge, -1).permute(0, 3, 1, 2).float()
        scale = ((height + 0.1) / self.edge, (width + 0.1) / self.edge)
        grid = F.interpolate(grid, scale_factor=scale, mode=self.config.interpolate_mode, align_corners=False)
        return grid.permute(0, 2, 3, 1).reshape(-1, grid.shape[1]).to(dtype)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Embed flat patches.

        Args:
            pixel_values: Tensor of shape [patches, channels * patch_size * patch_size], row-major per image.
            grid_thw: Integer tensor of shape [images, 3] of (time, height, width) patch counts.

        Returns:
            Tensor of shape [patches, hidden].
        """
        weight = self.patch_embedding.weight
        patches = pixel_values.to(weight.dtype).reshape(-1, self.config.num_channels, self.patch_size, self.patch_size)
        embeds = self.patch_embedding(patches).flatten(1)
        positions = [
            self.interpolate_positions(int(h), int(w), embeds.dtype).repeat(int(t), 1) for t, h, w in grid_thw.tolist()
        ]
        return embeds + torch.cat(positions)


class HY4FlashVisionAttention(nn.Module):
    """Bidirectional multi-head attention with biases."""

    def __init__(self, config: HY4FlashVisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Attend inside each image.

        Args:
            hidden_states: Tensor of shape [patches, hidden], images concatenated.
            lengths: Patch count of each image.

        Returns:
            Tensor of shape [patches, hidden].
        """
        q, k, v = (
            proj(hidden_states).view(-1, self.num_heads, self.head_dim)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        outputs = []
        for q_i, k_i, v_i in zip(q.split(lengths), k.split(lengths), v.split(lengths)):
            out = F.scaled_dot_product_attention(*(t.transpose(0, 1).unsqueeze(0) for t in (q_i, k_i, v_i)))
            outputs.append(out[0].transpose(0, 1).reshape(q_i.shape[0], -1))
        return self.o_proj(torch.cat(outputs))


class HY4FlashVisionMLP(nn.Module):
    """GELU MLP named like the release (``dense_h_to_4h`` / ``dense_4h_to_h``)."""

    def __init__(self, config: HY4FlashVisionConfig) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to a tensor of shape [..., hidden]."""
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(hidden_states)))


class HY4FlashVisionBlock(nn.Module):
    """Pre-norm ViT block."""

    def __init__(self, config: HY4FlashVisionConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HY4FlashVisionAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HY4FlashVisionMLP(config)

    def forward(self, hidden_states: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Run the block on a tensor of shape [patches, hidden]."""
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), lengths)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class HY4FlashPerceive(nn.Module):
    """Spatial-merge projector: RMSNorm, 2x2 conv, GELU, 1x1 conv, plus a newline embedding per merged row."""

    def __init__(self, config: HY4FlashVisionConfig) -> None:
        super().__init__()
        merge = config.spatial_merge_size
        self.merge = merge
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # ``proj`` keeps the released indices: 0 = merge conv, 1 = activation (no weights), 2 = output conv.
        self.proj = nn.ModuleList(
            [
                nn.Conv2d(config.hidden_size, config.hidden_size * merge**2, merge, merge),
                nn.GELU(),
                nn.Conv2d(config.hidden_size * merge**2, config.out_hidden_size, 1),
            ]
        )
        self.image_newline = nn.Parameter(torch.empty(config.out_hidden_size))

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Merge and project per image (frames are merged independently).

        Args:
            hidden_states: Tensor of shape [patches, hidden].
            grid_thw: Integer tensor of shape [images, 3].

        Returns:
            Tensor of shape [tokens, out_hidden]; each frame contributes ``(h/m) * (w/m + 1)`` row-major tokens.
        """
        outputs = []
        start = 0
        for t, h, w in grid_thw.tolist():
            count = t * h * w
            x = self.norm(hidden_states[start : start + count])
            start += count
            x = x.reshape(t, h, w, -1).permute(0, 3, 1, 2)
            for layer in self.proj:
                x = layer(x)
            frames, channels, rows, cols = x.shape
            newline = self.image_newline.to(x.dtype).reshape(1, channels, 1, 1).expand(frames, channels, rows, 1)
            x = torch.cat((x, newline), dim=-1).reshape(frames, channels, -1).permute(0, 2, 1)
            outputs.append(x.reshape(-1, channels))
        return torch.cat(outputs)


class HY4FlashVisionTower(nn.Module):
    """``vit``: embeddings, blocks and the perceive projector."""

    def __init__(self, config: HY4FlashVisionConfig) -> None:
        super().__init__()
        self.embeddings = HY4FlashVisionEmbeddings(config)
        self.layers = nn.ModuleList(HY4FlashVisionBlock(config) for _ in range(config.num_hidden_layers))
        self.perceive = HY4FlashPerceive(config)

    @torch.no_grad()
    def reset_parameters(self, std: float) -> None:
        """Initialize as the reference does: zero biases, unit norm weights, normal everything else."""
        for name, parameter in self.named_parameters():
            if name.endswith("bias"):
                nn.init.zeros_(parameter)
            elif parameter.ndim == 1 and "norm" in name:
                nn.init.ones_(parameter)
            else:
                nn.init.normal_(parameter, std=std)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode images into language-width tokens.

        Args:
            pixel_values: Tensor of shape [patches, channels * patch_size * patch_size].
            grid_thw: Integer tensor of shape [images, 3] of (time, height, width) patch counts.

        Returns:
            Tensor of shape [tokens, out_hidden] where images' tokens are concatenated.
        """
        # Attention is bidirectional inside each frame; frames of a video do not attend to each other.
        lengths = [h * w for t, h, w in grid_thw.tolist() for _ in range(t)]
        hidden_states = self.embeddings(pixel_values, grid_thw)
        for layer in self.layers:
            hidden_states = layer(hidden_states, lengths)
        return self.perceive(hidden_states, grid_thw)
