# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NaViT vision tower for Muse Spark Open image and video inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

from nemo_automodel.components.models.muse_spark.config import MuseSpark1xOmniConfig
from nemo_automodel.components.models.muse_spark.layers import next_dtype


def _cumulative_lengths(lengths: list[int], device: torch.device) -> torch.Tensor:
    """Build int32 packed-sequence boundaries for a list of segment lengths."""
    output = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    if lengths:
        output[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return output


def _rope_frequencies(
    head_dim: int,
    grid_h: int,
    grid_w: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    """Return adjacent-pair 2D RoPE frequencies shaped ``[H*W,D/2,2]``."""
    if head_dim % 4:
        raise ValueError("Muse Spark Open NaViT head_dim must be divisible by four")
    half_dim = head_dim // 2
    exponents = torch.arange(0, half_dim, 2, dtype=torch.float32, device=device)
    inverse = theta ** (-exponents / half_dim)
    pos_h = torch.arange(1, grid_h + 1, dtype=torch.float32, device=device)
    pos_w = torch.arange(1, grid_w + 1, dtype=torch.float32, device=device)
    mesh_h, mesh_w = torch.meshgrid(pos_h, pos_w, indexing="ij")
    freq_h = torch.einsum("hw,d->hwd", mesh_h, inverse)
    freq_w = torch.einsum("hw,d->hwd", mesh_w, inverse)
    angles = torch.cat((freq_w, freq_h), dim=-1)
    return torch.stack((angles.cos(), angles.sin()), dim=-1).reshape(grid_h * grid_w, head_dim // 2, 2)


def _apply_vision_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply adjacent-pair RoPE to Q/K tensors shaped ``[tokens,heads,head_dim]``."""
    cosine = frequencies[..., 0].unsqueeze(1)
    sine = frequencies[..., 1].unsqueeze(1)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        pairs = value.float().reshape(*value.shape[:-1], -1, 2)
        first, second = pairs.unbind(dim=-1)
        output = torch.stack(
            (first * cosine - second * sine, first * sine + second * cosine),
            dim=-1,
        ).flatten(-2)
        return output.to(value.dtype)

    return rotate(query), rotate(key)


def _pixel_shuffle_order(
    offset: int,
    grid_h: int,
    grid_w: int,
    factor: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Return block-major indices for one row-major ``[H,W]`` patch grid."""
    if grid_h % factor or grid_w % factor:
        raise ValueError("Muse Spark Open image grid must be divisible by the downsample factor")
    order = torch.arange(offset, offset + grid_h * grid_w, device=device)
    order = order.reshape(grid_h // factor, factor, grid_w // factor, factor).permute(0, 2, 1, 3).reshape(-1)
    return order, offset + grid_h * grid_w


def _sparse_order(
    offset: int,
    grid_h: int,
    grid_w: int,
    block_h: int,
    block_w: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], int]:
    """Pack one patch grid into fixed spatial blocks for sparse vision layers."""
    padded_h = math.ceil(grid_h / block_h) * block_h
    padded_w = math.ceil(grid_w / block_w) * block_w
    order = torch.arange(offset, offset + grid_h * grid_w, device=device).reshape(grid_h, grid_w)
    order = F.pad(order, (0, padded_w - grid_w, 0, padded_h - grid_h), value=-1)
    rows, columns = padded_h // block_h, padded_w // block_w
    order = order.reshape(rows, block_h, columns, block_w).permute(0, 2, 1, 3)
    order = order.reshape(rows * columns, block_h * block_w)
    lengths = (order != -1).sum(dim=1).tolist()
    return order[order != -1], lengths, offset + grid_h * grid_w


@dataclass(frozen=True)
class MuseSparkVisionOutput:
    """Flattened visual features after pixel shuffle and the adapter."""

    last_hidden_state: torch.Tensor


class MuseSparkVisionAttention(nn.Module):
    """Packed bidirectional attention over independent image/video-frame grids."""

    def __init__(self, hidden_size: int, num_heads: int, *, use_rope: bool, dtype: torch.dtype) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("Muse Spark Open vision hidden size must be divisible by its heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.use_rope = use_rope
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        frequencies: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Attend over ``[patches,hidden]`` using int32 segment boundaries."""
        length = hidden_states.shape[0]
        query = self.q_proj(hidden_states).reshape(length, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).reshape(length, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).reshape(length, self.num_heads, self.head_dim)
        if self.use_rope:
            query, key = _apply_vision_rope(query, key, frequencies)
        boundaries = cu_seqlens.cpu().tolist()
        outputs = []
        for start, end in zip(boundaries, boundaries[1:]):
            q = query[start:end].transpose(0, 1).unsqueeze(0)
            k = key[start:end].transpose(0, 1).unsqueeze(0)
            v = value[start:end].transpose(0, 1).unsqueeze(0)
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
            outputs.append(attended.squeeze(0).transpose(0, 1))
        if not outputs:
            return hidden_states.new_empty((0, self.num_heads * self.head_dim))
        return self.out_proj(torch.cat(outputs).reshape(length, -1))


class MuseSparkVisionMLP(nn.Module):
    """Bias-bearing GELU feed-forward used by NaViT."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size, bias=True, dtype=dtype)
        self.c_proj = nn.Linear(intermediate_size, hidden_size, bias=True, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform a ``[patches,hidden]`` visual sequence."""
        return self.c_proj(F.gelu(self.c_fc(hidden_states)))


class MuseSparkVisionBlock(GradientCheckpointingLayer):
    """One pre-LayerNorm NaViT encoder block."""

    def __init__(self, config: dict[str, Any], dtype: torch.dtype) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.ln_1 = nn.LayerNorm(hidden_size, eps=float(config.get("norm_eps", 1e-5)), dtype=dtype)
        self.attn = MuseSparkVisionAttention(
            hidden_size,
            int(config["num_attention_heads"]),
            use_rope=bool(config.get("use_rope", True)),
            dtype=dtype,
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=float(config.get("norm_eps", 1e-5)), dtype=dtype)
        self.mlp = MuseSparkVisionMLP(hidden_size, int(config["intermediate_size"]), dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        frequencies: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Transform packed ``[patches,hidden]`` visual states."""
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states), frequencies, cu_seqlens)
        return hidden_states + self.mlp(self.ln_2(hidden_states))


class MuseSparkVisionEncoder(nn.Module):
    """NaViT layer stack alternating sparse and full bidirectional attention."""

    def __init__(self, config: dict[str, Any], dtype: torch.dtype) -> None:
        super().__init__()
        self.sparse_attention_factor = int(config.get("sparse_attention_factor", 4))
        self.layer_types = config.get("layer_types")
        self.layers = nn.ModuleList(
            [MuseSparkVisionBlock(config, dtype) for _ in range(int(config["num_hidden_layers"]))]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        frequencies: torch.Tensor,
        full_cu_seqlens: torch.Tensor,
        sparse_cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Run packed ``[patches,hidden]`` states through the NaViT stack."""
        for layer_idx, layer in enumerate(self.layers):
            full_attention = (
                self.sparse_attention_factor <= 1
                or layer_idx == len(self.layers) - 1
                or (layer_idx + 1) % self.sparse_attention_factor == 0
            )
            if self.layer_types is not None:
                full_attention = self.layer_types[layer_idx] == "full_attention"
            hidden_states = layer(
                hidden_states,
                frequencies,
                full_cu_seqlens if full_attention else sparse_cu_seqlens,
            )
        return hidden_states


class MuseSparkVisionAdapter(nn.Module):
    """Two-layer GELU adapter from pixel-shuffled patches to projector width."""

    def __init__(self, input_size: int, output_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.c_fc = nn.Linear(input_size, output_size, bias=False, dtype=dtype)
        self.c_proj = nn.Linear(output_size, output_size, bias=False, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map ``[visual_tokens,input_size]`` to ``[visual_tokens,output_size]``."""
        return F.gelu(self.c_proj(F.gelu(self.c_fc(hidden_states))))


class MuseSparkVisionModel(nn.Module):
    """Variable-resolution Muse Spark Open image/video encoder."""

    def __init__(self, config: MuseSpark1xOmniConfig) -> None:
        super().__init__()
        navit = config.vision_navit_config
        dtype = next_dtype(config)
        self.config = navit
        self.patch_size = int(navit.get("patch_size", 14))
        self.patch_temporal = int(navit.get("patch_temporal", 2))
        self.downsample_factor = int(navit.get("downsample_factor", 2))
        self.sparse_attention_factor = int(navit.get("sparse_attention_factor", 4))
        self.grid_sample_mode = str(navit.get("grid_sample_mode", "bilinear"))
        hidden_size = int(navit.get("hidden_size", 1536))
        vision_output_dim = int(navit.get("vision_output_dim", hidden_size * self.downsample_factor**2))
        if vision_output_dim != hidden_size * self.downsample_factor**2:
            raise ValueError("Muse Spark Open vision_output_dim must match pixel-shuffled hidden size")
        image_size = int(navit.get("image_size", 448))
        self.position_grid_h = int(navit.get("position_grid_h", image_size // self.patch_size))
        self.position_grid_w = int(navit.get("position_grid_w", image_size // self.patch_size))
        self.head_dim = hidden_size // int(navit.get("num_attention_heads", 16))
        patch_width = int(navit.get("num_channels", 3)) * self.patch_temporal * self.patch_size**2
        self.conv1_linear = nn.Linear(patch_width, hidden_size, bias=False, dtype=dtype)
        self.positional_embedding_vlm = nn.Parameter(
            torch.empty(self.position_grid_h * self.position_grid_w, hidden_size, dtype=dtype)
        )
        self.ln_pre = nn.LayerNorm(hidden_size, eps=float(navit.get("norm_eps", 1e-5)), dtype=dtype)
        self.encoder = MuseSparkVisionEncoder(navit, dtype)
        self.ln_post = nn.LayerNorm(hidden_size, eps=float(navit.get("norm_eps", 1e-5)), dtype=dtype)
        adapter_output = int(navit.get("projector_input_dim", vision_output_dim))
        self.adapter = MuseSparkVisionAdapter(vision_output_dim, adapter_output, dtype)
        self.output_size = adapter_output

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> MuseSparkVisionOutput:
        """Encode row-major patches.

        Args:
            pixel_values: Flattened temporal patches with shape
                ``[sum(t*h*w), 3*patch_temporal*patch_size**2]``.
            grid_thw: Per-media clip grids with shape ``[media,3]``. Each row is
                ``[temporal_clips,grid_height,grid_width]``.

        Returns:
            Adapted visual tokens with shape
            ``[sum(t*h*w/downsample_factor**2), output_size]``.
        """
        if pixel_values.ndim != 2 or grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError("Muse Spark Open vision expects flattened patches and [media,3] grid_thw")
        device = pixel_values.device
        positions: list[torch.Tensor] = []
        frequencies: list[torch.Tensor] = []
        full_lengths: list[int] = []
        sparse_lengths: list[int] = []
        downsample_orders: list[torch.Tensor] = []
        sparse_orders: list[torch.Tensor] = []
        offset = 0
        sparse_offset = 0
        for temporal, grid_h, grid_w in grid_thw.tolist():
            temporal, grid_h, grid_w = int(temporal), int(grid_h), int(grid_w)
            pos_h = torch.linspace(-1 + 1 / grid_h, 1 - 1 / grid_h, grid_h, device=device)
            pos_w = torch.linspace(-1 + 1 / grid_w, 1 - 1 / grid_w, grid_w, device=device)
            frame_positions = torch.stack(torch.meshgrid(pos_h, pos_w, indexing="xy"), dim=-1).reshape(-1, 2)
            frame_frequencies = _rope_frequencies(
                self.head_dim,
                grid_h,
                grid_w,
                float(self.config.get("rope_theta", 10000.0)),
                device,
            )
            for _ in range(temporal):
                positions.append(frame_positions)
                frequencies.append(frame_frequencies)
                order, offset = _pixel_shuffle_order(
                    offset,
                    grid_h,
                    grid_w,
                    self.downsample_factor,
                    device,
                )
                downsample_orders.append(order)
                full_lengths.append(grid_h * grid_w)
                if self.sparse_attention_factor > 1:
                    order, lengths, sparse_offset = _sparse_order(
                        sparse_offset,
                        grid_h,
                        grid_w,
                        self.position_grid_h,
                        self.position_grid_w,
                        device,
                    )
                    sparse_orders.append(order)
                    sparse_lengths.extend(lengths)
                else:
                    sparse_lengths.append(grid_h * grid_w)

        expected_patches = sum(full_lengths)
        if pixel_values.shape[0] != expected_patches:
            raise ValueError(
                f"Muse Spark Open received {pixel_values.shape[0]} patches but grid_thw describes {expected_patches}"
            )
        hidden_states = self.conv1_linear(pixel_values.to(self.conv1_linear.weight.dtype))
        position_grid = (
            self.positional_embedding_vlm.reshape(self.position_grid_h, self.position_grid_w, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        position_stream = torch.cat(positions).to(hidden_states.dtype)
        interpolated = F.grid_sample(
            position_grid,
            position_stream[None, None],
            mode=self.grid_sample_mode,
            align_corners=False,
        )[0, :, 0].transpose(0, 1)
        hidden_states = self.ln_pre(hidden_states + interpolated.to(hidden_states.dtype))
        frequency_stream = torch.cat(frequencies)
        full_cu_seqlens = _cumulative_lengths(full_lengths, device)
        if self.sparse_attention_factor > 1:
            sparse_order = torch.cat(sparse_orders)
            hidden_states = hidden_states[sparse_order]
            frequency_stream = frequency_stream[sparse_order]
            sparse_cu_seqlens = _cumulative_lengths(sparse_lengths, device)
        else:
            sparse_order = None
            sparse_cu_seqlens = full_cu_seqlens
        hidden_states = self.encoder(
            hidden_states,
            frequency_stream,
            full_cu_seqlens,
            sparse_cu_seqlens,
        )
        if sparse_order is not None:
            restored = torch.empty_like(hidden_states)
            restored.index_copy_(0, sparse_order, hidden_states)
            hidden_states = restored
        hidden_states = self.ln_post(hidden_states)
        hidden_states = hidden_states[torch.cat(downsample_orders)]
        factor_squared = self.downsample_factor**2
        hidden_states = hidden_states.reshape(-1, factor_squared, hidden_states.shape[-1])
        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], -1)
        return MuseSparkVisionOutput(last_hidden_state=self.adapter(hidden_states))

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize a checkpoint-free vision model on ``buffer_device``."""
        with buffer_device:
            nn.init.normal_(self.positional_embedding_vlm, std=int(self.config["hidden_size"]) ** -0.5)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    module.reset_parameters()


__all__ = ["MuseSparkVisionModel", "MuseSparkVisionOutput"]
