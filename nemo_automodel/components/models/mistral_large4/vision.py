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

"""Native-layout Pixtral correctness path; no checkpoint matrix permutation.

The released tower uses FP32 interleaved axial rotation and BF16 source weights.
Packing images once per forward keeps FSDP collective counts independent of the
number of images on a rank. Attention remains isolated between images.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.common.utils import Float32RMSNorm


@dataclass(frozen=True)
class MistralLarge4VisionConfig:
    """Typed native vision settings, parsed once at the checkpoint boundary."""

    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 14
    image_size: int = 1540
    rope_theta: float = 10000.0
    spatial_merge_size: int = 2
    adapter_bias: bool = False
    image_token_id: int = 10
    image_break_token_id: int = 12
    image_end_token_id: int = 13

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_channels,
            self.patch_size,
            self.image_size,
            self.spatial_merge_size,
        )
        if min(dimensions) <= 0 or self.hidden_size % self.num_attention_heads:
            raise ValueError("Pixtral dimensions must be positive and heads must divide hidden_size")
        if (self.hidden_size // self.num_attention_heads) % 4 or self.rope_theta <= 0:
            raise ValueError("Pixtral axial RoPE requires head_dim divisible by four and positive rope_theta")
        if self.adapter_bias:
            raise ValueError("Large 4 requires bias-free vision adapters")
        if self.image_size % (self.patch_size * self.spatial_merge_size):
            raise ValueError("image_size must be divisible by patch_size * spatial_merge_size")
        if (
            min(self.image_token_id, self.image_break_token_id, self.image_end_token_id) < 0
            or len({self.image_token_id, self.image_break_token_id, self.image_end_token_id}) != 3
        ):
            raise ValueError("Image, break and end token IDs must be distinct nonnegative integers")

    def build(self, *, dtype: torch.dtype) -> "MistralLarge4VisionEncoder":
        """Construct the native-layout encoder without changing declarative settings."""
        return MistralLarge4VisionEncoder(self, dtype=dtype)

    def grid(self, pixels: torch.Tensor) -> tuple[int, int]:
        """Validate a complete preprocessed image and return its merged grid.

        Args:
            pixels: Floating tensor [channels, height, width], before patch convolution.
                Height/width must be positive multiples of patch_size * merge_size.

        Returns:
            Merged grid height and width in language image tokens.
        """
        unit = self.patch_size * self.spatial_merge_size
        if pixels.ndim != 3 or pixels.shape[0] != self.num_channels or not pixels.is_floating_point():
            raise ValueError("pixel_values entries must be floating [channels, height, width] images")
        height, width = pixels.shape[-2:]
        if min(height, width) <= 0 or max(height, width) > self.image_size or height % unit or width % unit:
            raise ValueError("Image dimensions must fit image_size and contain complete merged patches")
        return height // unit, width // unit


def _rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply axial rotation without rounding trigonometric tables to BF16.

    Args:
        x: Floating [tokens, heads, head_dim], adjacent real/imaginary channel pairs.
        angles: FP32 [tokens, head_dim / 2], height frequencies followed by width frequencies.

    Returns:
        Independent rotated tensor with x's shape and dtype.
    """
    pairs = x.float().unflatten(-1, (-1, 2))
    real, imag = pairs.unbind(-1)
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]
    return torch.stack((real * cos - imag * sin, imag * cos + real * sin), -1).flatten(-2).to(x.dtype)


class _VisionAttention(nn.Module):
    def __init__(self, config: MistralLarge4VisionConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.heads
        self.wq = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype)
        self.wk = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype)
        self.wv = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, angles: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """Attend bidirectionally within each image.

        Args:
            x: Floating packed [tokens, hidden] image features.
            angles: FP32 [tokens, head_dim / 2] axial angles.
            keep: Boolean [tokens, tokens] image membership mask, true permits attention.

        Returns:
            Independent projected features [tokens, hidden].
        """
        q = _rotate(self.wq(x).view(-1, self.heads, self.head_dim), angles)
        k = _rotate(self.wk(x).view(-1, self.heads, self.head_dim), angles)
        v = self.wv(x).view(-1, self.heads, self.head_dim)
        out = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), keep)
        return self.wo(out.transpose(0, 1).reshape(x.shape))


class _VisionMLP(nn.Module):
    def __init__(self, config: MistralLarge4VisionConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=dtype)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the native SwiGLU projections.

        Args:
            x: Floating [tokens, hidden] features.

        Returns:
            Independent features [tokens, hidden].
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MistralLarge4VisionBlock(nn.Module):
    """A single native Pixtral block and FSDP/recomputation boundary."""

    def __init__(self, config: MistralLarge4VisionConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.attention_norm = Float32RMSNorm(config.hidden_size, eps=1e-5, dtype=dtype)
        self.attention = _VisionAttention(config, dtype=dtype)
        self.ffn_norm = Float32RMSNorm(config.hidden_size, eps=1e-5, dtype=dtype)
        self.feed_forward = _VisionMLP(config, dtype=dtype)

    def forward(self, x: torch.Tensor, angles: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """Apply attention and SwiGLU residuals without mutating inputs.

        Args:
            x: Floating packed [tokens, hidden] image features.
            angles: FP32 [tokens, head_dim / 2] axial angles.
            keep: Boolean [tokens, tokens] image membership mask.

        Returns:
            Independent features [tokens, hidden].
        """
        x = x + self.attention(self.attention_norm(x), angles, keep)
        return x + self.feed_forward(self.ffn_norm(x))


class MistralLarge4VisionEncoder(nn.Module):
    """Pack variable images into one set of per-layer calls with native weight names."""

    def __init__(self, config: MistralLarge4VisionConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.patch_conv = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
            bias=False,
            dtype=dtype,
        )
        self.ln_pre = Float32RMSNorm(config.hidden_size, eps=1e-5, dtype=dtype)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList(
            [MistralLarge4VisionBlock(config, dtype=dtype) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, pixels: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Encode complete images; positions reset for each image.

        Args:
            pixels: Nonempty list of floating [channels, height, width] tensors,
                on the model device. Inputs are neither resized nor modified.

        Returns:
            One independent [patch_rows * patch_columns, vision_hidden] tensor per
            image, in encounter order and row-major patch order.
        """
        if not pixels:
            raise ValueError("Vision encoder requires at least one image")
        for image in pixels:
            self.config.grid(image)
        height, width = max(p.shape[1] for p in pixels), max(p.shape[2] for p in pixels)
        batch = torch.stack([F.pad(p, (0, width - p.shape[2], 0, height - p.shape[1])) for p in pixels])
        patches = self.patch_conv(batch.to(self.patch_conv.weight.dtype))
        sizes = [(p.shape[1] // self.config.patch_size, p.shape[2] // self.config.patch_size) for p in pixels]
        hidden = self.ln_pre(torch.cat([p[:, :h, :w].flatten(1).T for p, (h, w) in zip(patches, sizes)]))
        dim = self.config.hidden_size // self.config.num_attention_heads
        freq = self.config.rope_theta ** (-torch.arange(0, dim, 2, device=hidden.device, dtype=torch.float32) / dim)
        angles, ids = [], []
        for i, (h, w) in enumerate(sizes):
            rows, cols = torch.meshgrid(
                torch.arange(h, device=hidden.device), torch.arange(w, device=hidden.device), indexing="ij"
            )
            angles.append(torch.cat((rows.reshape(-1, 1) * freq[::2], cols.reshape(-1, 1) * freq[1::2]), -1))
            ids.append(torch.full((h * w,), i, device=hidden.device))
        angle, image_ids = torch.cat(angles), torch.cat(ids)
        keep = image_ids[:, None] == image_ids[None, :]
        for layer in self.transformer.layers:
            hidden = layer(hidden, angle, keep)
        return hidden.split([h * w for h, w in sizes])


class MistralLarge4PatchMerger(nn.Module):
    """Channel-major spatial patch merging, before the GELU projector."""

    def __init__(self, config: MistralLarge4VisionConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.merge = config.spatial_merge_size
        self.merging_layer = nn.Linear(config.hidden_size * self.merge**2, config.hidden_size, bias=False, dtype=dtype)

    def forward(self, features: tuple[torch.Tensor, ...], sizes: list[tuple[int, int]]) -> torch.Tensor:
        """Merge nonoverlapping complete patch groups.

        Args:
            features: Per-image floating [patch_rows * patch_columns, vision_hidden] tensors.
            sizes: Per-image patch grid (rows, columns), divisible by merge size.

        Returns:
            Independent [total_merged_patches, vision_hidden] tensor, in image order.
        """
        groups = []
        for x, (h, w) in zip(features, sizes, strict=True):
            # Last axes before flattening are channel, row-within-group, column-within-group.
            grouped = x.view(h // self.merge, self.merge, w // self.merge, self.merge, -1)
            groups.append(grouped.permute(0, 2, 4, 1, 3).reshape(-1, x.shape[-1] * self.merge**2))
        return self.merging_layer(torch.cat(groups))


class MistralLarge4VisionAdapter(nn.Module):
    """Bias-free vision-to-language GELU projection with released parameter names."""

    def __init__(self, config: MistralLarge4VisionConfig, text_hidden_size: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.w_in = nn.Linear(config.hidden_size, text_hidden_size, bias=False, dtype=dtype)
        self.w_out = nn.Linear(text_hidden_size, text_hidden_size, bias=False, dtype=dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project merged image features.

        Args:
            features: Floating [merged_patches, vision_hidden] features.

        Returns:
            Independent floating [merged_patches, text_hidden] features.
        """
        return self.w_out(F.gelu(self.w_in(features)))
