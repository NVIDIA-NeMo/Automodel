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

"""Native Inkling2 image encoder with independent per-image SDPA attention.

The tensor contracts follow the released Inkling-2-Small checkpoint configuration.
This implementation uses ordinary Torch modules and SDPA, preserving the released
weight names for bounded checkpoint loading.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from .config import Inkling2VisionConfig


class Inkling2VisionEmbeddings(nn.Module):
    """Native patch projection, interpolated positions and a learned class token."""

    def __init__(self, config: Inkling2VisionConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.grid_size = config.posemb_grid_size
        width = config.hidden_size
        self.patch_embedding = nn.Linear(3 * config.native_patch_size**2, width, bias=False, dtype=dtype)
        self.position_embedding = nn.Parameter(torch.empty(config.posemb_grid_size**2, width, dtype=dtype))
        self.class_embedding = nn.Parameter(torch.empty(width, dtype=dtype))
        self.cls_position_embedding = nn.Parameter(torch.empty(width, dtype=dtype))

    def forward(self, pixels: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Embed one image's native patches and add interpolated positions.

        Args:
            pixels: Tensor of shape [height * width, native_patch_size ** 2 * 3]. Rows follow
                the native patch grid in row-major order; each row flattens patch height,
                patch width and RGB channels, in that order. Its device and dtype must match
                the patch projection.
            height: Native patch rows of the image.
            width: Native patch columns of the image.

        Returns:
            New [1 + height * width, hidden] tokens in the projection dtype, with the class
            token first; the input is unchanged.
        """
        grid = self.position_embedding.reshape(self.grid_size, self.grid_size, -1)
        axes = []
        for size in (height, width):
            centers = (torch.arange(size, device=pixels.device, dtype=torch.float32) + 0.5) * self.grid_size / size
            samples = torch.arange(self.grid_size, device=pixels.device, dtype=torch.float32) + 0.5
            weights = (1 - (centers[:, None] - samples).abs()).clamp_min(0)
            axes.append(weights / weights.sum(-1, keepdim=True))
        positions = torch.einsum("hi,wj,ijd->hwd", axes[0], axes[1], grid.float()).flatten(0, 1).to(grid.dtype)
        hidden = self.patch_embedding(pixels) + positions
        return torch.cat(((self.class_embedding + self.cls_position_embedding)[None], hidden))


class Inkling2VisionAttention(nn.Module):
    """Full attention with separate horizontal and vertical rotary coordinates."""

    def __init__(self, config: Inkling2VisionConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, dtype=dtype)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Attend over one image, including CLS; never mix different images.

        Args:
            hidden: Compute-dtype [tokens, hidden] features.
            positions: Integer [tokens, 2] (y, x) coordinates; CLS uses zero.

        Returns:
            Projected features with the same shape and dtype as hidden.
        """
        shape = (hidden.shape[0], self.num_heads, self.head_dim)
        q, k, v = (projection(hidden).reshape(shape) for projection in (self.q_proj, self.k_proj, self.v_proj))
        frequency = 10000 ** (
            -torch.arange(0, self.head_dim, 4, device=hidden.device, dtype=torch.float32) / self.head_dim
        )
        angles = torch.cat((positions[:, 1, None] * frequency, positions[:, 0, None] * frequency), -1)
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]
        rotated = []
        for value in (q, k):
            even, odd = value.float().reshape(*shape[:-1], -1, 2).unbind(-1)
            rotated.append(
                torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2).to(value.dtype)
            )
        attended = (
            F.scaled_dot_product_attention(
                rotated[0].transpose(0, 1)[None],
                rotated[1].transpose(0, 1)[None],
                v.transpose(0, 1)[None],
                dropout_p=0.0,
                is_causal=False,
            )
            .squeeze(0)
            .transpose(0, 1)
            .reshape(hidden.shape)
        )
        return self.out_proj(attended)


class Inkling2VisionMLP(nn.Module):
    """The image encoder's ordinary GELU feed-forward block."""

    def __init__(self, config: Inkling2VisionConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, int(config.hidden_size * config.mlp_ratio), dtype=dtype)
        self.c_proj = nn.Linear(int(config.hidden_size * config.mlp_ratio), config.hidden_size, dtype=dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the GELU feed-forward block.

        Args:
            hidden: [tokens, hidden] features in the block's dtype.

        Returns:
            New [tokens, hidden] features; the input is unchanged.
        """
        return self.c_proj(F.gelu(self.c_fc(hidden)))


class Inkling2VisionLayerScale(nn.Module):
    """Learned channel scale on a vision residual branch."""

    def __init__(self, width: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width, dtype=dtype))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Scale a residual branch per channel.

        Args:
            hidden: [tokens, hidden] branch output.

        Returns:
            New [tokens, hidden] features, each channel multiplied by the learned [hidden]
            vector; the input is unchanged.
        """
        return hidden * self.gamma


class Inkling2VisionLayer(nn.Module):
    """Pre-LayerNorm attention and GELU branches with learned LayerScale."""

    def __init__(self, config: Inkling2VisionConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype)
        self.self_attn = Inkling2VisionAttention(config, dtype=dtype)
        self.mlp = Inkling2VisionMLP(config, dtype=dtype)
        self.ls_1 = Inkling2VisionLayerScale(config.hidden_size, dtype=dtype)
        self.ls_2 = Inkling2VisionLayerScale(config.hidden_size, dtype=dtype)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply one pre-LayerNorm attention and feed-forward block.

        Args:
            hidden: [tokens, hidden] features of one image, class token first.
            positions: Integer [tokens, 2] (y, x) patch coordinates; the class token uses zero.

        Returns:
            New [tokens, hidden] features; the inputs are unchanged.
        """
        hidden = hidden + self.ls_1(self.self_attn(self.ln_1(hidden), positions))
        return hidden + self.ls_2(self.mlp(self.ln_2(hidden)))


class Inkling2VisionBackbone(nn.Module):
    """Image transformer before spatial merging and text projection."""

    def __init__(self, config: Inkling2VisionConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.embeddings = Inkling2VisionEmbeddings(config, dtype=dtype)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype)
        self.layers = nn.ModuleList([Inkling2VisionLayer(config, dtype=dtype) for _ in range(config.num_hidden_layers)])

    def forward(self, pixels: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Encode one image's native patches.

        Args:
            pixels: [height * width, native_patch_size ** 2 * 3] rows in native row-major
                patch order, as consumed by the embeddings.
            height: Native patch rows of the image.
            width: Native patch columns of the image.

        Returns:
            New [height * width, hidden] features in the same row-major patch order, without
            the class token; the input is unchanged.
        """
        indices = torch.arange(height * width, device=pixels.device)
        spatial = torch.stack((indices // width + 1, indices % width + 1), -1)
        positions = torch.cat((spatial.new_zeros((1, 2)), spatial))
        hidden = self.ln_pre(self.embeddings(pixels, height=height, width=width))
        for layer in self.layers:
            hidden = layer(hidden, positions)
        return hidden[1:]


class Inkling2VisionEncoder(nn.Module):
    """Encode merged image patches into one text feature per image-body token."""

    _clip_mean: torch.Tensor
    _clip_std: torch.Tensor

    def __init__(self, config: Inkling2VisionConfig, text_hidden_size: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.image_cp_mesh: DeviceMesh | None = None
        self.config = config
        self.text_hidden_size = text_hidden_size
        self.backbone = Inkling2VisionBackbone(config, dtype=dtype)
        self.proj_in = nn.Linear(
            config.adapter_merge_size**2 * config.hidden_size,
            config.projection_hidden_dim,
            bias=config.projector_bias,
            dtype=dtype,
        )
        self.proj_out = nn.Linear(
            config.projection_hidden_dim, text_hidden_size, bias=config.projector_bias, dtype=dtype
        )
        self.register_buffer("_clip_mean", torch.empty(3, dtype=torch.float32), persistent=False)
        self.register_buffer("_clip_std", torch.empty(3, dtype=torch.float32), persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize all parameters and regenerate nonpersistent normalization buffers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()
            elif isinstance(module, Inkling2VisionLayerScale):
                module.gamma.fill_(1)
        embeddings = self.backbone.embeddings
        for parameter in (embeddings.position_embedding, embeddings.class_embedding, embeddings.cls_position_embedding):
            nn.init.normal_(parameter, std=0.02)
        self._clip_mean.copy_(self._clip_mean.new_tensor((0.48145466, 0.4578275, 0.40821073)))
        self._clip_std.copy_(self._clip_std.new_tensor((0.26862954, 0.26130258, 0.27577711)))

    def forward(self, vision_patches_bthwc: torch.Tensor, vision_grid_thw: torch.Tensor) -> torch.Tensor:
        """Project independent images into text embeddings.

        Args:
            vision_patches_bthwc: BF16 normalized patches [patches, temporal, side, side, 3],
                where side = native_patch_size * adapter_merge_size. Input is unchanged.
            vision_grid_thw: Integer [images, 3] grid metadata (1, rows, columns).
                Grids and patches follow the same image order; metadata may be on CPU or GPU.

        Returns:
            Compute-dtype [patches, text_hidden] features, in row-major merged-patch order.
        """
        cfg = self.config
        m, p = cfg.adapter_merge_size, cfg.native_patch_size
        if (
            vision_patches_bthwc.dtype != torch.bfloat16
            or vision_patches_bthwc.ndim != 5
            or tuple(vision_patches_bthwc.shape[1:]) != (cfg.temporal_patch_size, m * p, m * p, 3)
        ):
            raise ValueError("Inkling2 patches must be BF16 [patches, temporal, side, side, 3]")
        if (
            vision_grid_thw.ndim != 2
            or vision_grid_thw.shape[1] != 3
            or vision_grid_thw.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Inkling2 image grids must be integer [images, 3]")
        grid = vision_grid_thw.tolist()
        if (
            any(t != 1 or h <= 0 or w <= 0 for t, h, w in grid)
            or sum(h * w for _, h, w in grid) != vision_patches_bthwc.shape[0]
        ):
            raise ValueError("Inkling2 image grids and patch counts disagree")
        image_count = len(grid)
        if self.image_cp_mesh is not None:
            # Per-image backbone calls must agree across all FSDP DP/CP peers.
            # Dummy features retain zero-valued backward edges to every vision layer.
            count = torch.tensor(image_count, device=vision_patches_bthwc.device, dtype=torch.long)
            dist.all_reduce(count, op=dist.ReduceOp.MAX, group=self.image_cp_mesh.get_group())
            image_count = max(1, int(count))
        elif not grid:
            return self.proj_out.weight.new_empty((0, self.text_hidden_size))
        # Preserve the release's BF16 input renormalization before projection.
        pixels = (
            2
            * (
                vision_patches_bthwc * self._clip_std.to(vision_patches_bthwc.dtype)
                + self._clip_mean.to(vision_patches_bthwc.dtype)
            )
            - 1
        )
        pixels = pixels.mean(1).permute(0, 3, 1, 2).contiguous().to(self.proj_out.weight.dtype)
        pooled = []
        images = pixels.split([h * w for _, h, w in grid])
        for index in range(image_count):
            real = index < len(grid)
            if real:
                _, h, w = grid[index]
                image = images[index]
            else:
                h = w = 1
                image = pixels.new_zeros((1, 3, m * p, m * p))
            native = image.reshape(h, w, 3, m, p, m, p).permute(0, 3, 1, 5, 4, 6, 2).reshape(h * w * m * m, p * p * 3)
            hidden = self.backbone(native, height=h * m, width=w * m)
            merged = (
                hidden.reshape(h, m, w, m, cfg.hidden_size)
                .permute(0, 2, 1, 3, 4)
                .reshape(h * w, m * m * cfg.hidden_size)
            )
            pooled.append(merged if real else merged[:0])
        return self.proj_out(F.gelu(self.proj_in(torch.cat(pooled))))
