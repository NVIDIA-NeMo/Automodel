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

"""DeepSeek vision encoder and spatial merger with independent batched images."""

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling

from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.vision import (
    DeepseekV4VisionAligner,
    DeepseekV4VisionTransformer,
    get_vision_cos_sin,
)

from .config import DeepseekV41Config


def _vision_config(config: DeepseekV41Config) -> DeepseekV4Config:
    """Translate the typed V4.1 config into the existing vision implementation's config."""
    vision = config.vision_config
    return DeepseekV4Config(
        hidden_size=config.text_config.hidden_size,
        dtype=config.dtype,
        vision_n_layers=vision.num_hidden_layers,
        vision_dim=vision.hidden_size,
        vision_n_heads=vision.num_attention_heads,
        vision_inter_dim=vision.intermediate_size,
        vision_patch_size=vision.patch_size,
        vision_rope_theta=vision.rope_theta,
        vision_downsample_ratio=vision.downsample_ratio,
        vision_max_n_token=vision.max_image_tokens,
        vision_min_pixels=vision.min_pixels,
        vision_max_wh_ratio=vision.max_wh_ratio,
    )


class DeepseekV41VisionTransformer(DeepseekV4VisionTransformer):
    """The released full-attention 2D-RoPE ViT, configured by nested V4.1 fields.

    Args:
        config: Top-level V4.1 configuration containing the vision dimensions.
    """

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__(_vision_config(config))
        self.config = config.vision_config

    # The shared frame sharder gathers unmerged patch features. Spatial merging
    # happens afterwards, so odd image grids retain their reference ceil semantics.
    spatial_merge_size: int = 1

    def forward(
        self,
        patches: torch.Tensor,
        n_h: int | None = None,
        n_w: int | None = None,
        *,
        grid_thw: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | BaseModelOutputWithPooling:
        """Encode independent images with one call per FSDP/checkpointed block.

        Args:
            patches: Normalized RGB patches [patches, 3, patch_size, patch_size]
                or flattened [patches, 3 * patch_size**2], in image order.
            n_h: Patch rows for the backward-compatible single-image call.
            n_w: Patch columns for the backward-compatible single-image call.
            grid_thw: Integer grids [images, 3], each (frames, height, width).
                Frames are independent images, including frame-sharder dummies.
            return_dict: Return the pooler-output contract used by frame sharding.

        Returns:
            Unmerged features [patches, vision_dim], or a ModelOutput containing
            that tensor as pooler_output. No attention crosses image boundaries.
        """
        if grid_thw is None:
            if n_h is None or n_w is None:
                raise ValueError("Provide grid_thw or both n_h and n_w")
            grids = [(n_h, n_w)]
        else:
            if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
                raise ValueError("grid_thw must have shape [images, 3]")
            rows = grid_thw.detach().cpu().tolist()
            if any(t <= 0 or h <= 0 or w <= 0 for t, h, w in rows):
                raise ValueError("Image grids must have positive dimensions")
            grids = [(h, w) for t, h, w in rows for _ in range(t)]
        lengths = tuple(h * w for h, w in grids)
        if not lengths or sum(lengths) != patches.shape[0]:
            raise ValueError("Image grids must account for every pixel patch")
        x = self.patch_embed(patches)
        rotary = [get_vision_cos_sin(h, w, self.rope_dim, self.rope_theta, device=x.device) for h, w in grids]
        cos = torch.cat([pair[0] for pair in rotary])
        sin = torch.cat([pair[1] for pair in rotary])
        for block in self.blocks:
            x = block(x, cos, sin, lengths)
        x = self.norm(x)
        return BaseModelOutputWithPooling(pooler_output=x) if return_dict else x


class DeepseekV41VisionAligner(DeepseekV4VisionAligner):
    """The released padded spatial merger and GELU projection into text width.

    Args:
        config: Top-level V4.1 configuration containing vision and text widths.
    """

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__(_vision_config(config))

    def forward(
        self,
        x: torch.Tensor,
        n_h: int | None = None,
        n_w: int | None = None,
        *,
        image_grid_hws: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Merge each image independently, then batch the projection GEMMs.

        Args:
            x: Concatenated unmerged image features [patches, vision_dim].
            n_h: Patch rows for a single-image call.
            n_w: Patch columns for a single-image call.
            image_grid_hws: Integer patch grids [images, 2] in feature order.

        Returns:
            Features [sum(ceil(height / ratio) * ceil(width / ratio)), hidden].
            Padding stays inside each image; no patches mix across images.
        """
        if image_grid_hws is None:
            if n_h is None or n_w is None:
                raise ValueError("Provide image_grid_hws or both n_h and n_w")
            return super().forward(x, n_h, n_w)
        if image_grid_hws.ndim != 2 or image_grid_hws.shape[1] != 2:
            raise ValueError("image_grid_hws must have shape [images, 2]")
        grids = image_grid_hws.detach().cpu().tolist()
        if not grids or any(h <= 0 or w <= 0 for h, w in grids) or sum(h * w for h, w in grids) != x.shape[0]:
            raise ValueError("Image grids must account for every encoded patch")
        merged = []
        ratio = self.downsample_ratio
        for features, (h, w) in zip(x.split([h * w for h, w in grids]), grids, strict=True):
            features = features.view(h, w, -1).permute(2, 0, 1)
            features = F.pad(features, (0, -w % ratio, 0, -h % ratio))
            merged.append(F.unfold(features.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1))
        return self.w2(F.gelu(self.w1(torch.cat(merged))))
