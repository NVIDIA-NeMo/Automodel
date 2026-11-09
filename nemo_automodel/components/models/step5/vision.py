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

"""Step perception encoder with shared, configuration-derived rotary tables."""

import torch
from torch import nn

from nemo_automodel.components.models.step3p7.vision_encoder import (
    EncoderVisionAttention,
    StepRoboticsVisionEncoder,
    apply_rotary_emb,
)
from nemo_automodel.components.models.step5.configuration import Step5VisionConfig


class Step5VisionRotary(nn.Module):
    """Share one derived table across vision layers; never serialize or cast it."""

    def __init__(self, config: Step5VisionConfig) -> None:
        super().__init__()
        self.dim = config.width // config.heads
        self.theta = config.rope_theta * config.rope_theta_rescale_factor ** (self.dim / (self.dim - 2))
        self.maximum_grid = config.image_size // config.patch_size
        self.use_cls_token = config.use_cls_token
        self._signature: tuple[torch.device, torch.dtype, int, int] | None = None
        # Intentionally not a buffer: Module.to/to_empty must not round or
        # invalidate derived tables without changing their signature.
        self._frequencies: torch.Tensor | None = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, grid_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the source encoder's interleaved 2D RoPE.

        Args:
            q: Tensor [images, heads, image_tokens, head_dim].
            k: Tensor with q's shape and dtype.
            grid_hw: Patch-grid height and width; image_tokens includes the
                optional leading class token.

        Returns:
            Rotated Q and K with their original shape and dtype. Frequencies
            are computed in FP32 and materialized in the input dtype before
            sine/cosine, as required by the released vision implementation.
        """
        height, width = grid_hw
        if not (0 < height <= self.maximum_grid and 0 < width <= self.maximum_grid):
            raise ValueError("Step5 vision grid exceeds its configured rotary grid")
        signature = (q.device, q.dtype, height, width)
        if self._signature != signature:
            half = self.dim // 2
            inv = 1 / self.theta ** (torch.arange(0, half, 2, device=q.device, dtype=torch.float32) / half)
            rows = torch.arange(height, device=q.device, dtype=torch.float32) + int(self.use_cls_token)
            columns = torch.arange(width, device=q.device, dtype=torch.float32) + int(self.use_cls_token)
            vertical = (rows[:, None] * inv).repeat_interleave(2, -1)[:, None].expand(height, width, -1)
            horizontal = (columns[:, None] * inv).repeat_interleave(2, -1)[None].expand(height, width, -1)
            frequencies = torch.cat((horizontal, vertical), -1).reshape(height * width, self.dim)
            if self.use_cls_token:
                frequencies = torch.cat((frequencies.new_zeros(1, self.dim), frequencies))
            self._frequencies = frequencies[None, None].to(q.dtype)
            self._signature = signature
        assert self._frequencies is not None
        return apply_rotary_emb(self._frequencies, q), apply_rotary_emb(self._frequencies, k)


class Step5VisionEncoder(StepRoboticsVisionEncoder):
    """Reuse the Step encoder while owning derived-state initialization."""

    def __init__(self, config: Step5VisionConfig) -> None:
        super().__init__(config)
        if config.use_rope2d:
            rotary = Step5VisionRotary(config)
            for block in self.transformer.resblocks:
                block.attn.rope = rotary

    def forward(self, pixels: torch.Tensor, *, downsample: bool = False) -> torch.Tensor:
        """Encode images, optionally completing projection inputs inside the FSDP owner.

        Args:
            pixels: Images of shape [images, channels, height, width].
            downsample: Apply both checkpoint-owned stride-2 convolutions before
                FSDP reshards the vision module's parameters.

        Returns:
            Patch features [images, patches, width], or downsampled features
            [images, image_tokens, 4 * width] for the language projector.
        """
        features = super().forward(pixels)
        if downsample:
            grid_h, grid_w = (side // self.patch_size for side in pixels.shape[-2:])
            features = features.transpose(1, 2).reshape(pixels.shape[0], -1, grid_h, grid_w)
            features = self.vit_downsampler2(self.vit_downsampler1(features))
            features = features.flatten(2).transpose(1, 2)
        return features

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize every vision parameter after allocation from meta."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
                module.reset_parameters()
            elif isinstance(module, EncoderVisionAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
        if self.use_abs_posemb:
            nn.init.normal_(self.positional_embedding, std=self.hidden_size**-0.5)
        if self.class_embedding is not None:
            nn.init.normal_(self.class_embedding, std=self.hidden_size**-0.5)
        for block in self.transformer.resblocks:
            nn.init.constant_(block.ls_1.gamma, self.config.ls_init_value)
            nn.init.constant_(block.ls_2.gamma, self.config.ls_init_value)
