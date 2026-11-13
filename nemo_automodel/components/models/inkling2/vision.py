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

# The vision tower structure, submodule names and numerics follow the HuggingFace Inkling 2
# implementation (Apache-2.0).
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Source: https://github.com/huggingface/transformers/pull/50682
# Revision: 109a7d19bf667168837f32929e00b791e195d14d
# Original file: src/transformers/models/inkling2/modeling_inkling2.py

"""Native Inkling 2 vision tower: a ViT with 2D axial RoPE and a patch-merging projector.

Code provenance: the module structure, submodule names and numerics follow the Inkling 2
vision model in Hugging Face Transformers (https://github.com/huggingface/transformers,
commit 109a7d19bf667168837f32929e00b791e195d14d,
``src/transformers/models/inkling2/modeling_inkling2.py``, Apache-2.0, Copyright 2026 the
HuggingFace Team), so that the reference checkpoint key conversion applies unchanged.
The grid helpers are imported from ``transformers.vision_utils``, which is identical in
Transformers 5.18.0 and that commit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

if TYPE_CHECKING:
    from .configuration import Inkling2VisionConfig


def _axial_rotary_cos_sin(
    position_ids: torch.Tensor, head_dim: int, rope_theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the interleaved 2D axial rotary tables in float32.

    The frequencies are recomputed from the config on every call instead of being stored in
    a buffer, so a model-wide bf16 cast cannot round them.

    Args:
        position_ids: Long tensor of shape ``[tokens, 2]``, where column 0 is the row and
            column 1 is the column of each token on its image grid.
        head_dim: Attention head dimension; must be divisible by 4.
        rope_theta: Rotary base.

    Returns:
        ``(cos, sin)``, each a float32 tensor of shape ``[tokens, head_dim]``. The first half of
        the head dimension rotates with the column position and the second half with the row
        position; every frequency covers one adjacent channel pair.
    """
    spatial_dim = head_dim // 2
    exponent = torch.arange(0, spatial_dim, 2, dtype=torch.float32, device=position_ids.device) / spatial_dim
    inv_freq = 1.0 / (rope_theta**exponent)
    freqs = position_ids[..., None].float() * inv_freq
    freqs = torch.cat([freqs[:, 1], freqs[:, 0]], dim=-1).repeat_interleave(2, dim=-1)
    return freqs.cos(), freqs.sin()


def _rotate_pairs(hidden_states: torch.Tensor) -> torch.Tensor:
    """Rotate each adjacent channel pair ``(x1, x2)`` to ``(-x2, x1)``.

    Args:
        hidden_states: Tensor of shape ``[..., head_dim]`` with an even ``head_dim``.

    Returns:
        Tensor of shape ``[..., head_dim]``.
    """
    pairs = hidden_states.unflatten(-1, (-1, 2))
    first, second = pairs.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(start_dim=-2)


def _apply_axial_rotary(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the interleaved axial rotary embedding in float32.

    Args:
        hidden_states: Tensor of shape ``[tokens, heads, head_dim]``.
        cos: Float32 tensor of shape ``[tokens, head_dim]``.
        sin: Float32 tensor of shape ``[tokens, head_dim]``.

    Returns:
        Tensor of shape ``[tokens, heads, head_dim]`` in the dtype of ``hidden_states``.
    """
    states = hidden_states.float()
    rotated = states * cos.unsqueeze(-2) + _rotate_pairs(states) * sin.unsqueeze(-2)
    return rotated.to(hidden_states.dtype)


class Inkling2VisionAttention(nn.Module):
    """Bidirectional multi-head self-attention restricted to each image."""

    def __init__(self, config: Inkling2VisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim or self.head_dim % 4 != 0:
            raise ValueError(
                f"hidden_size ({self.embed_dim}) must split into num_attention_heads ({self.num_heads}) heads "
                "whose size is divisible by 4 for the axial rotary embedding"
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: list[int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Attend within each image sequence.

        Args:
            hidden_states: Tensor of shape ``[tokens, hidden]`` holding every image's sequence
                (class token, then its patches) back to back.
            sequence_lengths: Length of each image sequence; sums to ``tokens``.
            position_embeddings: ``(cos, sin)`` float32 tensors of shape ``[tokens, head_dim]``.

        Returns:
            Tensor of shape ``[tokens, hidden]``.
        """
        tokens = hidden_states.shape[0]
        query = self.q_proj(hidden_states).view(tokens, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(tokens, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(tokens, self.num_heads, self.head_dim)
        cos, sin = position_embeddings
        query = _apply_axial_rotary(query, cos, sin)
        key = _apply_axial_rotary(key, cos, sin)

        # [tokens, heads, head_dim] -> [1, heads, tokens, head_dim], split per image along tokens.
        query, key, value = (tensor.transpose(0, 1).unsqueeze(0) for tensor in (query, key, value))
        dropout = self.attention_dropout if self.training else 0.0
        outputs = [
            F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=False, scale=self.scaling)
            for q, k, v in zip(
                query.split(sequence_lengths, dim=2),
                key.split(sequence_lengths, dim=2),
                value.split(sequence_lengths, dim=2),
            )
        ]
        attn_output = torch.cat(outputs, dim=2).squeeze(0).transpose(0, 1).reshape(tokens, self.embed_dim)
        return self.out_proj(attn_output)


class Inkling2VisionMLP(nn.Module):
    """Two-layer feed-forward block of a vision encoder layer."""

    def __init__(self, config: Inkling2VisionConfig) -> None:
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward block.

        Args:
            hidden_states: Tensor of shape ``[tokens, hidden]``.

        Returns:
            Tensor of shape ``[tokens, hidden]``.
        """
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class Inkling2VisionEncoderLayer(nn.Module):
    """Pre-norm ViT layer with per-channel layer scales on both residual branches."""

    def __init__(self, config: Inkling2VisionConfig) -> None:
        super().__init__()
        self.layer_scale_init_value = config.layer_scale_init_value
        self.self_attn = Inkling2VisionAttention(config)
        self.mlp = Inkling2VisionMLP(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lambda_1 = nn.Parameter(torch.full((config.hidden_size,), float(config.layer_scale_init_value)))
        self.lambda_2 = nn.Parameter(torch.full((config.hidden_size,), float(config.layer_scale_init_value)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: list[int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run one encoder layer.

        Args:
            hidden_states: Tensor of shape ``[tokens, hidden]``, image sequences back to back.
            sequence_lengths: Length of each image sequence; sums to ``tokens``.
            position_embeddings: ``(cos, sin)`` float32 tensors of shape ``[tokens, head_dim]``.

        Returns:
            Tensor of shape ``[tokens, hidden]``.
        """
        attn_output = self.self_attn(self.layernorm_before(hidden_states), sequence_lengths, position_embeddings)
        hidden_states = hidden_states + self.lambda_1 * attn_output
        return hidden_states + self.lambda_2 * self.mlp(self.layernorm_after(hidden_states))


class Inkling2VisionEmbeddings(nn.Module):
    """Patch projection, interpolated learned positions and one class token per image."""

    def __init__(self, config: Inkling2VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.merge_size = config.spatial_merge_size
        self.grid_size = config.posemb_grid_size
        self.patch_embedding = nn.Linear(config.num_channels * config.patch_size**2, config.hidden_size, bias=False)
        self.position_embedding = nn.Embedding(config.posemb_grid_size**2, config.hidden_size)
        self.class_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        self.cls_position_embedding = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(
        self, pixel_values: torch.Tensor, patch_grid_thw: torch.Tensor, patch_index: torch.Tensor
    ) -> torch.Tensor:
        """Embed patches and interleave the class tokens.

        Args:
            pixel_values: Tensor of shape ``[lm_tokens, merge * patch, merge * patch, channels]``, one
                square of ``merge x merge`` native patches per language-model image token, in raster
                order within each image.
            patch_grid_thw: CPU long tensor of shape ``[images, 3]`` with each image's
                ``(time, rows, columns)`` in native patches.
            patch_index: Long tensor of shape ``[patches]`` giving each patch's position in the
                output sequence; the slot before each image's first patch holds its class token.

        Returns:
            Tensor of shape ``[patches + images, hidden]``: per image, the class token followed by
            its patches in spatial-merge-block order.
        """
        merge, patch = self.merge_size, self.patch_size
        # [lm_tokens, merge, patch, merge, patch, C] -> [lm_tokens, merge, merge, patch, patch, C]
        patches = pixel_values.reshape(-1, merge, patch, merge, patch, pixel_values.shape[-1]).transpose(2, 3)
        patch_embeddings = self.patch_embedding(patches.reshape(-1, self.patch_embedding.in_features))

        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            patch_grid_thw,
            num_grid_per_side=self.grid_size,
            mode="bilinear",
            align_corners=False,
            spatial_merge_size=merge,
        )
        interp_indices = interp_indices.to(pixel_values.device, non_blocking=True)
        interp_weights = interp_weights.to(pixel_values.device, non_blocking=True)
        position_embeddings = (self.position_embedding(interp_indices) * interp_weights[:, :, None]).sum(1)
        patch_embeddings = patch_embeddings + position_embeddings.to(patch_embeddings.dtype)

        cls_embedding = (self.class_embedding + self.cls_position_embedding).to(patch_embeddings.dtype)
        embeddings = cls_embedding.repeat(patch_embeddings.shape[0] + patch_grid_thw.shape[0], 1)
        embeddings[patch_index] = patch_embeddings
        return embeddings


class Inkling2VisionMerger(nn.Module):
    """Project each ``merge x merge`` square of patch features to one language-model embedding."""

    def __init__(self, config: Inkling2VisionConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(
            config.spatial_merge_size**2 * config.hidden_size,
            config.projection_hidden_dim,
            bias=config.projector_bias,
        )
        self.act = ACT2FN[config.hidden_act]
        self.linear_2 = nn.Linear(config.projection_hidden_dim, config.text_hidden_size, bias=config.projector_bias)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project merged patch features.

        Args:
            image_features: Tensor of shape ``[lm_tokens, merge * merge * hidden]``, the features of
                one square of patches concatenated in row-major order within the square.

        Returns:
            Tensor of shape ``[lm_tokens, text_hidden]``.
        """
        return self.linear_2(self.act(self.linear_1(image_features)))


class Inkling2VisionModel(nn.Module):
    """Inkling 2 vision tower producing one language-model embedding per image token.

    Submodule names match the Transformers reference after its checkpoint conversion
    (``embeddings.*``, ``pre_layernorm``, ``layers.N.*``, ``merger.linear_{1,2}``).
    """

    def __init__(self, config: Inkling2VisionConfig) -> None:
        super().__init__()
        rope_parameters = config.rope_parameters
        if rope_parameters["rope_type"] != "axial":
            raise ValueError(f"Inkling 2 vision supports only axial rope, got {rope_parameters['rope_type']!r}")
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = float(rope_parameters["rope_theta"])
        self.embeddings = Inkling2VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([Inkling2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.merger = Inkling2VisionMerger(config)

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, **kwargs: Any
    ) -> BaseModelOutputWithPooling:
        """Encode a batch of images.

        Args:
            pixel_values: Tensor of shape ``[lm_tokens, merge * patch, merge * patch, channels]``
                (channels last, normalized to about ``[-1, 1]``), one entry per language-model image
                token in raster order within each image and images back to back.
            image_grid_thw: Long tensor of shape ``[images, 3]`` with each image's
                ``(time, rows, columns)`` counted in language-model image tokens.
            **kwargs: Accepted for the shared multimodal calling convention and ignored.

        Returns:
            ``BaseModelOutputWithPooling`` with ``last_hidden_state`` of shape
            ``[patches + images, hidden]`` (per image, the class token then its patches in
            spatial-merge-block order) and ``pooler_output`` of shape ``[lm_tokens, text_hidden]``
            in the same order as ``pixel_values``.
        """
        del kwargs
        merge = self.config.spatial_merge_size
        patch_grid_thw = image_grid_thw.cpu() * torch.tensor([1, merge, merge])
        patches_per_image = patch_grid_thw.prod(-1)
        num_patches = int(patches_per_image.sum())
        side = merge * self.config.patch_size
        if tuple(pixel_values.shape[1:]) != (side, side, self.config.num_channels) or (
            pixel_values.shape[0] * merge**2 != num_patches
        ):
            raise ValueError(
                f"pixel_values of shape {tuple(pixel_values.shape)} do not match image_grid_thw "
                f"{image_grid_thw.tolist()}: expected [{num_patches // merge**2}, {side}, {side}, "
                f"{self.config.num_channels}]"
            )

        device = pixel_values.device
        # Sequence slot of every patch, skipping the class token in front of each image.
        image_offsets = torch.repeat_interleave(torch.arange(1, patch_grid_thw.shape[0] + 1), patches_per_image)
        patch_index = (torch.arange(num_patches) + image_offsets).to(device, non_blocking=True)
        sequence_lengths = (patches_per_image + 1).tolist()

        pixel_values = pixel_values.to(self.embeddings.patch_embedding.weight.dtype)
        hidden_states = self.pre_layernorm(self.embeddings(pixel_values, patch_grid_thw, patch_index))

        # Patches take 1-indexed (row, column) grid positions; class tokens sit at (0, 0).
        patch_positions = get_vision_position_ids(patch_grid_thw, merge).to(device, non_blocking=True)
        position_ids = patch_positions.new_zeros(hidden_states.shape[0], 2).index_copy(
            0, patch_index, patch_positions + 1
        )
        position_embeddings = _axial_rotary_cos_sin(position_ids, self.head_dim, self.rope_theta)

        for layer in self.layers:
            hidden_states = layer(hidden_states, sequence_lengths, position_embeddings)

        patch_states = hidden_states[patch_index.to(hidden_states.device)]
        pooler_output = self.merger(patch_states.reshape(-1, merge**2 * hidden_states.shape[-1]))
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=pooler_output)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize every parameter for checkpoint-free construction.

        Linear and embedding weights draw from ``N(0, initializer_range)``, biases and layer-norm
        shifts are zero, layer-norm scales are one and the layer scales start at
        ``layer_scale_init_value``, as in the reference initialization.
        """
        std = self.config.initializer_range
        embeddings = self.embeddings
        for parameter in (
            embeddings.patch_embedding.weight,
            embeddings.position_embedding.weight,
            embeddings.class_embedding,
            embeddings.cls_position_embedding,
        ):
            nn.init.normal_(parameter, mean=0.0, std=std)
        linears = [self.merger.linear_1, self.merger.linear_2]
        norms = [self.pre_layernorm]
        for layer in self.layers:
            attention = layer.self_attn
            linears.extend((attention.q_proj, attention.k_proj, attention.v_proj, attention.out_proj))
            linears.extend((layer.mlp.fc1, layer.mlp.fc2))
            norms.extend((layer.layernorm_before, layer.layernorm_after))
            nn.init.constant_(layer.lambda_1, layer.layer_scale_init_value)
            nn.init.constant_(layer.lambda_2, layer.layer_scale_init_value)
        for linear in linears:
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        for norm in norms:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)


__all__ = ["Inkling2VisionModel"]
