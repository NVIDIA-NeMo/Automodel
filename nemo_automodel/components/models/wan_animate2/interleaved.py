# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Interleave the two upstream Wan-Animate-2 streams inside each FSDP unit.

Diffusers 0.40 exposes reference extraction and denoising through one forward
with ``kv_cache_mode``. Training needs gradients through both passes, but FSDP2
must gather each block only once. A fresh cache inside each block also makes
activation-checkpoint recomputation independent of later forwards.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, TypedDict

import torch
from torch import nn

from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_wan_animate_2 import WanAnimate2Transformer3DModel


class WanAnimate2Inputs(TypedDict):
    """Training inputs; tensor layouts are defined by the adapter's prepare_inputs."""

    x: list[torch.Tensor]
    y: list[torch.Tensor]
    x_ref: list[torch.Tensor]
    condition_y: list[torch.Tensor]
    clip_fea: torch.Tensor
    clip_fea_ref: torch.Tensor
    context: list[torch.Tensor]
    context_ref: list[torch.Tensor]
    seq_len: int
    seq_len_ref: int
    grid_sizes_ref: torch.Tensor
    timestep: torch.Tensor
    origin_len: int
    origin_area: list[int]
    _target_latent_frames: int
    _compute_dtype: torch.dtype


class _BlockArgs(TypedDict, total=False):
    """Upstream block arguments, with layouts documented on _block_forward_origin."""

    temb: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_hidden_states_image: torch.Tensor | None
    rotary_emb: torch.Tensor
    grid_sizes: torch.Tensor
    reference_rotary_emb: torch.Tensor
    reference_grid_sizes: torch.Tensor
    attention_mask: object
    origin_latent_frames: int
    origin_latent_hw: int


def _block_forward_origin(
    self: nn.Module,
    x: torch.Tensor,
    x_ref: torch.Tensor,
    ref_args: _BlockArgs,
    gen_args: _BlockArgs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run both streams inside a single block call and a fresh local cache.

    Args:
        self: Upstream block with its original forward saved at installation.
        x: Generation tensor of shape [batch, generation_tokens, hidden].
        x_ref: Driving tensor of shape [batch, reference_tokens, hidden].
        ref_args: Reference block arguments: temb [batch, 6, hidden],
            encoder_hidden_states [batch, text_tokens, hidden], optional
            encoder_hidden_states_image [batch, image_tokens, hidden],
            rotary_emb [512, head_dim / 2] (complex), and grid_sizes
            [batch, 3] in temporal, height, width patch order.
        gen_args: Same layouts for the generation stream, plus
            reference_rotary_emb [512, head_dim / 2], reference_grid_sizes
            [batch, 3], and the upstream sparse attention BlockMask.

    Returns:
        Generation and reference tensors with the same layouts as the inputs.
        Both remain differentiable through the upstream attention and weights.
    """
    # This object never crosses an FSDP/checkpoint boundary. Recomputing the
    # whole block therefore reconstructs its cache before consuming it.
    cache = self._wan_animate2_cache_type()
    x_ref = self._wan_animate2_forward(x_ref, kv_cache=cache, kv_cache_mode="extract", **ref_args)
    x = self._wan_animate2_forward(x, kv_cache=cache, kv_cache_mode="cached", **gen_args)
    return x, x_ref


def install_forward_origin(model: nn.Module) -> None:
    """Install the training traversal on this model instance before sharding.

    Uses the released Diffusers blocks unchanged. Instance-bound methods keep
    other model instances and the upstream classes untouched; parameter names
    and checkpoint layouts also stay unchanged. The adapter calls this for
    unsharded execution, and the parallelization strategy calls it before FSDP.

    Args:
        model: WanAnimate2Transformer3DModel, optionally inside DDP.
    """
    inner = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    if getattr(inner, "_wan_animate2_training", False):
        return
    available, upstream = safe_import("diffusers.models.transformers.transformer_wan_animate_2")
    if not available or not isinstance(inner, upstream.WanAnimate2Transformer3DModel):
        raise TypeError("Wan-Animate-2 training requires WanAnimate2Transformer3DModel from diffusers>=0.40.0")
    lora_classes: dict[tuple[type, ...], type] = {}
    for block in inner.blocks:
        # LoRA injection creates a distinct empty subclass for each projection.
        # Reuse equivalent classes only on this model so type guards do not
        # exhaust Dynamo's recompile budget and change checkpoint replay to eager.
        for name in ("to_q", "to_k", "to_v"):
            projection = getattr(block.self_attn, name)
            projection_class = type(projection)
            if (
                projection_class.__name__ == "PatchedLinearLoRA"
                and projection_class.__module__ == "nemo_automodel.components._peft.lora"
            ):
                projection.__class__ = lora_classes.setdefault(projection_class.__bases__, projection_class)
        block._wan_animate2_forward = block.forward
        block._wan_animate2_cache_type = upstream.WanAnimate2KVLayerCache
        block.forward = MethodType(_block_forward_origin, block)
        if inner.patch_embedding.weight.device.type == "cuda":
            # Flex attention must compile at video resolutions. Compile only
            # these blocks, without graph capture, so checkpoint replay can
            # recreate activations. No upstream/global compiler state changes.
            block.compile(fullgraph=False, mode="max-autotune-no-cudagraphs")
    inner.forward = MethodType(_forward_origin, inner)
    inner._wan_animate2_training = True


def _forward_origin(self: WanAnimate2Transformer3DModel, inputs: WanAnimate2Inputs) -> list[torch.Tensor]:
    """Embed both streams, then enter each transformer block exactly once.

    Args:
        self: Transformer whose blocks have the interleaved training forward.
        inputs: Adapter mapping: x/y are per-sample tensors [16/20,
            target_frames + 1, height, width]; x_ref/condition_y are
            [16/20, target_frames, height, width]. context/context_ref hold
            [text_tokens, text_dim], clip_fea/clip_fea_ref hold
            [batch, image_tokens, 1280], timestep is [batch], and
            grid_sizes_ref is [batch, 3] in temporal, height, width patch order.
            Spatial dimensions are in latent pixels. See the adapter for the
            remaining scalar geometry and slicing metadata.

    Returns:
        Per-sample float32 predictions [16, target_frames + 1, height, width].
    """
    device = self.patch_embedding.weight.device

    def embed(latents: list[torch.Tensor], condition: list[torch.Tensor], seq_len: int):
        """Patch-embed one stream with the upstream exact-length contract.

        Args:
            latents: Per-sample tensors [16, frames, height, width].
            condition: Per-sample tensors [20, frames, height, width].
            seq_len: Expected post-patch tokens in every sample.

        Returns:
            Tokens [batch, seq_len, hidden] and CPU grids [batch, 3] in
            temporal, height, width patch order.
        """
        embedded = [self.patch_embedding(torch.cat([x, y], dim=0).unsqueeze(0)) for x, y in zip(latents, condition)]
        grid = torch.tensor([x.shape[2:] for x in embedded], dtype=torch.long)
        flat = [x.flatten(2).transpose(1, 2) for x in embedded]
        if any(x.shape[1] != seq_len for x in flat):
            raise ValueError(f"Wan-Animate-2 expects exactly {seq_len} tokens per sample")
        return torch.cat(flat), grid

    def context(text: list[torch.Tensor], image: torch.Tensor):
        """Embed text and image conditioning with upstream padding.

        Args:
            text: Per-sample tensors [text_tokens, text_dim].
            image: CLIP features [batch, image_tokens, 1280].

        Returns:
            Text [batch, config.text_len, hidden] and optional image features
            [batch, image_tokens, hidden].
        """
        text_embeds = self.text_embedding(
            torch.stack([torch.cat([x, x.new_zeros(self.config.text_len - x.shape[0], x.shape[1])]) for x in text])
        )
        return text_embeds, self.img_emb(image) if self.config.use_img_emb else None

    def time_embedding(timestep: torch.Tensor):
        """Project stream timesteps using the upstream precision policy.

        Args:
            timestep: Tensor of shape [batch].

        Returns:
            Head modulation [batch, hidden] and block modulation
            [batch, 6, hidden].
        """
        temb = self.time_embedding(self.timesteps_proj(timestep).to(next(self.time_embedding.parameters()).dtype))
        return temb, self.time_projection(temb).unflatten(1, (6, self.config.dim))

    x_ref, reference_grid = embed(inputs["x_ref"], inputs["condition_y"], inputs["seq_len_ref"])
    x, grid = embed(inputs["x"], inputs["y"], inputs["seq_len"])
    offsets = tuple(
        offset if offset >= 0 else reference_grid[0, axis].item()
        for axis, offset in enumerate(
            (self.config.refer_offset_t, self.config.refer_offset_h, self.config.refer_offset_w)
        )
    )
    rotary_ref = self._rope_freqs(offsets, device)
    _, temb_ref = time_embedding(inputs["timestep"] * 0 + 1)
    temb_head, temb = time_embedding(inputs["timestep"])
    text_ref, image_ref = context(inputs["context_ref"], inputs["clip_fea_ref"])
    text, image = context(inputs["context"], inputs["clip_fea"])

    origin_frames = inputs["origin_len"] // 4 + 1
    origin_hw = inputs["origin_area"][0] * inputs["origin_area"][1] // 256
    mask_key = (origin_frames, origin_hw)
    if mask_key not in self.block_masks:
        self.block_masks[mask_key] = self.create_mask(origin_frames, origin_hw, device)
    ref_args: _BlockArgs = dict(
        temb=temb_ref,
        encoder_hidden_states=text_ref,
        encoder_hidden_states_image=image_ref,
        rotary_emb=rotary_ref,
        grid_sizes=reference_grid,
    )
    gen_args: _BlockArgs = dict(
        temb=temb,
        encoder_hidden_states=text,
        encoder_hidden_states_image=image,
        rotary_emb=self._rope_freqs((0, 0, 0), device),
        grid_sizes=grid,
        reference_rotary_emb=rotary_ref,
        reference_grid_sizes=reference_grid,
        attention_mask=self.block_masks[mask_key],
        origin_latent_frames=origin_frames,
        origin_latent_hw=origin_hw,
    )
    for block in self.blocks:
        x, x_ref = block(x, x_ref, ref_args, gen_args)
    return [prediction.float() for prediction in self.unpatchify(self.head(x, temb_head), grid)]
