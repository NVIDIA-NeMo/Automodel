# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 Mistral and the HuggingFace Inc. team. All rights reserved.
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

"""Temporary Pixtral forward backport for Transformers 5.17.0, 5.18.0, and 5.19.0.

Adapted from Hugging Face Transformers (Apache-2.0), with the image-boundary
fix from https://github.com/huggingface/transformers/pull/49373
(ff704ffd47d800e31b31f2f81a5a2952fb0fbf71). Remove this module and its binding
when AutoModel pins a release containing that fix.
"""

import torch
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel, generate_block_attention_mask
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import is_flash_attention_requested, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs


@merge_with_config_defaults
@capture_outputs
def _pixtral_vision_forward(
    self: PixtralVisionModel,
    pixel_values: torch.Tensor,
    image_sizes: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | BaseModelOutput | BaseModelOutputWithPooling:
    """Preserve axial positions while passing explicit FlashAttention image boundaries.

    Args:
        pixel_values: Tensor of shape [images, channels, height, width], padded
            to a common image size.
        image_sizes: Tensor of shape [images, 2] containing each unpadded height
            and width. A list of height/width pairs is also accepted upstream.
        **kwargs: Options following ``PixtralVisionModel.forward``'s contract.

    Returns:
        The ``PixtralVisionModel.forward`` output contract: last hidden state
        of shape [1, tokens, hidden], with tokens packed across all images,
        plus optional recorded hidden states and attentions.
    """
    if image_sizes is None:
        batch_size, _, height, width = pixel_values.shape
        image_sizes = [(height, width)] * batch_size

    target_dtype = self.patch_conv.weight.dtype
    patch_embeds = self.patch_conv(pixel_values.to(dtype=target_dtype))
    patch_embeds_list = [
        embed[..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)]
        for embed, size in zip(patch_embeds, image_sizes)
    ]
    patch_embeds = torch.cat([p.flatten(1).T for p in patch_embeds_list], dim=0).unsqueeze(0)
    patch_embeds = self.ln_pre(patch_embeds)

    position_ids = []
    for patch in patch_embeds_list:
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(patch.shape[-2], device=patch.device),
            torch.arange(patch.shape[-1], device=patch.device),
            indexing="ij",
        )
        position_ids.append(torch.stack([hpos_ids.flatten(), wpos_ids.flatten()], dim=-1))
    position_ids = torch.cat(position_ids, dim=0)
    position_embeddings = self.patch_positional_embedding(patch_embeds, position_ids)

    if is_flash_attention_requested(self.config):
        # Axial positions contain two coordinates per patch, not image boundaries.
        sequence_lengths = [p.shape[-2] * p.shape[-1] for p in patch_embeds_list]
        cu_seqlens = torch.tensor([0, *sequence_lengths], device=patch_embeds.device, dtype=torch.int32)
        cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
        kwargs.update(
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max(sequence_lengths),
            max_length_k=max(sequence_lengths),
        )
        attention_mask = None
    else:
        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )

    return self.transformer(
        patch_embeds,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        **kwargs,
    )
