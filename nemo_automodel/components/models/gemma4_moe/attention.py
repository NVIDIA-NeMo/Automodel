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

"""Composition of document-causal masks with Gemma4's native attention policy."""

import torch
from transformers.configuration_utils import PreTrainedConfig

from nemo_automodel.shared.import_utils import safe_import

_, _gemma4 = safe_import("transformers.models.gemma4.modeling_gemma4")
_, _masking = safe_import("transformers.masking_utils")


def prepare_gemma4_document_mask(
    config: PreTrainedConfig,
    attention_mask: torch.Tensor,
    mm_token_type_ids: torch.Tensor | None,
    document_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Compose native full/sliding attention with an explicit document-causal mask.

    Call only for the document-causal contract marked by ``_packed_seq_ids``;
    arbitrary custom 4D masks are already complete and must not be interpreted here.
    HF otherwise treats a 4D input as complete, bypassing its vision/window policy.

    Args:
        config: Gemma4 text configuration using eager or SDPA attention.
        attention_mask: Boolean-allowed or additive-zero/minimum mask [batch, 1,
            sequence, sequence]. This carries the caller's selected document
            isolation and excludes padding/fake media on both axes.
        mm_token_type_ids: Processor modality IDs [batch, sequence], or None for text.
        document_ids: Original document IDs [batch, sequence]. These keep image
            groups separate even when the selected mask allows cross-document attention.
        dtype: Model floating point dtype for the additive native masks.

    Returns:
        Full and sliding masks [batch, 1, sequence, sequence]: boolean for SDPA
        and model-dtype additive for eager. Full
        attention retains the supplied policy. Only native same-image future
        edges may be added to sliding attention, within the selected documents.
        Inputs and model position IDs are never changed. A boolean full mask
        may alias the supplied attention mask and must be treated as read-only.
    """
    if config._attn_implementation not in ("sdpa", "eager"):
        raise ValueError("Gemma4 document-causal masks currently support SDPA or eager attention")
    allowed = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    batch_size, _, seq_len, _ = allowed.shape
    eligible = allowed[:, 0].diagonal(dim1=-2, dim2=-1)
    mask_kwargs = {
        "config": config,
        "inputs_embeds": torch.empty((batch_size, seq_len, 1), dtype=dtype, device=allowed.device),
        "attention_mask": eligible,
        "past_key_values": None,
        # Mask coordinates are monotonic; resetting RoPE positions is independent.
        "position_ids": torch.arange(seq_len, device=allowed.device).unsqueeze(0),
    }
    if config.use_bidirectional_attention == "vision" and mm_token_type_ids is not None:
        groups = _gemma4.get_block_sequence_ids_for_mask(mm_token_type_ids, device=allowed.device)
        groups = torch.where(groups >= 0, groups * (seq_len + 1) + document_ids, -1)
        native = _gemma4.create_masks_for_vision_model(block_sequence_ids=groups, **mask_kwargs)
        sliding = native["sliding_attention"]
    else:
        sliding = _masking.create_sliding_window_causal_mask(**mask_kwargs, allow_is_causal_skip=False)
    native_allowed = sliding if sliding.dtype == torch.bool else sliding == 0
    document_pairs = allowed | allowed.transpose(-1, -2)
    document_pairs &= eligible[:, None, :, None]
    document_pairs &= eligible[:, None, None, :]
    native_allowed &= document_pairs
    if config._attn_implementation == "sdpa":
        return {"full_attention": allowed, "sliding_attention": native_allowed}
    minimum = torch.finfo(dtype).min
    full = torch.zeros_like(allowed, dtype=dtype).masked_fill(~allowed, minimum)
    sliding = torch.zeros_like(allowed, dtype=dtype).masked_fill(~native_allowed, minimum)
    return {"full_attention": full, "sliding_attention": sliding}
