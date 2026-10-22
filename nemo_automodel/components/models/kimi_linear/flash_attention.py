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

"""Packed FlashAttention-2 for Kimi Linear's unequal QK and value widths."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nemo_automodel.shared.import_utils import UnavailableError, safe_import_from

_FLASH_MSG = "Kimi Linear flash_attention_2 requires flash-attn. Install with `uv sync --extra fa`."
_FLASH_OK, flash_attn_varlen_func = safe_import_from("flash_attn", "flash_attn_varlen_func", msg=_FLASH_MSG)


@dataclass(frozen=True)
class MLAFlashLayout:
    """Unpadding metadata shared by all non-CP MLA layers in one forward.

    Attributes:
        indices: Tensor of shape [valid_tokens] of int64 indices into flattened [batch,
            sequence] rows, grouped by document while preserving token order.
        cu_seqlens: Tensor of shape [documents + 1] of int32 cumulative document lengths.
        max_seqlen: Maximum valid document length, or zero for an empty batch.
    """

    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int


def build_flash_layout(doc_ids: torch.Tensor) -> MLAFlashLayout:
    """Group valid tokens into independent documents, including across batch rows.

    Run outside activation-checkpointed layers so dynamic metadata and its host
    synchronization happen once per model forward. Zero-valued padding is
    omitted. Repeated document IDs in different batch rows remain independent.

    Args:
        doc_ids: Integer tensor of shape [batch, sequence] with positive document IDs and
            zero padding, on the same device as the attention tensors.

    Returns:
        Layout whose indices address flattened batch/sequence positions and
        whose cumulative lengths describe the gathered document-major stream.
    """
    indices = torch.nonzero(doc_ids.flatten() > 0, as_tuple=False).flatten()
    if indices.numel() == 0:
        return MLAFlashLayout(indices, torch.zeros(1, dtype=torch.int32, device=doc_ids.device), 0)
    ids = doc_ids.flatten().index_select(0, indices).long()
    keys = (indices // doc_ids.shape[1]) * (doc_ids.max().long() + 1) + ids
    order = torch.argsort(keys, stable=True)
    indices = indices.index_select(0, order)
    _, lengths = torch.unique_consecutive(keys.index_select(0, order), return_counts=True)
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    return MLAFlashLayout(indices, cu_seqlens, int(lengths.max().item()))


def packed_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: MLAFlashLayout,
    scale: float,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Apply causal attention per document without dense sequence-square masks.

    FlashAttention-2 requires equal head widths. Pad Q/K/V to their largest
    width (aligned to eight), keep the original QK softmax scale, and slice V's
    output back to its original width. Padding query outputs are zero.

    Args:
        query: CUDA fp16/bf16 tensor of shape [batch, heads, sequence, qk_dim].
        key: CUDA fp16/bf16 tensor of shape [batch, heads, sequence, qk_dim].
        value: CUDA fp16/bf16 tensor of shape [batch, heads, sequence, value_dim].
        layout: Document-major indices and cumulative lengths for this batch;
            its tensor fields reside on the input device (see MLAFlashLayout).
        scale: Softmax scale based on the original, unpadded QK width.
        dropout_p: Attention dropout probability; callers pass zero for eval.

    Returns:
        Tensor of shape [batch, heads, sequence, value_dim], independent of the inputs,
        with zero output for padding tokens and gradients through valid tokens.
    """
    if not _FLASH_OK:
        raise UnavailableError(_FLASH_MSG)
    if layout.indices.numel() == 0:
        return torch.zeros_like(value) + (query.sum() + key.sum() + value.sum()) * 0
    batch, heads, sequence, value_dim = value.shape
    width = ((max(query.shape[-1], value_dim) + 7) // 8) * 8
    if width > 256:
        raise ValueError("Kimi Linear FlashAttention-2 requires padded head width <= 256")
    gathered = []
    for tensor in (query, key, value):
        tokens = tensor.transpose(1, 2).reshape(batch * sequence, heads, tensor.shape[-1])
        tokens = tokens.index_select(0, layout.indices)
        if tokens.shape[-1] != width:
            tokens = F.pad(tokens, (0, width - tokens.shape[-1]))
        gathered.append(tokens)
    attended = flash_attn_varlen_func(
        *gathered,
        layout.cu_seqlens,
        layout.cu_seqlens,
        layout.max_seqlen,
        layout.max_seqlen,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=True,
    )[..., :value_dim]
    padded = attended.new_zeros((batch * sequence, heads, value_dim)).index_copy(0, layout.indices, attended)
    return padded.reshape(batch, sequence, heads, value_dim).transpose(1, 2)
