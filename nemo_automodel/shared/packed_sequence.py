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

"""Shared packed-sequence metadata helpers."""

import torch
import torch.nn.functional as F


def get_seqlens_in_batch(attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract document lengths from an indexed packed-sequence mask.

    Args:
        attention_mask: Integer tensor of shape [batch, sequence]. Each nonzero
            value is a 1-based document index local to its batch row; zero marks
            padding.

    Returns:
        Tensor of shape [documents] containing nonzero document lengths in
        batch-major, document-index order.
    """
    batch_size = attention_mask.size(0)
    dtype, device = attention_mask.dtype, attention_mask.device
    max_documents = int(torch.max(attention_mask).item())
    counts = torch.zeros((batch_size, max_documents), dtype=dtype, device=device)
    for document_idx in range(max_documents):
        counts[:, document_idx] = torch.sum(attention_mask == (document_idx + 1), dim=-1)

    counts = counts.flatten()
    return counts[counts.nonzero().squeeze(dim=-1)]


def get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build varlen metadata for an indexed or binary attention mask.

    Indexed masks treat every distinct positive document index in each batch
    row as a separate sequence. Binary masks treat every nonempty batch row as
    one sequence. Padding tokens are omitted from the flattened token stream.

    Args:
        attention_mask: Integer or boolean tensor of shape [batch, sequence].
            Positive values identify valid tokens and zero marks padding.

    Returns:
        A tuple containing ``indices`` of shape [tokens] into the flattened
        padded stream, ``cu_seqlens`` of shape [documents + 1] with dtype
        int32, and the largest document length as an integer.

    Raises:
        ValueError: If the mask is not rank two or contains no valid tokens.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, sequence], got {tuple(attention_mask.shape)}")
    if attention_mask.numel() == 0 or not bool(attention_mask.bool().any().item()):
        raise ValueError("attention_mask must contain at least one valid token")

    if attention_mask.dtype == torch.bool or int(attention_mask.max().item()) <= 1:
        seqlens_in_batch = attention_mask.bool().sum(dim=-1)
        seqlens_in_batch = seqlens_in_batch[seqlens_in_batch > 0]
    else:
        seqlens_in_batch = get_seqlens_in_batch(attention_mask)

    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch
