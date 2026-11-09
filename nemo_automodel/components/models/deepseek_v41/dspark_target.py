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

"""Packed DSpark target forward for DeepSeek V4.1."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from nemo_automodel.components.models.deepseek_v41.packing import PackedSequenceLayout, packed_layout


def aligned_packed_layout(
    seq_lens: torch.Tensor, *, input_shape: tuple[int, int], alignment: int, sync_across_ranks: bool
) -> PackedSequenceLayout:
    """Lay out packed rows with every document starting on a compression-group boundary.

    Args:
        seq_lens: Integer document lengths of shape [batch, documents]; zero and -1000
            entries denote absent documents.
        input_shape: Original [batch, sequence] dimensions; the aligned rows are at least
            ``sequence`` tokens wide.
        alignment: Positive multiple for each document's physical span. A compressed group
            must start where the document's first group would start on its own, so this
            is the least common multiple of the active compression ratios.
        sync_across_ranks: Max-reduce the row length over the default process group.
            HybridEP needs the same token count from every rank, and rows of the same
            width need different amounts of alignment padding.

    Returns:
        Layout of the aligned rows; its tensors have shape [batch, aligned_sequence].
    """
    minimum_length = input_shape[1]
    if sync_across_ranks and dist.is_available() and dist.is_initialized():
        spans = (seq_lens.clamp_min(0) + alignment - 1) // alignment * alignment
        length = spans.sum(-1).max().clamp_min(minimum_length)
        dist.all_reduce(length, op=dist.ReduceOp.MAX)
        minimum_length = int(length)
    return packed_layout(
        seq_lens, seq_lens_padded=None, input_shape=input_shape, alignment=alignment, minimum_length=minimum_length
    )


def dspark_packed_forward(model: nn.Module, input_ids: torch.Tensor, *, seq_lens: torch.Tensor) -> PackedSequenceLayout:
    """Run a frozen DeepSeek V4.1 forward on packed rows for DSpark feature capture.

    Documents are aligned as in the ``seq_lens`` path of the model's ``forward``, and the
    prepared ``packed_seq_ids`` keep every attention path inside its document. With the
    HybridEP dispatcher every rank pads to the longest aligned row of all ranks. DSpark
    reads hooked activations, so only the last position's logits are projected.

    Args:
        model: ``DeepseekV41ForCausalLM`` target.
        input_ids: Token IDs of shape [batch, sequence] holding packed documents.
        seq_lens: Document lengths of shape [batch, documents]; each row sums to
            ``sequence`` and zero entries pad the document axis.

    Returns:
        Layout whose ``restore`` maps a tensor of shape [batch, aligned_sequence, ...]
        back to [batch, sequence, ...].
    """
    text_config = model.config.text_config
    layout = aligned_packed_layout(
        seq_lens,
        input_shape=tuple(input_ids.shape),
        alignment=model._packed_alignment,
        sync_across_ranks=model.backend.dispatcher == "hybridep",
    )
    model(
        layout.pack(input_ids, fill=text_config.pad_token_id or 0),
        attention_mask=layout.sequence_ids > 0,
        position_ids=layout.position_ids,
        packed_seq_ids=layout.sequence_ids,
        logits_to_keep=1,
    )
    return layout
