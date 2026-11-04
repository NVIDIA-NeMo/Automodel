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

"""Packed DSpark target forward for DeepSeek V4."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nemo_automodel.components.models.deepseek_v41.dspark_target import aligned_packed_layout
from nemo_automodel.components.models.deepseek_v41.packing import PackedSequenceLayout


def dspark_packed_forward(model: nn.Module, input_ids: torch.Tensor, *, seq_lens: torch.Tensor) -> PackedSequenceLayout:
    """Run a frozen DeepSeek V4 forward on packed rows for DSpark feature capture.

    The THD path derives document IDs from its ``seq_lens``, so sliding-window attention,
    compressed KV, and the indexer stay inside each document. A document pools the same
    compressed groups as it does alone only if it starts on a compression-group boundary,
    so every document is moved to one. The alignment padding runs as its own THD
    segments, which keeps a legal key in every query row. With the HybridEP dispatcher
    every rank pads to the longest aligned row of all ranks. DSpark reads hooked
    activations, so only the last position's logits are projected.

    Args:
        model: ``DeepseekV4ForCausalLM`` target.
        input_ids: Token IDs of shape [batch, sequence] holding packed documents.
        seq_lens: Document lengths of shape [batch, documents]; each row sums to
            ``sequence`` and zero entries pad the document axis.

    Returns:
        Layout whose ``restore`` maps a tensor of shape [batch, aligned_sequence, ...]
        back to [batch, sequence, ...].
    """
    ratios = model.config.compress_ratios[: model.config.num_hidden_layers]
    alignment = math.lcm(*(int(r) for r in ratios if r))
    layout = aligned_packed_layout(
        seq_lens,
        input_shape=tuple(input_ids.shape),
        alignment=alignment,
        sync_across_ranks=model.backend.dispatcher == "hybridep",
    )
    # THD segments of shape [batch, 2 * documents + 1]: each document, then its
    # alignment padding, then the row's trailing padding. Empty segments are skipped.
    lengths = seq_lens.clamp_min(0)
    spans = (lengths + alignment - 1) // alignment * alignment
    tail = layout.sequence_ids.shape[1] - spans.sum(-1, keepdim=True)
    segment_lens = torch.cat((torch.stack((lengths, spans - lengths), dim=-1).flatten(1), tail), dim=1)
    model(
        layout.pack(input_ids, fill=model.config.pad_token_id or 0),
        position_ids=layout.position_ids,
        qkv_format="thd",
        seq_lens=segment_lens,
        logits_to_keep=1,
    )
    return layout
