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

"""AutoModel batch representations after Zephon collation and attention policy."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from nemo_automodel.components.datasets.packing import BatchOutputFormat
from nemo_automodel.components.datasets.zephon.attention import TrainingBatch


def convert_batch_output(batch: TrainingBatch, *, output_format: BatchOutputFormat) -> TrainingBatch:
    """Describe packed documents using the selected AutoModel input format.

    Zephon owns packing, padding and shifting. THD adds the native length-based
    schema; AutoModel's existing preparation flattens tensors later, after this
    map. No token is moved or re-tokenized, and positions are never reconstructed
    from the length metadata. Xdoc policy has already selected the attention IDs.

    Args:
        batch: CPU token, label, padding and document tensors of shape
            [batch, sequence]. Positions have shape [batch, sequence] or
            [axes, batch, sequence]. Image tensors retain their processor layouts.
            Positive attention_document_ids delimit contiguous subdocuments;
            zero marks exclusions. Padding must be a trailing suffix in each
            packed document. Each packed document must contain a real token.
        output_format: "batched" preserves the mapping. "thd" selects the native
            seq_lens input route; it does not select a packing algorithm.

    Returns:
        The input mapping for batched output, or a shallow copy for THD with
        int64 seq_lens/seq_lens_padded [batch, max_subdocuments] (-1000 for unused
        entries) and qkv_format="thd". Real lengths exclude padding; physical
        lengths cover every token slot, attaching trailing padding to the last
        subdocument. Original document IDs, labels, positions and media alias
        their input tensors. No input mapping or tensor is mutated.

    Raises:
        ValueError: For unknown formats, non-suffix padding, noncontiguous
            subdocuments, empty documents or processor-excluded real tokens.
            Those exclusions require the batched mask/metadata representation.
    """
    if output_format == "batched":
        return batch
    if output_format != "thd":
        raise ValueError("output_format must be batched or thd")
    document_ids = batch["attention_document_ids"]
    padding = batch["padding_mask"]
    if document_ids.ndim != 2 or document_ids.shape != padding.shape:
        raise ValueError("THD document IDs and padding must have shape [batch, sequence]")
    if not torch.equal(document_ids.gt(0), ~padding):
        raise ValueError("THD cannot represent processor-excluded real tokens; use batched output")
    lengths = []
    physical_lengths = []
    sequence_length = document_ids.shape[1]
    for ids, mask in zip(document_ids, padding):
        real_length = int((~mask).sum())
        if not real_length or bool(mask[:real_length].any()):
            raise ValueError("THD requires nonempty documents with suffix padding")
        membership, counts = torch.unique_consecutive(ids[:real_length], return_counts=True)
        if bool((membership[1:] <= membership[:-1]).any()):
            raise ValueError("THD requires contiguous subdocuments in token order")
        lengths.append(counts)
        slots = counts.clone()
        slots[-1] += sequence_length - real_length
        physical_lengths.append(slots)
    result = batch.copy()
    # Gathered-token metadata has different offsets; native THD prepares its own.
    for key in ("packed_token_indices", "cu_seqlens", "max_seqlen", "attention_mask"):
        result.pop(key, None)
    result.update(
        seq_lens=pad_sequence(lengths, batch_first=True, padding_value=-1000),
        seq_lens_padded=pad_sequence(physical_lengths, batch_first=True, padding_value=-1000),
        qkv_format="thd",
    )
    return result
