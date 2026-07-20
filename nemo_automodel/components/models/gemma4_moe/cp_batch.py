# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Gemma4's contiguous-shard context-parallel batch sharding.

Contiguously shards the sequence across CP ranks (each rank keeps one
``seq_start:seq_end`` slice) so Gemma4 can run its own p2p ring FlexAttention
over the shards. It performs no collective -- the transport lives in Gemma4's
attention (see ``cp_attention.py``).

The generic slicing lives in ``components/distributed/cp_sharder.py``; this
module owns everything Gemma4-specific: the ``_packed_seq_ids`` synthesis its
manual CP attention mask builder needs, and the Gemma4 per-token batch keys
(``mm_token_type_ids``, ``per_layer_inputs``) that must be padded and sharded
alongside the sequence. Gemma4's ``prepare_model_inputs_for_cp`` exposes it
through the ``ContextParallelismSharder`` it returns under the ``"cp_sharder"`` batch key,
which the CP dispatch invokes in place of the default
load-balanced ``context_parallel`` path.
"""

from typing import Any

import torch

from nemo_automodel.components.distributed.cp_sharder import (
    convert_attention_mask_to_padding_mask,
    shard_batch_aux_only_contiguous,
    shard_batch_contiguous,
)

# Gemma4 per-token batch keys sharded alongside the sequence, with their seq
# dims and pad sentinels (0 keeps MoE routers and mask builders inert on pads).
_GEMMA4_SEQ_KEYS: dict[str, int] = {
    "mm_token_type_ids": 1,
    "per_layer_inputs": 1,
    "_packed_seq_ids": 1,
}
_GEMMA4_PAD_VALUES: dict[str, Any] = {
    "mm_token_type_ids": 0,
    "per_layer_inputs": 0,
    "_packed_seq_ids": 0,
}


def _synthesize_single_document_seq_ids(batch: dict, seq_len: int) -> None:
    """Materialize the trivial single-document ``_packed_seq_ids`` map.

    Collates emit ``_packed_seq_ids`` only when 2+ documents are packed, but
    Gemma4's manual CP attention mask builder needs document boundaries even
    for one document (1 = real token, 0 = pad). Derived from ``padding_mask``
    when present, else all-ones. A no-op when ``_packed_seq_ids`` already
    exists (genuinely packed input).

    Args:
        batch: The CP batch dict; mutated in place to add ``_packed_seq_ids``.
        seq_len: The pre-pad sequence length.
    """
    if "_packed_seq_ids" in batch:
        return
    primary = batch["inputs_embeds"] if "inputs_embeds" in batch else batch["input_ids"]
    padding_mask = batch.get("padding_mask")
    if padding_mask is not None:
        # padding_mask is [B, S], True == pad -> real-token map is its inverse.
        batch["_packed_seq_ids"] = (~padding_mask.bool()).to(torch.long)
    else:
        batch["_packed_seq_ids"] = torch.ones((primary.shape[0], seq_len), dtype=torch.long, device=primary.device)


def make_contiguous_shard_cp_batch_and_ctx(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    extra_seq_keys: dict[str, int] | None = None,
    extra_pad_values: dict[str, Any] | None = None,
):
    """Prepare and contiguously shard a batch for Gemma4's ring CP.

    Exposed as ``ContextParallelismSharder.shard_batch`` by Gemma4's ``_cp_shard_batch``;
    the CP dispatch invokes it. Synthesizes the
    ``_packed_seq_ids`` boundary map, then delegates to the shared contiguous
    sharder with Gemma4's per-token keys, keeping one contiguous sequence
    slice per CP rank (no collective; the transport lives in Gemma4's own
    ring attention).
    """
    # Derive padding_mask before the synthesis reads it; the shared prep's own
    # conversion is idempotent afterwards.
    convert_attention_mask_to_padding_mask(batch)
    primary = batch["inputs_embeds"] if "inputs_embeds" in batch else batch["input_ids"]
    _synthesize_single_document_seq_ids(batch, primary.shape[1])
    return shard_batch_contiguous(
        cp_mesh,
        tp_mesh,
        batch,
        loss_mask=loss_mask,
        padding_token_id=padding_token_id,
        extra_seq_keys={**_GEMMA4_SEQ_KEYS, **(extra_seq_keys or {})},
        extra_pad_values={**_GEMMA4_PAD_VALUES, **(extra_pad_values or {})},
    )


def make_contiguous_aux_only_shard_cp_batch_and_ctx(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    extra_seq_keys: dict[str, int] | None = None,
    extra_pad_values: dict[str, Any] | None = None,
):
    """Aux-only contiguous CP shard for Gemma4's sunk (in-forward) pre-embed.

    The sunk peer of :func:`make_contiguous_shard_cp_batch_and_ctx`: exposed as
    ``ContextParallelismSharder.shard_batch`` by Gemma4's sharder-only
    ``prepare_model_inputs_for_cp``. It shards only the no-grad auxiliary streams
    (``labels`` / ``position_ids`` / ``loss_mask`` / ``padding_mask`` plus the
    synthesized ``_packed_seq_ids`` document map) and leaves ``input_ids`` /
    ``pixel_values`` / ``mm_token_type_ids`` FULL-length in the batch. The model
    forward then embeds, splices vision, builds ``per_layer_inputs`` and the
    ``_gemma4_vision_group_ids`` / ``mm_token_type_ids`` ring metadata on the full
    sequence and contiguously slices them per microbatch (see
    ``slice_sequence_for_cp_contiguous``), so the embeddings and vision tower are
    trainable under CP and the PP×CP shared pre-embed graph no longer exists.

    ``_packed_seq_ids`` is synthesized full (from the full ``padding_mask``) and
    sharded here rather than in the forward: its pad-region zeros depend on the
    global pad tail, which the forward — holding only this rank's slice — cannot
    reconstruct. Every other flex-ring mask input the forward owns is a pure
    per-token or cumsum-over-full-then-slice quantity, so it slices to the same
    contiguous layout this sharder applies.
    """
    convert_attention_mask_to_padding_mask(batch)
    primary = batch["inputs_embeds"] if "inputs_embeds" in batch else batch["input_ids"]
    _synthesize_single_document_seq_ids(batch, primary.shape[1])
    return shard_batch_aux_only_contiguous(
        cp_mesh,
        tp_mesh,
        batch,
        loss_mask=loss_mask,
        padding_token_id=padding_token_id,
        extra_seq_keys={"_packed_seq_ids": 1, **(extra_seq_keys or {})},
        extra_pad_values={"_packed_seq_ids": 0, **(extra_pad_values or {})},
    )
