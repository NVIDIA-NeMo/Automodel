# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import torch


def process_input_for_thd(
    batch: dict[str, torch.Tensor],
    seq_lens_padding_value: int = -1000,
    padding_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Process inputs for THD (total, hidden, depth) format.

    This function converts batched inputs from BSHD (batch, sequence, hidden, depth) format
    to THD format for packed sequence processing. In THD format, the batch dimension is
    collapsed and all sequences are concatenated along the sequence dimension. This supports
    both 2D token IDs and 3D embeddings for pipeline parallelism scenarios.

    The function filters out padding values in seq_lens and seq_lens_padded (indicated by
    seq_lens_padding_value) and computes cumulative sequence lengths for efficient attention
    computation with Transformer Engine or other packed sequence implementations.

    Args:
        batch: Dictionary containing:
            - 'input_ids': Input tensor of shape [batch_size, seq_len] for token IDs or
                [batch_size, seq_len, hidden_dim] for embeddings (in pipeline parallel scenarios)
            - 'labels': Labels tensor of shape [batch_size, seq_len]
            - 'position_ids': Position IDs tensor of shape [batch_size, seq_len] (required)
            - 'seq_lens': Sequence lengths tensor of shape [batch_size, num_packs] containing
                actual sequence lengths (excluding padding/separators). Values matching
                seq_lens_padding_value indicate padding and are filtered out.
            - 'seq_lens_padded': Padded sequence lengths tensor of shape [batch_size, num_packs]
                containing lengths including separator tokens. Values matching
                seq_lens_padding_value indicate padding and are filtered out.
        seq_lens_padding_value: Value used to indicate padding in seq_lens/seq_lens_padded
            tensors that should be filtered out (default: -1000)
        padding_token_id: Token ID used for padding in input_ids to generate padding_mask (default: 0)

    Returns:
        Dictionary containing:
            - 'input_ids': Reshaped tensor of shape [total_tokens] for 2D token IDs or
                [total_tokens, hidden_dim] for 3D embeddings
            - 'labels': Reshaped labels tensor of shape [total_tokens]
            - 'position_ids': Reshaped tensor of shape [total_tokens]
            - 'cu_seqlens': Cumulative padded sequence lengths tensor of shape [num_sequences + 1] (int32)
                where num_sequences is the total count of non-padded sequences across the batch.
                NOTE: This contains cumulative lengths from seq_lens_padded (not seq_lens) since
                CP doesn't support padding between sequences (resulting in NaNs). The labels or loss mask
                will ensure that loss is computed correctly.
            - 'padding_mask': Boolean tensor of shape [total_tokens] indicating padding positions
            - Non-tensor keys from input batch are preserved (e.g., 'qkv_format')

    Example:
        >>> batch_size, seq_len = 2, 6
        >>> # 2D Token IDs case with packed sequences
        >>> batch = {
        ...     'input_ids': torch.tensor([[1, 2, 3, 99, 4, 5], [6, 7, 8, 9, 10, 11]]),
        ...     'labels': torch.tensor([[2, 3, 99, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
        ...     'position_ids': torch.tensor([[0, 1, 2, 0, 0, 1], [0, 1, 2, 3, 4, 5]]),
        ...     'seq_lens': torch.tensor([[3, 2], [6, -1000]]),  # Second batch has only 1 sequence
        ...     'seq_lens_padded': torch.tensor([[4, 2], [6, -1000]])
        ... }
        >>>
        >>> result = process_input_for_thd(batch)
        >>> # result['input_ids'].shape: [12] (2D input collapsed to 1D)
        >>> # result['labels'].shape: [12]
        >>> # result['position_ids'].shape: [12]
        >>> # result['cu_seqlens']: tensor([0, 4, 6, 12], dtype=torch.int32)
        >>> #   Breakdown: [0] + cumsum([4, 2, 6]) = [0, 4, 6, 12] (from seq_lens_padded)
        >>> # result['padding_mask'].shape: [12]
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    position_ids = batch["position_ids"]
    seq_lens = batch["seq_lens"]
    seq_lens_padded = batch["seq_lens_padded"]

    # Reshape to THD format: collapse batch dimension
    # Get total number of tokens from input_ids
    batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
    total_tokens = batch_size * seq_len

    position_ids_thd = position_ids.reshape(-1) if position_ids is not None else None
    input_ids_thd = input_ids.reshape(total_tokens, -1).squeeze(-1)
    labels_thd = labels.reshape(total_tokens, -1).squeeze(-1)

    cu_seqlens = None
    cu_seqlens_padded = None
    max_seqlen = None
    if seq_lens is not None:
        # Filter out padding values and flatten
        # seq_lens shape: [batch_size, num_packs] -> flatten and remove padding values
        seq_lens_flat = seq_lens.reshape(-1)
        valid_seq_lens = seq_lens_flat[seq_lens_flat != seq_lens_padding_value]

        # Compute cumulative sequence lengths for attention
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=valid_seq_lens.dtype, device=valid_seq_lens.device),
                torch.cumsum(valid_seq_lens, dim=0),
            ]
        )
        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(device=valid_seq_lens.device)
        # NOTE: max_seqlen is computed AFTER the trailing-pad absorption below,
        # from the final cu_seqlens, so it satisfies TE's contract that
        # ``max_seqlen_q >= max(cu_seqlens_q[i+1] - cu_seqlens_q[i])``.
        # See transformer_engine/pytorch/cpp_extensions/fused_attn.py:152-159.

        if seq_lens_padded is not None:
            # Same processing for padded sequence lengths
            seq_lens_padded_flat = seq_lens_padded.reshape(-1)
            valid_seq_lens_padded = seq_lens_padded_flat[seq_lens_padded_flat != seq_lens_padding_value]

            cu_seqlens_padded = torch.cat(
                [torch.tensor([0], device=valid_seq_lens_padded.device), torch.cumsum(valid_seq_lens_padded, dim=0)]
            )
            cu_seqlens_padded = cu_seqlens_padded.to(dtype=torch.int32).to(device=valid_seq_lens_padded.device)

        # Absorb trailing pack-level pad into the last sub-sequence's
        # ``cu_seqlens`` entry — but ONLY when the padding is purely trailing
        # (i.e., CP is off / cp_size == 1).
        #
        # Two distinct kinds of padding can be in ``cu_seqlens_padded``:
        #   1. PURELY TRAILING (cp_size==1): only the last sub-sequence's
        #      length grows to absorb pack-level padding. ``cu_seqlens`` and
        #      ``cu_seqlens_padded`` differ ONLY in the last entry. We can
        #      absorb the trailing pad into ``cu_seqlens`` and avoid TE's
        #      ``pad_between_seqs=True`` code path entirely.
        #
        #   2. CP-DIVISIBILITY PADDING (cp_size>1): each sub-sequence is
        #      individually rounded up to a multiple of ``2*cp_size``
        #      (see ``packed_sequence.py:_pad_pack``). This places real
        #      padding tokens BETWEEN sub-sequences so CP can shard
        #      sequence-dim cleanly. ``cu_seqlens`` and ``cu_seqlens_padded``
        #      differ in multiple entries. Here ``pad_between_seqs=True`` is
        #      required and we must NOT collapse the two — the kernel needs
        #      both to compute attention over real tokens while reading
        #      from the padded buffer.
        #
        # Detection: only absorb when the two arrays agree on every entry
        # except the last (case 1 above). The post-absorption ``max_seqlen``
        # (computed below from the final cu_seqlens) correctly reflects the
        # absorbed slot width, so the kernel never sees a slot wider than
        # the max_seqlen we claim. Verified safe via
        # /opt/Automodel/te_bug_report/te_thd_repro_MINIMAL.py: feeding TE
        # cu_seqlens=[0,112,224,336,448,1024] with truthful max_seqlen=576
        # runs cleanly; the original captured crash was specifically caused
        # by lying (max_seqlen=112 with a 576-wide slot).
        if (
            cu_seqlens is not None
            and cu_seqlens_padded is not None
            and cu_seqlens.numel() == cu_seqlens_padded.numel()
            and cu_seqlens.numel() > 1
            and torch.equal(cu_seqlens[:-1], cu_seqlens_padded[:-1])
        ):
            _total = int(total_tokens)
            _real_total = int(cu_seqlens[-1].item())
            if _real_total < _total:
                _extended = cu_seqlens.clone()
                _extended[-1] = _total
                cu_seqlens = _extended
                # Drop ``cu_seqlens_padded`` from the emit by marking it
                # identical to ``cu_seqlens``; the emit gate below then
                # skips it, which skips the ``pad_between_seqs=True``
                # branch in attention/utils.py.
                cu_seqlens_padded = cu_seqlens.clone()
        # CP case (cu_seqlens differs from cu_seqlens_padded in multiple
        # entries) falls through unchanged — both tensors are emitted and TE
        # uses pad_between_seqs=True correctly.

        # Compute max_seqlen from the FINAL cu_seqlens so TE's contract
        # (max_seqlen_q >= max(cu_seqlens[i+1] - cu_seqlens[i])) is satisfied.
        # The absorption above can grow cu_seqlens[-1] to total_tokens;
        # max_seqlen must reflect that growth or the cuDNN-fused-attn-bwd
        # kernel writes OOB.
        if cu_seqlens is not None and cu_seqlens.numel() > 1:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().to(dtype=torch.int32)

    # Emit cu_seqlens (unpadded — real sequence lengths) and cu_seqlens_padded
    # (with intra-pack padding included) as SEPARATE keys, plus max_seqlen.
    # Downstream consumers:
    #   * TE attention preprocess (attention/utils.py) requires both to set
    #     ``pad_between_seqs=True`` correctly when the pack has pad tokens.
    #     Without ``cu_seqlens_padded``, TE computes attention over pad
    #     positions, which has produced cuDNN err 700 in BWD on this
    #     container.
    #   * Mamba's seq_idx (nemotron_v3/layers.py:282) accepts either; the
    #     padded variant works because mamba scans the full token stream.
    #   * MTP loss (loss/mtp.py::calculate_mtp_loss) uses the unpadded
    #     ``cu_seqlens`` for per-sequence label-roll masking.
    result = {
        "input_ids": input_ids_thd,
        "position_ids": position_ids_thd,
        "cu_seqlens": cu_seqlens,
        "labels": labels_thd,
        "padding_mask": (input_ids_thd == padding_token_id),
    }
    # Only emit cu_seqlens_padded when it actually differs from cu_seqlens
    # (i.e., when padding exists between sub-sequences). After the trailing-
    # pad absorption above, cu_seqlens_padded == cu_seqlens in the common
    # case, and emitting it would force TE's ``pad_between_seqs=True``
    # branch unnecessarily — which TE flash-attn handles by returning only
    # the cu_seqlens-defined real tokens, breaking the residual addition
    # downstream.
    if cu_seqlens_padded is not None and not torch.equal(cu_seqlens_padded, cu_seqlens):
        result["cu_seqlens_padded"] = cu_seqlens_padded
    if max_seqlen is not None:
        result["max_seqlen"] = max_seqlen

    # Preserve qkv_format and other non-tensor keys from the original batch
    for key, value in batch.items():
        if key not in result and not isinstance(value, torch.Tensor):
            result[key] = value

    return result


def split_batch_into_thd_chunks(
    batch: dict[str, torch.Tensor],
    num_chunks: int,
    seq_lens_padding_value: int = -1000,
    padding_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Process inputs for THD format by splitting batch into chunks for context parallelism.

    This function splits the batch along the batch dimension into num_chunks chunks,
    processes each chunk with process_input_for_thd, and stacks the tensor results.
    This is useful for context parallelism where different chunks are processed on
    different devices/ranks.

    The cu_seqlens tensors from different chunks may have different lengths depending on
    the number of sequences in each chunk. These are padded with seq_lens_padding_value
    to ensure uniform length across chunks for stacking.

    Args:
        batch: Dictionary containing input tensors with same structure as process_input_for_thd:
            - 'input_ids': [batch_size, seq_len] or [batch_size, seq_len, hidden_dim]
            - 'labels': [batch_size, seq_len]
            - 'position_ids': [batch_size, seq_len] (required)
            - 'seq_lens': [batch_size, num_packs]
            - 'seq_lens_padded': [batch_size, num_packs]
        num_chunks: Number of chunks to split the batch into. Must evenly divide batch_size.
            If num_chunks <= 1, returns the result from process_input_for_thd directly.
        seq_lens_padding_value: Value used to indicate padding in seq_lens/seq_lens_padded
            tensors and for padding cu_seqlens to uniform length (default: -1000)
        padding_token_id: Token ID used for padding in input_ids to generate padding_mask (default: 0)

    Returns:
        Dictionary containing:
        - When num_chunks > 1:
            - 'input_ids': [num_chunks, tokens_per_chunk] or [num_chunks, tokens_per_chunk, hidden_dim]
            - 'labels': [num_chunks, tokens_per_chunk]
            - 'position_ids': [num_chunks, tokens_per_chunk]
            - 'cu_seqlens': [num_chunks, max_sequences_per_chunk + 1] (padded with seq_lens_padding_value).
                Contains cumulative lengths from seq_lens_padded for CP compatibility.
            - 'padding_mask': [num_chunks, tokens_per_chunk]
            - Non-tensor keys from input batch are preserved
        - When num_chunks <= 1:
            Returns the same format as process_input_for_thd (no chunk dimension)

    Example:
        >>> batch_size, seq_len = 4, 6
        >>> batch = {
        ...     'input_ids': torch.tensor([[1,2,3,4,5,6], [7,8,9,10,11,12],
        ...                                [13,14,15,16,17,18], [19,20,21,22,23,24]]),
        ...     'labels': torch.tensor([[2,3,4,5,6,7], [8,9,10,11,12,13],
        ...                            [14,15,16,17,18,19], [20,21,22,23,24,25]]),
        ...     'position_ids': torch.tensor([[0,1,2,3,4,5], [0,1,2,3,4,5],
        ...                                   [0,1,2,3,4,5], [0,1,2,3,4,5]]),
        ...     'seq_lens': torch.tensor([[6], [6], [6], [6]]),
        ...     'seq_lens_padded': torch.tensor([[6], [6], [6], [6]]),
        ... }
        >>>
        >>> result = split_batch_into_thd_chunks(batch, num_chunks=2)
        >>> # result['input_ids'].shape: [2, 12] (2 chunks, each with 2*6=12 tokens)
        >>> # result['cu_seqlens'].shape: [2, 3] (2 chunks, each with [0, 6, 12])
        >>> # result['cu_seqlens'][0]: tensor([0, 6, 12], dtype=torch.int32)
        >>> # result['cu_seqlens'][1]: tensor([0, 6, 12], dtype=torch.int32)
    """
    if num_chunks <= 1:
        return process_input_for_thd(batch, seq_lens_padding_value, padding_token_id)

    def pad_and_stack(tensor_list, padding_value):
        """Pad tensors to same length and stack them."""
        max_len = max(len(t) for t in tensor_list)
        padded = []
        for t in tensor_list:
            if len(t) < max_len:
                pad = torch.full((max_len - len(t),), padding_value, dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad])
            padded.append(t)
        return torch.stack(padded)

    chunk_size = batch["input_ids"].shape[0] // num_chunks

    # Process all chunks
    chunk_results = [
        process_input_for_thd(
            {
                k: v[i * chunk_size : (i + 1) * chunk_size] if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            },
            seq_lens_padding_value,
            padding_token_id,
        )
        for i in range(num_chunks)
    ]

    # Stack results
    stacked: dict = {
        "input_ids": torch.stack([c["input_ids"] for c in chunk_results]),
        "labels": torch.stack([c["labels"] for c in chunk_results]),
        "position_ids": torch.stack([c["position_ids"] for c in chunk_results]),
        "cu_seqlens": pad_and_stack([c["cu_seqlens"] for c in chunk_results], seq_lens_padding_value),
        "padding_mask": torch.stack([c["padding_mask"] for c in chunk_results]),
    }
    # Stack ``cu_seqlens_padded`` whenever ANY chunk emits it (i.e. did not
    # absorb its trailing pack-pad). For chunks that did absorb (no padded
    # variant in their result), fall back to ``cu_seqlens`` — those rows have
    # cu_seqlens_padded == cu_seqlens semantically. This keeps the stacked
    # tensor rectangular and lets per-microbatch TE attention consumers see
    # consistent kwargs across chunks.
    if any("cu_seqlens_padded" in c for c in chunk_results):
        stacked["cu_seqlens_padded"] = pad_and_stack(
            [c.get("cu_seqlens_padded", c["cu_seqlens"]) for c in chunk_results],
            seq_lens_padding_value,
        )
    if all("max_seqlen" in c for c in chunk_results):
        stacked["max_seqlen"] = torch.stack([c["max_seqlen"] for c in chunk_results])
    stacked.update({k: v for k, v in chunk_results[0].items() if not isinstance(v, torch.Tensor)})
    return stacked
