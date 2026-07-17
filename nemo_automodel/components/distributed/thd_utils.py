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


def _padding_mask_from_packed_lengths(
    seq_lens: torch.Tensor,
    seq_lens_padded: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    seq_lens_padding_value: int,
) -> torch.Tensor:
    """Build a token padding mask from packed-sequence slot metadata."""
    if seq_lens.shape != seq_lens_padded.shape:
        raise ValueError(
            "seq_lens and seq_lens_padded must have identical shapes; "
            f"got {tuple(seq_lens.shape)} and {tuple(seq_lens_padded.shape)}."
        )
    if seq_lens.device != seq_lens_padded.device:
        raise ValueError("seq_lens and seq_lens_padded must be on the same device.")
    if seq_lens.numel() % batch_size != 0:
        raise ValueError(
            "Packed sequence lengths with shape "
            f"{tuple(seq_lens.shape)} cannot be reshaped for batch size {batch_size}."
        )

    real_lens = seq_lens.reshape(batch_size, -1)
    padded_lens = seq_lens_padded.reshape(batch_size, -1)
    real_valid = real_lens != seq_lens_padding_value
    padded_valid = padded_lens != seq_lens_padding_value
    valid = real_valid & padded_valid
    lengths_are_valid = (
        (real_valid == padded_valid)
        & (~valid | ((real_lens >= 0) & (padded_lens >= real_lens)))
        & ~(valid & (~valid).to(torch.int32).cumsum(dim=1).gt(0))
    ).all()
    padded_totals = torch.where(valid, padded_lens, 0).sum(dim=1)
    padded_totals_are_valid = padded_totals.le(seq_len).all()
    if not bool((lengths_are_valid & padded_totals_are_valid).item()):
        raise ValueError(
            "Packed lengths must be non-negative, have matching trailing sentinels, satisfy "
            "seq_lens <= seq_lens_padded, and fit within the input sequence length."
        )

    real_lens = torch.where(valid, real_lens, 0)
    padded_lens = torch.where(valid, padded_lens, 0)

    slot_starts = padded_lens.cumsum(dim=1) - padded_lens
    pad_starts = (slot_starts + real_lens).clamp(min=0, max=seq_len)
    pad_ends = (slot_starts + padded_lens).clamp(min=0, max=seq_len)
    has_padding = valid & (padded_lens > real_lens)

    # Mark each [pad_start, pad_end) interval through a difference array. This
    # avoids materializing a [batch, num_sequences, sequence] comparison tensor.
    deltas = torch.zeros((batch_size, seq_len + 1), dtype=torch.int32, device=seq_lens.device)
    updates = has_padding.to(torch.int32)
    deltas.scatter_add_(1, pad_starts.to(torch.long), updates)
    deltas.scatter_add_(1, pad_ends.to(torch.long), -updates)
    trailing_updates = padded_totals.lt(seq_len).to(torch.int32)
    deltas.scatter_add_(1, padded_totals.to(torch.long).unsqueeze(1), trailing_updates.unsqueeze(1))
    deltas[:, seq_len] -= trailing_updates
    return deltas[:, :-1].cumsum(dim=1).gt(0)


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
            - 'padding_mask': Optional explicit boolean mask of shape [batch_size, seq_len].
                When absent, padding is derived from seq_lens and seq_lens_padded.
        seq_lens_padding_value: Value used to indicate padding in seq_lens/seq_lens_padded
            tensors that should be filtered out (default: -1000)
        padding_token_id: Fallback token ID used to generate padding_mask only when neither
            an explicit mask nor packed-length metadata is available (default: 0)

    Returns:
        Dictionary containing:
            - 'input_ids': Reshaped tensor of shape [total_tokens] for 2D token IDs or
                [total_tokens, hidden_dim] for 3D embeddings
            - 'labels': Reshaped labels tensor of shape [total_tokens]
            - 'position_ids': Reshaped tensor of shape [total_tokens]
            - 'cu_seqlens': Cumulative REAL sequence lengths tensor of shape [num_sequences + 1] (int32)
                where num_sequences is the total count of non-padded sequences across the batch.
                Built from seq_lens (the unpadded real lengths). When the trailing pack-pad is
                purely at the end (cp_size == 1), the last entry is grown to total_tokens to absorb
                that pad and avoid TE's ``pad_between_seqs=True`` path; see the absorption block in
                the function body for the gate.
            - 'cu_seqlens_padded': (optional) Cumulative PADDED sequence lengths tensor of the same
                shape as ``cu_seqlens``. Only emitted when it differs from ``cu_seqlens`` after
                absorption (i.e., when padding lives between sub-sequences, which is the CP case).
                Forwarded to TE as ``cu_seqlens_q_padded`` / ``cu_seqlens_kv_padded`` with
                ``pad_between_seqs=True`` so the kernel reads memory offsets from the padded
                variant while attending only over the real-length slots.
            - 'max_seqlen': Scalar int32 tensor equal to ``max(cu_seqlens[i+1] - cu_seqlens[i])``
                after any absorption. Honors TE's contract that
                ``max_seqlen_q >= max(cu_seqlens_q[i+1] - cu_seqlens_q[i])``.
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
        >>> # result['cu_seqlens']: tensor([0, 3, 5, 11], dtype=torch.int32)
        >>> #   Breakdown: [0] + cumsum([3, 2, 6]) = [0, 3, 5, 11] (from seq_lens — real lengths)
        >>> # result['cu_seqlens_padded']: tensor([0, 4, 6, 12], dtype=torch.int32)
        >>> #   Breakdown: [0] + cumsum([4, 2, 6]) = [0, 4, 6, 12] (from seq_lens_padded)
        >>> # result['max_seqlen']: tensor(6, dtype=torch.int32)  # max slot width in cu_seqlens
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
        seq_lens_flat = seq_lens.reshape(-1)
        valid_seq_lens = seq_lens_flat[seq_lens_flat != seq_lens_padding_value]

        cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=valid_seq_lens.dtype, device=valid_seq_lens.device),
                torch.cumsum(valid_seq_lens, dim=0),
            ]
        )
        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(device=valid_seq_lens.device)

        if seq_lens_padded is not None:
            seq_lens_padded_flat = seq_lens_padded.reshape(-1)
            valid_seq_lens_padded = seq_lens_padded_flat[seq_lens_padded_flat != seq_lens_padding_value]

            cu_seqlens_padded = torch.cat(
                [torch.tensor([0], device=valid_seq_lens_padded.device), torch.cumsum(valid_seq_lens_padded, dim=0)]
            )
            cu_seqlens_padded = cu_seqlens_padded.to(dtype=torch.int32).to(device=valid_seq_lens_padded.device)

        # Trailing-only pack-pad (cp_size==1): absorb into cu_seqlens[-1] so
        # the emit gate below drops cu_seqlens_padded and TE skips its
        # pad_between_seqs=True path. CP>1 differs in multiple entries and
        # falls through; both arrays are emitted and TE handles padding.
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
                cu_seqlens_padded = cu_seqlens.clone()

        # Compute max_seqlen from the FINAL cu_seqlens to honor TE's contract
        # (``max_seqlen_q >= max(cu_seqlens[i+1] - cu_seqlens[i])``, see TE's
        # cpp_extensions/fused_attn.py:152-159).
        if cu_seqlens is not None and cu_seqlens.numel() > 1:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().to(dtype=torch.int32)

    explicit_padding_mask = batch.get("padding_mask")
    if explicit_padding_mask is not None:
        if explicit_padding_mask.numel() != total_tokens:
            raise ValueError(
                "padding_mask must contain one value per input token; "
                f"got shape {tuple(explicit_padding_mask.shape)} for input shape {tuple(input_ids.shape)}."
            )
        padding_mask_thd = explicit_padding_mask.reshape(total_tokens).bool()
    elif seq_lens is not None and seq_lens_padded is not None:
        padding_mask_thd = _padding_mask_from_packed_lengths(
            seq_lens,
            seq_lens_padded,
            batch_size=batch_size,
            seq_len=seq_len,
            seq_lens_padding_value=seq_lens_padding_value,
        ).reshape(total_tokens)
    else:
        padding_mask_thd = input_ids_thd == padding_token_id

    result = {
        "input_ids": input_ids_thd,
        "position_ids": position_ids_thd,
        "cu_seqlens": cu_seqlens,
        "labels": labels_thd,
        "padding_mask": padding_mask_thd,
    }
    # Emit cu_seqlens_padded only when it differs from cu_seqlens — its
    # presence is what flips TE's pad_between_seqs=True path in
    # attention/utils.py.
    if cu_seqlens_padded is not None and not torch.equal(cu_seqlens_padded, cu_seqlens):
        result["cu_seqlens_padded"] = cu_seqlens_padded
    if max_seqlen is not None:
        result["max_seqlen"] = max_seqlen

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
            - 'cu_seqlens': [num_chunks, max_sequences_per_chunk + 1] (right-padded with
                seq_lens_padding_value across chunks for rectangularity). Built from seq_lens
                (real lengths) per chunk; see ``process_input_for_thd`` for the absorption
                semantics applied per chunk.
            - 'cu_seqlens_padded': (optional) Same shape, emitted whenever ANY chunk emits it.
                For chunks that absorbed (no separate padded variant), this row equals the
                chunk's ``cu_seqlens``.
            - 'max_seqlen': [num_chunks] per-chunk scalar tensor.
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

    stacked: dict = {
        "input_ids": torch.stack([c["input_ids"] for c in chunk_results]),
        "labels": torch.stack([c["labels"] for c in chunk_results]),
        "position_ids": torch.stack([c["position_ids"] for c in chunk_results]),
        "cu_seqlens": pad_and_stack([c["cu_seqlens"] for c in chunk_results], seq_lens_padding_value),
        "padding_mask": torch.stack([c["padding_mask"] for c in chunk_results]),
    }
    # Emit cu_seqlens_padded whenever any chunk emits it; absorbed chunks
    # fall back to their cu_seqlens (semantically equal) for rectangularity.
    if any("cu_seqlens_padded" in c for c in chunk_results):
        stacked["cu_seqlens_padded"] = pad_and_stack(
            [c.get("cu_seqlens_padded", c["cu_seqlens"]) for c in chunk_results],
            seq_lens_padding_value,
        )
    if all("max_seqlen" in c for c in chunk_results):
        stacked["max_seqlen"] = torch.stack([c["max_seqlen"] for c in chunk_results])
    stacked.update({k: v for k, v in chunk_results[0].items() if not isinstance(v, torch.Tensor)})
    return stacked
