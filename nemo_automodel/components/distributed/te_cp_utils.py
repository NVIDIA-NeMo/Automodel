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


def pad_thd_sequences_for_cp(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cu_seqlens: torch.Tensor,
    divisibility_factor: int,
    padding_token_id: int = 0,
    padding_label_id: int = -100,
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads sequences to be divisible by the divisibility factor or to a maximum total length.

    Args:
        input_ids: Tensor of shape (1, N) or (N,) containing concatenated sequences
        labels: Tensor of shape (1, N) or (N,) containing labels for each token
        cu_seqlens: Tensor of shape (M,) containing cumulative sequence lengths
        divisibility_factor: Each sequence length must be divisible by this factor
        padding_token_id: Token ID to use for padding (default: 0)
        padding_label_id: Label ID to use for padding (default: -100)
        max_seq_len: Optional maximum total length for all packed sequences. If provided,
            the entire concatenated output will be padded to this length. This represents
            the total length of all sequences packed together, not the max length of an
            individual sequence. Must be >= sum of padded individual sequence lengths.

    Returns:
        Tuple of:
        - input_ids_padded: Padded input_ids tensor
        - labels_padded: Padded labels tensor
        - cu_seqlens_padded: Cumulative sequence lengths accounting for padding
    """
    # Flatten input_ids and labels if needed
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    if labels.dim() == 2:
        labels = labels.squeeze(0)

    # Compute the sequence lengths from cu_seqlens
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

    # List: amount of padding needed for each sequence (make length a multiple of divisibility_factor)
    padding_amounts = [
        ((seq_len.item() + divisibility_factor - 1) // divisibility_factor) * divisibility_factor - seq_len.item()
        for seq_len in seqlens
    ]

    # Extract sequences and labels for each batch item
    batch_sequences = [input_ids[start.item() : end.item()] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
    batch_labels = [labels[start.item() : end.item()] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]

    # Pad sequences and labels to required length
    input_ids_padded = torch.cat(
        [
            (
                torch.cat([seq, torch.full((pad,), padding_token_id, dtype=seq.dtype, device=seq.device)])
                if pad > 0
                else seq
            )
            for seq, pad in zip(batch_sequences, padding_amounts)
        ]
    )
    labels_padded = torch.cat(
        [
            (
                torch.cat([seq, torch.full((pad,), padding_label_id, dtype=seq.dtype, device=seq.device)])
                if pad > 0
                else seq
            )
            for seq, pad in zip(batch_labels, padding_amounts)
        ]
    )

    # Compute cumulative padded sequence lengths, starting from 0
    padded_lengths = seqlens + torch.tensor(padding_amounts, dtype=seqlens.dtype, device=seqlens.device)
    cu_seqlens_padded = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=cu_seqlens.dtype, device=cu_seqlens.device), padded_lengths]), dim=0
    )

    # If max_seq_len is provided, pad the entire packed sequence to that total length
    if max_seq_len is not None:
        current_total_len = input_ids_padded.shape[0]
        if max_seq_len < current_total_len:
            raise ValueError(f"max_seq_len ({max_seq_len}) must be >= total padded length ({current_total_len})")
        if max_seq_len > current_total_len:
            additional_padding = max_seq_len - current_total_len
            assert additional_padding % divisibility_factor == 0, (
                f"{additional_padding=} must be divisible by {divisibility_factor=} when providing {max_seq_len=}"
            )
            input_ids_padded = torch.cat(
                [
                    input_ids_padded,
                    torch.full(
                        (additional_padding,),
                        padding_token_id,
                        dtype=input_ids_padded.dtype,
                        device=input_ids_padded.device,
                    ),
                ]
            )
            labels_padded = torch.cat(
                [
                    labels_padded,
                    torch.full(
                        (additional_padding,), padding_label_id, dtype=labels_padded.dtype, device=labels_padded.device
                    ),
                ]
            )
            # Update cu_seqlens_padded to reflect the final total length
            cu_seqlens_padded = cu_seqlens_padded.clone()
            cu_seqlens_padded[-1] = max_seq_len

    return input_ids_padded, labels_padded, cu_seqlens_padded


def generate_positional_ids_for_cp(
    cu_seqlens: torch.Tensor,
    divisibility_factor: int,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Generate positional IDs for sequences padded to be divisible by divisibility_factor.

    Args:
        cu_seqlens: Tensor of shape (M,) containing cumulative sequence lengths
        divisibility_factor: Each sequence length must be divisible by this factor
        dtype: Data type for the generated positional IDs (default: torch.long)

    Returns:
        Generated positional_ids tensor where each sequence starts from 0 and continues through padding
    """
    # Compute the sequence lengths from cu_seqlens
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

    # List: amount of padding needed for each sequence
    padding_amounts = [
        ((seq_len.item() + divisibility_factor - 1) // divisibility_factor) * divisibility_factor - seq_len.item()
        for seq_len in seqlens
    ]

    # Generate positional IDs for each padded sequence (each starts from 0)
    padded_lengths = seqlens + torch.tensor(padding_amounts, dtype=seqlens.dtype, device=seqlens.device)
    positional_ids = torch.cat([torch.arange(0, int(length), dtype=dtype) for length in padded_lengths]).to(
        device=seqlens.device
    )

    return positional_ids


def get_batch_on_this_cp_rank(
    cu_seqlens_padded: torch.Tensor,
    input_ids_padded: torch.Tensor,
    labels_padded: torch.Tensor,
    position_ids_padded: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    qvk_format: str = "thd",
):
    """Slice batch input along sequence dimension into multiple chunks for THD format.

    This function is inteded for use in self attention. It will not work for cross attention because
    it does not handle the case where the sequence length of the query and key are different.

    Which are parallelized across GPUs in a context parallel group.
    This version works with variable-length sequences using cumulative sequence lengths.
    """
    if qvk_format not in ["thd", "bshd", "sbhd"]:
        raise ValueError(f"Unsupported qvk_format: {qvk_format}!")
    if qvk_format == "thd":
        # Get context parallel size and rank
        if cp_size > 1:
            # Calculate the chunk sizes for each sequence
            total_slices_of_any_sequence = 2 * cp_size
            slice_sizes = (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]) // total_slices_of_any_sequence

            # Process each tensor directly instead of using keys_to_change loop
            def process_tensor(val):
                if val is None:
                    return val
                # Determine which dimension is the sequence dimension
                # Ensure cu_seqlens_padded[-1] is a Python int, not a 0-dim tensor
                if isinstance(cu_seqlens_padded[-1], torch.Tensor):
                    seq_len_val = cu_seqlens_padded[-1].item()
                else:
                    seq_len_val = cu_seqlens_padded[-1]

                # Handle 1D tensors (like position_ids that don't have batch dimension)
                if val.ndim == 1:
                    if val.shape[0] == seq_len_val:
                        current_seq_dim = 0
                    else:
                        raise ValueError(
                            "1D tensor shape doesn't match expected sequence length. Make sure the"
                            " inputs are in THD format and padded correctly."
                        )
                elif val.ndim >= 2:
                    if val.shape[1] == seq_len_val:
                        current_seq_dim = 1
                    elif val.shape[0] == seq_len_val:
                        current_seq_dim = 0
                    else:
                        raise ValueError("Make sure the inputs are in THD format and padded correctly.")
                else:
                    raise ValueError("Tensor must be at least 1D")

                # On this particular rank, for each sequence, get two slices, one from the beginning
                # and one from the end.
                cp_rank_slices = []
                for slice_size, seq_start in zip(slice_sizes, cu_seqlens_padded[:-1]):
                    # 1st segment
                    cp_rank_slices.append(
                        torch.arange(
                            seq_start + (cp_rank * slice_size),
                            seq_start + ((cp_rank + 1) * slice_size),
                            device=val.device,
                        )
                    )

                    # 2nd segment
                    cp_rank_slices.append(
                        torch.arange(
                            seq_start + ((total_slices_of_any_sequence - cp_rank - 1) * slice_size),
                            seq_start + ((total_slices_of_any_sequence - cp_rank) * slice_size),
                            device=val.device,
                        )
                    )

                return val.index_select(current_seq_dim, torch.cat(cp_rank_slices))

            # Process each tensor directly
            input_ids_padded = process_tensor(input_ids_padded)
            labels_padded = process_tensor(labels_padded)
            position_ids_padded = process_tensor(position_ids_padded)
    elif qvk_format == "bshd":

        def process_tensor(val):
            if val is None:
                return val

            # Dynamically determine sequence dimension based on format
            # For bshd format: batch, sequence, heads, dim
            seq_dim = 1

            # Validate tensor has enough dimensions
            if val.ndim < 2:
                raise ValueError(f"Tensor must have at least 2 dimensions for bshd format, got {val.ndim}")

            # Validate sequence dimension is divisible by 2*cp_size
            if val.shape[seq_dim] % (2 * cp_size) != 0:
                raise ValueError(
                    f"Sequence dimension (dim {seq_dim}) with size {val.shape[seq_dim]} "
                    f"must be divisible by 2*cp_size={2 * cp_size}"
                )

            # Reshape tensor to separate chunks
            try:
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to reshape tensor from shape {list(val.shape)} to chunk-separated shape. Error: {e}"
                )

            # Create index tensor on the same device as input to avoid CPU-GPU sync
            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device, dtype=torch.long)

            # Select the chunks for this rank
            val = val.index_select(seq_dim, index)

            # Reshape back to original format with reduced sequence dimension
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
            return val

        if cp_size > 1:
            input_ids_padded = process_tensor(input_ids_padded)
            labels_padded = process_tensor(labels_padded)
            position_ids_padded = process_tensor(position_ids_padded)

    elif qvk_format == "sbhd":

        def process_tensor(val):
            if val is None:
                return val

            # Dynamically determine sequence dimension based on format
            # For sbhd format: sequence, batch, heads, dim
            seq_dim = 0

            # Validate tensor has enough dimensions
            if val.ndim < 2:
                raise ValueError(f"Tensor must have at least 2 dimensions for sbhd format, got {val.ndim}")

            # Validate sequence dimension is divisible by 2*cp_size
            if val.shape[seq_dim] % (2 * cp_size) != 0:
                raise ValueError(
                    f"Sequence dimension (dim {seq_dim}) with size {val.shape[seq_dim]} "
                    f"must be divisible by 2*cp_size={2 * cp_size}"
                )

            # Reshape tensor to separate chunks
            try:
                val = val.view(
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to reshape tensor from shape {list(val.shape)} to chunk-separated shape. Error: {e}"
                )

            # Create index tensor on the same device as input to avoid CPU-GPU sync
            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device, dtype=torch.long)

            # Select the chunks for this rank (dim 0 for sbhd after reshape)
            val = val.index_select(0, index)

            # Reshape back to original format with reduced sequence dimension
            val = val.view(-1, *val.shape[2:])
            return val

        if cp_size > 1:
            input_ids_padded = process_tensor(input_ids_padded)
            labels_padded = process_tensor(labels_padded)
            position_ids_padded = process_tensor(position_ids_padded)

    else:
        raise ValueError(f"Support not implemented yet for qvk_format: {qvk_format}!")

    return input_ids_padded, labels_padded, position_ids_padded
