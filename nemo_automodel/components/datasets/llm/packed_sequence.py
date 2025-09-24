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

import logging

import torch
from datasets import Dataset
from torch.nn import functional as F

logger = logging.getLogger(__name__)

CROSS_ENTROPY_IGNORE_IDX = -100
PACK_TYPE = dict[str, torch.Tensor | list[int]]


# based on https://github.com/pytorch/torchtune/blob/v0.6.1/torchtune/datasets/_packed.py#L17

def _fill_labels_with_cross_entropy_ignore_idx(labels: list[int], loss_mask: list[int]) -> list[int]:
    for i, mask in enumerate(loss_mask):
        if mask == 0:
            labels[i] = CROSS_ENTROPY_IGNORE_IDX
    return labels


def _pad_pack(pack: PACK_TYPE, padding_idx: int, packed_sequence_size: int, cross_entropy_ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX) -> PACK_TYPE:
    """
    Pads a pack to ``packed_sequence_size``.
    """
    # Pad tokens
    num_padding_tokens = packed_sequence_size - len(pack["input_ids"])
    padded_tokens = F.pad(
        pack["input_ids"],
        (0, num_padding_tokens),
        value=padding_idx,
    )

    # Pad labels
    padded_labels = F.pad(
        pack["labels"],
        (0, packed_sequence_size - len(pack["labels"])),
        value=cross_entropy_ignore_idx,
    )

    # Add padding tokens as a last seq len to ensure sum is packed_sequence_size
    padded_seq_lens = (
        torch.cat([pack["seq_lens"], torch.tensor([num_padding_tokens])])
        if num_padding_tokens > 0
        else pack["seq_lens"]
    )

    # Pad position_ids continuing the sequence from last value
    # in position_ids
    # e.g. [0 1 2] -> [0 1 2 3 4 5] for packed_sequence_size = 6
    num_range = torch.arange(
        pack["position_ids"][-1] + 1,
        pack["position_ids"][-1] + packed_sequence_size - len(pack["position_ids"]) + 1,
    )
    # Clamp to packed_sequence_size - 1 to avoid out of bounds error
    clamped_num_range = torch.clamp(num_range, 0, packed_sequence_size - 1)
    padded_position_ids = torch.cat([pack["position_ids"], clamped_num_range])

    padded_pack = {
        "input_ids": padded_tokens,
        "labels": padded_labels,
        "position_ids": padded_position_ids,
        "seq_lens": padded_seq_lens,
    }
    return padded_pack


def _convert_to_tensors(pack: PACK_TYPE) -> PACK_TYPE:
    """
    Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors.
    """
    tensor_pack = {
        "input_ids": torch.tensor(pack["input_ids"], dtype=torch.long),
        "labels": torch.tensor(pack["labels"], dtype=torch.long),
        "position_ids": torch.tensor(pack["position_ids"], dtype=torch.long),
        "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long),
    }
    return tensor_pack

def _tensorize_and_pad_pack(pack: PACK_TYPE, padding_idx: int, packed_sequence_size: int, cross_entropy_ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX) -> None:
    """
    converts to tensors, pads a pack and returns it.
    """
    pack = _convert_to_tensors(pack)
    pack = _pad_pack(pack, padding_idx=padding_idx, packed_sequence_size=packed_sequence_size, cross_entropy_ignore_idx=cross_entropy_ignore_idx)
    return pack

def _should_stop_packing(max_packs: int, packs: list[PACK_TYPE]) -> bool:
    """
    If max packs is set, stop packing when we reach that number.
    """
    if max_packs is not None and len(packs) == max_packs:
        return True
    return False


def _calculate_leftover_seq_len(current_pack: PACK_TYPE, split_across_pack, previous_sample_boundary, packed_sequence_size) -> int:
    if split_across_pack:
        boundary = packed_sequence_size
        # The last elem in ``seq_lens`` ensures that ``sum(seq_lens) == packed_sequence_size``
        leftover_seq_len = packed_sequence_size - sum(current_pack["seq_lens"][:-1])
        seq_len_padding = [leftover_seq_len] if leftover_seq_len > 0 else []
    else:
        boundary = previous_sample_boundary
        # If we aren't splitting across packs, we leave out the last sample b/c
        # it will go into the next pack
        seq_len_padding = []
    return boundary, seq_len_padding


def _split_and_add_pack(current_pack: PACK_TYPE, packs: list[PACK_TYPE], split_across_pack: bool, previous_sample_boundary: int, packed_sequence_size: int, padding_idx: int, cross_entropy_ignore_idx=CROSS_ENTROPY_IGNORE_IDX) -> PACK_TYPE:
    """
    Splits the current pack at the boundary, processes it, adds it to ``packs``.

    ...and returns the start of the next pack.

    TODO(@akoumparouli): refactor.
    """
    boundary, seq_len_padding = _calculate_leftover_seq_len(
        current_pack,
        split_across_pack,
        previous_sample_boundary,
        packed_sequence_size,
    )

    pack = {
        "input_ids": current_pack["input_ids"][:boundary],
        "labels": current_pack["labels"][:boundary],
        "position_ids": current_pack["position_ids"][:boundary],
        "seq_lens": current_pack["seq_lens"][:-1] + seq_len_padding,
    }

    # Process and add the pack
    packs.append(_tensorize_and_pad_pack(
        pack,
        padding_idx=padding_idx,
        packed_sequence_size=packed_sequence_size,
        cross_entropy_ignore_idx=cross_entropy_ignore_idx,
    ))

    # Return the length of the first sample in next pack if we are splitting across packs,
    # otherwise return the length of the last sample in the current pack
    next_seq_len = (
        len(current_pack["input_ids"][boundary:])
        if split_across_pack else
        current_pack["seq_lens"][-1]
    )

    output_dict = {
        "input_ids": current_pack["input_ids"][boundary:],
        "labels": current_pack["labels"][boundary:],
        "position_ids": current_pack["position_ids"][boundary:],
        "seq_lens": [next_seq_len],
    }
    return output_dict


def pack_dataset(dataset, split=None, packed_sequence_size, split_across_pack=False, max_packs=None, padding_idx=0):
    """
    Pack the dataset to defined length.

    In particulat, it will iterate through the dataset. Use a buffer to hold samples until
    packed_sequence_size, then append the buffer to packs as a single "packed" sample.
    Continue until max_packs or end of dataset.

    Args:
        dataset: Actual dataset (can be 'train', 'val' or 'test')
        split (str): Whether the dataset is 'train', 'val' or 'test'
        packed_sequence_size (int): Number of tokens in a pack
        split_across_pack (bool): If the last sample in a pack does not fit in
            ``packed_sequence_size``, split the sample into the next pack, or move it entirely
            to the beginning of the next pack. Default: False
        max_packs (int): Maximum number of packs. Default: None
    """
    packs: list[PACK_TYPE] = []
    if split is not None:
        dataset = dataset[split]

    # Buffer to hold samples until they are long enough to be added to packs
    current_pack = {
        "input_ids": [],
        "labels": [],
        "position_ids": [],
        "seq_lens": [],
    }
    previous_sample_boundary: int = 0

    for sample in dataset:
        input_ids, labels = sample["input_ids"], sample["labels"]
        if loss_mask := sample.pop("loss_mask", None):
            labels = _fill_labels_with_cross_entropy_ignore_idx(labels, loss_mask)
        # If the dataset outputs samples that are larger than the specified
        # packed_sequence_size and we're unable to split it, user needs to modify
        # one of the two parameters
        seq_len = len(input_ids)
        if seq_len > packed_sequence_size and not split_across_pack:
            raise ValueError(
                f"Dataset sample is too long ({seq_len} > {packed_sequence_size}). "
                "Please set `split_across_pack=True` or increase `packed_sequence_size`.",
            )
        # Update the current pack
        # "position_ids" is the pos ids, "seq_lens" is the len of each seq within the pack
        current_pack["input_ids"] += input_ids
        current_pack["labels"] += labels
        current_pack["position_ids"] += [x % packed_sequence_size for x in range(seq_len)]
        current_pack["seq_lens"] += [seq_len]

        # If the current pack is over the packed_sequence_size, add it to packs and
        # retain any truncated or bumped samples for next pack
        while len(current_pack["input_ids"]) > packed_sequence_size and not _should_stop_packing(max_packs, packs):
            current_pack = _split_and_add_pack(
                current_pack,
                packs=packs,
                split_across_pack=split_across_pack,
                previous_sample_boundary=previous_sample_boundary,
                packed_sequence_size=packed_sequence_size,
                padding_idx=padding_idx,
                cross_entropy_ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
            )


        # Keep track of previous sample boundary
        previous_sample_boundary = len(current_pack["input_ids"])

        if _should_stop_packing(max_packs, packs):
            break

    # Handle the last pack if there's leftover and we haven't filled up the max packs
    if len(current_pack["input_ids"]) > 0 and (max_packs is None or len(packs) < max_packs):
        # No need to handle splitting at this point so we can just add the current pack
        packs.append(_tensorize_and_pad_pack(
            current_pack,
            padding_idx=padding_idx,
            packed_sequence_size=packed_sequence_size,
            cross_entropy_ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
        ))

    # After packing all samples, convert packs to a Dataset object
    logger.info("Total number of packs created: {}".format(len(packs)))
    return Dataset.from_dict(
        {key: [pack[key] for pack in packs] for key in packs[0].keys()}
    )



def create_block_causal_mask(seq_lens: list[torch.Tensor]) -> torch.Tensor:
    """
    Creates causal mask block for specified lengths.

    In particular, given a batch tensor of seq lens defining the lengths of samples in each pack,
    Construct a 2D block causal mask for each pack in the batch. For example, if
    a single sample's seq_lens is [3, 2, 1], the mask would be::
        mask = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.

    Returns:
        Tensor: Block causal mask of shape (batch_size, packed_sequence_size, packed_sequence_size).
    """
    batch_block_attn_masks = []
    batch_size = len(seq_lens)
    for sample_idx in range(batch_size):
        block_attn_masks = [
            torch.tril(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                ),
            )
            for i, seq_len in enumerate(seq_lens[sample_idx])
        ]

        batch_block_attn_masks.append(torch.block_diag(*block_attn_masks))
    # Transformers expects the attn_mask to be 4d [bs, 1, packed_sequence_size, packed_sequence_size], hence adding
    # singleton (size 1) dimension at position 1.
    return torch.stack(batch_block_attn_masks).unsqueeze(1)


def packed_block_causal_mask(seq_lens: list[torch.Tensor]):
    """
    Create a 2D block causal document mask for a batch of packed sequences.

    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.

    Returns:
        _MaskType: BlockMask or Tensor if torch version < 2.5.0.
    """
    return create_block_causal_mask(seq_lens=seq_lens)
