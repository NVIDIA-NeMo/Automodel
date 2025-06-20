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
from tqdm import tqdm


logger = logging.getLogger(__name__)

CROSS_ENTROPY_IGNORE_IDX = -100
PACK_TYPE = dict[str, torch.Tensor | list[int]]


# based on https://github.com/pytorch/torchtune/blob/v0.6.1/torchtune/datasets/_packed.py#L17
class PackedSequence:
    """
    Implements Packed Sequence for input dataset.

    Args:
        dataset: Actual dataset (can be 'train', 'val' or 'test')
        split (str): Whether the dataset is 'train', 'val' or 'test'
        packed_sequence_size (int): Number of tokens in a pack
        split_across_pack (bool): If the last sample in a pack does not fit in
            ``packed_sequence_size``, split the sample into the next pack, or move it entirely
            to the beginning of the next pack. Default: False
        max_packs (int): Maximum number of packs. Default: None
    """

    def __init__(self, dataset, split, packed_sequence_size, split_across_pack=False, max_packs=None):
        """
        Packed Sequence constructor.

        Given the dataset and the rest of the arguments, it will create (using the .pack) method
        another dataset containing packed sequences.

        Args:
            dataset: Actual dataset (can be 'train', 'val' or 'test')
            split (str): Whether the dataset is 'train', 'val' or 'test'
            packed_sequence_size (int): Number of tokens in a pack
            split_across_pack (bool): If the last sample in a pack does not fit in
                ``packed_sequence_size``, split the sample into the next pack, or move it entirely
                to the beginning of the next pack. Default: False
            max_packs (int): Maximum number of packs. Default: None
        """
        self.dataset = dataset
        self.split = split
        self.padding_idx = 0  # Padding value to pack a sequence to self.packed_sequence_size
        self.contains_loss_mask = False
        self.packed_sequence_size = packed_sequence_size
        self.split_across_pack = split_across_pack
        self.max_packs = max_packs
        self.packs: list[PACK_TYPE] = []

    # --------------------------------- public API ---------------------------------
    def pack(self) -> Dataset:
        """
        Pack the dataset to defined length.

        In particular, it will iterate through the dataset. Use a buffer to hold samples until
        packed_sequence_size, then append the buffer to self.packs as a single "packed" sample.
        Continue until max_packs or end of dataset.
        """
        self._start_packing()

        for sample in self.dataset:
            self._process_sample(sample)
            if self._should_stop_packing():
                break

        self._finalize_packing()
        return self.packed_dataset

    # ----------------------------- private helpers --------------------------------
    def _start_packing(self) -> None:
        """Prepare state and progress-bar."""
        self.contains_loss_mask = "loss_mask" in self.dataset[0]
        self.current_pack      = self._new_empty_pack()
        self.previous_sample_boundary = 0
        self.rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if self.rank == 0:
            self.pbar = tqdm(total=len(self.dataset),
                            desc=f"Packing {self.split} dataset",
                            dynamic_ncols=True)

    def _process_sample(self, sample: dict) -> None:
        """Ingest one sample, handle overflow, update progress-bar/boundary."""
        self._append_to_pack(sample)
        self._handle_overflow()

        if self.rank == 0:
            self.pbar.update()
        self.previous_sample_boundary = len(self.current_pack["input_ids"])

    def _handle_overflow(self) -> None:
        """Split current_pack while it is too large and we are still allowed to."""
        while (len(self.current_pack["input_ids"]) > self.packed_sequence_size
            and not self._should_stop_packing()):
            self.current_pack = self._split_and_add_pack(self.current_pack)

    def _append_to_pack(self, sample: dict) -> None:
        """Concatenate one sample onto current_pack."""
        ids, lbls = sample["input_ids"], sample["labels"]
        if len(ids) > self.packed_sequence_size and not self.split_across_pack:
            raise ValueError(
                f"Dataset sample is too long ({len(ids)} > {self.packed_sequence_size}). "
                "Set `split_across_pack=True` or increase `packed_sequence_size`."
            )

        self.current_pack["input_ids"]   += ids
        self.current_pack["labels"]      += lbls
        self.current_pack["position_ids"]+= [i % self.packed_sequence_size for i in range(len(ids))]
        self.current_pack["seq_lens"]    += [len(ids)]
        if self.contains_loss_mask:
            self.current_pack["loss_mask"] += sample["loss_mask"]

    def _finalize_packing(self) -> None:
        """Flush leftovers and build the final HF-Dataset."""
        if (self.current_pack["input_ids"] and
            (self.max_packs is None or len(self.packs) < self.max_packs)):
            self._add_pack(self.current_pack)

        self.packed_dataset = Dataset.from_dict(
            {k: [p[k] for p in self.packs] for k in self.packs[0]}
        )
        logger.info(f">>>>> Total number of packs created: {len(self.packs)} <<<<<")

    # -------------------- tiny utility: create empty pack dict --------------------
    def _new_empty_pack(self) -> dict:
        d = {"input_ids": [], "labels": [], "position_ids": [], "seq_lens": []}
        if getattr(self, "contains_loss_mask", False):
            d["loss_mask"] = []
        return d

    def _should_stop_packing(self) -> bool:
        """
        If max packs is set, stop packing when we reach that number.
        """
        if self.max_packs is not None and len(self.packs) == self.max_packs:
            return True
        return False

    def _split_and_add_pack(self, current_pack: PACK_TYPE) -> PACK_TYPE:
        """
        Splits the current pack at the boundary, processes it, adds it to ``self.packs``.

        ...and returns the start of the next pack.

        TODO(@akoumparouli): refactor.
        """
        if self.split_across_pack:
            boundary = self.packed_sequence_size
            # The last elem in ``seq_lens`` ensures that ``sum(seq_lens) == self.packed_sequence_size``
            leftover_seq_len = self.packed_sequence_size - sum(current_pack["seq_lens"][:-1])
            seq_len_padding = [leftover_seq_len] if leftover_seq_len > 0 else []
        else:
            boundary = self.previous_sample_boundary
            # If we aren't splitting across packs, we leave out the last sample b/c
            # it will go into the next pack
            seq_len_padding = []

        pack = {
            "input_ids": current_pack["input_ids"][:boundary],
            "labels": current_pack["labels"][:boundary],
            "position_ids": current_pack["position_ids"][:boundary],
            "seq_lens": current_pack["seq_lens"][:-1] + seq_len_padding,
        }
        if self.contains_loss_mask:
            pack["loss_mask"] = current_pack["loss_mask"][:boundary]

        # Process and add the pack
        self._add_pack(pack)

        # Return the length of the first sample in next pack if we are splitting across packs,
        # otherwise return the length of the last sample in the current pack
        next_seq_len = (
            len(current_pack["input_ids"][boundary:]) if self.split_across_pack else current_pack["seq_lens"][-1]
        )

        output_dict = {
            "input_ids": current_pack["input_ids"][boundary:],
            "labels": current_pack["labels"][boundary:],
            "position_ids": current_pack["position_ids"][boundary:],
            "seq_lens": [next_seq_len],
        }
        if self.contains_loss_mask:
            output_dict["loss_mask"] = current_pack["loss_mask"][boundary:]
        return output_dict

    def _add_pack(self, pack: PACK_TYPE) -> None:
        """
        Processes, pads and adds a pack to ``self.packs``.
        """
        pack = self._convert_to_tensors(pack)
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        self.packs.append(pack)

    def _convert_to_tensors(self, pack: PACK_TYPE) -> PACK_TYPE:
        """
        Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors.
        """
        tensor_pack = {
            "input_ids": torch.tensor(pack["input_ids"], dtype=torch.long),
            "labels": torch.tensor(pack["labels"], dtype=torch.long),
            "position_ids": torch.tensor(pack["position_ids"], dtype=torch.long),
            "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long),
        }
        if self.contains_loss_mask:
            tensor_pack["loss_mask"] = torch.tensor(pack["loss_mask"], dtype=torch.long)
        return tensor_pack

    def _pad_pack(self, pack: PACK_TYPE, padding_idx: int) -> PACK_TYPE:
        """
        Pads a pack to ``self.packed_sequence_size``.
        """
        # Pad tokens
        num_padding_tokens = self.packed_sequence_size - len(pack["input_ids"])
        padded_tokens = F.pad(
            pack["input_ids"],
            (0, num_padding_tokens),
            value=padding_idx,
        )

        # Pad labels
        padded_labels = F.pad(
            pack["labels"],
            (0, self.packed_sequence_size - len(pack["labels"])),
            value=CROSS_ENTROPY_IGNORE_IDX,
        )

        # Pad loss_mask
        if self.contains_loss_mask:
            padded_loss_mask = F.pad(
                pack["loss_mask"],
                (0, self.packed_sequence_size - len(pack["loss_mask"])),
                value=0,
            )

        # Add padding tokens as a last seq len to ensure sum is packed_sequence_size
        padded_seq_lens = (
            torch.cat([pack["seq_lens"], torch.tensor([num_padding_tokens])])
            if num_padding_tokens > 0
            else pack["seq_lens"]
        )

        # Pad position_ids continuing the sequence from last value
        # in position_ids
        # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.packed_sequence_size = 6
        num_range = torch.arange(
            pack["position_ids"][-1] + 1,
            pack["position_ids"][-1] + self.packed_sequence_size - len(pack["position_ids"]) + 1,
        )
        # Clamp to packed_sequence_size - 1 to avoid out of bounds error
        clamped_num_range = torch.clamp(num_range, 0, self.packed_sequence_size - 1)
        padded_position_ids = torch.cat([pack["position_ids"], clamped_num_range])

        padded_pack = {
            "input_ids": padded_tokens,
            "labels": padded_labels,
            "position_ids": padded_position_ids,
            "seq_lens": padded_seq_lens,
        }
        if self.contains_loss_mask:
            padded_pack["loss_mask"] = padded_loss_mask
        return padded_pack


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
