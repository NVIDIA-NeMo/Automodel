# Copyright (c) 2025 NVIDIA CORPORATION.
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

import functools
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

# need to remove
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy
import torch

from nemo_automodel.datasets.llm.megatron.indexed_dataset import IndexedDataset

import json
import hashlib
from collections import OrderedDict

# taken and modified from https://github.com/NVIDIA/Megatron-LM/blob/5e798111e60f45e82c336ef7b89d8d793c93208f/megatron/core/datasets/gpt_dataset.py
logger = logging.getLogger(__name__)

_PAD_TOKEN_ID = -100


class Split(Enum):
    train = 0
    valid = 1
    test = 2


@dataclass
class BlendedMegatronDatasetConfig:
    """Configuration object for Megatron Core datasets"""

    random_seed: int
    """The seed for all RNG during dataset creation."""

    sequence_length: int
    """The sequence length."""

    blend: Optional[Tuple[List[str], Optional[List[float]]]] = None
    """The blend, consisting of a list of dataset prefixes and optionally a list of dataset
       weights. For example, [["dataset-path1", "dataset-path2"], [0.3, 0.7]]. When the weights are
       None, they are inferred from the lengths of the contributing datasets. Not to be used with
       'blend_per_split'. Defaults to None.
    """

    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]] = None
    """A set of blends, as defined above, one for each split distribution. Not to be used with
       'blend'. Defauls to None.
    """

    split: Optional[str] = None
    """The split string, a comma separated weighting for the dataset splits when drawing samples
       from a single distribution. Not to be used with 'blend_per_split'.  Defaults to None.
    """

    split_matrix: Optional[List[Tuple[float, float]]] = field(init=False, default=None)
    """The split matrix consisting of non-overlapping book-ends of each split in order. For more
       information, refer to 'convert_split_vector_to_split_matrix'. Created automatically from
       'split'. Not to be passed in to the constructor.
    """

    num_dataset_builder_threads: int = 1
    """The number of threads to use for dataset building."""

    path_to_cache: Optional[str] = None
    """Where all re-useable dataset indices are to be cached."""

    mmap_bin_files: bool = True
    """Whether to mmap the .bin files or use file pointers."""

    mock: bool = field(init=False, default=False)
    """Whether to bypass real data loading and validation in favor of mock data generation.
       Created automatically from 'blend' and 'blend_per_split'. Not to be passed in to the
       constructor.
    """

    tokenizer: Optional[MegatronTokenizer] = None
    """The MegatronTokenizer instance. Required for datasets that do online tokenization."""

    mid_level_dataset_surplus: float = 0.005
    """The sample surplus to build for the mid-level datasets(s). Defaults arbitrarily to 0.005.
       This value is irrelevant for single source data blends. This value may need to be increased
       if the top level dataset oversamples the mid level dataset(s). This value may be set to 0.0
       in future if the top level dataset is constrained to not oversample the mid level
       datasets(s).
    """

    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        if self.blend_per_split is not None and any(self.blend_per_split):
            assert self.blend is None, "blend and blend_per_split are incompatible"
            assert self.split is None, "split and blend_per_split are incompatible"
            assert len(self.blend_per_split) == len(Split), f"blend_per_split must contain {len(Split)} blends"
            for split in Split:
                if self.blend_per_split[split.value] is None:
                    logger.info(f"blend not provided for {split.name} split")
                else:
                    assert self.blend_per_split[split.value][1] is None or len(
                        self.blend_per_split[split.value][0]
                    ) == len(self.blend_per_split[split.value][1]), (
                        "blend per split prefixes and weights must be equal in number"
                    )
        else:
            if self.blend is not None:
                assert self.blend[1] is None or len(self.blend[0]) == len(self.blend[1]), (
                    "blend prefixes and weights must be equal in number"
                )
                assert self.split is not None, "split must be provided when blend is not None"
            else:
                self.mock = True
                logger.info("Let mock = True, as both blend and blend_per_split are None")
                self.split = "1,1,1"
                logger.info(f"Let split = {self.split}, an arbitrarily even split, as mock is True")
            split_vector = parse_and_normalize_split(self.split)
            self.split_matrix = convert_split_vector_to_split_matrix(split_vector)
            logger.info(f"Let split_matrix = {self.split_matrix}")


def parse_and_normalize_split(split: str) -> List[float]:
    """Parse the dataset split ratios from a string

    Args:
        split (str): The train valid test split string e.g. "99,1,0"

    Returns:
        List[float]: The trian valid test split ratios e.g. [0.99, 0.01, 0.0]
    """
    split = list(map(float, re.findall(r"[.0-9]+", split)))
    split = split + [0.0 for _ in range(len(Split) - len(split))]

    assert len(split) == len(Split)
    assert all(map(lambda _: _ >= 0.0, split))

    split = _normalize(split)

    return split


def convert_split_vector_to_split_matrix(
    vector_a: List[float], vector_b: Optional[List[float]] = None
) -> List[Optional[Tuple[float, float]]]:
    """Build the split matrix from one or optionally two contributing split vectors.

    Ex. a standard conversion:

    [0.99, 0.01, 0.0] -> [(0, 0.99), (0.99, 1.0), None]

    Ex. a conversion for Retro when Retro pretraining uses a [0.99, 0.01, 0.0] split and Retro
    preprocessing used a [0.98, 0.02, 0.0] split:

    [0.99, 0.01, 0.0], [0.98, 0.02, 0.0] -> [(0, 0.98), (0.99, 1.0), None]

    Args:
        vector_a (List[float]): The primary split vector

        vector_b (Optional[List[float]]): An optional secondary split vector which constrains the
            primary split vector. Defaults to None.

    Returns:
        List[Tuple[float, float]]: The split matrix consisting of book-ends of each split in order
    """
    if vector_b is None:
        vector_b = vector_a

    # [.900, .090, .010] -> [0.00, .900, .990, 100]
    expansion_a = functools.reduce(lambda a, b: a + [a[len(a) - 1] + b], [[0], *vector_a])
    expansion_b = functools.reduce(lambda a, b: a + [a[len(a) - 1] + b], [[0], *vector_b])

    # [0.00, .900, .990, 100.0] -> [(0.00, .900), (.900, .990), (.990, 100)]
    bookends_a = list(zip(expansion_a[:-1], expansion_a[1:]))
    bookends_b = list(zip(expansion_b[:-1], expansion_b[1:]))

    # gather per-split overlap or None
    matrix = []
    for bookend_a, bookend_b in zip(bookends_a, bookends_b):
        if min(bookend_a[1], bookend_b[1]) <= max(bookend_a[0], bookend_b[0]):
            overlap = None
        else:
            overlap = (max(bookend_a[0], bookend_b[0]), min(bookend_a[1], bookend_b[1]))
        matrix.append(overlap)

    return matrix


@dataclass
class GPTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for Megatron Core GPT datasets"""

    reset_position_ids: Optional[bool] = None
    """Option to reset the position IDs in the dataset at an interval"""

    reset_attention_mask: Optional[bool] = None
    """Option to reset the attention mask from the dataset"""

    eod_mask_loss: Optional[bool] = None
    """Option to enable the EOD mask loss"""

    create_attention_mask: bool = True
    """Option to enable the attention masks generation. Can be disabled if attention kernel
       generates masks by itself.
    """

    drop_last_partial_validation_sequence: bool = True
    """Option to drop the last partial validation sequence"""

    add_extra_token_to_sequence: bool = True
    """Option to draw sequences with one extra token to ensure the sample input tokens and sample
       output tokens are both of the desired sequence length
    """

    object_storage_cache_path: Optional[str] = None
    """Path for caching indices for s3 or msc dataloading."""

    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        super().__post_init__()

        assert self.tokenizer is not None

        assert self.reset_position_ids is not None
        assert self.reset_attention_mask is not None
        assert self.eod_mask_loss is not None


class GPTDataset(torch.utils.data.Dataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the GPTDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset. When
            None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        self.dataset = indexed_dataset
        self.dataset_path = dataset_path
        self.indices = indexed_indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config

        self.unique_identifiers = OrderedDict()

        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["dataset_path"] = self.dataset_path
        self.unique_identifiers["num_samples"] = self.num_samples
        self.unique_identifiers["index_split"] = self.index_split.name
        for attr in self._key_config_attributes():
            self.unique_identifiers[attr] = getattr(self.config, attr)

        self.unique_description = json.dumps(
            self.unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
        )
        self.unique_description_hash = hashlib.md5(self.unique_description.encode("utf-8")).hexdigest()
        self.masks_and_position_ids_are_cacheable = not any(
            [
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
            ]
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

        try:
            self._pad_token_id = self.config.tokenizer.pad
        except Exception:
            self._pad_token_id = _PAD_TOKEN_ID

        (self.document_index, self.sample_index, self.shuffle_index) = self._build_document_sample_shuffle_indices()

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Return all config attributes which contribute to uniquely identifying the dataset.

        These attributes will be used to build a uniquely identifying string and MD5 hash which
        will be used to cache/load dataset resources from run to run.

        Returns:
            List[str]: The key config attributes
        """
        return ["random_seed", "sequence_length", "split", "split_matrix", "tokenizer"]

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying IndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique elements in the underlying IndexedDataset
        """
        return low_level_dataset.sequence_lengths.shape[0]

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> IndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (GPTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        return IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return self.sample_index.shape[0] - 1

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence so the index does not matter
            text, _ = self._query_document_sample_shuffle_indices(0)
        else:
            text, _ = self._query_document_sample_shuffle_indices(idx)

        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        if not self.masks_and_position_ids_are_cacheable or not self.masks_and_position_ids_are_cached:
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

    def _query_document_sample_shuffle_indices(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset - doc_index_beg_offset + self.config.add_extra_token_to_sequence,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = None if i < doc_index_end else doc_index_end_offset + self.config.add_extra_token_to_sequence
                sample_parts.append(self.dataset.get(self.document_index[i], offset=offset, length=length))
        assert len(document_ids) == len(sample_parts), (
            f"len(document_ids) ({len(document_ids)}) != len(sample_parts) ({len(sample_parts)})"
        )

        length = sum(map(len, sample_parts))

        # Pad the sample if necessary
        if length < (self.config.sequence_length + self.config.add_extra_token_to_sequence):
            sample_parts.append(
                [self._pad_token_id] * (self.config.sequence_length + self.config.add_extra_token_to_sequence - length)
            )

        return (
            numpy.concatenate(sample_parts, dtype=numpy.int64),
            numpy.array(document_ids, dtype=numpy.int64),
        )

    def _build_document_sample_shuffle_indices(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index

        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The document index, the sample
            index, and the shuffle index
        """
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices")

        if path_to_cache:
            base = f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}"
            get_path_to = lambda affix: os.path.join(path_to_cache, f"{base}-{affix}")
            path_to_description = get_path_to("description.txt")
            path_to_document_index = get_path_to("document_index.npy")
            path_to_sample_index = get_path_to("sample_index.npy")
            path_to_shuffle_index = get_path_to("shuffle_index.npy")
            cache_hit = all(
                map(
                    os.path.isfile,
                    [
                        path_to_description,
                        path_to_document_index,
                        path_to_sample_index,
                        path_to_shuffle_index,
                    ],
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (
            not cache_hit and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):
            logger.info(f"Build and save the {type(self).__name__} {self.index_split.name} indices")
            t_beg = time.time()

            sequence_length = self.config.sequence_length
            num_tokens_per_epoch = self._get_num_tokens_per_epoch()
            num_epochs = self._get_num_epochs(num_tokens_per_epoch)

            if num_epochs == 1:
                separate_final_epoch = False
            else:
                # Get the number of samples for the last epoch
                num_samples_sans_final_epoch = (
                    (num_epochs - 1) * num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                ) // sequence_length
                num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
                num_samples_per_epoch = (
                    num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                ) // sequence_length

                # num_samples_from_final_epoch should be non-negative
                assert num_samples_from_final_epoch >= 0

                # num_samples_from_final_epoch should not exceed max value
                assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

                # Separate the final epoch if it falls below the threshold
                threshold = 0.80
                separate_final_epoch = num_samples_from_final_epoch < int(threshold * num_samples_per_epoch)

                logger.debug(f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}")
                logger.debug(f"> threshold: {threshold}")
                logger.debug(f"> num_samples_per_epoch: {num_samples_per_epoch}")

            logger.debug(f"> separate_final_epoch: {separate_final_epoch}")

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            # Build the document index
            document_index = _build_document_index(self.indices, num_epochs, numpy_random_state, separate_final_epoch)

            drop_last_partial_sequence = True
            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence

            # Build the sample index
            from megatron.core.datasets import helpers

            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence
            else:
                drop_last_partial_sequence = True

            assert document_index.dtype == numpy.int32
            assert self.dataset.sequence_lengths.dtype == numpy.int32
            if len(document_index) * 2 > len(self.dataset.sequence_lengths):
                # If "access density" of sequence_lengths is high, force load the mmap-ed array
                # into memory by making a copy.
                #
                # System performance benefits come from two aspects:
                #   1. We sequentially pre-load the whole file, most of which we expect to read
                #   2. The GIL is held when entering the c++ program, improving the speed of which
                #      improves parallelism
                sequence_lengths_for_cpp = self.dataset.sequence_lengths.copy()
            else:
                sequence_lengths_for_cpp = self.dataset.sequence_lengths
            sample_index = helpers.build_sample_idx(
                sequence_lengths_for_cpp,
                document_index,
                sequence_length,
                num_epochs,
                num_tokens_per_epoch,
                drop_last_partial_sequence,
                self.config.add_extra_token_to_sequence,
            )

            # Build the shuffle index
            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
                )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                numpy.save(path_to_document_index, document_index, allow_pickle=True)
                numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
                numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            else:
                logger.warning(f"Unable to save {type(self).__name__} indexes because path_to_cache is None")

            t_end = time.time()
            logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            logger.info(f"> total number of samples: {sample_index.shape[0] - 1}")
            logger.info(f"> total number of epochs: {num_epochs}")

            return document_index, sample_index, shuffle_index

        logger.info(f"Load the {type(self).__name__} {self.index_split.name} indices")

        logger.info(f"\tLoad the document index from {os.path.basename(path_to_document_index)}")
        t_beg = time.time()
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        logger.info(f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}")
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        logger.info(f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}")
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        logger.info(f"> total number of samples: {sample_index.shape[0] - 1}")

        return document_index, sample_index, shuffle_index

    def _get_num_tokens_per_epoch(self) -> int:
        """Calculate the number of tokens in a single epoch

        Returns:
            int: The number of tokens in a single epoch
        """
        return int(numpy.sum(self.dataset.sequence_lengths[self.indices]))

    def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
        """Calculate the number of epochs

        Args:
            num_tokens_per_epoch (int): The number of tokens in a single epoch

        Returns:
            int: The number of epochs
        """
        num_epochs = 1
        num_tokens = num_tokens_per_epoch
        if self.num_samples is None:
            return num_epochs
        else:
            num_tokens_requested = (
                self.num_samples * self.config.sequence_length
            ) + self.config.add_extra_token_to_sequence
            while num_tokens < num_tokens_requested:
                num_epochs += 1
                num_tokens += num_tokens_per_epoch
        return num_epochs


def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
) -> numpy.ndarray:
    """Build an array with length = num epochs * num documents

    Args:
        documents (numpy.ndarray): the subset of exposed document indices

        num_epochs (int): The number of epochs

        numpy_random_state (numpy.random.RandomState): The NumPy random state

        separate_final_epoch (bool): Whether to exclude the last epoch from the global shuffle

    Returns:
        numpy.ndarray: The document index
    """
    if not separate_final_epoch or num_epochs == 1:
        document_index = numpy.mgrid[0:num_epochs, 0 : len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1)
        document_index = document_index.astype(numpy.int32)
        numpy_random_state.shuffle(document_index)
        return document_index

    doc_idx_first = _build_document_index(documents, num_epochs - 1, numpy_random_state, False)
    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False)
    return numpy.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: numpy.random.RandomState
) -> numpy.ndarray:
    """Build the range [0, size) and shuffle

    Args:
        num_samples (int): The size of the first shuffle range [0, num_samples)

        total_size (int): The size of the entire index. If larger than 'num_samples', it defines
            the second shuffle range [num_samples, total_size)

        numpy_random_state (numpy.random.RandomState): The NumPy random state

    Returns:
        numpy.ndarray: The shuffle index
    """
    dtype_ = numpy.uint32
    if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
        dtype_ = numpy.int64

    shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = numpy.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_last)

    return numpy.concatenate((shuffle_idx_first, shuffle_idx_last))


def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    create_attention_mask: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

        create_attention_mask (bool): Switch to enable the attention masks generation. Can be
            disabled if attention kernel generates masks by itself.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    seq_length = data.numel()

    if create_attention_mask:
        attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=data.device)).unsqueeze(0)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def _normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """
    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w
