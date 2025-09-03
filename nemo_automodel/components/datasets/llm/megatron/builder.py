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

import hashlib
import json
import logging
import math
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy
import torch

from nemo_automodel.components.datasets.llm.megatron.gpt_dataset import GPTDataset, GPTDatasetConfig, Split, normalize

logger = logging.getLogger(__name__)

_VERBOSE = False


class BlendedDataset(torch.utils.data.Dataset):
    """Conjugating class for a set of MegatronDataset instances

    Args:
        datasets (List[MegatronDataset]): The MegatronDataset instances to blend

        weights (List[Union[int, float]]): The weights that determine the dataset blend ratios

        size (Optional[int]): The number of samples to draw from the blend. If None, for each
            dataset index idx draw exactly weights[idx] samples from datasets[idx].

        config (BlendedMegatronDatasetConfig): The config

    Raises:
        RuntimeError: When the dataset has fewer or more samples than 'size' post-initialization
    """

    def __init__(
        self,
        datasets: List[GPTDataset],
        weights: List[Union[int, float]],
        size: Optional[int],
        config: GPTDatasetConfig,
    ) -> None:
        assert len(datasets) == len(weights)
        assert len(datasets) < 32767
        assert all(map(lambda _: type(_) == type(datasets[0]), datasets))
        assert all(map(lambda _: _.index_split == datasets[0].index_split, datasets))
        assert all(map(lambda _: _ > 0, weights))
        assert all(map(lambda _: type(_) == type(weights[0]), weights))
        if size is None and isinstance(weights[0], float):
            assert all(map(lambda _: _ == int(_), weights))

        # Alert user to unnecessary blending
        if len(datasets) == 1:
            logger.warning("Building a BlendedDataset for a single MegatronDataset")

        if size is not None:
            weights = normalize(weights)

        self.datasets = datasets
        self.split = self.datasets[0].index_split
        self.weights = weights
        self.size = size
        self.config = config

        unique_identifiers = OrderedDict()
        unique_identifiers["class"] = type(self).__name__
        unique_identifiers["datasets"] = [dataset.unique_identifiers for dataset in self.datasets]
        unique_identifiers["split"] = self.split.name
        unique_identifiers["weights"] = self.weights
        unique_identifiers["size"] = self.size
        unique_identifiers["tokenizer"] = self.config.tokenizer.name_or_path

        self.unique_description = json.dumps(unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers)
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

        self.dataset_index, self.dataset_sample_index = self._build_indices()

    def __len__(self) -> int:
        return self.dataset_index.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Union[int, numpy.ndarray]]:
        dataset_id = self.dataset_index[idx]
        dataset_sample_id = self.dataset_sample_index[idx]
        return {"dataset_id": dataset_id, **self.datasets[dataset_id][dataset_sample_id]}

    def _build_indices(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Build and optionally cache the dataset index and the dataset sample index

        The dataset index is a 1-D mapping which determines the dataset to query. The dataset
        sample index is a 1-D mapping which determines the sample to request from the queried
        dataset.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The dataset index and the dataset sample index
        """

        path_to_cache = self.config.path_to_cache

        if path_to_cache:
            get_path_to = lambda suffix: os.path.join(
                path_to_cache,
                f"{self.unique_description_hash}-{type(self).__name__}-{self.split.name}-{suffix}",
            )
            path_to_description = get_path_to("description.txt")
            path_to_dataset_index = get_path_to("dataset_index.npy")
            path_to_dataset_sample_index = get_path_to("dataset_sample_index.npy")
            cache_hit = all(
                map(
                    os.path.isfile,
                    [path_to_description, path_to_dataset_index, path_to_dataset_sample_index],
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (not cache_hit and torch.distributed.get_rank() == 0):
            logger.info(f"Build and save the {type(self).__name__} indices")

            # Build the dataset and dataset sample indexes
            logger.info("\tBuild and save the dataset and dataset sample indexes")
            t_beg = time.time()
            from nemo_automodel.components.datasets.llm.megatron import helpers_cpp

            if self.size is not None:
                dataset_index = numpy.zeros(self.size, dtype=numpy.int16)
                dataset_sample_index = numpy.zeros(self.size, dtype=numpy.int64)
                helpers_cpp.build_blending_indices(
                    dataset_index,
                    dataset_sample_index,
                    self.weights,
                    len(self.datasets),
                    self.size,
                    _VERBOSE,
                )
            else:
                size = sum(self.weights)
                dataset_index = numpy.zeros(size, dtype=numpy.int16)
                dataset_sample_index = numpy.zeros(size, dtype=numpy.int64)
                helpers_cpp.build_exhaustive_blending_indices(
                    dataset_index, dataset_sample_index, self.weights, len(self.datasets)
                )

            dataset_indices, dataset_sizes = numpy.unique(dataset_index, return_counts=True)
            for i, (_index, _size) in enumerate(zip(dataset_indices, dataset_sizes)):
                if len(self.datasets[_index]) < _size:
                    raise IndexError(
                        f"The {self.split.name} blend oversamples the contributing datasets and, "
                        f"for example, requests {_size} samples from "
                        f"{type(self.datasets[_index]).__name__} number {i} in excess of its size "
                        f"{len(self.datasets[_index])}. The current value of the config attribute "
                        f"mid_level_dataset_surplus may be increased, e.g. two- or ten-fold, from "
                        f"its current value ({self.config.mid_level_dataset_surplus}) to ensure a "
                        f"sufficient mid-level dataset sample margin from which to draw."
                    )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                # Save the indexes
                numpy.save(path_to_dataset_index, dataset_index, allow_pickle=True)
                numpy.save(path_to_dataset_sample_index, dataset_sample_index, allow_pickle=True)
            else:
                logger.warning(f"Cannot save the {type(self).__name__} indexes because path_to_cache is None")

            t_end = time.time()
            logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            return dataset_index, dataset_sample_index

        logger.info(f"Load the {type(self).__name__} indices")

        logger.info(f"\tLoad the dataset index from {path_to_dataset_index}")
        t_beg = time.time()
        dataset_index = numpy.load(path_to_dataset_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        logger.info(f"\tLoad the dataset sample index from {path_to_dataset_sample_index}")
        t_beg = time.time()
        dataset_sample_index = numpy.load(path_to_dataset_sample_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        return dataset_index, dataset_sample_index


class BlendedMegatronDatasetBuilder:
    """Builder class for the BlendedDataset and MegatronDataset classes

    Args:

        sizes (List[Optional[int]]): The minimum total number of samples to draw, or None, per split

        is_built_on_rank (Callable): A callable which returns True if the dataset should be built on
            the current rank and False otherwise. It should be Megatron Core parallelism aware i.e.
            global rank, local group rank, and virtual rank may inform its return value.

        config (BlendedMegatronDatasetConfig): The config object which informs dataset creation
    """

    def __init__(
        self,
        sizes: list[int],
        is_built_on_rank: Callable,
        config: GPTDatasetConfig,
        enabled_splits: Optional[list[str]] = None,
    ):
        self.sizes = sizes
        self.is_built_on_rank = is_built_on_rank
        self.config = config
        self.enabled_splits_names = enabled_splits

        repr = []
        for k, v in self.config.__dict__.items():
            repr.append(f"{k}: {v}" if k != "tokenizer" else f"{k}: {v.name_or_path}")
        logger.info(f"Building {GPTDataset.__name__} splits with sizes={self.sizes} and config=[{', '.join(repr)}]")

        if torch.distributed.is_initialized():
            gb_rank = torch.distributed.get_rank()
            if gb_rank == 0:
                assert self.is_built_on_rank(), "is_built_on_rank must return True when global rank = 0"

    def build(self) -> List[Optional[GPTDataset]]:
        """Build all dataset splits according to the provided blend(s)

        This method is distributed-aware and must be called on all ranks.

        The dataset splits returned can vary according to the config. Supply config.blend and
        config.split to build BlendedDataset and/or MegatronDataset splits from the same
        distribution. Supply config.blend_per_split to build BlendedDataset and/or MegatronDataset
        splits from separate distributions. In either case, for each split, handle the following
        cases:

        (1) The split is None
            - do nothing

        (2) The split has one contributing dataset, and...

            (a) 'size' is not None
                - Build a mid-level dataset with low-level dataset sampling in proportion to the
                size

            (b) 'size' is None
                - Build mid-level datasets with no excess low-level dataset sampling

        (3) The split has multiple contributing datasets, and...

            (a) 'weights' is not None and 'size' is not None
                - Build mid-level datasets with low-level dataset sampling in proportion to their
                weights and the size
                - Build a top-level dataset of length marginally greater than 'size' with mid-level
                dataset sampling in proportion to their weights and the size

            (b) 'weights' is not None and 'size' is None
                - Error

            (c) 'weights' is None and 'size' is not None
                - Build mid-level datasets with no excess low-level dataset sampling
                - Build a top-level dataset of length 'size' (capped at the sum of the mid-level
                dataset lengths) with mid-level dataset sampling in proportion to their lengths
                and the size

            (d) 'weights' is None and 'size' is None
                - Build mid-level datasets with no excess low-level dataset sampling
                - Build a top-level dataset with no excess mid-level dataset sampling

        Returns:
            List[Optional[GPTDataset]]: A list containing a dataset instance (or None) per
                split
        """
        datasets = self._build_blended_dataset_splits()

        for dataset in datasets:
            if dataset is not None and len(dataset) > 0:
                if isinstance(dataset, BlendedDataset):
                    assert dataset.size is None or dataset.size == len(dataset)
                elif isinstance(dataset, GPTDataset):
                    assert dataset.num_samples is None or dataset.num_samples <= len(dataset)

        return datasets

    def _build_blended_dataset_splits(self) -> List[Optional[GPTDataset]]:
        """Build all dataset splits according to the provided blend(s)

        See the BlendedMegatronDatasetBuilder.build alias for more information.

        Returns:
            List[Optional[GPTDataset]]: A list containing a dataset instance (or None) per
                split
        """
        if self.config.blend:
            prefixes, weights = self.config.blend
            if weights is not None:
                weights = normalize(weights)

            split = self._masked_split_matrix(self.config.split_matrix)

            # Blend consists of a single prefix
            if len(prefixes) == 1 and weights is None:
                return self._build_megatron_dataset_splits(prefixes[0], split, self.sizes)

            # Build the mid-level datasets
            if weights is None:
                # Build only one "epoch"
                sizes_per_dataset_buffer = [[None for split in Split] for prefix in prefixes]
            else:
                # The number of samples we plan to use per dataset
                # Mask sizes for disabled splits so we don't process None entries
                masked_sizes = [
                    self.sizes[i] if split[i] is not None and self.sizes[i] is not None else 0
                    for i in range(len(Split))
                ]
                sizes_per_dataset_target = _get_size_per_split_per_dataset(weights, masked_sizes)
                # The number of samples we plan to build per dataset
                sizes_per_dataset_buffer = _get_size_per_split_per_dataset(
                    weights,
                    masked_sizes,
                    surplus=self.config.mid_level_dataset_surplus,
                )

            # Build each dataset in parallel
            megatron_datasets = self._build_megatron_datasets_parallel(prefixes, split, sizes_per_dataset_buffer)

            # Build the top-level datasets
            blended_datasets = [None] * len(Split)
            for i in range(len(Split)):
                if split[i] is not None:
                    weights_i = weights
                    if weights_i is not None and self.sizes[i] is not None:
                        # Blend according to client-specified weights and client-specified size
                        size_per_dataset = list(zip(*sizes_per_dataset_target))[i]
                        size_i = sum(size_per_dataset)
                    elif weights_i is None:
                        # Blend according to dataset sizes as-is and (maybe) client-specified size
                        try:
                            weights_i = [len(megatron_dataset) for megatron_dataset in megatron_datasets[i]]
                        except TypeError:
                            weights_i = [0 for _ in prefixes]
                        if self.sizes[i] is not None:
                            size_i = min(self.sizes[i], sum(weights_i))
                        else:
                            # Build exhaustive indices
                            size_i = None
                    else:
                        raise ValueError("Using client-specified weights requires client-specified size")
                    blended_datasets[i] = self.build_generic_dataset(
                        BlendedDataset,
                        self.is_built_on_rank,
                        True,  # synchronize_ranks, default behavior to build on rank-0 first
                        megatron_datasets[i],
                        weights_i,
                        size_i,
                        self.config,
                    )

            return blended_datasets

        ##
        # Each split comes from a separate distribution
        ##
        else:
            blended_datasets = [None] * len(Split)
            for i in range(len(Split)):
                # Skip building if this split is disabled
                if not self._is_enabled_index(i):
                    continue
                split_spoof = [None] * len(Split)
                split_spoof[i] = (0.0, 1.0)
                sizes_spoof = [0] * len(Split)
                sizes_spoof[i] = self.sizes[i]

                # Blend is provided for the split
                blend = self.config.blend_per_split[i]
                if blend is not None:
                    prefixes, weights = blend
                    if weights is not None:
                        weights = normalize(weights)

                    # Blend consists of a sigle prefix
                    if len(prefixes) == 1:
                        blended_datasets[i] = self._build_megatron_dataset_splits(
                            prefixes[0], split_spoof, sizes_spoof
                        )[i]
                        continue
                    elif self.config.multiple_validation_sets and i == Split.valid.value:
                        # handle multiple validation sets
                        validation_datasets = []
                        if self.config.full_validation:
                            # verify that size is None, which causes a single epoch dataset
                            # to be built
                            assert sizes_spoof[i] is None
                        for prefix in prefixes:
                            ds = self._build_megatron_dataset_splits(prefix, split_spoof, sizes_spoof)[i]
                            validation_datasets.append(ds)
                        blended_datasets[i] = validation_datasets
                        continue

                    # Build mid-level datasets
                    if weights is None:
                        sizes_per_dataset_buffer = [[None for split in Split] for prefix in prefixes]
                    else:
                        # The number of samples we plan to use per dataset
                        sizes_per_dataset_target = _get_size_per_split_per_dataset(weights, sizes_spoof)
                        # The number of samples we plan to build per dataset
                        sizes_per_dataset_buffer = _get_size_per_split_per_dataset(
                            weights, sizes_spoof, surplus=self.config.mid_level_dataset_surplus
                        )

                    # Build each dataset in parallel
                    megatron_datasets = self._build_megatron_datasets_parallel(
                        prefixes, split_spoof, sizes_per_dataset_buffer
                    )[i]

                    # Build top-level dataset
                    if weights is not None and self.sizes[i] is not None:
                        # Blend according to client-specified weights and client-specified size
                        size_per_dataset = list(zip(*sizes_per_dataset_target))[i]
                        size = sum(size_per_dataset)
                    elif weights is None:
                        # Blend according to dataset sizes as-is and (maybe) client-specified size
                        try:
                            weights = [len(megatron_dataset) for megatron_dataset in megatron_datasets]
                        except TypeError:
                            weights = [0 for _ in prefixes]
                        if self.sizes[i] is not None:
                            size = min(self.sizes[i], sum(weights))
                        else:
                            # Build exhaustive indices
                            size = None
                    else:
                        raise RuntimeError
                    blended_datasets[i] = self.build_generic_dataset(
                        BlendedDataset,
                        self.is_built_on_rank,
                        True,  # synchronize_ranks, default behavior to build on rank-0 first
                        megatron_datasets,
                        weights,
                        size,
                        self.config,
                    )

            return blended_datasets

    def _build_megatron_dataset_splits(
        self,
        dataset_path: Optional[str],
        split: List[float],
        sizes: List[int],
        synchronize_ranks: bool = True,
    ) -> List[Optional[GPTDataset]]:
        """Build each MidLevelDataset split from a single LowLevelDataset

        Args:
            dataset_path (Optional[str]): The path on disk which defines the underlying
                LowLevelDataset, or None for mock dataset classes

            split (List[Tuple[float, float]]): The dataset split matrix

            sizes (List[int]): The number of total samples to draw from each split

            synchronize_ranks (bool): Whether to call barrier for rank-0 / barrier / other-ranks
                behavior. Set to False when we enforce this behavior at higher level.

        Returns:
            List[Optional[GPTDataset]]: The GPTDataset (or None) per split
        """
        # short-cut if we are not building on this rank
        if torch.distributed.is_initialized() and not self.is_built_on_rank():
            for i in range(len(Split)):
                if split[i] is not None and synchronize_ranks:
                    torch.distributed.barrier()
            return [None] * len(Split)

        # Build the low level dataset
        low_level_dataset = GPTDataset.build_low_level_dataset(dataset_path, self.config)

        # Build the split indices for the low level dataset
        num_elements = GPTDataset.numel_low_level_dataset(low_level_dataset)
        split_indices = []
        for i, _ in enumerate(Split):
            if split[i] is not None:
                beg = int(round(split[i][0] * float(num_elements)))
                end = int(round(split[i][1] * float(num_elements)))
                split_indices.append(numpy.arange(start=beg, stop=end, step=1, dtype=numpy.int32))
            else:
                split_indices.append(None)

        # Build the mid level dataset
        mid_level_datasets = []
        for i, _split in enumerate(Split):
            if split[i] is None:
                mid_level_datasets.append(None)
            else:
                mid_level_datasets.append(
                    self.build_generic_dataset(
                        GPTDataset,
                        self.is_built_on_rank,
                        synchronize_ranks,
                        low_level_dataset,
                        dataset_path,
                        split_indices[i],
                        sizes[i],
                        _split,
                        self.config,
                    )
                )

        return mid_level_datasets

    def _build_megatron_datasets_parallel(
        self, prefixes: List[str], split: List[float], sizes_per_dataset: List[List[int]]
    ) -> List[List[Optional[GPTDataset]]]:
        """Build the megatron datasets for a list of prefixes in parallel

        Args:
            prefixes (List[str]): The list of prefix strings

            split (List[float]): The dataset split ratios (must sum to 1.00)

            sizes_per_dataset (List[List[int]]): The number of samples to request
            per MegatronDataset per spilt

        Returns:
            List[List[Optional[GPTDataset]]]: For each split, have a list of
            GPTDataset per prefix
        """

        # Helper function to wrap the threading logic
        def _threading_helper(
            megatron_datasets: List[List[Optional[GPTDataset]]],
            num_workers: int,
            prefixes: List[str],
            split: List[float],
            sizes_per_dataset: List[List[int]],
        ) -> None:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                all_futures = []
                for i in range(len(prefixes)):
                    all_futures.append(
                        executor.submit(
                            self._build_megatron_dataset_splits,
                            prefixes[i],
                            split,
                            sizes_per_dataset[i],
                            False,  # synchronize_ranks, barrier is called in this function
                        )
                    )
                for future in all_futures:
                    try:
                        megatron_datasets_split = future.result()
                        for j in range(len(megatron_datasets_split)):
                            megatron_datasets[j].append(megatron_datasets_split[j])
                    except Exception as err:
                        raise err

        megatron_datasets = [[] for _ in range(len(Split))]
        num_dataset_builder_threads = self.config.num_dataset_builder_threads

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            # First, build on rank 0
            if rank == 0:
                num_workers = num_dataset_builder_threads
                if num_workers > 1:
                    # since only rank 0 is running, scale up the thread count
                    # but not too much to avoid overloading storage on miss path.
                    # if user set num_dataset_builder_threads to 1,
                    # i.e. meant for serial build, do not scale up.
                    num_workers *= min(2, max(1, torch.cuda.device_count()))
                _threading_helper(megatron_datasets, num_workers, prefixes, split, sizes_per_dataset)

            torch.distributed.barrier()

            # Then, build on other ranks; guaranteed to be data_cache hit
            if rank != 0:
                _threading_helper(
                    megatron_datasets,
                    num_dataset_builder_threads,
                    prefixes,
                    split,
                    sizes_per_dataset,
                )
        else:
            _threading_helper(megatron_datasets, num_dataset_builder_threads, prefixes, split, sizes_per_dataset)

        return megatron_datasets

    @staticmethod
    def build_generic_dataset(
        cls: Union[Type[GPTDataset | BlendedDataset], Callable],
        is_built_on_rank: Callable,
        synchronize_ranks: bool,
        *args: Any,
    ) -> Optional[Union[GPTDataset | BlendedDataset, Iterable]]:
        """Build the GPTDataset or BlendedDataset

        Return None if and only if the underlying dataset class is not built on the current rank
        and torch.distributed is initialized.

        Args:
            cls (Union[Type[GPTDataset | BlendedDataset], Callable]): The GPTDataset or BlendedDataset class to be
                built. In special cases, e.g. when we are building the low level dataset for a
                RawMegatronDataset instance, we can accept a Callable which returns an Iterable.

            synchronize_ranks (bool): Whether to call barrier for rank-0 / barrier / other-ranks
                behavior. Set to False when we enforce this behavior at higher level.

            args (Tuple[Any]): The positional arguments used to build the provided
                GPTDataset or BlendedDataset class

        Raises:
            Exception: When the dataset constructor raises an OSError

        Returns:
            Optional[Union[GPTDataset | BlendedDataset, Iterable]]: The GPTDataset or BlendedDataset instantion, the
                Iterable instantiation, or None
        """
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

            dataset = None

            # First, build on rank 0
            if rank == 0 and is_built_on_rank():
                try:
                    dataset = cls(*args)
                except OSError as err:
                    log = (
                        "Failed to write dataset materials to the data cache directory. Please "
                        "supply a directory to which you have write access via the path_to_cache "
                        "attribute in BlendedMegatronDatasetConfig and retry. Refer to the "
                        "preserved traceback above for more information."
                    )
                    raise Exception(log) from err

            if synchronize_ranks:
                torch.distributed.barrier()

            # After, build on other ranks
            if rank != 0 and is_built_on_rank():
                dataset = cls(*args)

            return dataset

        return cls(*args)

    def _is_enabled_index(self, idx: int) -> bool:
        """Return True if a given split index should be built.

        If no enabled_splits were provided, all splits are enabled.
        """
        if self.enabled_splits_names is None:
            return True
        name_map = {Split.train.value: "train", Split.valid.value: "validation", Split.test.value: "test"}
        return name_map.get(idx) in self.enabled_splits_names

    def _masked_split_matrix(self, split_matrix: List[Optional[tuple]]) -> List[Optional[tuple]]:
        """Mask splits that are not enabled by setting their bookends to None.

        This preserves the original split ratios while skipping construction for disabled splits.
        """
        if self.enabled_splits_names is None:
            return split_matrix
        masked = []
        for i, val in enumerate(split_matrix):
            masked.append(val if self._is_enabled_index(i) else None)
        return masked


def _get_size_per_split_per_dataset(
    normalized_weights: List[float], target_size_per_split: List[int], surplus: float = 0.0
) -> List[List[int]]:
    """Determine the contribution of the MegatronDataset splits to the BlendedDataset splits

    Args:
        normalized_weights (List[float]): e.g. [0.3, 0.7]

        target_size_per_split (List[int]): The number of samples to target for each BlendedDataset
            split

        surplus (float): The sample surplus to build per split per dataset

    Returns:
        List[List[int]]: The number of samples to request per MegatronDataset per split
    """

    assert numpy.isclose(sum(normalized_weights), 1.0)

    # Use margin as buffer to ensure we satiate the request
    sizes_per_dataset = [
        [int(math.ceil(math.ceil(target_size * weight) * (1 + surplus))) for target_size in target_size_per_split]
        for weight in normalized_weights
    ]

    return sizes_per_dataset
