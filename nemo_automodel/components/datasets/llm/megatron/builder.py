# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Any, Callable, Iterable, List, Optional, Type, Union
from enum import Enum

import numpy
import torch

from nemo_automodel.components.datasets.llm.megatron.gpt_dataset import _normalize, GPTDataset, GPTDatasetConfig

logger = logging.getLogger(__name__)

class Split(Enum):
    train = 0
    valid = 1
    test = 2

class BlendedMegatronDatasetBuilder:
    """Builder class for the BlendedDataset and MegatronDataset classes

    Args:
        cls (Type[MegatronDataset]): The class to instantiate, must inherit from MegatronDataset

        sizes (List[Optional[int]]): The minimum total number of samples to draw, or None, per split

        is_built_on_rank (Callable): A callable which returns True if the dataset should be built on
            the current rank and False otherwise. It should be Megatron Core parallelism aware i.e.
            global rank, local group rank, and virtual rank may inform its return value.

        config (BlendedMegatronDatasetConfig): The config object which informs dataset creation
    """

    def __init__(
        self,
        cls: GPTDataset,
        sizes: list[int],
        is_built_on_rank: Callable,
        config: GPTDatasetConfig,
    ):
        self.cls = cls
        self.sizes = sizes
        self.is_built_on_rank = is_built_on_rank
        self.config = config

        logger.info(
            f"Building {cls.__name__} splits with sizes={self.sizes} and config={self.config}",
        )

        if torch.distributed.is_initialized():
            gb_rank = torch.distributed.get_rank()
            if gb_rank == 0:
                assert (
                    self.is_built_on_rank()
                ), "is_built_on_rank must return True when global rank = 0"

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
        return self._build_blended_dataset_splits()

    def _build_blended_dataset_splits(self) -> List[Optional[GPTDataset]]:
        """Build all dataset splits according to the provided blend(s)

        See the BlendedMegatronDatasetBuilder.build alias for more information.

        Returns:
            List[Optional[GPTDataset]]: A list containing a dataset instance (or None) per
                split
        """
        if self.config.blend:
            prefixes, weights = self.config.blend
            assert len(prefixes) == 1, "Dataset blending not supported yet"
            if weights is not None:
                weights = _normalize(weights)

            split = self.config.split_matrix

            # Blend consists of a single prefix
            if len(prefixes) == 1 and weights is None:
                return self._build_megatron_dataset_splits(prefixes[0], split, self.sizes)
        else:
            # missing blend error
            raise ValueError("Missing blend in config")

    def _build_megatron_dataset_splits(
        self,
        dataset_path: Optional[str],
        split: List[float],
        sizes: List[int],
        synchronize_ranks: bool = True,
    ) -> List[Optional["MidLevelDataset"]]:
        """Build each MidLevelDataset split from a single LowLevelDataset

        Args:
            dataset_path (Optional[str]): The path on disk which defines the underlying
                LowLevelDataset, or None for mock dataset classes

            split (List[Tuple[float, float]]): The dataset split matrix

            sizes (List[int]): The number of total samples to draw from each split

            synchronize_ranks (bool): Whether to call barrier for rank-0 / barrier / other-ranks
                behavior. Set to False when we enforce this behavior at higher level.

        Returns:
            List[Optional[MidLevelDataset]]: The MidLevelDataset (or None) per split
        """
        # short-cut if we are not building on this rank
        if torch.distributed.is_initialized() and not self.is_built_on_rank():
            for i in range(len(Split)):
                if split[i] is not None and synchronize_ranks:
                    torch.distributed.barrier()
            return [None] * len(Split)

        # Build the low level dataset
        low_level_dataset = self.cls.build_low_level_dataset(dataset_path, self.config)

        # Build the split indices for the low level dataset
        num_elements = self.cls.numel_low_level_dataset(low_level_dataset)
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
                        self.cls,
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

    @staticmethod
    def build_generic_dataset(
        cls: Union[Type["DistributedDataset"], Callable],
        is_built_on_rank: Callable,
        synchronize_ranks: bool,
        *args: Any,
    ) -> Optional[Union["DistributedDataset", Iterable]]:
        """Build the DistributedDataset

        Return None if and only if the underlying dataset class is not built on the current rank
        and torch.distributed is initialized.

        Args:
            cls (Union[Type[DistributedDataset], Callable]): The DistributedDataset class to be
                built. In special cases, e.g. when we are building the low level dataset for a
                RawMegatronDataset instance, we can accept a Callable which returns an Iterable.

            synchronize_ranks (bool): Whether to call barrier for rank-0 / barrier / other-ranks
                behavior. Set to False when we enforce this behavior at higher level.

            args (Tuple[Any]): The positional arguments used to build the provided
                DistributedDataset class

        Raises:
            Exception: When the dataset constructor raises an OSError

        Returns:
            Optional[Union[DistributedDataset, Iterable]]: The DistributedDataset instantion, the
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
                        f"Failed to write dataset materials to the data cache directory. Please "
                        f"supply a directory to which you have write access via the path_to_cache "
                        f"attribute in BlendedMegatronDatasetConfig and retry. Refer to the "
                        f"preserved traceback above for more information."
                    )
                    raise Exception(log) from err

            if synchronize_ranks:
                torch.distributed.barrier()

            # After, build on other ranks
            if rank != 0 and is_built_on_rank():
                dataset = cls(*args)

            return dataset

        return cls(*args)

