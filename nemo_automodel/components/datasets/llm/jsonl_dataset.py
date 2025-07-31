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

from contextlib import ExitStack
from typing import TYPE_CHECKING

import torch
from torch.utils.data import IterableDataset

from nemo_automodel.components.datasets.llm.lingua_assets.core import (
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class JSONLDataset(IterableDataset):
    def __init__(
        self,
        root_dir: str,
        rank: int,
        world_size: int,
        tokenizer: "PreTrainedTokenizerBase",
        sources: dict[str, float],
        batch_size: int,
        packed_sequence_size: int,
        seed: int,
        split: str,
        add_bos: bool = True,
        add_eos: bool = True,
        load_async: bool = False,
        prefetch_size: int = 64,
        n_views: int = 2,
        infinite: bool = True,
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            rank: Rank of the process
            world_size: World size (number of processes)
            tokenizer: Tokenizer to be used for tokenization
            sources: Dictionary of sources to be used for the dataset. For example, {"fineweb_edu_10bt_shuffled": 50.0, "other_dataset": 50.0}
            batch_size: Batch size
            packed_sequence_size: Size of the packed sequence
            seed: Seed for the random number generator
            split: Split to be used for the dataset. Must be either "train" or "validation"
            add_bos: Whether to add the beginning of sentence token
            add_eos: Whether to add the end of sentence token
            load_async: Whether to load the dataset asynchronously
            prefetch_size: Size of the prefetch buffer
            n_views: Number of views to be used for the dataset. Each view is offset by 1 from the previous one.
                We use 2 views for the dataset. The first view is the input sequence and the second view is the target sequence.
            infinite: If True, the dataset will loop infinitely. If False, the dataset will stop after one pass through all data.
        """
        assert split in ["train", "validation"], "Split must be either train or validation"
        # Persist constructor args so we can rebuild the dataloader after a checkpoint restore
        self._root_dir = root_dir
        self._rank = rank
        self._world_size = world_size
        self._tokenizer = tokenizer
        self._sources = sources
        self._batch_size = batch_size
        self._packed_sequence_size = packed_sequence_size
        self._seed = seed
        self._split = split
        self._add_bos = add_bos
        self._add_eos = add_eos
        self._load_async = load_async
        self._prefetch_size = prefetch_size
        self._n_views = n_views
        self._infinite = infinite

        # Initialize dataloader state
        self.data_loader_state = init_dataloader_state_from_args(
            root_dir,
            rank,
            world_size,
            sources,
            batch_size,
            packed_sequence_size,
            seed,
            add_bos,
            add_eos,
            prefetch_size,
            n_views,
            split,
            not infinite,
        )

        # Create the context stack to manage the dataloader lifecycle and build the dataloader
        self._build_dataloader()

    def __iter__(self):
        """
        Yields batches of data from the dataset.
        """
        for batch, state in self.data_loader:
            self.data_loader_state = state

            batch = torch.from_numpy(batch)
            bs, s, _ = batch.shape
            loss_mask = torch.ones((bs, s), dtype=torch.int64)
            return_batch = {
                "input_ids": batch[:, :, 0],
                "labels": batch[:, :, 1],
                "loss_mask": loss_mask,
            }
            yield return_batch

    def __del__(self):
        # Ensure resources are closed even if build failed partially
        if hasattr(self, "context_stack"):
            self.context_stack.close()

    def state_dict(self):
        """
        Returns the state of the dataloader.
        """
        return self.data_loader_state

    def load_state_dict(self, state_dict):
        """Restore dataloader state from a checkpoint.

        We recreate the underlying dataloader so that it picks up the new state
        (the original dataloader captured the old state at construction time).
        """
        # Close the existing dataloader to release file handles / worker processes
        if hasattr(self, "context_stack"):
            self.context_stack.close()

        # Replace state dict and rebuild the dataloader
        self.data_loader_state = state_dict
        self._build_dataloader()

    def _build_dataloader(self):
        """Helper to (re)build the internal dataloader using the current state.

        This is called when the state is restored from a checkpoint.
        """
        # Always create a fresh context stack so resources are correctly managed
        self.context_stack = ExitStack()
        self.data_loader = self.context_stack.enter_context(
            build_dataloader_from_args(
                self._tokenizer, self._load_async, self._prefetch_size, state=self.data_loader_state
            )
        )
