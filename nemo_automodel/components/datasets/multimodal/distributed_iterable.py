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
#
# Includes Apache-2.0 code adapted from ByteDance-Seed/Bagel. Upstream references:
#   https://github.com/bytedance-seed/BAGEL
#   data/distributed_iterable_dataset.py
# Upstream copyright: Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Class name and attributes match the data-status round-trip contract.

"""DistributedIterableDataset base for BAGEL-style data pipelines."""

from __future__ import annotations

import copy
import random

import torch


class DistributedIterableDataset(torch.utils.data.IterableDataset):
    """Base class for rank/worker-aware iterable datasets.

    Owns a private ``rng`` used only to shuffle file paths deterministically
    in :meth:`set_epoch` — NOT used for per-sample randomness. Per-sample
    randomness still goes through the Python global ``random`` module (see
    :mod:`packing` for the reseed hook).
    """

    def __init__(self, dataset_name, local_rank=0, world_size=1, num_workers=8):
        self.dataset_name = dataset_name
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.rng = random.Random()
        self.data_paths = None
        self.data_status = None
        self._resume_data_status = None

    def set_data_status(self, data_status):
        self.data_status = copy.deepcopy(data_status)
        self._resume_data_status = copy.deepcopy(data_status)

    def _get_worker_data_status(self, worker_id):
        if self.data_status is None:
            return None
        return self.data_status[worker_id]

    def _set_worker_resume_data_status(self, worker_id, status):
        if self._resume_data_status is None:
            self._resume_data_status = [None] * max(int(self.num_workers), worker_id + 1)
        while len(self._resume_data_status) <= worker_id:
            self._resume_data_status.append(None)
        self._resume_data_status[worker_id] = copy.deepcopy(status)

    def state_dict(self):
        return {"data_status": copy.deepcopy(self._resume_data_status)}

    def load_state_dict(self, state_dict):
        if state_dict is None:
            self.set_data_status(None)
            return
        self.set_data_status(state_dict.get("data_status"))

    def get_data_paths(self, *args, **kwargs):
        raise NotImplementedError

    def set_epoch(self, seed=42):
        if self.data_paths is None:
            return

        if isinstance(self.data_paths[0], tuple):
            data_paths = sorted(self.data_paths, key=lambda x: (x[0], x[1]))
        elif isinstance(self.data_paths[0], str):
            data_paths = sorted(self.data_paths)
        else:
            raise ValueError(f"Unknown data_paths type: {type(self.data_paths[0])}")

        self.rng.seed(seed)
        self.rng.shuffle(data_paths)

        num_files_per_rank = len(data_paths) // self.world_size
        local_start = self.local_rank * num_files_per_rank
        local_end = (self.local_rank + 1) * num_files_per_rank
        self.num_files_per_rank = num_files_per_rank
        self.data_paths_per_rank = data_paths[local_start:local_end]

    def get_data_paths_per_worker(self):
        if self.data_paths is None:
            return None

        info = torch.utils.data.get_worker_info()
        if info is None:
            # Single worker: Use all files assigned to the rank
            return self.data_paths_per_rank, 0

        worker_id = info.id
        num_files_per_worker = self.num_files_per_rank // info.num_workers
        start = num_files_per_worker * worker_id
        end = num_files_per_worker * (worker_id + 1)
        data_paths_per_worker = self.data_paths_per_rank[start:end]

        return data_paths_per_worker[::-1], worker_id

    def __iter__(self):
        raise NotImplementedError
