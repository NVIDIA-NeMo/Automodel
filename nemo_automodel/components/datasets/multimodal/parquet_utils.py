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
#   data/parquet_utils.py
# Upstream copyright: Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Function names and semantics match the T2I / editing dataset path contract.

"""Parquet shard discovery + filesystem factory for BAGEL T2I / edit data.

Only the local-filesystem path is exercised in our current tests; the
HDFS branch is preserved for upstream compatibility but the cluster-
specific host / port / extra_conf hooks remain stubs. Customise them in
your own deployment if you actually have HDFS-backed parquet shards.
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def get_parquet_data_paths(data_dir_list, num_sampled_data_paths, rank=0, world_size=1):
    """Return a flat list of parquet file paths sharded across ranks.

    Directories are split across ranks via
    ``chunk_size = ceil(num_dirs / world_size)``. Each rank lists its local
    directories, repeats the file list to reach ``num_sampled_data_paths`` per
    directory, then all-gathers across ranks so every rank ends up with the
    same combined list.
    """
    num_data_dirs = len(data_dir_list)
    if world_size > 1:
        chunk_size = (num_data_dirs + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, num_data_dirs)
        local_data_dir_list = data_dir_list[start_idx:end_idx]
        local_num_sampled_data_paths = num_sampled_data_paths[start_idx:end_idx]
    else:
        local_data_dir_list = data_dir_list
        local_num_sampled_data_paths = num_sampled_data_paths

    local_data_paths = []
    for data_dir, num_data_path in zip(local_data_dir_list, local_num_sampled_data_paths):
        if data_dir.startswith("hdfs://"):
            files = hdfs_ls_cmd(data_dir)
            data_paths_per_dir = [file for file in files if file.endswith(".parquet")]
        else:
            files = os.listdir(data_dir)
            data_paths_per_dir = [os.path.join(data_dir, name) for name in files if name.endswith(".parquet")]
        repeat = num_data_path // len(data_paths_per_dir)
        data_paths_per_dir = data_paths_per_dir * (repeat + 1)
        local_data_paths.extend(data_paths_per_dir[:num_data_path])

    if world_size > 1:
        # Lazy import so CPU-only / non-distributed users don't pay for it.
        import torch.distributed as dist

        gather_list = [None] * world_size
        dist.all_gather_object(gather_list, local_data_paths)

        combined_chunks = []
        for chunk_list in gather_list:
            if chunk_list is not None:
                combined_chunks.extend(chunk_list)
    else:
        combined_chunks = local_data_paths

    return combined_chunks


# NOTE: customize these three functions for your cluster if you want HDFS.
def get_hdfs_host():  # pragma: no cover - cluster hook
    """Return the HDFS host URI used by BAGEL parquet readers."""
    return "hdfs://xxx"


def get_hdfs_block_size():  # pragma: no cover - cluster hook
    """Return the HDFS read buffer size for pyarrow."""
    return 134217728


def get_hdfs_extra_conf():  # pragma: no cover - cluster hook
    """Return optional pyarrow HDFS configuration overrides."""
    return None


def init_arrow_pf_fs(parquet_file_path):
    """Return a pyarrow filesystem for ``parquet_file_path``.

    ``pyarrow`` is imported lazily because not every AM install carries it,
    and the import is only needed when an actually-parquet-backed dataset
    (T2I, UnifiedEdit) is constructed.
    """
    import pyarrow.fs as pf

    if parquet_file_path.startswith("hdfs://"):
        fs = pf.HadoopFileSystem(
            host=get_hdfs_host(),
            port=0,
            buffer_size=get_hdfs_block_size(),
            extra_conf=get_hdfs_extra_conf(),
        )
    else:
        fs = pf.LocalFileSystem()
    return fs


def hdfs_ls_cmd(dir):  # pragma: no cover - cluster hook
    """List HDFS parquet directory entries with the native hdfs CLI."""
    result = subprocess.run(["hdfs", "dfs", "ls", dir], capture_output=True, text=True).stdout
    return ["hdfs://" + i.split("hdfs://")[-1].strip() for i in result.split("\n") if "hdfs://" in i]
