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

import fcntl
import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import List, Tuple

_HELPER_CACHE_ROOT = Path("/tmp") / f"nemo_automodel_{os.getuid()}"


def get_blend_from_list(
    blend: List[str] | None,
) -> Tuple[List[str], List[float] | None] | None:
    """Get the megatron.core.datasets.blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend from the blend list

    Args:
        blend (Optional[List[str]]): The blend list, which can be either (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        Optional[Tuple[List[str], Optional[List[float]]]]: The blend, consisting of a list of dataset prefixes and optionally a list of dataset weights, e.g. [["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], [30.0, 70.0]].
    """
    if blend is None:
        return None

    if len(blend) % 2 == 1:
        weight_per_dataset = None
        raw_prefix_per_dataset = blend
    else:
        raw_weight_per_dataset, raw_prefix_per_dataset = zip(
            *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
        )

        weight_per_dataset = []
        for rwpd in raw_weight_per_dataset:
            try:
                weight = float(rwpd)
            except ValueError:
                weight = None
            weight_per_dataset.append(weight)

        is_none = map(lambda _: _ is None, weight_per_dataset)
        if any(is_none):
            assert all(is_none)
            weight_per_dataset = None
            raw_prefix_per_dataset = blend

    prefix_per_dataset = [rppd.strip() for rppd in raw_prefix_per_dataset]

    return prefix_per_dataset, weight_per_dataset


@lru_cache(maxsize=1)
def compile_helper() -> ModuleType:
    """Load the CPU dataset helpers from a locked, node-local build cache.

    Requires a C++ compiler and Ninja. Sources and build artifacts live in a
    per-user directory under ``/tmp``, independent of ``TORCH_EXTENSIONS_DIR``
    and ``TMPDIR``. Each source and Python/PyTorch environment has its own
    cache entry. An exclusive file lock covers source generation, compilation,
    and import, so ranks on the same node can call this without a distributed
    barrier. The loaded module is retained for the lifetime of this process.

    Returns:
        The compiled module containing the dataset indexing functions.
    """
    import torch
    from torch.utils.cpp_extension import load_inline

    from nemo_automodel.components.datasets.llm.megatron._helpers_source import CPP_SOURCE

    cache_key = hashlib.sha256(
        f"{sys.version}\n{torch.__version__}\n{torch.__file__}\n{CPP_SOURCE}".encode()
    ).hexdigest()
    _HELPER_CACHE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    build_directory = _HELPER_CACHE_ROOT / cache_key
    build_directory.mkdir(exist_ok=True)
    # load_inline writes main.cpp before taking PyTorch's own build lock.
    # Keep this separate lock file in place: unlinking it can split waiters
    # across different inodes and allow concurrent source writers again.
    with (build_directory / "compile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return load_inline(
            name="nemo_automodel_megatron_helpers",
            cpp_sources=CPP_SOURCE,
            extra_cflags=["-O3"],
            with_cuda=False,
            build_directory=str(build_directory),
        )
