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

from functools import lru_cache
from types import ModuleType
from typing import List, Tuple


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
    """Load the CPU dataset helpers, compiling once into PyTorch's extension cache.

    Requires a C++ compiler and Ninja. ``TORCH_EXTENSIONS_DIR`` can override
    the cache location; the installed package directory is never modified.
    PyTorch's build lock coordinates processes sharing the same cache, so
    each rank can call this without a distributed barrier. The loaded module
    is retained for the lifetime of this process.

    Returns:
        The compiled module containing the dataset indexing functions.
    """
    from torch.utils.cpp_extension import load_inline

    from nemo_automodel.components.datasets.llm.megatron._helpers_source import CPP_SOURCE

    return load_inline(
        name="nemo_automodel_megatron_helpers",
        cpp_sources=CPP_SOURCE,
        extra_cflags=["-O3"],
        with_cuda=False,
    )
