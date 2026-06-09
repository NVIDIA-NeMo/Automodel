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

"""BAGEL-style multimodal data pipeline for packed three-group training.

Ports the subset of upstream BAGEL ``data`` needed to feed BAGEL training from
fully AM-native code: VLM SFT, T2I pretrain, and unified image editing. The
packed batch schema is shared by Stage 1 and Stage 2; whether VAE /
flow-matching tensors are consumed is controlled by the model stage.
"""

from __future__ import annotations

from .collate_fns import SimpleCustomBatch, bagel_packed_collate_fn, collate_wrapper
from .datasets import (
    DATASET_REGISTRY,
    DEFAULT_DATASET_INFO,
    SftJSONLIterableDataset,
    T2IIterableDataset,
    UnifiedEditIterableDataset,
    make_bagel_multimodal_dataset,
)
from .distributed_iterable import DistributedIterableDataset
from .interleave import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from .packing import DataConfig, PackedDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs
from .transforms import ImageTransform, MaxLongEdgeMinShortEdgeResize
from .utils import (
    add_special_tokens,
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    len2weight,
    patchify,
    pil_img2rgb,
    prepare_attention_mask_per_sample,
)
from .video import FrameSampler

__all__ = [
    "DATASET_REGISTRY",
    "DEFAULT_DATASET_INFO",
    "DataConfig",
    "DistributedIterableDataset",
    "FrameSampler",
    "ImageTransform",
    "InterleavedBaseIterableDataset",
    "MaxLongEdgeMinShortEdgeResize",
    "PackedDataset",
    "ParquetStandardIterableDataset",
    "SftJSONLIterableDataset",
    "SimpleCustomBatch",
    "T2IIterableDataset",
    "UnifiedEditIterableDataset",
    "add_special_tokens",
    "bagel_packed_collate_fn",
    "collate_wrapper",
    "get_flattened_position_ids_extrapolate",
    "get_flattened_position_ids_interpolate",
    "get_parquet_data_paths",
    "init_arrow_pf_fs",
    "len2weight",
    "make_bagel_multimodal_dataset",
    "patchify",
    "pil_img2rgb",
    "prepare_attention_mask_per_sample",
]
