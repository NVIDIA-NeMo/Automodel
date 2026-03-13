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

"""Diffusion model datasets and dataloaders."""

import importlib

_LAZY_ATTRS = {
    # Dataset classes
    "BaseMultiresolutionDataset": (".base_dataset", "BaseMultiresolutionDataset"),
    "TextToImageDataset": (".text_to_image_dataset", "TextToImageDataset"),
    "TextToVideoDataset": (".text_to_video_dataset", "TextToVideoDataset"),
    "MetaFilesDataset": (".meta_files_dataset", "MetaFilesDataset"),
    # Utilities
    "MultiTierBucketCalculator": (".multi_tier_bucketing", "MultiTierBucketCalculator"),
    "SequentialBucketSampler": (".sampler", "SequentialBucketSampler"),
    "VIDEO_OPTIONAL_FIELDS": (".text_to_video_dataset", "VIDEO_OPTIONAL_FIELDS"),
    # Collate functions
    "collate_fn_text_to_image": (".collate_fns", "collate_fn_text_to_image"),
    "collate_fn_video": (".collate_fns", "collate_fn_video"),
    "collate_fn_production": (".collate_fns", "collate_fn_production"),
    # Dataloader builders
    "build_text_to_image_multiresolution_dataloader": (
        ".collate_fns",
        "build_text_to_image_multiresolution_dataloader",
    ),
    "build_video_multiresolution_dataloader": (".collate_fns", "build_video_multiresolution_dataloader"),
    # Legacy (non-multiresolution)
    "build_dataloader": (".meta_files_dataset", "build_dataloader"),
    # Mock/test
    "build_mock_dataloader": (".mock_dataloader", "build_mock_dataloader"),
}

__all__ = sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
