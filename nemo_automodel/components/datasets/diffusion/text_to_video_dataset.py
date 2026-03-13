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

import pickle
from pathlib import Path
from typing import Dict, List

import torch

from .base_dataset import BaseMultiresolutionDataset

VIDEO_OPTIONAL_FIELDS = ("text_mask", "text_embeddings_2", "text_mask_2", "image_embeds")


def load_optional_video_fields(data: dict, device: str = "cpu") -> dict:
    """Extract optional model-specific fields, moving to device."""
    result = {}
    for key in VIDEO_OPTIONAL_FIELDS:
        if key in data and data[key] is not None:
            result[key] = data[key].to(device)
    return result


def collate_optional_video_fields(batch: List[Dict], result: dict) -> None:
    """Concatenate optional video fields present in batch into result dict."""
    if not batch:
        return
    for key in VIDEO_OPTIONAL_FIELDS:
        if key in batch[0]:
            result[key] = torch.cat([item[key] for item in batch], dim=0)


class TextToVideoDataset(BaseMultiresolutionDataset):
    """Text-to-Video dataset with multiresolution bucket organization.

    Loads preprocessed .meta files organized by resolution bucket.
    Compatible with SequentialBucketSampler for multiresolution training.
    """

    def __init__(self, cache_dir: str, model_type: str = "wan", device: str = "cpu"):
        """
        Args:
            cache_dir: Directory containing preprocessed cache (metadata.json + shards + WxH/*.meta)
            model_type: Model type for model-specific fields ("wan", "hunyuan", etc.)
            device: Device to load tensors to
        """
        self.model_type = model_type
        self.device = device
        super().__init__(cache_dir, quantization=8)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single video sample from its .meta file."""
        item = self.metadata[idx]
        cache_file = Path(item["cache_file"])

        with open(cache_file, "rb") as f:
            data = pickle.load(f)

        video_latents = data["video_latents"].to(self.device)
        text_embeddings = data["text_embeddings"].to(self.device)

        output = {
            "video_latents": video_latents,
            "text_embeddings": text_embeddings,
            "bucket_resolution": torch.tensor(item["bucket_resolution"]),
            "aspect_ratio": item.get("aspect_ratio", 1.0),
        }

        # Model-specific optional fields
        output.update(load_optional_video_fields(data, self.device))

        return output
