# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Static regression fixture: ty check --no-force-exclude --error all <this file>."""

import torch
from datasets import Dataset
from PIL import Image
from transformers import ProcessorMixin

from nemo_automodel.components.datasets.vlm.conversation import ConversationSample
from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapperConfig, RdrDatasetConfig
from nemo_automodel.components.datasets.vlm.mock import MockVlmDatasetConfig
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import NeatPackConfig


def check_builders(processor: ProcessorMixin) -> None:
    text: ConversationSample = {"conversation": [{"role": "user", "content": "Describe this image."}]}
    media: ConversationSample = {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (2, 2))},
                    {"type": "image", "image": b"image-bytes"},
                    {"type": "image", "image": torch.zeros(3, 2, 2)},
                ],
            }
        ],
        "mm_inputs_meta": {"height": 2, "width": 2},
    }
    for raw in (RdrDatasetConfig().build(), MockVlmDatasetConfig().build(), [text, media]):
        tokenized = PreTokenizedDatasetWrapperConfig().build(dataset=raw, processor=processor)
        NeatPackConfig().build(dataset=tokenized, padding_idx=0, ds_raw=raw, processor=processor)


def check_external_datasets(
    raw: Dataset | torch.utils.data.Dataset[ConversationSample], processor: ProcessorMixin
) -> None:
    tokenized = PreTokenizedDatasetWrapperConfig().build(dataset=raw, processor=processor)
    NeatPackConfig().build(dataset=tokenized, padding_idx=0, ds_raw=raw, processor=processor)
