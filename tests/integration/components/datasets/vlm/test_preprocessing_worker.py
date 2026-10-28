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

"""Shared VLM preprocessing crosses a spawned worker boundary without a model."""

import multiprocessing

import torch
from PIL import Image

from nemo_automodel.components.datasets.vlm.datasets import encode_vlm_document
from tests.utils.vlm_fixtures import processor as processor


def test_encoder_spawn_transfer_has_no_model_dependency(processor):
    document = {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("L", (8, 12), color=20)},
                    {"type": "image", "image": Image.new("L", (16, 8), color=30)},
                    {"type": "text", "text": "Describe image"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
        ]
    }
    expected = encode_vlm_document(document, processor, inject_fake_images=False)
    with multiprocessing.get_context("spawn").Pool(1) as workers:
        actual = workers.apply(encode_vlm_document, (document, processor), {"inject_fake_images": False})
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
