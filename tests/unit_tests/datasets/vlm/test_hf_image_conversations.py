# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Embedded HF image ordering and invalid-index checks."""

import copy
import io

import pytest
from PIL import Image

from nemo_automodel.components.datasets.vlm.datasets import _image_conversations


def _record(color: int) -> dict[str, bytes | None]:
    buffer = io.BytesIO()
    Image.new("L", (3, 5), color).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def test_indexed_images_preserve_order_repetition_and_source() -> None:
    source = {
        "messages": [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 1},
                        {"type": "text", "text": "compare"},
                        {"type": "image", "index": 0},
                        {"type": "image", "index": 1},
                    ],
                }
            ]
        ],
        "images": [[_record(20), _record(90)]],
    }
    before = copy.deepcopy(source)
    output = _image_conversations(source)["conversation"][0][0]["content"]
    assert source == before
    assert [item["image"].getpixel((0, 0)) for item in output if item["type"] == "image"] == [90, 20, 90]
    assert output[1] == {"type": "text", "text": "compare"}


@pytest.mark.parametrize("index", [-1, 1, True, None, "0"])
def test_invalid_image_index_is_rejected(index: int | bool | str | None) -> None:
    source = {
        "messages": [[{"role": "user", "content": [{"type": "image", "index": index}]}]],
        "images": [[_record(20)]],
    }
    with pytest.raises(ValueError, match="Invalid image index"):
        _image_conversations(source)


def test_text_only_and_unreferenced_image() -> None:
    source = {"messages": [[{"role": "assistant", "content": [{"type": "text", "text": "answer"}]}]], "images": [[]]}
    assert _image_conversations(source)["conversation"] == source["messages"]
    source["images"] = [[_record(20)]]
    with pytest.raises(ValueError, match="Every supplied image"):
        _image_conversations(source)
