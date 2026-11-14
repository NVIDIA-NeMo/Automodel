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

"""Fetched image normalization, including storage ownership and schema errors."""

import io
from array import array

import pytest
from PIL import Image

from nemo_automodel.components.datasets.vlm.datasets import VlmMediaDecodeError
from nemo_automodel.components.datasets.zephon.vlm_input import (
    VlmInputOptions,
    decode_embedded_image,
    normalize_image_document,
)


def _png(color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (3, 5), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_bytes_hf_struct_and_pil_are_owned_and_paths_are_never_opened(monkeypatch):
    data = _png()
    original = Image.new("L", (3, 5), color=40)
    real_open = Image.open

    def memory_only(value, *args, **kwargs):
        assert isinstance(value, io.BytesIO)
        return real_open(value, *args, **kwargs)

    monkeypatch.setattr(Image, "open", memory_only)
    for value in (
        data,
        bytearray(data),
        memoryview(data),
        array("B", data),
        {"bytes": data, "path": "s3://ignored/image"},
        {"bytes": array("B", data), "path": "s3://ignored/image"},
        original,
    ):
        image = decode_embedded_image(value)
        assert image.mode == "RGB" and image.size == (3, 5) and image is not original
        assert image.getpixel((0, 0)) == ((40, 40, 40) if value is original else (255, 0, 0))
    assert original.mode == "L"
    for value in ("/missing.png", "https://host/image", {"path": "/missing.png", "bytes": None}):
        with pytest.raises(ValueError, match="self-contained Parquet or LitData"):
            decode_embedded_image(value)
    with pytest.raises(VlmMediaDecodeError, match="embedded image bytes"):
        decode_embedded_image(b"corrupt")


def test_common_schemas_preserve_multiturn_order_and_image_placement():
    images = [_png("red"), _png("blue")]
    turns = [{"from": "human", "value": "before<image>between<image>after"}, {"from": "gpt", "value": "answer"}]
    document = normalize_image_document({"conversations": turns, "images": images})
    content = document["conversation"][0]["content"]
    assert [part["type"] for part in content] == ["text", "image", "text", "image", "text"]
    assert [part["text"] for part in content if part["type"] == "text"] == ["before", "between", "after"]
    assert [part["image"].getpixel((0, 0)) for part in content if part["type"] == "image"] == [(255, 0, 0), (0, 0, 255)]
    assert [turn["role"] for turn in document["conversation"]] == ["user", "assistant"]
    assert turns[0]["value"] == "before<image>between<image>after" and images[0] == _png("red")
    mapped = normalize_image_document(
        {"dialog": turns, "photos": images},
        options=VlmInputOptions(column_mapping={"messages": "dialog", "images": "photos"}),
    )
    assert [part["type"] for part in mapped["conversation"][0]["content"]] == [part["type"] for part in content]
    qa = normalize_image_document(
        {"image": {"bytes": images[0], "path": "ignored"}, "question": "why?", "answer": "yes"}
    )
    assert [part["type"] for part in qa["conversation"][0]["content"]] == ["image", "text"]
    structured = normalize_image_document(
        {
            "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "why?"}]}],
            "image": images[0],
        }
    )
    assert structured["conversation"][0]["content"][0]["image"].size == (3, 5)
    text = normalize_image_document({"conversation": '[{"role":"assistant","content":"text"}]'})
    assert text["conversation"][0]["content"] == [{"type": "text", "text": "text"}]


def _custom(payload):
    return {"conversation": [{"role": "user", "content": [{"type": "image", "image": payload["blob"]}]}]}


def test_custom_normalizer_is_revalidated_and_no_unknown_media_is_dropped():
    result = normalize_image_document({"blob": _png()}, options=VlmInputOptions(normalizer=_custom))
    assert result["conversation"][0]["content"][0]["image"].mode == "RGB"
    invalid = [
        {"messages": [{"role": "user", "content": "<image>"}]},
        {"messages": [{"role": "user", "content": "no marker"}], "image": _png()},
        {"messages": [{"role": "user", "content": [{"type": "video", "video": "pointer"}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image", "image": "/tmp/no.png"}]}]},
        {"conversation": [], "messages": []},
    ]
    for payload in invalid:
        with pytest.raises(ValueError, match="my source"):
            normalize_image_document(payload, source_name="my source")
    with pytest.raises(ValueError, match="Unknown VLM"):
        VlmInputOptions(column_mapping={"typo": "value"})


def test_pointer_columns_and_late_pointers_fail_before_any_image_decode(monkeypatch):
    def must_not_decode(*args, **kwargs):
        raise AssertionError("schema validation must happen before image decode")

    monkeypatch.setattr(Image, "open", must_not_decode)
    for payload in (
        {"image_path": "/data/p.png", "question": "q", "answer": "a"},
        {"messages": [{"role": "user", "content": "<image><image>"}], "images": [b"corrupt", "/data/p.png"]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": b"corrupt"},
                        {"type": "image", "image": {"path": "s3://bucket/p.png"}},
                    ],
                }
            ]
        },
    ):
        with pytest.raises(ValueError, match="self-contained Parquet or LitData"):
            normalize_image_document(payload)
