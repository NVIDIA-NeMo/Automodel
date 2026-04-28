# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

import json
import os
from typing import Dict, List

import pytest
from PIL import Image

import nemo_automodel.components.datasets.vlm.datasets as ds
import nemo_automodel.components.datasets.vlm.utils as vlm_utils


@pytest.fixture(autouse=True)
def _isolate_random_choice(monkeypatch):
    """
    Make `random.choice` deterministic.  The monkeypatch is autouse so it
    applies to every test in this file.
    """
    monkeypatch.setattr(ds.random, "choice", lambda seq: seq[0])


@pytest.fixture
def stub_json2token(monkeypatch):
    """
    Replace `json2token` with a function that returns a stable,
    easily verifiable string.  It also records its inputs so we
    can assert call semantics later.
    """

    calls: List[Dict] = []

    def _fake_json2token(value, *, sort_json_key):  # noqa: D401
        """Very small stand-in for the real helper."""
        calls.append(
            {"value": value, "sort_json_key": sort_json_key},
        )
        return f"TOK::{json.dumps(value, sort_keys=sort_json_key)}"

    monkeypatch.setattr(ds, "json2token", _fake_json2token)
    return calls  # The test can inspect this list if it wants.


def test_make_rdr_dataset(monkeypatch):
    """End-to-end sanity check for `make_rdr_dataset`."""
    fake_ds = [
        {"image": "img_001", "text": "some label"},
        {"image": "img_002", "text": "another label"},
    ]

    # Patch `load_dataset` so no network call is issued.
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **k: fake_ds)

    result = ds.make_rdr_dataset()

    assert len(result) == len(fake_ds)
    for sample, src in zip(result, fake_ds, strict=True):
        assert list(sample) == ["conversation"]

        conversation = sample["conversation"]
        assert len(conversation) == 2

        # user turn
        user_turn = conversation[0]
        assert user_turn["role"] == "user"
        assert user_turn["content"][0] == {"type": "image", "image": src["image"]}
        assert user_turn["content"][1]["type"] == "text"

        # assistant turn
        assistant_turn = conversation[1]
        assert assistant_turn["role"] == "assistant"
        assistant_payload = assistant_turn["content"][0]
        assert assistant_payload == {"type": "text", "text": src["text"]}


@pytest.mark.parametrize(
    "ground_key,wrapper",
    [
        pytest.param(
            "gt_parses",
            lambda: {"gt_parses": [{"a": 1}, {"b": 2}]},
            id="multiple-parses",
        ),
        pytest.param(
            "gt_parse",
            lambda: {"gt_parse": {"answer": 42}},
            id="single-parse",
        ),
    ],
)
def test_make_cord_v2_dataset(monkeypatch, stub_json2token, ground_key, wrapper):
    """
    Parametrised test for the two possible CORD-V2 JSON layouts.
    """
    # One fake sample is enough for behaviour coverage.
    fake_ds = [
        {
            "image": "img_1337",
            "ground_truth": json.dumps(wrapper()),
        },
    ]
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **k: fake_ds)

    # Run
    result = ds.make_cord_v2_dataset()

    assert len(result) == 1
    convo = result[0]["conversation"]
    assert len(convo) == 2

    user_turn, assistant_turn = convo
    assert user_turn["role"] == "user"
    assert user_turn["content"][0] == {"type": "image", "image": "img_1337"}

    # The assistant text must be exactly what json2token produced
    assistant_payload = assistant_turn["content"][0]
    assert assistant_payload["text"].startswith("TOK::")

    # Called exactly once per GT-json, always with sort_json_key=True
    if ground_key == "gt_parses":
        expected_calls = len(json.loads(fake_ds[0]["ground_truth"])[ground_key])
    else:  # "gt_parse"
        expected_calls = 1
    assert len(stub_json2token) == expected_calls
    for call in stub_json2token:
        assert call["sort_json_key"] is True


def test_make_medpix_dataset(monkeypatch):
    """End-to-end sanity check for `make_medpix_dataset`."""
    fake_ds = [
        {
            "image_id": "medpix_001.jpg",
            "question": "What is shown in this medical image?",
            "answer": "This is a chest X-ray showing normal lung fields.",
        },
        {
            "image_id": "medpix_002.jpg",
            "question": "Describe the findings in this image.",
            "answer": "The image shows a fracture in the left femur.",
        },
    ]

    # Patch `load_dataset` so no network call is issued.
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **k: fake_ds)

    result = ds.make_medpix_dataset()

    assert len(result) == len(fake_ds)
    for sample, src in zip(result, fake_ds, strict=True):
        assert list(sample) == ["conversation"]

        conversation = sample["conversation"]
        assert len(conversation) == 2

        # user turn
        user_turn = conversation[0]
        assert user_turn["role"] == "user"
        assert user_turn["content"][0] == {"type": "image", "image": src["image_id"]}
        assert user_turn["content"][1] == {"type": "text", "text": src["question"]}

        # assistant turn
        assistant_turn = conversation[1]
        assert assistant_turn["role"] == "assistant"
        assistant_payload = assistant_turn["content"][0]
        assert assistant_payload == {"type": "text", "text": src["answer"]}


def test_make_cv17_dataset(monkeypatch):
    """End-to-end sanity check for `make_cv17_dataset`."""
    # Mock dataset with audio data and extra columns to test column removal
    class MockDataset:
        def __init__(self, data):
            self.data = data
            self.column_names = ["audio", "transcription", "extra_col1", "extra_col2", "unwanted_col"]

        def remove_columns(self, columns_to_remove):
            # Simulate column removal
            expected_removed = ["extra_col1", "extra_col2", "unwanted_col"]
            assert set(columns_to_remove) == set(expected_removed)
            return self.data

        def __iter__(self):
            return iter(self.data)

    fake_audio_data = [
        {
            "audio": {
                "array": [0.1, 0.2, 0.3, -0.1, -0.2],
                "sampling_rate": 16000
            },
            "transcription": "Merhaba, nasılsınız?"
        },
        {
            "audio": {
                "array": [0.5, -0.3, 0.8, 0.2, -0.1],
                "sampling_rate": 16000
            },
            "transcription": "Bu bir test cümlesidir."
        },
    ]

    mock_dataset = MockDataset(fake_audio_data)

    # Patch `load_dataset` so no network call is issued
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **k: mock_dataset)

    result = ds.make_cv17_dataset()

    assert len(result) == len(fake_audio_data)
    for sample, src in zip(result, fake_audio_data, strict=True):
        assert set(sample.keys()) == {"conversation", "audio"}

        # Test conversation structure
        conversation = sample["conversation"]
        assert len(conversation) == 2

        # Test user turn
        user_turn = conversation[0]
        assert user_turn["role"] == "user"
        assert user_turn["content"] == "<|endoftext11|>Transcribe the Turkish audio clip."

        # Test assistant turn
        assistant_turn = conversation[1]
        assert assistant_turn["role"] == "assistant"
        assert assistant_turn["content"] == src["transcription"]

        # Test audio data processing
        audio_array, sampling_rate = sample["audio"]
        assert audio_array == src["audio"]["array"]
        assert sampling_rate == src["audio"]["sampling_rate"]


def test_make_unimm_chat_dataset(monkeypatch):
    """End-to-end sanity check for `make_unimm_chat_dataset`."""
    fake_ds = [
        {
            "image": "img_A",
            "conversation": json.dumps(
                [
                    {"from": "human", "value": "Describe <image> please <IMAGE   > now."},
                    {"from": "gpt", "value": "  Response 1  "},
                ],
            ),
        },
        {
            "image": "img_B",
            "conversation": json.dumps(
                [
                    {"from": "human", "value": "<image>"},
                    {"from": "system", "value": "should be ignored"},
                    {"from": "gpt", "value": "Answer 2"},
                ],
            ),
        },
    ]

    # Patch `load_dataset` so no network call is issued.
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **k: fake_ds)

    result = ds.make_unimm_chat_dataset()

    assert len(result) == len(fake_ds)

    # First sample exercises mixed text/image content and whitespace trimming.
    convo_a = result[0]["conversation"]
    assert len(convo_a) == 2

    user_turn_a, assistant_turn_a = convo_a
    assert user_turn_a["role"] == "user"
    assert user_turn_a["content"] == [
        {"type": "text", "text": "Describe"},
        {"type": "image", "image": "img_A"},
        {"type": "text", "text": "please"},
        {"type": "image", "image": "img_A"},
        {"type": "text", "text": "now."},
    ]

    assert assistant_turn_a["role"] == "assistant"
    assert assistant_turn_a["content"] == [{"type": "text", "text": "Response 1"}]

    # Second sample shows placeholder-only inputs and ignored speaker roles.
    convo_b = result[1]["conversation"]
    assert len(convo_b) == 2

    user_turn_b, assistant_turn_b = convo_b
    assert user_turn_b["role"] == "user"
    assert user_turn_b["content"] == [{"type": "image", "image": "img_B"}]

    assert assistant_turn_b["role"] == "assistant"
    assert assistant_turn_b["content"] == [{"type": "text", "text": "Answer 2"}]


# ---------------------------------------------------------------------------
# Tests for _convert_sharegpt_to_conversation
# ---------------------------------------------------------------------------


class TestConvertSharegptToConversation:
    """Tests for the sharegpt-to-conversation conversion helper."""

    def test_basic_text_only(self):
        """Text-only messages without media."""
        example = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        result = ds._convert_sharegpt_to_conversation(example)
        assert result == {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]},
            ],
        }

    def test_image_placeholder(self):
        """User message with <image> placeholder replaced by actual path."""
        example = {
            "messages": [
                {"role": "user", "content": "<image>\nDescribe this image."},
                {"role": "assistant", "content": "A cat."},
            ],
            "images": ["cat.jpg"],
        }
        result = ds._convert_sharegpt_to_conversation(example)
        conv = result["conversation"]
        assert conv[0]["role"] == "user"
        assert conv[0]["content"] == [
            {"type": "image", "image": "cat.jpg"},
            {"type": "text", "text": "Describe this image."},
        ]
        assert conv[1] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "A cat."}],
        }

    def test_video_placeholder(self):
        """User message with <video> placeholder."""
        example = {
            "messages": [
                {"role": "user", "content": "<video>\nDescribe this video."},
                {"role": "assistant", "content": "A video of a dog."},
            ],
            "videos": ["dog.mp4"],
        }
        result = ds._convert_sharegpt_to_conversation(example)
        conv = result["conversation"]
        assert conv[0]["content"] == [
            {"type": "video", "video": "dog.mp4"},
            {"type": "text", "text": "Describe this video."},
        ]

    def test_media_dir_prepended(self):
        """Relative media paths are joined with media_dir."""
        example = {
            "messages": [
                {"role": "user", "content": "<image>\nWhat is this?"},
                {"role": "assistant", "content": "A photo."},
            ],
            "images": ["sub/img.jpg"],
        }
        result = ds._convert_sharegpt_to_conversation(
            example, media_dir="/data/media",
        )
        assert result["conversation"][0]["content"][0] == {
            "type": "image",
            "image": "/data/media/sub/img.jpg",
        }

    def test_absolute_media_path_not_modified(self):
        """Absolute media paths are not modified even when media_dir is set."""
        example = {
            "messages": [
                {"role": "user", "content": "<image>\nDescribe."},
                {"role": "assistant", "content": "Ok."},
            ],
            "images": ["/abs/path/img.jpg"],
        }
        result = ds._convert_sharegpt_to_conversation(
            example, media_dir="/data/media",
        )
        assert result["conversation"][0]["content"][0]["image"] == "/abs/path/img.jpg"

    def test_multiple_images_and_videos(self):
        """Multiple <image> and <video> placeholders consumed in order."""
        example = {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\n<video>\n<image>\nDescribe all.",
                },
                {"role": "assistant", "content": "Done."},
            ],
            "images": ["a.jpg", "b.jpg"],
            "videos": ["v.mp4"],
        }
        result = ds._convert_sharegpt_to_conversation(example)
        user_content = result["conversation"][0]["content"]
        assert user_content[0] == {"type": "image", "image": "a.jpg"}
        assert user_content[1] == {"type": "video", "video": "v.mp4"}
        assert user_content[2] == {"type": "image", "image": "b.jpg"}
        assert user_content[3] == {"type": "text", "text": "Describe all."}

    def test_custom_columns_and_tags(self):
        """Custom column names and tag mappings."""
        example = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"},
            ],
        }
        result = ds._convert_sharegpt_to_conversation(
            example,
            columns={"messages": "conversations"},
            tags={
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        )
        conv = result["conversation"]
        assert conv[0] == {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
        assert conv[1] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
        }

    def test_unknown_role_skipped(self):
        """Messages with unrecognized roles are silently skipped."""
        example = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        result = ds._convert_sharegpt_to_conversation(example)
        assert len(result["conversation"]) == 2

    def test_mm_inputs_meta_passthrough(self):
        """mm_inputs_meta is passed through to the output."""
        example = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "mm_inputs_meta": {"fps": 1, "nframes": 64},
        }
        result = ds._convert_sharegpt_to_conversation(example)
        assert result["mm_inputs_meta"] == {"fps": 1, "nframes": 64}


# ---------------------------------------------------------------------------
# Tests for make_meta_dataset
# ---------------------------------------------------------------------------


class TestMakeMetaDataset:
    """Tests for the meta-file dataset loading function."""

    def test_basic_jsonl(self, tmp_path):
        """Load a single dataset from a JSONL file."""
        # Create data file
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            json.dumps({
                "messages": [
                    {"role": "user", "content": "<image>\nWhat is this?"},
                    {"role": "assistant", "content": "A photo of a cat."},
                ],
                "images": ["cat.jpg"],
            }) + "\n"
            + json.dumps({
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
            }) + "\n",
        )

        # Create meta file
        meta_file = tmp_path / "dataset_info.json"
        meta_file.write_text(json.dumps({
            "my_dataset": {
                "file_name": "train.jsonl",
                "media_dir": "/data/images",
            },
        }))

        result = ds.make_meta_dataset(str(meta_file))

        assert len(result) == 2
        # First example: image + text
        conv0 = result[0]["conversation"]
        assert conv0[0]["content"][0] == {"type": "image", "image": "/data/images/cat.jpg"}
        assert conv0[0]["content"][1] == {"type": "text", "text": "What is this?"}
        assert conv0[1]["content"][0] == {"type": "text", "text": "A photo of a cat."}
        # Second example: text only
        conv1 = result[1]["conversation"]
        assert conv1[0]["content"] == [{"type": "text", "text": "Hello"}]

    def test_json_array_file(self, tmp_path):
        """Load from a plain JSON array file."""
        data_file = tmp_path / "train.json"
        data_file.write_text(json.dumps([
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
            },
        ]))

        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({
            "ds1": {"file_name": "train.json"},
        }))

        result = ds.make_meta_dataset(str(meta_file))
        assert len(result) == 1

    def test_multiple_datasets_combined(self, tmp_path):
        """Multiple datasets in one meta file are merged."""
        for name in ("a.jsonl", "b.jsonl"):
            (tmp_path / name).write_text(
                json.dumps({
                    "messages": [
                        {"role": "user", "content": f"From {name}"},
                        {"role": "assistant", "content": "Ok"},
                    ],
                }) + "\n",
            )

        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({
            "dataset_a": {"file_name": "a.jsonl"},
            "dataset_b": {"file_name": "b.jsonl"},
        }))

        result = ds.make_meta_dataset(str(meta_file))
        assert len(result) == 2

    def test_dataset_names_filter(self, tmp_path):
        """Only selected datasets are loaded when dataset_names is specified."""
        for name in ("a.jsonl", "b.jsonl"):
            (tmp_path / name).write_text(
                json.dumps({
                    "messages": [
                        {"role": "user", "content": f"From {name}"},
                        {"role": "assistant", "content": "Ok"},
                    ],
                }) + "\n",
            )

        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({
            "dataset_a": {"file_name": "a.jsonl"},
            "dataset_b": {"file_name": "b.jsonl"},
        }))

        result = ds.make_meta_dataset(str(meta_file), dataset_names=["dataset_a"])
        assert len(result) == 1
        assert result[0]["conversation"][0]["content"][0]["text"] == "From a.jsonl"

    def test_dataset_names_missing_raises(self, tmp_path):
        """Requesting a non-existent dataset name raises ValueError."""
        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({"ds1": {"file_name": "x.jsonl"}}))

        with pytest.raises(ValueError, match="not found in meta file"):
            ds.make_meta_dataset(str(meta_file), dataset_names=["nonexistent"])

    def test_missing_file_name_raises(self, tmp_path):
        """Dataset entry without file_name raises ValueError."""
        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({"ds1": {"media_dir": "/tmp"}}))

        with pytest.raises(ValueError, match="missing 'file_name'"):
            ds.make_meta_dataset(str(meta_file))

    def test_sample_ratio(self, tmp_path):
        """sample_ratio < 1.0 reduces the number of loaded examples."""
        data_file = tmp_path / "train.jsonl"
        lines = []
        for i in range(10):
            lines.append(json.dumps({
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
            }))
        data_file.write_text("\n".join(lines) + "\n")

        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({
            "ds1": {"file_name": "train.jsonl", "sample_ratio": 0.5},
        }))

        result = ds.make_meta_dataset(str(meta_file))
        assert len(result) == 5

    def test_sample_ratio_upsample(self, tmp_path):
        """sample_ratio > 1.0 duplicates data (integer ratio)."""
        data_file = tmp_path / "train.jsonl"
        lines = []
        for i in range(10):
            lines.append(json.dumps({
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
            }))
        data_file.write_text("\n".join(lines) + "\n")

        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({
            "ds1": {"file_name": "train.jsonl", "sample_ratio": 2.0},
        }))

        result = ds.make_meta_dataset(str(meta_file))
        assert len(result) == 20

    def test_sample_ratio_upsample_fractional(self, tmp_path):
        """sample_ratio with fractional part (e.g. 1.5) adds partial extra copy."""
        data_file = tmp_path / "train.jsonl"
        lines = []
        for i in range(10):
            lines.append(json.dumps({
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
            }))
        data_file.write_text("\n".join(lines) + "\n")

        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({
            "ds1": {"file_name": "train.jsonl", "sample_ratio": 1.5},
        }))

        result = ds.make_meta_dataset(str(meta_file))
        # 1 full copy (10) + floor(10 * 0.5) = 5 extra = 15
        assert len(result) == 15

    def test_absolute_file_path(self, tmp_path):
        """Absolute file_name paths are used as-is."""
        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            json.dumps({
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
            }) + "\n",
        )

        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({
            "ds1": {"file_name": str(data_file)},
        }))

        result = ds.make_meta_dataset(str(meta_file))
        assert len(result) == 1

    def test_custom_tags(self, tmp_path):
        """Custom tags mapping works end-to-end through make_meta_dataset."""
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            json.dumps({
                "conversations": [
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello"},
                ],
            }) + "\n",
        )

        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({
            "ds1": {
                "file_name": "train.jsonl",
                "columns": {"messages": "conversations"},
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                },
            },
        }))

        result = ds.make_meta_dataset(str(meta_file))
        conv = result[0]["conversation"]
        assert conv[0] == {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
        assert conv[1] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
        }

    # -----------------------------------------------------------------------
    # shard_data tests
    # -----------------------------------------------------------------------

    def _make_10_sample_meta(self, tmp_path):
        """Helper: create a 10-sample JSONL file with a meta JSON."""
        data_file = tmp_path / "train.jsonl"
        lines = []
        for i in range(10):
            lines.append(json.dumps({
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
            }))
        data_file.write_text("\n".join(lines) + "\n")

        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({
            "ds1": {"file_name": "train.jsonl"},
        }))
        return meta_file

    def test_shard_data_rank0_of_2(self, tmp_path):
        """Rank 0 of 2 loads even-indexed samples (0, 2, 4, 6, 8)."""
        meta_file = self._make_10_sample_meta(tmp_path)
        result = ds.make_meta_dataset(str(meta_file), shard_data=True, rank=0, world_size=2)
        assert len(result) == 5
        texts = [r["conversation"][0]["content"][0]["text"] for r in result]
        assert texts == ["Q0", "Q2", "Q4", "Q6", "Q8"]

    def test_shard_data_rank1_of_2(self, tmp_path):
        """Rank 1 of 2 loads odd-indexed samples (1, 3, 5, 7, 9)."""
        meta_file = self._make_10_sample_meta(tmp_path)
        result = ds.make_meta_dataset(str(meta_file), shard_data=True, rank=1, world_size=2)
        assert len(result) == 5
        texts = [r["conversation"][0]["content"][0]["text"] for r in result]
        assert texts == ["Q1", "Q3", "Q5", "Q7", "Q9"]

    def test_shard_data_all_ranks_cover_full_dataset(self, tmp_path):
        """All shards combined cover the dataset with equal counts per rank (tail dropped)."""
        meta_file = self._make_10_sample_meta(tmp_path)
        world_size = 3
        per_rank = 10 // world_size  # 3
        all_texts = []
        for rank in range(world_size):
            result = ds.make_meta_dataset(str(meta_file), shard_data=True, rank=rank, world_size=world_size)
            assert len(result) == per_rank
            all_texts.extend(r["conversation"][0]["content"][0]["text"] for r in result)
        # Tail samples are dropped to ensure equal counts across ranks
        assert len(all_texts) == per_rank * world_size
        assert len(set(all_texts)) == len(all_texts)  # no duplicates

    def test_shard_data_with_sample_ratio(self, tmp_path):
        """sample_ratio is applied before sharding."""
        data_file = tmp_path / "train.jsonl"
        lines = []
        for i in range(10):
            lines.append(json.dumps({
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
            }))
        data_file.write_text("\n".join(lines) + "\n")

        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({
            "ds1": {"file_name": "train.jsonl", "sample_ratio": 0.6},
        }))

        # sample_ratio=0.6 on 10 items -> 6 items, then rank 0/2 gets 3
        result = ds.make_meta_dataset(str(meta_file), shard_data=True, rank=0, world_size=2)
        assert len(result) == 3

    def test_shard_data_with_upsample(self, tmp_path):
        """sample_ratio > 1.0 is applied before sharding."""
        data_file = tmp_path / "train.jsonl"
        lines = []
        for i in range(10):
            lines.append(json.dumps({
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
            }))
        data_file.write_text("\n".join(lines) + "\n")

        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({
            "ds1": {"file_name": "train.jsonl", "sample_ratio": 2.0},
        }))

        # sample_ratio=2.0 on 10 items -> 20 items, then rank 0/2 gets 10
        result = ds.make_meta_dataset(str(meta_file), shard_data=True, rank=0, world_size=2)
        assert len(result) == 10

    def test_shard_data_world_size_1_returns_all(self, tmp_path):
        """world_size=1 returns all data (no-op shard)."""
        meta_file = self._make_10_sample_meta(tmp_path)
        result = ds.make_meta_dataset(str(meta_file), shard_data=True, rank=0, world_size=1)
        assert len(result) == 10

    def test_shard_data_false_returns_all(self, tmp_path):
        """shard_data=False (default) always returns full dataset."""
        meta_file = self._make_10_sample_meta(tmp_path)
        result = ds.make_meta_dataset(str(meta_file), shard_data=False, rank=0, world_size=2)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# Tests for _preload_media
# ---------------------------------------------------------------------------


class TestPreloadMedia:
    """Tests for the _preload_media helper function."""

    def test_loads_image_from_path(self, tmp_path):
        """String path is loaded and converted to a PIL RGB Image."""
        img = Image.new("RGBA", (4, 4), color="red")
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        example = {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text", "text": "Describe."},
                    ],
                },
            ],
        }

        result = vlm_utils._preload_media(example)
        loaded = result["conversation"][0]["content"][0]["image"]
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"

    def test_converts_pil_image_to_rgb(self):
        """An existing PIL Image in non-RGB mode is converted to RGB."""
        rgba_img = Image.new("RGBA", (4, 4), color="blue")
        example = {
            "conversation": [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": rgba_img}],
                },
            ],
        }

        result = vlm_utils._preload_media(example)
        loaded = result["conversation"][0]["content"][0]["image"]
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"

    @pytest.fixture
    def _mock_decord(self, monkeypatch):
        """Mock decord so video tests don't need real files."""
        import numpy as np

        total = 120
        all_frames = np.random.randint(0, 255, (total, 4, 4, 3), dtype=np.uint8)

        class FakeVideoReader:
            def __init__(self, path):
                self.path = path

            def __len__(self):
                return total

            def get_avg_fps(self):
                return 30.0

            def get_batch(self, indices):
                class FakeBatch:
                    def asnumpy(self_inner):
                        return all_frames[list(indices)]
                return FakeBatch()

        fake_decord = type("decord", (), {
            "VideoReader": FakeVideoReader,
            "bridge": type("bridge", (), {"set_bridge": staticmethod(lambda x: None)})(),
        })()
        monkeypatch.setitem(__import__("sys").modules, "decord", fake_decord)

    def test_video_preloaded_to_pil_frames(self, _mock_decord):
        """Video path is decoded into a list of PIL RGB Images."""
        example = {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "/data/clip.mp4"},
                        {"type": "text", "text": "Describe."},
                    ],
                },
            ],
        }

        result = vlm_utils._preload_media(example)
        loaded = result["conversation"][0]["content"][0]["video"]
        assert isinstance(loaded, list)
        assert len(loaded) == 120
        assert all(isinstance(f, Image.Image) for f in loaded)
        assert all(f.mode == "RGB" for f in loaded)

    def test_video_with_frame_indices(self, _mock_decord):
        """Video with frame_indices only reads the specified frames (padded to even)."""
        example = {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": "/data/clip.mp4",
                            "frame_indices": [0, 15, 30, 45, 60],
                        },
                        {"type": "text", "text": "Describe."},
                    ],
                },
            ],
        }

        result = vlm_utils._preload_media(example)
        loaded = result["conversation"][0]["content"][0]["video"]
        assert isinstance(loaded, list)
        # 5 frames → padded to 6 (even alignment)
        assert len(loaded) == 6
        assert all(isinstance(f, Image.Image) for f in loaded)

    def test_text_only_passthrough(self):
        """Examples with only text content are returned unchanged."""
        example = {
            "conversation": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi"}],
                },
            ],
        }

        result = vlm_utils._preload_media(example)
        assert result["conversation"][0]["content"][0]["text"] == "Hello"

    def test_no_conversation_key(self):
        """Example without a 'conversation' key is returned as-is."""
        example = {"other_key": "value"}
        result = vlm_utils._preload_media(example)
        assert result == {"other_key": "value"}

    def test_missing_image_file_raises(self):
        """Missing image file raises an exception."""
        example = {
            "conversation": [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "/nonexistent/path.png"}],
                },
            ],
        }

        with pytest.raises(FileNotFoundError):
            vlm_utils._preload_media(example)


# ---------------------------------------------------------------------------
# Tests for _read_video_frames
# ---------------------------------------------------------------------------


class TestReadVideoFrames:
    """Tests for the _read_video_frames helper function."""

    @pytest.fixture(autouse=True)
    def _mock_decord(self, monkeypatch):
        """Mock decord so tests don't need real video files."""
        import numpy as np

        self._total_frames = 120
        self._video_fps = 30.0
        all_frames = np.random.randint(0, 255, (self._total_frames, 4, 4, 3), dtype=np.uint8)

        class FakeVideoReader:
            def __init__(vr, path):
                vr.path = path

            def __len__(vr):
                return self._total_frames

            def get_avg_fps(vr):
                return self._video_fps

            def get_batch(vr, indices):
                class FakeBatch:
                    def asnumpy(self_inner):
                        return all_frames[list(indices)]
                return FakeBatch()

        fake_decord = type("decord", (), {
            "VideoReader": FakeVideoReader,
            "bridge": type("bridge", (), {"set_bridge": staticmethod(lambda x: None)})(),
        })()
        monkeypatch.setitem(__import__("sys").modules, "decord", fake_decord)

    def test_returns_pil_images(self):
        """Returns a list of PIL RGB Images."""
        frames = vlm_utils._read_video_frames("/fake.mp4")
        assert isinstance(frames, list)
        assert all(isinstance(f, Image.Image) for f in frames)
        assert all(f.mode == "RGB" for f in frames)

    def test_respects_max_frames(self):
        """Frame count is clamped to max_frames from processor."""
        processor = type("P", (), {
            "video_processor": type("VP", (), {"fps": None, "max_frames": 8, "min_frames": 4})(),
        })()
        frames = vlm_utils._read_video_frames("/fake.mp4", processor=processor)
        assert len(frames) == 8

    def test_respects_fps_sampling(self):
        """Frames are subsampled according to target fps."""
        # 120 frames at 30fps video, target 2fps → interval=15 → 8 frames
        processor = type("P", (), {
            "video_processor": type("VP", (), {"fps": 2, "max_frames": None, "min_frames": 4})(),
        })()
        frames = vlm_utils._read_video_frames("/fake.mp4", processor=processor)
        assert len(frames) == 8

    def test_no_processor_reads_all_frames(self):
        """Without processor, all frames are returned."""
        frames = vlm_utils._read_video_frames("/fake.mp4")
        assert len(frames) == self._total_frames

    def test_fps_with_max_frames_clamp(self):
        """fps sampling + max_frames clamp work together."""
        # 120 frames at 30fps, target 10fps → interval=3 → 40 frames, clamp to 16
        processor = type("P", (), {
            "video_processor": type("VP", (), {"fps": 10, "max_frames": 16, "min_frames": 4})(),
        })()
        frames = vlm_utils._read_video_frames("/fake.mp4", processor=processor)
        assert len(frames) == 16

    def test_explicit_frame_indices(self):
        """Explicit frame_indices overrides processor fps/max_frames, padded to even."""
        processor = type("P", (), {
            "video_processor": type("VP", (), {
                "fps": 2, "max_frames": 4, "min_frames": 2, "temporal_patch_size": 2,
            })(),
        })()
        indices = [0, 15, 30, 45, 60]
        frames = vlm_utils._read_video_frames("/fake.mp4", processor=processor, frame_indices=indices)
        # 5 frames → padded to 6 (next even)
        assert len(frames) == 6

    def test_frame_indices_clamped_to_valid_range(self):
        """frame_indices beyond total_frames are clamped to the last frame."""
        # total_frames = 120, so index 999 → 119; 3 frames → padded to 4 (even)
        frames = vlm_utils._read_video_frames("/fake.mp4", frame_indices=[0, 10, 999])
        assert len(frames) == 4

    def test_even_frame_indices_not_padded(self):
        """Even number of frame_indices is not padded."""
        frames = vlm_utils._read_video_frames("/fake.mp4", frame_indices=[0, 10, 20, 30])
        assert len(frames) == 4

    def test_temporal_patch_size_alignment(self):
        """Frame count is aligned to temporal_patch_size from processor."""
        processor = type("P", (), {
            "video_processor": type("VP", (), {
                "fps": None, "max_frames": None, "min_frames": 4, "temporal_patch_size": 4,
            })(),
        })()
        # 120 frames, no fps sampling → 120 frames, 120 % 4 == 0, no padding
        frames = vlm_utils._read_video_frames("/fake.mp4", processor=processor)
        assert len(frames) % 4 == 0

    def test_round_up_not_down(self):
        """Frame count rounds UP to temporal_patch_size boundary, not down.

        This ensures consistency with the sampler, HF video processor,
        and LLaMA-Factory (all round up).
        """
        # 120 frames at 30fps, target 1fps → nframes = 120/30*1 = 4.0
        # max_frames=5 → min(4,5)=4, but let's use a case where rounding matters:
        # 120 frames at 30fps, target 3fps → nframes = 120/30*3 = 12.0
        # max_frames=5 → min(12,5)=5, temporal_patch_size=4
        # Round UP: 5 → 8 (next multiple of 4)
        # Round DOWN would give: 5 → 4
        processor = type("P", (), {
            "video_processor": type("VP", (), {
                "fps": 3, "max_frames": 5, "min_frames": 2, "temporal_patch_size": 4,
            })(),
        })()
        frames = vlm_utils._read_video_frames("/fake.mp4", processor=processor)
        assert len(frames) == 8  # rounded UP from 5 to 8, not down to 4


# ---------------------------------------------------------------------------
# Tests for RobustDatasetWrapper preload toggle
# ---------------------------------------------------------------------------


class TestRobustDatasetWrapperPreload:
    """Tests for the preload_media toggle on RobustDatasetWrapper."""

    def test_preload_default_false(self):
        """preload_media defaults to False and processor to None."""
        wrapper = ds.RobustDatasetWrapper([{"conversation": []}])
        assert wrapper.preload_media is False
        assert wrapper.processor is None

    def test_preload_enabled_returns_pil(self, tmp_path):
        """When preload_media=True, __getitem__ returns PIL Images."""
        img = Image.new("RGB", (4, 4), color="green")
        img_path = tmp_path / "img.png"
        img.save(str(img_path))

        data = [
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": str(img_path)}],
                    },
                ],
            },
        ]
        wrapper = ds.RobustDatasetWrapper(data)
        wrapper.preload_media = True

        result = wrapper[0]
        loaded = result["conversation"][0]["content"][0]["image"]
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"

    def test_preload_disabled_returns_string(self, tmp_path):
        """When preload_media=False, __getitem__ returns path strings."""
        img = Image.new("RGB", (4, 4), color="green")
        img_path = tmp_path / "img.png"
        img.save(str(img_path))

        data = [
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": str(img_path)}],
                    },
                ],
            },
        ]
        wrapper = ds.RobustDatasetWrapper(data)
        # preload_media is False by default

        result = wrapper[0]
        assert result["conversation"][0]["content"][0]["image"] == str(img_path)

    def test_preload_failure_retries(self):
        """When preload fails on one sample, retry picks a different sample."""
        good_img = Image.new("RGB", (4, 4), color="red")
        data = [
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": "/nonexistent.png"}],
                    },
                ],
            },
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": good_img}],
                    },
                ],
            },
        ]
        wrapper = ds.RobustDatasetWrapper(data, max_retries=10)
        wrapper.preload_media = True

        # Requesting index 0 (bad path) should eventually retry and succeed
        # with a random fallback sample
        result = wrapper[0]
        loaded = result["conversation"][0]["content"][0]["image"]
        assert isinstance(loaded, Image.Image)


# ---------------------------------------------------------------------------
# Tests for dataset-level fake image injection (FSDP / Zero3)
# ---------------------------------------------------------------------------


class TestRobustDatasetWrapperFakeImageInjection:
    """RobustDatasetWrapper injects fake images into pure-text samples at __getitem__ time."""

    def test_text_only_gets_fake_image_when_preload(self):
        """Pure-text sample gets a fake image injected when preload_media=True."""
        data = [
            {
                "conversation": [
                    {"role": "user", "content": [{"type": "text", "text": "What is 1+1?"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "2"}]},
                ],
            },
        ]
        wrapper = ds.RobustDatasetWrapper(data)
        wrapper.preload_media = True

        result = wrapper[0]
        # Should have _injected_fake flag
        assert result.get("_injected_fake") is True
        # First user content item should now be a fake image
        user_content = result["conversation"][0]["content"]
        assert user_content[0]["type"] == "image"
        assert isinstance(user_content[0]["image"], Image.Image)
        # Original text should still be present
        assert user_content[1] == {"type": "text", "text": "What is 1+1?"}

    def test_image_sample_not_injected(self, tmp_path):
        """Sample with real image should NOT get fake image injected."""
        img = Image.new("RGB", (4, 4), color="blue")
        img_path = tmp_path / "img.png"
        img.save(str(img_path))

        data = [
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img_path)},
                            {"type": "text", "text": "Describe"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "A blue image"}]},
                ],
            },
        ]
        wrapper = ds.RobustDatasetWrapper(data)
        wrapper.preload_media = True

        result = wrapper[0]
        assert "_injected_fake" not in result
        # Only the original image + text, no extra fake image
        user_content = result["conversation"][0]["content"]
        assert len(user_content) == 2
        assert user_content[0]["type"] == "image"

    def test_no_injection_when_preload_disabled(self):
        """When preload_media=False (eval mode), no injection happens."""
        data = [
            {
                "conversation": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
                ],
            },
        ]
        wrapper = ds.RobustDatasetWrapper(data)
        # preload_media defaults to False

        result = wrapper[0]
        assert "_injected_fake" not in result
        # Content should be unchanged
        user_content = result["conversation"][0]["content"]
        assert len(user_content) == 1
        assert user_content[0]["type"] == "text"

    def test_does_not_mutate_original(self):
        """Injection should not mutate the original dataset sample."""
        original_conv = [
            {"role": "user", "content": [{"type": "text", "text": "test"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
        data = [{"conversation": original_conv}]
        wrapper = ds.RobustDatasetWrapper(data)
        wrapper.preload_media = True

        result = wrapper[0]
        assert result.get("_injected_fake") is True
        # Original conversation should be unchanged
        assert len(original_conv[0]["content"]) == 1
        assert original_conv[0]["content"][0]["type"] == "text"
