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
from typing import Dict, List

import pytest

import nemo_automodel.components.datasets.vlm.datasets as ds


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
        assert user_turn["content"] == "<|audio_1|>Transcribe the Turkish audio clip."

        # Test assistant turn
        assistant_turn = conversation[1]
        assert assistant_turn["role"] == "assistant"
        assert assistant_turn["content"] == src["transcription"]

        # Test audio data processing
        audio_array, sampling_rate = sample["audio"]
        assert audio_array == src["audio"]["array"]
        assert sampling_rate == src["audio"]["sampling_rate"]
