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

import pytest
from datasets import Dataset


def _build_tiny_dataset():
    return Dataset.from_dict(
        {
            "ctx": ["ctx 1", "ctx 2"],
            "endings": [
                ["e1_a", "e1_b", "e1_c", "e1_d"],
                ["e2_a", "e2_b", "e2_c", "e2_d"],
            ],
            "label": [2, 0],  # → e1_c, e2_a
            "attention_mask": [[1, 1], [1, 1]],
        }
    )


@pytest.fixture(autouse=True)
def _patch_external_libs(monkeypatch):
    # 1) Patch datasets.load_dataset everywhere
    def _fake_load_dataset(path_or_dataset, split=None, trust_remote_code=True):
        # We only check that the slice expression is propagated
        assert split in (None, "train", "train[:1]")
        return _build_tiny_dataset()

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    # 2) Patch the NeMo pre-processor on its real import path
    class _DummyPreprocessor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def process(self, ds, _parent):
            # Return dataset unchanged
            return ds

    monkeypatch.setattr(
        "nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor",
        _DummyPreprocessor,
        raising=False,
    )

    yield


def test_dataset_basic():
    # Import after patching so the class sees the fakes
    from nemo_automodel.components.datasets.llm.hellaswag import HellaSwag

    dummy_tokenizer = object()
    ds = HellaSwag(path_or_dataset="ignored", tokenizer=dummy_tokenizer)

    # Length
    assert len(ds) == 2

    # Context
    ctxs = ds.get_context(_build_tiny_dataset())
    assert ctxs == ["ctx 1", "ctx 2"]

    # Target
    tgts = ds.get_target(_build_tiny_dataset())
    assert tgts == ["e1_c", "e2_a"]

    # __getitem__ strips attention_mask
    row = ds[0]
    assert "attention_mask" not in row
    assert row["ctx"] == "ctx 1"


def test_sample_limiting():
    from nemo_automodel.components.datasets.llm.hellaswag import HellaSwag

    dummy_tokenizer = object()
    ds = HellaSwag(
        path_or_dataset="ignored",
        tokenizer=dummy_tokenizer,
        num_samples_limit=1,  # forces split 'train[:1]'
    )
    # Our stub still returns the same two rows
    assert len(ds) == 2
