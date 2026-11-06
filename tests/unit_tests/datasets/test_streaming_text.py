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
"""Fixed-length real text retains token alignment across documents and resumes."""

from itertools import islice

import pytest
import torch
from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_automodel.components.datasets.llm.streaming_text import StreamingTextConfig, StreamingTextDataset


class IntegerTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [int(value) for value in text.split()]


def make_data() -> StreamingTextDataset:
    source = Dataset.from_dict({"text": ["1 2 3", "4 5", "6 7 8 9"]}).to_iterable_dataset()
    return StreamingTextDataset(source, IntegerTokenizer(), 4)


def test_fixed_length_document_and_chunk_boundaries() -> None:
    chunks = list(islice(make_data(), 3))
    assert [c["input_ids"].tolist() for c in chunks] == [[1, 2, 3, 0], [4, 5, 0, 6], [7, 8, 9, 0]]
    assert [c["labels"].tolist() for c in chunks] == [[2, 3, 0, 4], [5, 0, 6, 7], [8, 9, 0, 1]]
    assert all(c["labels"].dtype == torch.long for c in chunks)
    assert all(set(c) == {"input_ids", "labels"} for c in chunks)


def test_dp_sharding_with_fewer_source_files_than_ranks() -> None:
    data = make_data()
    first = next(iter(data.shard(2, 0)))
    second = next(iter(data.shard(2, 1)))
    assert first["input_ids"].tolist() == [1, 2, 3, 0]
    assert second["input_ids"].tolist() == [4, 5, 0, 4]
    assert next(iter(data))["labels"].tolist() == [2, 3, 0, 4]


@pytest.mark.parametrize("workers", [0, 2])
def test_stateful_loader_resumes_mid_document(workers: int) -> None:
    source = Dataset.from_dict({"text": ["1 2 3 4 5 6 7 8 9", "10 11 12 13 14 15 16 17 18"]})

    def loader() -> StatefulDataLoader:
        data = StreamingTextDataset(source.to_iterable_dataset(num_shards=2), IntegerTokenizer(), 3)
        return StatefulDataLoader(data, batch_size=2, num_workers=workers)

    original = loader()
    iterator = iter(original)
    next(iterator)
    state = original.state_dict()
    expected = list(islice(iterator, 4))
    resumed = loader()
    resumed.load_state_dict(state)
    actual = list(islice(iter(resumed), 4))
    for a, b in zip(actual, expected, strict=True):
        for key in a:
            torch.testing.assert_close(a[key], b[key])


def test_empty_source_fails_instead_of_hanging() -> None:
    source = Dataset.from_dict({"text": [""]}).to_iterable_dataset()
    with pytest.raises(ValueError, match="no nonempty"):
        next(iter(StreamingTextDataset(source, IntegerTokenizer(), 4)))


@pytest.mark.parametrize("length", [0, -1])
def test_invalid_chunk_length(length: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        StreamingTextConfig(seq_length=length).build(tokenizer=IntegerTokenizer())


def test_missing_tokenizer_fails_before_network_access() -> None:
    with pytest.raises(ValueError, match="EOS"):
        StreamingTextConfig().build(tokenizer=None)


def test_idle_workers_with_one_source_shard() -> None:
    loader = StatefulDataLoader(make_data(), batch_size=2, num_workers=4)
    batches = list(islice(loader, 3))
    assert all(batch["input_ids"].shape == (2, 4) for batch in batches)


def test_typed_recipe_uses_canonical_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.components.datasets.llm import streaming_text
    from nemo_automodel.recipes._typed_config import RecipeConfig

    monkeypatch.setattr(streaming_text.datasets, "load_dataset", lambda *args, **kwargs: make_data().source)
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.llm.streaming_text.StreamingTextConfig",
                    "seq_length": 4,
                },
                "dataloader": {"collate_fn": "torch.utils.data.default_collate"},
                "step_scheduler": {"local_batch_size": 3, "global_batch_size": 6, "max_steps": 2},
            }
        )
    ).dataloader
    loader = config.build(tokenizer=IntegerTokenizer(), dp_rank=1, dp_world_size=2)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (3, 4)
    assert batch["input_ids"][0].tolist() == [4, 5, 0, 4]
