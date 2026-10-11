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

import itertools

import pytest
import torch

from nemo_automodel.components.datasets.llm.mock_iterable_dataset import MockIterableDataset, MockIterableDatasetConfig
from nemo_automodel.components.datasets.loader import DataloaderConfig, ParallelAwareDataloader


def _fingerprints(batch: dict[str, torch.Tensor]) -> tuple[bytes, bytes, bytes]:
    """Serialize each CPU int64 [batch, sequence] tensor without changing its values.

    Args:
        batch: Input IDs, shifted labels, and positions, each [batch, sequence].

    Returns:
        Complete byte strings in input, label, position order.
    """
    return tuple(batch[key].numpy().tobytes() for key in ("input_ids", "labels", "position_ids"))


def _loader(dp_rank: int, *, workers: int = 0, seed: int | None = 42) -> ParallelAwareDataloader:
    config = DataloaderConfig(
        dataset_config=MockIterableDatasetConfig(
            vocab_size=131, seq_len=32, num_samples=29, batch_size=2, exclude_token_ids=[1, 7], seed=seed
        ),
        batch_size=None,
        num_workers=workers,
        seed=177,
    )
    return config.build(dp_rank=dp_rank, dp_world_size=4, pp_enabled=True)


def test_pipeline_peers_use_same_native_loader_samples_despite_global_rng() -> None:
    peers = {}
    for global_rank in range(8):
        torch.manual_seed(937 + global_rank)
        loader = _loader(global_rank % 4)
        iterator = iter(loader)
        batches = []
        for _ in range(6):
            torch.rand(global_rank + 11)  # model construction/other CPU RNG use may differ by stage
            batch = next(iterator)
            assert torch.equal(batch["labels"][:, :-1], batch["input_ids"][:, 1:])
            assert (batch["labels"][:, -1] == -100).all()
            assert not torch.isin(batch["input_ids"], torch.tensor([1, 7])).any()
            batches.append(_fingerprints(batch))
        peers[global_rank] = batches
    for dp_rank in range(4):
        assert peers[dp_rank] == peers[dp_rank + 4]
    assert len({peers[rank][0][0] for rank in range(4)}) == 4


def test_old_process_global_rng_negative_control(monkeypatch: pytest.MonkeyPatch) -> None:
    original = MockIterableDataset._sample_tokens

    def old_global_rng(self: MockIterableDataset, generator: torch.Generator | None = None) -> torch.Tensor:
        """Reproduce the old global-RNG sampling; return IDs [batch, sequence]."""
        return original(self, generator=None)

    monkeypatch.setattr(MockIterableDataset, "_sample_tokens", old_global_rng)
    with pytest.raises(AssertionError):
        test_pipeline_peers_use_same_native_loader_samples_despite_global_rng()


def test_seeded_dataset_does_not_consume_global_rng() -> None:
    torch.manual_seed(13)
    state = torch.get_rng_state().clone()
    list(MockIterableDataset(vocab_size=31, seq_len=9, num_samples=3, seed=0))
    assert torch.equal(state, torch.get_rng_state())


def test_seed_none_preserves_existing_stream_and_length() -> None:
    torch.manual_seed(99)
    expected = [torch.randint(0, 31, (2, 9)) for _ in range(3)]
    torch.manual_seed(99)
    data = MockIterableDataset(vocab_size=31, seq_len=9, num_samples=3, batch_size=2).shard(4, 2)
    assert len(data) == 3
    assert all(torch.equal(batch["input_ids"], tokens) for batch, tokens in zip(data, expected))


def test_dp_shards_reconstruct_global_stream_and_do_not_mutate_parent() -> None:
    dataset = MockIterableDataset(vocab_size=31, seq_len=9, num_samples=11, seed=13)
    expected = [_fingerprints(batch) for batch in dataset]
    shards = [dataset.shard(4, rank) for rank in range(4)]
    assert len(dataset) == 11
    assert [len(shard) for shard in shards] == [3, 3, 3, 2]
    actual = {
        rank + local * 4: _fingerprints(batch) for rank, shard in enumerate(shards) for local, batch in enumerate(shard)
    }
    assert [actual[index] for index in range(11)] == expected
    assert [_fingerprints(batch) for batch in dataset.shard(2, 1).shard(2, 1)] == expected[3::4]
    assert len(dataset.shard(20, 17)) == 0
    assert list(dataset.shard(20, 17)) == []


def test_stateful_native_loader_resume_reproduces_remaining_samples() -> None:
    torch.manual_seed(3)
    loader = _loader(1)
    iterator = iter(loader)
    list(itertools.islice(iterator, 3))
    state = loader.state_dict()
    expected = [_fingerprints(batch) for batch in iterator]
    torch.manual_seed(789)
    restored = _loader(1)
    restored.load_state_dict(state)
    assert [_fingerprints(batch) for batch in restored] == expected


@pytest.mark.runtime_budget(
    20, reason="Two real spawned StatefulDataLoader workers must import the native dataset stack"
)
def test_native_loader_workers_reconstruct_single_worker_sequence() -> None:
    expected = [_fingerprints(batch) for batch in _loader(2)]
    torch.manual_seed(812)
    assert [_fingerprints(batch) for batch in _loader(2, workers=2)] == expected


def test_repeated_iterators_and_wraparound_seed() -> None:
    dataset = MockIterableDataset(vocab_size=31, seq_len=9, num_samples=3, seed=2**64 - 1)
    assert [_fingerprints(batch) for batch in dataset] == [_fingerprints(batch) for batch in dataset]


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_invalid_seed_fails(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        MockIterableDataset(seed=seed)


@pytest.mark.parametrize("shards,index", [(0, 0), (2, 2), (2, -1)])
def test_invalid_shard_fails(shards: int, index: int) -> None:
    with pytest.raises(ValueError, match="shard"):
        MockIterableDataset(seed=0).shard(shards, index)
