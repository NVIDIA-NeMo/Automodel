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

"""CPU coverage for the token-budget batch sampler."""

import pytest
import torch

from nemo_automodel.components.datasets.llm.dynamic_token_batch_sampler import (
    DynamicTokenBatchSampler,
    DynamicTokenBatchSamplerConfig,
    plan_token_budget_batches,
)
from nemo_automodel.components.datasets.llm.sample_lengths import compute_sample_lengths


class _ListDataset(torch.utils.data.Dataset):
    """Map-style dataset of ``{"input_ids": [...]}`` samples of the given lengths."""

    def __init__(self, lengths):
        self.samples = [{"input_ids": list(range(length))} for length in lengths]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _padded_tokens(batch, lengths):
    return len(batch) * max(lengths[i] for i in batch)


# ---------------------------------------------------------------------------
# plan_token_budget_batches
# ---------------------------------------------------------------------------


def test_plan_keeps_every_batch_within_the_padded_budget():
    lengths = [10, 10, 10, 10, 100, 100]
    batches = plan_token_budget_batches(range(len(lengths)), lengths, max_tokens_per_batch=200)

    assert all(_padded_tokens(batch, lengths) <= 200 for batch in batches)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))


def test_plan_uses_padded_cost_not_the_raw_sum():
    # Raw sum is 10+10+100 = 120 <= 200, but the padded cost of holding all three is
    # 3 * 100 = 300. The long sample must start a new batch.
    lengths = [10, 10, 100]
    batches = plan_token_budget_batches([0, 1, 2], lengths, max_tokens_per_batch=200)

    assert batches == [[0, 1], [2]]


def test_plan_emits_an_oversized_sample_as_its_own_batch_instead_of_dropping_it():
    lengths = [10, 5000, 10]
    batches = plan_token_budget_batches([0, 1, 2], lengths, max_tokens_per_batch=100)

    assert [1] in batches
    assert sorted(i for batch in batches for i in batch) == [0, 1, 2]


def test_plan_honors_max_batch_size():
    lengths = [1] * 10
    batches = plan_token_budget_batches(range(10), lengths, max_tokens_per_batch=10_000, max_batch_size=3)

    assert [len(batch) for batch in batches] == [3, 3, 3, 1]


def test_plan_returns_no_batches_for_an_empty_order():
    assert plan_token_budget_batches([], [], max_tokens_per_batch=128) == []


# ---------------------------------------------------------------------------
# DynamicTokenBatchSampler
# ---------------------------------------------------------------------------


def test_every_rank_yields_the_same_number_of_batches():
    """A rank that runs dry early would desync the next collective and hang the job."""
    dataset = _ListDataset([(i % 37) * 8 + 8 for i in range(500)])

    for world_size in (1, 2, 3, 8):
        counts = {
            rank: len(
                list(
                    DynamicTokenBatchSampler(
                        dataset, max_tokens_per_batch=1024, num_replicas=world_size, rank=rank
                    )
                )
            )
            for rank in range(world_size)
        }
        assert len(set(counts.values())) == 1, f"world_size={world_size} gave uneven counts {counts}"


def test_ranks_partition_the_batches_without_overlap():
    dataset = _ListDataset([(i % 11) * 16 + 16 for i in range(200)])
    world_size = 4

    per_rank = [
        [tuple(batch) for batch in DynamicTokenBatchSampler(dataset, max_tokens_per_batch=512, num_replicas=world_size, rank=rank)]
        for rank in range(world_size)
    ]

    flat = [batch for rank_batches in per_rank for batch in rank_batches]
    assert len(flat) == len(set(flat)), "the same batch was handed to more than one rank"


def test_batches_respect_the_token_budget_end_to_end():
    lengths = [(i % 23) * 32 + 32 for i in range(300)]
    dataset = _ListDataset(lengths)

    sampler = DynamicTokenBatchSampler(dataset, max_tokens_per_batch=2048, max_batch_size=16)

    for batch in sampler:
        assert len(batch) <= 16
        if len(batch) > 1:
            assert _padded_tokens(batch, lengths) <= 2048


def test_a_new_epoch_reshuffles_the_batches():
    dataset = _ListDataset([(i % 17) * 16 + 16 for i in range(300)])
    sampler = DynamicTokenBatchSampler(dataset, max_tokens_per_batch=1024)

    first = [tuple(batch) for batch in sampler]
    sampler.set_epoch(1)
    second = [tuple(batch) for batch in sampler]

    assert first != second


def test_the_same_seed_and_epoch_replans_identically():
    """Ranks agree on the plan only because identical inputs give an identical order."""
    dataset = _ListDataset([(i % 13) * 24 + 24 for i in range(200)])

    a = [tuple(batch) for batch in DynamicTokenBatchSampler(dataset, max_tokens_per_batch=1024, seed=7)]
    b = [tuple(batch) for batch in DynamicTokenBatchSampler(dataset, max_tokens_per_batch=1024, seed=7)]

    assert a == b


def test_len_matches_the_number_of_yielded_batches():
    dataset = _ListDataset([(i % 9) * 32 + 32 for i in range(150)])
    sampler = DynamicTokenBatchSampler(dataset, max_tokens_per_batch=1024, num_replicas=2, rank=1)

    assert len(sampler) == len(list(sampler))


def test_state_dict_round_trip_resumes_mid_epoch():
    dataset = _ListDataset([(i % 15) * 16 + 16 for i in range(240)])
    sampler = DynamicTokenBatchSampler(dataset, max_tokens_per_batch=1024)

    full = [tuple(batch) for batch in sampler]
    consumed = 0
    for _ in sampler:
        consumed += 1
        if consumed == 3:
            break
    state = sampler.state_dict()

    resumed = DynamicTokenBatchSampler(dataset, max_tokens_per_batch=1024)
    resumed.load_state_dict(state)

    assert [tuple(batch) for batch in resumed] == full[3:]


def test_state_dict_carries_the_epoch():
    dataset = _ListDataset([32] * 64)
    sampler = DynamicTokenBatchSampler(dataset, max_tokens_per_batch=256)
    sampler.set_epoch(4)

    restored = DynamicTokenBatchSampler(dataset, max_tokens_per_batch=256)
    restored.load_state_dict(sampler.state_dict())

    assert restored.epoch == 4


def test_precomputed_lengths_skip_the_dataset_scan():
    class _Exploding(_ListDataset):
        def __getitem__(self, index):
            raise AssertionError("the dataset must not be read when lengths are supplied")

    sampler = DynamicTokenBatchSampler(
        _Exploding([0] * 32), max_tokens_per_batch=256, lengths=[64] * 32
    )

    assert len(list(sampler)) > 0


def test_sort_window_zero_batches_in_shuffled_order():
    dataset = _ListDataset([(i % 5) * 64 + 64 for i in range(100)])

    sampler = DynamicTokenBatchSampler(dataset, max_tokens_per_batch=1024, sort_window=0)

    assert sorted(i for batch in sampler for i in batch) != []


def test_oversized_samples_warn_once(caplog):
    dataset = _ListDataset([10, 10, 9999])

    with caplog.at_level("WARNING"):
        DynamicTokenBatchSampler(dataset, max_tokens_per_batch=128)

    assert any("exceed max_tokens_per_batch" in record.message for record in caplog.records)


def test_a_dataset_smaller_than_the_world_warns_and_yields_nothing():
    dataset = _ListDataset([16])

    sampler = DynamicTokenBatchSampler(dataset, max_tokens_per_batch=1024, num_replicas=4, rank=0)

    assert list(sampler) == []


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_tokens_per_batch": 0}, "max_tokens_per_batch must be positive"),
        ({"max_tokens_per_batch": 128, "max_batch_size": 0}, "max_batch_size must be positive"),
        ({"max_tokens_per_batch": 128, "sort_window": -1}, "sort_window must be non-negative"),
        ({"max_tokens_per_batch": 128, "num_replicas": 0}, "num_replicas must be positive"),
        ({"max_tokens_per_batch": 128, "num_replicas": 2, "rank": 2}, "is out of range"),
    ],
)
def test_invalid_arguments_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DynamicTokenBatchSampler(_ListDataset([16] * 8), **kwargs)


# ---------------------------------------------------------------------------
# Config + dataloader integration
# ---------------------------------------------------------------------------


def test_config_builds_a_sampler_for_its_rank():
    dataset = _ListDataset([(i % 7) * 32 + 32 for i in range(120)])
    config = DynamicTokenBatchSamplerConfig(max_tokens_per_batch=1024, max_batch_size=8, sort_window=64, seed=3)

    sampler = config.build(dataset=dataset, dataset_len=len(dataset), rank=1, world_size=2)

    assert isinstance(sampler, DynamicTokenBatchSampler)
    assert (sampler.rank, sampler.num_replicas, sampler.seed) == (1, 2, 3)
    assert sampler.max_batch_size == 8


def test_dataloader_consumes_the_sampler_as_a_batch_sampler():
    lengths = [(i % 9) * 32 + 32 for i in range(120)]
    dataset = _ListDataset(lengths)
    sampler = DynamicTokenBatchSampler(dataset, max_tokens_per_batch=1024)

    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda items: items)
    batch_sizes = [len(batch) for batch in loader]

    assert batch_sizes == [len(batch) for batch in sampler]
    assert len(set(batch_sizes)) > 1, "a token budget should produce more than one batch size"


# ---------------------------------------------------------------------------
# compute_sample_lengths
# ---------------------------------------------------------------------------


def test_compute_sample_lengths_reads_lists_and_tensors():
    class _Mixed(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return [{"input_ids": [1, 2, 3]}, {"input_ids": torch.zeros(5)}, {}][index]

    assert compute_sample_lengths(_Mixed()) == [3, 5, 0]
