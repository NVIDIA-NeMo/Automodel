# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for LengthGroupedSampler stateful checkpoint/resume logic."""

from nemo_automodel.components.datasets.llm.length_grouped_sampler import LengthGroupedSampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(lengths):
    """Create a simple list dataset with given token lengths."""
    return [{"input_ids": list(range(length))} for length in lengths]


def _collect(sampler):
    """Exhaust a sampler and return all indices as a list."""
    return list(sampler)


# ---------------------------------------------------------------------------
# state_dict / load_state_dict
# ---------------------------------------------------------------------------


class TestStateDictBasic:
    def test_state_dict_initial(self):
        ds = _make_dataset([10, 20, 30, 40])
        sampler = LengthGroupedSampler(ds, batch_size=2, seed=0, num_replicas=1, rank=0)
        state = sampler.state_dict()
        assert state == {"yielded": 0, "epoch": 0}

    def test_state_dict_after_full_iteration(self):
        ds = _make_dataset([10, 20, 30, 40])
        sampler = LengthGroupedSampler(ds, batch_size=2, seed=0, num_replicas=1, rank=0)
        _collect(sampler)
        state = sampler.state_dict()
        assert state["yielded"] == len(sampler)
        assert state["epoch"] == 0

    def test_state_dict_after_set_epoch(self):
        ds = _make_dataset([10, 20, 30, 40])
        sampler = LengthGroupedSampler(ds, batch_size=2, seed=0, num_replicas=1, rank=0)
        sampler.set_epoch(3)
        _collect(sampler)
        state = sampler.state_dict()
        assert state["epoch"] == 3

    def test_load_state_dict_restores_epoch(self):
        ds = _make_dataset([10, 20, 30, 40])
        sampler = LengthGroupedSampler(ds, batch_size=2, seed=0, num_replicas=1, rank=0)
        sampler.load_state_dict({"yielded": 0, "epoch": 5})
        assert sampler.epoch == 5


# ---------------------------------------------------------------------------
# Mid-epoch resume
# ---------------------------------------------------------------------------


class TestMidEpochResume:
    def test_resume_yields_remaining_indices(self):
        """Interrupting at index K and resuming should yield the remaining N-K indices."""
        ds = _make_dataset([5, 10, 15, 20, 25, 30, 35, 40])
        sampler = LengthGroupedSampler(ds, batch_size=2, seed=42, num_replicas=1, rank=0)

        # Full pass to get reference order
        full = _collect(sampler)

        # Simulate interruption after 3 samples
        stop_after = 3
        sampler2 = LengthGroupedSampler(ds, batch_size=2, seed=42, num_replicas=1, rank=0)
        partial = []
        for i, idx in enumerate(sampler2):
            partial.append(idx)
            if i + 1 == stop_after:
                break

        # Save state and resume
        state = sampler2.state_dict()
        assert state["yielded"] == stop_after

        sampler3 = LengthGroupedSampler(ds, batch_size=2, seed=42, num_replicas=1, rank=0)
        sampler3.load_state_dict(state)
        remaining = _collect(sampler3)

        assert partial + remaining == full

    def test_resume_at_zero_gives_full_epoch(self):
        ds = _make_dataset([10, 20, 30, 40])
        sampler = LengthGroupedSampler(ds, batch_size=2, seed=7, num_replicas=1, rank=0)
        full = _collect(sampler)

        sampler2 = LengthGroupedSampler(ds, batch_size=2, seed=7, num_replicas=1, rank=0)
        sampler2.load_state_dict({"yielded": 0, "epoch": 0})
        resumed = _collect(sampler2)

        assert resumed == full

    def test_resume_at_end_yields_nothing(self):
        ds = _make_dataset([10, 20, 30, 40])
        sampler = LengthGroupedSampler(ds, batch_size=2, seed=0, num_replicas=1, rank=0)
        n = len(sampler)
        sampler.load_state_dict({"yielded": n, "epoch": 0})
        assert _collect(sampler) == []

    def test_resume_preserves_epoch_seed(self):
        """Resuming at epoch 2 should produce the same order as a fresh epoch 2 run."""
        ds = _make_dataset([5, 10, 15, 20, 25, 30, 35, 40])
        sampler_fresh = LengthGroupedSampler(ds, batch_size=2, seed=0, num_replicas=1, rank=0)
        sampler_fresh.set_epoch(2)
        full_epoch2 = _collect(sampler_fresh)

        # Resume at epoch 2 from halfway
        mid = len(full_epoch2) // 2
        sampler_resume = LengthGroupedSampler(ds, batch_size=2, seed=0, num_replicas=1, rank=0)
        sampler_resume.load_state_dict({"yielded": mid, "epoch": 2})
        remaining = _collect(sampler_resume)

        assert remaining == full_epoch2[mid:]


# ---------------------------------------------------------------------------
# Yielded counter
# ---------------------------------------------------------------------------


class TestYieldedTracking:
    def test_yielded_increments(self):
        ds = _make_dataset([10, 20, 30, 40])
        sampler = LengthGroupedSampler(ds, batch_size=2, seed=0, num_replicas=1, rank=0)
        it = iter(sampler)
        next(it)
        assert sampler.yielded == 1
        next(it)
        assert sampler.yielded == 2

    def test_yielded_resets_on_new_iter(self):
        ds = _make_dataset([10, 20, 30, 40])
        sampler = LengthGroupedSampler(ds, batch_size=2, seed=0, num_replicas=1, rank=0)
        _collect(sampler)
        assert sampler.yielded == len(sampler)
        # Generator body runs on first next(), which resets yielded to 0 then increments to 1
        it = iter(sampler)
        next(it)
        assert sampler.yielded == 1

    def test_next_yielded_consumed_once(self):
        """_next_yielded should be consumed on first __iter__ call, not persist."""
        ds = _make_dataset([10, 20, 30, 40])
        sampler = LengthGroupedSampler(ds, batch_size=2, seed=0, num_replicas=1, rank=0)
        sampler.load_state_dict({"yielded": 2, "epoch": 0})

        # First iteration skips 2
        first = _collect(sampler)
        assert len(first) == len(sampler) - 2

        # Second iteration without another load_state_dict should yield all
        second = _collect(sampler)
        assert len(second) == len(sampler)
