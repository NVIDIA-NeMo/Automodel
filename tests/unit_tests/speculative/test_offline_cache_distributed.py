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

"""CPU unit tests for the distributed offline-cache writer.

Covers the contiguous, shard-aligned block partition (:func:`partition_cache_shards`)
and the distributed write loop (:func:`write_cache_shards_distributed`): every shard
is owned by exactly one rank, the union of per-rank writes reconstructs the identical
sequential cache the single-process path produces, and ranks pad their forward-step
count to a global maximum (the collective-lockstep guarantee) while writing only real
samples. Multi-rank behavior is simulated by calling the writer once per rank into a
shared directory with a fixed ``sync_max_steps`` (standing in for the all-reduce MAX).
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from nemo_automodel.components.datasets.llm.dspark_cache import CachedDSparkDataset, write_manifest, write_shard
from nemo_automodel.components.datasets.llm.offline_cache import (
    partition_cache_shards,
    write_cache_shards,
    write_cache_shards_distributed,
)

_SEQ = 4
_HIDDEN = 8
_LAYERS = [1, 3]

logger = logging.getLogger(__name__)


class _TokenDataset(Dataset):
    """A tiny map-style dataset of identifiable tokenized samples."""

    def __init__(self, num_samples: int):
        self._n = num_samples

    def __len__(self):
        return self._n

    def __getitem__(self, index):
        # Encode the global index into every position so cache order is verifiable.
        return {
            "input_ids": torch.full((_SEQ,), index, dtype=torch.long),
            "attention_mask": torch.ones(_SEQ, dtype=torch.long),
            "loss_mask": torch.ones(_SEQ, dtype=torch.long),
        }


def _collate(features):
    return {k: torch.stack([f[k] for f in features], dim=0) for k in features[0]}


def _make_compute_batch(counter: list[int]):
    def _compute(batch):
        counter[0] += 1
        bsz = batch["input_ids"].shape[0]
        return {
            "input_ids": batch["input_ids"],
            "loss_mask": batch["loss_mask"],
            "target_hidden_states": batch["input_ids"][..., None].float().expand(bsz, _SEQ, _HIDDEN * len(_LAYERS)),
            "target_last_hidden_states": batch["input_ids"][..., None].float().expand(bsz, _SEQ, _HIDDEN),
        }

    return _compute


def _full_loader(num_samples: int, batch_size: int) -> DataLoader:
    return DataLoader(
        _TokenDataset(num_samples), batch_size=batch_size, shuffle=False, collate_fn=_collate, drop_last=False
    )


def _write_manifest_for(cache_dir: str, num_samples: int, shard_size: int) -> None:
    write_manifest(
        cache_dir,
        {
            "target_model": "tiny",
            "target_model_type": "qwen3",
            "target_vocab_size": 16,
            "hidden_size": _HIDDEN,
            "num_hidden_layers": 6,
            "seq_length": _SEQ,
            "dtype": "fp32",
            "num_samples": num_samples,
            "shard_size": shard_size,
            "target_hidden_dim": _HIDDEN * len(_LAYERS),
            "target_last_hidden_dim": _HIDDEN,
            "target_layer_ids": list(_LAYERS),
        },
    )


def _run_all_ranks(cache_dir: str, num_samples: int, shard_size: int, batch_size: int, world_size: int) -> int:
    """Simulate every rank's write into a shared dir; return total compute_batch calls."""
    # The real all-reduce MAX is simulated by pre-computing the global max step count.
    local_steps = []
    for rank in range(world_size):
        _, _, n_local = partition_cache_shards(num_samples, shard_size, world_size, rank)
        local_steps.append((n_local + batch_size - 1) // batch_size)
    global_max = max(local_steps) if local_steps else 0

    counter = [0]
    for rank in range(world_size):
        write_cache_shards_distributed(
            dataloader=_full_loader(num_samples, batch_size),
            output_dir=cache_dir,
            shard_size=shard_size,
            world_size=world_size,
            rank=rank,
            compute_batch=_make_compute_batch(counter),
            write_shard_fn=write_shard,
            logger=logger,
            sync_max_steps=lambda _local, _m=global_max: _m,
        )
    return counter[0]


# ---------------------------------------------------------------------------
# partition_cache_shards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_samples,shard_size,world_size",
    [
        (10, 2, 1),
        (10, 2, 3),
        (10, 2, 5),
        (10, 2, 8),  # world_size > total_shards -> some ranks get 0
        (9, 2, 4),  # partial last shard
        (7, 4, 3),
        (0, 2, 2),  # empty dataset
    ],
)
def test_partition_tiles_samples_without_overlap(num_samples, shard_size, world_size):
    covered = []
    total_shards = (num_samples + shard_size - 1) // shard_size
    seen_shards = 0
    for rank in range(world_size):
        start_shard, start_sample, n_local = partition_cache_shards(num_samples, shard_size, world_size, rank)
        if n_local > 0:
            covered.append((start_sample, start_sample + n_local))
            assert start_sample == start_shard * shard_size
        seen_shards += (n_local + shard_size - 1) // shard_size
    covered.sort()
    # Contiguous, non-overlapping cover of exactly [0, num_samples).
    assert sum(end - start for start, end in covered) == num_samples
    expected_start = 0
    for start, end in covered:
        assert start == expected_start
        expected_start = end
    assert expected_start == num_samples
    assert seen_shards == total_shards


def test_partition_rejects_bad_rank_and_world_size():
    with pytest.raises(ValueError, match="world_size"):
        partition_cache_shards(4, 2, 0, 0)
    with pytest.raises(ValueError, match="rank"):
        partition_cache_shards(4, 2, 2, 2)


# ---------------------------------------------------------------------------
# write_cache_shards_distributed
# ---------------------------------------------------------------------------


def test_single_rank_matches_single_process_writer(tmp_path):
    """world_size=1 must produce exactly the sequential single-process cache."""
    dist_dir = str(tmp_path / "dist")
    seq_dir = str(tmp_path / "seq")
    num_samples, shard_size, batch_size = 6, 2, 2

    _write_manifest_for(dist_dir, num_samples, shard_size)
    _run_all_ranks(dist_dir, num_samples, shard_size, batch_size, world_size=1)

    _write_manifest_for(seq_dir, num_samples, shard_size)
    write_cache_shards(
        dataloader=_full_loader(num_samples, batch_size),
        output_dir=seq_dir,
        shard_size=shard_size,
        start_shard_index=0,
        compute_batch=_make_compute_batch([0]),
        write_shard_fn=write_shard,
        logger=logger,
    )

    dist_ds, seq_ds = CachedDSparkDataset(dist_dir), CachedDSparkDataset(seq_dir)
    assert len(dist_ds) == len(seq_ds) == num_samples
    for i in range(num_samples):
        torch.testing.assert_close(dist_ds[i]["input_ids"], seq_ds[i]["input_ids"])


@pytest.mark.parametrize(
    "num_samples,shard_size,batch_size,world_size",
    [
        (12, 2, 2, 3),  # even split
        (12, 2, 1, 4),  # batch_size 1
        (9, 3, 3, 2),  # uneven blocks + partial tail
        (10, 2, 2, 8),  # more ranks than shards
        (5, 5, 1, 3),  # single shard, extra ranks idle
    ],
)
def test_multi_rank_reconstructs_ordered_cache(tmp_path, num_samples, shard_size, batch_size, world_size):
    cache_dir = str(tmp_path / "cache")
    _write_manifest_for(cache_dir, num_samples, shard_size)
    total_calls = _run_all_ranks(cache_dir, num_samples, shard_size, batch_size, world_size)

    dataset = CachedDSparkDataset(cache_dir)
    assert len(dataset) == num_samples
    # Global order is preserved: sample i encodes the global index i in every position.
    for i in range(num_samples):
        assert torch.equal(dataset[i]["input_ids"], torch.full((_SEQ,), i, dtype=torch.long))

    # Lockstep: every rank runs the same number of forward steps (global max), so the
    # total call count is world_size * global_max, not just the real-sample steps.
    local_steps = [
        (partition_cache_shards(num_samples, shard_size, world_size, r)[2] + batch_size - 1) // batch_size
        for r in range(world_size)
    ]
    assert total_calls == world_size * max(local_steps)


def test_empty_dataset_writes_nothing_and_does_not_crash(tmp_path):
    """num_samples == 0 must not read dataset[0] for the dummy batch (lazy build)."""
    cache_dir = str(tmp_path / "cache")
    counter = [0]
    for rank in range(2):
        write_cache_shards_distributed(
            dataloader=_full_loader(0, 2),
            output_dir=cache_dir,
            shard_size=2,
            world_size=2,
            rank=rank,
            compute_batch=_make_compute_batch(counter),
            write_shard_fn=write_shard,
            logger=logger,
            sync_max_steps=lambda _local: 0,
        )
    assert counter[0] == 0
    import os

    assert not any(name.startswith("shard-") for name in os.listdir(cache_dir)) if os.path.isdir(cache_dir) else True


def test_local_loader_propagates_worker_settings(tmp_path, monkeypatch):
    """The per-rank Subset loader must inherit the source loader's worker settings.

    The precompute loads the target onto CUDA before iterating, so dropping the
    source's ``multiprocessing_context`` (forkserver) would fork workers that inherit
    a live CUDA context and abort. Capture the DataLoader kwargs instead of spawning
    real workers.
    """
    import nemo_automodel.components.datasets.llm.offline_cache as oc

    captured = {}
    real_loader_cls = oc.DataLoader

    class _CapturingLoader(real_loader_cls):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            kwargs.pop("multiprocessing_context", None)
            kwargs.pop("persistent_workers", None)
            kwargs["num_workers"] = 0  # do not actually spawn workers in the test
            super().__init__(*args, **kwargs)

    source = real_loader_cls(
        _TokenDataset(4),
        batch_size=2,
        shuffle=False,
        collate_fn=_collate,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="forkserver",
        drop_last=False,
    )
    monkeypatch.setattr(oc, "DataLoader", _CapturingLoader)
    write_cache_shards_distributed(
        dataloader=source,
        output_dir=str(tmp_path / "cache"),
        shard_size=2,
        world_size=1,
        rank=0,
        compute_batch=_make_compute_batch([0]),
        write_shard_fn=write_shard,
        logger=logger,
    )
    assert captured["num_workers"] == 2
    assert captured["persistent_workers"] is True
    assert captured["multiprocessing_context"] is not None


def test_idle_rank_still_runs_lockstep_forwards_but_writes_nothing(tmp_path):
    """A rank that owns no shards must still enter every collective (dummy forwards)."""
    cache_dir = str(tmp_path / "cache")
    num_samples, shard_size, batch_size, world_size = 2, 2, 1, 4  # 1 shard, ranks 1..3 idle
    _write_manifest_for(cache_dir, num_samples, shard_size)

    per_rank_calls = []
    global_max = 2  # rank 0 has 2 samples at batch_size 1
    for rank in range(world_size):
        counter = [0]
        write_cache_shards_distributed(
            dataloader=_full_loader(num_samples, batch_size),
            output_dir=cache_dir,
            shard_size=shard_size,
            world_size=world_size,
            rank=rank,
            compute_batch=_make_compute_batch(counter),
            write_shard_fn=write_shard,
            logger=logger,
            sync_max_steps=lambda _local: global_max,
        )
        per_rank_calls.append(counter[0])

    assert per_rank_calls == [global_max] * world_size  # idle ranks pad with dummy forwards
    assert len(CachedDSparkDataset(cache_dir)) == num_samples
