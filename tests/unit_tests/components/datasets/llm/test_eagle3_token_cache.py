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

"""Unit tests for the EAGLE-3 draft-vocab token-map cache.

The recipe otherwise scans the whole training set every setup to rank token
frequencies; ``selected_token_ids_path`` caches the result so reruns skip the
scan. These tests cover save/load round-trip, the incompatible-cache rebuild
paths, and the ``load_or_build`` orchestrator (cache miss builds + writes; cache
hit skips the scan).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from nemo_automodel.components.datasets.llm.eagle3 import (
    _expected_draft_vocab_size,
    build_eagle3_token_mapping,
    load_eagle3_token_mapping,
    load_or_build_eagle3_token_mapping,
    save_eagle3_token_mapping,
)


class _CountingLoader:
    """Yields fixed batches and records how many times it was iterated.

    ``build_eagle3_token_mapping`` iterates its dataloader once to count token
    frequencies; a cache hit must not iterate at all.
    """

    def __init__(self, batches):
        self._batches = batches
        self.iter_count = 0

    def __iter__(self):
        self.iter_count += 1
        return iter(self._batches)


def _one_batch_loader():
    return _CountingLoader(
        [
            {
                "input_ids": torch.tensor([[5, 9, 9, 3]], dtype=torch.long),
                "loss_mask": torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
            }
        ]
    )


# Shared mapping config for the orchestrator tests: 16-token target, 4-token
# draft (so build actually scans), special tokens [0, 1].
_CFG = {"target_vocab_size": 16, "draft_vocab_size": 4, "special_token_ids": [0, 1]}


# ---------------------------------------------------------------------------
# _expected_draft_vocab_size
# ---------------------------------------------------------------------------


def test_expected_draft_vocab_size():
    assert _expected_draft_vocab_size(16, None) == 16  # None -> full vocab
    assert _expected_draft_vocab_size(16, 32) == 16  # too large -> full vocab
    assert _expected_draft_vocab_size(16, 16) == 16  # equal -> full vocab
    assert _expected_draft_vocab_size(16, 4) == 4  # shrunk


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    path = str(tmp_path / "map.pt")
    ids = torch.tensor([0, 1, 9, 5], dtype=torch.long)
    save_eagle3_token_mapping(path, ids, target_vocab_size=16)

    loaded = load_eagle3_token_mapping(path, target_vocab_size=16, draft_vocab_size=4)
    assert loaded is not None
    loaded_ids, loaded_mask = loaded
    assert torch.equal(loaded_ids, ids)
    expected_mask = torch.zeros(16, dtype=torch.bool)
    expected_mask[ids] = True
    assert torch.equal(loaded_mask, expected_mask)


def test_save_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "map.pt")
    ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    save_eagle3_token_mapping(path, ids, target_vocab_size=8)
    assert load_eagle3_token_mapping(path, target_vocab_size=8, draft_vocab_size=4) is not None


# ---------------------------------------------------------------------------
# load: incompatible / missing -> None (caller rebuilds)
# ---------------------------------------------------------------------------


def test_load_missing_returns_none(tmp_path):
    assert load_eagle3_token_mapping(str(tmp_path / "absent.pt"), target_vocab_size=16, draft_vocab_size=4) is None


def test_load_rejects_mismatched_target_vocab_size(tmp_path):
    path = str(tmp_path / "map.pt")
    save_eagle3_token_mapping(path, torch.tensor([0, 1, 2, 3], dtype=torch.long), target_vocab_size=16)
    # Same file, different target vocab -> incompatible.
    assert load_eagle3_token_mapping(path, target_vocab_size=32, draft_vocab_size=4) is None


def test_load_rejects_mismatched_draft_vocab_size(tmp_path):
    path = str(tmp_path / "map.pt")
    save_eagle3_token_mapping(path, torch.tensor([0, 1, 2, 3], dtype=torch.long), target_vocab_size=16)
    # Cache has 4 ids; config now wants 8 -> incompatible.
    assert load_eagle3_token_mapping(path, target_vocab_size=16, draft_vocab_size=8) is None


def test_load_rejects_corrupted_file(tmp_path):
    path = tmp_path / "map.pt"
    path.write_text("not a torch checkpoint")
    assert load_eagle3_token_mapping(str(path), target_vocab_size=16, draft_vocab_size=4) is None


# ---------------------------------------------------------------------------
# load_or_build orchestrator
# ---------------------------------------------------------------------------


def test_load_or_build_without_cache_path_builds(tmp_path):
    loader = _one_batch_loader()
    ids, mask = load_or_build_eagle3_token_mapping(loader, **_CFG, cache_path=None)
    assert ids.shape == (4,)
    assert mask.shape == (16,)
    assert loader.iter_count == 1  # scanned


def test_load_or_build_writes_then_reuses_cache(tmp_path):
    path = str(tmp_path / "map.pt")
    loader = _one_batch_loader()

    # First call: cache miss -> build (scan once) + write.
    ids1, mask1 = load_or_build_eagle3_token_mapping(loader, **_CFG, cache_path=path)
    assert loader.iter_count == 1
    assert os.path.exists(path)

    # Second call on the SAME loader: cache hit -> no further scan.
    ids2, mask2 = load_or_build_eagle3_token_mapping(loader, **_CFG, cache_path=path)
    assert loader.iter_count == 1  # not scanned again
    assert torch.equal(ids1, ids2)
    assert torch.equal(mask1, mask2)


def test_load_or_build_matches_direct_build(tmp_path):
    """The cached result must equal a direct ``build_eagle3_token_mapping`` call."""
    direct_ids, direct_mask = build_eagle3_token_mapping(_one_batch_loader(), **_CFG)
    cached_ids, cached_mask = load_or_build_eagle3_token_mapping(
        _one_batch_loader(), **_CFG, cache_path=str(tmp_path / "map.pt")
    )
    assert torch.equal(direct_ids, cached_ids)
    assert torch.equal(direct_mask, cached_mask)


def test_load_or_build_rebuilds_on_incompatible_cache(tmp_path):
    path = str(tmp_path / "map.pt")
    # Pre-seed an incompatible cache (wrong target vocab).
    save_eagle3_token_mapping(path, torch.tensor([0, 1], dtype=torch.long), target_vocab_size=999)
    loader = _one_batch_loader()
    ids, _ = load_or_build_eagle3_token_mapping(loader, **_CFG, cache_path=path)
    assert loader.iter_count == 1  # incompatible cache -> rebuilt
    assert ids.shape == (4,)
    # The stale cache was overwritten with a compatible one.
    assert load_eagle3_token_mapping(path, target_vocab_size=16, draft_vocab_size=4) is not None


# ---------------------------------------------------------------------------
# Distributed path: single-process gloo exercises the rank-0 load + broadcast
# ---------------------------------------------------------------------------


def test_load_or_build_distributed_build_then_load(tmp_path):
    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this session")

    path = str(tmp_path / "map.pt")
    dist.init_process_group(backend="gloo", init_method=f"file://{tmp_path / 'pg'}", world_size=1, rank=0)
    try:
        # Cache miss under dist -> build + save on rank 0.
        ids1, _ = load_or_build_eagle3_token_mapping(_one_batch_loader(), **_CFG, cache_path=path)
        # Cache hit under dist -> rank 0 loads and broadcasts the ids.
        loader = _one_batch_loader()
        ids2, _ = load_or_build_eagle3_token_mapping(loader, **_CFG, cache_path=path)
        assert loader.iter_count == 0  # served from cache via broadcast, no scan
        assert torch.equal(ids1, ids2)
    finally:
        dist.destroy_process_group()
