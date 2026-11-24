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

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from nemo_automodel.components.datasets.wam.episode_windows import EpisodeWindowDataset

WINDOW_FRAMES = 4


@pytest.fixture(autouse=True)
def fake_praxis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-in for ``praxis.robot_io`` so the dataset builds without the praxis package."""
    robot_io = types.ModuleType("praxis.robot_io")
    robot_io.contiguous_chunk_geometry = lambda action_chunk, n_chunks: [(0, 2), (2, 2)]
    monkeypatch.setitem(sys.modules, "praxis", types.ModuleType("praxis"))
    monkeypatch.setitem(sys.modules, "praxis.robot_io", robot_io)


def _write_episode(root: Path, name: str, frames: int) -> None:
    np.savez(root / f"{name}.npz", obs8=np.zeros((frames, 8), dtype=np.float32))


def _dataset(root: Path) -> EpisodeWindowDataset:
    config = {"chunking": {"action_chunk": 2, "n_chunks": 2}, "video": {"height": 8, "width": 8}}
    ds = EpisodeWindowDataset(root, model_config=config, seed=0)
    assert ds.window_frames == WINDOW_FRAMES
    return ds


def _count_loads(monkeypatch: pytest.MonkeyPatch, limit: int = 500) -> None:
    """Fail the test instead of hanging when ``__iter__`` keeps reloading episodes forever."""
    real_load = np.load
    calls = {"n": 0}

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        assert calls["n"] <= limit, "EpisodeWindowDataset.__iter__ did not stop after loading episodes repeatedly"
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counting_load)


def test_iter_raises_when_every_episode_is_shorter_than_a_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_episode(tmp_path, "a", WINDOW_FRAMES - 1)
    _write_episode(tmp_path, "b", 1)
    ds = _dataset(tmp_path)
    _count_loads(monkeypatch)

    with pytest.raises(ValueError, match="shorter than the 4-frame window"):
        next(iter(ds))


def test_iter_skips_short_episodes_when_a_long_one_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_episode(tmp_path, "short", WINDOW_FRAMES - 1)
    _write_episode(tmp_path, "long", WINDOW_FRAMES + 2)
    ds = _dataset(tmp_path)
    monkeypatch.setattr(ds, "window", lambda file, start: (file.stem, start))
    _count_loads(monkeypatch)

    windows = [w for _, w in zip(range(20), ds)]

    assert {stem for stem, _ in windows} == {"long"}
    assert {start for _, start in windows} <= {0, 1, 2}


def test_iter_accepts_an_episode_exactly_one_window_long(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_episode(tmp_path, "exact", WINDOW_FRAMES)
    ds = _dataset(tmp_path)
    monkeypatch.setattr(ds, "window", lambda file, start: (file.stem, start))

    assert next(iter(ds)) == ("exact", 0)
