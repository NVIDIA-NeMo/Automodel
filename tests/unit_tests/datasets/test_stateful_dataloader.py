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

from __future__ import annotations

import os

import pytest
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.datasets.llm.megatron.sampler import create_megatron_sampler
from nemo_automodel.components.datasets.stateful_dataloader import DPAwareStatefulDataLoader


class RangeDataset(Dataset):
    def __init__(self, size: int = 64) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def _distributed_loader(rank: int, world_size: int, batch_size: int = 2) -> DPAwareStatefulDataLoader:
    dataset = RangeDataset()
    sampler = StatefulDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    return DPAwareStatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        dp_rank=rank,
        dp_world_size=world_size,
    )


def _megatron_loader(rank: int, world_size: int, micro_batch_size: int = 2) -> DPAwareStatefulDataLoader:
    dataset = RangeDataset()
    batch_sampler = create_megatron_sampler(
        dataset_len=len(dataset),
        micro_batch_size=micro_batch_size,
        global_batch_size=micro_batch_size * world_size,
        dataloader_type="single",
        rank=rank,
        world_size=world_size,
    )
    return DPAwareStatefulDataLoader(
        dataset,
        batch_sampler=batch_sampler,
        dp_rank=rank,
        dp_world_size=world_size,
    )


def _consume(loader: DPAwareStatefulDataLoader, num_batches: int) -> list[list[int]]:
    iterator = iter(loader)
    batches = []
    for _ in range(num_batches):
        batch = next(iterator)
        if isinstance(batch, torch.Tensor):
            batch = batch.tolist()
        batches.append(batch)
    return batches


def _rank_states(world_size: int, num_batches: int) -> dict[int, dict]:
    states = {}
    for rank in range(world_size):
        loader = _distributed_loader(rank, world_size)
        _consume(loader, num_batches)
        states[rank] = loader.state_dict()
    return states


def _legacy_rank_states(world_size: int, num_batches: int) -> dict[int, dict]:
    states = _rank_states(world_size, num_batches)
    for state in states.values():
        state.pop("_automodel_dp_state", None)
    return states


def _next_global_window(world_size: int, rank_states: dict[int, dict]) -> list[int]:
    samples = []
    for rank in range(world_size):
        loader = _distributed_loader(rank, world_size)
        loader.load_state_dict_from_dp_rank_states(rank_states)
        samples.extend(_consume(loader, 1)[0])
    return sorted(samples)


def test_dp_aware_stateful_dataloader_reshards_when_dp_size_increases() -> None:
    rank_states = _rank_states(world_size=2, num_batches=2)

    assert _next_global_window(world_size=4, rank_states=rank_states) == list(range(8, 16))


def test_dp_aware_stateful_dataloader_reshards_when_dp_size_decreases() -> None:
    rank_states = _rank_states(world_size=4, num_batches=1)

    assert _next_global_window(world_size=2, rank_states=rank_states) == list(range(8, 12))


def test_dp_aware_stateful_dataloader_preserves_same_dp_resume() -> None:
    rank_states = _rank_states(world_size=2, num_batches=2)
    loader = _distributed_loader(rank=0, world_size=2)

    loader.load_state_dict_from_dp_rank_states(rank_states)

    assert _consume(loader, 1)[0] == [8, 10]


def test_dp_aware_stateful_dataloader_allows_legacy_same_dp_resume() -> None:
    rank_states = _legacy_rank_states(world_size=2, num_batches=2)
    loader = _distributed_loader(rank=0, world_size=2)

    loader.load_state_dict_from_dp_rank_states(rank_states)

    assert _consume(loader, 1)[0] == [8, 10]


def test_dp_aware_stateful_dataloader_rejects_legacy_dp_reshard() -> None:
    rank_states = _legacy_rank_states(world_size=4, num_batches=1)
    loader = _distributed_loader(rank=0, world_size=2)

    with pytest.raises(ValueError, match="Cannot reshard a legacy dataloader checkpoint"):
        loader.load_state_dict_from_dp_rank_states(rank_states)


def test_dp_aware_stateful_dataloader_reshards_megatron_batch_sampler_state() -> None:
    rank_states = {}
    for rank in range(2):
        loader = _megatron_loader(rank, world_size=2)
        _consume(loader, 2)
        rank_states[rank] = loader.state_dict()

    samples = []
    for rank in range(4):
        loader = _megatron_loader(rank, world_size=4)
        loader.load_state_dict_from_dp_rank_states(rank_states)
        samples.extend(_consume(loader, 1)[0])

    assert sorted(samples) == list(range(8, 16))


def test_checkpointer_uses_one_saved_rank_file_for_dp_aware_scale_up(tmp_path, monkeypatch) -> None:
    state_name = "dataloader"
    state_dir = tmp_path / state_name
    state_dir.mkdir()
    for rank, state in _rank_states(world_size=2, num_batches=2).items():
        torch.save(state, state_dir / f"{state_name}_dp_rank_{rank}.pt")

    loaded_files = []
    real_torch_load = torch.load

    def tracking_torch_load(path, *args, **kwargs):
        loaded_files.append(os.path.basename(str(path)))
        return real_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(
        "nemo_automodel.components.checkpoint.checkpointing.torch.load",
        tracking_torch_load,
    )
    loader = _distributed_loader(rank=3, world_size=4)
    checkpointer = CheckpointingConfig(
        checkpoint_dir=tmp_path,
        model_cache_dir=tmp_path / "cache",
        save_consolidated=False,
    ).build(dp_rank=3, tp_rank=0, pp_rank=0)

    checkpointer.load_on_dp_ranks(loader, state_name, str(tmp_path))

    assert loaded_files == [f"{state_name}_dp_rank_0.pt"]
    assert _consume(loader, 1)[0] == [11, 15]


def test_checkpointer_loads_all_rank_states_without_single_rank_capability(tmp_path) -> None:
    class AllRankState:
        def __init__(self) -> None:
            self.rank_states = None

        def load_state_dict_from_dp_rank_states(self, rank_states):
            self.rank_states = rank_states

    state_name = "custom"
    state_dir = tmp_path / state_name
    state_dir.mkdir()
    for rank in range(2):
        torch.save({"rank": rank}, state_dir / f"{state_name}_dp_rank_{rank}.pt")

    state = AllRankState()
    checkpointer = CheckpointingConfig(
        checkpoint_dir=tmp_path,
        model_cache_dir=tmp_path / "cache",
        save_consolidated=False,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)

    checkpointer.load_on_dp_ranks(state, state_name, str(tmp_path))

    assert state.rank_states == {0: {"rank": 0}, 1: {"rank": 1}}


def test_checkpointer_uses_current_rank_file_when_metadata_is_available(tmp_path, monkeypatch) -> None:
    state_name = "dataloader"
    state_dir = tmp_path / state_name
    state_dir.mkdir()
    for rank, state in _rank_states(world_size=2, num_batches=2).items():
        torch.save(state, state_dir / f"{state_name}_dp_rank_{rank}.pt")

    loaded_files = []
    real_torch_load = torch.load

    def tracking_torch_load(path, *args, **kwargs):
        loaded_files.append(os.path.basename(str(path)))
        return real_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(
        "nemo_automodel.components.checkpoint.checkpointing.torch.load",
        tracking_torch_load,
    )
    loader = _distributed_loader(rank=0, world_size=2)
    checkpointer = CheckpointingConfig(
        checkpoint_dir=tmp_path,
        model_cache_dir=tmp_path / "cache",
        save_consolidated=False,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)

    checkpointer.load_on_dp_ranks(loader, state_name, str(tmp_path))

    assert loaded_files == [f"{state_name}_dp_rank_0.pt"]
    assert _consume(loader, 1)[0] == [8, 10]


def test_checkpointer_uses_current_rank_file_for_legacy_same_dp_resume(tmp_path, monkeypatch) -> None:
    state_name = "dataloader"
    state_dir = tmp_path / state_name
    state_dir.mkdir()
    for rank, state in _legacy_rank_states(world_size=2, num_batches=2).items():
        torch.save(state, state_dir / f"{state_name}_dp_rank_{rank}.pt")

    loaded_files = []
    real_torch_load = torch.load

    def tracking_torch_load(path, *args, **kwargs):
        loaded_files.append(os.path.basename(str(path)))
        return real_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(
        "nemo_automodel.components.checkpoint.checkpointing.torch.load",
        tracking_torch_load,
    )
    loader = _distributed_loader(rank=0, world_size=2)
    checkpointer = CheckpointingConfig(
        checkpoint_dir=tmp_path,
        model_cache_dir=tmp_path / "cache",
        save_consolidated=False,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)

    checkpointer.load_on_dp_ranks(loader, state_name, str(tmp_path))

    assert loaded_files == [f"{state_name}_dp_rank_0.pt"]
    assert _consume(loader, 1)[0] == [8, 10]


def test_checkpointer_rejects_legacy_dp_reshard(tmp_path) -> None:
    state_name = "dataloader"
    state_dir = tmp_path / state_name
    state_dir.mkdir()
    for rank, state in _legacy_rank_states(world_size=4, num_batches=1).items():
        torch.save(state, state_dir / f"{state_name}_dp_rank_{rank}.pt")

    loader = _distributed_loader(rank=0, world_size=2)
    checkpointer = CheckpointingConfig(
        checkpoint_dir=tmp_path,
        model_cache_dir=tmp_path / "cache",
        save_consolidated=False,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)

    with pytest.raises(ValueError, match="Cannot reshard a legacy dataloader checkpoint"):
        checkpointer.load_on_dp_ranks(loader, state_name, str(tmp_path))
