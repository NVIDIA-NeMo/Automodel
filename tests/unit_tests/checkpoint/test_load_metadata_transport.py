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


"""Opt-in checkpoint metadata transport and process-group ownership."""

import gc
import os
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.checkpoint.state_dict_adapter import CheckpointLoadPart

pytestmark = pytest.mark.cpu_only


def test_load_metadata_transport_defaults_to_existing_group() -> None:
    assert CheckpointingConfig(save_consolidated=False).load_metadata_with_gloo is False


@pytest.mark.parametrize("value", ["true", 1, None])
def test_load_metadata_transport_requires_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="load_metadata_with_gloo must be a boolean"):
        CheckpointingConfig(save_consolidated=False, load_metadata_with_gloo=value)


@pytest.mark.parametrize(
    "enabled,initialized,world_size,backend",
    [(True, False, 1, "nccl"), (True, True, 1, "nccl"), (True, True, 2, "gloo"), (False, True, 2, "nccl")],
)
def test_no_extra_group_without_multi_rank_nccl(
    enabled: bool, initialized: bool, world_size: int, backend: str
) -> None:
    """No shared group unless the option is on and the caller's group is a multi-rank NCCL group."""
    original = object()
    with (
        patch("torch.distributed.is_initialized", return_value=initialized),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_backend", return_value=backend),
        patch("torch.distributed.new_group") as create,
        patch("torch.distributed.destroy_process_group") as destroy,
    ):
        checkpointer = CheckpointingConfig(save_consolidated=False, load_metadata_with_gloo=enabled).build(
            0, 0, 0, process_group=original
        )
        assert checkpointer.process_group is original
        assert checkpointer._get_load_process_group() is original
        assert checkpointer._load_process_group is None
        checkpointer.close()
    create.assert_not_called()
    destroy.assert_not_called()


def test_load_group_preserves_membership_and_survives_checkpointer_close() -> None:
    original, owned = Mock(spec=dist.ProcessGroup), Mock(spec=dist.ProcessGroup)
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_backend", return_value="nccl"),
        patch("torch.distributed.get_process_group_ranks", return_value=[2, 3]),
        patch("torch.distributed.new_group", return_value=owned) as create,
        patch("torch.distributed.destroy_process_group") as destroy,
    ):
        checkpointer = CheckpointingConfig(save_consolidated=False, load_metadata_with_gloo=True).build(
            0, 0, 0, process_group=original
        )
        assert checkpointer.process_group is original
        create.assert_not_called()
        assert checkpointer._get_load_process_group() is owned
        assert checkpointer._get_load_process_group() is owned
        assert checkpointer._load_process_group is owned
        create.assert_called_once_with(ranks=[2, 3], backend="gloo", use_local_synchronization=True)
        checkpointer.close()
        checkpointer.close()
        assert checkpointer._load_process_group is None
        successor = CheckpointingConfig(save_consolidated=False, load_metadata_with_gloo=True).build(
            0, 0, 0, process_group=original
        )
        assert successor._get_load_process_group() is owned
        successor.close()
        create.assert_called_once()
    destroy.assert_not_called()


def test_default_does_not_create_metadata_group() -> None:
    original = object()
    with patch("torch.distributed.new_group") as create:
        checkpointer = CheckpointingConfig(save_consolidated=False).build(0, 0, 0, process_group=original)
        assert checkpointer._get_load_process_group() is original
    create.assert_not_called()


def test_load_uses_cpu_group_and_save_keeps_callers_group() -> None:
    original, owned = Mock(spec=dist.ProcessGroup), Mock(spec=dist.ProcessGroup)
    checkpoint_module = "nemo_automodel.components.checkpoint.checkpointing"
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_backend", return_value="nccl"),
        patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]),
        patch("torch.distributed.new_group", return_value=owned),
        patch("torch.distributed.destroy_process_group") as destroy,
        patch(f"{checkpoint_module}.dcp.load") as load,
        patch(f"{checkpoint_module}.dcp.save") as save,
        patch(f"{checkpoint_module}._ensure_shared_dirs"),
    ):
        checkpointer = CheckpointingConfig(save_consolidated=False, load_metadata_with_gloo=True).build(
            0, 0, 0, process_group=original
        )
        tensors = {"weight": torch.zeros(2, 3)}
        state = SimpleNamespace(state_dict=lambda: tensors, load_state_dict=Mock())
        try:
            checkpointer._do_load(tensors, "/checkpoint/model")
            assert load.call_args.kwargs["process_group"] is owned
            checkpointer.load_distributed_state(state, "auxiliary", "/checkpoint")
            assert load.call_args.kwargs["process_group"] is owned
            state.load_state_dict.assert_called_once_with(tensors)
            checkpointer.save_distributed_state(state, "auxiliary", "/checkpoint")
            assert save.call_args.kwargs["process_group"] is original
        finally:
            checkpointer.close()
    destroy.assert_not_called()


def test_part_wise_model_load_uses_owned_group() -> None:
    """The adapter-driven MoE load path hands the shared group to every per-part dcp.load."""
    original, owned = Mock(spec=dist.ProcessGroup), Mock(spec=dist.ProcessGroup)
    checkpoint_module = "nemo_automodel.components.checkpoint.checkpointing"
    tensors = {"a": torch.zeros(2), "b": torch.zeros(3)}
    parts = [
        CheckpointLoadPart(
            checkpoint_tensors={name: tensors[name]},
            model_keys=frozenset({name}),
            temporary_checkpoint_keys=frozenset(),
            finish=Mock(),
        )
        for name in tensors
    ]
    metadata = SimpleNamespace(state_dict_metadata=dict.fromkeys(tensors))
    model_state = SimpleNamespace(uses_tied_lm_head=False, is_peft=False, model=[None])
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_backend", return_value="nccl"),
        patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]),
        patch("torch.distributed.new_group", return_value=owned),
        patch("torch.distributed.destroy_process_group") as destroy,
        patch(f"{checkpoint_module}._get_checkpoint_metadata", return_value=metadata),
        patch(f"{checkpoint_module}.dcp.load") as load,
    ):
        checkpointer = CheckpointingConfig(save_consolidated=False, load_metadata_with_gloo=True).build(
            0, 0, 0, process_group=original
        )
        try:
            checkpointer._load_model_in_parts(model_state, iter(parts), tensors, "/checkpoint/model", Mock())
        finally:
            checkpointer.close()
    assert load.call_count == len(parts)
    assert all(call.kwargs["process_group"] is owned for call in load.call_args_list)
    for part in parts:
        part.finish.assert_called_once_with()
    destroy.assert_not_called()


def _real_gloo_worker(rank: int, world_size: int, init_file: str, root: str) -> None:
    """Load a real sharded DCP checkpoint through a shared Gloo group and leave the caller's group intact."""
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    checkpointer = None
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        full = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        dcp.save({"weight": distribute_tensor(full, mesh, [Shard(0)])}, checkpoint_id=f"{root}/ckpt")
        checkpointer = CheckpointingConfig(
            enabled=False, save_consolidated=False, load_metadata_with_gloo=True, checkpoint_dir=f"{root}/unused"
        ).build(rank, 0, 0, process_group=dist.group.WORLD)
        destination = {"weight": DTensor.from_local(torch.zeros(8 // world_size, 4), mesh, [Shard(0)])}
        # The caller's group is Gloo on CPU; report it as NCCL so the option takes effect as on GPU.
        with patch("torch.distributed.get_backend", return_value="nccl"):
            owned = checkpointer._get_load_process_group()
        assert owned is not dist.group.WORLD
        assert dist.get_backend(owned) == "gloo"
        assert dist.get_process_group_ranks(owned) == list(range(world_size))
        assert checkpointer._get_load_process_group() is owned
        with patch("torch.distributed.checkpoint.load", wraps=dcp.load) as spy:
            checkpointer._do_load(destination, f"{root}/ckpt")
        assert spy.call_args.kwargs["process_group"] is owned
        torch.testing.assert_close(destination["weight"].to_local(), full.chunk(world_size)[rank], rtol=0, atol=0)
        checkpointer.close()
        checkpointer.close()
        assert checkpointer._load_process_group is None
        check = torch.ones(1)
        dist.all_reduce(check)
        assert check.item() == world_size
    finally:
        if checkpointer is not None:
            checkpointer.close()
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=90,
    reason="Spawns two CPU ranks that re-import torch and perform a real DCP round trip.",
)
def test_real_gloo_group_loads_sharded_checkpoint(tmp_path) -> None:
    """Two CPU ranks: the shared Gloo group carries DCP planning and every shard loads bit-exactly."""
    mp.spawn(_real_gloo_worker, args=(2, str(tmp_path / "dist_init"), str(tmp_path)), nprocs=2, join=True)


def _successive_checkpointers_worker(rank: int, init_file: str, root: str, enabled: bool) -> None:
    """Exercise subgroup-only loads across checkpointer and distributed-world lifetimes."""
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    for generation in range(2):
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}-{generation}",
            rank=rank,
            world_size=3,
            timeout=timedelta(seconds=30),
        )
        try:
            caller = dist.new_group(ranks=[1, 2], backend="gloo")
            if rank in (1, 2):
                mesh = torch.distributed.device_mesh.DeviceMesh.from_group(caller, "cpu", mesh_dim_names=("dp",))
                full = torch.arange(32, dtype=torch.float32).reshape(8, 4) + generation
                path = f"{root}/ckpt-{generation}"
                dcp.save(
                    {"weight": distribute_tensor(full, mesh, [Shard(0)])}, checkpoint_id=path, process_group=caller
                )
                for iteration in range(3):
                    if iteration and rank == 2:
                        time.sleep(1)
                    checkpointer = CheckpointingConfig(
                        enabled=False, save_consolidated=False, load_metadata_with_gloo=enabled
                    ).build(rank - 1, 0, 0, process_group=caller)
                    destination = {"weight": DTensor.from_local(torch.zeros(4, 4), mesh, [Shard(0)])}
                    try:
                        # Real CPU groups and DCP I/O; only select the NCCL-caller branch.
                        with patch("torch.distributed.get_backend", return_value="nccl"):
                            checkpointer._get_load_process_group()
                        checkpointer._do_load(destination, path)
                        torch.testing.assert_close(
                            destination["weight"].to_local(), full.chunk(2)[rank - 1], rtol=0, atol=0
                        )
                    finally:
                        checkpointer.close()
                    del checkpointer
                    gc.collect()
                check = torch.ones(1)
                dist.all_reduce(check, group=caller)
                assert check.item() == 2
            # Rank zero never enters the subgroup's checkpoint operations.
            check = torch.ones(1)
            dist.all_reduce(check)
            assert check.item() == 3
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.runtime_budget(
    30,
    hard_timeout=90,
    reason="Spawns three CPU ranks, loads through successive checkpointers with a delayed rank, and reinitializes the world.",
)
def test_successive_checkpointers_load_on_subgroup(tmp_path, enabled: bool) -> None:
    processes = mp.spawn(
        _successive_checkpointers_worker,
        args=(str(tmp_path / "dist_init"), str(tmp_path), enabled),
        nprocs=3,
        join=False,
    )
    deadline = time.monotonic() + 55
    try:
        while not processes.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Successive subgroup checkpointers stalled during load or group creation")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
