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

"""Actual DCP loading into CUDA-mesh expert shards, including CPU offload."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from tests.unit_tests.models.minimax_m3_vl.test_quantized_checkpoint import _save_checkpoint, _streamed_fixture


def _load_worker(rank: int, directory: str, storage_device: str, ep_size: int = 1, n_experts: int = 1) -> None:
    """Load actual checkpoint parts on two input-axis shards per expert rank.

    Args:
        rank: Global process rank.
        directory: Shared fixture checkpoint and per-rank evidence directory.
        storage_device: Final local parameter storage, cpu or cuda.
        ep_size: Expert-parallel size; CI uses one, optional workspace probe uses four.
        n_experts: Global expert count, divisible by ep_size.
    """
    world_size = ep_size * 2
    torch.cuda.set_device(rank)
    root = Path(directory)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    try:
        if ep_size == 1:
            mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("ep", "ep_shard"))
        else:
            mesh = MeshContext.build(FSDP2Config(), ParallelismSizes(ep_size=ep_size), world_size=world_size).moe_mesh
        assert mesh is not None
        coordinates = dict(zip(mesh.mesh_dim_names, mesh.get_coordinate()))
        adapter, checkpoint, native, expected = _streamed_fixture("text", n_experts)
        expert_start = coordinates["ep"] * (n_experts // ep_size)
        expert_ids = set(range(expert_start, expert_start + n_experts // ep_size))
        sharded_keys = [key for key in native if ".experts." in key or ".indexer." in key]
        for key in sharded_keys:
            placements = tuple(
                Shard(1) if axis == "ep_shard" else (Shard(0) if ".experts." in key else Replicate())
                for axis in mesh.mesh_dim_names
            )
            tensor = distribute_tensor(native[key].to(rank), mesh, placements=placements)
            native[key] = tensor.to(storage_device)
            assert native[key].to_local().device.type == storage_device
            assert native[key].device_mesh.device_type == "cuda"
        for key in native.keys() - set(sharded_keys):
            native[key] = native[key].to(storage_device)

        pointers = {
            key: (value.to_local() if isinstance(value, DTensor) else value).untyped_storage().data_ptr()
            for key, value in native.items()
        }
        reader = _HuggingFaceStorageReader(str(root), key_mapping=None)
        parts = adapter.iter_checkpoint_load_parts(native, device_mesh=mesh)
        assert parts is not None
        completed = set()
        checkpoint_keys = set()
        for part in parts:
            assert not completed.intersection(part.model_keys)
            dcp.load(part.checkpoint_tensors, storage_reader=reader)
            part.finish()
            completed.update(part.model_keys)
            assert not checkpoint_keys.intersection(part.checkpoint_tensors)
            checkpoint_keys.update(part.checkpoint_tensors)
        assert completed == set(native)
        expected_keys = {
            key
            for key in checkpoint
            if ".experts." not in key or int(key.split(".experts.")[1].split(".")[0]) in expert_ids
        }
        assert checkpoint_keys == expected_keys
        for key, tensor in native.items():
            local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            assert local.device.type == storage_device, key
            assert local.untyped_storage().data_ptr() == pointers[key], key
            wanted = expected[key]
            if ".experts." in key:
                wanted = wanted.chunk(ep_size, dim=0)[coordinates["ep"]]
            if key in sharded_keys:
                wanted = wanted.chunk(2, dim=1)[coordinates["ep_shard"]]
            torch.testing.assert_close(local.cpu(), wanted, rtol=0, atol=0)
        (root / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "mesh_names": list(mesh.mesh_dim_names),
                    "mesh_shape": list(mesh.shape),
                    "coordinates": coordinates,
                    "expert_ids": sorted(expert_ids),
                    "storage_device": storage_device,
                    "checkpoint_key_count": len(checkpoint_keys),
                    "model_key_count": len(completed),
                    "exact_values": True,
                    "original_storage_preserved": True,
                },
                indent=2,
            )
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required for actual shard placement")
@pytest.mark.parametrize("storage_device", ["cpu", "cuda"])
def test_quantized_dcp_preserves_inner_shard_storage(tmp_path, storage_device):
    _, checkpoint, _, _ = _streamed_fixture("text")
    _save_checkpoint(checkpoint, tmp_path)
    mp.spawn(_load_worker, args=(str(tmp_path), storage_device), nprocs=2, join=True)
