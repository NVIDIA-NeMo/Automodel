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

"""Two-process checkpoint placement checks; these do not qualify distributed training."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Shard, distribute_tensor

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel.components.models.inkling2.state_dict_adapter import Inkling2StateDictAdapter

from .test_checkpoint import checkpoint_config, decode_oracle, write_checkpoint

pytestmark = [pytest.mark.cpu_only, pytest.mark.timeout(120)]


def shard_worker(rank: int, folder: str) -> None:
    """Read [experts,input,output] DTensor shards directly from real safetensors."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(folder) / "rendezvous"),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("shard",))
        adapter = Inkling2StateDictAdapter(checkpoint_config())
        for layer in [0, 1]:
            path = Path(folder) / str(layer)
            fp4 = layer == 0
            packed = torch.full(
                (4, 16, 32 if fp4 else 64), 0x31 if fp4 else 1, dtype=torch.uint8 if fp4 else torch.float8_e4m3fn
            )
            scales = torch.full(
                (4, 16, 4 if fp4 else 2), 2 if fp4 else 128, dtype=torch.float8_e4m3fn if fp4 else torch.uint8
            )
            amax = torch.tensor([1.0, 2.0, 4.0, 8.0]) if fp4 else None
            denom = torch.tensor([1536.0]) if fp4 else None
            wanted = decode_oracle(packed, scales, amax, denom).transpose(-1, -2)
            key = f"layers.{layer}.mlp.experts.gate_and_up_projs"
            for axis in [0, 1, 2]:
                destination = distribute_tensor(torch.full_like(wanted, float("nan")), mesh, [Shard(axis)])
                pointer = destination.to_local().untyped_storage().data_ptr()
                reader = _HuggingFaceStorageReader(str(path), key_mapping=None)
                targets = adapter.to_hf({key: destination}, quantization=True, for_checkpoint_load=True)
                dcp.load(targets, storage_reader=reader)
                assert adapter.from_hf(targets) == {}
                assert adapter.view_loaded_native_keys == {key}
                expected = distribute_tensor(wanted, mesh, [Shard(axis)])
                assert pointer == destination.to_local().untyped_storage().data_ptr()
                torch.testing.assert_close(destination.to_local(), expected.to_local(), rtol=0, atol=0)
        # A mathematically partial tensor cannot be interpreted as final checkpoint storage.
        invalid = DTensor.from_local(torch.zeros(4, 64, 16), mesh, [Partial()])
        with pytest.raises(ValueError, match="Partial"):
            adapter.to_hf(
                {"layers.0.mlp.experts.gate_and_up_projs": invalid}, quantization=True, for_checkpoint_load=True
            )
        unaligned = distribute_tensor(torch.zeros(4, 48, 16), mesh, [Shard(1)])
        with pytest.raises(ValueError, match="align"):
            adapter.to_hf(
                {"layers.0.mlp.experts.gate_and_up_projs": unaligned}, quantization=True, for_checkpoint_load=True
            )
        # Floating resume must ignore the original source quantization config.
        # Row splits cross a rank boundary; singleton query rows leave one rank empty.
        floating = {
            "output_query.weight": torch.arange(32, dtype=torch.float32).reshape(1, 32),
            "layers.0.mlp_query.weight": torch.arange(32, dtype=torch.float32).reshape(1, 32) + 17,
            "layers.1.attn_query.weight": torch.arange(96, dtype=torch.float32).reshape(3, 32),
            "layers.0.attn.qkvr.weight": torch.arange(112 * 32, dtype=torch.float32).reshape(112, 32),
        }
        path = Path(folder) / "floating"
        wanted_hf = adapter.to_hf(floating)
        if rank == 0:
            path.mkdir()
            write_checkpoint(wanted_hf, path)
        dist.barrier()
        for axis in (0, 1):
            source = {name: distribute_tensor(value, mesh, [Shard(axis)]) for name, value in floating.items()}
            exported = adapter.to_hf(source)
            for key, value in exported.items():
                torch.testing.assert_close(value.full_tensor(), wanted_hf[key], rtol=0, atol=0)
            destination = {
                name: distribute_tensor(torch.full_like(value, float("nan")), mesh, [Shard(axis)])
                for name, value in floating.items()
            }
            pointers = {name: value.to_local().untyped_storage().data_ptr() for name, value in destination.items()}
            reader = _HuggingFaceStorageReader(str(path), key_mapping=None)
            targets = adapter.to_hf(destination, for_checkpoint_load=True)
            dcp.load(targets, storage_reader=reader)
            restored = adapter.from_hf(targets)
            # Shard(0) crosses the fused row split and the squeezed singleton rows, so those three keys go
            # through replicated temporaries. The [3, 32] query and every Shard(1) tensor alias their source
            # views; fused DTensor views are complete after DCP and are only reported, not returned.
            completed = adapter.view_loaded_native_keys
            if axis == 0:
                assert completed == {"output_query.weight", "layers.0.mlp_query.weight", "layers.0.attn.qkvr.weight"}
            else:
                assert completed == {"layers.0.attn.qkvr.weight"}
            assert set(restored) | completed == set(destination)
            assert not set(restored) & completed
            for name, value in destination.items():
                assert value.to_local().untyped_storage().data_ptr() == pointers[name]
                torch.testing.assert_close(value.full_tensor(), floating[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_process_expert_input_and_output_shards(tmp_path: Path) -> None:
    for layer in [0, 1]:
        path = tmp_path / str(layer)
        path.mkdir()
        key = f"model.llm.layers.{layer}.mlp.experts.w13_weight"
        fp4 = layer == 0
        checkpoint = {
            key: torch.full(
                (4, 16, 32 if fp4 else 64), 0x31 if fp4 else 1, dtype=torch.uint8 if fp4 else torch.float8_e4m3fn
            ),
            key + "_scale": torch.full(
                (4, 16, 4 if fp4 else 2), 2 if fp4 else 128, dtype=torch.float8_e4m3fn if fp4 else torch.uint8
            ),
        }
        if fp4:
            checkpoint.update(
                {
                    key + "_amax": torch.tensor([1.0, 2.0, 4.0, 8.0]),
                    key + "_outer_scale_denominator": torch.tensor([1536.0]),
                }
            )
        write_checkpoint(checkpoint, path)
    mp.spawn(shard_worker, args=(str(tmp_path),), nprocs=2, join=True)
