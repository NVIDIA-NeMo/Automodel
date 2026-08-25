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

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel._transformers.models.gemma4_moe.state_dict_adapter import Gemma4MoEStateDictAdapter

N_EXPERTS = 4
HIDDEN = 64
EXPERT_INTER = 32


@pytest.fixture
def adapter() -> Gemma4MoEStateDictAdapter:
    return Gemma4MoEStateDictAdapter(
        config=SimpleNamespace(),
        moe_config=SimpleNamespace(n_routed_experts=N_EXPERTS),
        backend=SimpleNamespace(),
        dtype=torch.float32,
    )


def _run_ep_sharded_load_without_full_copy(rank: int, world_size: int, init_file: str, model_path: str) -> None:
    """Load this rank's expert weights directly into its part of the model's weight memory."""
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))
        adapter = Gemma4MoEStateDictAdapter(
            config=SimpleNamespace(),
            moe_config=SimpleNamespace(n_routed_experts=N_EXPERTS),
            backend=SimpleNamespace(),
            dtype=torch.float32,
        )
        local_experts = N_EXPERTS // world_size
        gate_local = torch.full((local_experts, HIDDEN, 2 * EXPERT_INTER), -1.0)
        down_local = torch.full((local_experts, EXPERT_INTER, HIDDEN), -1.0)
        gate = DTensor.from_local(gate_local, mesh, [Shard(0)])
        down = DTensor.from_local(down_local, mesh, [Shard(0)])
        native_state = {
            "model.language_model.layers.0.moe.experts.gate_and_up_projs": gate,
            "model.language_model.layers.0.moe.experts.down_projs": down,
        }

        destinations = adapter.to_hf(
            native_state,
            device_mesh=mesh,
            for_checkpoint_load=True,
        )
        gate_destination = destinations["model.language_model.layers.0.experts.gate_up_proj"]
        down_destination = destinations["model.language_model.layers.0.experts.down_proj"]
        scale_destination = destinations["model.language_model.layers.0.router.per_expert_scale"]
        assert isinstance(gate_destination, DTensor)
        assert isinstance(down_destination, DTensor)
        assert isinstance(scale_destination, DTensor)
        assert gate_destination.shape == (N_EXPERTS, 2 * EXPERT_INTER, HIDDEN)
        assert down_destination.shape == (N_EXPERTS, HIDDEN, EXPERT_INTER)
        assert scale_destination.shape == (N_EXPERTS,)
        assert gate_destination.to_local().untyped_storage().data_ptr() == gate_local.untyped_storage().data_ptr()
        assert down_destination.to_local().untyped_storage().data_ptr() == down_local.untyped_storage().data_ptr()
        gate_chunk = gate_destination.__create_chunk_list__()[0]
        assert gate_chunk.offsets == torch.Size([rank * local_experts, 0, 0])
        assert gate_chunk.sizes == torch.Size([local_experts, 2 * EXPERT_INTER, HIDDEN])

        dcp.load(
            destinations,
            checkpoint_id=model_path,
            storage_reader=_HuggingFaceStorageReader(model_path),
        )
        converted = adapter.from_hf(destinations, device_mesh=mesh)

        start_expert = rank * local_experts
        end_expert = start_expert + local_experts
        checkpoint_gate = torch.arange(
            N_EXPERTS * 2 * EXPERT_INTER * HIDDEN,
            dtype=torch.float32,
        ).reshape(N_EXPERTS, 2 * EXPERT_INTER, HIDDEN)
        checkpoint_down = torch.arange(
            N_EXPERTS * HIDDEN * EXPERT_INTER,
            dtype=torch.float32,
        ).reshape(N_EXPERTS, HIDDEN, EXPERT_INTER)
        checkpoint_scale = torch.arange(1, N_EXPERTS + 1, dtype=torch.float32)
        loaded_gate = converted["model.language_model.layers.0.moe.experts.gate_and_up_projs"]
        loaded_down = converted["model.language_model.layers.0.moe.experts.down_projs"]
        torch.testing.assert_close(
            loaded_gate.to_local(),
            checkpoint_gate[start_expert:end_expert].transpose(-2, -1),
        )
        torch.testing.assert_close(
            loaded_down.to_local(),
            checkpoint_down[start_expert:end_expert].transpose(-2, -1)
            * checkpoint_scale[start_expert:end_expert, None, None],
        )
        assert loaded_gate.to_local().untyped_storage().data_ptr() == gate_local.untyped_storage().data_ptr()
        assert loaded_down.to_local().untyped_storage().data_ptr() == down_local.untyped_storage().data_ptr()
    finally:
        dist.destroy_process_group()


def test_ep_load_reads_only_local_grouped_experts_into_final_storage(tmp_path) -> None:
    """Two EP ranks load disjoint grouped-expert slices without global materialization."""
    model_path = tmp_path / "model"
    model_path.mkdir()
    checkpoint_gate = torch.arange(
        N_EXPERTS * 2 * EXPERT_INTER * HIDDEN,
        dtype=torch.float32,
    ).reshape(N_EXPERTS, 2 * EXPERT_INTER, HIDDEN)
    checkpoint_down = torch.arange(
        N_EXPERTS * HIDDEN * EXPERT_INTER,
        dtype=torch.float32,
    ).reshape(N_EXPERTS, HIDDEN, EXPERT_INTER)
    save_file(
        {
            "model.language_model.layers.0.experts.gate_up_proj": checkpoint_gate,
            "model.language_model.layers.0.experts.down_proj": checkpoint_down,
            "model.language_model.layers.0.router.per_expert_scale": torch.arange(
                1, N_EXPERTS + 1, dtype=torch.float32
            ),
        },
        model_path / "model.safetensors",
    )

    mp.spawn(
        _run_ep_sharded_load_without_full_copy,
        args=(2, str(tmp_path / "dist_init"), str(model_path)),
        nprocs=2,
        join=True,
    )


def test_large_destinations_use_model_weight_memory_and_scale_stays_small(
    adapter: Gemma4MoEStateDictAdapter,
) -> None:
    gate_and_up = torch.zeros(N_EXPERTS, HIDDEN, 2 * EXPERT_INTER)
    down = torch.zeros(N_EXPERTS, EXPERT_INTER, HIDDEN)
    state_dict = {
        "model.language_model.layers.0.moe.experts.gate_and_up_projs": gate_and_up,
        "model.language_model.layers.0.moe.experts.down_projs": down,
    }

    destinations = adapter.to_hf(state_dict, for_checkpoint_load=True)

    gate_destination = destinations["model.language_model.layers.0.experts.gate_up_proj"]
    down_destination = destinations["model.language_model.layers.0.experts.down_proj"]
    scale_destination = destinations["model.language_model.layers.0.router.per_expert_scale"]
    assert gate_destination.untyped_storage().data_ptr() == gate_and_up.untyped_storage().data_ptr()
    assert down_destination.untyped_storage().data_ptr() == down.untyped_storage().data_ptr()
    assert scale_destination.untyped_storage().data_ptr() not in {
        gate_and_up.untyped_storage().data_ptr(),
        down.untyped_storage().data_ptr(),
    }
    assert scale_destination.shape == (N_EXPERTS,)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_from_hf_applies_scales_to_loaded_model_weights(adapter: Gemma4MoEStateDictAdapter, dtype: torch.dtype) -> None:
    adapter.dtype = dtype
    expected_gate_hf = torch.randn(N_EXPERTS, 2 * EXPERT_INTER, HIDDEN, dtype=dtype)
    expected_down_hf = torch.randn(N_EXPERTS, HIDDEN, EXPERT_INTER, dtype=dtype)
    expected_scale = torch.arange(1, N_EXPERTS + 1, dtype=dtype)
    reference = adapter.from_hf(
        {
            "model.language_model.layers.0.experts.gate_up_proj": expected_gate_hf.clone(),
            "model.language_model.layers.0.experts.down_proj": expected_down_hf.clone(),
            "model.language_model.layers.0.router.per_expert_scale": expected_scale.clone(),
        }
    )
    gate_and_up = torch.zeros(N_EXPERTS, HIDDEN, 2 * EXPERT_INTER, dtype=dtype)
    down = torch.zeros(N_EXPERTS, EXPERT_INTER, HIDDEN, dtype=dtype)
    state_dict = {
        "model.language_model.layers.0.moe.experts.gate_and_up_projs": gate_and_up,
        "model.language_model.layers.0.moe.experts.down_projs": down,
    }
    destinations = adapter.to_hf(state_dict, for_checkpoint_load=True)
    destinations["model.language_model.layers.0.experts.gate_up_proj"].copy_(expected_gate_hf)
    destinations["model.language_model.layers.0.experts.down_proj"].copy_(expected_down_hf)
    destinations["model.language_model.layers.0.router.per_expert_scale"].copy_(expected_scale)

    converted = adapter.from_hf(destinations)

    converted_gate = converted["model.language_model.layers.0.moe.experts.gate_and_up_projs"]
    converted_down = converted["model.language_model.layers.0.moe.experts.down_projs"]
    assert converted_gate.untyped_storage().data_ptr() == gate_and_up.untyped_storage().data_ptr()
    assert converted_down.untyped_storage().data_ptr() == down.untyped_storage().data_ptr()
    torch.testing.assert_close(
        converted_gate,
        reference["model.language_model.layers.0.moe.experts.gate_and_up_projs"],
    )
    torch.testing.assert_close(
        converted_down,
        reference["model.language_model.layers.0.moe.experts.down_projs"],
    )


def test_export_destinations_remain_independent_contiguous_tensors(adapter: Gemma4MoEStateDictAdapter) -> None:
    gate_and_up = torch.zeros(N_EXPERTS, HIDDEN, 2 * EXPERT_INTER)
    down = torch.zeros(N_EXPERTS, EXPERT_INTER, HIDDEN)

    exported = adapter.to_hf(
        {
            "model.language_model.layers.0.moe.experts.gate_and_up_projs": gate_and_up,
            "model.language_model.layers.0.moe.experts.down_projs": down,
        }
    )

    exported_gate = exported["model.language_model.layers.0.experts.gate_up_proj"]
    exported_down = exported["model.language_model.layers.0.experts.down_proj"]
    assert exported_gate.is_contiguous()
    assert exported_down.is_contiguous()
    assert exported_gate.untyped_storage().data_ptr() != gate_and_up.untyped_storage().data_ptr()
    assert exported_down.untyped_storage().data_ptr() != down.untyped_storage().data_ptr()
