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

"""Real DTensor checkpoint coverage for uneven EP and composed EP/FSDP meshes."""

import os
import socket
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import PretrainedConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p5.state_dict_adapter import Step3p5StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    split_experts_weights_dtensor_aware,
)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _checkpoint_worker(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        ep_mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("ep",))
        for n_experts in (10, 12, 2):
            full = torch.arange(n_experts * 8 * 12, dtype=torch.float32).view(n_experts, 8, 12)
            sharded = distribute_tensor(full, ep_mesh, [Shard(0)])
            chunks = torch.arange(n_experts).chunk(4)
            expected_ids = chunks[rank] if rank < len(chunks) else torch.empty(0, dtype=torch.int64)
            first = sum(chunk.numel() for chunk in chunks[:rank])
            actual_range = get_expert_range_for_rank_from_mesh(ep_mesh, n_experts)
            ownership = torch.tensor([*actual_range, first, first + expected_ids.numel()])
            all_ownership = [torch.empty_like(ownership) for _ in range(4)]
            dist.all_gather(all_ownership, ownership)
            ownership = torch.stack(all_ownership)
            assert torch.equal(ownership[:, :2], ownership[:, 2:]), ownership
            exported, ids = split_experts_weights_dtensor_aware(sharded, n_experts)
            assert ids == expected_ids.tolist()
            for expert, expert_id in zip(exported, ids):
                torch.testing.assert_close(expert, full[expert_id])

            restored = create_dtensor_from_local(sharded.to_local(), ep_mesh, n_experts=n_experts)
            assert restored.shape == full.shape
            torch.testing.assert_close(restored.full_tensor(), full)

        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_shard", "ep"))
        config = MoEConfig(
            dim=8,
            inter_dim=12,
            moe_inter_dim=6,
            n_routed_experts=3,
            n_shared_experts=0,
            n_activated_experts=1,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            norm_topk_prob=False,
            expert_bias=False,
            dtype=torch.float32,
        )
        for n_experts in (3, 1):
            config = replace(config, n_routed_experts=n_experts)
            adapter = Step3p5StateDictAdapter(PretrainedConfig(), config, BackendConfig(), dtype=torch.float32)
            native = {
                "model.layers.0.moe.experts.gate_and_up_projs": torch.arange(
                    n_experts * 8 * 12, dtype=torch.float32
                ).view(n_experts, 8, 12),
                "model.layers.0.moe.experts.down_projs": torch.arange(n_experts * 6 * 8, dtype=torch.float32).view(
                    n_experts, 6, 8
                ),
            }
            sharded = {name: distribute_tensor(value, mesh, [Shard(1), Shard(0)]) for name, value in native.items()}
            hf_expected = adapter.to_hf(native)
            hf_actual = adapter.to_hf(sharded)
            for name, value in hf_actual.items():
                assert value.shape == hf_expected[name].shape
                torch.testing.assert_close(value.full_tensor(), hf_expected[name])
            for checkpoint in (hf_actual, hf_expected):
                restored = adapter.from_hf(dict(checkpoint), device_mesh=mesh)
                for name, value in restored.items():
                    assert value.shape == native[name].shape, (rank, n_experts, name, value.shape, native[name].shape)
                    assert value.placements == sharded[name].placements
                    torch.testing.assert_close(value.to_local(), sharded[name].to_local())
        uneven_inner = {"model.layers.0.moe.down_proj.weight": torch.zeros(1, 8, 7)}
        with pytest.raises(ValueError, match="input dimensions must be divisible"):
            adapter.from_hf(uneven_inner, device_mesh=mesh)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_uneven_checkpoint_sharding_and_roundtrip() -> None:
    mp.spawn(_checkpoint_worker, args=(_port(),), nprocs=4, join=True)
