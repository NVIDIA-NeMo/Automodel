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

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate

from nemo_automodel.recipes.llm import kd as llm_kd


class _MixedMeshModel(nn.Module):
    def __init__(self, expert_mesh: DeviceMesh, dense_mesh: DeviceMesh) -> None:
        super().__init__()
        self.register_parameter(
            "expert",
            nn.Parameter(
                DTensor.from_local(
                    torch.tensor([10.0]),
                    expert_mesh,
                    (Replicate(),),
                    run_check=False,
                )
            ),
        )
        self.register_parameter(
            "dense",
            nn.Parameter(
                DTensor.from_local(
                    torch.tensor([20.0]),
                    dense_mesh,
                    (Replicate(),),
                    run_check=False,
                )
            ),
        )


def test_llm_kd_non_pp_step_clips_gradients_across_device_meshes():
    """The non-PP KD step clips mixed EP and DP DTensor gradients together."""
    assert not dist.is_initialized()
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        expert_mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("ep",))
        dense_mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("dp_shard_cp",))
        device_mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("tp",))

        model = _MixedMeshModel(expert_mesh, dense_mesh)
        model.expert.grad = DTensor.from_local(
            torch.tensor([3.0]),
            expert_mesh,
            (Replicate(),),
            run_check=False,
        )
        model.dense.grad = DTensor.from_local(
            torch.tensor([4.0]),
            dense_mesh,
            (Replicate(),),
            run_check=False,
        )

        recipe = object.__new__(llm_kd.KnowledgeDistillationRecipeForNextTokenPrediction)
        recipe.model_parts = [model]
        recipe.pp_enabled = False
        recipe.device_mesh = device_mesh
        recipe.moe_mesh = expert_mesh
        recipe.optimizer = [torch.optim.SGD(model.parameters(), lr=0.1)]
        recipe.lr_scheduler = None
        recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
        recipe.cfg = {}
        recipe.timestamp = time.perf_counter() - 1.0
        recipe.step_scheduler = SimpleNamespace(step=1, epoch=0)
        recipe.kd_ratio = 0.5
        recipe.kd_loss_fn = SimpleNamespace(temperature=1.0)
        recipe._ce_loss_buffer = []
        recipe._kd_loss_buffer = []
        recipe._dp_allreduce = lambda tensor, include_cp=False: tensor
        recipe._get_dp_group_size = lambda include_cp=False: 1
        recipe._forward_backward_step = Mock(return_value=(torch.tensor(1.0), torch.tensor(0.75), torch.tensor(0.25)))

        metrics = recipe._run_train_optim_step(
            [{"labels": torch.tensor([[1, 2]])}],
            max_grad_norm=2.5,
        )

        assert metrics.metrics["grad_norm"] == pytest.approx(5.0)
        torch.testing.assert_close(model.expert.to_local().detach(), torch.tensor([9.85]))
        torch.testing.assert_close(model.dense.to_local().detach(), torch.tensor([19.8]))
        assert model.expert.grad is None
        assert model.dense.grad is None
    finally:
        dist.destroy_process_group()
