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
"""Expert parallelism of the Praxis MoE layers through the shared MoE machinery (CPU, two processes).

Each rank holds the full tiny model as a reference. After the experts are made expert parallel, the
MoE layer must give the same output as the reference on this rank's tokens, its expert gradients must
be the sum of both ranks' contributions, and the shared gradient scaling must turn that sum into the
data-parallel mean that FSDP gives every other parameter.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.models.praxis.backbone import MoeTrainingSettings
from nemo_automodel.components.models.praxis.config import PraxisConfig
from nemo_automodel.components.models.praxis.model import PraxisPolicy
from nemo_automodel.components.models.praxis.parallelize import _expert_meshes
from nemo_automodel.components.moe.parallelizer import apply_ep_to_moe
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from tests.unit_tests.models.praxis.test_praxis_model import TINY_CONFIG, cpu_backend

# Over the default budget on purpose: the test spawns two worker processes that import torch.
pytestmark = pytest.mark.timeout(120)

TOKENS = 32
EXPERT_PARAMS = ("gate_and_up_projs", "down_projs")


def _policy() -> PraxisPolicy:
    torch.manual_seed(0)
    settings = MoeTrainingSettings(gate_bias_update_factor=0.0, aux_loss_coeff=0.0)
    policy = PraxisPolicy(PraxisConfig.from_json(TINY_CONFIG), cpu_backend(), settings).to(torch.float32)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.normal_()
    return policy.train()


def _worker(rank: int, world_size: int, store_path: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size)
    try:
        moe_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))
        ep_mesh, ep_shard_mesh = _expert_meshes(moe_mesh)
        assert ep_mesh is not None and ep_shard_mesh is None
        assert _expert_meshes(init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))) == (None, None)
        assert _expert_meshes(None) == (None, None)

        policy = _policy()
        reference = copy.deepcopy(policy)
        layers, reference_layers = policy.moe_layers(), reference.moe_layers()
        assert len(layers) == 3
        for layer in layers:
            apply_ep_to_moe(layer, ep_mesh, moe_mesh=moe_mesh)
        n_experts = layers[0].n_routed_experts
        n_local = n_experts // world_size
        for layer in layers:
            for name in EXPERT_PARAMS:
                parameter = getattr(layer.experts, name)
                assert isinstance(parameter, DTensor)
                assert parameter.placements == (Shard(0),) and parameter.device_mesh == ep_mesh
                assert parameter.to_local().shape[0] == n_local

        dim = int(TINY_CONFIG["backbone"]["dim"])
        x = torch.randn(TOKENS, dim, generator=torch.Generator().manual_seed(100 + rank))
        layer, reference_layer = layers[0], reference_layers[0]
        out, reference_out = layer(x), reference_layer(x)
        torch.testing.assert_close(out, reference_out, rtol=1e-5, atol=1e-5)
        out.sum().backward()
        reference_out.sum().backward()

        # The expert-parallel gradient sums the contributions of every rank's tokens.
        start = rank * n_local
        summed = {}
        for name in EXPERT_PARAMS:
            reference_grad = getattr(reference_layer.experts, name).grad.clone()
            dist.all_reduce(reference_grad)
            summed[name] = reference_grad[start : start + n_local]
            # fp32 sums over the gathered tokens in another order: agreement to 1e-4, not bitwise
            torch.testing.assert_close(getattr(layer.experts, name).grad.to_local(), summed[name], rtol=1e-4, atol=1e-4)
        # The shared scaling turns it into the data-parallel mean that FSDP gives the other parameters;
        # those keep their local gradient.
        scale_grads_and_clip_grad_norm(None, [policy], moe_mesh=moe_mesh, ep_axis_name="ep", dp_group_size=world_size)
        for name in EXPERT_PARAMS:
            torch.testing.assert_close(
                getattr(layer.experts, name).grad.to_local(), summed[name] / world_size, rtol=1e-4, atol=1e-4
            )
        torch.testing.assert_close(layer.gate.weight.grad, reference_layer.gate.weight.grad, rtol=1e-5, atol=1e-5)
    finally:
        dist.destroy_process_group()


def test_praxis_experts_run_expert_parallel_through_the_shared_moe_path(tmp_path) -> None:
    mp.spawn(_worker, args=(2, str(tmp_path / "rendezvous")), nprocs=2, join=True)
