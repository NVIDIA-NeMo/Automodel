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

"""The routers' selection biases step from the expert load of all data-parallel ranks."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate

from nemo_automodel.components.models.praxis.backbone import MoeTrainingSettings
from nemo_automodel.components.models.praxis.config import PraxisConfig
from nemo_automodel.components.models.praxis.model import PraxisPolicy
from nemo_automodel.components.models.praxis.parallelize import replicate_router_biases
from tests.unit_tests.models.praxis.test_praxis_model import TINY_CONFIG, cpu_backend

# Over the default budget on purpose: the test spawns two worker processes that import torch.
pytestmark = pytest.mark.timeout(120)

FACTOR = 1e-3
TOKENS = 64


def _policy() -> PraxisPolicy:
    torch.manual_seed(0)
    settings = MoeTrainingSettings(gate_bias_update_factor=FACTOR, aux_loss_coeff=0.0)
    policy = PraxisPolicy(PraxisConfig.from_json(TINY_CONFIG), cpu_backend(), settings)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.normal_()
    return policy.train()


def _worker(rank: int, world_size: int, store_path: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size)
    try:
        policy = _policy()
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        replicate_router_biases(policy, mesh)
        layers = policy.moe_layers()
        assert layers
        dim = int(TINY_CONFIG["backbone"]["dim"])
        token_mask = torch.ones(TOKENS, dtype=torch.bool)
        expected = []
        for layer in layers:
            gate = layer.gate
            assert isinstance(gate.e_score_correction_bias, DTensor)
            assert all(isinstance(p, Replicate) for p in gate.e_score_correction_bias.placements)
            # Every rank routes its own tokens; the oracle counts the picks of all ranks.
            x = torch.randn(TOKENS, dim, generator=torch.Generator().manual_seed(100 + rank))
            _, indices, _ = gate(x, token_mask, None)
            picks = [torch.zeros_like(indices) for _ in range(world_size)]
            dist.all_gather(picks, indices)
            load = torch.bincount(torch.cat(picks).flatten(), minlength=gate.n_experts).float()
            expected.append(FACTOR * torch.sign(load.mean() - load))
        policy.update_moe_gate_bias()
        for layer, step in zip(layers, expected, strict=True):
            bias = layer.gate.e_score_correction_bias.to_local()
            torch.testing.assert_close(bias, step, rtol=0, atol=1e-7)
            copies = [torch.zeros_like(bias) for _ in range(world_size)]
            dist.all_gather(copies, bias)
            for other in copies:
                torch.testing.assert_close(other, bias, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_router_bias_steps_from_the_load_of_all_ranks(tmp_path: Path) -> None:
    mp.spawn(_worker, args=(2, str(tmp_path / "rendezvous")), nprocs=2, join=True)
