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

"""Distributed random initialization must not repeat router rows across shards."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import Gate


def _worker(rank: int, init_file: str, dtype: torch.dtype) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        config = MoEConfig(
            n_routed_experts=256,
            n_shared_experts=1,
            n_activated_experts=6,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.001,
            aux_loss_coeff=0.0,
            score_func="sqrtsoftplus",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=32,
            norm_topk_prob=True,
            dtype=dtype,
        )
        mesh = init_device_mesh("cuda", (2,))
        with torch.device("meta"):
            gate = Gate(config)
        fully_shard(gate, mesh=mesh)
        gate.to_empty(device=torch.device("cuda", rank))
        torch.manual_seed(1234)
        with torch.no_grad():
            gate.init_weights(torch.device("cuda", rank))
        weight = gate.weight.full_tensor()
        # With local-shard RNG, the identical per-rank seed repeats every row
        # of the first shard on the second shard, creating tied routing scores.
        assert not torch.equal(weight[:128], weight[128:]), "FSDP router shards repeat identical rows"
        assert torch.unique(weight, dim=0).shape[0] == 256
        assert 0.018 < weight.float().std().item() < 0.022
        assert abs(weight.float().mean().item()) < 0.001
        expected = weight.clone()
        dist.broadcast(expected, src=0)
        torch.testing.assert_close(weight, expected, rtol=0, atol=0)
        torch.testing.assert_close(gate.e_score_correction_bias, torch.zeros(256, device="cuda"))
        assert gate.e_score_correction_bias.dtype == torch.float32

        # The initialized gate must route and backpropagate on its first use.
        # A nonconstant upstream gradient avoids the constant-sum loss of
        # normalized routing weights.
        inputs = torch.randn(4, config.dim, device=weight.device, dtype=dtype, requires_grad=True)
        token_mask = torch.ones(4, device=weight.device, dtype=torch.bool)
        routing_weights, selected, _ = gate(inputs, token_mask, cp_mesh=None)
        assert torch.isfinite(routing_weights).all()
        assert ((selected >= 0) & (selected < config.n_routed_experts)).all()
        assert (selected.sort(dim=-1).values.diff(dim=-1) > 0).all()
        routing_weights.backward(torch.randn_like(routing_weights))
        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()
        assert inputs.grad.abs().sum() > 0
        assert gate.weight.grad is not None
        assert torch.isfinite(gate.weight.grad.to_local()).all()
        assert gate.weight.grad.full_tensor().abs().sum() > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fsdp_gate_initialization_has_distinct_shards(tmp_path: Path, dtype: torch.dtype) -> None:
    mp.spawn(_worker, args=(str(tmp_path / "init"), dtype), nprocs=2, join=True)
