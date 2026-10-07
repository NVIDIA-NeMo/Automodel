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

"""MOK initialization must preserve weights when FSDP splits local experts."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts
from nemo_automodel.components.moe.mok_experts import GroupedExpertsMoK


def _shard_features(parameter: nn.Parameter) -> Shard:
    """Match the production expert FSDP placement.

    Args:
        parameter: Parameter of shape [experts, output_features, input_features].

    Returns:
        Placement splitting output features across FSDP ranks.
    """
    return Shard(1)


def _check_init(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=90)
    )
    try:
        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=1536,
            norm_topk_prob=True,
        )
        # Expert parallelism alone, FSDP alone, and their composition. MOK's
        # kernel is not needed to exercise its real distributed parameters.
        for fsdp_size, ep_size in ((1, 4), (4, 1), (2, 2)):
            mesh = init_device_mesh("cpu", (fsdp_size, ep_size), mesh_dim_names=("ep_shard", "ep"))
            for dtype in (torch.bfloat16, torch.float32):
                reference = GroupedExperts(replace(config, n_routed_experts=config.n_routed_experts // ep_size))
                reference.to(dtype=dtype)
                torch.manual_seed(1234)
                with torch.no_grad():
                    reference.init_weights(torch.device("cpu"))
                expected_rng = torch.get_rng_state()
                expected = {
                    "routed_gate_weights": reference.gate_and_up_projs[..., : config.moe_inter_dim].transpose(-1, -2),
                    "routed_up_weights": reference.gate_and_up_projs[..., config.moe_inter_dim :].transpose(-1, -2),
                    "routed_down_weights": reference.down_projs.transpose(-1, -2),
                }

                with torch.device("meta"):
                    model = GroupedExpertsMoK(config, BackendConfig(dispatcher="mok")).to(dtype=dtype)
                if ep_size > 1:
                    for name, parameter in list(model.named_parameters()):
                        setattr(model, name, nn.Parameter(distribute_tensor(parameter, mesh["ep"], [Shard(0)])))
                if fsdp_size > 1:
                    fully_shard(model, mesh=mesh["ep_shard"], shard_placement_fn=_shard_features)
                model.to_empty(device="cpu")
                for name, parameter in list(model.named_parameters()):
                    with torch.no_grad():
                        parameter.to_local().fill_(float("nan"))
                originals = {
                    name: (parameter, parameter.to_local().data_ptr()) for name, parameter in model.named_parameters()
                }

                torch.manual_seed(1234)
                with torch.no_grad():
                    model.init_weights(torch.device("cpu"))

                assert torch.equal(torch.get_rng_state(), expected_rng)
                for name, parameter in model.named_parameters():
                    original, storage = originals[name]
                    assert parameter is original
                    assert parameter.to_local().data_ptr() == storage
                    assert torch.isfinite(parameter.to_local()).all()
                    # Existing EP initialization draws the same canonical stream
                    # independently for each EP rank. FSDP must only slice it.
                    reference_full = expected[name].repeat(ep_size, 1, 1)
                    torch.testing.assert_close(parameter.full_tensor(), reference_full, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    45, reason="Four fresh Gloo workers validate composed EP/FSDP placements and both weight dtypes."
)
def test_mok_init_preserves_canonical_weights_across_sharding(tmp_path: Path) -> None:
    mp.spawn(_check_init, args=(str(tmp_path / "rendezvous"),), nprocs=4, join=True)
