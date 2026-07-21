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

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from nemo_automodel.components.models.bagel.hf_backbone_loader import initialize_bagel_non_backbone_weights


class _PositionEmbedding(nn.Module):
    def __init__(self, hidden_size: int = 8, max_num_patch_per_side: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.pos_embed = nn.Parameter(
            torch.empty(max_num_patch_per_side**2, hidden_size),
            requires_grad=False,
        )


class _BagelOwnedModules(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_embedder = nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 8))
        self.vae2llm = nn.Linear(4, 8)
        self.llm2vae = nn.Linear(8, 4)
        self.latent_pos_embed = _PositionEmbedding()
        self.connector = nn.Sequential(nn.Linear(6, 8), nn.GELU(), nn.Linear(8, 8))
        self.vit_pos_embed = _PositionEmbedding()


class _BagelModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _BagelOwnedModules()
        self.config = SimpleNamespace(visual_gen=True, visual_und=True)


def test_non_backbone_initialization_is_seeded_and_does_not_advance_global_rng() -> None:
    torch.manual_seed(1)
    first = _BagelModel()
    torch.manual_seed(2)
    second = _BagelModel()

    torch.manual_seed(123)
    expected_next = torch.rand(4)
    torch.manual_seed(123)
    initialize_bagel_non_backbone_weights(first, seed=4396)
    actual_next = torch.rand(4)
    initialize_bagel_non_backbone_weights(second, seed=4396)

    assert torch.equal(actual_next, expected_next)
    for name, first_parameter in first.named_parameters():
        assert torch.equal(first_parameter, dict(second.named_parameters())[name])
    assert torch.count_nonzero(first.model.llm2vae.weight) == 0
    assert torch.count_nonzero(first.model.llm2vae.bias) == 0


def test_non_backbone_initialization_changes_with_seed() -> None:
    first = _BagelModel()
    second = _BagelModel()

    initialize_bagel_non_backbone_weights(first, seed=1)
    initialize_bagel_non_backbone_weights(second, seed=2)

    assert not torch.equal(first.model.connector[0].weight, second.model.connector[0].weight)


def test_non_backbone_initialization_supports_dtensor_parameters() -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    started_process_group = False
    if not dist.is_initialized():
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
        started_process_group = True

    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        reference = _BagelModel()
        sharded = _BagelModel()
        for module in sharded.modules():
            for name, parameter in list(module.named_parameters(recurse=False)):
                distributed = distribute_tensor(parameter.detach(), mesh, [Shard(0)])
                setattr(module, name, nn.Parameter(distributed, requires_grad=parameter.requires_grad))

        initialize_bagel_non_backbone_weights(reference, seed=4396)
        initialize_bagel_non_backbone_weights(sharded, seed=4396)

        reference_parameters = dict(reference.named_parameters())
        for name, parameter in sharded.named_parameters():
            assert isinstance(parameter, DTensor)
            assert torch.equal(parameter.full_tensor(), reference_parameters[name])
    finally:
        if started_process_group:
            dist.destroy_process_group()
