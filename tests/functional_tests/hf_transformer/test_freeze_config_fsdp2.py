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

"""Real two-rank FSDP2 lifecycle coverage for generic freeze configuration."""

import copy
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel._transformers.infrastructure import apply_model_infrastructure
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.mesh import MeshContext

_WORLD_SIZE = 2
_FEATURES = 4


class _TinyBackbone(nn.Module):
    """One-layer backbone exposing AutoModel's generic layer-container contract."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(_FEATURES, _FEATURES, bias=False)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the backbone layer.

        Args:
            inputs: Tensor of shape [batch, features].

        Returns:
            Tensor of shape [batch, features].
        """
        return self.layers[0](inputs)


class _TinyFreezeModel(nn.Module):
    """Small model with frozen model-owned state and an explicitly selected head."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()
        self.classifier = nn.Linear(_FEATURES, 1, bias=False)
        self.classifier.requires_grad_(False)
        self.model_constant = nn.Parameter(torch.ones(1), requires_grad=False)
        self.config = SimpleNamespace(use_cache=False, num_kv_shared_layers=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the frozen backbone and selected classifier.

        Args:
            inputs: Tensor of shape [batch, features].

        Returns:
            Tensor of shape [batch, 1].
        """
        return self.classifier(self.backbone(inputs))


def _free_port() -> int:
    """Return an available localhost TCP port for the spawned process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Replicate a possibly sharded tensor on every rank.

    Args:
        tensor: Tensor of arbitrary shape, either local or sharded on the FSDP mesh.

    Returns:
        Tensor with the input's global shape, replicated on every rank.
    """
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _worker(rank: int, port: int) -> None:
    """Run one rank of the real FSDP2 trainability lifecycle regression."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        torch.manual_seed(1234)
        model = _TinyFreezeModel().cuda(rank)
        reference = copy.deepcopy(model)
        reference.backbone.requires_grad_(False)
        reference.classifier.requires_grad_(True)

        mesh = init_device_mesh(
            "cuda",
            (1, _WORLD_SIZE, 1),
            mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"),
        )
        config = FSDP2Config(
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            ),
            enable_fsdp2_prefetch=False,
        )
        model = apply_model_infrastructure(
            model=model,
            is_meta_device=False,
            device=torch.device("cuda", rank),
            load_base_model=False,
            model_wrapper=FSDP2Manager(config, device_mesh=mesh),
            mesh=MeshContext.from_meshes(mesh),
            freeze_config={
                "freeze_modules": [{"path": "backbone"}],
                "unfreeze_modules": [{"path": "classifier"}],
            },
        )

        assert isinstance(model.backbone.layers[0].weight, DTensor)
        assert isinstance(model.classifier.weight, DTensor)
        assert isinstance(model.model_constant, DTensor)
        assert not model.backbone.layers[0].weight.requires_grad
        assert model.classifier.weight.requires_grad
        assert not model.model_constant.requires_grad
        assert [name for name, parameter in model.named_parameters() if parameter.requires_grad] == [
            "classifier.weight"
        ]

        optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.1)
        reference_optimizer = torch.optim.SGD(
            (parameter for parameter in reference.parameters() if parameter.requires_grad), lr=0.1
        )

        rank_inputs = torch.full((2, _FEATURES), float(rank + 1), device=rank)
        model(rank_inputs).sum().backward()

        reference_loss = (
            sum(
                reference(torch.full((2, _FEATURES), float(source_rank + 1), device=rank)).sum()
                for source_rank in range(_WORLD_SIZE)
            )
            / _WORLD_SIZE
        )
        reference_loss.backward()

        assert model.backbone.layers[0].weight.grad is None
        assert model.classifier.weight.grad is not None
        torch.testing.assert_close(
            _full_tensor(model.classifier.weight.grad),
            reference.classifier.weight.grad,
        )

        optimizer.step()
        reference_optimizer.step()
        full_classifier_weight = _full_tensor(model.classifier.weight)
        torch.testing.assert_close(full_classifier_weight, reference.classifier.weight)

        gathered_weights = [torch.empty_like(full_classifier_weight) for _ in range(_WORLD_SIZE)]
        dist.all_gather(gathered_weights, full_classifier_weight)
        for gathered_weight in gathered_weights:
            torch.testing.assert_close(gathered_weight, full_classifier_weight)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason="requires two CUDA GPUs",
)
def test_freeze_config_survives_real_fsdp2_forward_backward_and_optimizer_step() -> None:
    """Selected FSDP2 parameters remain trainable and synchronized through one update."""
    mp.spawn(_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)
