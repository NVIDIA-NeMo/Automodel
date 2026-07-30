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

"""Distributed gradient parity for the multi-domain objective."""

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.training.domain_mixture import DomainMixtureConfig, DomainWeightConfig
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


class _TinyTokenClassifier(nn.Module):
    """Linear token classifier used by the distributed parity check."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        """Project token features of shape ``[batch, sequence, 3]`` to four logits."""
        return SimpleNamespace(logits=self.projection(input_ids))


def _mixture():
    return DomainMixtureConfig(
        domains=(
            DomainWeightConfig(name="web", sampling_weight=0.5, objective_weight=0.25),
            DomainWeightConfig(name="code", sampling_weight=0.5, objective_weight=0.75),
        )
    ).build()


def _run_distributed_gradient_parity(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(23)
        model = DistributedDataParallel(_TinyTokenClassifier())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        cfg = ConfigNode({})
        recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
        object.__setattr__(recipe, "dist_env", SimpleNamespace(device=torch.device("cpu")))
        object.__setattr__(recipe, "device_mesh", None)
        object.__setattr__(recipe, "pp_enabled", False)
        object.__setattr__(recipe, "pp", None)
        object.__setattr__(recipe, "tokenizer", SimpleNamespace(pad_token_id=0))
        object.__setattr__(recipe, "te_fp8", None)
        object.__setattr__(recipe, "distributed_config", SimpleNamespace(defer_fsdp_grad_sync=True))
        object.__setattr__(recipe, "model_parts", [model])
        object.__setattr__(recipe, "loss_fn", MaskedCrossEntropy(fp32_upcast=False))
        object.__setattr__(recipe, "domain_mixture", _mixture())

        features = (
            torch.tensor([[[0.2, -0.4, 0.6], [1.0, 0.5, -0.5]]])
            if rank == 0
            else torch.tensor([[[-0.3, 0.7, 0.1], [0.8, -0.2, 0.4]]])
        )
        labels = torch.tensor([[0, 1]]) if rank == 0 else torch.tensor([[2, 3]])
        batch = {"input_ids": features, "labels": labels, "dataset_id": torch.tensor([rank])}
        loss_buffer = []

        recipe._forward_backward_step(
            0,
            batch,
            loss_buffer=loss_buffer,
            num_label_tokens=4,
            num_batches=1,
            is_train=True,
        )

        reference = _TinyTokenClassifier()
        reference.load_state_dict(model.module.state_dict())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        all_features = torch.cat(
            (
                torch.tensor([[[0.2, -0.4, 0.6], [1.0, 0.5, -0.5]]]),
                torch.tensor([[[-0.3, 0.7, 0.1], [0.8, -0.2, 0.4]]]),
            )
        )
        all_labels = torch.tensor([[0, 1], [2, 3]])
        per_token = F.cross_entropy(
            reference(all_features).logits.reshape(-1, 4),
            all_labels.reshape(-1),
            reduction="none",
        ).reshape_as(all_labels)
        reference_loss = (per_token * torch.tensor([[0.5, 0.5], [1.5, 1.5]])).sum() / 4
        reference_loss.backward()

        torch.testing.assert_close(model.module.projection.weight.grad, reference.projection.weight.grad)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.module.projection.weight, reference.projection.weight)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_domain_mixture_gradient_matches_global_two_rank_reference(tmp_path: Path) -> None:
    """Two real DP ranks must match a single-process weighted global objective."""
    mp.spawn(
        _run_distributed_gradient_parity,
        args=(2, str(tmp_path / "domain_mixture_process_group")),
        nprocs=2,
        join=True,
    )
