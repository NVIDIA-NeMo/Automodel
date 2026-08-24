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

"""Real MegatronFSDP SUM-vs-average gradient parity through Engine.

Run with::

    torchrun --standalone --nproc-per-node=2 run_megatron_fsdp_per_token_loss.py
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn

from nemo_automodel.components.distributed.config import MegatronFSDPConfig
from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.engine import Engine


class TinyBlock(nn.Module):
    """One real MegatronFSDP wrapping unit operating on ``[batch, 4]`` tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project float token features from ``[batch, 4]`` to ``[batch, 4]``."""
        return self.projection(tokens)


class TinyTokenModel(nn.Module):
    """Tiny model whose block class is auto-derived as a MegatronFSDP unit."""

    _no_split_modules = ["TinyBlock"]

    def __init__(self) -> None:
        super().__init__()
        self.block = TinyBlock()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map numeric token rows ``[batch, 4]`` to outputs of the same shape."""
        return self.block(input_ids.to(torch.float32))


def _windows(rank: int, device: torch.device) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Build two rank-local windows with unequal global token denominators.

    Args:
        rank: Data-parallel rank, zero or one.
        device: CUDA device on which to construct each microbatch.

    Returns:
        Two ``(input_ids, weights)`` microbatches with shape ``[1, 4]``.
        Across DP, their weight sums are three and four, respectively.
    """
    if rank == 0:
        values_a, weights_a = [1, 2, 3, 4], [1.0, 0.0, 0.0, 0.0]
        values_b, weights_b = [5, 6, 7, 8], [0.0, 1.0, 1.0, 1.0]
    else:
        values_a, weights_a = [9, 10, 11, 12], [1.0, 1.0, 0.0, 0.0]
        values_b, weights_b = [13, 14, 15, 16], [0.0, 0.0, 0.0, 1.0]

    def microbatch(values: list[int], weights: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([values], device=device),
            torch.tensor([weights], device=device),
        )

    return microbatch(values_a, weights_a), microbatch(values_b, weights_b)


def _local_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot float32 local shards of all named model parameters."""
    parameters = {}
    for name, parameter in model.named_parameters():
        value = parameter.detach()
        value = value.to_local() if hasattr(value, "to_local") else value
        parameters[name] = value.float().cpu().clone()
    return parameters


def _weighted_identity_loss(output: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return the weighted model-output numerator.

    Args:
        output: Model values shaped ``[batch=1, sequence=4]``.
        weights: Per-token weights with the same ``[batch=1, sequence=4]`` layout.

    Returns:
        Scalar rank-local weighted numerator.
    """
    assert output.shape == weights.shape
    return (output * weights.to(output)).sum()


def _run_mode(
    mesh_context: MeshContext,
    *,
    summed_gradients: bool,
) -> tuple[tuple[float, float, float], float, dict[str, torch.Tensor]]:
    """Run one complete update through a real MegatronFSDP reduction mode.

    Args:
        mesh_context: Two-rank ``[dp=2, cp=1, tp=1]`` CUDA mesh.
        summed_gradients: Value passed as ``calculate_per_token_loss``. True
            selects SUM gradient collectives; false selects averaged gradients.

    Returns:
        The complete-window loss sum, weight sum, and normalized loss; the
        global gradient norm; and float32 local parameter shards after the
        update.
    """
    torch.manual_seed(1234)
    model = TinyTokenModel().cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = MegatronFSDPConfig(
        zero_dp_strategy=3,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        check_for_nan_in_grad=False,
        disable_bucketing=True,
        calculate_per_token_loss=summed_gradients,
    )
    model, optimizer = MegatronFSDPManager(config, mesh_context.device_mesh).parallelize(model, optimizer)
    windows = _windows(dist.get_rank(), device)
    engine = Engine(
        model,
        optimizer=optimizer,
        mesh_context=mesh_context,
        max_grad_norm=1e9,
        gradient_accumulation_steps=len(windows),
    )

    global_weight_sum = sum(weights.sum() for _, weights in windows)
    dist.all_reduce(global_weight_sum)
    reduction_scale = dist.get_world_size() / global_weight_sum

    local_loss_sum = torch.zeros((), device=device)
    for input_ids, weights in windows:
        loss_sum = _weighted_identity_loss(engine(input_ids=input_ids), weights)
        local_loss_sum += loss_sum.detach()
        engine.backward(loss_sum * reduction_scale, scale_wrt_gas=False)
        engine.step()

    global_loss_sum = local_loss_sum.clone()
    dist.all_reduce(global_loss_sum)

    statistics = (
        global_loss_sum.item(),
        global_weight_sum.item(),
        (global_loss_sum / global_weight_sum).item(),
    )
    return statistics, float(engine.get_global_grad_norm()), _local_parameters(model)


def main() -> None:
    """Assert real MegatronFSDP SUM and average modes produce one update."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise ValueError(f"MegatronFSDP per-token parity requires two ranks, got {dist.get_world_size()}")
    torch.cuda.set_device(int(torch.distributed.get_rank() % torch.cuda.device_count()))

    mesh_context = MeshContext.build(
        MegatronFSDPConfig(),
        ParallelismSizes(dp_size=2, cp_size=1, tp_size=1),
        world_size=2,
    )
    averaged = _run_mode(mesh_context, summed_gradients=False)
    dist.barrier()
    summed = _run_mode(mesh_context, summed_gradients=True)

    assert math.isfinite(averaged[1]) and averaged[1] > 0
    assert math.isfinite(summed[1]) and summed[1] > 0
    torch.testing.assert_close(torch.tensor(summed[0]), torch.tensor(averaged[0]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.tensor(summed[1]), torch.tensor(averaged[1]), rtol=1e-5, atol=1e-5)
    assert summed[0][1] == 7.0
    assert set(summed[2]) == set(averaged[2])
    for name in sorted(averaged[2]):
        torch.testing.assert_close(summed[2][name], averaged[2][name], rtol=1e-5, atol=1e-5)

    if rank == 0:
        print("MegatronFSDP calculate_per_token_loss SUM gradients match averaged-gradient Engine update")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
