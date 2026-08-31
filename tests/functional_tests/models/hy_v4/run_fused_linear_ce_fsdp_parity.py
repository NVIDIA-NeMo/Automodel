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

"""Distributed HY V4 LM-head gradient parity for materialized and fused CE."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.models.hy_v4.model import HyV4LMHead


def _local_grad(parameter: torch.nn.Parameter) -> torch.Tensor:
    """Return one rank's local parameter-gradient shard in FP32."""
    grad = parameter.grad
    if grad is None:
        raise AssertionError("expected an LM-head gradient")
    if isinstance(grad, DTensor):
        grad = grad.to_local()
    return grad.detach().float()


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return relative L2 and cosine metrics as CUDA scalars."""
    delta = actual - expected
    rel_l2 = delta.norm() / expected.norm().clamp_min(1.0e-30)
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).clamp(-1.0, 1.0)
    return rel_l2, cosine


def main() -> None:
    """Compare the exact FSDP-sharded LM-head path on every data rank."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))

        torch.manual_seed(1234)
        weight = torch.randn(args.vocab, args.hidden, device=device, dtype=torch.bfloat16) / args.hidden**0.5
        reference_head = HyV4LMHead(
            args.hidden,
            args.vocab,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        fused_head = HyV4LMHead(
            args.hidden,
            args.vocab,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            reference_head.weight.copy_(weight)
            fused_head.weight.copy_(weight)
        del weight
        fully_shard(reference_head, mesh=mesh, reshard_after_forward=True)
        fully_shard(fused_head, mesh=mesh, reshard_after_forward=True)

        generator = torch.Generator(device=device).manual_seed(9000 + rank)
        hidden = torch.randn(
            1,
            args.tokens,
            args.hidden,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        labels = torch.randint(
            0,
            args.vocab,
            (1, args.tokens),
            device=device,
            dtype=torch.long,
            generator=generator,
        )
        labels[:, ::17] = -100

        reference_hidden = hidden.detach().clone().requires_grad_(True)
        reference_logits = reference_head(reference_hidden)
        reference_loss = F.cross_entropy(
            reference_logits.reshape(-1, args.vocab),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        reference_loss.backward()
        reference_weight_grad = _local_grad(reference_head.weight).clone()
        reference_hidden_grad = reference_hidden.grad.detach().float().clone()

        fused_hidden = hidden.detach().clone().requires_grad_(True)
        fused_loss = FusedLinearCrossEntropy(ignore_index=-100, reduction="sum")(
            fused_hidden,
            labels,
            fused_head.weight,
            grad_reduce_group=dist.group.WORLD,
        )
        fused_loss.backward()
        fused_weight_grad = _local_grad(fused_head.weight)
        fused_hidden_grad = fused_hidden.grad.detach().float()

        hidden_rel_l2, hidden_cosine = _metrics(fused_hidden_grad, reference_hidden_grad)
        weight_rel_l2, weight_cosine = _metrics(fused_weight_grad, reference_weight_grad)
        local_loss_abs = (fused_loss - reference_loss).abs().float()
        metric_tensor = torch.stack(
            [local_loss_abs, hidden_rel_l2, 1.0 - hidden_cosine, weight_rel_l2, 1.0 - weight_cosine]
        )
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.MAX)

        passed = bool(
            metric_tensor[0] <= 1.0e-3
            and metric_tensor[1] <= 2.0e-2
            and metric_tensor[2] <= 2.0e-4
            and metric_tensor[3] <= 5.0e-3
            and metric_tensor[4] <= 1.0e-5
        )
        if rank == 0:
            result = {
                "passed": passed,
                "world_size": world,
                "shape_per_rank": {"tokens": args.tokens, "hidden": args.hidden, "vocab": args.vocab},
                "max_across_ranks": {
                    "loss_abs": metric_tensor[0].item(),
                    "hidden_grad_rel_l2": metric_tensor[1].item(),
                    "hidden_grad_one_minus_cosine": metric_tensor[2].item(),
                    "weight_grad_rel_l2": metric_tensor[3].item(),
                    "weight_grad_one_minus_cosine": metric_tensor[4].item(),
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2), flush=True)
        if not passed:
            raise SystemExit(1)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
