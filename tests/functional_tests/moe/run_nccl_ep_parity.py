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

"""Run with torchrun on two SM90+ GPUs in an NCCL-EP-enabled container."""

import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.moe.nccl_ep import finalize_nccl_ep

logger = logging.getLogger(__name__)


def _reference(hidden, probs, indices, mask, up, down):
    """Compute independent FP32 routed expert outputs.

    Args:
        hidden: Local tokens [tokens, hidden].
        probs: Routing weights [tokens, top_k].
        indices: Global expert IDs [tokens, top_k].
        mask: Valid-token mask [tokens].
        up: Global gate/up matrices [experts, hidden, 2 * intermediate].
        down: Global down matrices [experts, intermediate, hidden].

    Returns:
        Local output [tokens, hidden].
    """
    output = hidden * 0
    for expert in range(up.shape[0]):
        gate, value = (hidden @ up[expert]).chunk(2, dim=-1)
        weight = (probs * (indices == expert) * mask[:, None]).sum(-1, keepdim=True)
        output = output + ((F.silu(gate) * value) * weight) @ down[expert]
    return output


def _check_case(experts, global_up, global_down, *, rank, world_size, case, checkpointed):
    """Compare a two-microbatch step with the FP32 reference.

    Args:
        experts: Sharded production expert module using NCCL-EP.
        global_up: Initial global gate/up tensor [experts, hidden, 2 * intermediate].
        global_down: Initial global down tensor [experts, intermediate, hidden].
        rank: EP rank.
        world_size: EP size.
        case: Equal, uneven, empty-source, empty-receiver, or masked routing.
        checkpointed: Whether to recompute the production forward during backward.
    """
    local_experts = global_up.shape[0] // world_size
    start = rank * local_experts
    with torch.no_grad():
        experts.gate_and_up_projs.to_local().copy_(global_up[start : start + local_experts])
        experts.down_projs.to_local().copy_(global_down[start : start + local_experts])
    experts.zero_grad(set_to_none=True)
    reference_up = global_up.float().detach().requires_grad_()
    reference_down = global_down.float().detach().requires_grad_()
    actual_losses, reference_losses, inputs = [], [], []
    for microbatch in range(2):
        torch.manual_seed(37 + rank + microbatch * 13)
        tokens = 5 if case != "uneven" or rank == 0 else 3
        if case == "empty_source" and rank != 0:
            tokens = 0
        hidden = (torch.randn(tokens, 512, device="cuda") * 0.2).bfloat16().requires_grad_()
        probs = torch.randn(tokens, 2, device="cuda").softmax(-1).requires_grad_()
        indices = (torch.arange(tokens * 2, device="cuda").reshape(tokens, 2) + rank) % global_up.shape[0]
        if case == "empty_receiver":
            indices = torch.tensor([0, 1], device="cuda").expand(tokens, -1)
        mask = torch.ones(tokens, device="cuda", dtype=torch.bool)
        if case == "masked":
            mask.fill_(False)
        elif tokens:
            mask[-1] = False
        reference_hidden = hidden.detach().float().requires_grad_()
        reference_probs = probs.detach().clone().requires_grad_()
        actual = (
            checkpoint(experts, hidden, mask, probs, indices, use_reentrant=False)
            if checkpointed
            else experts(hidden, mask, probs, indices)
        )
        expected = _reference(reference_hidden, reference_probs, indices, mask, reference_up, reference_down)
        # BF16 GEMMs and activation rounding differ from the FP32 oracle.
        torch.testing.assert_close(actual.float(), expected, rtol=0.05, atol=2e-4)
        upstream = torch.randn_like(expected)
        actual_losses.append((actual.float() * upstream).sum())
        reference_losses.append((expected * upstream).sum())
        inputs.append((hidden, probs, reference_hidden, reference_probs))
    sum(actual_losses).backward()
    sum(reference_losses).backward()
    for hidden, probs, reference_hidden, reference_probs in inputs:
        torch.testing.assert_close(hidden.grad.float(), reference_hidden.grad, rtol=0.05, atol=2e-4)
        torch.testing.assert_close(probs.grad, reference_probs.grad, rtol=0.05, atol=2e-4)
    actual_norm_sq = torch.zeros((), device="cuda")
    expected_norm_sq = torch.zeros((), device="cuda")
    for parameter, reference in zip((experts.gate_and_up_projs, experts.down_projs), (reference_up, reference_down)):
        dist.all_reduce(reference.grad)
        expected_grad = reference.grad[start : start + local_experts]
        actual_grad = parameter.grad.to_local().float()
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0.05, atol=3e-4)
        actual_norm_sq += actual_grad.square().sum()
        expected_norm_sq += expected_grad.square().sum()
        with torch.no_grad():
            parameter.to_local().add_(parameter.grad.to_local(), alpha=-0.01)
            expected_parameter = reference.detach()[start : start + local_experts] - 0.01 * expected_grad
        torch.testing.assert_close(parameter.to_local().float(), expected_parameter, rtol=0.01, atol=2e-4)
    dist.all_reduce(actual_norm_sq)
    dist.all_reduce(expected_norm_sq)
    torch.testing.assert_close(actual_norm_sq.sqrt(), expected_norm_sq.sqrt(), rtol=0.025, atol=2e-4)
    logger.info("rank=%d case=%s checkpoint=%s: output, gradients, norm and update passed", rank, case, checkpointed)


def _check_checkpoint_recompute(mesh, *, rank: int, world_size: int) -> None:
    """Compare gradients with and without non-reentrant recomputation at a realistic size.

    NCCL-EP backward reuses the forward's routing handle while recomputation
    redispatches the tokens, so the recomputed expert layout must match the forward's.
    A few tokens keep their order by accident; thousands of routes per expert do not
    unless the transport fixes the order of each expert's receive slots.

    Args:
        mesh: One-dimensional EP device mesh.
        rank: EP rank.
        world_size: EP size.
    """
    tokens, hidden_dim, top_k = 4096, 512, 4
    config = MoEConfig(
        n_routed_experts=world_size * 8,
        n_shared_experts=0,
        n_activated_experts=top_k,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=hidden_dim,
        inter_dim=256,
        moe_inter_dim=256,
        norm_topk_prob=False,
    )
    backend = BackendConfig(
        dispatcher="nccl_ep", experts="torch_mm", nccl_ep={"max_tokens_per_rank": tokens}, attn="sdpa", linear="torch"
    )
    experts = MoE(config, backend).experts.cuda().bfloat16()
    torch.manual_seed(29)
    global_up = (torch.randn_like(experts.gate_and_up_projs.float()) * 0.02).bfloat16()
    global_down = (torch.randn_like(experts.down_projs.float()) * 0.02).bfloat16()
    experts.gate_and_up_projs = torch.nn.Parameter(distribute_tensor(global_up, mesh, [Shard(0)]))
    experts.down_projs = torch.nn.Parameter(distribute_tensor(global_down, mesh, [Shard(0)]))
    experts.init_token_dispatcher(mesh)
    torch.manual_seed(31 + rank)
    hidden = (torch.randn(tokens, hidden_dim, device="cuda") * 0.2).bfloat16()
    probs, indices = torch.randn(tokens, config.n_routed_experts, device="cuda").softmax(-1).topk(top_k, dim=-1)
    mask = torch.ones(tokens, device="cuda", dtype=torch.bool)
    upstream = torch.randn(tokens, hidden_dim, device="cuda")
    grads = []
    for checkpointed in (False, True):
        experts.zero_grad(set_to_none=True)
        hidden_in = hidden.detach().requires_grad_()
        probs_in = probs.detach().requires_grad_()
        out = (
            checkpoint(experts, hidden_in, mask, probs_in, indices, use_reentrant=False)
            if checkpointed
            else experts(hidden_in, mask, probs_in, indices)
        )
        (out.float() * upstream).sum().backward()
        grads.append(
            (
                hidden_in.grad,
                probs_in.grad,
                experts.gate_and_up_projs.grad.to_local(),
                experts.down_projs.grad.to_local(),
            )
        )
    for name, eager, recomputed in zip(("hidden", "probs", "gate_and_up_projs", "down_projs"), *grads):
        torch.testing.assert_close(recomputed, eager, rtol=0, atol=0, msg=f"{name} gradients differ under recompute")
    logger.info("rank=%d case=recompute tokens=%d: gradients match without recomputation", rank, tokens)


def main() -> None:
    """Validate routed expert outputs and a training step on the EP group."""
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        config = MoEConfig(
            n_routed_experts=world_size * 2,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=512,
            inter_dim=64,
            moe_inter_dim=64,
            norm_topk_prob=False,
        )
        backend = BackendConfig(
            dispatcher="nccl_ep", experts="torch_mm", nccl_ep={"max_tokens_per_rank": 5}, attn="sdpa", linear="torch"
        )
        experts = MoE(config, backend).experts.cuda().bfloat16()
        torch.manual_seed(17)
        global_up = (torch.randn_like(experts.gate_and_up_projs.float()) * 0.02).bfloat16()
        global_down = (torch.randn_like(experts.down_projs.float()) * 0.02).bfloat16()
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ep",))
        experts.gate_and_up_projs = torch.nn.Parameter(distribute_tensor(global_up, mesh, [Shard(0)]))
        experts.down_projs = torch.nn.Parameter(distribute_tensor(global_down, mesh, [Shard(0)]))
        experts.init_token_dispatcher(mesh)
        for checkpointed in (False, True):
            for case in ("equal", "uneven", "empty_source", "empty_receiver", "masked"):
                _check_case(
                    experts,
                    global_up,
                    global_down,
                    rank=rank,
                    world_size=world_size,
                    case=case,
                    checkpointed=checkpointed,
                )
        # NCCL-EP keeps one configuration per process; rebuild it at the larger size.
        finalize_nccl_ep()
        _check_checkpoint_recompute(mesh, rank=rank, world_size=world_size)
    finally:
        torch.cuda.synchronize()
        finalize_nccl_ep()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
