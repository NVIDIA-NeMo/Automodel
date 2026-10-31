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
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common.utils import BackendConfig, TEFp8Config
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


_GRADIENT_NAMES = ("hidden", "probs", "gate_and_up_projs", "down_projs")


def _large_routing_gradients(mesh, *, rank: int, world_size: int, nccl_ep: dict) -> list[tuple[torch.Tensor, ...]]:
    """Run one expert layer at a realistic size without and with non-reentrant recomputation.

    NCCL-EP backward reuses the forward's routing handle while recomputation
    redispatches the tokens, so the recomputed expert layout must match the forward's.
    A few tokens keep their order by accident; thousands of routes per expert do not
    unless the transport fixes the order of each expert's receive slots. The layer is
    rebuilt for each NCCL-EP configuration because a process holds one at a time.

    Args:
        mesh: One-dimensional EP device mesh.
        rank: EP rank.
        world_size: EP size.
        nccl_ep: ``BackendConfig.nccl_ep`` settings for this run.

    Returns:
        Gradients without and with recomputation, each ordered as ``_GRADIENT_NAMES``:
        input [tokens, hidden], routing weights [tokens, top_k], local gate/up weights
        [local_experts, hidden, 2 * inter], and local down weights [local_experts, inter, hidden].
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
    backend = BackendConfig(dispatcher="nccl_ep", experts="torch_mm", nccl_ep=nccl_ep, attn="sdpa", linear="torch")
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
    try:
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
    finally:
        torch.cuda.synchronize()
        finalize_nccl_ep()
    return grads


def _check_large_routing(mesh, *, rank: int, world_size: int) -> None:
    """Require recomputation to be exact and a fixed receive capacity to match per-dispatch buffers.

    Args:
        mesh: One-dimensional EP device mesh.
        rank: EP rank.
        world_size: EP size.
    """
    tokens = 4096
    per_dispatch = _large_routing_gradients(
        mesh, rank=rank, world_size=world_size, nccl_ep={"max_tokens_per_rank": tokens}
    )
    # Factor 2 covers every route this layer can receive, so the fixed buffer cannot overflow.
    fixed = _large_routing_gradients(
        mesh, rank=rank, world_size=world_size, nccl_ep={"max_tokens_per_rank": tokens, "recv_capacity_factor": 2.0}
    )
    for label, (eager, recomputed) in (("recompute", per_dispatch), ("fixed capacity recompute", fixed)):
        for name, expected, actual in zip(_GRADIENT_NAMES, eager, recomputed):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"{label}: {name} gradients differ")
        logger.info("rank=%d case=%s tokens=%d: gradients match without recomputation", rank, label, tokens)
    # Both buffers feed the same grouped GEMMs and SwiGLU kernels the same routed rows.
    for name, expected, actual in zip(_GRADIENT_NAMES, per_dispatch[0], fixed[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"fixed capacity: {name} gradients differ")
    logger.info("rank=%d case=fixed capacity tokens=%d: gradients match per-dispatch buffers", rank, tokens)


def _te_step(
    mesh,
    *,
    rank: int,
    world_size: int,
    nccl_ep: dict,
    te_grouped_tensor: bool,
    te_fp8: str | None,
    outer_recipe: str | None = None,
    checkpointed: bool = False,
) -> list[torch.Tensor]:
    """Run one Transformer Engine expert layer step on fixed weights and routing.

    Args:
        mesh: One-dimensional EP device mesh.
        rank: EP rank.
        world_size: EP size.
        nccl_ep: ``BackendConfig.nccl_ep`` settings.
        te_grouped_tensor: Whether the experts use TE's native grouped-tensor GEMMs.
        te_fp8: ``BackendConfig.te_fp8`` recipe, for which NCCL-EP aligns the expert rows.
        outer_recipe: Recipe enabled outside the backend instead, leaving the rows to TE's padding.
        checkpointed: Whether to recompute the layer in backward.

    Returns:
        Output [tokens, hidden], gradients of the input [tokens, hidden] and routing weights
        [tokens, top_k], then the gradient of every local GEMM weight.
    """
    tokens, hidden_dim, inter_dim, top_k = 1024, 512, 256, 4
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
        inter_dim=inter_dim,
        moe_inter_dim=inter_dim,
        norm_topk_prob=False,
        dtype=torch.bfloat16,
    )
    backend = BackendConfig(
        dispatcher="nccl_ep",
        experts="te",
        te_grouped_tensor=te_grouped_tensor,
        nccl_ep=nccl_ep,
        te_fp8=None if te_fp8 is None else {"recipe": te_fp8},
        attn="sdpa",
        linear="torch",
    )
    recipe = te_fp8 or outer_recipe
    with torch.device("meta"):
        experts = MoE(config, backend).experts
    experts.init_token_dispatcher(mesh)
    experts.to_empty(device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(43 + rank)
    local = config.n_routed_experts // world_size
    gate_up = torch.randn(local, hidden_dim, 2 * inter_dim, device="cuda", generator=generator) * hidden_dim**-0.5
    down = torch.randn(local, inter_dim, hidden_dim, device="cuda", generator=generator) * inter_dim**-0.5
    experts.gate_and_up_projs = gate_up.bfloat16()
    experts.down_projs = down.bfloat16()
    hidden = (torch.randn(tokens, hidden_dim, device="cuda", generator=generator) * 0.5).bfloat16().requires_grad_()
    scores = torch.randn(tokens, config.n_routed_experts, device="cuda", generator=generator)
    probs, indices = scores.softmax(-1).topk(top_k, dim=-1)
    probs.requires_grad_()
    mask = torch.ones(tokens, device="cuda", dtype=torch.bool)
    upstream = torch.randn(tokens, hidden_dim, device="cuda", generator=generator)
    try:
        # As in the training recipes, backward and its recomputation run inside the FP8 context.
        with TEFp8Config(recipe=recipe).maybe_te_autocast() if recipe is not None else nullcontext():
            out = (
                checkpoint(experts, hidden, mask, probs, indices, use_reentrant=False)
                if checkpointed
                else experts(hidden, mask, probs, indices)
            )
            (out.float() * upstream).sum().backward()
        weights = [
            getattr(linear, f"weight{i}").grad
            for linear in (experts.gate_up_linear, experts.down_linear)
            for i in range(linear.num_gemms)
        ]
        return [out.detach(), hidden.grad, probs.grad, *weights]
    finally:
        torch.cuda.synchronize()
        finalize_nccl_ep()


def _check_te_experts(mesh, *, rank: int, world_size: int) -> None:
    """Require exact results from aligned rows, fixed capacity, and zero copy with TE experts.

    NCCL-EP zero-pads each expert's rows to the multiple the FP8 GEMMs need, which TE's
    padding would otherwise add, and the grouped GEMMs never read a fixed buffer's unused
    rows, so each configuration must reproduce the reference bit for bit.

    Args:
        mesh: One-dimensional EP device mesh.
        rank: EP rank.
        world_size: EP size.
    """
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.module.grouped_linear import is_module_grouped_tensor_path_supported

    blackwell = torch.cuda.get_device_capability()[0] >= 10
    max_tokens = {"max_tokens_per_rank": 1024}
    fixed = {**max_tokens, "recv_capacity_factor": 2.0}

    def same(label: str, expected: list[torch.Tensor], actual: list[torch.Tensor]) -> None:
        for index, (reference, value) in enumerate(zip(expected, actual, strict=True)):
            torch.testing.assert_close(value, reference, rtol=0, atol=0, msg=f"{label}: tensor {index} differs")
        logger.info("rank=%d case=te %s: output and gradients match", rank, label)

    def step(**kwargs) -> list[torch.Tensor]:
        return _te_step(mesh, rank=rank, world_size=world_size, **kwargs)

    for recipe in (None, "current", "mxfp8"):
        if recipe == "mxfp8" and not blackwell:
            continue
        name = recipe or "bf16"
        if recipe is not None:
            padded = step(nccl_ep=max_tokens, te_grouped_tensor=False, te_fp8=None, outer_recipe=recipe)
            aligned = step(nccl_ep=max_tokens, te_grouped_tensor=False, te_fp8=recipe)
            same(f"{name} aligned rows", padded, aligned)
        te_recipe = None if recipe is None else TEFp8Config(recipe=recipe).build_recipe()
        if not is_module_grouped_tensor_path_supported(te_recipe, torch.bfloat16):
            continue
        reference = step(nccl_ep=max_tokens, te_grouped_tensor=True, te_fp8=recipe)
        cases = [
            ("recompute", max_tokens, True),
            ("fixed capacity", fixed, False),
            ("fixed capacity recompute", fixed, True),
        ]
        if tex.ep_zero_copy_supported():
            zero_copy = {**fixed, "zero_copy": True}
            cases += [("zero copy", zero_copy, False), ("zero copy recompute", zero_copy, True)]
        for label, nccl_ep, checkpointed in cases:
            actual = step(nccl_ep=nccl_ep, te_grouped_tensor=True, te_fp8=recipe, checkpointed=checkpointed)
            same(f"{name} {label}", reference, actual)


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
        _check_large_routing(mesh, rank=rank, world_size=world_size)
        _check_te_experts(mesh, rank=rank, world_size=world_size)
    finally:
        torch.cuda.synchronize()
        finalize_nccl_ep()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
