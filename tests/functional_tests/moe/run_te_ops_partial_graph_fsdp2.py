#!/usr/bin/env python
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

"""Two-GPU eager-vs-partial-graph convergence parity guard for TE-ops experts.

Run with::

    CUDA_VISIBLE_DEVICES=0,1 TE_OPS_GRAPH_MODE=bf16 torchrun --nproc-per-node=2 \
        tests/functional_tests/moe/run_te_ops_partial_graph_fsdp2.py

The runner builds two identically initialized ``GroupedExpertsTeOps`` layers.
Both use FSDP2 ``ep_shard=2``, ``reshard_after_forward=True``, and PyTorch's
non-reentrant activation checkpointing. Only one twin is managed by
``PartialCudaGraphManager``. Every training step compares the graph twin with
the always-eager reference at the output, loss, input-gradient, local sharded
parameter-gradient, and optimizer-updated local-shard boundaries.

Iteration zero is an ordinary eager optimizer step and supplies the graph
capture sample. Later steps change route counts and per-expert splits; an
overflow call exercises eager fallback before a final graph replay verifies
that the fallback did not perturb shared TE operation/quantizer state. Setting
``TE_OPS_GRAPH_ASYMMETRIC_CAPTURE=1`` gives rank 1 zero routes on iteration zero;
it must still join graph-storage collectives, skip local capture, and match its
eager twin on later steps.

``TE_OPS_GRAPH_MODE=mxfp8`` additionally requires
``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1``. It enables TE MXFP8 autocast with BF16 DPA
disabled and requires the full CuTe grouped-MLP block-32 layout. The local
dispatcher models the fused path's paged post-dispatch contract: non-empty
in-bucket calls have 128 physical rows, overflow calls use the next multiple of
128, real split values are retained, and padded output rows are removed before
unpermutation.
"""

from __future__ import annotations

import atexit
import logging
import math
import os
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsTeOps
from nemo_automodel.components.moe.parallelizer import _moe_shard_placement
from nemo_automodel.recipes.llm.partial_cuda_graphs import PartialCudaGraphManager

logger = logging.getLogger(__name__)

NUM_EXPERTS = 4
TOP_K = 2
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 64
TOKENS = 66
EXPERT_BUCKET_TOKENS = 128
CAPTURE_VALID_TOKENS = 11
REPLAY_VALID_TOKENS = (7, 19, 31)
OVERFLOW_VALID_TOKENS = TOKENS
POST_OVERFLOW_VALID_TOKENS = 23

OWNER_SHARD_DIMS = {
    "gate_up_linear._stacked_weight": 2,
    "gate_up_linear._stacked_bias": 0,
    "down_linear._stacked_weight": 2,
    "down_linear._stacked_bias": 0,
}


@dataclass(frozen=True)
class _Tolerance:
    """Numerical parity limits for one compared training signal."""

    atol: float
    rtol: float
    min_cosine: float


_BF16_TOLERANCES = {
    "output": _Tolerance(atol=2e-2, rtol=2e-2, min_cosine=0.9999),
    "loss": _Tolerance(atol=2e-4, rtol=2e-2, min_cosine=1.0),
    "input_grad": _Tolerance(atol=5e-3, rtol=5e-2, min_cosine=0.999),
    "param_grad": _Tolerance(atol=5e-3, rtol=5e-2, min_cosine=0.999),
    "updated_shard": _Tolerance(atol=2e-3, rtol=2e-2, min_cosine=0.9999),
}

_MXFP8_TOLERANCES = {
    "output": _Tolerance(atol=8e-2, rtol=8e-2, min_cosine=0.995),
    "loss": _Tolerance(atol=2e-3, rtol=8e-2, min_cosine=1.0),
    "input_grad": _Tolerance(atol=2e-2, rtol=1e-1, min_cosine=0.99),
    "param_grad": _Tolerance(atol=2e-2, rtol=1e-1, min_cosine=0.99),
    "updated_shard": _Tolerance(atol=5e-3, rtol=5e-2, min_cosine=0.999),
}


class _LocalPermutationDispatcher:
    """Group real routes locally and optionally expose an MXFP8 physical page."""

    def __init__(self, num_experts: int, *, paged_mxfp8: bool) -> None:
        self.num_experts = num_experts
        self.paged_mxfp8 = paged_mxfp8
        self._token_ids: torch.Tensor | None = None
        self._num_tokens = 0
        self._real_route_count = 0
        self.last_splits: tuple[int, ...] | None = None
        self.last_physical_tokens = 0
        self.history: list[tuple[int, ...]] = []
        self.physical_history: list[int] = []
        self.runtime_alias_check: Callable[[], None] | None = None

    def token_permutation2(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.runtime_alias_check is not None:
            # This point is inside the FSDP pre-forward/post-forward window.
            self.runtime_alias_check()
        self._num_tokens = num_local_tokens
        flat_indices = token_indices.reshape(-1)
        flat_probs = token_probs.reshape(-1)
        token_ids = (
            torch.arange(num_local_tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(-1, token_indices.shape[1])
            .reshape(-1)
        )

        valid = flat_indices >= 0
        expert_ids = flat_indices[valid]
        token_ids = token_ids[valid]
        flat_probs = flat_probs[valid]
        order = torch.argsort(expert_ids, stable=True)
        expert_ids = expert_ids[order]
        self._token_ids = token_ids[order]
        permuted_probs = flat_probs[order]
        permuted_hidden_states = hidden_states[self._token_ids]
        tokens_per_expert = torch.bincount(expert_ids, minlength=self.num_experts)

        self._real_route_count = token_ids.numel()
        physical_tokens = self._real_route_count
        # Keep an actually empty capture sample empty. That is what exercises
        # synchronized FSDP storage preparation with one locally skipped graph.
        if self.paged_mxfp8 and self._real_route_count > 0:
            physical_tokens = max(
                EXPERT_BUCKET_TOKENS,
                math.ceil(self._real_route_count / EXPERT_BUCKET_TOKENS) * EXPERT_BUCKET_TOKENS,
            )
            padding = physical_tokens - self._real_route_count
            if padding:
                permuted_hidden_states = torch.cat(
                    (
                        permuted_hidden_states,
                        permuted_hidden_states.new_zeros((padding, permuted_hidden_states.shape[-1])),
                    ),
                    dim=0,
                )
                permuted_probs = torch.cat((permuted_probs, permuted_probs.new_zeros(padding)), dim=0)

        self.last_splits = tuple(int(value) for value in tokens_per_expert.cpu().tolist())
        self.last_physical_tokens = physical_tokens
        self.history.append(self.last_splits)
        self.physical_history.append(physical_tokens)
        return permuted_hidden_states, tokens_per_expert, permuted_probs

    def token_unpermutation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._token_ids is None:
            raise RuntimeError("token_unpermutation called before token_permutation2")
        if hidden_states.shape[0] < self._real_route_count:
            raise RuntimeError(
                f"expert output has {hidden_states.shape[0]} rows for {self._real_route_count} real routes"
            )
        # MXFP8 can return a full physical page. Only real routed rows may enter
        # the combine/unpermutation step.
        hidden_states = hidden_states.narrow(0, 0, self._real_route_count)
        output = torch.zeros(
            self._num_tokens,
            hidden_states.shape[-1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return output.index_add(0, self._token_ids, hidden_states)


class _MoEContainer(nn.Module):
    """Expose the gate-and-experts structure used by generic graph discovery."""

    def __init__(self, experts: GroupedExpertsTeOps) -> None:
        super().__init__()
        self.gate = nn.Identity()
        self.experts = experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.experts(hidden_states, token_mask, weights, indices)


class _ExpertsLayer(nn.Module):
    """A minimal generic transformer layer with one MoE sublayer."""

    def __init__(self, experts: GroupedExpertsTeOps) -> None:
        super().__init__()
        self.mlp = _MoEContainer(experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.mlp(hidden_states, token_mask, weights, indices)


class _GraphGuardModel(nn.Module):
    """Provide the normal model/backend surface consumed by graph discovery."""

    def __init__(self, layer: nn.Module, backend: BackendConfig) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="generic_moe_graph_parity_guard")
        self.backend = backend
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({"0": layer})


@dataclass
class _Twin:
    """One side of the eager-vs-graph comparison."""

    name: str
    experts: GroupedExpertsTeOps
    dispatcher: _LocalPermutationDispatcher
    checkpointed_layer: nn.Module
    optimizer: torch.optim.Optimizer
    backend: BackendConfig
    alias_checks: list[int]
    manager: PartialCudaGraphManager | None = None


@dataclass
class _StepResult:
    """Detached training signals compared after one forward/backward call."""

    output: torch.Tensor
    loss: torch.Tensor
    input_grad: torch.Tensor
    param_grads: dict[str, torch.Tensor]
    splits: tuple[int, ...]
    physical_tokens: int


def _backend(mode: str) -> BackendConfig:
    te_fp8 = {"recipe": "mxfp8", "fp8_dpa": False} if mode == "mxfp8" else None
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        experts="te_ops",
        dispatcher="deepep",
        te_fp8=te_fp8,
        partial_cuda_graph_experts=True,
        partial_cuda_graph_expert_bucket_tokens=EXPERT_BUCKET_TOKENS,
        partial_cuda_graph_layer_limit=1,
    )


def _moe_config() -> MoEConfig:
    return MoEConfig(
        n_routed_experts=NUM_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=TOP_K,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=HIDDEN_SIZE,
        inter_dim=INTERMEDIATE_SIZE,
        moe_inter_dim=INTERMEDIATE_SIZE,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=True,
        expert_activation="quick_geglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.bfloat16,
    )


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _routes(
    rank: int,
    step: int,
    valid_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic masks, route weights, and rank-specific expert IDs."""
    token_ids = torch.arange(TOKENS, device=device)
    # Rotate the valid-token window so split changes are not just prefixes of
    # the same routing pattern. The overflow step naturally marks every token.
    token_mask = ((token_ids - (5 * step + 3 * rank)) % TOKENS) < valid_tokens

    primary = (token_ids + step + rank) % NUM_EXPERTS
    secondary = (3 * token_ids + 2 * step + rank + 1) % NUM_EXPERTS
    secondary = torch.where(secondary == primary, (secondary + 1) % NUM_EXPERTS, secondary)
    indices = torch.stack((primary, secondary), dim=-1).to(torch.int64)

    generator = torch.Generator(device="cpu").manual_seed(19000 + 101 * rank + step)
    logits = torch.randn(TOKENS, TOP_K, generator=generator, dtype=torch.float32)
    weights = torch.softmax(logits, dim=-1).to(device=device, dtype=torch.bfloat16)
    return token_mask, weights, indices


def _input(rank: int, step: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(23000 + 101 * rank + step)
    return torch.randn(TOKENS, HIDDEN_SIZE, generator=generator, dtype=torch.float32).to(
        device=device,
        dtype=torch.bfloat16,
    )


def _assert_logically_sharded(experts: GroupedExpertsTeOps, world_size: int) -> None:
    owners = dict(experts.named_parameters())
    assert set(owners) == set(OWNER_SHARD_DIMS), f"unexpected TE-ops owners: {sorted(owners)}"
    for name, parameter in owners.items():
        assert isinstance(parameter, DTensor), f"{name} is not logically sharded outside the FSDP call"
        assert len(parameter.placements) == 1
        shard_dim = OWNER_SHARD_DIMS[name]
        placement = parameter.placements[0]
        assert isinstance(placement, Shard) and placement.dim == shard_dim, (
            name,
            placement,
            shard_dim,
        )
        assert _local(parameter).shape[shard_dim] * world_size == parameter.shape[shard_dim]


def _assert_parity(
    *,
    label: str,
    graph_value: torch.Tensor,
    eager_value: torch.Tensor,
    tolerance: _Tolerance,
    rank: int,
) -> None:
    graph_float = graph_value.detach().float()
    eager_float = eager_value.detach().float()
    assert graph_float.shape == eager_float.shape, (label, graph_float.shape, eager_float.shape)
    assert torch.isfinite(graph_float).all(), f"{label}: graph value is non-finite"
    assert torch.isfinite(eager_float).all(), f"{label}: eager value is non-finite"
    torch.testing.assert_close(
        graph_float,
        eager_float,
        atol=tolerance.atol,
        rtol=tolerance.rtol,
        msg=lambda message: f"{label} eager-vs-graph mismatch: {message}",
    )

    max_diff = (graph_float - eager_float).abs().max().item() if graph_float.numel() else 0.0
    if graph_float.numel() <= 1:
        cosine = 1.0
    elif graph_float.norm().item() == 0.0 and eager_float.norm().item() == 0.0:
        cosine = 1.0
    else:
        cosine = F.cosine_similarity(graph_float.flatten(), eager_float.flatten(), dim=0).item()
    assert cosine >= tolerance.min_cosine, (
        f"{label}: cosine={cosine:.8f} is below {tolerance.min_cosine:.8f}; max_diff={max_diff:.6e}"
    )
    if rank == 0:
        logger.info("parity %-52s max_diff=%.6e cosine=%.8f", label, max_diff, cosine)


def _build_twin(
    *,
    name: str,
    mode: str,
    moe_mesh: DeviceMesh,
    device: torch.device,
    graph_enabled: bool,
) -> _Twin:
    backend = _backend(mode)
    assert backend.te_fp8 is None or not backend.te_fp8.fp8_dpa
    experts = GroupedExpertsTeOps(_moe_config(), backend=backend, dispatcher_backend="deepep")
    experts.ep_size = 1
    dispatcher = _LocalPermutationDispatcher(NUM_EXPERTS, paged_mxfp8=mode == "mxfp8")
    experts.token_dispatcher = dispatcher
    experts.set_moe_mesh(moe_mesh)

    if mode == "mxfp8":
        assert experts._te_ops_uses_padded_capacity, "MXFP8 guard did not select paged-capacity fusion"
        assert experts._te_ops_full_mxfp8_fusion_eligible, "MXFP8 guard is not eligible for full grouped-MLP fusion"
        assert experts._te_glu_interleave_size == 32, "MXFP8 quick_geglu did not select block-32 layout"
    else:
        assert not experts._te_ops_uses_padded_capacity
        assert experts._te_glu_interleave_size is None

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    fully_shard(
        experts,
        mesh=moe_mesh["ep_shard"],
        shard_placement_fn=_moe_shard_placement,
        reshard_after_forward=True,
        mp_policy=mp_policy,
    )
    to_empty_parameters_only(experts, device=device)
    torch.manual_seed(20260629)
    torch.cuda.manual_seed_all(20260629)
    experts.init_weights(device)
    _assert_logically_sharded(experts, dist.get_world_size())

    alias_checks = [0]

    def _assert_runtime_aliases() -> None:
        for linear in (experts.gate_up_linear, experts.down_linear):
            weight_owner = _local(linear._parameters["_stacked_weight"])
            expected_weight_shape = (linear.num_groups, linear.out_features, linear.in_features)
            assert tuple(weight_owner.shape) == expected_weight_shape
            assert linear.weight.rowwise_data.data_ptr() == weight_owner.data_ptr()

            bias_owner = _local(linear._parameters["_stacked_bias"])
            assert tuple(bias_owner.shape) == (linear.num_groups, linear.out_features)
            assert linear.bias.rowwise_data.data_ptr() == bias_owner.data_ptr()
        alias_checks[0] += 1

    dispatcher.runtime_alias_check = _assert_runtime_aliases
    layer = _ExpertsLayer(experts)
    checkpointed_layer = ptd_checkpoint_wrapper(
        layer,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=True,
    )

    manager = None
    if graph_enabled:
        model = _GraphGuardModel(checkpointed_layer, backend)
        manager = PartialCudaGraphManager.from_model_parts(
            [model],
            activation_checkpointing=True,
        )
        assert manager is not None
        assert len(manager.entries) == 1
        assert manager.stats() == {"captured": 0, "replayed": 0, "fallback": 0}
        assert manager.expert_storage_stats() == {"entries": 1, "active": 0, "retained_bytes": 0}

    optimizer = torch.optim.AdamW(experts.parameters(), lr=2e-2, weight_decay=0.0, foreach=False)
    return _Twin(
        name=name,
        experts=experts,
        dispatcher=dispatcher,
        checkpointed_layer=checkpointed_layer,
        optimizer=optimizer,
        backend=backend,
        alias_checks=alias_checks,
        manager=manager,
    )


def _make_twins(
    *,
    mode: str,
    moe_mesh: DeviceMesh,
    device: torch.device,
) -> tuple[_Twin, _Twin]:
    graph = _build_twin(name="graph", mode=mode, moe_mesh=moe_mesh, device=device, graph_enabled=True)
    eager = _build_twin(name="eager", mode=mode, moe_mesh=moe_mesh, device=device, graph_enabled=False)

    graph_parameters = dict(graph.experts.named_parameters())
    eager_parameters = dict(eager.experts.named_parameters())
    assert graph_parameters.keys() == eager_parameters.keys()
    with torch.no_grad():
        for name in graph_parameters:
            # Copy corresponding local shards instead of relying only on RNG
            # sequencing. This makes identical initialization an explicit test
            # precondition without replacing either FSDP-owned Parameter.
            _local(eager_parameters[name]).copy_(_local(graph_parameters[name]))
            torch.testing.assert_close(
                _local(graph_parameters[name]),
                _local(eager_parameters[name]),
                atol=0,
                rtol=0,
                msg=lambda message, owner=name: f"initial local shard mismatch for {owner}: {message}",
            )
    return graph, eager


def _expected_physical_tokens(mode: str, real_routes: int) -> int:
    if mode != "mxfp8" or real_routes == 0:
        return real_routes
    return max(EXPERT_BUCKET_TOKENS, math.ceil(real_routes / EXPERT_BUCKET_TOKENS) * EXPERT_BUCKET_TOKENS)


def _execute_forward_backward(
    *,
    twin: _Twin,
    mode: str,
    rank: int,
    world_size: int,
    step: int,
    valid_tokens: int,
    device: torch.device,
) -> _StepResult:
    hidden_states = _input(rank, step, device).requires_grad_(True)
    token_mask, weights, indices = _routes(rank, step, valid_tokens, device)
    history_start = len(twin.dispatcher.history)
    physical_history_start = len(twin.dispatcher.physical_history)

    fp8_context: Any = twin.backend.te_fp8.maybe_te_autocast() if twin.backend.te_fp8 is not None else nullcontext()
    # Keep backward inside TE autocast. Non-reentrant checkpoint recomputation
    # occurs during backward and must see the same MXFP8 recipe as the original
    # forward, matching the real non-PP training loop.
    with fp8_context:
        output = twin.checkpointed_layer(hidden_states, token_mask, weights, indices)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        saved_output = output.detach().clone()
        loss = output.float().square().mean() + 0.01 * output.float().mean()
        assert torch.isfinite(loss)
        saved_loss = loss.detach().clone()
        loss.backward()

    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
    assert twin.dispatcher.last_splits is not None
    forward_splits = twin.dispatcher.last_splits
    real_routes = valid_tokens * TOP_K
    assert sum(forward_splits) == real_routes
    if mode == "mxfp8" and real_routes > 0:
        assert twin.experts._te_ops_fusion_checked, (
            f"{twin.name}: MXFP8 autocast did not confirm full CuTe grouped-MLP forward/backward fusion"
        )
    expected_physical_tokens = _expected_physical_tokens(mode, real_routes)
    assert twin.dispatcher.last_physical_tokens == expected_physical_tokens

    # Non-reentrant checkpointing must rerun the same dynamic route and physical
    # layout during backward, including when the nested expert call is replayed.
    assert twin.dispatcher.history[history_start:] == [forward_splits, forward_splits]
    assert twin.dispatcher.physical_history[physical_history_start:] == [
        expected_physical_tokens,
        expected_physical_tokens,
    ]

    param_grads = {}
    for name, parameter in twin.experts.named_parameters():
        assert parameter.grad is not None, f"{twin.name}: missing gradient for {name}"
        local_grad = _local(parameter.grad)
        assert torch.isfinite(local_grad).all(), f"{twin.name}: non-finite gradient for {name}"
        assert torch.count_nonzero(local_grad).item() > 0, f"{twin.name}: zero gradient for {name}"
        param_grads[name] = local_grad.detach().clone()
    _assert_logically_sharded(twin.experts, world_size)
    return _StepResult(
        output=saved_output,
        loss=saved_loss,
        input_grad=hidden_states.grad.detach().clone(),
        param_grads=param_grads,
        splits=forward_splits,
        physical_tokens=expected_physical_tokens,
    )


def _run_parity_step(
    *,
    graph: _Twin,
    eager: _Twin,
    mode: str,
    rank: int,
    world_size: int,
    step: int,
    valid_tokens: int,
    device: torch.device,
) -> tuple[int, tuple[int, ...]]:
    graph.optimizer.zero_grad(set_to_none=False)
    eager.optimizer.zero_grad(set_to_none=False)

    before_graph = {name: _local(parameter).detach().clone() for name, parameter in graph.experts.named_parameters()}

    graph_result = _execute_forward_backward(
        twin=graph,
        mode=mode,
        rank=rank,
        world_size=world_size,
        step=step,
        valid_tokens=valid_tokens,
        device=device,
    )
    eager_result = _execute_forward_backward(
        twin=eager,
        mode=mode,
        rank=rank,
        world_size=world_size,
        step=step,
        valid_tokens=valid_tokens,
        device=device,
    )

    assert graph_result.splits == eager_result.splits
    assert graph_result.physical_tokens == eager_result.physical_tokens
    tolerances = _MXFP8_TOLERANCES if mode == "mxfp8" else _BF16_TOLERANCES
    label_prefix = f"{mode}/step={step}/rank={rank}"
    _assert_parity(
        label=f"{label_prefix}/output",
        graph_value=graph_result.output,
        eager_value=eager_result.output,
        tolerance=tolerances["output"],
        rank=rank,
    )
    _assert_parity(
        label=f"{label_prefix}/loss",
        graph_value=graph_result.loss,
        eager_value=eager_result.loss,
        tolerance=tolerances["loss"],
        rank=rank,
    )
    _assert_parity(
        label=f"{label_prefix}/input_grad",
        graph_value=graph_result.input_grad,
        eager_value=eager_result.input_grad,
        tolerance=tolerances["input_grad"],
        rank=rank,
    )
    assert graph_result.param_grads.keys() == eager_result.param_grads.keys()
    for name in graph_result.param_grads:
        _assert_parity(
            label=f"{label_prefix}/param_grad/{name}",
            graph_value=graph_result.param_grads[name],
            eager_value=eager_result.param_grads[name],
            tolerance=tolerances["param_grad"],
            rank=rank,
        )

    graph.optimizer.step()
    eager.optimizer.step()
    changed_parameters = 0
    for name, graph_parameter in graph.experts.named_parameters():
        eager_parameter = dict(eager.experts.named_parameters())[name]
        graph_shard = _local(graph_parameter)
        eager_shard = _local(eager_parameter)
        _assert_parity(
            label=f"{label_prefix}/updated_shard/{name}",
            graph_value=graph_shard,
            eager_value=eager_shard,
            tolerance=tolerances["updated_shard"],
            rank=rank,
        )
        changed_parameters += int(not torch.equal(graph_shard, before_graph[name]))
    assert changed_parameters == len(before_graph), (
        f"{label_prefix}: optimizer updated only {changed_parameters}/{len(before_graph)} local shards"
    )
    _assert_logically_sharded(graph.experts, world_size)
    _assert_logically_sharded(eager.experts, world_size)
    return valid_tokens * TOP_K, graph_result.splits


def _assert_manager_stats(
    *,
    manager: PartialCudaGraphManager,
    mode: str,
    capture_on_rank: bool,
    replay_route_counts: list[int],
    retained_bytes: int,
) -> None:
    expected_replays = 2 * len(replay_route_counts) if capture_on_rank else 0
    assert manager.stats() == {
        "captured": int(capture_on_rank),
        "replayed": expected_replays,
        "fallback": 2 if capture_on_rank else 0,
    }

    expected_padding = 0
    if mode == "bf16" and capture_on_rank:
        expected_padding = 2 * sum(EXPERT_BUCKET_TOKENS - route_count for route_count in replay_route_counts)
    assert manager.expert_bucket_stats() == {
        "entries": 1,
        "capacity_tokens": EXPERT_BUCKET_TOKENS,
        "bucketed_replay": expected_replays,
        "padding_tokens": expected_padding,
        "overflow_fallback": 2 if capture_on_rank else 0,
        "empty_fallback": 0,
        "capture_overflow_skip": 0,
        "capture_empty_skip": int(not capture_on_rank),
    }
    assert manager.expert_storage_stats() == {
        "entries": 1,
        "active": int(capture_on_rank),
        "retained_bytes": retained_bytes,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mode = os.environ.get("TE_OPS_GRAPH_MODE", "bf16").lower()
    if mode not in ("bf16", "mxfp8"):
        raise ValueError(f"TE_OPS_GRAPH_MODE must be 'bf16' or 'mxfp8', got {mode!r}")
    if mode == "mxfp8" and int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
        raise RuntimeError("MXFP8 parity guard requires NVTE_CUTEDSL_FUSED_GROUPED_MLP=1")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size != 2:
        raise RuntimeError(f"TE-ops partial graph FSDP2 parity guard requires exactly 2 ranks, got {world_size}")
    capability = torch.cuda.get_device_capability(device)
    if capability < (10, 0):
        raise RuntimeError(f"TE-ops dynamic-split graph parity guard requires SM100+, got {capability}")

    asymmetric_capture = os.environ.get("TE_OPS_GRAPH_ASYMMETRIC_CAPTURE", "0") == "1"
    capture_on_rank = not asymmetric_capture or rank == 0
    moe_mesh = init_device_mesh(
        "cuda",
        (world_size, 1),
        mesh_dim_names=("ep_shard", "ep"),
    )

    torch.manual_seed(20260629)
    torch.cuda.manual_seed_all(20260629)
    graph, eager = _make_twins(mode=mode, moe_mesh=moe_mesh, device=device)
    assert graph.manager is not None
    # Keep failure paths deterministic too: an uncaught parity assertion must
    # release TE graphs before Python tears down CUDA/NCCL module globals.
    close_manager = graph.manager.close
    atexit.register(close_manager)

    # Iteration zero is eager for both twins and provides the graph twin's real
    # post-dispatch capture sample. It is a complete optimizer step before
    # capture, exactly as in the production training lifecycle.
    capture_valid_tokens = CAPTURE_VALID_TOKENS if capture_on_rank else 0
    capture_routes, capture_splits = _run_parity_step(
        graph=graph,
        eager=eager,
        mode=mode,
        rank=rank,
        world_size=world_size,
        step=0,
        valid_tokens=capture_valid_tokens,
        device=device,
    )
    assert capture_routes < EXPERT_BUCKET_TOKENS
    graph_entry = graph.manager.entries[0]
    assert graph_entry.expert_bucket_uses_paged_capacity
    if capture_on_rank:
        captured_call = graph_entry.captured_call
        assert captured_call is not None
        assert captured_call.sample_tensors[0].shape == (EXPERT_BUCKET_TOKENS, HIDDEN_SIZE)
        captured_splits = captured_call.sample_tensors[1]
        torch.testing.assert_close(
            captured_splits,
            torch.tensor(capture_splits, dtype=torch.int64, device=device),
            atol=0,
            rtol=0,
        )
        assert captured_splits.sum().item() == capture_routes
    graph.manager.capture()
    assert graph.manager.stats() == {"captured": int(capture_on_rank), "replayed": 0, "fallback": 0}
    storage_after_capture = graph.manager.expert_storage_stats()
    assert storage_after_capture["entries"] == 1
    assert storage_after_capture["active"] == int(capture_on_rank)
    assert (storage_after_capture["retained_bytes"] > 0) == capture_on_rank
    retained_bytes = storage_after_capture["retained_bytes"]
    _assert_logically_sharded(graph.experts, world_size)
    _assert_logically_sharded(eager.experts, world_size)

    replay_route_counts: list[int] = []
    observed_splits = [capture_splits]
    for step, valid_tokens in enumerate(REPLAY_VALID_TOKENS, start=1):
        route_count, splits = _run_parity_step(
            graph=graph,
            eager=eager,
            mode=mode,
            rank=rank,
            world_size=world_size,
            step=step,
            valid_tokens=valid_tokens,
            device=device,
        )
        assert route_count < EXPERT_BUCKET_TOKENS
        replay_route_counts.append(route_count)
        observed_splits.append(splits)
        assert graph.manager.expert_storage_stats()["retained_bytes"] == retained_bytes

    # The real route count exceeds 128. BF16 reaches the manager with 132 rows;
    # MXFP8 reaches it with the dispatcher's next physical page (256 rows). Both
    # must bypass the graph and remain numerically aligned with eager execution.
    overflow_step = len(REPLAY_VALID_TOKENS) + 1
    overflow_routes, overflow_splits = _run_parity_step(
        graph=graph,
        eager=eager,
        mode=mode,
        rank=rank,
        world_size=world_size,
        step=overflow_step,
        valid_tokens=OVERFLOW_VALID_TOKENS,
        device=device,
    )
    assert overflow_routes > EXPERT_BUCKET_TOKENS
    observed_splits.append(overflow_splits)

    # The TE Sequential used for capture shares BasicOperations and FP8
    # quantizers with the eager fallback target. Replay once more after the
    # overflow to prove that eager execution did not perturb graph state.
    post_overflow_step = overflow_step + 1
    post_overflow_routes, post_overflow_splits = _run_parity_step(
        graph=graph,
        eager=eager,
        mode=mode,
        rank=rank,
        world_size=world_size,
        step=post_overflow_step,
        valid_tokens=POST_OVERFLOW_VALID_TOKENS,
        device=device,
    )
    assert post_overflow_routes < EXPERT_BUCKET_TOKENS
    replay_route_counts.append(post_overflow_routes)
    observed_splits.append(post_overflow_splits)

    assert len(set(replay_route_counts)) == len(REPLAY_VALID_TOKENS) + 1
    assert len(set(observed_splits)) == len(observed_splits), f"expert splits did not vary: {observed_splits}"
    _assert_manager_stats(
        manager=graph.manager,
        mode=mode,
        capture_on_rank=capture_on_rank,
        replay_route_counts=replay_route_counts,
        retained_bytes=retained_bytes,
    )

    # Every real full-experts call runs once in the original forward and once
    # in non-reentrant checkpoint recomputation. Graph capture warmups target
    # the inner TE op directly and therefore do not increment these counters.
    expected_optimizer_steps = 3 + len(REPLAY_VALID_TOKENS)
    assert graph.alias_checks[0] == 2 * expected_optimizer_steps
    assert eager.alias_checks[0] == 2 * expected_optimizer_steps
    assert graph.dispatcher.history == eager.dispatcher.history
    assert graph.dispatcher.physical_history == eager.dispatcher.physical_history

    all_splits: list[list[tuple[int, ...]] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(all_splits, observed_splits)
    assert len({split for rank_splits in all_splits if rank_splits is not None for split in rank_splits}) > len(
        observed_splits
    )

    if rank == 0:
        capture_mode = "asymmetric zero-token-rank" if asymmetric_capture else "symmetric"
        precision = "TE MXFP8 full-fused block32" if mode == "mxfp8" else "BF16"
        logger.info(
            "PASS: eager-vs-partial-graph numerical parity for %s GroupedExpertsTeOps + "
            "FSDP2 ep_shard=2 + non-reentrant AC + dynamic splits/counts + overflow fallback + %s capture",
            precision,
            capture_mode,
        )

    # TE graphed callables own forward/backward CUDAGraph objects, while the
    # manager's FSDP handle intercepts normal unsharded-storage freeing. Tear
    # both down deterministically before NCCL/process-group destruction so the
    # four sequential torchrun cases never depend on interpreter finalization.
    close_manager()
    atexit.unregister(close_manager)
    assert graph.manager.expert_storage_stats() == {"entries": 0, "active": 0, "retained_bytes": 0}
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
