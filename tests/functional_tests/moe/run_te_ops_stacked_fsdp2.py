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

"""Two-GPU FSDP2 lifecycle guard for stacked ``GroupedExpertsTeOps`` owners.

Run with::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 \
        tests/functional_tests/moe/run_te_ops_stacked_fsdp2.py

The test deliberately uses a local permutation dispatcher. Both ranks therefore
form one ``ep_shard=2`` group while expert parallelism itself is size one. This
isolates the stacked-parameter/FSDP contract from DeepEP or HybridEP communication:

* TE-ops projections register ordinary stacked ``nn.Parameter`` owners on meta;
* AutoModel's production meta materializer and initializer preserve owner identity;
* FSDP shards TE-layout weights on dim 2 and biases on dim 1;
* three activation-checkpointed BF16 forward/backward/optimizer steps use different
  per-expert token splits in both the original forward and recomputation;
* ``reshard_after_forward=True`` repeatedly invalidates and rebuilds runtime views;
* canonical expert state keys survive a distributed-checkpoint round trip.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsTeOps
from nemo_automodel.components.moe.parallelizer import _moe_shard_placement

NUM_EXPERTS = 4
TOP_K = 2
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 64
TOKENS = 12
STEPS = 3

CANONICAL_SHAPES = {
    "gate_and_up_projs": (NUM_EXPERTS, HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE),
    "down_projs": (NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
    "gate_up_proj_bias": (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE),
    "down_proj_bias": (NUM_EXPERTS, HIDDEN_SIZE),
}

OWNER_SHARD_DIMS = {
    "gate_up_linear._stacked_weight": 2,
    "gate_up_linear._stacked_bias": 1,
    "down_linear._stacked_weight": 2,
    "down_linear._stacked_bias": 1,
}


class _LocalPermutationDispatcher:
    """Group compact top-k routes by expert without doing EP communication."""

    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self._token_ids: torch.Tensor | None = None
        self._num_tokens = 0
        self.last_splits: tuple[int, ...] | None = None
        self.runtime_alias_check: Callable[[], None] | None = None

    def token_permutation2(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.runtime_alias_check is not None:
            # This runs inside GroupedExpertsTeOps.forward, after FSDP's root
            # pre-forward hook has all-gathered the stacked owners.
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
        tokens_per_expert = torch.bincount(expert_ids, minlength=self.num_experts)
        self.last_splits = tuple(int(value) for value in tokens_per_expert.cpu().tolist())
        return hidden_states[self._token_ids], tokens_per_expert, permuted_probs

    def token_unpermutation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._token_ids is None:
            raise RuntimeError("token_unpermutation called before token_permutation2")
        output = torch.zeros(
            self._num_tokens,
            hidden_states.shape[-1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return output.index_add(0, self._token_ids, hidden_states)


class _ExpertsBlock(nn.Module):
    """Put the experts behind the same non-reentrant AC boundary used in production."""

    def __init__(self, experts: GroupedExpertsTeOps) -> None:
        super().__init__()
        self.experts = experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.experts(hidden_states, token_mask, weights, indices)


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


def _backend() -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        experts="te_ops",
        dispatcher="deepep",
    )


def _routes(rank: int, step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic, distinct top-k routes and normalized weights."""
    route_pairs = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((2, 3), (0, 1)),
    )
    primary, secondary = route_pairs[step]
    pair_by_token = [primary if token % 3 else secondary for token in range(TOKENS)]
    if rank == 1:
        pair_by_token = [(right, left) for left, right in reversed(pair_by_token)]
    indices = torch.tensor(pair_by_token, dtype=torch.int64, device=device)

    generator = torch.Generator(device="cpu").manual_seed(7100 + 101 * rank + step)
    logits = torch.randn(TOKENS, TOP_K, generator=generator, dtype=torch.float32)
    weights = torch.softmax(logits, dim=-1).to(device=device, dtype=torch.bfloat16)
    return indices, weights


def _input(rank: int, step: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(8300 + 101 * rank + step)
    return torch.randn(TOKENS, HIDDEN_SIZE, generator=generator, dtype=torch.float32).to(
        device=device,
        dtype=torch.bfloat16,
    )


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _assert_plain_stacked_owners(experts: GroupedExpertsTeOps) -> dict[str, nn.Parameter]:
    owners = dict(experts.named_parameters())
    assert set(owners) == set(OWNER_SHARD_DIMS), f"unexpected TE-ops owners: {sorted(owners)}"
    for name, owner in owners.items():
        assert type(owner) is nn.Parameter, f"{name} must be a plain nn.Parameter, got {type(owner)}"
        assert "GroupedTensor" not in type(owner).__name__, f"{name} unexpectedly owns a TE GroupedTensor"
        assert owner.is_meta, f"{name} should start on meta"
    return owners


def _assert_canonical_state(state: dict[str, torch.Tensor]) -> None:
    assert set(state) == set(CANONICAL_SHAPES), f"non-canonical state keys: {sorted(state)}"
    assert all("_stacked" not in key for key in state)
    for key, expected_shape in CANONICAL_SHAPES.items():
        assert tuple(state[key].shape) == expected_shape, (key, state[key].shape, expected_shape)


def _checkpoint_roundtrip(
    experts: GroupedExpertsTeOps,
    reference_input: torch.Tensor,
    reference_mask: torch.Tensor,
    reference_weights: torch.Tensor,
    reference_indices: torch.Tensor,
    rank: int,
) -> None:
    with torch.no_grad():
        expected = experts(reference_input, reference_mask, reference_weights, reference_indices).detach().clone()

    model_state = ModelState(experts)
    state_to_save = model_state.state_dict()
    _assert_canonical_state(state_to_save)

    path_holder: list[str | None] = [tempfile.mkdtemp(prefix="te_ops_stacked_fsdp2_") if rank == 0 else None]
    dist.broadcast_object_list(path_holder, src=0)
    checkpoint_dir = path_holder[0]
    assert checkpoint_dir is not None

    dcp.save(state_to_save, checkpoint_id=checkpoint_dir)
    dist.barrier()

    with torch.no_grad():
        for parameter in experts.parameters():
            _local(parameter).zero_()
        corrupted = experts(reference_input, reference_mask, reference_weights, reference_indices)
    assert not torch.allclose(corrupted.float(), expected.float(), atol=1e-2, rtol=1e-2), (
        "checkpoint corruption did not change output"
    )

    state_to_load = model_state.state_dict()
    _assert_canonical_state(state_to_load)
    dcp.load(state_to_load, checkpoint_id=checkpoint_dir)
    model_state.load_state_dict(state_to_load)

    with torch.no_grad():
        restored = experts(reference_input, reference_mask, reference_weights, reference_indices)
    torch.testing.assert_close(restored.float(), expected.float(), atol=2e-2, rtol=2e-2)

    dist.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_dir)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size != 2:
        raise RuntimeError(f"GroupedExpertsTeOps FSDP2 guard requires exactly 2 ranks, got {world_size}")

    moe_mesh = init_device_mesh(
        "cuda",
        (world_size, 1),
        mesh_dim_names=("ep_shard", "ep"),
    )
    ep_shard_mesh = moe_mesh["ep_shard"]

    torch.manual_seed(20260629)
    torch.cuda.manual_seed_all(20260629)
    experts = GroupedExpertsTeOps(_moe_config(), backend=_backend(), dispatcher_backend="deepep")
    experts.ep_size = 1
    dispatcher = _LocalPermutationDispatcher(NUM_EXPERTS)
    experts.token_dispatcher = dispatcher

    _assert_plain_stacked_owners(experts)
    _assert_canonical_state(experts.state_dict())

    # Production wires the full MoE mesh in init_token_dispatcher before FSDP.
    # The canonical state adapter needs it to reconstruct global DTensor metadata
    # from the input-dimension shards exposed by FSDP outside an unshard window.
    experts.set_moe_mesh(moe_mesh)
    checkpointed_block = ptd_checkpoint_wrapper(
        _ExpertsBlock(experts),
        preserve_rng_state=True,
    )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    fully_shard(
        experts,
        mesh=ep_shard_mesh,
        shard_placement_fn=_moe_shard_placement,
        reshard_after_forward=True,
        mp_policy=mp_policy,
    )

    sharded_owners = dict(experts.named_parameters())
    assert set(sharded_owners) == set(OWNER_SHARD_DIMS)
    sharded_owner_ids = {name: id(parameter) for name, parameter in sharded_owners.items()}
    for name, parameter in sharded_owners.items():
        assert isinstance(parameter, DTensor), f"{name} was not FSDP2-sharded"
        assert _local(parameter).is_meta, f"{name} should still be meta immediately after fully_shard"
        assert len(parameter.placements) == 1
        placement = parameter.placements[0]
        expected_dim = OWNER_SHARD_DIMS[name]
        assert isinstance(placement, Shard) and placement.dim == expected_dim, (
            f"{name} placement is {placement}, expected Shard({expected_dim})"
        )

    # This is the production order: the model is FSDP-parallelized while still on
    # meta, then the checkpointer materializes the already-sharded DTensor owners.
    to_empty_parameters_only(experts, device=device)
    assert sharded_owner_ids == {name: id(owner) for name, owner in experts.named_parameters()}, (
        "post-FSDP meta materialization replaced stacked DTensor owner Parameters"
    )
    assert all(not _local(owner).is_meta and _local(owner).device == device for owner in experts.parameters())
    experts.init_weights(device)
    assert sharded_owner_ids == {name: id(owner) for name, owner in experts.named_parameters()}, (
        "post-FSDP initialization replaced stacked DTensor owner Parameters"
    )

    alias_checks = [0]

    def _assert_runtime_aliases() -> None:
        """Validate TE GroupedTensor aliases inside FSDP's unshard window."""
        for linear in (experts.gate_up_linear, experts.down_linear):
            weight_owner = _local(linear._parameters["_stacked_weight"])
            expected_weight_shape = (linear.num_groups, linear.out_features, linear.in_features)
            assert tuple(weight_owner.shape) == expected_weight_shape
            weight_alias = linear.weight
            assert type(weight_alias).__name__ == "GroupedTensor"
            assert weight_alias.rowwise_data.data_ptr() == weight_owner.data_ptr()

            bias_owner = _local(linear._parameters["_stacked_bias"])
            assert tuple(bias_owner.shape) == (linear.num_groups, linear.out_features)
            bias_alias = linear.bias
            assert type(bias_alias).__name__ == "GroupedTensor"
            assert bias_alias.rowwise_data.data_ptr() == bias_owner.data_ptr()
        alias_checks[0] += 1

    dispatcher.runtime_alias_check = _assert_runtime_aliases

    optimizer = torch.optim.AdamW(experts.parameters(), lr=2e-2, weight_decay=0.0, foreach=False)
    initial_shards = {name: _local(parameter).detach().clone() for name, parameter in experts.named_parameters()}
    observed_splits: list[tuple[int, ...]] = []
    token_mask = torch.ones(TOKENS, dtype=torch.bool, device=device)

    experts.train()
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        hidden_states = _input(rank, step, device).requires_grad_(True)
        indices, weights = _routes(rank, step, device)
        output = checkpointed_block(hidden_states, token_mask, weights, indices)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        # reshard_after_forward=True must restore every persistent owner to its
        # local shard before backward; the runtime aliases above saw full tensors.
        for name, parameter in experts.named_parameters():
            shard_dim = OWNER_SHARD_DIMS[name]
            assert _local(parameter).shape[shard_dim] * world_size == parameter.shape[shard_dim]
        assert dispatcher.last_splits is not None
        observed_splits.append(dispatcher.last_splits)

        loss = output.float().square().mean() + 0.01 * output.float().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()

        for name, parameter in experts.named_parameters():
            assert parameter.grad is not None, f"missing gradient for {name}"
            local_grad = _local(parameter.grad)
            assert torch.isfinite(local_grad).all(), f"non-finite gradient for {name}"
            assert torch.count_nonzero(local_grad).item() > 0, f"zero gradient for {name}"
        optimizer.step()

    assert len(set(observed_splits)) == STEPS, f"expert splits did not vary: {observed_splits}"
    # One call in the original forward and one in non-reentrant checkpoint
    # recomputation during backward for every optimizer step.
    assert alias_checks[0] == 2 * STEPS
    all_splits: list[list[tuple[int, ...]] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(all_splits, observed_splits)
    assert len({split for rank_splits in all_splits for split in rank_splits}) >= STEPS

    for name, parameter in experts.named_parameters():
        assert not torch.equal(_local(parameter), initial_shards[name]), f"optimizer did not update {name}"

    optimizer.zero_grad(set_to_none=True)
    reference_input = _input(rank, STEPS + 1, device)
    reference_indices, reference_weights = _routes(rank, 1, device)
    _checkpoint_roundtrip(
        experts,
        reference_input,
        token_mask,
        reference_weights,
        reference_indices,
        rank,
    )

    if rank == 0:
        print(
            "PASS: GroupedExpertsTeOps plain stacked owners + FSDP2 ep_shard=2 + "
            "activation checkpointing + dynamic splits + optimizer + DCP roundtrip",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
