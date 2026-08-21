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

"""Scheduled two-rank CPU parity test for the Engine and finalizer under TP x EP.

The custom-MoE tensor-parallel path keeps the token path (attention, router)
replicated across TP ranks, so every TP rank feeds the same tokens into the
expert-parallel all-gather and each expert gradient accumulates ``tp_size``
identical contributions. This test drives the real ``GroupedExperts`` through
the Datum Engine with tp=2 replicated tokens and a 2-rank EP mesh, then asserts
that the existing gradient finalizer restores the single-process fp32 loss,
gradients, global gradient norm, and one-step parameter update.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_automodel.components.datasets.datum import Datum
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts
from nemo_automodel.engine import Engine, collate_prebatched

_TP_SIZE = 2
_WORLD_SIZE = _TP_SIZE
_N_EXPERTS = 4
_TOP_K = 2
_DIM = 16
_MOE_INTER_DIM = 32
_NUM_TOKENS = 6
_LEARNING_RATE = 0.05


class _ExpertModel(nn.Module):
    """Expose ``GroupedExperts`` through the Engine's primary-input convention."""

    def __init__(self, experts: GroupedExperts) -> None:
        super().__init__()
        self.experts = experts

    def forward(
        self,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
        router_weights: torch.Tensor,
        router_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run the expert layer.

        Args:
            input_ids: Tensor of shape [tokens, hidden] containing expert inputs.
            token_mask: Boolean tensor of shape [tokens] selecting active tokens.
            router_weights: Tensor of shape [tokens, top_k] containing route weights.
            router_indices: Integer tensor of shape [tokens, top_k] containing expert IDs.

        Returns:
            Tensor of shape [tokens, hidden] containing the combined expert outputs.
        """
        return self.experts(input_ids, token_mask, router_weights, router_indices)


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _tiny_moe_config() -> MoEConfig:
    return MoEConfig(
        n_routed_experts=_N_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=_TOP_K,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=_DIM,
        inter_dim=_MOE_INTER_DIM,
        moe_inter_dim=_MOE_INTER_DIM,
        norm_topk_prob=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.float32,
    )


def _global_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic global batch; under TP every rank sees the full batch."""
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(_NUM_TOKENS, _DIM, generator=generator)
    router_logits = torch.randn(_NUM_TOKENS, _N_EXPERTS, generator=generator)
    weights, indices = router_logits.softmax(dim=-1).topk(_TOP_K, dim=-1)
    token_mask = torch.ones(_NUM_TOKENS, dtype=torch.bool)
    return x, weights, indices, token_mask


def _loss_weights() -> torch.Tensor:
    """Return deterministic nonuniform positive weights of shape [tokens, hidden]."""
    return torch.linspace(0.25, 1.25, steps=_NUM_TOKENS * _DIM).reshape(_NUM_TOKENS, _DIM)


def _build_experts(config: MoEConfig) -> GroupedExperts:
    generator = torch.Generator().manual_seed(4321)
    experts = GroupedExperts(config)
    with torch.no_grad():
        experts.gate_and_up_projs.copy_(torch.randn(experts.gate_and_up_projs.shape, generator=generator) * 0.05)
        experts.down_projs.copy_(torch.randn(experts.down_projs.shape, generator=generator) * 0.05)
    return experts


def _reference_forward_backward() -> tuple[GroupedExperts, torch.Tensor, torch.Tensor]:
    """Single-process (tp=1, ep=1) fp32 forward/backward as ground truth."""
    experts = _build_experts(_tiny_moe_config())
    x, weights, indices, token_mask = _global_inputs()
    y = experts(x, token_mask, weights, indices)
    loss_weights = _loss_weights()
    loss = (y.square() * loss_weights).sum() / loss_weights.sum()
    loss.backward()
    assert experts.gate_and_up_projs.grad is not None
    assert experts.down_projs.grad is not None
    return experts, y.detach(), loss.detach()


def _ep_tp_grad_parity_worker(rank: int, world_size: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        reference_experts, y_ref, loss_ref = _reference_forward_backward()
        gate_up_grad_ref = reference_experts.gate_and_up_projs.grad.detach().clone()
        down_grad_ref = reference_experts.down_projs.grad.detach().clone()
        reference_grad_norm = torch.linalg.vector_norm(
            torch.cat((gate_up_grad_ref.reshape(-1), down_grad_ref.reshape(-1))).to(torch.float64)
        )

        # Composed TP x EP topology on 2 ranks: the same two ranks form the TP
        # replica group of the token path and the EP group of the experts, as
        # in production where the EP mesh flattens the dp*tp ranks.
        world_mesh = init_device_mesh("cpu", (1, _TP_SIZE), mesh_dim_names=("dp", "tp"))
        ep_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))

        experts = _build_experts(_tiny_moe_config())
        experts.gate_and_up_projs = nn.Parameter(
            distribute_tensor(experts.gate_and_up_projs.detach(), ep_mesh, [Shard(0)])
        )
        experts.down_projs = nn.Parameter(distribute_tensor(experts.down_projs.detach(), ep_mesh, [Shard(0)]))
        # parallelize_model sets this marker on the model whenever the custom-MoE
        # TP path is active; get_expert_tp_replication_factor keys off it.
        experts._nemo_moe_tp_requires_replica_sync = True

        # The Engine sees the TP-replicated token path and the EP experts as one
        # ordinary forward/backward boundary.
        x, weights, indices, token_mask = _global_inputs()
        loss_weights = _loss_weights()
        datum = Datum(
            model_inputs={
                "input_ids": x,
                "token_mask": token_mask,
                "router_weights": weights,
                "router_indices": indices,
            },
            loss_fn_inputs={"weights": loss_weights},
        )
        observed: dict[str, torch.Tensor] = {}

        def loss_fn(output: torch.Tensor, loss_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            """Return squared expert outputs in the Engine loss-weight layout.

            Args:
                output: Tensor of shape [tokens, hidden] containing expert outputs.
                loss_inputs: Mapping whose ``weights`` tensor has shape [tokens, hidden].

            Returns:
                Tensor of shape [tokens, hidden] containing per-element squared loss.
            """
            observed["output"] = output.detach()
            return output.square()

        model = _ExpertModel(experts)
        model._nemo_moe_tp_requires_replica_sync = True
        optimizer = torch.optim.SGD(model.parameters(), lr=_LEARNING_RATE)
        engine = Engine(
            model,
            device="cpu",
            mesh_context=MeshContext.from_meshes(world_mesh, ep_mesh),
            collate_fn=collate_prebatched,
            optimizers=optimizer,
            max_grad_norm=1e6,
        )
        engine_loss, _ = engine.forward_backward([datum], loss_fn)
        torch.testing.assert_close(observed["output"], y_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(engine_loss, loss_ref.to(torch.float64), rtol=1e-5, atol=1e-7)

        n_local_experts = _N_EXPERTS // world_size
        start = rank * n_local_experts
        end = start + n_local_experts
        gate_up_grad_ref_local = gate_up_grad_ref[start:end]
        down_grad_ref_local = down_grad_ref[start:end]
        assert gate_up_grad_ref_local.abs().sum() > 0
        assert down_grad_ref_local.abs().sum() > 0

        # Each expert gradient accumulated tp_size identical token replicas
        # through the EP all-gather.
        torch.testing.assert_close(
            experts.gate_and_up_projs.grad.to_local(), _TP_SIZE * gate_up_grad_ref_local, rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(
            experts.down_projs.grad.to_local(), _TP_SIZE * down_grad_ref_local, rtol=1e-4, atol=1e-5
        )

        # Engine.optim_step must remove exactly that factor before any parameter
        # mutation. Capture the corrected gradients at its staging-fence hook;
        # the method then performs the SGD update and clears all gradients.
        corrected_grads: dict[str, torch.Tensor] = {}

        def capture_corrected_grads() -> None:
            corrected_grads["gate_up"] = experts.gate_and_up_projs.grad.to_local().detach().clone()
            corrected_grads["down"] = experts.down_projs.grad.to_local().detach().clone()

        step_result = engine.optim_step(before_optimizer_step=capture_corrected_grads)
        torch.testing.assert_close(corrected_grads["gate_up"], gate_up_grad_ref_local, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(corrected_grads["down"], down_grad_ref_local, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(step_result.grad_norm, reference_grad_norm, rtol=1e-5, atol=1e-7)
        assert step_result.learning_rates == (_LEARNING_RATE,)
        assert experts.gate_and_up_projs.grad is None
        assert experts.down_projs.grad is None

        torch.optim.SGD(reference_experts.parameters(), lr=_LEARNING_RATE).step()
        torch.testing.assert_close(
            experts.gate_and_up_projs.to_local(),
            reference_experts.gate_and_up_projs[start:end],
            rtol=1e-4,
            atol=1e-5,
        )
        torch.testing.assert_close(
            experts.down_projs.to_local(),
            reference_experts.down_projs[start:end],
            rtol=1e-4,
            atol=1e-5,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_engine_tp_replicated_tokens_through_ep_matches_reference_after_finalization():
    mp.spawn(_ep_tp_grad_parity_worker, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)
