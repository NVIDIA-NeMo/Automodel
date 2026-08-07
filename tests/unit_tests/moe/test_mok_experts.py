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

import gc
import weakref
from dataclasses import dataclass

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig, MoKBackendConfig
from nemo_automodel.components.moe import mok_experts
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.moe.mok_experts import GroupedExpertsMoK


def _valid_moe_config(**overrides: object) -> MoEConfig:
    values = {
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
        "n_expert_groups": 1,
        "n_limited_groups": 1,
        "train_gate": True,
        "gate_bias_update_factor": 0.0,
        "aux_loss_coeff": 0.0,
        "score_func": "softmax",
        "route_scale": 1.0,
        "dim": 256,
        "inter_dim": 512,
        "moe_inter_dim": 256,
        "norm_topk_prob": True,
        "dtype": torch.bfloat16,
    }
    values.update(overrides)
    return MoEConfig(**values)


def test_mok_functional_import_is_lazy_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    functional = object()
    imports: list[str] = []

    def fake_safe_import(path: str, **kwargs: object) -> tuple[bool, object]:
        del kwargs
        imports.append(path)
        return True, functional

    monkeypatch.setattr(mok_experts, "_mok_functional", None)
    monkeypatch.setattr(mok_experts, "safe_import", fake_safe_import)

    assert mok_experts._mok_functional is None
    assert mok_experts._load_mok_functional() is functional
    assert mok_experts._load_mok_functional() is functional
    assert imports == ["mok.functional"]


def test_mok_backend_config_build_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeFunctional:
        class MoKConfig:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

    monkeypatch.setattr(
        "nemo_automodel.components.models.common.utils.safe_import",
        lambda *args, **kwargs: (True, FakeFunctional),
    )
    config = MoKBackendConfig(
        fwd_num_comm_sms=24,
        bwd_num_comm_sms=20,
        minibatch_size=2048,
        macrobatch_size=8192,
        schedule_capacity_multiplier=0.75,
        all_gather_top_experts_chunk_bytes=1024,
    )

    config.build()

    assert captured == {
        "fwd_num_comm_sms": 24,
        "bwd_num_comm_sms": 20,
        "minibatch_size": 2048,
        "macrobatch_size": 8192,
        "schedule_capacity_multiplier": 0.75,
        "all_gather_top_experts_chunk_bytes": 1024,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_shared_experts": 0}, "n_shared_experts=0"),
        ({"dim": 384}, "dim=384"),
        ({"moe_inter_dim": 384}, "moe_inter_dim=384"),
        ({"expert_bias": True}, "expert_bias=True"),
        ({"shared_expert_gate": True}, "shared_expert_gate=True"),
        ({"moe_latent_size": 128}, "moe_latent_size=128"),
        ({"expert_activation": "relu2"}, "activations must both be swiglu"),
    ],
)
def test_mok_rejects_unsupported_moe_contract(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GroupedExpertsMoK(_valid_moe_config(**overrides), BackendConfig(dispatcher="mok"))


def test_mok_state_dict_preserves_combined_expert_layout() -> None:
    config = _valid_moe_config()
    source = GroupedExpertsMoK(config, BackendConfig(dispatcher="mok"))
    with torch.no_grad():
        source.routed_gate_weights.copy_(
            torch.arange(source.routed_gate_weights.numel(), dtype=torch.float32)
            .reshape_as(source.routed_gate_weights)
            .to(torch.bfloat16)
        )
        source.routed_up_weights.copy_(source.routed_gate_weights + 1)
        source.routed_down_weights.copy_(
            torch.arange(source.routed_down_weights.numel(), dtype=torch.float32)
            .reshape_as(source.routed_down_weights)
            .to(torch.bfloat16)
        )

    state = source.state_dict()

    assert set(state) == {"gate_and_up_projs", "down_projs"}
    assert state["gate_and_up_projs"].shape == (4, 256, 512)
    assert state["down_projs"].shape == (4, 256, 256)
    torch.testing.assert_close(state["gate_and_up_projs"][..., :256], source.routed_gate_weights.transpose(-1, -2))
    torch.testing.assert_close(state["gate_and_up_projs"][..., 256:], source.routed_up_weights.transpose(-1, -2))
    torch.testing.assert_close(state["down_projs"], source.routed_down_weights.transpose(-1, -2))

    restored = GroupedExpertsMoK(config, BackendConfig(dispatcher="mok"))
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.routed_gate_weights, source.routed_gate_weights)
    torch.testing.assert_close(restored.routed_up_weights, source.routed_up_weights)
    torch.testing.assert_close(restored.routed_down_weights, source.routed_down_weights)


def test_mok_state_dict_loads_virtual_expert_keys_independently() -> None:
    config = _valid_moe_config()
    source = GroupedExpertsMoK(config, BackendConfig(dispatcher="mok"))
    target = GroupedExpertsMoK(config, BackendConfig(dispatcher="mok"))
    with torch.no_grad():
        source.routed_gate_weights.fill_(1)
        source.routed_up_weights.fill_(2)
        source.routed_down_weights.fill_(3)
        target.routed_gate_weights.zero_()
        target.routed_up_weights.zero_()
        target.routed_down_weights.zero_()

    source_state = source.state_dict()
    target.load_state_dict({"gate_and_up_projs": source_state["gate_and_up_projs"]}, strict=False)

    torch.testing.assert_close(target.routed_gate_weights, source.routed_gate_weights)
    torch.testing.assert_close(target.routed_up_weights, source.routed_up_weights)
    torch.testing.assert_close(target.routed_down_weights, torch.zeros_like(target.routed_down_weights))

    target.load_state_dict({"down_projs": source_state["down_projs"]}, strict=False)

    torch.testing.assert_close(target.routed_down_weights, source.routed_down_weights)


def test_mok_compacts_padding_and_restores_token_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _valid_moe_config()
    monkeypatch.setattr("nemo_automodel.components.moe.layers.get_world_size_safe", lambda: 4)
    moe = MoE(config, BackendConfig(dispatcher="mok", linear="torch"))
    captured: dict[str, torch.Tensor] = {}

    def fake_experts(
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *shared_weights: torch.Tensor,
    ) -> torch.Tensor:
        del shared_weights
        captured["x"] = x
        captured["token_mask"] = token_mask
        captured["weights"] = weights
        captured["indices"] = indices
        return x + 1

    monkeypatch.setattr(moe.experts, "forward", fake_experts)
    x = torch.randn(1, 4, 256, dtype=torch.bfloat16, requires_grad=True)
    padding_mask = torch.tensor([[False, True, False, True]])

    output = moe(x, padding_mask=padding_mask)

    assert output.shape == x.shape
    assert captured["x"].shape == (512, 256)
    torch.testing.assert_close(captured["x"][:2], x.detach().view(-1, 256)[[0, 2]])
    torch.testing.assert_close(captured["x"][2:], torch.zeros_like(captured["x"][2:]))
    assert captured["token_mask"].all()
    assert captured["token_mask"].numel() == 512
    torch.testing.assert_close(captured["weights"][2:], torch.zeros_like(captured["weights"][2:]))
    dummy_load = torch.bincount(captured["indices"][2:].flatten(), minlength=config.n_routed_experts)
    torch.testing.assert_close(dummy_load, torch.full_like(dummy_load, dummy_load[0]))
    torch.testing.assert_close(output[0, 0], x[0, 0] + 1)
    torch.testing.assert_close(output[0, 2], x[0, 2] + 1)
    torch.testing.assert_close(output[0, 1], torch.zeros_like(output[0, 1]))
    torch.testing.assert_close(output[0, 3], torch.zeros_like(output[0, 3]))

    output.sum().backward()
    torch.testing.assert_close(x.grad[0, 0], torch.ones_like(x.grad[0, 0]))
    torch.testing.assert_close(x.grad[0, 2], torch.ones_like(x.grad[0, 2]))
    torch.testing.assert_close(x.grad[0, 1], torch.zeros_like(x.grad[0, 1]))
    torch.testing.assert_close(x.grad[0, 3], torch.zeros_like(x.grad[0, 3]))


def test_mok_all_padding_still_enters_collective_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _valid_moe_config()
    monkeypatch.setattr("nemo_automodel.components.moe.layers.get_world_size_safe", lambda: 4)
    moe = MoE(config, BackendConfig(dispatcher="mok", linear="torch"))
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def fake_experts(
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *shared_weights: torch.Tensor,
    ) -> torch.Tensor:
        del shared_weights
        calls.append((x, weights, indices))
        assert token_mask.all()
        return torch.zeros_like(x)

    monkeypatch.setattr(moe.experts, "forward", fake_experts)
    x = torch.randn(1, 4, 256, dtype=torch.bfloat16)

    output = moe(x, padding_mask=torch.ones(1, 4, dtype=torch.bool))

    assert len(calls) == 1
    expert_x, weights, indices = calls[0]
    assert expert_x.shape == (512, 256)
    torch.testing.assert_close(expert_x, torch.zeros_like(expert_x))
    torch.testing.assert_close(weights, torch.zeros_like(weights))
    dummy_load = torch.bincount(indices.flatten(), minlength=config.n_routed_experts)
    torch.testing.assert_close(dummy_load, torch.full_like(dummy_load, dummy_load[0]))
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_mok_manual_backward_maps_gradients_to_autograd_inputs() -> None:
    @dataclass(frozen=True)
    class FakeSchedule:
        peer_rank: torch.Tensor

    @dataclass(frozen=True)
    class FakeForwardContext:
        hidden: torch.Tensor

    class FakeRuntime:
        def forward(self, x: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, object, object]:
            del args
            return x.clone(), FakeSchedule(torch.zeros(1, dtype=torch.int32)), FakeForwardContext(x.clone())

        def backward(
            self,
            schedule: object,
            forward_context: object,
            grad_output: torch.Tensor,
            x: torch.Tensor,
            router_weights: torch.Tensor,
            shared_gate_weights: torch.Tensor,
            shared_up_weights: torch.Tensor,
            shared_down_weights: torch.Tensor,
            routed_gate_weights: torch.Tensor,
            routed_up_weights: torch.Tensor,
            routed_down_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            del schedule, forward_context, grad_output
            return (
                torch.full_like(x, 1),
                torch.full_like(router_weights, 2),
                torch.full_like(routed_gate_weights, 3),
                torch.full_like(routed_up_weights, 4),
                torch.full_like(routed_down_weights, 5),
                torch.full_like(shared_gate_weights, 6),
                torch.full_like(shared_up_weights, 7),
                torch.full_like(shared_down_weights, 8),
            )

    experts = GroupedExpertsMoK(_valid_moe_config(), BackendConfig(dispatcher="mok"))
    experts.runtime = FakeRuntime()
    x = torch.randn(4, 256, dtype=torch.bfloat16, requires_grad=True)
    router_weights = torch.randn(4, 2, dtype=torch.bfloat16, requires_grad=True)
    top_experts = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    shared_gate = torch.randn(256, 256, dtype=torch.bfloat16, requires_grad=True)
    shared_up = torch.randn(256, 256, dtype=torch.bfloat16, requires_grad=True)
    shared_down = torch.randn(256, 256, dtype=torch.bfloat16, requires_grad=True)

    output = experts(
        x,
        torch.ones(4, dtype=torch.bool),
        router_weights,
        top_experts,
        shared_gate,
        shared_up,
        shared_down,
    )
    output.backward(torch.ones_like(output))

    for actual, expected in (
        (x.grad, 1),
        (router_weights.grad, 2),
        (experts.routed_gate_weights.grad, 3),
        (experts.routed_up_weights.grad, 4),
        (experts.routed_down_weights.grad, 5),
        (shared_gate.grad, 6),
        (shared_up.grad, 7),
        (shared_down.grad, 8),
    ):
        torch.testing.assert_close(actual, torch.full_like(actual, expected))


def test_mok_context_uses_checkpoint_saved_tensor_hooks() -> None:
    @dataclass(frozen=True)
    class FakeSchedule:
        peer_rank: torch.Tensor
        counts: tuple[torch.Tensor, torch.Tensor]

    @dataclass(frozen=True)
    class FakeForwardContext:
        x_routed: torch.Tensor
        hidden_routed: tuple[torch.Tensor, torch.Tensor]

    class FakeRuntime:
        def __init__(self) -> None:
            self.forward_calls = 0
            self.first_context_refs: list[weakref.ReferenceType[torch.Tensor]] = []
            self.backward_state: tuple[object, object] | None = None

        def forward(self, x: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, object, object]:
            del args
            self.forward_calls += 1
            schedule = FakeSchedule(
                peer_rank=torch.zeros(2, dtype=torch.int32),
                counts=(torch.ones(1, dtype=torch.int32), torch.ones(2, dtype=torch.int32)),
            )
            forward_context = FakeForwardContext(
                x_routed=torch.ones(128, 128),
                hidden_routed=(torch.ones(64, 64), torch.ones(32, 32)),
            )
            if self.forward_calls == 1:
                self.first_context_refs = [
                    weakref.ref(forward_context.x_routed),
                    *(weakref.ref(item) for item in forward_context.hidden_routed),
                ]
            return x.clone(), schedule, forward_context

        def backward(
            self,
            schedule: object,
            forward_context: object,
            grad_output: torch.Tensor,
            *inputs: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            self.backward_state = (schedule, forward_context)
            return tuple(torch.ones_like(tensor) for tensor in (inputs[0], inputs[1], *inputs[5:], *inputs[2:5]))

    runtime = FakeRuntime()
    tensors = [torch.randn(2, 2, requires_grad=True) for _ in range(8)]
    top_experts = torch.zeros(2, 2, dtype=torch.int64)

    def run(x: torch.Tensor) -> torch.Tensor:
        return mok_experts._MoKAutogradFunction.apply(
            runtime,
            x,
            tensors[1],
            top_experts,
            tensors[2],
            tensors[3],
            tensors[4],
            tensors[5],
            tensors[6],
            tensors[7],
        )

    output = checkpoint(run, tensors[0], use_reentrant=False)
    gc.collect()

    assert runtime.forward_calls == 1
    assert runtime.first_context_refs
    assert all(ref() is None for ref in runtime.first_context_refs)

    output.sum().backward()

    assert runtime.forward_calls == 2
    assert runtime.backward_state is not None
    schedule, forward_context = runtime.backward_state
    assert isinstance(schedule, FakeSchedule)
    assert isinstance(forward_context, FakeForwardContext)
    assert all(tensor.grad is not None for tensor in tensors)
