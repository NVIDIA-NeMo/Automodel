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

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules, patch_moe_module
from nemo_automodel.components.models.common import BackendConfig, MoKBackendConfig
from nemo_automodel.components.moe import mok_experts
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts
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


def test_mok_ops_import_is_lazy_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    ops = object()
    imports: list[str] = []

    def fake_safe_import(path: str, **kwargs: object) -> tuple[bool, object]:
        del kwargs
        imports.append(path)
        return True, ops

    monkeypatch.setattr(mok_experts, "_mok_ops", None)
    monkeypatch.setattr(mok_experts, "safe_import", fake_safe_import)

    assert mok_experts._mok_ops is None
    assert mok_experts._load_mok_ops() is ops
    assert mok_experts._load_mok_ops() is ops
    assert imports == ["mok.ops"]


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
        precision="mxfp8",
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


def test_mok_backend_config_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="mok.precision must be 'bf16' or 'mxfp8'; got 'fp8'"):
        MoKBackendConfig(precision="fp8")


def test_mok_accepts_dsv4_clamped_swiglu() -> None:
    experts = GroupedExpertsMoK(
        _valid_moe_config(swiglu_limit=10.0),
        BackendConfig(dispatcher="mok"),
    )

    assert experts.runtime.swiglu_limit == 10.0


@pytest.mark.parametrize(("configured_limit", "functional_limit"), [(0.0, None), (10.0, 10.0)])
def test_mok_passes_swiglu_limit_to_functional_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch,
    configured_limit: float,
    functional_limit: float | None,
) -> None:
    calls: dict[str, object] = {}

    class FakeFunctional:
        @staticmethod
        def get_workspace(*args: object, **kwargs: object) -> object:
            del args, kwargs
            return object()

        @staticmethod
        def build_schedule(*args: object, **kwargs: object) -> object:
            del args, kwargs
            return object()

        @staticmethod
        def forward(*args: object) -> tuple[torch.Tensor, object]:
            calls["forward_limit"] = args[-1]
            return args[3], object()

        @staticmethod
        def backward(*args: object) -> tuple[object, ...]:
            calls["backward_limit"] = args[-1]
            return ()

    monkeypatch.setattr(mok_experts, "_mok_functional", FakeFunctional)
    runtime = mok_experts._MoKRuntime(MoKBackendConfig(), swiglu_limit=configured_limit)
    runtime.config = object()
    runtime.ep_group = object()
    x = torch.empty(4, 256, dtype=torch.bfloat16)
    router_weights = torch.empty(4, 2, dtype=torch.float32)
    top_experts = torch.empty(4, 2, dtype=torch.int64)
    shared_gate = torch.empty(256, 256, dtype=torch.bfloat16)
    shared_up = torch.empty(256, 256, dtype=torch.bfloat16)
    shared_down = torch.empty(256, 256, dtype=torch.bfloat16)
    routed_gate = torch.empty(1, 256, 256, dtype=torch.bfloat16)
    routed_up = torch.empty(1, 256, 256, dtype=torch.bfloat16)
    routed_down = torch.empty(1, 256, 256, dtype=torch.bfloat16)

    _, schedule, forward_context, mxfp8_weights = runtime.forward(
        x,
        router_weights,
        top_experts,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate,
        routed_up,
        routed_down,
    )
    runtime.backward(
        schedule,
        forward_context,
        x,
        x,
        router_weights,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate,
        routed_up,
        routed_down,
        mxfp8_weights,
    )

    assert calls == {"forward_limit": functional_limit, "backward_limit": functional_limit}


@pytest.mark.parametrize("retain_until_optimizer_step", [False, True])
def test_mok_mxfp8_quantizes_only_routed_weights_with_required_layouts(
    monkeypatch: pytest.MonkeyPatch,
    retain_until_optimizer_step: bool,
) -> None:
    functional_calls: dict[str, tuple[object, ...]] = {}
    quantization_calls: list[tuple[torch.Tensor, bool, bool]] = []
    quantization_results: list[tuple[torch.Tensor | None, ...]] = []

    class FakeFunctional:
        @staticmethod
        def get_workspace(*args: object, **kwargs: object) -> object:
            del args, kwargs
            return object()

        @staticmethod
        def build_schedule(*args: object, **kwargs: object) -> object:
            del args, kwargs
            return object()

        @staticmethod
        def forward(*args: object) -> tuple[torch.Tensor, object]:
            functional_calls["forward"] = args
            return args[3], object()

        @staticmethod
        def backward(*args: object) -> tuple[object, ...]:
            functional_calls["backward"] = args
            return ()

    class FakeOps:
        @staticmethod
        def mxfp8_quantize(
            weight: torch.Tensor,
            return_normal: bool,
            return_transposed: bool,
        ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            quantization_calls.append((weight, return_normal, return_transposed))
            call_id = len(quantization_calls)
            normal = weight.clone() if return_normal else None
            normal_scale = torch.tensor([call_id], dtype=torch.uint8) if return_normal else None
            transposed = weight.transpose(-1, -2).contiguous() if return_transposed else None
            transposed_scale = torch.tensor([call_id + 10], dtype=torch.uint8) if return_transposed else None
            result = (normal, normal_scale, transposed, transposed_scale)
            quantization_results.append(result)
            return result

    monkeypatch.setattr(mok_experts, "_mok_functional", FakeFunctional)
    monkeypatch.setattr(mok_experts, "_mok_ops", FakeOps)
    runtime = mok_experts._MoKRuntime(MoKBackendConfig(precision="mxfp8"), swiglu_limit=0.0)
    runtime._retain_mxfp8_cache_until_optimizer_step = retain_until_optimizer_step
    runtime.config = object()
    runtime.ep_group = object()
    x = torch.empty(4, 256, dtype=torch.bfloat16)
    router_weights = torch.empty(4, 2, dtype=torch.float32)
    top_experts = torch.empty(4, 2, dtype=torch.int64)
    shared_gate = torch.empty(256, 256, dtype=torch.bfloat16)
    shared_up = torch.empty(256, 256, dtype=torch.bfloat16)
    shared_down = torch.empty(256, 256, dtype=torch.bfloat16)
    routed_gate = torch.empty(1, 256, 256, dtype=torch.bfloat16)
    routed_up = torch.empty(1, 256, 256, dtype=torch.bfloat16)
    routed_down = torch.empty(1, 256, 256, dtype=torch.bfloat16)

    _, schedule, forward_context, mxfp8_weights = runtime.forward(
        x,
        router_weights,
        top_experts,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate,
        routed_up,
        routed_down,
    )
    _, _, _, recompute_mxfp8_weights = runtime.forward(
        x,
        router_weights,
        top_experts,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate,
        routed_up,
        routed_down,
    )
    runtime.backward(
        schedule,
        forward_context,
        x,
        x,
        router_weights,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate,
        routed_up,
        routed_down,
        recompute_mxfp8_weights,
    )

    expected_layouts = (
        (True, True),
        (True, True),
        (True, True),
    )
    for (actual_weight, *actual_layout), expected_weight, expected_layout in zip(
        quantization_calls,
        (routed_gate, routed_up, routed_down),
        expected_layouts,
        strict=True,
    ):
        assert actual_weight is expected_weight
        assert tuple(actual_layout) == expected_layout
    forward_args = functional_calls["forward"]
    assert all(
        actual is expected
        for actual, expected in zip(forward_args[5:8], (shared_gate, shared_up, shared_down), strict=True)
    )
    for actual, quantized in zip(mxfp8_weights, quantization_results, strict=True):
        assert all(actual_item is expected_item for actual_item, expected_item in zip(actual, quantized, strict=True))
    for actual, expected in zip(forward_args[8:11], mxfp8_weights, strict=True):
        assert all(
            actual_item is expected_item for actual_item, expected_item in zip(actual, expected[:2], strict=True)
        )
    assert len(mxfp8_weights) == 3
    assert len(recompute_mxfp8_weights) == 3
    assert all(actual is expected for actual, expected in zip(recompute_mxfp8_weights, mxfp8_weights, strict=True))
    backward_args = functional_calls["backward"]
    assert all(
        actual is expected
        for actual, expected in zip(backward_args[7:10], (shared_gate, shared_up, shared_down), strict=True)
    )
    for actual, expected in (
        (backward_args[10], quantization_results[0]),
        (backward_args[11], quantization_results[1]),
        (backward_args[12], quantization_results[2][2:]),
    ):
        assert all(actual_item is expected_item for actual_item, expected_item in zip(actual, expected, strict=True))
    if retain_until_optimizer_step:
        assert runtime._mxfp8_weights is mxfp8_weights
        runtime._invalidate_mxfp8_cache()
    assert runtime._mxfp8_weights is None


def test_mok_mxfp8_optimizer_post_hook_refreshes_cached_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    quantization_calls: list[torch.Tensor] = []

    def fake_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        quantization_calls.append(weight)
        return weight, weight, weight, weight

    monkeypatch.setattr(mok_experts, "_mxfp8_weight_both", fake_quantize)
    experts = GroupedExpertsMoK(
        _valid_moe_config(),
        BackendConfig(dispatcher="mok", mok={"precision": "mxfp8"}),
    )
    optimizer = torch.optim.AdamW(experts.parameters(), lr=1.0e-3, foreach=False)
    mok_experts.enable_mok_mxfp8_optimizer_step_cache([experts], [optimizer])
    routed_weights = (
        experts.routed_gate_weights,
        experts.routed_up_weights,
        experts.routed_down_weights,
    )

    first = experts.runtime._get_mxfp8_weights(*routed_weights)
    assert experts.runtime._get_mxfp8_weights(*routed_weights) is first
    assert len(quantization_calls) == 3
    generation = experts.runtime._mxfp8_cache_generation

    sum(parameter.float().sum() for parameter in experts.parameters()).backward()
    optimizer.step()

    assert experts.runtime._mxfp8_cache_generation == generation + 1
    assert experts.runtime._mxfp8_weights is None
    refreshed = experts.runtime._get_mxfp8_weights(*routed_weights)
    assert refreshed is not first
    assert len(quantization_calls) == 6


def test_mok_mxfp8_cache_requires_one_optimizer_per_model_part() -> None:
    experts = GroupedExpertsMoK(
        _valid_moe_config(),
        BackendConfig(dispatcher="mok", mok={"precision": "mxfp8"}),
    )

    with pytest.raises(ValueError, match="one optimizer per model part"):
        mok_experts.enable_mok_mxfp8_optimizer_step_cache([experts], [])


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
        ({"swiglu_limit": -1.0}, "swiglu_limit=-1.0"),
        ({"swiglu_limit": float("nan")}, "swiglu_limit=nan"),
    ],
)
def test_mok_rejects_unsupported_moe_contract(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GroupedExpertsMoK(_valid_moe_config(**overrides), BackendConfig(dispatcher="mok"))


def test_mok_state_dict_preserves_combined_expert_layout() -> None:
    config = _valid_moe_config()
    backend = BackendConfig(dispatcher="mok", mok={"precision": "mxfp8"})
    source = GroupedExpertsMoK(config, backend)
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

    restored = GroupedExpertsMoK(config, backend)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.routed_gate_weights, source.routed_gate_weights)
    torch.testing.assert_close(restored.routed_up_weights, source.routed_up_weights)
    torch.testing.assert_close(restored.routed_down_weights, source.routed_down_weights)


def test_mok_random_init_matches_canonical_grouped_expert_layout() -> None:
    """A dispatcher switch must not change random-init model weights."""
    config = _valid_moe_config()

    torch.manual_seed(1234)
    reference = GroupedExperts(config)
    with torch.no_grad():
        reference.init_weights(torch.device("cpu"))

    torch.manual_seed(1234)
    mok = GroupedExpertsMoK(config, BackendConfig(dispatcher="mok"))
    with torch.no_grad():
        mok.init_weights(torch.device("cpu"))

    mok_state = mok.state_dict()
    torch.testing.assert_close(mok_state["gate_and_up_projs"], reference.gate_and_up_projs, rtol=0, atol=0)
    torch.testing.assert_close(mok_state["down_projs"], reference.down_projs, rtol=0, atol=0)


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


def test_mok_rejects_lora_patching() -> None:
    experts = GroupedExpertsMoK(_valid_moe_config(), BackendConfig(dispatcher="mok"))

    with pytest.raises(NotImplementedError, match="LoRA is not supported for Mixture-of-Kittens"):
        patch_moe_module(experts)

    model = torch.nn.Module()
    model.experts = experts
    with pytest.raises(NotImplementedError, match="LoRA is not supported for Mixture-of-Kittens"):
        apply_lora_to_linear_modules(model, PeftConfig(target_modules=["experts"]))


@pytest.mark.parametrize("world_size", [1, 2, 3, 5, 6])
def test_mok_requires_world_size_divisible_by_four(monkeypatch: pytest.MonkeyPatch, world_size: int) -> None:
    monkeypatch.setattr("nemo_automodel.components.moe.layers.get_world_size_safe", lambda: world_size)

    with pytest.raises(ValueError, match=rf"world size to be divisible by 4; got {world_size}"):
        MoE(_valid_moe_config(), BackendConfig(dispatcher="mok", linear="torch"))


def test_mok_dispatches_pre_aligned_extent_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _valid_moe_config()
    monkeypatch.setattr("nemo_automodel.components.moe.layers.get_world_size_safe", lambda: 4)
    moe = MoE(config, BackendConfig(dispatcher="mok", linear="torch"))
    captured: dict[str, torch.Tensor] = {}

    def fake_experts(
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *shared_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Capture expert inputs and return an identity-like output.

        Args:
            x: BF16 tensor of shape [tokens, hidden].
            weights: Tensor of shape [tokens, activated_experts].
            indices: Int64 tensor of shape [tokens, activated_experts].
            *shared_weights: Shared-expert tensors in their projection layouts.

        Returns:
            Tensor of shape [tokens, hidden].
        """
        del shared_weights
        captured["x"] = x
        captured["weights"] = weights
        captured["indices"] = indices
        return x + 1

    monkeypatch.setattr(moe.experts, "forward", fake_experts)
    x = torch.randn(1, 512, 256, dtype=torch.bfloat16, requires_grad=True)

    output = moe(x)

    assert output.shape == x.shape
    assert captured["x"].shape == (512, 256)
    torch.testing.assert_close(captured["x"], x.detach().view(-1, 256))
    torch.testing.assert_close(output, x + 1)

    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x.grad))


def test_mok_delegates_invalid_extent_without_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _valid_moe_config()
    monkeypatch.setattr("nemo_automodel.components.moe.layers.get_world_size_safe", lambda: 4)
    moe = MoE(config, BackendConfig(dispatcher="mok", linear="torch"))

    def fake_experts(
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *shared_weights: torch.Tensor,
    ) -> torch.Tensor:
        del weights, indices, shared_weights
        if x.size(0) % 256 != 0:
            raise ValueError("num_local_tokens must be divisible by 256")
        return x

    monkeypatch.setattr(moe.experts, "forward", fake_experts)
    x = torch.randn(1, 513, 256, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="num_local_tokens must be divisible by 256"):
        moe(x)


def test_mok_packed_thd_dispatches_padding_as_physical_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _valid_moe_config()
    monkeypatch.setattr("nemo_automodel.components.moe.layers.get_world_size_safe", lambda: 4)
    moe = MoE(config, BackendConfig(dispatcher="mok", linear="torch"))
    captured: dict[str, torch.Tensor] = {}
    original_gate_forward = moe.gate.forward

    def capture_gate(
        x: torch.Tensor, token_mask: torch.Tensor, cp_mesh: object
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        captured["gate_x"] = x
        captured["gate_token_mask"] = token_mask
        weights, indices, aux_loss = original_gate_forward(x, token_mask, cp_mesh)
        captured["gate_weights"] = weights
        captured["gate_indices"] = indices
        return weights, indices, aux_loss

    def fake_experts(
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *shared_weights: torch.Tensor,
    ) -> torch.Tensor:
        del shared_weights
        captured["x"] = x
        captured["weights"] = weights
        captured["indices"] = indices
        return x + 1

    monkeypatch.setattr(moe.gate, "forward", capture_gate)
    monkeypatch.setattr(moe.experts, "forward", fake_experts)
    x = torch.randn(512, 256, dtype=torch.bfloat16, requires_grad=True)
    padding_mask = torch.zeros(512, dtype=torch.bool)
    padding_mask[-2:] = True

    output = moe(x, padding_mask=padding_mask)

    assert output.shape == x.shape
    torch.testing.assert_close(captured["gate_x"], x.detach())
    torch.testing.assert_close(captured["gate_token_mask"], ~padding_mask)
    torch.testing.assert_close(captured["x"], x.detach())
    assert captured["weights"] is captured["gate_weights"]
    assert captured["indices"] is captured["gate_indices"]
    torch.testing.assert_close(output, x + 1)

    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x.grad))


def test_mok_packed_thd_accepts_padding_between_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _valid_moe_config()
    monkeypatch.setattr("nemo_automodel.components.moe.layers.get_world_size_safe", lambda: 4)
    moe = MoE(config, BackendConfig(dispatcher="mok", linear="torch"))
    captured: dict[str, torch.Tensor] = {}
    original_gate_forward = moe.gate.forward

    def capture_gate(
        x: torch.Tensor, token_mask: torch.Tensor, cp_mesh: object
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        captured["token_mask"] = token_mask
        return original_gate_forward(x, token_mask, cp_mesh)

    def fake_experts(
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *shared_weights: torch.Tensor,
    ) -> torch.Tensor:
        del weights, indices, shared_weights
        return x

    monkeypatch.setattr(moe.gate, "forward", capture_gate)
    monkeypatch.setattr(moe.experts, "forward", fake_experts)
    x = torch.randn(512, 256, dtype=torch.bfloat16)
    padding_mask = torch.zeros(512, dtype=torch.bool)
    padding_mask[[1, 511]] = True

    output = moe(x, padding_mask=padding_mask)

    torch.testing.assert_close(captured["token_mask"], ~padding_mask)
    torch.testing.assert_close(output, x)


def test_mok_all_padding_still_enters_collective_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _valid_moe_config()
    monkeypatch.setattr("nemo_automodel.components.moe.layers.get_world_size_safe", lambda: 4)
    moe = MoE(config, BackendConfig(dispatcher="mok", linear="torch"))
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    gate_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    gate_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_gate_forward = moe.gate.forward

    def capture_gate(
        x: torch.Tensor, token_mask: torch.Tensor, cp_mesh: object
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        gate_calls.append((x, token_mask))
        weights, indices, aux_loss = original_gate_forward(x, token_mask, cp_mesh)
        gate_outputs.append((weights, indices))
        return weights, indices, aux_loss

    def fake_experts(
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *shared_weights: torch.Tensor,
    ) -> torch.Tensor:
        del shared_weights
        calls.append((x, weights, indices))
        return torch.zeros_like(x)

    monkeypatch.setattr(moe.gate, "forward", capture_gate)
    monkeypatch.setattr(moe.experts, "forward", fake_experts)
    x = torch.randn(512, 256, dtype=torch.bfloat16)

    output = moe(x, padding_mask=torch.ones(512, dtype=torch.bool))

    assert len(gate_calls) == 1
    gate_x, gate_token_mask = gate_calls[0]
    torch.testing.assert_close(gate_x, x)
    assert not gate_token_mask.any()
    assert len(calls) == 1
    expert_x, weights, indices = calls[0]
    torch.testing.assert_close(expert_x, x)
    assert weights is gate_outputs[0][0]
    assert indices is gate_outputs[0][1]
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_mok_manual_backward_maps_gradients_to_autograd_inputs() -> None:
    @dataclass(frozen=True)
    class FakeSchedule:
        peer_rank: torch.Tensor

    @dataclass(frozen=True)
    class FakeForwardContext:
        hidden: torch.Tensor

    class FakeRuntime:
        def forward(self, x: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, object, object, None]:
            del args
            return x.clone(), FakeSchedule(torch.zeros(1, dtype=torch.int32)), FakeForwardContext(x.clone()), None

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
            mxfp8_forward_weights: None,
        ) -> tuple[torch.Tensor, ...]:
            del schedule, forward_context, grad_output
            assert mxfp8_forward_weights is None
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


def test_mok_mxfp8_autograd_restores_saved_weight_layouts() -> None:
    @dataclass(frozen=True)
    class FakeSchedule:
        peer_rank: torch.Tensor

    @dataclass(frozen=True)
    class FakeForwardContext:
        hidden: torch.Tensor

    routed_layouts = tuple(
        tuple(torch.full((1,), projection * 4 + layout) for layout in range(4)) for projection in range(3)
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.backward_layouts: tuple[tuple[torch.Tensor, ...], ...] | None = None

        def forward(
            self, x: torch.Tensor, *args: torch.Tensor
        ) -> tuple[torch.Tensor, object, object, tuple[tuple[torch.Tensor, ...], ...]]:
            del args
            return (
                x.clone(),
                FakeSchedule(torch.zeros(1, dtype=torch.int32)),
                FakeForwardContext(x.clone()),
                routed_layouts,
            )

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
            mxfp8_weights: tuple[tuple[torch.Tensor, ...], ...],
        ) -> tuple[torch.Tensor, ...]:
            del schedule, forward_context, grad_output
            self.backward_layouts = mxfp8_weights
            return (
                torch.ones_like(x),
                torch.ones_like(router_weights),
                torch.ones_like(routed_gate_weights),
                torch.ones_like(routed_up_weights),
                torch.ones_like(routed_down_weights),
                torch.ones_like(shared_gate_weights),
                torch.ones_like(shared_up_weights),
                torch.ones_like(shared_down_weights),
            )

    runtime = FakeRuntime()
    tensors = [torch.randn(2, 2, requires_grad=True) for _ in range(8)]
    output = mok_experts._MoKAutogradFunction.apply(
        runtime,
        tensors[0],
        tensors[1],
        torch.zeros(2, 2, dtype=torch.int64),
        tensors[2],
        tensors[3],
        tensors[4],
        tensors[5],
        tensors[6],
        tensors[7],
    )
    output.sum().backward()

    assert runtime.backward_layouts is not None
    for actual_projection, expected_projection in zip(runtime.backward_layouts, routed_layouts, strict=True):
        for actual, expected in zip(actual_projection, expected_projection, strict=True):
            torch.testing.assert_close(actual, expected)
    assert all(tensor.grad is not None for tensor in tensors)


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

        def forward(self, x: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, object, object, None]:
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
            return x.clone(), schedule, forward_context, None

        def backward(
            self,
            schedule: object,
            forward_context: object,
            grad_output: torch.Tensor,
            *inputs: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            self.backward_state = (schedule, forward_context)
            assert inputs[-1] is None
            inputs = inputs[:-1]
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
