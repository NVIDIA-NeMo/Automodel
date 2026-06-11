# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared GatedDeltaNet fp32-compute helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.common.gated_delta_net_fp32 import (
    FP32_GDN_PARAM_NAMES,
    HOLDER_NAME,
    Fp32GateParamHolder,
    force_fp32_gated_delta_net_params,
    is_gated_delta_net_fp32_param_key,
    isolate_fp32_params,
    isolate_gated_delta_net_fp32_params,
    mark_keep_in_fp32_modules_strict,
    route_fp32_holder_key,
    strip_fp32_holder_key,
    upcast_gated_delta_net_fp32_state_tensor,
)


class _FakeGatedDeltaNet(nn.Module):
    """Minimal GatedDeltaNet-like module with bare A_log/dt_bias params."""

    def __init__(self, a_dtype: torch.dtype = torch.float32, dt_dtype: torch.dtype = torch.float32, num_v: int = 4):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(num_v, dtype=a_dtype))
        self.dt_bias = nn.Parameter(torch.ones(num_v, dtype=dt_dtype))
        # A bf16 submodule param (must NOT be moved into the holder).
        self.in_proj = nn.Linear(8, 8, bias=False)


class _FakeSsmMixer(nn.Module):
    """Minimal non-GDN SSM-like module with A_log/dt_bias params."""

    def __init__(self, a_dtype: torch.dtype = torch.bfloat16, dt_dtype: torch.dtype = torch.bfloat16, num_v: int = 4):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(num_v, dtype=a_dtype))
        self.dt_bias = nn.Parameter(torch.ones(num_v, dtype=dt_dtype))


def _wrap_in_model(gdn: nn.Module) -> nn.Module:
    """Build a tiny model with ``layers.0.linear_attn = gdn``."""
    model = nn.Module()
    layer = nn.Module()
    layer.linear_attn = gdn
    model.layers = nn.ModuleList([layer])
    return model


class TestIsolateFp32Params:
    def test_moves_fp32_bare_params_into_holder(self):
        gdn = _FakeGatedDeltaNet(a_dtype=torch.float32, dt_dtype=torch.float32)
        holder = isolate_fp32_params(gdn)

        assert isinstance(holder, Fp32GateParamHolder)
        assert "A_log" not in gdn._parameters
        assert "dt_bias" not in gdn._parameters
        assert HOLDER_NAME in gdn._modules
        # __getattr__ redirect resolves to the holder-owned params.
        assert gdn.A_log is holder._parameters["A_log"]
        assert gdn.dt_bias is holder._parameters["dt_bias"]
        # The bf16 submodule weight stays put.
        assert gdn.in_proj.weight is not None

    def test_forces_bf16_tracked_params_to_fp32_before_isolation(self):
        gdn = _FakeGatedDeltaNet(a_dtype=torch.float32, dt_dtype=torch.bfloat16)
        holder = isolate_fp32_params(gdn)

        assert holder is not None
        assert "A_log" not in gdn._parameters
        assert "dt_bias" not in gdn._parameters
        assert gdn.A_log.dtype == torch.float32
        assert gdn.dt_bias.dtype == torch.float32
        assert holder.A_log.dtype == torch.float32
        assert holder.dt_bias.dtype == torch.float32

    def test_all_bf16_tracked_params_are_forced_and_isolated(self):
        gdn = _FakeGatedDeltaNet(a_dtype=torch.bfloat16, dt_dtype=torch.bfloat16)
        holder = isolate_fp32_params(gdn)
        assert holder is not None
        assert HOLDER_NAME in gdn._modules
        assert "A_log" not in gdn._parameters
        assert "dt_bias" not in gdn._parameters
        assert holder.A_log.dtype == torch.float32
        assert holder.dt_bias.dtype == torch.float32

    def test_force_fp32_gated_delta_net_params_without_isolation(self):
        gdn = _FakeGatedDeltaNet(a_dtype=torch.bfloat16, dt_dtype=torch.bfloat16)
        assert force_fp32_gated_delta_net_params(gdn) is True
        assert HOLDER_NAME not in gdn._modules
        assert gdn.A_log.dtype == torch.float32
        assert gdn.dt_bias.dtype == torch.float32

    def test_idempotent_no_nested_holder(self):
        gdn = _FakeGatedDeltaNet()
        h1 = isolate_fp32_params(gdn)
        h2 = isolate_fp32_params(gdn)
        assert h1 is h2
        # The holder must not get its own nested holder.
        assert HOLDER_NAME not in h1._modules

    def test_getattr_resolves_after_param_replacement(self):
        gdn = _FakeGatedDeltaNet()
        isolate_fp32_params(gdn)
        new_tensor = nn.Parameter(torch.ones(4, dtype=torch.float32))
        gdn._fp32_params._parameters["A_log"] = new_tensor
        assert gdn.A_log is new_tensor

    def test_missing_attr_still_raises(self):
        gdn = _FakeGatedDeltaNet()
        isolate_fp32_params(gdn)
        with pytest.raises(AttributeError):
            _ = gdn.does_not_exist_xyz


class TestHolderForward:
    def test_uses_own_dt_bias_when_present(self):
        holder = Fp32GateParamHolder()
        holder.A_log = nn.Parameter(torch.ones(4, dtype=torch.float32))
        holder.dt_bias = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        a = torch.zeros(4)

        g = holder(a)
        expected = -torch.ones(4).exp() * F.softplus(torch.zeros(4))
        assert torch.allclose(g, expected, atol=1e-5)

        # A dt_bias passed by the caller is ignored when the holder owns dt_bias.
        g_ignored = holder(a, dt_bias=torch.full((4,), 100.0))
        assert torch.allclose(g, g_ignored, atol=1e-5)

    def test_falls_back_to_arg_dt_bias(self):
        holder = Fp32GateParamHolder()
        holder.A_log = nn.Parameter(torch.ones(4, dtype=torch.float32))
        g = holder(torch.zeros(4), dt_bias=torch.zeros(4))
        expected = -torch.ones(4).exp() * F.softplus(torch.zeros(4))
        assert torch.allclose(g, expected, atol=1e-5)

    def test_computes_in_fp32_with_bf16_inputs(self):
        holder = Fp32GateParamHolder()
        holder.A_log = nn.Parameter(torch.ones(4, dtype=torch.float32))
        holder.dt_bias = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        g = holder(torch.zeros(4, dtype=torch.bfloat16))
        assert g.dtype == torch.float32


class TestIsolateAcrossModel:
    def test_isolates_and_marks(self):
        model = _wrap_in_model(_FakeGatedDeltaNet())
        ok = isolate_gated_delta_net_fp32_params(model)
        assert ok is True
        assert model._keep_in_fp32_modules_strict == (HOLDER_NAME,)
        assert HOLDER_NAME in model.layers[0].linear_attn._modules

    def test_idempotent(self):
        model = _wrap_in_model(_FakeGatedDeltaNet())
        assert isolate_gated_delta_net_fp32_params(model) is True
        assert isolate_gated_delta_net_fp32_params(model) is True
        # Still marked exactly once.
        assert model._keep_in_fp32_modules_strict == (HOLDER_NAME,)
        # No nested holder created inside the holder.
        holder = model.layers[0].linear_attn._fp32_params
        assert HOLDER_NAME not in holder._modules

    def test_bf16_gdn_params_are_forced_and_marked(self):
        model = _wrap_in_model(_FakeGatedDeltaNet(a_dtype=torch.bfloat16, dt_dtype=torch.bfloat16))
        assert isolate_gated_delta_net_fp32_params(model) is True
        assert model._keep_in_fp32_modules_strict == (HOLDER_NAME,)
        holder = model.layers[0].linear_attn._fp32_params
        assert holder.A_log.dtype == torch.float32
        assert holder.dt_bias.dtype == torch.float32

    def test_no_gdn_is_noop(self):
        model = nn.Module()
        model.mlp = nn.Linear(4, 4)
        assert isolate_gated_delta_net_fp32_params(model) is False

    def test_non_linear_attn_a_log_module_is_noop(self):
        model = nn.Module()
        model.mixer = _FakeSsmMixer()

        assert isolate_gated_delta_net_fp32_params(model) is False
        assert HOLDER_NAME not in model.mixer._modules
        assert model.mixer.A_log.dtype == torch.bfloat16
        assert model.mixer.dt_bias.dtype == torch.bfloat16
        assert not hasattr(model, "_keep_in_fp32_modules_strict")


class TestMarkKeepInFp32:
    def test_idempotent_and_preserves_existing(self):
        model = nn.Module()
        model._keep_in_fp32_modules_strict = ("foo",)
        mark_keep_in_fp32_modules_strict(model)
        mark_keep_in_fp32_modules_strict(model)
        assert model._keep_in_fp32_modules_strict == ("foo", HOLDER_NAME)


class TestKeyHelpers:
    def test_strip_fp32_holder_key(self):
        assert (
            strip_fp32_holder_key("model.layers.0.linear_attn._fp32_params.A_log") == "model.layers.0.linear_attn.A_log"
        )
        assert (
            strip_fp32_holder_key("model.layers.0.linear_attn._fp32_params.dt_bias")
            == "model.layers.0.linear_attn.dt_bias"
        )
        # Already bare / unrelated keys are unchanged.
        assert strip_fp32_holder_key("model.layers.0.linear_attn.A_log") == "model.layers.0.linear_attn.A_log"
        assert strip_fp32_holder_key("model.layers.0.mlp.gate.weight") == "model.layers.0.mlp.gate.weight"

    def test_route_fp32_holder_key(self):
        for name in FP32_GDN_PARAM_NAMES:
            assert (
                route_fp32_holder_key(f"model.layers.0.linear_attn.{name}")
                == f"model.layers.0.linear_attn._fp32_params.{name}"
            )
        # Already routed -> unchanged.
        assert (
            route_fp32_holder_key("model.layers.0.linear_attn._fp32_params.A_log")
            == "model.layers.0.linear_attn._fp32_params.A_log"
        )
        # Not a tracked GDN param -> unchanged.
        assert (
            route_fp32_holder_key("model.layers.0.linear_attn.conv1d.weight")
            == "model.layers.0.linear_attn.conv1d.weight"
        )
        # A_log outside linear_attn -> unchanged.
        assert route_fp32_holder_key("model.layers.0.self_attn.A_log") == "model.layers.0.self_attn.A_log"

    def test_strip_route_round_trip(self):
        bare = "model.layers.3.linear_attn.A_log"
        routed = route_fp32_holder_key(bare)
        assert routed == "model.layers.3.linear_attn._fp32_params.A_log"
        assert strip_fp32_holder_key(routed) == bare

    def test_identifies_bare_and_routed_gdn_fp32_param_keys(self):
        assert is_gated_delta_net_fp32_param_key("model.layers.0.linear_attn.A_log")
        assert is_gated_delta_net_fp32_param_key("model.layers.0.linear_attn._fp32_params.dt_bias")
        assert not is_gated_delta_net_fp32_param_key("model.layers.0.self_attn.A_log")
        assert not is_gated_delta_net_fp32_param_key("model.layers.0.linear_attn.conv1d.weight")


class TestStateTensorUpcast:
    def test_upcasts_bf16_gdn_state_tensor(self):
        tensor = torch.ones(4, dtype=torch.bfloat16)

        out = upcast_gated_delta_net_fp32_state_tensor("model.layers.0.linear_attn._fp32_params.dt_bias", tensor)

        assert out.dtype == torch.float32
        torch.testing.assert_close(out, tensor.float())

    def test_leaves_unrelated_bf16_tensor_by_reference(self):
        tensor = torch.ones(4, dtype=torch.bfloat16)

        out = upcast_gated_delta_net_fp32_state_tensor("model.layers.0.self_attn.q_proj.weight", tensor)

        assert out is tensor
        assert out.dtype == torch.bfloat16

    def test_leaves_non_floating_gdn_tensor_by_reference(self):
        tensor = torch.ones(4, dtype=torch.int64)

        out = upcast_gated_delta_net_fp32_state_tensor("model.layers.0.linear_attn.A_log", tensor)

        assert out is tensor
        assert out.dtype == torch.int64
