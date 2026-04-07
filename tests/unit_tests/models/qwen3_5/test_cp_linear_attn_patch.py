# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Qwen3.5 dense CP + FSDP mixed-dtype patching."""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn


class _FakeGatedDeltaNet(nn.Module):
    """Mimics HF Qwen3_5GatedDeltaNet with mixed-dtype bare params."""

    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        self.conv1d = nn.Conv1d(4, 4, 1)
        self.norm = nn.LayerNorm(4)
        # Force norm to float32
        self.norm.weight.data = self.norm.weight.data.float()
        self.norm.bias.data = self.norm.bias.data.float()
        self.layer_idx = 0


@pytest.fixture()
def fake_model():
    """Build a minimal model with a fake GatedDeltaNet layer."""
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].linear_attn = _FakeGatedDeltaNet()
    model.layers[0].layer_type = "linear_attention"
    return model


class TestPatchHfModel:
    @staticmethod
    def _stub_qwen3_5_modules(monkeypatch):
        """Stub transformers.models.qwen3_5* so cp_linear_attn can be imported."""
        for path in (
            "transformers.models.qwen3_5_moe",
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
            "transformers.models.qwen3_5",
            "transformers.models.qwen3_5.modeling_qwen3_5",
        ):
            if path not in sys.modules:
                stub = types.ModuleType(path)
                stub.Qwen3_5MoeGatedDeltaNet = _FakeGatedDeltaNet
                stub.Qwen3_5GatedDeltaNet = _FakeGatedDeltaNet
                monkeypatch.setitem(sys.modules, path, stub)

    def test_fp32_params_moved_to_holder(self, fake_model, monkeypatch):
        """Float32 bare params are moved into _fp32_params submodule via real patch_hf_model."""
        self._stub_qwen3_5_modules(monkeypatch)

        # Remove cached cp_linear_attn so re-import picks up our stubs
        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)

        from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import patch_hf_model

        la = fake_model.layers[0].linear_attn
        assert la.A_log.dtype == torch.float32
        assert la.dt_bias.dtype == torch.bfloat16

        patch_hf_model(fake_model, cp_enabled=False)

        # A_log (float32) should be moved out of _parameters into __dict__
        assert "A_log" not in la._parameters
        assert la.A_log.dtype == torch.float32
        # dt_bias (bfloat16) stays as a regular parameter
        assert "dt_bias" in la._parameters
        # _fp32_params submodule holds the moved param
        assert hasattr(la, "_fp32_params")
        assert la._fp32_params.A_log.dtype == torch.float32
        # __dict__ reference and holder share the same tensor
        assert la.A_log is la._fp32_params.A_log

    def test_no_class_swap_when_cp_disabled(self, fake_model, monkeypatch):
        """With cp_enabled=False, class should not change to CPAwareGatedDeltaNet."""
        self._stub_qwen3_5_modules(monkeypatch)

        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)

        from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import patch_hf_model

        la = fake_model.layers[0].linear_attn
        patch_hf_model(fake_model, cp_enabled=False)
        assert type(la) is _FakeGatedDeltaNet

    def test_dict_access_preserves_tensor_identity(self, fake_model):
        """__dict__ reference and _fp32_params hold the same tensor."""
        la = fake_model.layers[0].linear_attn
        original_A_log = la.A_log

        holder = nn.Module()
        setattr(holder, "A_log", la.A_log)
        del la._parameters["A_log"]
        la.__dict__["A_log"] = original_A_log
        la.add_module("_fp32_params", holder)

        assert la.A_log is la._fp32_params.A_log
        assert id(la.A_log) == id(original_A_log)


class TestAttachLinearAttnPositionHooks:
    def test_hook_caches_position_ids(self, fake_model):
        """Pre-hook stores position_ids on linear_attn module."""
        from nemo_automodel.components.distributed.cp_utils import attach_linear_attn_position_hooks

        attach_linear_attn_position_hooks(fake_model)

        layer = fake_model.layers[0]
        pos_ids = torch.arange(10)

        # Simulate decoder layer forward call with position_ids kwarg
        # The hook fires on the layer (which has linear_attn + layer_type)
        for hook in layer._forward_pre_hooks.values():
            hook(layer, (), {"position_ids": pos_ids})

        assert layer.linear_attn._cached_position_ids is pos_ids

    def test_hook_deduplication(self, fake_model):
        """Calling twice does not register duplicate hooks."""
        from nemo_automodel.components.distributed.cp_utils import attach_linear_attn_position_hooks

        attach_linear_attn_position_hooks(fake_model)
        n_hooks = len(fake_model.layers[0]._forward_pre_hooks)

        attach_linear_attn_position_hooks(fake_model)
        assert len(fake_model.layers[0]._forward_pre_hooks) == n_hooks

    def test_no_hook_on_non_linear_attn_layers(self):
        """Layers without linear_attn don't get hooks."""
        from nemo_automodel.components.distributed.cp_utils import attach_linear_attn_position_hooks

        model = nn.Module()
        model.layers = nn.ModuleList([nn.Module()])
        model.layers[0].self_attn = nn.Linear(4, 4)
        model.layers[0].layer_type = "full_attention"

        attach_linear_attn_position_hooks(model)
        assert len(model.layers[0]._forward_pre_hooks) == 0


class TestQwen35ParallelizationStrategyRegistration:
    def test_strategy_registered(self):
        """Qwen3.5 model classes are in the strategy registry."""
        from nemo_automodel.components.distributed.parallelizer import PARALLELIZATION_STRATEGIES

        assert "Qwen3_5ForConditionalGeneration" in PARALLELIZATION_STRATEGIES
        assert "Qwen3_5ForCausalLM" in PARALLELIZATION_STRATEGIES

    def test_strategy_type(self):
        """Strategy is Qwen3_5ParallelizationStrategy."""
        from nemo_automodel.components.distributed.parallelizer import (
            PARALLELIZATION_STRATEGIES,
            Qwen3_5ParallelizationStrategy,
        )

        assert isinstance(PARALLELIZATION_STRATEGIES["Qwen3_5ForCausalLM"], Qwen3_5ParallelizationStrategy)
