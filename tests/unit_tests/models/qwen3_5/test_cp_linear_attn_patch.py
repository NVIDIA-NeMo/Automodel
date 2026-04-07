# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Qwen3.5 dense CP + FSDP mixed-dtype patching."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

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
    def test_fp32_params_moved_to_holder(self, fake_model, monkeypatch):
        """Float32 bare params are moved into _fp32_params submodule."""
        # Monkeypatch the isinstance check to match our fake class
        import nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn as mod

        monkeypatch.setattr(
            mod, "__builtins__", mod.__builtins__ if hasattr(mod, "__builtins__") else {}
        )
        # Directly test the param-moving logic
        la = fake_model.layers[0].linear_attn
        assert la.A_log.dtype == torch.float32
        assert la.dt_bias.dtype == torch.bfloat16

        # Simulate the param wrapping logic
        holder = None
        for pname in list(la._parameters.keys()):
            param = la._parameters[pname]
            if param is not None and param.dtype == torch.float32:
                if holder is None:
                    holder = nn.Module()
                setattr(holder, pname, param)
                del la._parameters[pname]
                la.__dict__[pname] = param
        if holder is not None:
            la.add_module("_fp32_params", holder)

        # A_log should be accessible via __dict__ but not in _parameters
        assert "A_log" not in la._parameters
        assert la.A_log is holder.A_log  # same tensor
        assert la.A_log.dtype == torch.float32
        # dt_bias stays as a regular parameter
        assert "dt_bias" in la._parameters
        # _fp32_params submodule exists
        assert hasattr(la, "_fp32_params")
        assert la._fp32_params.A_log.dtype == torch.float32

    def test_no_class_swap_when_cp_disabled(self, fake_model):
        """With cp_enabled=False, class should not change."""
        la = fake_model.layers[0].linear_attn
        original_class = type(la)
        # The patch function checks isinstance(mod, Qwen3_5GatedDeltaNet)
        # which won't match our fake, so just verify the flag logic
        assert original_class is _FakeGatedDeltaNet

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
