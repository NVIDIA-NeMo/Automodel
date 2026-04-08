# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Qwen3.5 dense CP + FSDP mixed-dtype patching."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

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

    def test_class_swap_when_cp_enabled(self, fake_model, monkeypatch):
        """With cp_enabled=True, class is swapped to CPAwareGatedDeltaNet."""
        self._stub_qwen3_5_modules(monkeypatch)

        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)

        from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
            CPAwareGatedDeltaNet,
            patch_hf_model,
        )

        la = fake_model.layers[0].linear_attn
        patch_hf_model(fake_model, cp_enabled=True)
        assert type(la) is CPAwareGatedDeltaNet
        assert la._cp_mesh is None

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


class TestQwen35ParallelizationStrategyParallelize:
    """Tests for Qwen3_5ParallelizationStrategy.parallelize() method."""

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

    @pytest.fixture()
    def mock_device_mesh(self):
        """Create a mock device mesh with CP support."""
        from torch.distributed.device_mesh import DeviceMesh

        mesh = MagicMock(spec=DeviceMesh)
        dp_shard_mesh = MagicMock()
        dp_shard_mesh.size.return_value = 2
        dp_shard_mesh.ndim = 1
        tp_mesh = MagicMock()
        tp_mesh.size.return_value = 1
        tp_mesh.ndim = 1
        cp_mesh = MagicMock()
        cp_mesh.size.return_value = 1
        cp_mesh.ndim = 1

        mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp", "tp")
        mesh.__getitem__ = MagicMock(side_effect=lambda key: {
            "dp_replicate": MagicMock(size=MagicMock(return_value=1), ndim=1),
            "dp_shard_cp": dp_shard_mesh,
            "tp": tp_mesh,
            "cp": cp_mesh,
            ("dp_replicate", "dp_shard_cp"): dp_shard_mesh,
        }[key])

        return mesh, cp_mesh, tp_mesh

    @pytest.fixture()
    def mock_env(self, monkeypatch):
        """Mock the distributed functions used by DefaultParallelizationStrategy."""
        import nemo_automodel.components.distributed.parallelizer as par_mod
        import nemo_automodel.components.distributed.parallelizer_utils as par_utils

        fully_shard_mock = MagicMock(side_effect=lambda model, **kw: model)
        monkeypatch.setattr(par_mod, "fully_shard", fully_shard_mock, raising=False)

        apply_fsdp_mock = MagicMock()
        monkeypatch.setattr(par_mod, "apply_fsdp2_sharding_recursively", apply_fsdp_mock, raising=False)

        # Also mock fully_shard_by_dtype which _fsdp_by_dtype calls
        fsdp_by_dtype_mock = MagicMock()
        monkeypatch.setattr(par_utils, "fully_shard_by_dtype", fsdp_by_dtype_mock, raising=False)

        # Mock _pre_shard_combined_projections which _fsdp_by_dtype calls
        monkeypatch.setattr(par_mod, "_pre_shard_combined_projections", MagicMock(), raising=False)

        extract_mock = MagicMock(return_value=[])
        monkeypatch.setattr(par_mod, "_extract_model_layers", extract_mock, raising=False)

        get_plan_mock = MagicMock(return_value={})
        monkeypatch.setattr(par_mod, "_get_parallel_plan", get_plan_mock, raising=False)

        validate_mock = MagicMock()
        monkeypatch.setattr(par_mod, "validate_tp_mesh", validate_mock, raising=False)

        parallelize_mod_mock = MagicMock()
        monkeypatch.setattr(par_mod, "parallelize_module", parallelize_mod_mock, raising=False)

        checkpoint_mock = MagicMock(side_effect=lambda x: x)
        monkeypatch.setattr(par_mod, "checkpoint_wrapper", checkpoint_mock, raising=False)

        return {
            "apply_fsdp": apply_fsdp_mock,
            "fully_shard": fully_shard_mock,
            "fully_shard_by_dtype": fsdp_by_dtype_mock,
        }

    def test_parallelize_calls_patch_and_delegates(self, fake_model, monkeypatch, mock_device_mesh, mock_env):
        """parallelize() patches the model and delegates to super()."""
        self._stub_qwen3_5_modules(monkeypatch)

        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)

        from nemo_automodel.components.distributed.parallelizer import Qwen3_5ParallelizationStrategy

        mesh, cp_mesh, tp_mesh = mock_device_mesh
        strategy = Qwen3_5ParallelizationStrategy()

        with patch(
            "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.patch_hf_model"
        ) as mock_patch:
            result = strategy.parallelize(model=fake_model, device_mesh=mesh)

        # patch_hf_model was called (cp_enabled=False because "cp" not in mesh_dim_names)
        mock_patch.assert_called_once_with(fake_model, cp_enabled=False)
        # super().parallelize ran fully_shard
        mock_env["fully_shard"].assert_called()
        assert result is fake_model

    def test_parallelize_swaps_and_restores_fsdp_global(self, fake_model, monkeypatch, mock_device_mesh, mock_env):
        """The globals swap for apply_fsdp2_sharding_recursively is restored after call."""
        self._stub_qwen3_5_modules(monkeypatch)

        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)

        import nemo_automodel.components.distributed.parallelizer as par_mod
        from nemo_automodel.components.distributed.parallelizer import Qwen3_5ParallelizationStrategy

        original_fn = par_mod.apply_fsdp2_sharding_recursively
        strategy = Qwen3_5ParallelizationStrategy()

        # Track what function was used during super().parallelize()
        called_with = {}

        def spy_apply_fsdp(*args, **kwargs):
            # During super().parallelize, the global should be the custom _fsdp_by_dtype
            called_with["fn"] = par_mod.apply_fsdp2_sharding_recursively

        mock_env["apply_fsdp"].side_effect = spy_apply_fsdp

        with patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.patch_hf_model"):
            strategy.parallelize(model=fake_model, device_mesh=mock_device_mesh[0])

        # After call, global is restored
        assert par_mod.apply_fsdp2_sharding_recursively is original_fn

    def test_parallelize_restores_global_on_error(self, fake_model, monkeypatch, mock_device_mesh, mock_env):
        """Global is restored even if super().parallelize() raises."""
        self._stub_qwen3_5_modules(monkeypatch)

        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)

        import nemo_automodel.components.distributed.parallelizer as par_mod
        from nemo_automodel.components.distributed.parallelizer import Qwen3_5ParallelizationStrategy

        original_fn = par_mod.apply_fsdp2_sharding_recursively
        strategy = Qwen3_5ParallelizationStrategy()

        mock_env["fully_shard"].side_effect = RuntimeError("boom")

        with patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.patch_hf_model"):
            with pytest.raises(RuntimeError, match="boom"):
                strategy.parallelize(model=fake_model, device_mesh=mock_device_mesh[0])

        # Global still restored
        assert par_mod.apply_fsdp2_sharding_recursively is original_fn

    def test_parallelize_sets_cp_mesh_when_enabled(self, fake_model, monkeypatch, mock_device_mesh, mock_env):
        """When CP is enabled, _cp_mesh is set on CPAwareGatedDeltaNet modules."""
        self._stub_qwen3_5_modules(monkeypatch)

        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)

        from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
            CPAwareGatedDeltaNet,
            patch_hf_model,
        )
        from nemo_automodel.components.distributed.parallelizer import Qwen3_5ParallelizationStrategy

        mesh, cp_mesh, tp_mesh = mock_device_mesh
        # Enable CP by adding "cp" to mesh_dim_names and making cp_mesh.size() > 1
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp", "tp", "cp")
        cp_mesh.size.return_value = 2

        # Pre-patch the model so the module is CPAwareGatedDeltaNet
        patch_hf_model(fake_model, cp_enabled=True)
        la = fake_model.layers[0].linear_attn
        assert type(la) is CPAwareGatedDeltaNet
        assert la._cp_mesh is None

        strategy = Qwen3_5ParallelizationStrategy()

        with patch(
            "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.patch_hf_model"
        ):
            strategy.parallelize(model=fake_model, device_mesh=mesh)

        # CP mesh should be set
        assert la._cp_mesh is cp_mesh

    def test_fsdp_by_dtype_handles_module_list(self, monkeypatch, mock_device_mesh, mock_env):
        """The custom _fsdp_by_dtype correctly iterates ModuleList children."""
        self._stub_qwen3_5_modules(monkeypatch)

        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)

        import nemo_automodel.components.distributed.parallelizer as par_mod
        from nemo_automodel.components.distributed.parallelizer import Qwen3_5ParallelizationStrategy

        # Build a model with layers in a ModuleList
        model = nn.Module()
        model.config = types.SimpleNamespace(
            num_attention_heads=8, num_key_value_heads=8, hidden_size=64,
        )
        model.__class__.__name__ = "Qwen3_5ForCausalLM"
        inner = nn.Module()
        layer = nn.Module()
        layer.mlp = nn.Linear(4, 4)
        inner.layers = nn.ModuleList([layer])
        model.model = inner

        mesh, cp_mesh, tp_mesh = mock_device_mesh

        # Capture what the custom _fsdp_by_dtype does
        shard_by_dtype_calls = []
        with patch(
            "nemo_automodel.components.distributed.parallelizer_utils.fully_shard_by_dtype",
            side_effect=lambda *a, **kw: shard_by_dtype_calls.append(a[0]),
        ), patch(
            "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.patch_hf_model"
        ), patch(
            "nemo_automodel.components.distributed.parallelizer._pre_shard_combined_projections"
        ):
            # Make extract_layers return the real layers
            mock_env["apply_fsdp"].side_effect = lambda module, mesh, mp, offload=None: (
                par_mod.apply_fsdp2_sharding_recursively(module, mesh, mp, offload)
            )
            strategy = Qwen3_5ParallelizationStrategy()
            strategy.parallelize(model=model, device_mesh=mesh)

        # fully_shard_by_dtype should have been called for the layer child
        assert len(shard_by_dtype_calls) > 0
        assert layer in shard_by_dtype_calls
