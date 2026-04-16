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

        # A_log (float32) should be moved out of _parameters
        assert "A_log" not in la._parameters
        # Accessed via __getattr__ → _fp32_params
        assert la.A_log.dtype == torch.float32
        # dt_bias (bfloat16) stays as a regular parameter
        assert "dt_bias" in la._parameters
        # _fp32_params submodule holds the moved param
        assert hasattr(la, "_fp32_params")
        assert la._fp32_params.A_log.dtype == torch.float32
        # __getattr__ resolves to the same tensor in _fp32_params
        assert la.A_log is la._fp32_params.A_log

    def test_class_always_swapped_for_fsdp(self, fake_model, monkeypatch):
        """Class is always swapped to CPAwareGatedDeltaNet for FSDP fp32 unshard support."""
        self._stub_qwen3_5_modules(monkeypatch)

        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)

        from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
            CPAwareGatedDeltaNet,
            patch_hf_model,
        )

        la = fake_model.layers[0].linear_attn
        patch_hf_model(fake_model, cp_enabled=False)
        assert type(la) is CPAwareGatedDeltaNet
        assert la._cp_mesh is None

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

    def test_getattr_resolves_after_param_replacement(self, fake_model, monkeypatch):
        """__getattr__ resolves to _fp32_params even after the underlying tensor is replaced."""
        self._stub_qwen3_5_modules(monkeypatch)

        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)

        from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import patch_hf_model

        la = fake_model.layers[0].linear_attn
        patch_hf_model(fake_model, cp_enabled=False)

        # Simulate FSDP replacing the parameter in _fp32_params
        new_tensor = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        la._fp32_params._parameters["A_log"] = new_tensor

        # __getattr__ should resolve to the NEW tensor, not the old one
        assert la.A_log is new_tensor


class TestFp32ParamHolder:
    """Tests for _Fp32ParamHolder forward (gate computation)."""

    @staticmethod
    def _stub_and_import(monkeypatch):
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
        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)
        from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import _Fp32ParamHolder

        return _Fp32ParamHolder

    def test_holder_forward_computes_gate(self, monkeypatch):
        """_Fp32ParamHolder.forward returns g = -A_log.exp() * softplus(a + dt_bias)."""
        _Fp32ParamHolder = self._stub_and_import(monkeypatch)
        holder = _Fp32ParamHolder()
        holder.A_log = nn.Parameter(torch.ones(4, dtype=torch.float32))
        a = torch.zeros(4)
        dt_bias = torch.zeros(4)
        g = holder(a, dt_bias)
        # g = -exp(1) * softplus(0 + 0) = -e * softplus(0) = -e * ln(2)
        expected = -torch.ones(4).float().exp() * torch.nn.functional.softplus(torch.zeros(4))
        assert torch.allclose(g, expected, atol=1e-5)

    def test_holder_forward_dtype_is_float32(self, monkeypatch):
        """Gate computation happens in float32 even with bfloat16 inputs."""
        _Fp32ParamHolder = self._stub_and_import(monkeypatch)
        holder = _Fp32ParamHolder()
        holder.A_log = nn.Parameter(torch.ones(4, dtype=torch.float32))
        a = torch.zeros(4, dtype=torch.bfloat16)
        dt_bias = torch.zeros(4, dtype=torch.bfloat16)
        g = holder(a, dt_bias)
        assert g.dtype == torch.float32


class TestComputeGate:
    """Tests for CPAwareGatedDeltaNet._compute_gate routing."""

    @staticmethod
    def _stub_and_import(monkeypatch):
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
        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)
        from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
            CPAwareGatedDeltaNet,
            _Fp32ParamHolder,
            patch_hf_model,
        )

        return CPAwareGatedDeltaNet, _Fp32ParamHolder, patch_hf_model

    def test_compute_gate_routes_through_holder(self, fake_model, monkeypatch):
        """_compute_gate calls _fp32_params.forward when holder exists."""
        CPAwareGatedDeltaNet, _Fp32ParamHolder, patch_hf_model = self._stub_and_import(monkeypatch)
        patch_hf_model(fake_model, cp_enabled=False)
        la = fake_model.layers[0].linear_attn
        assert isinstance(la, CPAwareGatedDeltaNet)
        a = torch.zeros(4)
        with patch.object(la._fp32_params, "forward", return_value=torch.zeros(4)) as mock_fwd:
            la._compute_gate(a)
            mock_fwd.assert_called_once()

    def test_compute_gate_fallback_without_holder(self, fake_model, monkeypatch):
        """_compute_gate falls back to inline computation without _fp32_params."""
        CPAwareGatedDeltaNet, _, patch_hf_model = self._stub_and_import(monkeypatch)
        la = fake_model.layers[0].linear_attn
        la.__class__ = CPAwareGatedDeltaNet
        la._cp_mesh = None
        # No _fp32_params — A_log is still in _parameters
        a = torch.zeros(4)
        g = la._compute_gate(a)
        assert g.dtype == torch.float32
        assert g.shape == (4,)


class TestPatchHfModelSentinel:
    """Test that __getattr__ patching is idempotent."""

    @staticmethod
    def _stub_and_import(monkeypatch):
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
        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)
        from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import patch_hf_model

        return patch_hf_model

    def test_double_patch_does_not_rewrap_getattr(self, monkeypatch):
        """Calling patch_hf_model twice does not chain __getattr__ wrappers."""
        patch_hf_model = self._stub_and_import(monkeypatch)

        model1 = nn.Module()
        model1.layers = nn.ModuleList([nn.Module()])
        model1.layers[0].linear_attn = _FakeGatedDeltaNet()
        model1.layers[0].layer_type = "linear_attention"
        patch_hf_model(model1, cp_enabled=False)
        getattr_after_first = type(model1.layers[0].linear_attn).__getattr__

        model2 = nn.Module()
        model2.layers = nn.ModuleList([nn.Module()])
        model2.layers[0].linear_attn = _FakeGatedDeltaNet()
        model2.layers[0].layer_type = "linear_attention"
        patch_hf_model(model2, cp_enabled=False)
        getattr_after_second = type(model2.layers[0].linear_attn).__getattr__

        # Same function, not a wrapper of a wrapper
        assert getattr_after_first is getattr_after_second


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
        mesh.__getitem__ = MagicMock(
            side_effect=lambda key: {
                "dp_replicate": MagicMock(size=MagicMock(return_value=1), ndim=1),
                "dp_shard_cp": dp_shard_mesh,
                "tp": tp_mesh,
                "cp": cp_mesh,
                ("dp_replicate", "dp_shard_cp"): dp_shard_mesh,
            }[key]
        )

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

        with patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.patch_hf_model") as mock_patch:
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

        from nemo_automodel.components.distributed.parallelizer import Qwen3_5ParallelizationStrategy
        from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
            CPAwareGatedDeltaNet,
            patch_hf_model,
        )

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

        with patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.patch_hf_model"):
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
            num_attention_heads=8,
            num_key_value_heads=8,
            hidden_size=64,
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
        with (
            patch(
                "nemo_automodel.components.distributed.parallelizer_utils.fully_shard_by_dtype",
                side_effect=lambda *a, **kw: shard_by_dtype_calls.append(a[0]),
            ),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.patch_hf_model"),
            patch("nemo_automodel.components.distributed.parallelizer._pre_shard_combined_projections"),
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


# ---------------------------------------------------------------------------
# Helpers for _forward_no_cp tests
# ---------------------------------------------------------------------------

# Dimensions used throughout the _forward_no_cp tests.
_HIDDEN = 16
_NUM_K_HEADS = 2
_NUM_V_HEADS = 2
_HEAD_K_DIM = 4
_HEAD_V_DIM = 4
_KEY_DIM = _NUM_K_HEADS * _HEAD_K_DIM  # 8
_VALUE_DIM = _NUM_V_HEADS * _HEAD_V_DIM  # 8
_CONV_DIM = _KEY_DIM * 2 + _VALUE_DIM  # 24
_CONV_KERNEL = 4


class _IdentityNorm(nn.Module):
    """Simple norm replacement that accepts (x, z) and returns x unchanged."""

    def forward(self, x, z):
        return x


def _build_forward_module(monkeypatch):
    """Import CPAwareGatedDeltaNet with stubs and build a module ready for _forward_no_cp.

    Returns (module, CPAwareGatedDeltaNet_class).
    """
    # Stub transformers modules
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
            # apply_mask_to_padding_states is imported at runtime inside _forward_no_cp
            stub.apply_mask_to_padding_states = lambda hidden_states, mask: hidden_states
            monkeypatch.setitem(sys.modules, path, stub)
        else:
            # Ensure existing stub has the function
            existing = sys.modules[path]
            if not hasattr(existing, "apply_mask_to_padding_states"):
                existing.apply_mask_to_padding_states = lambda hidden_states, mask: hidden_states

    cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
    if cp_mod_key in sys.modules:
        monkeypatch.delitem(sys.modules, cp_mod_key)

    from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

    # Build a _FakeGatedDeltaNet and swap its class
    mod = _FakeGatedDeltaNet()
    mod.__class__ = CPAwareGatedDeltaNet
    mod._cp_mesh = None

    # -- Submodules expected by _forward_no_cp --
    mod.in_proj_qkv = nn.Linear(_HIDDEN, _CONV_DIM, bias=False)
    mod.in_proj_z = nn.Linear(_HIDDEN, _VALUE_DIM, bias=False)
    mod.in_proj_b = nn.Linear(_HIDDEN, _NUM_V_HEADS, bias=False)
    mod.in_proj_a = nn.Linear(_HIDDEN, _NUM_V_HEADS, bias=False)
    mod.conv1d = nn.Conv1d(
        _CONV_DIM,
        _CONV_DIM,
        _CONV_KERNEL,
        groups=_CONV_DIM,
        padding=_CONV_KERNEL - 1,
    )
    mod.out_proj = nn.Linear(_VALUE_DIM, _HIDDEN, bias=False)

    # Norm that accepts (x, z) -> returns x (simplified)
    mod.norm = _IdentityNorm()

    # Override A_log and dt_bias to match num_v_heads (the _FakeGatedDeltaNet
    # creates them with size 4, but in_proj_a output is num_v_heads=2)
    mod.A_log = nn.Parameter(torch.ones(_NUM_V_HEADS, dtype=torch.float32))
    mod.dt_bias = nn.Parameter(torch.ones(_NUM_V_HEADS, dtype=torch.bfloat16))

    # Scalar/dim attributes
    mod.head_k_dim = _HEAD_K_DIM
    mod.head_v_dim = _HEAD_V_DIM
    mod.num_k_heads = _NUM_K_HEADS
    mod.num_v_heads = _NUM_V_HEADS
    mod.key_dim = _KEY_DIM
    mod.value_dim = _VALUE_DIM
    mod.conv_kernel_size = _CONV_KERNEL
    mod.activation = "silu"
    mod.layer_idx = 0

    # chunk_gated_delta_rule — returns (output, state) with correct shape
    def _fake_chunk_gdn(query, key, value, *, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel):
        # output: same shape as value [B, S, H_v, D_v]
        return value.clone(), None

    mod.chunk_gated_delta_rule = _fake_chunk_gdn

    # causal_conv1d_fn — None forces the fallback conv path
    mod.causal_conv1d_fn = None

    return mod, CPAwareGatedDeltaNet


class TestForwardNoCp:
    """Tests for CPAwareGatedDeltaNet._forward_no_cp covering lines 91-193."""

    def test_basic_forward_pass(self, monkeypatch):
        """_forward_no_cp produces output with correct shape (basic path, cache_params=None)."""
        mod, _ = _build_forward_module(monkeypatch)
        x = torch.randn(2, 8, _HIDDEN)
        out = mod._forward_no_cp(x)
        assert out.shape == (2, 8, _HIDDEN)

    def test_forward_no_cp_cache_params_none(self, monkeypatch):
        """Training path: cache_params=None exercises the else-branch at line 133-146."""
        mod, _ = _build_forward_module(monkeypatch)
        x = torch.randn(1, 4, _HIDDEN)
        out = mod._forward_no_cp(x, cache_params=None, cache_position=None)
        assert out.shape == (1, 4, _HIDDEN)

    def test_forward_no_cp_causal_conv1d_fn_none_fallback(self, monkeypatch):
        """When causal_conv1d_fn is None, falls back to F.silu(conv1d(...))."""
        mod, _ = _build_forward_module(monkeypatch)
        assert mod.causal_conv1d_fn is None  # confirm fallback path
        x = torch.randn(1, 6, _HIDDEN)
        out = mod._forward_no_cp(x)
        assert out.shape == (1, 6, _HIDDEN)

    def test_forward_no_cp_with_causal_conv1d_fn(self, monkeypatch):
        """When causal_conv1d_fn is set, it is called instead of conv1d fallback."""
        mod, _ = _build_forward_module(monkeypatch)

        # Install a mock causal_conv1d_fn
        def _mock_causal_conv1d_fn(*, x, weight, bias, activation, seq_idx):
            return torch.nn.functional.silu(x)

        mod.causal_conv1d_fn = _mock_causal_conv1d_fn
        x = torch.randn(1, 6, _HIDDEN)
        out = mod._forward_no_cp(x)
        assert out.shape == (1, 6, _HIDDEN)

    def test_forward_no_cp_attention_mask(self, monkeypatch):
        """attention_mask is passed through apply_mask_to_padding_states."""
        mod, _ = _build_forward_module(monkeypatch)
        x = torch.randn(1, 4, _HIDDEN)
        mask = torch.ones(1, 4, dtype=torch.bool)
        out = mod._forward_no_cp(x, attention_mask=mask)
        assert out.shape == (1, 4, _HIDDEN)

    def test_forward_no_cp_gqa_repeat(self, monkeypatch):
        """When num_v_heads > num_k_heads, q/k are repeat-interleaved."""
        mod, _ = _build_forward_module(monkeypatch)
        # Make v_heads > k_heads to trigger the repeat_interleave branch
        new_num_v_heads = 4
        mod.num_v_heads = new_num_v_heads
        mod.num_k_heads = 2
        # Adjust value_dim to match new num_v_heads
        new_value_dim = new_num_v_heads * _HEAD_V_DIM  # 16
        mod.value_dim = new_value_dim
        conv_dim = _KEY_DIM * 2 + new_value_dim  # 32
        mod.in_proj_qkv = nn.Linear(_HIDDEN, conv_dim, bias=False)
        mod.in_proj_z = nn.Linear(_HIDDEN, new_value_dim, bias=False)
        mod.in_proj_b = nn.Linear(_HIDDEN, new_num_v_heads, bias=False)
        mod.in_proj_a = nn.Linear(_HIDDEN, new_num_v_heads, bias=False)
        mod.conv1d = nn.Conv1d(conv_dim, conv_dim, _CONV_KERNEL, groups=conv_dim, padding=_CONV_KERNEL - 1)
        mod.out_proj = nn.Linear(new_value_dim, _HIDDEN, bias=False)
        # Update A_log and dt_bias to match new num_v_heads
        mod.A_log = nn.Parameter(torch.ones(new_num_v_heads, dtype=torch.float32))
        mod.dt_bias = nn.Parameter(torch.ones(new_num_v_heads, dtype=torch.bfloat16))

        x = torch.randn(1, 4, _HIDDEN)
        out = mod._forward_no_cp(x)
        assert out.shape == (1, 4, _HIDDEN)

    def test_forward_no_cp_uses_compute_gate(self, monkeypatch):
        """_forward_no_cp delegates gate computation to _compute_gate."""
        mod, _ = _build_forward_module(monkeypatch)
        original_compute_gate = mod._compute_gate
        called = []

        def _tracking_compute_gate(a):
            called.append(True)
            return original_compute_gate(a)

        mod._compute_gate = _tracking_compute_gate
        x = torch.randn(1, 4, _HIDDEN)
        mod._forward_no_cp(x)
        assert len(called) == 1, "_compute_gate should be called exactly once"

    def test_forward_no_cp_output_dtype_matches_input(self, monkeypatch):
        """Output dtype follows the projection layers (float32 in this test)."""
        mod, _ = _build_forward_module(monkeypatch)
        x = torch.randn(1, 4, _HIDDEN, dtype=torch.float32)
        out = mod._forward_no_cp(x)
        assert out.dtype == torch.float32


class TestForwardDispatch:
    """Tests for forward() dispatching to _forward_no_cp (lines 207-213)."""

    def test_forward_delegates_when_cp_mesh_none(self, monkeypatch):
        """forward() calls _forward_no_cp when _cp_mesh is None."""
        mod, _ = _build_forward_module(monkeypatch)
        mod._cp_mesh = None
        x = torch.randn(1, 4, _HIDDEN)
        out = mod.forward(x)
        assert out.shape == (1, 4, _HIDDEN)

    def test_forward_delegates_when_cp_mesh_size_1(self, monkeypatch):
        """forward() calls _forward_no_cp when _cp_mesh.size() <= 1."""
        mod, _ = _build_forward_module(monkeypatch)
        mock_mesh = MagicMock()
        mock_mesh.size.return_value = 1
        mod._cp_mesh = mock_mesh
        x = torch.randn(1, 4, _HIDDEN)
        out = mod.forward(x)
        assert out.shape == (1, 4, _HIDDEN)

    def test_forward_passes_cache_params_through(self, monkeypatch):
        """forward() passes cache_params, cache_position, attention_mask to _forward_no_cp."""
        mod, _ = _build_forward_module(monkeypatch)
        mod._cp_mesh = None

        called_with = {}
        orig_fwd = mod._forward_no_cp

        def _spy(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
            called_with["cache_params"] = cache_params
            called_with["cache_position"] = cache_position
            called_with["attention_mask"] = attention_mask
            return orig_fwd(
                hidden_states, cache_params=cache_params, cache_position=cache_position, attention_mask=attention_mask
            )

        mod._forward_no_cp = _spy

        x = torch.randn(1, 4, _HIDDEN)
        mask = torch.ones(1, 4, dtype=torch.bool)
        mod.forward(x, attention_mask=mask)

        assert called_with["cache_params"] is None
        assert called_with["cache_position"] is None
        assert called_with["attention_mask"] is mask

    def test_forward_ignores_extra_cp_kwargs(self, monkeypatch):
        """forward() accepts position_ids, qkv_format, etc. but ignores them on no-CP path."""
        mod, _ = _build_forward_module(monkeypatch)
        mod._cp_mesh = None
        x = torch.randn(1, 4, _HIDDEN)
        out = mod.forward(
            x,
            position_ids=torch.arange(4).unsqueeze(0),
            qkv_format="bshd",
            cu_seqlens=None,
            seq_index=None,
        )
        assert out.shape == (1, 4, _HIDDEN)


class TestMakeFp32GetattrFallback:
    """Test _make_fp32_getattr fallback when attr is not in _fp32_params."""

    @staticmethod
    def _stub_and_import(monkeypatch):
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
        cp_mod_key = "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
        if cp_mod_key in sys.modules:
            monkeypatch.delitem(sys.modules, cp_mod_key)
        from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
            _make_fp32_getattr,
            patch_hf_model,
        )

        return _make_fp32_getattr, patch_hf_model

    def test_fallback_raises_attribute_error(self, fake_model, monkeypatch):
        """Accessing a non-existent attr via patched __getattr__ raises AttributeError."""
        _make_fp32_getattr, patch_hf_model = self._stub_and_import(monkeypatch)
        patch_hf_model(fake_model, cp_enabled=False)
        la = fake_model.layers[0].linear_attn
        # _fp32_params exists, but "nonexistent_xyz" is not in it
        with pytest.raises(AttributeError):
            _ = la.nonexistent_xyz

    def test_fallback_resolves_real_attrs(self, fake_model, monkeypatch):
        """Patched __getattr__ still resolves normal module attributes."""
        _make_fp32_getattr, patch_hf_model = self._stub_and_import(monkeypatch)
        patch_hf_model(fake_model, cp_enabled=False)
        la = fake_model.layers[0].linear_attn
        # conv1d is a real submodule — should resolve fine
        assert la.conv1d is not None
