# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for nemo_automodel._transformers.validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn

from nemo_automodel._transformers.validation import (
    ModelCapabilitiesMixin,
    attach_capabilities_and_validate,
)

_PARALLELIZE_PATH = "nemo_automodel.components.distributed.optimized_tp_plans.PARALLELIZE_FUNCTIONS"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Bare(nn.Module):
    """Model with no parallelism attributes."""

    def forward(self, input_ids):
        return input_ids


class _WithPP(nn.Module):
    _pp_plan = {"layers": "split"}

    def forward(self, input_ids):
        return input_ids


class _WithTP(nn.Module):
    _tp_plan = {"layers": "col"}

    def forward(self, input_ids):
        return input_ids


class _WithSDPA(nn.Module):
    _supports_sdpa = True

    def forward(self, input_ids):
        return input_ids


class _WithSeqLens(nn.Module):
    _supports_sdpa = True

    def forward(self, input_ids, seq_lens=None):
        return input_ids


class _WithKwargs(nn.Module):
    _supports_sdpa = True

    def forward(self, input_ids, **kwargs):
        return input_ids


def _mesh(tp=1, pp=1, cp=1, ep=1):
    return SimpleNamespace(tp_size=tp, pp_size=pp, cp_size=cp, ep_size=ep)


def _attach(model):
    """Inject mixin without validation (for testing supports properties)."""
    if isinstance(model, ModelCapabilitiesMixin):
        return model
    model.__class__ = type(
        model.__class__.__name__,
        (ModelCapabilitiesMixin, model.__class__),
        {},
    )
    return model


def _make_moe_cls():
    from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin

    class _MoE(MoEFSDPSyncMixin, nn.Module):
        def forward(self, input_ids):
            return input_ids

    return _MoE


def _make_moe_te_cls():
    """MoE model whose backend.attn == 'te'."""
    from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin

    class _MoETE(MoEFSDPSyncMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.backend = SimpleNamespace(attn="te")

        def forward(self, input_ids):
            return input_ids

    return _MoETE


# ---------------------------------------------------------------------------
# attach_capabilities
# ---------------------------------------------------------------------------


class TestAttachCapabilities:
    def test_injects_mixin(self):
        model = _Bare()
        _attach(model)
        assert isinstance(model, ModelCapabilitiesMixin)
        assert hasattr(model, "supports")
        assert hasattr(model, "validate_for_mesh")

    def test_idempotent(self):
        model = _Bare()
        _attach(model)
        cls_after_first = type(model)
        _attach(model)
        assert type(model) is cls_after_first

    def test_preserves_original_class_name(self):
        model = _Bare()
        _attach(model)
        assert type(model).__name__ == "_Bare"

    def test_attach_and_validate_calls_validation(self):
        model = _Bare()
        with patch(_PARALLELIZE_PATH, {}):
            with pytest.raises(ValueError, match="no TP plan"):
                attach_capabilities_and_validate(model, _mesh(tp=2))


# ---------------------------------------------------------------------------
# ModelSupports properties
# ---------------------------------------------------------------------------


class TestModelSupportsTP:
    def test_tp_true_with_optimized_plan(self):
        model = _Bare()
        _attach(model)
        with patch(_PARALLELIZE_PATH, {_Bare: lambda: {}}):
            assert model.supports.supports_tp is True

    def test_tp_true_with_hf_native_plan(self):
        model = _WithTP()
        _attach(model)
        with patch(_PARALLELIZE_PATH, {}):
            assert model.supports.supports_tp is True

    def test_tp_false_without_plan(self):
        model = _Bare()
        _attach(model)
        with patch(_PARALLELIZE_PATH, {}):
            assert model.supports.supports_tp is False


class TestModelSupportsPP:
    def test_pp_true(self):
        model = _WithPP()
        _attach(model)
        assert model.supports.supports_pp is True

    def test_pp_false(self):
        model = _Bare()
        _attach(model)
        assert model.supports.supports_pp is False


class TestModelSupportsCP:
    def test_hf_true_with_sdpa(self):
        model = _WithSDPA()
        _attach(model)
        assert model.supports.supports_cp is True

    def test_hf_false_without_sdpa(self):
        model = _Bare()
        _attach(model)
        assert model.supports.supports_cp is False

    def test_hf_false_for_hybrid(self):
        """HF hybrid model (Mamba layers) cannot use CP even with SDPA."""
        model = _WithSDPA()
        model.config = SimpleNamespace(hybrid_override_pattern="MMAMM")
        _attach(model)
        assert model.supports.supports_cp is False

    def test_hf_false_for_hybrid_layers_block_type(self):
        model = _WithSDPA()
        model.config = SimpleNamespace(layers_block_type=["attention", "M", "attention"])
        _attach(model)
        assert model.supports.supports_cp is False

    def test_custom_true_with_te(self):
        cls = _make_moe_te_cls()
        model = cls()
        _attach(model)
        assert model.supports.supports_cp is True

    def test_custom_false_with_flex(self):
        """Custom model using FlexAttention does not support CP."""

        class _Flex(nn.Module):
            def __init__(self):
                super().__init__()
                self.backend = SimpleNamespace(attn="flex")

            def forward(self, x):
                return x

        model = _Flex()
        _attach(model)
        assert model.supports.supports_cp is False

    def test_custom_moe_false_without_te(self):
        cls = _make_moe_cls()
        model = cls()
        _attach(model)
        assert model.supports.supports_cp is False


class TestModelSupportsEP:
    def test_ep_true_for_moe(self):
        cls = _make_moe_cls()
        model = cls()
        _attach(model)
        assert model.supports.supports_ep is True

    def test_ep_false_for_dense(self):
        model = _Bare()
        _attach(model)
        assert model.supports.supports_ep is False


class TestModelSupportsSequencePacking:
    def test_true_with_seq_lens(self):
        model = _WithSeqLens()
        _attach(model)
        assert model.supports.supports_sequence_packing is True

    def test_true_with_kwargs(self):
        model = _WithKwargs()
        _attach(model)
        assert model.supports.supports_sequence_packing is True

    def test_false_without_seq_lens(self):
        model = _Bare()
        _attach(model)
        assert model.supports.supports_sequence_packing is False


class TestModelSupportsGradientCheckpointing:
    def test_true_when_supported(self):
        class _WithGC(nn.Module):
            supports_gradient_checkpointing = True

            def forward(self, x):
                return x

        model = _WithGC()
        _attach(model)
        assert model.supports.supports_gradient_checkpointing is True

    def test_false_for_moe(self):
        cls = _make_moe_cls()
        model = cls()
        _attach(model)
        assert model.supports.supports_gradient_checkpointing is False

    def test_false_by_default(self):
        model = _Bare()
        _attach(model)
        assert model.supports.supports_gradient_checkpointing is False


class TestModelSupportsCPWithSequencePacking:
    def test_cp1_seq_packing_supported(self):
        """When cp_size=1, just checks seq_lens support."""
        model = _WithSeqLens()
        _attach(model)
        model._mesh = _mesh(cp=1)
        assert model.supports.supports_cp_with_sequence_packing is True

    def test_cp1_seq_packing_unsupported(self):
        model = _Bare()
        _attach(model)
        model._mesh = _mesh(cp=1)
        assert model.supports.supports_cp_with_sequence_packing is False

    def test_cp_gt1_with_te_and_seq_lens(self):
        """CP>1 + packing requires TE attention."""
        cls = _make_moe_te_cls()

        class _MoETESeqLens(cls):
            def forward(self, input_ids, seq_lens=None):
                return input_ids

        model = _MoETESeqLens()
        _attach(model)
        model._mesh = _mesh(cp=2)
        assert model.supports.supports_cp_with_sequence_packing is True

    def test_cp_gt1_without_te_fails(self):
        """CP>1 + packing but no TE → not supported."""
        model = _WithSeqLens()
        _attach(model)
        model._mesh = _mesh(cp=2)
        assert model.supports.supports_cp_with_sequence_packing is False

    def test_cp_gt1_no_seq_lens(self):
        model = _Bare()
        _attach(model)
        model._mesh = _mesh(cp=2)
        assert model.supports.supports_cp_with_sequence_packing is False


class TestModelSupportsRepr:
    def test_repr(self):
        model = _Bare()
        _attach(model)
        with patch(_PARALLELIZE_PATH, {}):
            r = repr(model.supports)
        assert "ModelSupports(" in r
        assert "tp=" in r
        assert "pp=" in r


# ---------------------------------------------------------------------------
# validate_for_mesh
# ---------------------------------------------------------------------------


class TestValidateForMesh:
    def test_tp_fails(self):
        model = _Bare()
        _attach(model)
        model._mesh = _mesh(tp=2)
        with patch(_PARALLELIZE_PATH, {}):
            with pytest.raises(ValueError, match="Tensor parallelism.*no TP plan"):
                model.validate_for_mesh()

    def test_tp_passes_with_plan(self):
        model = _WithTP()
        _attach(model)
        model._mesh = _mesh(tp=2)
        with patch(_PARALLELIZE_PATH, {}):
            model.validate_for_mesh()

    def test_pp_fails(self):
        model = _Bare()
        _attach(model)
        model._mesh = _mesh(pp=2)
        with pytest.raises(ValueError, match="Pipeline parallelism.*_pp_plan"):
            model.validate_for_mesh()

    def test_pp_passes_with_plan(self):
        model = _WithPP()
        _attach(model)
        model._mesh = _mesh(pp=2)
        model.validate_for_mesh()

    def test_cp_fails_hf_no_sdpa(self):
        model = _Bare()
        _attach(model)
        model._mesh = _mesh(cp=2)
        with pytest.raises(ValueError, match="Context parallelism.*_supports_sdpa"):
            model.validate_for_mesh()

    def test_cp_fails_hybrid(self):
        model = _WithSDPA()
        model.config = SimpleNamespace(hybrid_override_pattern="MMAMM")
        _attach(model)
        model._mesh = _mesh(cp=2)
        with pytest.raises(ValueError, match="Context parallelism.*hybrid.*Mamba"):
            model.validate_for_mesh()

    def test_cp_fails_custom_flex(self):
        class _Flex(nn.Module):
            def __init__(self):
                super().__init__()
                self.backend = SimpleNamespace(attn="flex")

            def forward(self, x):
                return x

        model = _Flex()
        _attach(model)
        model._mesh = _mesh(cp=2)
        with pytest.raises(ValueError, match="Context parallelism.*TE attention backend"):
            model.validate_for_mesh()

    def test_cp_passes_custom_te(self):
        cls = _make_moe_te_cls()
        model = cls()
        _attach(model)
        model._mesh = _mesh(cp=4)
        model.validate_for_mesh()

    def test_cp_passes_hf_sdpa(self):
        model = _WithSDPA()
        _attach(model)
        model._mesh = _mesh(cp=2)
        model.validate_for_mesh()

    def test_ep_fails_for_dense(self):
        model = _Bare()
        _attach(model)
        model._mesh = _mesh(ep=2)
        with pytest.raises(ValueError, match="Expert parallelism.*MoE model"):
            model.validate_for_mesh()

    def test_ep_passes_for_moe(self):
        cls = _make_moe_cls()
        model = cls()
        _attach(model)
        model._mesh = _mesh(ep=4)
        model.validate_for_mesh()

    def test_multiple_errors(self):
        model = _Bare()
        _attach(model)
        model._mesh = _mesh(tp=2, pp=4, ep=2, cp=2)
        with patch(_PARALLELIZE_PATH, {}):
            with pytest.raises(ValueError) as exc_info:
                model.validate_for_mesh()
            msg = str(exc_info.value)
            assert "no TP plan" in msg
            assert "_pp_plan" in msg
            assert "MoE" in msg
            assert "Context parallelism" in msg

    def test_no_mesh_is_noop(self):
        model = _Bare()
        _attach(model)
        model._mesh = None
        model.validate_for_mesh()

    def test_size_1_is_noop(self):
        model = _Bare()
        _attach(model)
        model._mesh = _mesh()
        model.validate_for_mesh()


# ---------------------------------------------------------------------------
# _is_config_compatible_with_custom_model + get_is_hf_model
# ---------------------------------------------------------------------------


class TestGetIsHfModel:
    """Verify that get_is_hf_model uses config compatibility checks."""

    def test_nemotron_h_v3_uses_custom(self):
        from unittest.mock import MagicMock

        from nemo_automodel._transformers.model_init import get_is_hf_model

        config = SimpleNamespace(
            architectures=["NemotronHForCausalLM"],
            n_routed_experts=8,
        )
        reg_mapping = MagicMock()
        reg_mapping.__contains__ = lambda self, k: k == "NemotronHForCausalLM"
        with patch("nemo_automodel._transformers.model_init.ModelRegistry") as mock_reg:
            mock_reg.model_arch_name_to_cls = reg_mapping
            assert get_is_hf_model(config, force_hf=False) is False

    def test_nemotron_h_v2_falls_through_to_hf(self):
        from unittest.mock import MagicMock

        from nemo_automodel._transformers.model_init import get_is_hf_model

        config = SimpleNamespace(
            architectures=["NemotronHForCausalLM"],
        )
        reg_mapping = MagicMock()
        reg_mapping.__contains__ = lambda self, k: k == "NemotronHForCausalLM"
        with patch("nemo_automodel._transformers.model_init.ModelRegistry") as mock_reg:
            mock_reg.model_arch_name_to_cls = reg_mapping
            assert get_is_hf_model(config, force_hf=False) is True

    def test_force_hf_always_returns_true(self):
        from unittest.mock import MagicMock

        from nemo_automodel._transformers.model_init import get_is_hf_model

        config = SimpleNamespace(architectures=["LlamaForCausalLM"])
        reg_mapping = MagicMock()
        reg_mapping.__contains__ = lambda self, k: True
        with patch("nemo_automodel._transformers.model_init.ModelRegistry") as mock_reg:
            mock_reg.model_arch_name_to_cls = reg_mapping
            assert get_is_hf_model(config, force_hf=True) is True
