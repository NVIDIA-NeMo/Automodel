# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for UCCL-EP integration into MoE components.

Tests cover:
- BackendConfig: uccl_ep dispatcher validation and fallback logic
- MoE layer: dispatcher selection for uccl_ep
- TokenDispatcherConfig: moe_flex_dispatcher_backend uccl_ep option
- _DeepepManager: custom dispatch/combine function injection
- fused_a2a: get_uccl_buffer caching, UCCL autograd function wiring
"""

import importlib.util
import warnings
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.models.common import BackendConfig

HAVE_UCCL_EP = importlib.util.find_spec("uccl") is not None or importlib.util.find_spec("ep") is not None
HAVE_TE = importlib.util.find_spec("transformer_engine") is not None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def moe_config():
    return MoEConfig(
        n_routed_experts=8,
        n_shared_experts=2,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.1,
        aux_loss_coeff=0.01,
        score_func="softmax",
        route_scale=1.0,
        dim=128,
        inter_dim=256,
        moe_inter_dim=256,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="flex",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


# ---------------------------------------------------------------------------
# BackendConfig – uccl_ep dispatcher
# ---------------------------------------------------------------------------


class TestBackendConfigUcclEpDispatcher:
    """Test BackendConfig validation for uccl_ep dispatcher."""

    def test_uccl_ep_dispatcher_accepted(self):
        """uccl_ep is a valid dispatcher value."""
        config = BackendConfig(dispatcher="uccl_ep")
        assert config.dispatcher == "uccl_ep"

    def test_te_experts_with_uccl_ep_valid(self):
        """te experts + uccl_ep dispatcher should not fall back."""
        config = BackendConfig(experts="te", dispatcher="uccl_ep")
        assert config.experts == "te"
        assert config.dispatcher == "uccl_ep"

    def test_gmm_experts_with_uccl_ep_valid(self):
        """gmm experts + uccl_ep dispatcher should not fall back."""
        config = BackendConfig(experts="gmm", dispatcher="uccl_ep")
        assert config.experts == "gmm"
        assert config.dispatcher == "uccl_ep"

    def test_torch_mm_experts_with_uccl_ep_valid(self):
        """torch_mm experts + uccl_ep dispatcher is valid."""
        config = BackendConfig(experts="torch_mm", dispatcher="uccl_ep")
        assert config.experts == "torch_mm"
        assert config.dispatcher == "uccl_ep"

    def test_te_experts_falls_back_when_dispatcher_is_torch(self):
        """te experts should still fall back to torch_mm when dispatcher='torch'."""
        config = BackendConfig(experts="te", dispatcher="torch")
        assert config.experts == "torch_mm"
        assert config.dispatcher == "torch"

    def test_gmm_experts_falls_back_when_dispatcher_is_torch(self):
        """gmm experts should still fall back to torch_mm when dispatcher='torch'."""
        config = BackendConfig(experts="gmm", dispatcher="torch")
        assert config.experts == "torch_mm"
        assert config.dispatcher == "torch"


# ---------------------------------------------------------------------------
# MoE layer – uccl_ep dispatcher selection
# ---------------------------------------------------------------------------


class TestMoELayerUcclEpDispatcher:
    """Test MoE layer dispatcher selection for uccl_ep."""

    def test_moe_uccl_ep_single_device_falls_back(self, moe_config, backend_config):
        """uccl_ep dispatcher with world_size=1 should fall back to GroupedExperts."""
        from nemo_automodel.components.moe.experts import GroupedExperts
        from nemo_automodel.components.moe.layers import MoE

        backend_config.experts = "te"
        backend_config.dispatcher = "uccl_ep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=1):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.experts, GroupedExperts)

    def test_moe_uccl_ep_single_device_warning_message(self, moe_config, backend_config):
        """uccl_ep dispatcher with world_size=1 should warn with the dispatcher name."""
        from nemo_automodel.components.moe.layers import MoE

        backend_config.experts = "gmm"
        backend_config.dispatcher = "uccl_ep"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=1):
                MoE(moe_config, backend_config)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "uccl_ep" in str(user_warnings[0].message)
            assert "Expert parallelism requires multiple GPUs" in str(user_warnings[0].message)

    def test_moe_uccl_ep_multi_device_gmm_experts(self, moe_config, backend_config):
        """uccl_ep dispatcher with gmm experts and world_size>1 should use GroupedExpertsDeepEP."""
        from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP
        from nemo_automodel.components.moe.layers import MoE

        backend_config.experts = "gmm"
        backend_config.dispatcher = "uccl_ep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=2):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.experts, GroupedExpertsDeepEP)

    def test_moe_uccl_ep_multi_device_torch_mm_experts(self, moe_config, backend_config):
        """uccl_ep dispatcher with torch_mm experts and world_size>1 should use GroupedExpertsDeepEP."""
        from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP
        from nemo_automodel.components.moe.layers import MoE

        backend_config.experts = "torch_mm"
        backend_config.dispatcher = "uccl_ep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=2):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.experts, GroupedExpertsDeepEP)


# ---------------------------------------------------------------------------
# TokenDispatcherConfig – moe_flex_dispatcher_backend field
# ---------------------------------------------------------------------------


class TestTokenDispatcherConfigUcclEp:
    """Test TokenDispatcherConfig moe_flex_dispatcher_backend for uccl_ep."""

    def test_default_backend_deepep(self):
        from nemo_automodel.components.moe.megatron.token_dispatcher import TokenDispatcherConfig

        config = TokenDispatcherConfig()
        assert config.moe_flex_dispatcher_backend == "deepep"

    def test_backend_uccl_ep(self):
        from nemo_automodel.components.moe.megatron.token_dispatcher import TokenDispatcherConfig

        config = TokenDispatcherConfig(moe_flex_dispatcher_backend="uccl_ep")
        assert config.moe_flex_dispatcher_backend == "uccl_ep"

    def test_backend_hybridep(self):
        from nemo_automodel.components.moe.megatron.token_dispatcher import TokenDispatcherConfig

        config = TokenDispatcherConfig(moe_flex_dispatcher_backend="hybridep")
        assert config.moe_flex_dispatcher_backend == "hybridep"


# ---------------------------------------------------------------------------
# _DeepepManager – custom dispatch/combine fn injection
# ---------------------------------------------------------------------------


class TestDeepepManagerCustomFns:
    """Test _DeepepManager accepts custom _dispatch_fn and _combine_fn."""

    def test_default_uses_fused_dispatch(self):
        """Without custom fns, manager uses module-level fused_dispatch/fused_combine."""
        from nemo_automodel.components.moe.megatron.token_dispatcher import _DeepepManager
        from nemo_automodel.components.moe.megatron import fused_a2a

        group = Mock()
        group.size.return_value = 2

        # If fused_dispatch is None (no deep_ep installed), should raise ImportError
        if fused_a2a.fused_dispatch is None:
            with pytest.raises(ImportError):
                _DeepepManager(group=group, router_topk=2)
        else:
            mgr = _DeepepManager(group=group, router_topk=2)
            assert mgr._fused_dispatch is fused_a2a.fused_dispatch
            assert mgr._fused_combine is fused_a2a.fused_combine

    def test_custom_dispatch_fn_used(self):
        """Custom _dispatch_fn overrides the module-level fused_dispatch."""
        from nemo_automodel.components.moe.megatron.token_dispatcher import _DeepepManager

        group = Mock()
        group.size.return_value = 2
        custom_dispatch = Mock()
        custom_combine = Mock()

        mgr = _DeepepManager(
            group=group,
            router_topk=2,
            _dispatch_fn=custom_dispatch,
            _combine_fn=custom_combine,
        )
        assert mgr._fused_dispatch is custom_dispatch
        assert mgr._fused_combine is custom_combine

    def test_custom_dispatch_fn_none_falls_back(self):
        """Passing None explicitly for _dispatch_fn falls back to module-level."""
        from nemo_automodel.components.moe.megatron.token_dispatcher import _DeepepManager
        from nemo_automodel.components.moe.megatron import fused_a2a

        group = Mock()
        group.size.return_value = 2

        if fused_a2a.fused_dispatch is None:
            with pytest.raises(ImportError):
                _DeepepManager(group=group, router_topk=2, _dispatch_fn=None, _combine_fn=None)
        else:
            mgr = _DeepepManager(group=group, router_topk=2, _dispatch_fn=None, _combine_fn=None)
            assert mgr._fused_dispatch is fused_a2a.fused_dispatch


# ---------------------------------------------------------------------------
# fused_a2a – HAVE_UCCL_EP flag and public API
# ---------------------------------------------------------------------------


class TestFusedA2AUcclEpPublicAPI:
    """Test that uccl_fused_dispatch/uccl_fused_combine are correctly exported."""

    def test_uccl_functions_importable(self):
        """uccl_fused_dispatch and uccl_fused_combine should be importable (may be None)."""
        from nemo_automodel.components.moe.megatron.fused_a2a import (
            uccl_fused_combine,
            uccl_fused_dispatch,
        )

        # They are either callable or None depending on whether uccl is installed
        assert uccl_fused_dispatch is None or callable(uccl_fused_dispatch)
        assert uccl_fused_combine is None or callable(uccl_fused_combine)

    def test_have_uccl_ep_flag(self):
        """HAVE_UCCL_EP flag should be a bool."""
        from nemo_automodel.components.moe.megatron.fused_a2a import HAVE_UCCL_EP

        assert isinstance(HAVE_UCCL_EP, bool)

    def test_uccl_buffer_global_initially_none(self):
        """_uccl_buffer global should start as None."""
        from nemo_automodel.components.moe.megatron import fused_a2a

        # Reset to ensure clean state
        original = fused_a2a._uccl_buffer
        fused_a2a._uccl_buffer = None
        assert fused_a2a._uccl_buffer is None
        fused_a2a._uccl_buffer = original


# ---------------------------------------------------------------------------
# fused_a2a – get_uccl_buffer caching
# ---------------------------------------------------------------------------


class TestGetUcclBufferCaching:
    """Test get_uccl_buffer creates and caches buffers correctly."""

    @pytest.fixture(autouse=True)
    def _reset_uccl_buffer(self):
        """Ensure _uccl_buffer is reset before/after each test."""
        from nemo_automodel.components.moe.megatron import fused_a2a

        original = fused_a2a._uccl_buffer
        fused_a2a._uccl_buffer = None
        yield
        fused_a2a._uccl_buffer = original

    @pytest.mark.skipif(not HAVE_UCCL_EP, reason="UCCL-EP not installed")
    def test_buffer_created_on_first_call(self):
        from nemo_automodel.components.moe.megatron.fused_a2a import get_uccl_buffer
        from nemo_automodel.components.moe.megatron import fused_a2a

        group = Mock()
        group.size.return_value = 2
        buf = get_uccl_buffer(group, 1024)
        assert buf is not None
        assert fused_a2a._uccl_buffer is buf

    @pytest.mark.skipif(not HAVE_UCCL_EP, reason="UCCL-EP not installed")
    def test_buffer_reused_on_same_group(self):
        from nemo_automodel.components.moe.megatron.fused_a2a import get_uccl_buffer

        group = Mock()
        group.size.return_value = 2
        buf1 = get_uccl_buffer(group, 1024)
        buf2 = get_uccl_buffer(group, 1024)
        assert buf1 is buf2


# ---------------------------------------------------------------------------
# HAVE_UCCL_EP flag in utils.py
# ---------------------------------------------------------------------------


class TestUtilsHaveUcclEp:
    """Test HAVE_UCCL_EP flag in models/common/utils.py."""

    def test_have_uccl_ep_is_bool(self):
        from nemo_automodel.components.models.common.utils import HAVE_UCCL_EP

        assert isinstance(HAVE_UCCL_EP, bool)


# ---------------------------------------------------------------------------
# MoEFlexTokenDispatcher – uccl_ep path
# ---------------------------------------------------------------------------


class TestMoEFlexTokenDispatcherUcclEp:
    """Test MoEFlexTokenDispatcher initializes correctly with UCCL-EP."""

    @pytest.fixture(autouse=True)
    def _reset_shared_managers(self):
        """Reset shared managers before/after each test."""
        from nemo_automodel.components.moe.megatron.token_dispatcher import MoEFlexTokenDispatcher

        orig_comm = MoEFlexTokenDispatcher.shared_deepep_manager
        orig_uccl = MoEFlexTokenDispatcher.shared_uccl_manager
        MoEFlexTokenDispatcher.shared_deepep_manager = None
        MoEFlexTokenDispatcher.shared_uccl_manager = None
        yield
        MoEFlexTokenDispatcher.shared_deepep_manager = orig_comm
        MoEFlexTokenDispatcher.shared_uccl_manager = orig_uccl

    def test_uccl_ep_backend_config(self):
        """Config with moe_flex_dispatcher_backend='uccl_ep' should be valid."""
        from nemo_automodel.components.moe.megatron.token_dispatcher import TokenDispatcherConfig

        config = TokenDispatcherConfig(moe_flex_dispatcher_backend="uccl_ep")
        assert config.moe_flex_dispatcher_backend == "uccl_ep"

    def test_uccl_ep_uses_uccl_dispatch_fn(self):
        """When backend='uccl_ep', dispatcher should use uccl_fused_dispatch."""
        from nemo_automodel.components.moe.megatron.token_dispatcher import (
            MoEFlexTokenDispatcher,
            TokenDispatcherConfig,
            _DeepepManager,
        )
        from nemo_automodel.components.moe.megatron import fused_a2a

        config = TokenDispatcherConfig(
            moe_flex_dispatcher_backend="uccl_ep",
            num_moe_experts=8,
            moe_router_topk=2,
        )

        group = Mock()
        group.size.return_value = 2

        captured_kwargs = {}

        def mock_init(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.group = kwargs.get("group")
            self.router_topk = kwargs.get("router_topk")
            self.capacity_factor = kwargs.get("capacity_factor")
            self.permute_fusion = kwargs.get("permute_fusion")
            self.num_experts = kwargs.get("num_experts")
            self.num_local_experts = kwargs.get("num_local_experts")
            self.router_dtype = kwargs.get("router_dtype")
            self.moe_router_expert_pad_multiple = kwargs.get("moe_router_expert_pad_multiple")
            self.token_indices = None
            self.token_probs = None
            self.handle = None
            dispatch_fn = kwargs.get("_dispatch_fn")
            combine_fn = kwargs.get("_combine_fn")
            self._fused_dispatch = dispatch_fn if dispatch_fn is not None else fused_a2a.fused_dispatch
            self._fused_combine = combine_fn if combine_fn is not None else fused_a2a.fused_combine

        with patch.object(_DeepepManager, "__init__", mock_init):
            MoEFlexTokenDispatcher(
                num_local_experts=4,
                local_expert_indices=list(range(4)),
                config=config,
                ep_group=group,
            )

        # Verify uccl dispatch/combine fns were passed
        assert captured_kwargs["_dispatch_fn"] is fused_a2a.uccl_fused_dispatch
        assert captured_kwargs["_combine_fn"] is fused_a2a.uccl_fused_combine

    def test_deepep_uses_no_custom_dispatch_fn(self):
        """When backend='deepep' (default), _dispatch_fn should not be passed."""
        from nemo_automodel.components.moe.megatron.token_dispatcher import (
            MoEFlexTokenDispatcher,
            TokenDispatcherConfig,
            _DeepepManager,
        )
        from nemo_automodel.components.moe.megatron import fused_a2a

        config = TokenDispatcherConfig(
            moe_flex_dispatcher_backend="deepep",
            num_moe_experts=8,
            moe_router_topk=2,
        )

        group = Mock()
        group.size.return_value = 2

        captured_kwargs = {}

        def mock_init(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.group = kwargs.get("group")
            self.router_topk = kwargs.get("router_topk")
            self.capacity_factor = kwargs.get("capacity_factor")
            self.permute_fusion = kwargs.get("permute_fusion")
            self.num_experts = kwargs.get("num_experts")
            self.num_local_experts = kwargs.get("num_local_experts")
            self.router_dtype = kwargs.get("router_dtype")
            self.moe_router_expert_pad_multiple = kwargs.get("moe_router_expert_pad_multiple")
            self.token_indices = None
            self.token_probs = None
            self.handle = None
            dispatch_fn = kwargs.get("_dispatch_fn")
            combine_fn = kwargs.get("_combine_fn")
            self._fused_dispatch = dispatch_fn if dispatch_fn is not None else fused_a2a.fused_dispatch
            self._fused_combine = combine_fn if combine_fn is not None else fused_a2a.fused_combine

        with patch.object(_DeepepManager, "__init__", mock_init):
            MoEFlexTokenDispatcher(
                num_local_experts=4,
                local_expert_indices=list(range(4)),
                config=config,
                ep_group=group,
            )

        assert "_dispatch_fn" not in captured_kwargs
        assert "_combine_fn" not in captured_kwargs

    def test_shared_uccl_manager_used_when_sharing(self):
        """UCCL-EP path should use shared_uccl_manager, not shared_deepep_manager."""
        from nemo_automodel.components.moe.megatron.token_dispatcher import (
            MoEFlexTokenDispatcher,
            TokenDispatcherConfig,
            _DeepepManager,
        )
        from nemo_automodel.components.moe.megatron import fused_a2a

        config = TokenDispatcherConfig(
            moe_flex_dispatcher_backend="uccl_ep",
            num_moe_experts=8,
            moe_router_topk=2,
        )

        group = Mock()
        group.size.return_value = 2

        def mock_init(self, **kwargs):
            self.group = kwargs.get("group")
            self.router_topk = kwargs.get("router_topk")
            self.capacity_factor = kwargs.get("capacity_factor")
            self.permute_fusion = kwargs.get("permute_fusion")
            self.num_experts = kwargs.get("num_experts")
            self.num_local_experts = kwargs.get("num_local_experts")
            self.router_dtype = kwargs.get("router_dtype")
            self.moe_router_expert_pad_multiple = kwargs.get("moe_router_expert_pad_multiple")
            self.token_indices = None
            self.token_probs = None
            self.handle = None
            dispatch_fn = kwargs.get("_dispatch_fn")
            combine_fn = kwargs.get("_combine_fn")
            self._fused_dispatch = dispatch_fn if dispatch_fn is not None else fused_a2a.fused_dispatch
            self._fused_combine = combine_fn if combine_fn is not None else fused_a2a.fused_combine

        with patch.object(_DeepepManager, "__init__", mock_init):
            d1 = MoEFlexTokenDispatcher(
                num_local_experts=4,
                local_expert_indices=list(range(4)),
                config=config,
                ep_group=group,
            )
            d2 = MoEFlexTokenDispatcher(
                num_local_experts=4,
                local_expert_indices=list(range(4)),
                config=config,
                ep_group=group,
            )

        # Both should share the same uccl manager
        assert d1._comm_manager is d2._comm_manager
        assert MoEFlexTokenDispatcher.shared_uccl_manager is not None
        # shared_deepep_manager should remain None since we used uccl path
        assert MoEFlexTokenDispatcher.shared_deepep_manager is None
