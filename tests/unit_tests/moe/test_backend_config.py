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

import warnings

import pytest
import torch

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.models.common import BackendConfig


class TestBackendConfigGatePrecision:
    """Test BackendConfig gate_precision field."""

    def test_gate_precision_string_input_fp32(self):
        """Test that BackendConfig gate_precision accepts string input and converts to torch.dtype."""
        backend_config = BackendConfig(gate_precision="torch.float32")
        assert backend_config.gate_precision == torch.float32

    def test_gate_precision_string_input_fp64(self):
        """Test that BackendConfig gate_precision accepts fp64 string input."""
        backend_config = BackendConfig(gate_precision="torch.float64")
        assert backend_config.gate_precision == torch.float64

    def test_gate_precision_string_input_short_form(self):
        """Test that BackendConfig gate_precision accepts short form string input."""
        backend_config = BackendConfig(gate_precision="float32")
        assert backend_config.gate_precision == torch.float32

    def test_gate_precision_none_default(self):
        """Test that BackendConfig gate_precision defaults to None."""
        backend_config = BackendConfig()
        assert backend_config.gate_precision is None

    def test_gate_precision_torch_dtype_input(self):
        """Test that BackendConfig gate_precision accepts torch.dtype directly."""
        backend_config = BackendConfig(gate_precision=torch.float32)
        assert backend_config.gate_precision == torch.float32


class TestBackendConfigExpertsDispatcherValidation:
    """Test BackendConfig validation for experts and dispatcher fields."""

    def test_te_experts_falls_back_to_torch(self):
        """Test that BackendConfig falls back te experts to torch when dispatcher is not deepep."""
        config = BackendConfig(experts="te", dispatcher="torch")
        assert config.experts == "torch"
        assert config.dispatcher == "torch"

    def test_gmm_experts_falls_back_to_torch(self):
        """Test that BackendConfig falls back gmm experts to torch when dispatcher is not deepep."""
        config = BackendConfig(experts="gmm", dispatcher="torch")
        assert config.experts == "torch"
        assert config.dispatcher == "torch"

    def test_te_experts_with_deepep_valid(self):
        """Test that te experts with deepep dispatcher is valid."""
        config = BackendConfig(experts="te", dispatcher="deepep")
        assert config.experts == "te"
        assert config.dispatcher == "deepep"

    def test_gmm_experts_with_deepep_valid(self):
        """Test that gmm experts with deepep dispatcher is valid."""
        config = BackendConfig(experts="gmm", dispatcher="deepep")
        assert config.experts == "gmm"
        assert config.dispatcher == "deepep"

    def test_torch_experts_with_torch_dispatcher_valid(self):
        """Test that torch experts with torch dispatcher is valid."""
        config = BackendConfig(experts="torch", dispatcher="torch")
        assert config.experts == "torch"
        assert config.dispatcher == "torch"

    def test_torch_experts_with_deepep_dispatcher_valid(self):
        """Test that torch experts with deepep dispatcher is valid."""
        config = BackendConfig(experts="torch", dispatcher="deepep")
        assert config.experts == "torch"
        assert config.dispatcher == "deepep"

    def test_torch_mm_experts_with_torch_dispatcher_valid(self):
        """Test that torch_mm experts with torch dispatcher is valid."""
        config = BackendConfig(experts="torch_mm", dispatcher="torch")
        assert config.experts == "torch_mm"
        assert config.dispatcher == "torch"

    def test_torch_mm_experts_with_deepep_dispatcher_valid(self):
        """Test that torch_mm experts with deepep dispatcher is valid."""
        config = BackendConfig(experts="torch_mm", dispatcher="deepep")
        assert config.experts == "torch_mm"
        assert config.dispatcher == "deepep"


class TestBackendConfigEnableDeepepDeprecation:
    """Test backwards compatibility for deprecated enable_deepep parameter."""

    def test_enable_deepep_true_sets_dispatcher_deepep_and_experts_gmm(self):
        """Test that enable_deepep=True sets dispatcher='deepep' and experts='gmm' with deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = BackendConfig(enable_deepep=True)
            assert config.dispatcher == "deepep"
            assert config.experts == "gmm"
            assert config.enable_deepep is None  # Should be cleared after conversion
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "enable_deepep is deprecated" in str(w[0].message)

    def test_enable_deepep_false_sets_dispatcher_torch(self):
        """Test that enable_deepep=False sets dispatcher='torch' with deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = BackendConfig(enable_deepep=False)
            assert config.dispatcher == "torch"
            assert config.experts == "torch"  # experts unchanged when enable_deepep=False
            assert config.enable_deepep is None  # Should be cleared after conversion
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "enable_deepep is deprecated" in str(w[0].message)

    def test_enable_deepep_none_no_warning(self):
        """Test that enable_deepep=None (default) does not trigger warning."""
        from nemo_automodel.components.models.common.utils import HAVE_DEEP_EP

        expected_dispatcher = "deepep" if HAVE_DEEP_EP and torch.cuda.is_available() else "torch"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = BackendConfig()
            assert config.enable_deepep is None
            # No deprecation warning should be raised
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

    def test_enable_deepep_overrides_dispatcher_and_experts(self):
        """Test that enable_deepep takes precedence over dispatcher and experts when both provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Even if dispatcher="torch" and experts="torch", enable_deepep=True should override them
            config = BackendConfig(dispatcher="torch", experts="torch", enable_deepep=True)
            assert config.dispatcher == "deepep"
            assert config.experts == "gmm"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_enable_deepep_false_overrides_dispatcher_deepep(self):
        """Test that enable_deepep=False overrides dispatcher='deepep'."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = BackendConfig(dispatcher="deepep", enable_deepep=False)
            assert config.dispatcher == "torch"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_dispatcher_without_enable_deepep(self):
        """Test that dispatcher works correctly without enable_deepep."""
        config = BackendConfig(dispatcher="deepep")
        assert config.dispatcher == "deepep"
        assert config.enable_deepep is None

        config = BackendConfig(dispatcher="torch")
        assert config.dispatcher == "torch"
        assert config.enable_deepep is None

    def test_deprecation_warning_message_content(self):
        """Test that deprecation warning message contains helpful migration info."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BackendConfig(enable_deepep=True)
            warning_message = str(w[0].message)
            assert "experts='gmm'" in warning_message
            assert "dispatcher='deepep'" in warning_message
            assert "dispatcher='torch'" in warning_message
            assert "will be removed in a future release" in warning_message


class TestMoEConfig:
    """Test MoEConfig dataclass."""

    @pytest.fixture
    def base_moe_config_kwargs(self):
        """Base kwargs for creating a MoEConfig."""
        return {
            "n_routed_experts": 8,
            "n_shared_experts": 0,
            "n_activated_experts": 2,
            "n_expert_groups": 1,
            "n_limited_groups": 1,
            "train_gate": False,
            "gate_bias_update_factor": 0.0,
            "aux_loss_coeff": 0.0,
            "score_func": "softmax",
            "route_scale": 1.0,
            "dim": 128,
            "inter_dim": 256,
            "moe_inter_dim": 256,
            "norm_topk_prob": False,
        }

    def test_dtype_string_input_torch_prefix(self, base_moe_config_kwargs):
        """Test that MoEConfig dtype accepts string input with torch prefix."""
        config = MoEConfig(**base_moe_config_kwargs, dtype="torch.float16")
        assert config.dtype == torch.float16

    def test_dtype_string_input_short_form(self, base_moe_config_kwargs):
        """Test that MoEConfig dtype accepts short form string input."""
        config = MoEConfig(**base_moe_config_kwargs, dtype="bfloat16")
        assert config.dtype == torch.bfloat16

    def test_dtype_torch_dtype_input(self, base_moe_config_kwargs):
        """Test that MoEConfig dtype accepts torch.dtype directly."""
        config = MoEConfig(**base_moe_config_kwargs, dtype=torch.float32)
        assert config.dtype == torch.float32

    def test_dtype_default_bfloat16(self, base_moe_config_kwargs):
        """Test that MoEConfig dtype defaults to bfloat16."""
        config = MoEConfig(**base_moe_config_kwargs)
        assert config.dtype == torch.bfloat16

    def test_expert_activation_default(self, base_moe_config_kwargs):
        """Test that expert_activation defaults to swiglu."""
        config = MoEConfig(**base_moe_config_kwargs)
        assert config.expert_activation == "swiglu"

    def test_expert_activation_quick_geglu(self, base_moe_config_kwargs):
        """Test that expert_activation can be set to quick_geglu."""
        config = MoEConfig(**base_moe_config_kwargs, expert_activation="quick_geglu")
        assert config.expert_activation == "quick_geglu"

    def test_optional_fields_defaults(self, base_moe_config_kwargs):
        """Test that optional fields have correct defaults."""
        config = MoEConfig(**base_moe_config_kwargs)
        assert config.router_bias is False
        assert config.expert_bias is False
        assert config.softmax_before_topk is False
        assert config.shared_expert_gate is False
        assert config.shared_expert_inter_dim is None

    def test_moeconfig_importable_from_layers(self, base_moe_config_kwargs):
        """Test that MoEConfig is still importable from layers for backwards compatibility."""
        from nemo_automodel.components.moe.layers import MoEConfig as MoEConfigFromLayers

        config = MoEConfigFromLayers(**base_moe_config_kwargs)
        assert config.n_routed_experts == 8
