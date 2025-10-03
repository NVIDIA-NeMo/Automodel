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

from unittest.mock import Mock, MagicMock, patch

from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.utils import FsdpOptimizationConfig


class MockFSDPModule:
    """Mock FSDP module for testing."""
    def __init__(self):
        self._is_last_backward = False
        self._reshard_after_backward = False
        self._requires_gradient_sync = False

    def set_is_last_backward(self, value):
        self._is_last_backward = value

    def set_reshard_after_backward(self, value):
        self._reshard_after_backward = value

    def set_requires_gradient_sync(self, value):
        self._requires_gradient_sync = value


class MockFSDPState:
    """Mock FSDP state for testing post_backward hooks."""
    def __init__(self, has_param_group=True):
        self._fsdp_param_group = MagicMock() if has_param_group else None


class MockFSDPStateCtx:
    """Mock FSDP state context."""
    def __init__(self, num_states=2):
        self.all_states = [MockFSDPState() for _ in range(num_states)]


class MockFullyShardState:
    """Mock fully_shard state."""
    def __init__(self):
        self._state_ctx = MockFSDPStateCtx()
        self._root_post_backward_final_callback = MagicMock()


class MockBackend:
    """Mock backend with FSDP optimization config."""
    def __init__(self, fsdp_optimization_config=None):
        self.fsdp_optimization_config = fsdp_optimization_config


class MockModel:
    """Mock model with layers structure."""
    def __init__(self, has_moe=True, num_layers=2):
        self.layers = Mock()
        blocks = []
        for i in range(num_layers):
            block = Mock()
            block.mlp = Mock()
            if has_moe:
                block.mlp.experts = MockFSDPModule()
            blocks.append((f"layer_{i}", block))
        self.layers.named_children = Mock(return_value=blocks)


class MockMoEModel(MoEFSDPSyncMixin):
    """Mock MoE model that uses the FSDP mixin."""
    def __init__(self, backend, model, has_lm_head=False):
        self.backend = backend
        self.model = model
        if has_lm_head:
            self.lm_head = MockFSDPModule()


class TestLastBackwardForFSDPModule:
    """Test static method for setting last backward states."""

    def test_sets_all_flags_true(self):
        fsdp_module = MockFSDPModule()

        MoEFSDPSyncMixin.last_backward_for_fsdp_module(fsdp_module)

        assert fsdp_module._is_last_backward is True
        assert fsdp_module._reshard_after_backward is True
        assert fsdp_module._requires_gradient_sync is True


class TestFirstForwardForFSDPModule:
    """Test static method for setting first forward states."""

    def test_default_flags_false(self):
        fsdp_module = MockFSDPModule()

        MoEFSDPSyncMixin.first_forward_for_fsdp_module(fsdp_module)

        assert fsdp_module._is_last_backward is False
        assert fsdp_module._reshard_after_backward is False
        assert fsdp_module._requires_gradient_sync is False

    def test_custom_flags(self):
        fsdp_module = MockFSDPModule()

        MoEFSDPSyncMixin.first_forward_for_fsdp_module(
            fsdp_module,
            requires_gradient_sync=True,
            reshard_after_backward=True
        )

        assert fsdp_module._is_last_backward is False
        assert fsdp_module._reshard_after_backward is True
        assert fsdp_module._requires_gradient_sync is True


@patch('nemo_automodel.components.moe.fsdp_mixin.fully_shard')
class TestPostBackwardForFSDPModules:
    """Test static method for post-backward finalization."""

    def test_single_module(self, mock_fully_shard):
        fsdp_module = MockFSDPModule()
        fsdp_state = MockFullyShardState()
        mock_fully_shard.state.return_value = fsdp_state

        MoEFSDPSyncMixin.post_backward_for_fsdp_modules([fsdp_module])

        # Verify last backward flags set
        assert fsdp_module._is_last_backward is True
        assert fsdp_module._reshard_after_backward is True
        assert fsdp_module._requires_gradient_sync is True

        # Verify post_backward called
        for state in fsdp_state._state_ctx.all_states:
            state._fsdp_param_group.post_backward.assert_called_once()

        # Verify final callback called
        fsdp_state._root_post_backward_final_callback.assert_called_once()

    def test_multiple_modules(self, mock_fully_shard):
        fsdp_modules = [MockFSDPModule(), MockFSDPModule()]
        fsdp_state1 = MockFullyShardState()
        fsdp_state2 = MockFullyShardState()

        # Return different states for the two loops in post_backward_for_fsdp_modules
        mock_fully_shard.state.side_effect = [
            fsdp_state1, fsdp_state2,  # First loop
            fsdp_state1, fsdp_state2   # Second loop
        ]

        MoEFSDPSyncMixin.post_backward_for_fsdp_modules(fsdp_modules)

        # Verify all modules processed
        for module in fsdp_modules:
            assert module._is_last_backward is True

        # Verify both states' callbacks called
        fsdp_state1._root_post_backward_final_callback.assert_called_once()
        fsdp_state2._root_post_backward_final_callback.assert_called_once()


class TestSetFSDPStatesForLastBackward:
    """Test instance method for setting states before last backward."""

    def test_no_config_returns_early(self):
        backend = MockBackend(fsdp_optimization_config=None)
        model = MockModel()
        moe_model = MockMoEModel(backend, model)

        # Should return early without error
        moe_model.set_fsdp_states_for_last_backward()

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_fsdp_model(self, mock_isinstance):
        # Mock isinstance to return True for FSDPModule check
        mock_isinstance.return_value = True

        config = FsdpOptimizationConfig()
        backend = MockBackend(fsdp_optimization_config=config)
        model = MockFSDPModule()
        moe_model = MockMoEModel(backend, model)

        moe_model.set_fsdp_states_for_last_backward()

        assert model._is_last_backward is True
        assert model._reshard_after_backward is True
        assert model._requires_gradient_sync is True

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_fsdp_model_and_lm_head(self, mock_isinstance):
        # Mock isinstance to return True for FSDPModule check
        mock_isinstance.return_value = True

        config = FsdpOptimizationConfig()
        backend = MockBackend(fsdp_optimization_config=config)
        model = MockFSDPModule()
        moe_model = MockMoEModel(backend, model, has_lm_head=True)

        moe_model.set_fsdp_states_for_last_backward()

        assert model._is_last_backward is True
        assert moe_model.lm_head._is_last_backward is True

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_moe_experts(self, mock_isinstance):
        # Mock isinstance to return True for FSDPModule check
        mock_isinstance.return_value = True

        config = FsdpOptimizationConfig()
        backend = MockBackend(fsdp_optimization_config=config)
        # Use MockFSDPModule for model since isinstance will return True
        model = MockFSDPModule()
        model.layers = Mock()

        # Create MoE layers
        blocks = []
        for i in range(3):
            block = Mock()
            block.mlp = Mock()
            block.mlp.experts = MockFSDPModule()
            blocks.append((f"layer_{i}", block))
        model.layers.named_children = Mock(return_value=blocks)

        moe_model = MockMoEModel(backend, model)

        moe_model.set_fsdp_states_for_last_backward()

        # Verify model and all expert modules have flags set
        assert model._is_last_backward is True
        for _, block in model.layers.named_children():
            experts = block.mlp.experts
            assert experts._is_last_backward is True
            assert experts._reshard_after_backward is True
            assert experts._requires_gradient_sync is True


class TestFinalizeFSDPStatesPostBackward:
    """Test instance method for finalizing FSDP states after backward."""

    def test_no_config_returns_early(self):
        backend = MockBackend(fsdp_optimization_config=None)
        model = MockModel()
        moe_model = MockMoEModel(backend, model)

        moe_model.finalize_fsdp_states_post_backward()
        # No assertions needed - just verify it doesn't crash

    @patch('nemo_automodel.components.moe.fsdp_mixin.MoEFSDPSyncMixin.post_backward_for_fsdp_modules')
    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_collects_all_fsdp_modules(self, mock_isinstance, mock_post_backward):
        # Mock isinstance to return True for FSDPModule check
        mock_isinstance.return_value = True

        config = FsdpOptimizationConfig()
        backend = MockBackend(fsdp_optimization_config=config)
        model = MockFSDPModule()
        model.layers = Mock()

        # Create MoE layers
        blocks = []
        for i in range(2):
            block = Mock()
            block.mlp = Mock()
            block.mlp.experts = MockFSDPModule()
            blocks.append((f"layer_{i}", block))
        model.layers.named_children = Mock(return_value=blocks)

        moe_model = MockMoEModel(backend, model, has_lm_head=True)

        moe_model.finalize_fsdp_states_post_backward()

        # Verify post_backward_for_fsdp_modules was called with all FSDP modules
        mock_post_backward.assert_called_once()
        fsdp_modules_arg = mock_post_backward.call_args[0][0]
        # Should have model + lm_head + 2 experts = 4 modules
        assert len(fsdp_modules_arg) == 4


class TestSetFSDPStatesForFirstForward:
    """Test instance method for setting states before first forward."""

    def test_no_config_returns_early(self):
        backend = MockBackend(fsdp_optimization_config=None)
        model = MockModel()
        moe_model = MockMoEModel(backend, model)

        moe_model.set_fsdp_states_for_first_forward()

    def test_defer_all_operations(self):
        config = FsdpOptimizationConfig(
            defer_grad_sync_for_model=True,
            defer_reshard_after_backward_for_model=True,
            defer_grad_sync_for_lm_head=True,
            defer_reshard_after_backward_for_lm_head=True,
            defer_grad_sync_for_experts=True,
            defer_reshard_after_backward_for_experts=True,
        )
        backend = MockBackend(fsdp_optimization_config=config)
        model = MockFSDPModule()
        model.layers = Mock()

        blocks = []
        block = Mock()
        block.mlp = Mock()
        block.mlp.experts = MockFSDPModule()
        blocks.append(("layer_0", block))
        model.layers.named_children = Mock(return_value=blocks)

        moe_model = MockMoEModel(backend, model, has_lm_head=True)

        moe_model.set_fsdp_states_for_first_forward()

        # All should defer (not requires_gradient_sync, not reshard_after_backward)
        assert model._requires_gradient_sync is False
        assert model._reshard_after_backward is False
        assert moe_model.lm_head._requires_gradient_sync is False
        assert moe_model.lm_head._reshard_after_backward is False
        assert block.mlp.experts._requires_gradient_sync is False
        assert block.mlp.experts._reshard_after_backward is False

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_no_defer_operations(self, mock_isinstance):
        # Mock isinstance to return True for FSDPModule check
        mock_isinstance.return_value = True

        config = FsdpOptimizationConfig(
            defer_grad_sync_for_model=False,
            defer_reshard_after_backward_for_model=False,
            defer_grad_sync_for_lm_head=False,
            defer_reshard_after_backward_for_lm_head=False,
            defer_grad_sync_for_experts=False,
            defer_reshard_after_backward_for_experts=False,
        )
        backend = MockBackend(fsdp_optimization_config=config)
        model = MockFSDPModule()
        model.layers = Mock()

        blocks = []
        block = Mock()
        block.mlp = Mock()
        block.mlp.experts = MockFSDPModule()
        blocks.append(("layer_0", block))
        model.layers.named_children = Mock(return_value=blocks)

        moe_model = MockMoEModel(backend, model, has_lm_head=True)

        moe_model.set_fsdp_states_for_first_forward()

        # All should sync and reshard immediately
        assert model._requires_gradient_sync is True
        assert model._reshard_after_backward is True
        assert moe_model.lm_head._requires_gradient_sync is True
        assert moe_model.lm_head._reshard_after_backward is True
        assert block.mlp.experts._requires_gradient_sync is True
        assert block.mlp.experts._reshard_after_backward is True

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_mixed_defer_strategy(self, mock_isinstance):
        # Mock isinstance to return True for FSDPModule check
        mock_isinstance.return_value = True

        config = FsdpOptimizationConfig(
            defer_grad_sync_for_model=True,
            defer_reshard_after_backward_for_model=False,
            defer_grad_sync_for_lm_head=False,
            defer_reshard_after_backward_for_lm_head=True,
            defer_grad_sync_for_experts=True,
            defer_reshard_after_backward_for_experts=True,
        )
        backend = MockBackend(fsdp_optimization_config=config)
        model = MockFSDPModule()
        model.layers = Mock()

        blocks = []
        block = Mock()
        block.mlp = Mock()
        block.mlp.experts = MockFSDPModule()
        blocks.append(("layer_0", block))
        model.layers.named_children = Mock(return_value=blocks)

        moe_model = MockMoEModel(backend, model, has_lm_head=True)

        moe_model.set_fsdp_states_for_first_forward()

        # Model: defer grad sync, immediate reshard
        assert model._requires_gradient_sync is False
        assert model._reshard_after_backward is True

        # LM head: immediate grad sync, defer reshard
        assert moe_model.lm_head._requires_gradient_sync is True
        assert moe_model.lm_head._reshard_after_backward is False

        # Experts: defer both
        assert block.mlp.experts._requires_gradient_sync is False
        assert block.mlp.experts._reshard_after_backward is False
