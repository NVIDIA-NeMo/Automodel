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

from unittest.mock import MagicMock, Mock, patch

from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin, _configure_fsdp_module, _iter_fsdp_modules
from nemo_automodel.components.moe.utils import BackendConfig


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


class MockBackend:
    """Mock backend with FSDP optimization config."""

    def __init__(self, enable_fsdp_optimizations=False):
        self.enable_fsdp_optimizations = enable_fsdp_optimizations


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

    def __init__(self, backend, model, has_lm_head=False, has_embed_tokens=False):
        self.backend = backend
        self.model = model
        if has_lm_head:
            self.lm_head = MockFSDPModule()
        if has_embed_tokens:
            model.embed_tokens = MockFSDPModule()


class TestConfigureFSDPModule:
    """Test _configure_fsdp_module helper function."""

    def test_sets_all_flags(self):
        fsdp_module = MockFSDPModule()

        _configure_fsdp_module(
            fsdp_module, is_last_backward=True, reshard_after_backward=True, requires_gradient_sync=True
        )

        assert fsdp_module._is_last_backward is True
        assert fsdp_module._reshard_after_backward is True
        assert fsdp_module._requires_gradient_sync is True

    def test_sets_flags_false(self):
        fsdp_module = MockFSDPModule()

        _configure_fsdp_module(
            fsdp_module, is_last_backward=False, reshard_after_backward=False, requires_gradient_sync=False
        )

        assert fsdp_module._is_last_backward is False
        assert fsdp_module._reshard_after_backward is False
        assert fsdp_module._requires_gradient_sync is False


class TestIterFSDPModules:
    """Test _iter_fsdp_modules helper function."""

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_iterates_model_only(self, mock_isinstance):
        # Mock isinstance to return True only for the model
        def isinstance_side_effect(obj, cls):
            if cls.__name__ == 'FSDPModule':
                return isinstance(obj, MockFSDPModule)
            return isinstance(obj, cls)
        mock_isinstance.side_effect = isinstance_side_effect

        model = MockFSDPModule()
        moe_model = MockMoEModel(MockBackend(), model)

        modules = list(_iter_fsdp_modules(moe_model))

        assert len(modules) == 1
        assert modules[0] is model

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_iterates_model_and_lm_head(self, mock_isinstance):
        def isinstance_side_effect(obj, cls):
            if cls.__name__ == 'FSDPModule':
                return isinstance(obj, MockFSDPModule)
            return isinstance(obj, cls)
        mock_isinstance.side_effect = isinstance_side_effect

        model = MockFSDPModule()
        moe_model = MockMoEModel(MockBackend(), model, has_lm_head=True)

        modules = list(_iter_fsdp_modules(moe_model))

        assert len(modules) == 2
        assert model in modules
        assert moe_model.lm_head in modules

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_iterates_model_embeddings_lm_head(self, mock_isinstance):
        def isinstance_side_effect(obj, cls):
            if cls.__name__ == 'FSDPModule':
                return isinstance(obj, MockFSDPModule)
            return isinstance(obj, cls)
        mock_isinstance.side_effect = isinstance_side_effect

        model = MockFSDPModule()
        moe_model = MockMoEModel(MockBackend(), model, has_lm_head=True, has_embed_tokens=True)

        modules = list(_iter_fsdp_modules(moe_model))

        assert len(modules) == 3
        assert model in modules
        assert model.embed_tokens in modules
        assert moe_model.lm_head in modules

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_iterates_with_experts(self, mock_isinstance):
        def isinstance_side_effect(obj, cls):
            if cls.__name__ == 'FSDPModule':
                return isinstance(obj, MockFSDPModule)
            return isinstance(obj, cls)
        mock_isinstance.side_effect = isinstance_side_effect

        model = MockFSDPModule()
        model.layers = Mock()
        blocks = []
        for i in range(2):
            block = Mock()
            block.mlp = Mock()
            block.mlp.experts = MockFSDPModule()
            blocks.append((f"layer_{i}", block))
        model.layers.named_children = Mock(return_value=blocks)

        moe_model = MockMoEModel(MockBackend(), model)

        modules = list(_iter_fsdp_modules(moe_model))

        # model + 2 experts
        assert len(modules) == 3
        assert model in modules


class TestPrepareForGradAccumulation:
    """Test prepare_for_grad_accumulation method."""

    def test_no_optimizations_returns_early(self):
        backend = MockBackend(enable_fsdp_optimizations=False)
        model = MockFSDPModule()
        moe_model = MockMoEModel(backend, model)

        # Should return early without error
        moe_model.prepare_for_grad_accumulation(pp_enabled=False)

    def test_pp_enabled_returns_early(self):
        backend = MockBackend(enable_fsdp_optimizations=True)
        model = MockFSDPModule()
        moe_model = MockMoEModel(backend, model)

        # PP enabled should return early (handled by patched backward)
        moe_model.prepare_for_grad_accumulation(pp_enabled=True)

        # Flags should remain unchanged
        assert model._is_last_backward is False
        assert model._reshard_after_backward is False
        assert model._requires_gradient_sync is False

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_defers_sync_and_resharding(self, mock_isinstance):
        def isinstance_side_effect(obj, cls):
            if cls.__name__ == 'FSDPModule':
                return isinstance(obj, MockFSDPModule)
            return isinstance(obj, cls)
        mock_isinstance.side_effect = isinstance_side_effect

        backend = MockBackend(enable_fsdp_optimizations=True)
        model = MockFSDPModule()
        model.layers = Mock()
        blocks = []
        block = Mock()
        block.mlp = Mock()
        block.mlp.experts = MockFSDPModule()
        blocks.append(("layer_0", block))
        model.layers.named_children = Mock(return_value=blocks)

        moe_model = MockMoEModel(backend, model, has_lm_head=True)

        moe_model.prepare_for_grad_accumulation(pp_enabled=False)

        # All should defer (not requires_gradient_sync, not reshard_after_backward)
        assert model._is_last_backward is False
        assert model._reshard_after_backward is False
        assert model._requires_gradient_sync is False
        assert moe_model.lm_head._is_last_backward is False
        assert moe_model.lm_head._reshard_after_backward is False
        assert moe_model.lm_head._requires_gradient_sync is False
        assert block.mlp.experts._is_last_backward is False
        assert block.mlp.experts._reshard_after_backward is False
        assert block.mlp.experts._requires_gradient_sync is False


class TestPrepareForFinalBackward:
    """Test prepare_for_final_backward method."""

    def test_no_optimizations_returns_early(self):
        backend = MockBackend(enable_fsdp_optimizations=False)
        model = MockFSDPModule()
        moe_model = MockMoEModel(backend, model)

        # Should return early without error
        moe_model.prepare_for_final_backward(pp_enabled=False)

    def test_pp_enabled_returns_early(self):
        backend = MockBackend(enable_fsdp_optimizations=True)
        model = MockFSDPModule()
        moe_model = MockMoEModel(backend, model)

        # PP enabled should return early (handled by patched backward)
        moe_model.prepare_for_final_backward(pp_enabled=True)

        # Flags should remain unchanged
        assert model._is_last_backward is False
        assert model._reshard_after_backward is False
        assert model._requires_gradient_sync is False

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_enables_sync_and_resharding(self, mock_isinstance):
        def isinstance_side_effect(obj, cls):
            if cls.__name__ == 'FSDPModule':
                return isinstance(obj, MockFSDPModule)
            return isinstance(obj, cls)
        mock_isinstance.side_effect = isinstance_side_effect

        backend = MockBackend(enable_fsdp_optimizations=True)
        model = MockFSDPModule()
        model.layers = Mock()
        blocks = []
        block = Mock()
        block.mlp = Mock()
        block.mlp.experts = MockFSDPModule()
        blocks.append(("layer_0", block))
        model.layers.named_children = Mock(return_value=blocks)

        moe_model = MockMoEModel(backend, model, has_lm_head=True)

        moe_model.prepare_for_final_backward(pp_enabled=False)

        # All should enable sync and resharding for the last backward
        assert model._is_last_backward is True
        assert model._reshard_after_backward is True
        assert model._requires_gradient_sync is True
        assert moe_model.lm_head._is_last_backward is True
        assert moe_model.lm_head._reshard_after_backward is True
        assert moe_model.lm_head._requires_gradient_sync is True
        assert block.mlp.experts._is_last_backward is True
        assert block.mlp.experts._reshard_after_backward is True
        assert block.mlp.experts._requires_gradient_sync is True


class TestFullWorkflow:
    """Test complete workflow with both methods."""

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_grad_accumulation_workflow(self, mock_isinstance):
        def isinstance_side_effect(obj, cls):
            if cls.__name__ == 'FSDPModule':
                return isinstance(obj, MockFSDPModule)
            return isinstance(obj, cls)
        mock_isinstance.side_effect = isinstance_side_effect

        backend = MockBackend(enable_fsdp_optimizations=True)
        model = MockFSDPModule()
        model.layers = Mock()
        blocks = []
        block = Mock()
        block.mlp = Mock()
        block.mlp.experts = MockFSDPModule()
        blocks.append(("layer_0", block))
        model.layers.named_children = Mock(return_value=blocks)

        moe_model = MockMoEModel(backend, model, has_lm_head=True)

        # Step 1: Prepare for gradient accumulation
        moe_model.prepare_for_grad_accumulation(pp_enabled=False)

        # Verify all defer
        assert model._requires_gradient_sync is False
        assert model._reshard_after_backward is False

        # Step 2: Prepare for final backward
        moe_model.prepare_for_final_backward(pp_enabled=False)

        # Verify all enable sync/resharding
        assert model._is_last_backward is True
        assert model._reshard_after_backward is True
        assert model._requires_gradient_sync is True
        assert moe_model.lm_head._is_last_backward is True
        assert moe_model.lm_head._reshard_after_backward is True
        assert moe_model.lm_head._requires_gradient_sync is True
        assert block.mlp.experts._is_last_backward is True
        assert block.mlp.experts._reshard_after_backward is True
        assert block.mlp.experts._requires_gradient_sync is True
