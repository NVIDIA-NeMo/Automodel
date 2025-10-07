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

from nemo_automodel.components.moe.fsdp_mixin import (
    MoEFSDPSyncMixin,
    _configure_fsdp_module,
    _disable_fsdp_for_moe_module,
    _iter_fsdp_modules,
    _run_post_backward_for_moe_module,
    _run_post_backward_hooks,
    get_is_optim_step,
    patched_backward_maybe_with_nosync,
    set_is_optim_step,
)
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


class TestGlobalOptimStepFlag:
    """Test global optimization step flag functions."""

    def test_set_and_get_is_optim_step(self):
        set_is_optim_step(True)
        assert get_is_optim_step() is True

        set_is_optim_step(False)
        assert get_is_optim_step() is False


class TestRunPostBackwardHooks:
    """Test _run_post_backward_hooks helper function."""

    @patch('nemo_automodel.components.moe.fsdp_mixin.fully_shard')
    def test_runs_post_backward_and_returns_callback(self, mock_fully_shard):
        fsdp_module = MockFSDPModule()

        # Create mock FSDP state
        mock_state1 = Mock()
        mock_state1._fsdp_param_group = Mock()
        mock_state2 = Mock()
        mock_state2._fsdp_param_group = None  # Test None case

        mock_state_ctx = Mock()
        mock_state_ctx.all_states = [mock_state1, mock_state2]

        mock_fsdp_state = Mock()
        mock_fsdp_state._state_ctx = mock_state_ctx
        mock_callback = Mock()
        mock_fsdp_state._root_post_backward_final_callback = mock_callback

        mock_fully_shard.state.return_value = mock_fsdp_state

        result = _run_post_backward_hooks(fsdp_module)

        # Verify post_backward was called only for state with param_group
        mock_state1._fsdp_param_group.post_backward.assert_called_once()

        # Verify callback is returned (not called yet)
        assert result is mock_callback


class TestDisableFsdpForMoeModule:
    """Test _disable_fsdp_for_moe_module helper function."""

    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_disables_all_fsdp_modules(self, mock_isinstance):
        def isinstance_side_effect(obj, cls):
            if cls.__name__ == 'FSDPModule':
                return isinstance(obj, MockFSDPModule)
            return isinstance(obj, cls)
        mock_isinstance.side_effect = isinstance_side_effect

        model = MockFSDPModule()
        model.layers = Mock()
        blocks = []
        block = Mock()
        block.mlp = Mock()
        block.mlp.experts = MockFSDPModule()
        blocks.append(("layer_0", block))
        model.layers.named_children = Mock(return_value=blocks)

        moe_model = MockMoEModel(MockBackend(), model, has_lm_head=True)

        _disable_fsdp_for_moe_module(moe_model)

        # Verify all modules are disabled
        assert model._is_last_backward is False
        assert model._reshard_after_backward is False
        assert model._requires_gradient_sync is False
        assert moe_model.lm_head._is_last_backward is False
        assert moe_model.lm_head._reshard_after_backward is False
        assert moe_model.lm_head._requires_gradient_sync is False
        assert block.mlp.experts._is_last_backward is False
        assert block.mlp.experts._reshard_after_backward is False
        assert block.mlp.experts._requires_gradient_sync is False


class TestRunPostBackwardForMoeModule:
    """Test _run_post_backward_for_moe_module helper function."""

    @patch('nemo_automodel.components.moe.fsdp_mixin.fully_shard')
    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_enables_and_runs_post_backward(self, mock_isinstance, mock_fully_shard):
        def isinstance_side_effect(obj, cls):
            if cls.__name__ == 'FSDPModule':
                return isinstance(obj, MockFSDPModule)
            return isinstance(obj, cls)
        mock_isinstance.side_effect = isinstance_side_effect

        model = MockFSDPModule()
        model.layers = Mock()
        blocks = []
        block = Mock()
        block.mlp = Mock()
        block.mlp.experts = MockFSDPModule()
        blocks.append(("layer_0", block))
        model.layers.named_children = Mock(return_value=blocks)

        moe_model = MockMoEModel(MockBackend(), model, has_lm_head=True)

        # Create mock FSDP states
        def create_mock_fsdp_state():
            mock_state = Mock()
            mock_state._fsdp_param_group = Mock()
            mock_state_ctx = Mock()
            mock_state_ctx.all_states = [mock_state]
            mock_fsdp_state = Mock()
            mock_fsdp_state._state_ctx = mock_state_ctx
            mock_fsdp_state._root_post_backward_final_callback = Mock()
            return mock_fsdp_state

        mock_fully_shard.state.side_effect = [
            create_mock_fsdp_state(),
            create_mock_fsdp_state(),
            create_mock_fsdp_state(),
        ]

        _run_post_backward_for_moe_module(moe_model)

        # Verify all modules are enabled
        assert model._is_last_backward is True
        assert model._reshard_after_backward is True
        assert model._requires_gradient_sync is True
        assert moe_model.lm_head._is_last_backward is True
        assert moe_model.lm_head._reshard_after_backward is True
        assert moe_model.lm_head._requires_gradient_sync is True
        assert block.mlp.experts._is_last_backward is True
        assert block.mlp.experts._reshard_after_backward is True
        assert block.mlp.experts._requires_gradient_sync is True


class TestPatchedBackwardMaybeWithNosync:
    """Test patched_backward_maybe_with_nosync function."""

    def test_ddp_last_backward(self):
        """Test DDP path with last_backward=True."""
        from torch.nn.parallel import DistributedDataParallel

        mock_stage = Mock()
        mock_ddp_module = Mock(spec=DistributedDataParallel)
        mock_reducer = Mock()
        mock_ddp_module.reducer = mock_reducer
        mock_stage.submod = mock_ddp_module

        bwd_kwargs = {
            "stage_output": Mock(),
            "output_grads": Mock(),
            "input_values": Mock(),
        }

        with patch('nemo_automodel.components.moe.fsdp_mixin.stage_backward') as mock_stage_backward:
            with patch('nemo_automodel.components.moe.fsdp_mixin.torch.nn.parallel.distributed._find_tensors') as mock_find_tensors:
                mock_find_tensors.return_value = []
                mock_stage_backward.return_value = ((), None)

                result = patched_backward_maybe_with_nosync(
                    mock_stage,
                    "full",
                    bwd_kwargs,
                    last_backward=True
                )

                # Verify prepare_for_backward was called
                mock_reducer.prepare_for_backward.assert_called_once()
                grads, param_groups = result
                assert grads == ((), None)
                assert param_groups is None

    def test_ddp_not_last_backward(self):
        """Test DDP path with last_backward=False."""
        from torch.nn.parallel import DistributedDataParallel
        from unittest.mock import MagicMock

        mock_stage = Mock()
        mock_ddp_module = Mock(spec=DistributedDataParallel)
        # Make no_sync return a context manager
        mock_ddp_module.no_sync = MagicMock()
        mock_ddp_module.no_sync.return_value.__enter__ = MagicMock()
        mock_ddp_module.no_sync.return_value.__exit__ = MagicMock()
        mock_stage.submod = mock_ddp_module

        bwd_kwargs = {
            "stage_output": Mock(),
            "output_grads": Mock(),
            "input_values": Mock(),
        }

        with patch('nemo_automodel.components.moe.fsdp_mixin.stage_backward') as mock_stage_backward:
            mock_stage_backward.return_value = ((), None)

            result = patched_backward_maybe_with_nosync(
                mock_stage,
                "full",
                bwd_kwargs,
                last_backward=False
            )

            # Verify no_sync was used
            mock_ddp_module.no_sync.assert_called_once()
            grads, param_groups = result
            assert grads == ((), None)
            assert param_groups is None

    @patch('nemo_automodel.components.moe.fsdp_mixin.fully_shard')
    def test_fsdp_module_last_backward(self, mock_fully_shard):
        """Test FSDP module path with last_backward=True."""
        mock_stage = Mock()
        mock_fsdp_module = MockFSDPModule()
        mock_stage.submod = mock_fsdp_module

        bwd_kwargs = {
            "stage_output": Mock(),
            "output_grads": Mock(),
            "input_values": Mock(),
        }

        # Create mock FSDP state
        mock_state = Mock()
        mock_state._fsdp_param_group = Mock()
        mock_state_ctx = Mock()
        mock_state_ctx.all_states = [mock_state]
        mock_fsdp_state = Mock()
        mock_fsdp_state._state_ctx = mock_state_ctx
        mock_fsdp_state._root_post_backward_final_callback = Mock()
        mock_fully_shard.state.return_value = mock_fsdp_state

        with patch('nemo_automodel.components.moe.fsdp_mixin.stage_backward') as mock_stage_backward:
            with patch('nemo_automodel.components.moe.fsdp_mixin.isinstance') as mock_isinstance:
                # Make isinstance return True for FSDPModule check
                mock_isinstance.side_effect = lambda obj, cls: cls.__name__ == 'FSDPModule'
                mock_stage_backward.return_value = ((), None)

                result = patched_backward_maybe_with_nosync(
                    mock_stage,
                    "full",
                    bwd_kwargs,
                    last_backward=True
                )

                # Verify post_backward was called
                mock_state._fsdp_param_group.post_backward.assert_called_once()
                mock_fsdp_state._root_post_backward_final_callback.assert_called_once()
                grads, param_groups = result
                assert grads == ((), None)
                assert param_groups is None

    @patch('nemo_automodel.components.moe.fsdp_mixin.get_is_optim_step')
    @patch('nemo_automodel.components.moe.fsdp_mixin.isinstance')
    def test_moe_fsdp_mixin_last_backward_with_optim_step(self, mock_isinstance, mock_get_optim):
        """Test MoEFSDPSyncMixin path with last_backward=True and IS_OPTIM_STEP=True."""
        def isinstance_side_effect(obj, cls):
            if cls == MoEFSDPSyncMixin:
                return True
            if cls.__name__ == 'FSDPModule':
                return isinstance(obj, MockFSDPModule)
            return False
        mock_isinstance.side_effect = isinstance_side_effect
        mock_get_optim.return_value = True

        mock_stage = Mock()
        model = MockFSDPModule()
        moe_model = MockMoEModel(MockBackend(), model)
        mock_stage.submod = moe_model

        bwd_kwargs = {
            "stage_output": Mock(),
            "output_grads": Mock(),
            "input_values": Mock(),
        }

        with patch('nemo_automodel.components.moe.fsdp_mixin.stage_backward') as mock_stage_backward:
            with patch('nemo_automodel.components.moe.fsdp_mixin._run_post_backward_for_moe_module') as mock_run_post:
                mock_stage_backward.return_value = ((), None)

                result = patched_backward_maybe_with_nosync(
                    mock_stage,
                    "full",
                    bwd_kwargs,
                    last_backward=True
                )

                # Verify post backward was called
                mock_run_post.assert_called_once_with(moe_model)
                grads, param_groups = result
                assert grads == ((), None)
                assert param_groups is None

    def test_backward_type_input(self):
        """Test backward_type='input' path."""
        mock_stage = Mock()
        mock_stage.submod = Mock()  # Non-DP module
        mock_stage.submod.parameters.return_value = []

        bwd_kwargs = {
            "stage_output": Mock(),
            "output_grads": Mock(),
            "input_values": Mock(),
        }

        with patch('nemo_automodel.components.moe.fsdp_mixin.stage_backward_input') as mock_backward_input:
            with patch('nemo_automodel.components.moe.fsdp_mixin.isinstance') as mock_isinstance:
                mock_isinstance.return_value = False
                mock_backward_input.return_value = ((), [])

                result = patched_backward_maybe_with_nosync(
                    mock_stage,
                    "input",
                    bwd_kwargs,
                    last_backward=False
                )

                mock_backward_input.assert_called_once()
                grads, param_groups = result
                assert grads == ()
                assert param_groups == []

    def test_backward_type_weight(self):
        """Test backward_type='weight' path."""
        mock_stage = Mock()
        mock_stage.submod = Mock()  # Non-DP module
        mock_stage.submod.parameters.return_value = []

        bwd_kwargs = {
            "param_groups": [],
        }

        with patch('nemo_automodel.components.moe.fsdp_mixin.stage_backward_weight') as mock_backward_weight:
            with patch('nemo_automodel.components.moe.fsdp_mixin.isinstance') as mock_isinstance:
                mock_isinstance.return_value = False
                mock_backward_weight.return_value = ()

                result = patched_backward_maybe_with_nosync(
                    mock_stage,
                    "weight",
                    bwd_kwargs,
                    last_backward=False
                )

                mock_backward_weight.assert_called_once()
                grads, param_groups = result
                assert grads == ()
                assert param_groups is None

    def test_backward_type_invalid(self):
        """Test invalid backward_type raises error."""
        mock_stage = Mock()
        mock_stage.submod = Mock()

        bwd_kwargs = {}

        with patch('nemo_automodel.components.moe.fsdp_mixin.isinstance') as mock_isinstance:
            mock_isinstance.return_value = False

            try:
                patched_backward_maybe_with_nosync(
                    mock_stage,
                    "invalid_type",
                    bwd_kwargs,
                    last_backward=False
                )
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "Unknown backward type" in str(e)
