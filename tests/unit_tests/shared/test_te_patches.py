# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

import nemo_automodel.shared.te_patches as te_patches_module
from nemo_automodel.shared.te_patches import (
    _apply_fused_adam_quantized_tensor_patch,
    apply_te_patches,
)

# All tests that call _apply_fused_adam_quantized_tensor_patch need to mock
# is_te_min_version since it is called at the top of that function.
_MOCK_TE_VERSION = "nemo_automodel.shared.import_utils.is_te_min_version"


class TestApplyTePatchesIdempotent:
    def setup_method(self):
        te_patches_module._TE_PATCHES_APPLIED = False

    def teardown_method(self):
        te_patches_module._TE_PATCHES_APPLIED = False

    @patch.object(te_patches_module, "_apply_fused_adam_quantized_tensor_patch")
    def test_apply_te_patches_calls_fused_adam_patch(self, mock_patch_fn):
        apply_te_patches()
        mock_patch_fn.assert_called_once()

    @patch.object(te_patches_module, "_apply_fused_adam_quantized_tensor_patch")
    def test_apply_te_patches_idempotent(self, mock_patch_fn):
        apply_te_patches()
        apply_te_patches()
        mock_patch_fn.assert_called_once()

    @patch.object(te_patches_module, "_apply_fused_adam_quantized_tensor_patch")
    def test_apply_te_patches_sets_flag(self, mock_patch_fn):
        assert not te_patches_module._TE_PATCHES_APPLIED
        apply_te_patches()
        assert te_patches_module._TE_PATCHES_APPLIED


class TestFusedAdamQuantizedTensorPatch:
    def teardown_method(self):
        te_patches_module._TE_PATCHES_APPLIED = False

    @patch(_MOCK_TE_VERSION, return_value=True)
    def test_skips_when_te_version_ge_2_12(self, _mock_ver):
        mock_fused_adam_cls = MagicMock()
        original_method = MagicMock()
        mock_fused_adam_cls._initialize_state = original_method

        mock_fused_adam_module = MagicMock()
        mock_fused_adam_module.FusedAdam = mock_fused_adam_cls

        with patch.dict("sys.modules", {
            "transformer_engine": MagicMock(),
            "transformer_engine.pytorch": MagicMock(),
            "transformer_engine.pytorch.optimizers": MagicMock(),
            "transformer_engine.pytorch.optimizers.fused_adam": mock_fused_adam_module,
            "transformer_engine.pytorch.quantized_tensor": MagicMock(),
        }):
            _apply_fused_adam_quantized_tensor_patch()

        # Should NOT patch when TE >= 2.12
        assert mock_fused_adam_cls._initialize_state is original_method

    @patch(_MOCK_TE_VERSION, return_value=False)
    def test_skips_when_te_not_installed(self, _mock_ver):
        with patch.dict("sys.modules", {
            "transformer_engine": None,
            "transformer_engine.pytorch": None,
            "transformer_engine.pytorch.optimizers": None,
            "transformer_engine.pytorch.optimizers.fused_adam": None,
            "transformer_engine.pytorch.quantized_tensor": None,
        }):
            # Should not raise when TE is not importable
            _apply_fused_adam_quantized_tensor_patch()

    @patch(_MOCK_TE_VERSION, return_value=False)
    def test_patches_fused_adam_when_te_available(self, _mock_ver):
        mock_quantized_tensor_cls = type("QuantizedTensor", (), {})

        mock_fused_adam_cls = MagicMock()
        # Source without the upstream fix lines means patch should apply
        original_method = MagicMock()
        original_method.__name__ = "_initialize_state"
        mock_fused_adam_cls._initialize_state = original_method

        mock_fused_adam_module = MagicMock()
        mock_fused_adam_module.FusedAdam = mock_fused_adam_cls

        mock_qt_module = MagicMock()
        mock_qt_module.QuantizedTensor = mock_quantized_tensor_cls

        # Old TE source that uses torch.zeros(param.shape, ...) without the fix
        old_source = (
            "def _initialize_state(self, param, state_name, zero_buffer):\n"
            "    data = torch.zeros(param.shape, dtype=torch.int16, device=param.device)\n"
            "    data = torch.empty(param.shape, dtype=dtype, device=param.device)\n"
        )

        with patch.dict("sys.modules", {
            "transformer_engine": MagicMock(),
            "transformer_engine.pytorch": MagicMock(),
            "transformer_engine.pytorch.optimizers": MagicMock(),
            "transformer_engine.pytorch.optimizers.fused_adam": mock_fused_adam_module,
            "transformer_engine.pytorch.quantized_tensor": mock_qt_module,
        }), patch("inspect.getsource", return_value=old_source):
            _apply_fused_adam_quantized_tensor_patch()

        # Verify _initialize_state was replaced
        assert mock_fused_adam_cls._initialize_state is not original_method
        assert callable(mock_fused_adam_cls._initialize_state)

    @patch(_MOCK_TE_VERSION, return_value=False)
    def test_patches_when_only_partial_upstream_fix(self, _mock_ver):
        mock_fused_adam_cls = MagicMock()
        original_method = MagicMock()
        original_method.__name__ = "_initialize_state"
        mock_fused_adam_cls._initialize_state = original_method

        mock_fused_adam_module = MagicMock()
        mock_fused_adam_module.FusedAdam = mock_fused_adam_cls

        mock_qt_module = MagicMock()
        mock_qt_module.QuantizedTensor = type("QuantizedTensor", (), {})

        # Source mentions QuantizedTensor but is missing the full fix lines
        partial_source = (
            "def _initialize_state(self, param):\n"
            "    # QuantizedTensor mentioned in a comment\n"
            "    data = torch.zeros(param.shape, dtype=torch.int16)\n"
        )

        with patch.dict("sys.modules", {
            "transformer_engine": MagicMock(),
            "transformer_engine.pytorch": MagicMock(),
            "transformer_engine.pytorch.optimizers": MagicMock(),
            "transformer_engine.pytorch.optimizers.fused_adam": mock_fused_adam_module,
            "transformer_engine.pytorch.quantized_tensor": mock_qt_module,
        }), patch("inspect.getsource", return_value=partial_source):
            _apply_fused_adam_quantized_tensor_patch()

        # Should still patch since the full upstream fix is not present
        assert mock_fused_adam_cls._initialize_state is not original_method

    @patch(_MOCK_TE_VERSION, return_value=False)
    def test_skips_patch_when_already_handled_upstream(self, _mock_ver):
        mock_fused_adam_cls = MagicMock()
        original_method = MagicMock()
        original_method.__name__ = "_initialize_state"
        mock_fused_adam_cls._initialize_state = original_method

        mock_fused_adam_module = MagicMock()
        mock_fused_adam_module.FusedAdam = mock_fused_adam_cls

        mock_qt_module = MagicMock()

        # Simulate upstream source that contains all three fix lines from TE PR #2535
        upstream_fixed_source = (
            "def _initialize_state(self, param, state_name, zero_buffer):\n"
            "    dtype = self.name_to_dtype_map[state_name]\n"
            "    param_for_empty = param.dequantize() if isinstance(param, QuantizedTensor) else param\n"
            "    if store_param_remainders:\n"
            "        data = torch.zeros_like(param_for_empty, dtype=torch.int16)\n"
            "    else:\n"
            "        data = torch.empty_like(param_for_empty, dtype=dtype)\n"
        )

        with patch.dict("sys.modules", {
            "transformer_engine": MagicMock(),
            "transformer_engine.pytorch": MagicMock(),
            "transformer_engine.pytorch.optimizers": MagicMock(),
            "transformer_engine.pytorch.optimizers.fused_adam": mock_fused_adam_module,
            "transformer_engine.pytorch.quantized_tensor": mock_qt_module,
        }), patch("inspect.getsource", return_value=upstream_fixed_source):
            _apply_fused_adam_quantized_tensor_patch()

        # Should NOT have been replaced since upstream already has the full fix
        assert mock_fused_adam_cls._initialize_state is original_method
