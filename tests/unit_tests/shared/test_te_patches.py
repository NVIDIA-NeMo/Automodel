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

from unittest.mock import MagicMock, patch

import nemo_automodel.shared.te_patches as te_patches_module
from nemo_automodel.shared.te_patches import (
    _apply_fused_adam_quantized_tensor_patch,
    apply_te_patches,
)


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

    def test_skips_when_te_not_installed(self):
        with patch.dict("sys.modules", {
            "transformer_engine": None,
            "transformer_engine.pytorch": None,
            "transformer_engine.pytorch.optimizers": None,
            "transformer_engine.pytorch.optimizers.fused_adam": None,
            "transformer_engine.pytorch.quantized_tensor": None,
        }):
            # Should not raise when TE is not importable
            _apply_fused_adam_quantized_tensor_patch()

    def test_patches_fused_adam_when_te_available(self):
        mock_quantized_tensor_cls = type("QuantizedTensor", (), {})

        mock_fused_adam_cls = MagicMock()
        # _initialize_state source without "QuantizedTensor" means patch should apply
        original_method = MagicMock()
        original_method.__name__ = "_initialize_state"
        mock_fused_adam_cls._initialize_state = original_method

        mock_fused_adam_module = MagicMock()
        mock_fused_adam_module.FusedAdam = mock_fused_adam_cls

        mock_qt_module = MagicMock()
        mock_qt_module.QuantizedTensor = mock_quantized_tensor_cls

        with patch.dict("sys.modules", {
            "transformer_engine": MagicMock(),
            "transformer_engine.pytorch": MagicMock(),
            "transformer_engine.pytorch.optimizers": MagicMock(),
            "transformer_engine.pytorch.optimizers.fused_adam": mock_fused_adam_module,
            "transformer_engine.pytorch.quantized_tensor": mock_qt_module,
        }), patch("inspect.getsource", return_value="def _initialize_state(self, param): pass"):
            _apply_fused_adam_quantized_tensor_patch()

        # Verify _initialize_state was replaced
        assert mock_fused_adam_cls._initialize_state is not original_method
        assert callable(mock_fused_adam_cls._initialize_state)

    def test_skips_patch_when_already_handled_upstream(self):
        mock_fused_adam_cls = MagicMock()
        original_method = MagicMock()
        original_method.__name__ = "_initialize_state"
        mock_fused_adam_cls._initialize_state = original_method

        mock_fused_adam_module = MagicMock()
        mock_fused_adam_module.FusedAdam = mock_fused_adam_cls

        mock_qt_module = MagicMock()

        with patch.dict("sys.modules", {
            "transformer_engine": MagicMock(),
            "transformer_engine.pytorch": MagicMock(),
            "transformer_engine.pytorch.optimizers": MagicMock(),
            "transformer_engine.pytorch.optimizers.fused_adam": mock_fused_adam_module,
            "transformer_engine.pytorch.quantized_tensor": mock_qt_module,
        }), patch(
            "inspect.getsource",
            return_value="def _initialize_state(self, param): QuantizedTensor handling here",
        ):
            _apply_fused_adam_quantized_tensor_patch()

        # Should NOT have been replaced since upstream already handles it
        assert mock_fused_adam_cls._initialize_state is original_method
