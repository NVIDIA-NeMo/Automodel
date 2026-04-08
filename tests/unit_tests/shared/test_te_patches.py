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

import torch

import nemo_automodel.shared.te_patches as te_patches_module
from nemo_automodel.shared.te_patches import (
    _apply_fused_adam_quantized_tensor_patch,
    apply_te_patches,
)

# All tests that call _apply_fused_adam_quantized_tensor_patch need to mock
# is_te_min_version since it is called at the top of that function.
_MOCK_TE_VERSION = "nemo_automodel.shared.import_utils.is_te_min_version"


def _build_te_mocks():
    """Build mock TE modules and return (fused_adam_cls, original_method, qt_cls, sys_modules_dict)."""
    qt_cls = type("QuantizedTensor", (), {})

    fused_adam_cls = MagicMock()
    original_method = MagicMock()
    original_method.__name__ = "_initialize_state"
    fused_adam_cls._initialize_state = original_method

    fused_adam_module = MagicMock()
    fused_adam_module.FusedAdam = fused_adam_cls

    qt_module = MagicMock()
    qt_module.QuantizedTensor = qt_cls

    modules = {
        "transformer_engine": MagicMock(),
        "transformer_engine.pytorch": MagicMock(),
        "transformer_engine.pytorch.optimizers": MagicMock(),
        "transformer_engine.pytorch.optimizers.fused_adam": fused_adam_module,
        "transformer_engine.pytorch.quantized_tensor": qt_module,
    }
    return fused_adam_cls, original_method, qt_cls, modules


# Old TE source that uses torch.zeros(param.shape, ...) without the fix
_OLD_TE_SOURCE = (
    "def _initialize_state(self, param, state_name, zero_buffer):\n"
    "    data = torch.zeros(param.shape, dtype=torch.int16, device=param.device)\n"
    "    data = torch.empty(param.shape, dtype=dtype, device=param.device)\n"
)


def _install_patch_and_get_fn(fused_adam_cls, sys_modules):
    """Run _apply_fused_adam_quantized_tensor_patch and return the installed function."""
    with patch(_MOCK_TE_VERSION, return_value=False), patch.dict(
        "sys.modules", sys_modules
    ), patch("inspect.getsource", return_value=_OLD_TE_SOURCE):
        _apply_fused_adam_quantized_tensor_patch()
    return fused_adam_cls._initialize_state


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
        fused_adam_cls, original_method, _, sys_modules = _build_te_mocks()

        with patch.dict("sys.modules", sys_modules):
            _apply_fused_adam_quantized_tensor_patch()

        # Should NOT patch when TE >= 2.12
        assert fused_adam_cls._initialize_state is original_method

    @patch(_MOCK_TE_VERSION, return_value=False)
    def test_skips_when_te_not_installed(self, _mock_ver):
        with patch.dict("sys.modules", {
            "transformer_engine": None,
            "transformer_engine.pytorch": None,
            "transformer_engine.pytorch.optimizers": None,
            "transformer_engine.pytorch.optimizers.fused_adam": None,
            "transformer_engine.pytorch.quantized_tensor": None,
        }):
            _apply_fused_adam_quantized_tensor_patch()

    @patch(_MOCK_TE_VERSION, return_value=False)
    def test_patches_fused_adam_when_te_available(self, _mock_ver):
        fused_adam_cls, original_method, _, sys_modules = _build_te_mocks()

        with patch.dict("sys.modules", sys_modules), patch(
            "inspect.getsource", return_value=_OLD_TE_SOURCE
        ):
            _apply_fused_adam_quantized_tensor_patch()

        assert fused_adam_cls._initialize_state is not original_method
        assert callable(fused_adam_cls._initialize_state)

    @patch(_MOCK_TE_VERSION, return_value=False)
    def test_patches_when_only_partial_upstream_fix(self, _mock_ver):
        fused_adam_cls, original_method, _, sys_modules = _build_te_mocks()

        partial_source = (
            "def _initialize_state(self, param):\n"
            "    # QuantizedTensor mentioned in a comment\n"
            "    data = torch.zeros(param.shape, dtype=torch.int16)\n"
        )

        with patch.dict("sys.modules", sys_modules), patch(
            "inspect.getsource", return_value=partial_source
        ):
            _apply_fused_adam_quantized_tensor_patch()

        assert fused_adam_cls._initialize_state is not original_method

    @patch(_MOCK_TE_VERSION, return_value=False)
    def test_skips_patch_when_already_handled_upstream(self, _mock_ver):
        fused_adam_cls, original_method, _, sys_modules = _build_te_mocks()

        upstream_fixed_source = (
            "def _initialize_state(self, param, state_name, zero_buffer):\n"
            "    dtype = self.name_to_dtype_map[state_name]\n"
            "    param_for_empty = param.dequantize() if isinstance(param, QuantizedTensor) else param\n"
            "    if store_param_remainders:\n"
            "        data = torch.zeros_like(param_for_empty, dtype=torch.int16)\n"
            "    else:\n"
            "        data = torch.empty_like(param_for_empty, dtype=dtype)\n"
        )

        with patch.dict("sys.modules", sys_modules), patch(
            "inspect.getsource", return_value=upstream_fixed_source
        ):
            _apply_fused_adam_quantized_tensor_patch()

        assert fused_adam_cls._initialize_state is original_method


class TestPatchedInitializeStateBehavior:
    """Test the actual behavior of the patched _initialize_state function."""

    def _get_patched_fn(self):
        fused_adam_cls, _, _, sys_modules = _build_te_mocks()
        return _install_patch_and_get_fn(fused_adam_cls, sys_modules)

    def _make_optimizer_self(self, state_dtype=torch.float32):
        opt = MagicMock()
        opt.name_to_dtype_map = {"exp_avg": state_dtype}
        opt.state = {}
        opt._scales = {}
        return opt

    def test_regular_param_zero_buffer(self):
        fn = self._get_patched_fn()
        opt = self._make_optimizer_self()
        param = torch.nn.Parameter(torch.randn(4, 4))
        opt.state[param] = {}

        fn(opt, param, "exp_avg", zero_buffer=True)

        result = opt.state[param]["exp_avg"]
        assert result.shape == (4, 4)
        assert result.dtype == torch.float32
        assert torch.all(result == 0)

    def test_regular_param_no_zero_buffer(self):
        fn = self._get_patched_fn()
        opt = self._make_optimizer_self()
        param = torch.nn.Parameter(torch.randn(3, 5))
        opt.state[param] = {}

        fn(opt, param, "exp_avg", zero_buffer=False)

        result = opt.state[param]["exp_avg"]
        assert result.shape == (3, 5)
        assert result.dtype == torch.float32

    def test_quantized_tensor_param_dequantizes(self):
        fn = self._get_patched_fn()
        opt = self._make_optimizer_self()

        # Create a mock QuantizedTensor that has a dequantize method
        fused_adam_cls, _, qt_cls, sys_modules = _build_te_mocks()
        patched_fn = _install_patch_and_get_fn(fused_adam_cls, sys_modules)

        dequantized = torch.randn(2, 3)
        param = MagicMock(spec=["dequantize", "device", "shape"])
        param.dequantize.return_value = dequantized
        param.device = torch.device("cpu")
        param.shape = (2, 3)
        # Make isinstance(param, QuantizedTensor) return True
        param.__class__ = qt_cls

        opt.state[param] = {}

        patched_fn(opt, param, "exp_avg", zero_buffer=True)

        param.dequantize.assert_called_once()
        result = opt.state[param]["exp_avg"]
        assert result.shape == (2, 3)
        assert result.dtype == torch.float32

    def test_store_param_remainders(self):
        fn = self._get_patched_fn()
        opt = self._make_optimizer_self()
        param = torch.nn.Parameter(torch.randn(4, 4))
        opt.state[param] = {}

        fn(opt, param, "exp_avg", zero_buffer=False, store_param_remainders=True)

        result = opt.state[param]["exp_avg"]
        assert result.dtype == torch.int16
        assert result.shape == (4, 4)
        assert torch.all(result == 0)  # zeros_like for remainders

    def test_non_float32_creates_scale(self):
        fn = self._get_patched_fn()
        opt = self._make_optimizer_self(state_dtype=torch.float16)
        param = torch.nn.Parameter(torch.randn(2, 2))
        opt.state[param] = {}

        fn(opt, param, "exp_avg", zero_buffer=True)

        assert param in opt._scales
        assert "exp_avg" in opt._scales[param]
        scale = opt._scales[param]["exp_avg"]
        assert scale.shape == (1,)
        assert scale.dtype == torch.float32
        assert scale.item() == 1.0

    def test_float32_does_not_create_scale(self):
        fn = self._get_patched_fn()
        opt = self._make_optimizer_self(state_dtype=torch.float32)
        param = torch.nn.Parameter(torch.randn(2, 2))
        opt.state[param] = {}

        fn(opt, param, "exp_avg", zero_buffer=True)

        assert param not in opt._scales

    def test_uint8_dtype_uses_fp8_quantizer(self):
        fn = self._get_patched_fn()
        opt = self._make_optimizer_self(state_dtype=torch.uint8)
        param = torch.nn.Parameter(torch.randn(2, 2))
        opt.state[param] = {}

        mock_quantized_output = MagicMock()
        mock_quantizer = MagicMock()
        mock_quantizer.make_empty.return_value = mock_quantized_output

        mock_float8_quantizer = MagicMock(return_value=mock_quantizer)
        mock_tex = MagicMock()

        with patch.dict("sys.modules", {
            "transformer_engine_torch": mock_tex,
            "transformer_engine": MagicMock(),
            "transformer_engine.pytorch": MagicMock(),
            "transformer_engine.pytorch.tensor": MagicMock(),
            "transformer_engine.pytorch.tensor.float8_tensor": MagicMock(Float8Quantizer=mock_float8_quantizer),
        }):
            fn(opt, param, "exp_avg", zero_buffer=True)

        mock_float8_quantizer.assert_called_once()
        mock_quantizer.make_empty.assert_called_once_with((2, 2))
        mock_quantized_output.quantize_.assert_called_once()
