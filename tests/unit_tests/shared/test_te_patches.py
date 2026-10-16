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

import copy
from unittest.mock import MagicMock, patch

import pytest
import torch

import nemo_automodel.shared.te_patches as te_patches_module
from nemo_automodel.shared.te_patches import (
    _apply_fused_adam_empty_shard_patch,
    _apply_fused_adam_in_place_load_patch,
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
    @patch.object(te_patches_module, "_apply_fused_adam_empty_shard_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_in_place_load_patch")
    def test_apply_te_patches_calls_fused_adam_patch(self, mock_load_patch, mock_empty_patch, mock_patch_fn):
        apply_te_patches()
        mock_patch_fn.assert_called_once()
        mock_empty_patch.assert_called_once()
        mock_load_patch.assert_called_once()

    @patch.object(te_patches_module, "_apply_fused_adam_quantized_tensor_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_empty_shard_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_in_place_load_patch")
    def test_apply_te_patches_idempotent(self, mock_load_patch, mock_empty_patch, mock_patch_fn):
        apply_te_patches()
        apply_te_patches()
        mock_patch_fn.assert_called_once()
        mock_empty_patch.assert_called_once()
        mock_load_patch.assert_called_once()

    @patch.object(te_patches_module, "_apply_fused_adam_quantized_tensor_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_empty_shard_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_in_place_load_patch")
    def test_apply_te_patches_sets_flag(self, mock_load_patch, mock_empty_patch, mock_patch_fn):
        assert not te_patches_module._TE_PATCHES_APPLIED
        apply_te_patches()
        assert te_patches_module._TE_PATCHES_APPLIED


class TestFusedAdamEmptyShardPatch:
    def _patched_applier(self):
        _, _, _, modules = _build_te_mocks()
        fused_adam = modules["transformer_engine.pytorch.optimizers.fused_adam"]
        original = MagicMock(return_value="updated")
        fused_adam.multi_tensor_applier = original
        modules["transformer_engine.pytorch.optimizers"].fused_adam = fused_adam
        with patch.dict("sys.modules", modules):
            _apply_fused_adam_empty_shard_patch()
            _apply_fused_adam_empty_shard_patch()
        return fused_adam.multi_tensor_applier, original

    def test_filters_last_empty_slot_across_all_five_lists(self):
        applier, original = self._patched_applier()
        columns = [[torch.ones(1) for _ in range(29)] + [torch.empty(0)] for _ in range(5)]
        result = applier("kernel", "overflow", columns, 1e-5)

        assert result == "updated"
        original.assert_called_once()
        sent = original.call_args.args[2]
        assert len(sent) == 5
        assert all(len(column) == 29 for column in sent)
        assert all(tensor.numel() == 1 for column in sent for tensor in column)
        assert all(len(column) == 30 for column in columns)

    def test_skips_all_empty_shards(self):
        applier, original = self._patched_applier()
        assert applier("kernel", "overflow", [[torch.empty(0)] for _ in range(5)]) is None
        original.assert_not_called()

    def test_checks_local_dtensor_shard_instead_of_global_shape(self):
        applier, original = self._patched_applier()

        class Shard:
            def __init__(self):
                self._local_tensor = torch.empty(0)

            def numel(self):
                return 8

        assert applier("kernel", "overflow", [[Shard()] for _ in range(5)]) is None
        original.assert_not_called()

    def test_keeps_nonempty_input_unchanged(self):
        applier, original = self._patched_applier()
        columns = [[torch.ones(1)] for _ in range(5)]
        assert applier("kernel", "overflow", columns) == "updated"
        assert original.call_args.args[2] is columns

    def test_rejects_mismatched_nonempty_parameter(self):
        applier, original = self._patched_applier()
        columns = [[torch.empty(0)], [torch.ones(1)], *[[torch.empty(0)] for _ in range(3)]]
        with pytest.raises(RuntimeError, match="empty gradient"):
            applier("kernel", "overflow", columns)
        original.assert_not_called()


class TestFusedAdamInPlaceLoadPatch:
    def test_installs_once(self):
        _, _, _, modules = _build_te_mocks()
        fused_adam = modules["transformer_engine.pytorch.optimizers.fused_adam"]
        original = MagicMock()
        fused_adam.FusedAdam.load_state_dict = original
        modules["transformer_engine.pytorch.optimizers"].fused_adam = fused_adam
        with patch.dict("sys.modules", modules):
            _apply_fused_adam_in_place_load_patch()
            installed = fused_adam.FusedAdam.load_state_dict
            _apply_fused_adam_in_place_load_patch()

        assert installed is not original
        assert installed._nemo_in_place_load_patch is True
        assert fused_adam.FusedAdam.load_state_dict is installed

    def test_skips_without_transformer_engine(self):
        with patch.dict("sys.modules", {"transformer_engine": None}):
            _apply_fused_adam_in_place_load_patch()


@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_adam_in_place_load_resumes_exactly_without_new_state_memory():
    """Resume a FusedAdam with fp32 masters (bf16 parameter + int16 remainder) and fp32 moments from a checkpoint copy.

    The load must reuse the state buffers that already exist, as they do after torch DCP initializes the optimizer
    state, allocate no new memory, and continue bit-identically to the uninterrupted optimizer.
    """
    fused_adam_module = pytest.importorskip("transformer_engine.pytorch.optimizers.fused_adam")
    original_load = fused_adam_module.FusedAdam.load_state_dict
    try:
        _apply_fused_adam_in_place_load_patch()
        torch.manual_seed(0)

        def make(params):
            return fused_adam_module.FusedAdam(
                params,
                lr=1e-2,
                betas=(0.9, 0.95),
                weight_decay=0.0,
                master_weights=True,
                master_weight_dtype=torch.float32,
                store_param_remainders=True,
                exp_avg_dtype=torch.float32,
                exp_avg_sq_dtype=torch.float32,
            )

        ref_params = [
            torch.nn.Parameter(torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)),
            torch.nn.Parameter(torch.randn(256, device="cuda", dtype=torch.float32)),
        ]
        ref = make(ref_params)
        grads = [[torch.randn_like(p) for p in ref_params] for _ in range(3)]
        for step_grads in grads[:2]:
            for param, grad in zip(ref_params, step_grads):
                param.grad = grad.clone()
            ref.step()
        saved = copy.deepcopy(ref.state_dict())

        res_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
        res = make(res_params)
        for param in res_params:
            param.grad = torch.zeros_like(param)
        res.param_groups[0]["lr"] = 0.0
        res.step()  # allocate the state buffers, as torch DCP does before loading into them
        res.zero_grad(set_to_none=True)
        buffers = {(i, name): t.data_ptr() for i, p in enumerate(res_params) for name, t in res.state[p].items()}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        res.load_state_dict(saved)
        torch.cuda.synchronize()
        peak_growth = torch.cuda.max_memory_allocated() - before

        assert {
            (i, name): t.data_ptr() for i, p in enumerate(res_params) for name, t in res.state[p].items()
        } == buffers
        assert peak_growth < 2**20, f"load allocated {peak_growth} bytes"
        assert res.param_groups[0]["lr"] == 1e-2
        assert res.param_groups[0]["step"] == ref.param_groups[0]["step"] == 2
        for p_ref, p_res in zip(ref_params, res_params):
            assert res.state[p_res].keys() == ref.state[p_ref].keys()
            for name, value in ref.state[p_ref].items():
                torch.testing.assert_close(res.state[p_res][name], value, rtol=0, atol=0)
        for params, opt in ((ref_params, ref), (res_params, res)):
            for param, grad in zip(params, grads[2]):
                param.grad = grad.clone()
            opt.step()
        for p_ref, p_res in zip(ref_params, res_params):
            torch.testing.assert_close(p_res, p_ref, rtol=0, atol=0)
    finally:
        fused_adam_module.FusedAdam.load_state_dict = original_load


@pytest.mark.parametrize("error_type", ["FileNotFoundError", "OSError", "ImportError"])
def test_apply_te_patches_tolerates_missing_native_extension(tmp_path, monkeypatch, error_type):
    """A successful top-level TE import does not guarantee its torch extension loads."""
    import importlib
    import sys

    package = tmp_path / "transformer_engine"
    (package / "pytorch").mkdir(parents=True)
    (package / "__init__.py").write_text(
        'import warnings\nwarnings.warn("PyTorch extension is unavailable", UserWarning)\n'
    )
    (package / "pytorch" / "__init__.py").write_text(
        f'raise {error_type}("Could not load Transformer Engine torch lib")\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with patch.dict(sys.modules):
        for name in list(sys.modules):
            if name == "transformer_engine" or name.startswith("transformer_engine."):
                del sys.modules[name]
        with pytest.warns(UserWarning, match="PyTorch extension is unavailable"):
            importlib.import_module("transformer_engine")
        monkeypatch.setattr(te_patches_module, "_TE_PATCHES_APPLIED", False)
        # TE >= 2.12 takes the existing version-gated patch's early return.
        with patch(_MOCK_TE_VERSION, return_value=True):
            apply_te_patches()
        assert te_patches_module._TE_PATCHES_APPLIED
        assert "transformer_engine.pytorch.optimizers.fused_adam" not in sys.modules


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
