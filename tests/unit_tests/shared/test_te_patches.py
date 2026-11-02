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

import collections
import copy
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

import nemo_automodel.shared.te_patches as te_patches_module
from nemo_automodel.shared.te_patches import (
    _apply_fused_adam_cached_lists_patch,
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
    with (
        patch(_MOCK_TE_VERSION, return_value=False),
        patch.dict("sys.modules", sys_modules),
        patch("inspect.getsource", return_value=_OLD_TE_SOURCE),
    ):
        _apply_fused_adam_quantized_tensor_patch()
    return fused_adam_cls._initialize_state


class TestApplyTePatchesIdempotent:
    def setup_method(self):
        te_patches_module._TE_PATCHES_APPLIED = False

    def teardown_method(self):
        te_patches_module._TE_PATCHES_APPLIED = False

    @patch.object(te_patches_module, "_apply_fused_adam_cached_lists_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_quantized_tensor_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_empty_shard_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_in_place_load_patch")
    def test_apply_te_patches_calls_fused_adam_patch(
        self, mock_load_patch, mock_empty_patch, mock_patch_fn, mock_cached_patch
    ):
        apply_te_patches()
        mock_patch_fn.assert_called_once()
        mock_empty_patch.assert_called_once()
        mock_load_patch.assert_called_once()
        mock_cached_patch.assert_called_once()

    @patch.object(te_patches_module, "_apply_fused_adam_cached_lists_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_quantized_tensor_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_empty_shard_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_in_place_load_patch")
    def test_apply_te_patches_idempotent(self, mock_load_patch, mock_empty_patch, mock_patch_fn, mock_cached_patch):
        apply_te_patches()
        apply_te_patches()
        mock_patch_fn.assert_called_once()
        mock_empty_patch.assert_called_once()
        mock_load_patch.assert_called_once()
        mock_cached_patch.assert_called_once()

    @patch.object(te_patches_module, "_apply_fused_adam_cached_lists_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_quantized_tensor_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_empty_shard_patch")
    @patch.object(te_patches_module, "_apply_fused_adam_in_place_load_patch")
    def test_apply_te_patches_sets_flag(self, mock_load_patch, mock_empty_patch, mock_patch_fn, mock_cached_patch):
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
        with patch.dict(
            "sys.modules",
            {
                "transformer_engine": None,
                "transformer_engine.pytorch": None,
                "transformer_engine.pytorch.optimizers": None,
                "transformer_engine.pytorch.optimizers.fused_adam": None,
                "transformer_engine.pytorch.quantized_tensor": None,
            },
        ):
            _apply_fused_adam_quantized_tensor_patch()

    @patch(_MOCK_TE_VERSION, return_value=False)
    def test_patches_fused_adam_when_te_available(self, _mock_ver):
        fused_adam_cls, original_method, _, sys_modules = _build_te_mocks()

        with patch.dict("sys.modules", sys_modules), patch("inspect.getsource", return_value=_OLD_TE_SOURCE):
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

        with patch.dict("sys.modules", sys_modules), patch("inspect.getsource", return_value=partial_source):
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

        with patch.dict("sys.modules", sys_modules), patch("inspect.getsource", return_value=upstream_fixed_source):
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
        self._get_patched_fn()
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

        with patch.dict(
            "sys.modules",
            {
                "transformer_engine_torch": mock_tex,
                "transformer_engine": MagicMock(),
                "transformer_engine.pytorch": MagicMock(),
                "transformer_engine.pytorch.tensor": MagicMock(),
                "transformer_engine.pytorch.tensor.float8_tensor": MagicMock(Float8Quantizer=mock_float8_quantizer),
            },
        ):
            fn(opt, param, "exp_avg", zero_buffer=True)

        mock_float8_quantizer.assert_called_once()
        mock_quantizer.make_empty.assert_called_once_with((2, 2))
        mock_quantized_output.quantize_.assert_called_once()


# ----------------------------------------------------------------------------------------- cached lists


class _FakeFusedAdam:
    """TE ``FusedAdam`` reduced to the attributes and the per-parameter loop the cached-lists patch replays.

    ``step`` is the reference: the bf16 / fp32 branches of the original loop, calling the module-level
    ``multi_tensor_applier`` with the same lists in the same order. ``original_calls`` counts how often it ran.
    """

    def __init__(self, params, *, master_weights=True, store_param_remainders=True, capturable=False):
        self.param_groups = [
            {
                "params": list(params),
                "lr": 1e-3,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.1,
                "bias_correction": True,
            }
        ]
        self.state = collections.defaultdict(dict)
        self.capturable = capturable
        self.use_decoupled_grad = False
        self.master_weights = master_weights
        self.store_param_remainders = store_param_remainders and master_weights
        self.fuse_unscale = True
        self.name_to_dtype_map = {
            "exp_avg": torch.bfloat16,
            "exp_avg_sq": torch.bfloat16,
            "master_param": torch.float32,
        }
        self.adam_w_mode = 1
        self._dummy_overflow_buf = torch.zeros(1, dtype=torch.int)
        self.multi_tensor_adam = "multi_tensor_adam"
        self.multi_tensor_adam_param_remainder = "multi_tensor_adam_param_remainder"
        self.original_calls = 0
        self.loaded = 0

    def initialize_state(self, param, store_param_remainders):
        state = self.state[param]
        state["exp_avg"] = torch.zeros(param.shape, dtype=torch.bfloat16)
        state["exp_avg_sq"] = torch.zeros(param.shape, dtype=torch.bfloat16)
        if self.master_weights and param.dtype != torch.float32:
            if store_param_remainders:
                state["master_param"] = torch.zeros(param.shape, dtype=torch.int16)
            else:
                state["master_param"] = param.detach().float()

    def step(self, closure=None, grad_scaler=None):
        self.original_calls += 1
        for group in self.param_groups:
            # TE 2.19 order: the step counter advances before an empty group is skipped
            group["step"] = group["step"] + 1 if "step" in group else 1
            if len(group["params"]) == 0:
                continue
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]
            p_f16, g_f16, m_f16, v_f16, main_f16 = [], [], [], [], []
            p_f32, g_f32, m_f32, v_f32 = [], [], [], []
            has_fp16 = has_bf16 = False
            for param in group["params"]:
                state = self.state[param]
                remainders = self.store_param_remainders and param.dtype == torch.bfloat16
                if len(state) == 0:
                    self.initialize_state(param, remainders)
                grad = param.grad
                if grad is None:
                    continue
                unscaled = {name: state[name] for name in ("exp_avg", "exp_avg_sq", "master_param") if name in state}
                if param.dtype in (torch.float16, torch.bfloat16):
                    has_fp16 = has_fp16 or param.dtype == torch.float16
                    has_bf16 = has_bf16 or param.dtype == torch.bfloat16
                    p_f16.append(param.data)
                    if self.master_weights:
                        main_f16.append(unscaled["master_param"].data)
                    g_f16.append(grad.data)
                    m_f16.append(unscaled["exp_avg"])
                    v_f16.append(unscaled["exp_avg_sq"])
                else:
                    p_f32.append(param.data)
                    g_f32.append(grad.data)
                    m_f32.append(unscaled["exp_avg"])
                    v_f32.append(unscaled["exp_avg_sq"])
            args = (
                group["lr"],
                beta1,
                beta2,
                group["eps"],
                group["step"],
                self.adam_w_mode,
                bias_correction,
                group["weight_decay"],
            )
            module = sys.modules["transformer_engine.pytorch.optimizers.fused_adam"]
            if p_f16:
                if self.master_weights:
                    lists = [g_f16, p_f16, m_f16, v_f16, main_f16]
                    func = (
                        self.multi_tensor_adam_param_remainder
                        if self.store_param_remainders and has_bf16 and not has_fp16
                        else self.multi_tensor_adam
                    )
                else:
                    lists = [g_f16, p_f16, m_f16, v_f16]
                    func = self.multi_tensor_adam
                module.multi_tensor_applier(func, self._dummy_overflow_buf, lists, *args)
            if p_f32:
                module.multi_tensor_applier(
                    self.multi_tensor_adam, self._dummy_overflow_buf, [g_f32, p_f32, m_f32, v_f32], *args
                )
        return None

    def load_state_dict(self, state_dict):
        self.loaded += 1

    def add_param_group(self, param_group):
        self.param_groups.append(param_group)


def _fake_params(seed=0):
    torch.manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(4, 3).to(torch.bfloat16)),
        torch.nn.Parameter(torch.randn(5).to(torch.bfloat16)),
        torch.nn.Parameter(torch.randn(2, 2)),
        torch.nn.Parameter(torch.randn(3)),
    ]


def _set_grads(params, seed, skip=()):
    torch.manual_seed(seed)
    for index, param in enumerate(params):
        param.grad = None if index in skip else torch.randn_like(param)


def _snapshot(calls):
    """(func, args, [[(data_ptr, clone)…] per column]) per applier call: identity and values."""
    return [
        (func, args, [[(t.data_ptr(), t.detach().clone()) for t in column] for column in lists])
        for func, _, lists, *args in calls
    ]


class TestFusedAdamCachedListsPatch:
    def _patched_module(self, monkeypatch):
        module = types.ModuleType("transformer_engine.pytorch.optimizers.fused_adam")
        calls = []

        def applier(func, noop, lists, *args):
            calls.append((func, noop, [list(column) for column in lists], *args))

        module.multi_tensor_applier = applier

        class FusedAdam(_FakeFusedAdam):
            pass

        module.FusedAdam = FusedAdam
        monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.optimizers.fused_adam", module)
        monkeypatch.setattr(te_patches_module, "safe_import_te", lambda: (True, MagicMock()), raising=False)
        monkeypatch.setattr(
            "nemo_automodel.shared.import_utils.safe_import_te", lambda: (True, MagicMock()), raising=False
        )
        _apply_fused_adam_cached_lists_patch()
        _apply_fused_adam_cached_lists_patch()  # idempotent: the second call keeps the first wrapper
        assert getattr(FusedAdam.step, "_nemo_cached_lists_patch", False) is True
        assert FusedAdam.step._nemo_original_step is _FakeFusedAdam.step
        return module, FusedAdam, calls

    def _run(self, optimizer, params, steps, calls, skips=None):
        out = []
        for step in range(steps):
            _set_grads(params, seed=100 + step, skip=(skips or {}).get(step, ()))
            del calls[:]
            optimizer.step()
            out.append(_snapshot(calls))
        return out

    def test_cached_steps_reproduce_the_original_calls(self, monkeypatch):
        module, FusedAdam, calls = self._patched_module(monkeypatch)
        params = _fake_params()
        reference_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
        patched = FusedAdam(params)
        reference = _FakeFusedAdam(reference_params)
        reference_steps = []
        for step in range(4):
            _set_grads(reference_params, seed=100 + step)
            del calls[:]
            reference.step()
            reference_steps.append(_snapshot(calls))
        patched_steps = self._run(patched, params, 4, calls)
        # the first step runs the original (states are initialised there); the next three are the cached path
        assert patched.original_calls == 1
        for ref_step, step in zip(reference_steps, patched_steps):
            # identity is per optimizer: compare pointers within each optimizer's own params/states
            assert [c[0] for c in step] == [c[0] for c in ref_step]
            assert [c[1] for c in step] == [c[1] for c in ref_step]
            for (_, _, ref_cols), (_, _, cols) in zip(ref_step, step):
                for ref_col, col in zip(ref_cols, cols):
                    for (_, ref_value), (_, value) in zip(ref_col, col):
                        torch.testing.assert_close(value, ref_value, rtol=0, atol=0)
        # the cached lists alias the live parameters, gradients and states
        last = patched_steps[-1]
        (func16, _, cols16), (func32, _, cols32) = last
        assert func16 == "multi_tensor_adam_param_remainder" and func32 == "multi_tensor_adam"
        f16_params, f32_params = params[:2], params[2:]
        assert [ptr for ptr, _ in cols16[0]] == [p.grad.data_ptr() for p in f16_params]
        assert [ptr for ptr, _ in cols16[1]] == [p.data_ptr() for p in f16_params]
        assert [ptr for ptr, _ in cols16[2]] == [patched.state[p]["exp_avg"].data_ptr() for p in f16_params]
        assert [ptr for ptr, _ in cols16[4]] == [patched.state[p]["master_param"].data_ptr() for p in f16_params]
        assert [ptr for ptr, _ in cols32[0]] == [p.grad.data_ptr() for p in f32_params]
        assert [ptr for ptr, _ in cols32[3]] == [patched.state[p]["exp_avg_sq"].data_ptr() for p in f32_params]
        assert patched.param_groups[0]["step"] == 4

    def test_consecutive_cached_steps_see_the_fresh_gradients(self, monkeypatch):
        _, FusedAdam, calls = self._patched_module(monkeypatch)
        params = _fake_params()
        optimizer = FusedAdam(params)
        steps = self._run(optimizer, params, 3, calls)
        grads_step2 = [value for _, value in steps[1][0][2][0]]
        grads_step3 = [value for _, value in steps[2][0][2][0]]
        assert all(not torch.equal(a, b) for a, b in zip(grads_step2, grads_step3))
        _set_grads(params, seed=102)
        assert all(torch.equal(value, p.grad) for value, p in zip(grads_step3, params[:2]))

    @pytest.mark.parametrize("count", [1024, 1025])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_cached_calls_bound_tensor_batches_without_losing_or_repeating_parameters(
        self, monkeypatch: pytest.MonkeyPatch, count: int, dtype: torch.dtype
    ) -> None:
        _, FusedAdam, calls = self._patched_module(monkeypatch)
        params = [torch.nn.Parameter(torch.ones(1, dtype=dtype)) for _ in range(count)]
        optimizer = FusedAdam(params)
        steps = self._run(optimizer, params, 3, calls)
        assert optimizer.original_calls == 1
        assert optimizer.param_groups[0]["step"] == 3
        for step in steps[1:]:
            assert len(step) == (1 if count == 1024 else 2)
            assert [len(columns[0]) for _, _, columns in step] == ([1024] if count == 1024 else [1024, 1])
            assert all(len(column) <= 1024 for _, _, columns in step for column in columns)
            assert all(len({len(column) for column in columns}) == 1 for _, _, columns in step)
            expected_kernel = "multi_tensor_adam_param_remainder" if dtype == torch.bfloat16 else "multi_tensor_adam"
            assert all(kernel == expected_kernel for kernel, _, _ in step)
            assert all(args == step[0][1] for _, args, _ in step)
            expected_columns = [params, [optimizer.state[param]["exp_avg"] for param in params]]
            expected_columns.append([optimizer.state[param]["exp_avg_sq"] for param in params])
            if dtype == torch.bfloat16:
                expected_columns.append([optimizer.state[param]["master_param"] for param in params])
            for index, tensors in enumerate(expected_columns, start=1):
                assert [ptr for _, _, columns in step for ptr, _ in columns[index]] == [
                    tensor.data_ptr() for tensor in tensors
                ]
        assert [ptr for _, _, columns in steps[-1] for ptr, _ in columns[0]] == [
            param.grad.data_ptr() for param in params
        ]

    def test_mixed_small_and_large_groups_update_each_parameter_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, FusedAdam, calls = self._patched_module(monkeypatch)
        small = _fake_params()
        large = [torch.nn.Parameter(torch.ones(1)) for _ in range(1025)]
        optimizer = FusedAdam(small)
        optimizer.add_param_group({**optimizer.param_groups[0], "params": large})
        steps = self._run(optimizer, small + large, 3, calls)
        assert optimizer.original_calls == 1
        assert [group["step"] for group in optimizer.param_groups] == [3, 3]
        for step in steps[1:]:
            assert len(step) == 4
            assert [ptr for _, _, columns in step for ptr, _ in columns[1]] == [
                param.data_ptr() for param in small + large
            ]

    def test_missing_gradient_falls_back_and_rebuilds(self, monkeypatch):
        _, FusedAdam, calls = self._patched_module(monkeypatch)
        params = _fake_params()
        optimizer = FusedAdam(params)
        steps = self._run(optimizer, params, 4, calls, skips={2: (1,)})
        # step 0: original (states initialised); steps 1-3 cached. A gradient that vanishes at step 2 invalidates
        # the plan; the rebuilt plan skips that parameter exactly as the original loop does, no original call needed.
        assert optimizer.original_calls == 1
        assert len(steps[1][0][2][0]) == 2
        assert len(steps[2][0][2][0]) == 1
        assert len(steps[3][0][2][0]) == 2
        assert optimizer.param_groups[0]["step"] == 4

    def test_parameter_without_gradient_stays_skipped_until_it_gets_one(self, monkeypatch):
        _, FusedAdam, calls = self._patched_module(monkeypatch)
        params = _fake_params()
        optimizer = FusedAdam(params)
        steps = self._run(optimizer, params, 3, calls, skips={0: (3,), 1: (3,)})
        assert optimizer.original_calls == 1
        assert len(steps[1][1][2][0]) == 1  # fp32 column without the skipped parameter, cached path
        assert len(steps[2][1][2][0]) == 2  # gradient appeared → plan rebuilt, still no original call
        assert optimizer.original_calls == 1

    @pytest.mark.parametrize("mode", ["capturable", "scaler", "closure", "decoupled"])
    def test_unsupported_modes_use_the_original_step(self, monkeypatch, mode):
        _, FusedAdam, calls = self._patched_module(monkeypatch)
        params = _fake_params()
        optimizer = FusedAdam(params, capturable=(mode == "capturable"))
        if mode == "decoupled":
            optimizer.use_decoupled_grad = True
        for step in range(3):
            _set_grads(params, seed=step)
            if mode == "scaler":
                optimizer.step(grad_scaler=object())
            elif mode == "closure":
                optimizer.step(closure=lambda: None)
            else:
                optimizer.step()
        assert optimizer.original_calls == 3

    def test_states_needing_an_unscale_pass_use_the_original_step(self, monkeypatch):
        _, FusedAdam, calls = self._patched_module(monkeypatch)
        params = _fake_params()
        optimizer = FusedAdam(params)
        optimizer.name_to_dtype_map["exp_avg"] = torch.float16
        self._run(optimizer, params, 3, calls)
        assert optimizer.original_calls == 3

    def test_quantized_parameter_uses_the_original_step(self, monkeypatch):
        _, FusedAdam, calls = self._patched_module(monkeypatch)

        class Float8Like(torch.Tensor):
            pass

        params = _fake_params()
        params.append(torch.nn.Parameter(Float8Like(torch.randn(2))))
        optimizer = FusedAdam(params)
        self._run(optimizer, params, 2, calls)
        assert optimizer.original_calls == 2

    def test_load_state_dict_and_add_param_group_drop_the_plan(self, monkeypatch):
        _, FusedAdam, calls = self._patched_module(monkeypatch)
        params = _fake_params()
        optimizer = FusedAdam(params)
        self._run(optimizer, params, 2, calls)
        assert optimizer.__dict__["_nemo_fused_adam_plans"][0] is not None
        optimizer.load_state_dict({})
        assert optimizer.loaded == 1 and "_nemo_fused_adam_plans" not in optimizer.__dict__
        self._run(optimizer, params, 1, calls)
        assert optimizer.original_calls == 1  # states already exist: the plan is rebuilt on the cached path
        extra = torch.nn.Parameter(torch.randn(2))
        optimizer.add_param_group(
            {
                "params": [extra],
                "lr": 1e-3,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "bias_correction": True,
            }
        )
        assert "_nemo_fused_adam_plans" not in optimizer.__dict__
        _set_grads(params + [extra], seed=7)
        del calls[:]
        optimizer.step()
        assert optimizer.original_calls == 2  # the new group's state is initialised by the original step
        assert len(calls) == 3

    def test_empty_parameter_group_keeps_its_step_counter_in_sync(self, monkeypatch):
        """TE advances ``group["step"]`` before skipping an empty group (a group can be empty on some DP ranks);
        the cached path must count the same, or a rank resuming from such a checkpoint applies a stale bias
        correction (HuiyingLi on #4285: ``[4, 1]`` vs ``[4, 4]`` after four steps)."""
        _, FusedAdam, calls = self._patched_module(monkeypatch)
        params = _fake_params()
        optimizer = FusedAdam(params)
        optimizer.add_param_group(
            {"params": [], "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0, "bias_correction": True}
        )
        reference = _FakeFusedAdam([torch.nn.Parameter(p.detach().clone()) for p in params])
        reference.add_param_group(
            {"params": [], "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0, "bias_correction": True}
        )
        for step in range(4):
            _set_grads(params, seed=100 + step)
            _set_grads(reference.param_groups[0]["params"], seed=100 + step)
            optimizer.step()
            reference.step()
        assert optimizer.original_calls == 1  # steps 2-4 took the cached path
        assert [g["step"] for g in optimizer.param_groups] == [4, 4]
        assert [g["step"] for g in optimizer.param_groups] == [g["step"] for g in reference.param_groups]

    def test_unsupported_group_is_remembered_until_the_hooks_drop_it(self, monkeypatch):
        """A group that can never take the cached path is negative-cached: the original step runs without a
        per-parameter walk first; ``load_state_dict`` drops the verdict with the plans."""
        _, FusedAdam, calls = self._patched_module(monkeypatch)

        class Float8Like(torch.Tensor):
            pass

        params = _fake_params()
        params.append(torch.nn.Parameter(Float8Like(torch.randn(2))))
        optimizer = FusedAdam(params)
        self._run(optimizer, params, 3, calls)
        plans = optimizer.__dict__["_nemo_fused_adam_plans"]
        assert plans[0] is te_patches_module._FUSED_ADAM_NO_PLAN
        assert optimizer.original_calls == 3
        optimizer.load_state_dict({})
        assert "_nemo_fused_adam_plans" not in optimizer.__dict__
        self._run(optimizer, params, 1, calls)
        assert optimizer.__dict__["_nemo_fused_adam_plans"][0] is te_patches_module._FUSED_ADAM_NO_PLAN
        assert optimizer.original_calls == 4

    def test_dtensor_parameters_hand_their_local_shards_to_the_kernel(self, monkeypatch):
        dist = pytest.importorskip("torch.distributed")
        if not dist.is_available():
            pytest.skip("torch.distributed unavailable")
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard, distribute_tensor

        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(29500 + os.getpid() % 1000))
        if not dist.is_initialized():
            dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            mesh = init_device_mesh("cpu", (1,))
            _, FusedAdam, calls = self._patched_module(monkeypatch)
            weight = torch.nn.Parameter(distribute_tensor(torch.randn(4, 3), mesh, [Shard(0)]))
            assert isinstance(weight, DTensor)
            optimizer = FusedAdam([weight])

            def dtensor_states(param, remainders):
                state = optimizer.state[param]
                local = param._local_tensor
                state["exp_avg"] = distribute_tensor(torch.zeros_like(local, dtype=torch.bfloat16), mesh, [Shard(0)])
                state["exp_avg_sq"] = distribute_tensor(torch.zeros_like(local, dtype=torch.bfloat16), mesh, [Shard(0)])

            optimizer.initialize_state = dtensor_states
            for step in range(3):
                weight.grad = distribute_tensor(torch.randn(4, 3), mesh, [Shard(0)])
                del calls[:]
                optimizer.step()
            assert optimizer.original_calls == 1
            (func, _, lists, *_args) = calls[0]
            assert func == "multi_tensor_adam"
            grads, datas, moments, velocities = lists
            assert all(type(t) is torch.Tensor for column in lists for t in column)
            assert grads[0].data_ptr() == weight.grad._local_tensor.data_ptr()
            assert datas[0].data_ptr() == weight._local_tensor.data_ptr()
            assert moments[0].data_ptr() == optimizer.state[weight]["exp_avg"]._local_tensor.data_ptr()
            assert velocities[0].data_ptr() == optimizer.state[weight]["exp_avg_sq"]._local_tensor.data_ptr()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native TE optimizer parity")
@pytest.mark.parametrize("count", [4, 1025])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cached_fused_adam_matches_native_updates(count: int, dtype: torch.dtype) -> None:
    """Compare cached and native CUDA updates, including large and empty parameter groups."""
    module = pytest.importorskip("transformer_engine.pytorch.optimizers.fused_adam")
    # Native numerical parity cannot detect a changed resource cap; TE 2.19 has no cap constant.
    assert (
        getattr(module, "_MAX_TENSORS_PER_MULTI_TENSOR_ADAM_CALL", 1024)
        == te_patches_module._FUSED_ADAM_MAX_TENSORS_PER_CALL
    )
    optimizer_type = module.FusedAdam
    # Recipe setup tests may already have installed this wrapper on the real TE class.
    original_step = getattr(optimizer_type.step, "_nemo_original_step", optimizer_type.step)
    native_step = MagicMock(wraps=original_step)
    torch.manual_seed(123)
    params = [torch.nn.Parameter(torch.randn(4, device="cuda", dtype=dtype)) for _ in range(count)]
    reference_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    moment_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    optimizer = optimizer_type(
        [{"params": params}, {"params": []}],
        master_weights=True,
        store_param_remainders=True,
        exp_avg_dtype=moment_dtype,
        exp_avg_sq_dtype=moment_dtype,
    )
    reference = optimizer_type(
        [{"params": reference_params}, {"params": []}],
        master_weights=True,
        store_param_remainders=True,
        exp_avg_dtype=moment_dtype,
        exp_avg_sq_dtype=moment_dtype,
    )
    # Restore all three methods changed by the compatibility patch after this test.
    with (
        torch.no_grad(),
        patch.object(optimizer_type, "step", native_step),
        patch.object(optimizer_type, "load_state_dict", optimizer_type.load_state_dict),
        patch.object(optimizer_type, "add_param_group", optimizer_type.add_param_group),
    ):
        _apply_fused_adam_cached_lists_patch()
        assert optimizer_type.step is not native_step
        for step in range(4):
            for index, (param, reference_param) in enumerate(zip(params, reference_params)):
                grad = None if step == 2 and index == 0 else torch.randn_like(param)
                param.grad = grad
                reference_param.grad = None if grad is None else grad.clone()
            original_step(reference)
            optimizer.step()
            torch.testing.assert_close(torch.stack(params), torch.stack(reference_params), rtol=0, atol=0)
            state_names = (
                ("exp_avg", "exp_avg_sq", "master_param") if dtype == torch.bfloat16 else ("exp_avg", "exp_avg_sq")
            )
            for name in state_names:
                torch.testing.assert_close(
                    torch.stack([optimizer.state[param][name] for param in params]),
                    torch.stack([reference.state[param][name] for param in reference_params]),
                    rtol=0,
                    atol=0,
                )
            assert [group["step"] for group in optimizer.param_groups] == [step + 1, step + 1]
            assert [group["step"] for group in reference.param_groups] == [step + 1, step + 1]
        native_step.assert_called_once()  # First step initializes state; the other three use the cache.
