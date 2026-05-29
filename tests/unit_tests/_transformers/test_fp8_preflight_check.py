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

"""Tests for the FP8 dequantize pre-flight check in model_init.

See https://github.com/NVIDIA-NeMo/Automodel/issues/2114 for background on why
this check exists.
"""

from types import SimpleNamespace

import pytest

from nemo_automodel._transformers.model_init import (
    _FP8_PREFLIGHT_DISABLE_ENV,
    _FP8_PREFLIGHT_FOOTPRINT_THRESHOLD,
    _check_fp8_dequantize_will_fit,
    _get_hf_param_count_estimate,
    _quant_config_is_fp8_full_materialize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hf_config(*, layers=80, hidden=12288, vocab=131072, inter=29568, n_experts=1):
    """Build a SimpleNamespace mimicking a transformer-style HF config."""
    return SimpleNamespace(
        num_hidden_layers=layers,
        hidden_size=hidden,
        vocab_size=vocab,
        intermediate_size=inter,
        num_local_experts=n_experts,
    )


def _fp8_dequant_cfg():
    return {"quant_method": "fp8", "dequantize": True}


def _fp8_no_dequant_cfg():
    return {"quant_method": "fp8", "dequantize": False}


# Mocked free-memory values used to drive the budget logic deterministically.
_MEM_200GB = 200 * 1024**3
_MEM_8GB = 8 * 1024**3


# ---------------------------------------------------------------------------
# _quant_config_is_fp8_full_materialize
# ---------------------------------------------------------------------------


class TestQuantConfigIsFp8FullMaterialize:
    def test_dict_fp8_dequantize_true(self):
        assert _quant_config_is_fp8_full_materialize({"quant_method": "fp8", "dequantize": True}) is True

    def test_dict_fp8_dequantize_false(self):
        assert _quant_config_is_fp8_full_materialize({"quant_method": "fp8", "dequantize": False}) is False

    def test_dict_fp8_no_dequantize_key(self):
        assert _quant_config_is_fp8_full_materialize({"quant_method": "fp8"}) is False

    def test_object_fp8_dequantize_true(self):
        cfg = SimpleNamespace(quant_method="fp8", dequantize=True)
        assert _quant_config_is_fp8_full_materialize(cfg) is True

    def test_non_fp8(self):
        assert _quant_config_is_fp8_full_materialize({"quant_method": "bnb_8bit", "dequantize": True}) is False

    def test_none(self):
        assert _quant_config_is_fp8_full_materialize(None) is False

    def test_accepts_dequant_aliases(self):
        assert _quant_config_is_fp8_full_materialize({"quant_method": "fp8", "is_dequantize": True}) is True
        assert _quant_config_is_fp8_full_materialize({"quant_method": "fp8", "dequant": True}) is True


# ---------------------------------------------------------------------------
# _get_hf_param_count_estimate
# ---------------------------------------------------------------------------


class TestGetHfParamCountEstimate:
    def test_uses_num_parameters_when_present(self):
        cfg = SimpleNamespace(num_parameters=12_345_000_000)
        assert _get_hf_param_count_estimate(cfg) == 12_345_000_000

    def test_formula_dense_model(self):
        # 7B-ish LLaMA shape should land in the single-digit billions.
        cfg = _make_hf_config(layers=32, hidden=4096, vocab=32000, inter=11008, n_experts=1)
        n = _get_hf_param_count_estimate(cfg)
        assert n is not None
        assert 5e9 < n < 2e10

    def test_formula_moe_inflates_param_count(self):
        # MoE inflates params via n_experts: a Mixtral-8x7B-shaped config
        # should produce noticeably more parameters than a dense 7B.
        moe = _get_hf_param_count_estimate(
            _make_hf_config(layers=32, hidden=4096, vocab=32000, inter=14336, n_experts=8)
        )
        dense = _get_hf_param_count_estimate(
            _make_hf_config(layers=32, hidden=4096, vocab=32000, inter=14336, n_experts=1)
        )
        assert moe > dense * 3

    def test_walks_text_subconfig(self):
        # Multimodal wrappers store layer counts on a text_config; the walker
        # must recurse rather than return zero. Use a real PretrainedConfig so
        # the isinstance check passes.
        from transformers import PretrainedConfig

        class _Sub(PretrainedConfig):
            model_type = "tiny"

        text = _Sub(num_hidden_layers=8, hidden_size=2048, vocab_size=32000, intermediate_size=8192)
        wrapper = SimpleNamespace(text_config=text)
        n = _get_hf_param_count_estimate(wrapper)
        assert n is not None and n > 0

    def test_returns_none_when_unestimable(self):
        assert _get_hf_param_count_estimate(SimpleNamespace()) is None


# ---------------------------------------------------------------------------
# _check_fp8_dequantize_will_fit
# ---------------------------------------------------------------------------


class TestCheckFp8DequantizeWillFit:
    def test_raises_when_footprint_exceeds_budget(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))

        # ~Devstral / Mistral Medium ballpark: ~246 GB BF16 vs an 8 GB budget.
        cfg = _make_hf_config(layers=88, hidden=12288, vocab=131072, inter=28672)

        with pytest.raises(RuntimeError) as excinfo:
            _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "mistralai/Mistral-Medium")
        msg = str(excinfo.value)
        assert "Mistral-Medium" in msg
        assert "BF16 footprint" in msg
        assert "Free CUDA memory" in msg
        assert "issues/2114" in msg
        assert "mistral3_vlm" in msg
        assert _FP8_PREFLIGHT_DISABLE_ENV in msg

    def test_returns_none_when_footprint_fits(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_200GB, _MEM_200GB))

        cfg = _make_hf_config(layers=32, hidden=4096, vocab=32000, inter=11008)  # ~7B
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "tinyllama/TinyLlama-FP8") is None

    def test_noop_when_quant_config_none(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        cfg = _make_hf_config()  # would-be huge model
        assert _check_fp8_dequantize_will_fit(cfg, None, "some/model") is None

    def test_noop_when_quant_method_not_fp8(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        cfg = _make_hf_config()
        assert _check_fp8_dequantize_will_fit(cfg, {"quant_method": "bnb_8bit", "dequantize": True}, "x") is None

    def test_noop_when_dequantize_false(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        cfg = _make_hf_config()
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_no_dequant_cfg(), "x") is None

    def test_noop_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

        def _boom(*a, **k):
            raise AssertionError("mem_get_info should not be called when CUDA is unavailable")

        monkeypatch.setattr("torch.cuda.mem_get_info", _boom)
        cfg = _make_hf_config()
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "x") is None

    def test_noop_when_disable_env_set(self, monkeypatch):
        monkeypatch.setenv(_FP8_PREFLIGHT_DISABLE_ENV, "1")

        def _boom(*a, **k):
            raise AssertionError("no CUDA calls should happen when the disable env is set")

        monkeypatch.setattr("torch.cuda.is_available", _boom)
        cfg = _make_hf_config()
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "x") is None

    def test_noop_when_param_count_unestimable(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        def _boom(*a, **k):
            raise AssertionError("mem_get_info should not run when the estimate is None")

        monkeypatch.setattr("torch.cuda.mem_get_info", _boom)
        assert _check_fp8_dequantize_will_fit(SimpleNamespace(), _fp8_dequant_cfg(), "x") is None

    def test_noop_when_mem_get_info_fails(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        def _raise(*a, **k):
            raise RuntimeError("no CUDA driver")

        monkeypatch.setattr("torch.cuda.mem_get_info", _raise)
        cfg = _make_hf_config(layers=88, hidden=12288)
        # Fail open: we'd rather let the load proceed than block on a flaky driver call.
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "x") is None

    def test_threshold_boundary(self, monkeypatch):
        """At exactly the budget, the check passes; one GB over and it trips."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        cfg = SimpleNamespace(num_parameters=1_000_000_000)  # 2 GB BF16

        exact_free = int((2 * 1024**3) / _FP8_PREFLIGHT_FOOTPRINT_THRESHOLD)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (exact_free, exact_free))
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "x") is None

        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (exact_free - 1024**3, exact_free))
        with pytest.raises(RuntimeError):
            _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "x")

    def test_object_quant_config(self, monkeypatch):
        """A FineGrainedFP8Config-like object (not a dict) must still be detected."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        cfg = _make_hf_config(layers=88, hidden=12288)
        quant = SimpleNamespace(quant_method="fp8", dequantize=True)
        with pytest.raises(RuntimeError):
            _check_fp8_dequantize_will_fit(cfg, quant, "any/model")


# ---------------------------------------------------------------------------
# Wiring: the force_hf branch must invoke the pre-flight before HF's loader.
# ---------------------------------------------------------------------------


class TestForceHfBranchWiring:
    def test_force_hf_invokes_preflight_before_loader(self, monkeypatch):
        """Drive _init_model with force_hf=True and a sentinel-raising preflight.

        If the sentinel propagates, the preflight is wired in. If
        _from_pretrained_parent_class is reached, the wiring is broken.
        """
        from nemo_automodel._transformers import model_init

        class _Sentinel(Exception):
            pass

        def _raise_sentinel(*a, **k):
            raise _Sentinel()

        monkeypatch.setattr(model_init, "_check_fp8_dequantize_will_fit", _raise_sentinel)
        monkeypatch.setattr(
            model_init,
            "get_hf_config",
            lambda path, attn, **_: SimpleNamespace(architectures=["Dummy"], quantization_config=None),
        )
        monkeypatch.setattr(model_init, "_propagate_torch_dtype_to_subconfigs", lambda *a, **k: None)
        monkeypatch.setattr(model_init, "_streaming_bnb_supported", lambda *a, **k: False)

        class _DummyCls:
            _model_mapping = {}

            @classmethod
            def _from_pretrained_parent_class(cls, *a, **k):
                raise AssertionError("HF loader must not run when preflight raises")

        with pytest.raises(_Sentinel):
            model_init._init_model(
                _DummyCls,
                "fake/repo",
                "sdpa",
                "bfloat16",
                None,  # quantization_config
                True,  # force_hf
            )
