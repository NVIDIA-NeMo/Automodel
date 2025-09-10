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
"""Unit tests for the private ``_get_parallel_plan`` helper.

The function selects a tensor-parallel sharding plan via the following priority:

1. A *custom* plan supplied by the caller (either a dictionary ‑or- an import
   path to a dict/function).
2. If requested, the HuggingFace-derived plan via ``get_hf_tp_shard_plan``.
3. A model-specific plan located in ``PARALLELIZE_FUNCTIONS``; on failure, try HF.
4. Otherwise, return a default base plan (with SP adjustments when enabled).

This test module covers every branch, including error conditions.
"""

from __future__ import annotations

import types
from types import SimpleNamespace
from typing import Dict

import pytest

# Function under test and collaborators
import nemo_automodel.components.distributed.parallelizer as parallelizer
from nemo_automodel.components.distributed.parallelizer import _get_parallel_plan


class _DummyModel:
    """Minimal model stand-in."""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure external state is isolated between tests."""
    # Backup original global dicts so we can restore them after each test
    original_plans: Dict = parallelizer.PARALLELIZE_FUNCTIONS.copy()
    original_model_cls = getattr(parallelizer, "model_cls", None)

    yield

    # Restore module-level globals that we tamper with
    parallelizer.PARALLELIZE_FUNCTIONS.clear()
    parallelizer.PARALLELIZE_FUNCTIONS.update(original_plans)

    if original_model_cls is not None:
        monkeypatch.setattr(parallelizer, "model_cls", original_model_cls, raising=False)
    else:
        # Ensure we do not leak the attr
        monkeypatch.delattr(parallelizer, "model_cls", raising=False)


def _set_global_model_cls(monkeypatch, cls):
    """Make the *module-global* ``model_cls`` visible to the helper."""
    monkeypatch.setattr(parallelizer, "model_cls", cls, raising=False)


# 1. Custom plan provided directly as *dict*
def test_custom_dict_plan(monkeypatch):
    plan = {"foo": "bar"}
    _set_global_model_cls(monkeypatch, _DummyModel)  # irrelevant but required
    result = _get_parallel_plan(_DummyModel(), sequence_parallel=False, tp_shard_plan=plan)
    assert result is plan  # identity check


# 2. Custom plan via *import path*
def test_custom_plan_imports_dict(monkeypatch):
    plan = {"baz": "qux"}

    # Fake import path resolution
    def _fake_import_class_from_path(path):  # noqa: D401
        assert path == "some.module.PLAN"
        return plan  # Dict returned directly

    monkeypatch.setattr(parallelizer, "import_class_from_path", _fake_import_class_from_path, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), tp_shard_plan="some.module.PLAN")
    assert result is plan


def test_custom_plan_imports_function(monkeypatch):
    plan = {"alpha": "omega"}

    def _dummy_fn():
        return plan

    def _fake_import(path):  # noqa: D401
        return _dummy_fn

    monkeypatch.setattr(parallelizer, "import_class_from_path", _fake_import, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), tp_shard_plan="some.module.func")
    assert result is plan


def test_custom_plan_invalid_path(monkeypatch):
    """Invalid import path should raise *ValueError* from helper."""

    def _fake_import(path):  # noqa: D401
        raise ImportError("boom")

    monkeypatch.setattr(parallelizer, "import_class_from_path", _fake_import, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    with pytest.raises(ValueError):
        _get_parallel_plan(_DummyModel(), tp_shard_plan="bad.path")


# 3. Optimised plan in ``PARALLELIZE_FUNCTIONS``
def test_optimised_plan_success(monkeypatch):
    plan = {"opt": "plan"}

    # Register dummy entry
    parallelizer.PARALLELIZE_FUNCTIONS[_DummyModel] = lambda m, sp: plan
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=False)
    assert result is plan


def test_optimised_plan_fallback_to_hf(monkeypatch):
    """If the optimised function raises, the helper should fallback to HF plan."""
    sentinel = {"hf": "plan"}

    def _broken_fn(model, seq):  # noqa: D401
        raise RuntimeError("fail")

    parallelizer.PARALLELIZE_FUNCTIONS[_DummyModel] = _broken_fn
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda m: sentinel, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=False)
    assert result is sentinel


# 4. HF fallback when no optimised plan exists
def test_hf_fallback(monkeypatch):
    # When no optimised plan exists and HF is not explicitly requested, the helper
    # should return the default base plan.
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda m: {"hf": "plan2"}, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=False)
    assert isinstance(result, dict)
    # base plan should include embed_tokens and lm_head entries
    assert "model.embed_tokens" in result
    assert "lm_head" in result


def test_hf_fallback_sequence_parallel_assert(monkeypatch):
    """When sequence_parallel=True and no optimised plan, helper should return base plan with SP entries."""
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda m: {}, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=True)
    assert isinstance(result, dict)
    # SP-adjusted entries should be present
    assert "model.norm" in result


def test_use_hf_tp_plan_sp_false(monkeypatch):
    """Explicit HF plan when requested and SP=False returns HF plan."""
    sentinel = {"hf": "plan"}
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda m: sentinel, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=False, use_hf_tp_plan=True)
    assert result is sentinel


def test_use_hf_tp_plan_sp_true_assert(monkeypatch):
    """Explicit HF plan with SP=True should assert."""
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda m: {"hf": "plan"}, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    with pytest.raises(AssertionError):
        _get_parallel_plan(_DummyModel(), sequence_parallel=True, use_hf_tp_plan=True)


def test_optimised_plan_and_hf_both_fail_raises_sp_false(monkeypatch):
    """Optimised plan raises and HF raises → runtime error (SP=False)."""
    def _broken_fn(model, seq):
        raise RuntimeError("fail")

    parallelizer.PARALLELIZE_FUNCTIONS[_DummyModel] = _broken_fn
    def _raise_hf(_model):
        raise RuntimeError("hf fail")
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_hf, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    with pytest.raises(RuntimeError, match="hf fail"):
        _get_parallel_plan(_DummyModel(), sequence_parallel=False)


def test_optimised_plan_and_hf_both_fail_assert_sp_true(monkeypatch):
    """Optimised plan raises then HF path asserts (SP=True)."""
    def _broken_fn(model, seq):
        raise RuntimeError("fail")

    parallelizer.PARALLELIZE_FUNCTIONS[_DummyModel] = _broken_fn
    def _raise_hf2(_model):
        raise RuntimeError("hf fail")
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_hf2, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    with pytest.raises(AssertionError):
        _get_parallel_plan(_DummyModel(), sequence_parallel=True)


def test_not_registered_and_hf_fail_base_plan(monkeypatch):
    """No optimised plan and HF raises → base plan (with/without SP)."""
    # Ensure dummy not in mapping
    parallelizer.PARALLELIZE_FUNCTIONS.pop(_DummyModel, None)
    def _raise_hf3(_model):
        raise RuntimeError("hf fail")
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_hf3, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    # SP=False
    result = _get_parallel_plan(_DummyModel(), sequence_parallel=False)
    assert "model.embed_tokens" in result and "lm_head" in result

    # SP=True
    result_sp = _get_parallel_plan(_DummyModel(), sequence_parallel=True)
    assert "model.norm" in result_sp


def test_custom_plan_imports_non_dict_raises(monkeypatch):
    """If import resolves but returns non-dict object, raise ValueError."""
    def _fake_import(path):
        return ["not", "a", "dict"]

    monkeypatch.setattr(parallelizer, "import_class_from_path", _fake_import, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    with pytest.raises(ValueError):
        _get_parallel_plan(_DummyModel(), tp_shard_plan="some.module.NOT_A_DICT")
