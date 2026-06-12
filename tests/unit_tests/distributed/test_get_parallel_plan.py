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

from types import SimpleNamespace
from typing import Dict

import pytest

# Function under test and collaborators
import nemo_automodel.components.distributed.parallelizer as parallelizer
from nemo_automodel.components.distributed.optimized_tp_plans import (
    LLAMA_NEMOTRON_SUPER_TP_PLAN_NAME,
    _get_class_qualname,
    get_decilm_nemotron_tp_plan,
    get_llama_nemotron_super_tp_plan,
)
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
    parallelizer.PARALLELIZE_FUNCTIONS[_get_class_qualname(_DummyModel)] = lambda m, sp: plan
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=False)
    assert result is plan


def test_optimised_plan_fallback_to_hf(monkeypatch):
    """If the optimised function raises, the helper should fallback to HF plan."""
    sentinel = {"hf": "plan"}

    def _broken_fn(model, seq):  # noqa: D401
        raise RuntimeError("fail")

    parallelizer.PARALLELIZE_FUNCTIONS[_get_class_qualname(_DummyModel)] = _broken_fn
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda m: sentinel, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=False)
    assert result is sentinel


# 4. HF plan is used when no optimised plan exists
def test_hf_fallback(monkeypatch):
    # When no optimised plan exists, the helper should prefer the HF-provided plan.
    hf_plan = {"model.embed_tokens": "embed", "lm_head": "head"}
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda m: hf_plan, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=False)
    assert result is hf_plan


def test_hf_fallback_sequence_parallel_assert(monkeypatch):
    """When sequence_parallel=True and no optimised plan, helper should return base plan with SP entries."""
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda m: {}, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=True)
    assert isinstance(result, dict)
    # SP-adjusted entries should be present
    assert "model.norm" in result


def test_optimised_plan_and_hf_both_fail_raises_sp_false(monkeypatch):
    """Optimised plan raises and HF raises → runtime error (SP=False)."""

    def _broken_fn(model, seq):
        raise RuntimeError("fail")

    parallelizer.PARALLELIZE_FUNCTIONS[_get_class_qualname(_DummyModel)] = _broken_fn

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

    parallelizer.PARALLELIZE_FUNCTIONS[_get_class_qualname(_DummyModel)] = _broken_fn

    def _raise_hf2(_model):
        raise RuntimeError("hf fail")

    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_hf2, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    with pytest.raises(RuntimeError, match="hf fail"):
        _get_parallel_plan(_DummyModel(), sequence_parallel=True)


def test_not_registered_and_hf_fail_base_plan(monkeypatch):
    """No optimised plan and HF raises → base plan (with/without SP)."""
    # Ensure dummy not in mapping
    parallelizer.PARALLELIZE_FUNCTIONS.pop(_get_class_qualname(_DummyModel), None)

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


class _RemoteCodeDummyModel:
    """Stand-in for an HF trust_remote_code model. HF places those classes
    under the ``transformers_modules.*`` namespace at import time."""


# Mimic HF's dynamic-module convention so the fail-fast guard triggers.
_RemoteCodeDummyModel.__module__ = "transformers_modules.fake_repo.modeling_fake"


def test_default_plan_fallthrough_remote_code_warns_at_tp_size_gt_1(monkeypatch, caplog):
    """tp_size > 1 + custom-code arch with no registered plan warns and uses the base plan.

    #2244 hard-raised here on the theory that the default base plan trips a
    ``shard_order`` assert in ``_redistribute`` on the first weight redistribute. That
    was too broad: build/parallelize/forward/training all succeed on the base plan at
    tp>1, and the assert only fired on one checkpoint-reload ``redistribute`` that is now
    fixed at its source (``checkpoint.stateful_wrappers`` shards via ``distribute_tensor``).
    Custom-code archs (loaded with ``trust_remote_code=True``, i.e. living under
    ``transformers_modules.*``) are now treated like any other fall-through: warn and
    return the base plan. See https://github.com/NVIDIA-NeMo/Automodel/issues/2243.
    """
    import logging as _logging

    parallelizer.PARALLELIZE_FUNCTIONS.pop(_get_class_qualname(_RemoteCodeDummyModel), None)

    def _raise_hf(_model):
        raise RuntimeError("hf fail")

    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_hf, raising=True)
    _set_global_model_cls(monkeypatch, _RemoteCodeDummyModel)

    for sp in (False, True):
        caplog.clear()
        with caplog.at_level(_logging.WARNING, logger=parallelizer.logger.name):
            result = _get_parallel_plan(_RemoteCodeDummyModel(), sequence_parallel=sp, tp_size=2)

        assert "model.embed_tokens" in result and "lm_head" in result
        records = [r.getMessage() for r in caplog.records]
        # Warning must name the offending class and the registration paths.
        assert any(_RemoteCodeDummyModel.__name__ in m for m in records)
        assert any("PARALLELIZE_FUNCTIONS" in m and "_tp_plan" in m and "tp_shard_plan" in m for m in records)
        # And it must flag that this is a trust_remote_code arch.
        assert any("trust_remote_code=True" in m for m in records)


def test_default_plan_fallthrough_known_hf_arch_warns_at_tp_size_gt_1(monkeypatch, caplog):
    """Known HF archs (not ``transformers_modules.*``) keep working at tp_size > 1.

    They have been working in practice on the default base plan, so the helper only
    logs a warning and still returns the base plan rather than raising.
    """
    import logging as _logging

    parallelizer.PARALLELIZE_FUNCTIONS.pop(_get_class_qualname(_DummyModel), None)

    def _raise_hf(_model):
        raise RuntimeError("hf fail")

    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_hf, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    with caplog.at_level(_logging.WARNING, logger=parallelizer.logger.name):
        result = _get_parallel_plan(_DummyModel(), sequence_parallel=False, tp_size=2)

    assert "model.embed_tokens" in result and "lm_head" in result
    records = [r.getMessage() for r in caplog.records]
    assert any("No registered tensor-parallel plan" in m for m in records)
    # A known HF arch is NOT a trust_remote_code class, so that note must be absent.
    assert not any("trust_remote_code=True" in m for m in records)


def test_default_plan_fallthrough_remote_code_folds_translator_diagnostic(monkeypatch, caplog):
    """Remote-code archs whose ``_tp_plan`` failed to translate get a diagnostic in the warning.

    Covers the case where the model author exposed a ``_tp_plan`` but
    ``get_hf_tp_shard_plan`` raised while translating it (e.g. because the styles are
    not recognized by nemo). The fall-through warning folds the translator's error
    message in so the user can distinguish "no `_tp_plan` at all" from "`_tp_plan`
    defined but unusable". See https://github.com/NVIDIA-NeMo/Automodel/pull/2244.
    """
    import logging as _logging

    parallelizer.PARALLELIZE_FUNCTIONS.pop(_get_class_qualname(_RemoteCodeDummyModel), None)

    def _raise_translator(_model):
        raise ValueError("Unknown parallel style: foo_bar")

    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_translator, raising=True)
    _set_global_model_cls(monkeypatch, _RemoteCodeDummyModel)

    with caplog.at_level(_logging.WARNING, logger=parallelizer.logger.name):
        result = _get_parallel_plan(_RemoteCodeDummyModel(), sequence_parallel=False, tp_size=2)

    assert "model.embed_tokens" in result and "lm_head" in result
    records = [r.getMessage() for r in caplog.records]
    # Diagnostic from get_hf_tp_shard_plan must be folded into the warning.
    assert any("Unknown parallel style: foo_bar" in m for m in records)
    # And the registration guidance must still be there.
    assert any("PARALLELIZE_FUNCTIONS" in m and "_tp_plan" in m and "tp_shard_plan" in m for m in records)


def test_default_plan_fallthrough_tp_size_1_still_returns_base_plan(monkeypatch):
    """tp_size == 1 keeps the existing behavior: the default base plan is returned.

    At tp_size == 1 no sharding actually happens, so the missing ``shard_order``
    metadata never matters. This preserves backwards compatibility for callers that
    do not pass ``tp_size`` (default is 1), including for custom-code archs.
    """
    parallelizer.PARALLELIZE_FUNCTIONS.pop(_get_class_qualname(_RemoteCodeDummyModel), None)

    def _raise_hf(_model):
        raise RuntimeError("hf fail")

    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_hf, raising=True)
    _set_global_model_cls(monkeypatch, _RemoteCodeDummyModel)

    # Explicit tp_size=1 — should still return the base plan, even for remote-code archs.
    result = _get_parallel_plan(_RemoteCodeDummyModel(), sequence_parallel=False, tp_size=1)
    assert "model.embed_tokens" in result and "lm_head" in result


def test_hf_native_plan_unaffected_at_tp_size_gt_1(monkeypatch):
    """Models that expose an HF-native ``_tp_plan`` must not trip the new guard.

    The fail-fast check only fires when path 4 (default base plan) would be taken
    *and* the model is a custom-code arch. If ``get_hf_tp_shard_plan`` returns a
    non-empty plan, that plan must be used regardless of ``tp_size``.
    """
    hf_plan = {"model.embed_tokens": "embed", "lm_head": "head"}
    parallelizer.PARALLELIZE_FUNCTIONS.pop(_get_class_qualname(_RemoteCodeDummyModel), None)
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda _m: hf_plan, raising=True)
    _set_global_model_cls(monkeypatch, _RemoteCodeDummyModel)

    result = _get_parallel_plan(_RemoteCodeDummyModel(), sequence_parallel=False, tp_size=4)
    assert result is hf_plan


def test_custom_plan_imports_non_dict_raises(monkeypatch):
    """If import resolves but returns non-dict object, raise ValueError."""

    def _fake_import(path):
        return ["not", "a", "dict"]

    monkeypatch.setattr(parallelizer, "import_class_from_path", _fake_import, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    with pytest.raises(ValueError):
        _get_parallel_plan(_DummyModel(), tp_shard_plan="some.module.NOT_A_DICT")


# ---------------------------------------------------------------------------
# Named TP plan constant and plan builder functions
# ---------------------------------------------------------------------------


def test_named_plan_constant_value():
    assert LLAMA_NEMOTRON_SUPER_TP_PLAN_NAME == "llama_nemotron_super_tp_plan"


class TestGetLlamaNemotronSuperTpPlan:
    def test_returns_expected_keys(self):
        plan = get_llama_nemotron_super_tp_plan(sequence_parallel=False)
        assert isinstance(plan, dict)
        assert "model.embed_tokens" in plan
        assert "model.layers.*.self_attn.qkv_proj" in plan
        assert "model.layers.*.self_attn.o_proj" in plan
        assert "model.layers.*.mlp.gate_up_proj" in plan
        assert "model.layers.*.mlp.down_proj" in plan
        assert "lm_head" in plan

    def test_sp_adds_norm_and_layernorm_entries(self):
        plan = get_llama_nemotron_super_tp_plan(sequence_parallel=True)
        assert "model.norm" in plan
        assert "model.layers.*.input_layernorm" in plan
        assert "model.layers.*.post_attention_layernorm" in plan


class TestGetDecilmNemotronTpPlan:
    def test_returns_separate_qkv_projections(self):
        plan = get_decilm_nemotron_tp_plan(sequence_parallel=False)
        assert isinstance(plan, dict)
        assert "model.layers.*.self_attn.q_proj" in plan
        assert "model.layers.*.self_attn.k_proj" in plan
        assert "model.layers.*.self_attn.v_proj" in plan
        assert "model.layers.*.self_attn.o_proj" in plan
        assert "model.layers.*.mlp.up_proj" in plan
        assert "model.layers.*.mlp.gate_proj" in plan
        assert "model.layers.*.mlp.down_proj" in plan
        assert "lm_head" in plan
        # Must NOT have fused projections (those are Llama-specific)
        assert "model.layers.*.self_attn.qkv_proj" not in plan
        assert "model.layers.*.mlp.gate_up_proj" not in plan

    def test_sp_adds_norm_entries(self):
        plan = get_decilm_nemotron_tp_plan(sequence_parallel=True)
        assert "model.norm" in plan
        assert "model.layers.*.input_layernorm" in plan


# ---------------------------------------------------------------------------
# Named plan resolution inside _get_parallel_plan
# ---------------------------------------------------------------------------


def test_named_plan_resolves_to_llama_for_generic_model():
    """Named plan on a model without DeciLM config → fused Llama plan."""
    model = _DummyModel()
    result = _get_parallel_plan(
        model,
        sequence_parallel=False,
        tp_shard_plan=LLAMA_NEMOTRON_SUPER_TP_PLAN_NAME,
    )
    assert "model.layers.*.self_attn.qkv_proj" in result


def test_named_plan_resolves_to_decilm_for_nemotron_nas():
    """Named plan on DeciLM/nemotron-nas model → separate-projection plan."""
    model = _DummyModel()
    model.config = SimpleNamespace(
        architectures=["DeciLMForCausalLM"],
        model_type="nemotron-nas",
    )
    result = _get_parallel_plan(
        model,
        sequence_parallel=False,
        tp_shard_plan=LLAMA_NEMOTRON_SUPER_TP_PLAN_NAME,
    )
    assert "model.layers.*.self_attn.q_proj" in result
    assert "model.layers.*.self_attn.k_proj" in result
    assert "model.layers.*.self_attn.v_proj" in result
    assert "model.layers.*.self_attn.qkv_proj" not in result


def test_named_plan_llama_with_sequence_parallel():
    """Named plan + SP on a generic model includes norm entries."""
    model = _DummyModel()
    result = _get_parallel_plan(
        model,
        sequence_parallel=True,
        tp_shard_plan=LLAMA_NEMOTRON_SUPER_TP_PLAN_NAME,
    )
    assert "model.norm" in result
    assert "model.layers.*.input_layernorm" in result


def test_named_plan_decilm_with_sequence_parallel():
    """Named plan + SP on DeciLM model includes norm entries."""
    model = _DummyModel()
    model.config = SimpleNamespace(
        architectures=["DeciLMForCausalLM"],
        model_type="nemotron-nas",
    )
    result = _get_parallel_plan(
        model,
        sequence_parallel=True,
        tp_shard_plan=LLAMA_NEMOTRON_SUPER_TP_PLAN_NAME,
    )
    assert "model.norm" in result
    assert "model.layers.*.self_attn.q_proj" in result


# ---------------------------------------------------------------------------
# Registered remote-code archs resolve via the bare-class-name fallback
# instead of tripping the trust_remote_code fail-fast guard.
# ---------------------------------------------------------------------------


class _DeciLMRemoteCodeModel:
    """Stand-in for the remote-code ``DeciLMForCausalLM`` class.

    Its bare ``__name__`` must match the ``PARALLELIZE_FUNCTIONS`` key, and its
    ``__module__`` lives under ``transformers_modules.*`` (HF's dynamic-module
    namespace) so we exercise the exact path the Super-49B recipe hits.
    """


_DeciLMRemoteCodeModel.__name__ = "DeciLMForCausalLM"
_DeciLMRemoteCodeModel.__qualname__ = "DeciLMForCausalLM"
_DeciLMRemoteCodeModel.__module__ = "transformers_modules.nvidia.Llama_3_3_Nemotron_Super_49B.abc123.modeling_decilm"


def test_decilm_registered_resolves_at_tp_size_gt_1(monkeypatch):
    """Registered DeciLMForCausalLM must NOT raise the remote-code fail-fast at tp>1.

    Regression test for the Super-49B squad ckpt-robustness job: the model is a
    trust_remote_code DeciLM whose qualified module path carries a snapshot hash,
    so it is keyed in PARALLELIZE_FUNCTIONS by bare class name and must be picked
    up by the ``model_cls.__name__`` fallback in ``_get_parallel_plan`` before the
    fail-fast guard fires. It must return the DeciLM (separate-projection) plan.
    """

    # HF translation deliberately unavailable so we would otherwise fall through
    # to the default base plan and trip the remote-code guard.
    def _raise_hf(_model):
        raise RuntimeError("hf fail")

    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_hf, raising=True)
    _set_global_model_cls(monkeypatch, _DeciLMRemoteCodeModel)

    result = _get_parallel_plan(_DeciLMRemoteCodeModel(), sequence_parallel=False, tp_size=4)

    # DeciLM is not fused: separate q/k/v + separate gate/up, no fused projections.
    assert "model.layers.*.self_attn.q_proj" in result
    assert "model.layers.*.self_attn.k_proj" in result
    assert "model.layers.*.self_attn.v_proj" in result
    assert "model.layers.*.mlp.gate_proj" in result
    assert "model.layers.*.mlp.up_proj" in result
    assert "model.layers.*.self_attn.qkv_proj" not in result
    assert "model.layers.*.mlp.gate_up_proj" not in result
