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
3. The model's ``ParallelSpec.tp_plan``; on failure, try HF.
4. Otherwise, return a default base plan (with SP adjustments when enabled).

This test module covers every branch, including error conditions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch.distributed.tensor.parallel import ColwiseParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

# Function under test and collaborators
import nemo_automodel.components.distributed.parallelizer as parallelizer
from nemo_automodel._transformers.model_init import _get_mixin_wrapped_class
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import _get_parallel_plan
from nemo_automodel.components.models.llama.parallelization import LLAMA_PARALLEL_SPEC
from nemo_automodel.components.models.nemotron_nas.parallelization import DeciLMForCausalLM as DeciLMDeclaration


class _DummyModel:
    """Minimal model stand-in."""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure external state is isolated between tests."""
    original_model_cls = getattr(parallelizer, "model_cls", None)

    yield

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


def test_custom_moe_explicit_plan_is_validated_even_when_ep_is_one(monkeypatch):
    _set_global_model_cls(monkeypatch, _DummyModel)
    monkeypatch.setattr(parallelizer, "_uses_custom_moe_modules", lambda model: True)

    with pytest.raises(ValueError, match="EP-owned"):
        _get_parallel_plan(
            _DummyModel(),
            sequence_parallel=False,
            tp_shard_plan={"model.layers.*.mlp.experts": object()},
            tp_size=2,
        )


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


# 3. Optimised plan from the model's ParallelSpec
def test_optimised_plan_success(monkeypatch):
    plan = {"opt": "plan"}

    # Register dummy entry
    monkeypatch.setattr(_DummyModel, "parallel_spec", ParallelSpec(tp_plan=plan), raising=False)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=False)
    assert result == plan
    assert result is not plan  # the declaration is handed out as a copy


def test_declared_plan_wins_over_hf(monkeypatch):
    """A declared plan is used even when the model also carries a translatable HF ``_tp_plan``."""
    plan = {"opt": "plan"}
    monkeypatch.setattr(_DummyModel, "parallel_spec", ParallelSpec(tp_plan=plan), raising=False)
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda m: {"hf": "plan"}, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)
    assert _get_parallel_plan(_DummyModel(), sequence_parallel=False) == plan


def test_sequence_parallel_overlay_and_missing_overlay_warning(monkeypatch, caplog):
    """SP overlays the declared plan; a spec without an overlay logs that the request is ignored."""
    base = {"lm_head": "base_head", "model.layers.*.mlp.down_proj": "down"}
    overlay = {"lm_head": "sp_head", "model.norm": "norm"}
    monkeypatch.setattr(
        _DummyModel, "parallel_spec", ParallelSpec(tp_plan=base, sequence_parallel_plan=overlay), raising=False
    )
    _set_global_model_cls(monkeypatch, _DummyModel)
    assert _get_parallel_plan(_DummyModel(), sequence_parallel=True) == {**base, **overlay}
    assert _get_parallel_plan(_DummyModel(), sequence_parallel=False) == base

    monkeypatch.setattr(_DummyModel, "parallel_spec", ParallelSpec(tp_plan=base), raising=False)
    with caplog.at_level("WARNING"):
        assert _get_parallel_plan(_DummyModel(), sequence_parallel=True) == base
    assert "declares no sequence-parallel plan" in caplog.text


# 4. HF plan is used when no optimised plan exists
def test_hf_fallback(monkeypatch):
    # When no optimised plan exists, the helper should prefer the HF-provided plan.
    hf_plan = {"model.embed_tokens": "embed", "lm_head": "head"}
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda m: hf_plan, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=False)
    assert result == hf_plan


def test_hf_fallback_sequence_parallel_assert(monkeypatch):
    """When sequence_parallel=True and no optimised plan, helper should return base plan with SP entries."""
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda m: {}, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    result = _get_parallel_plan(_DummyModel(), sequence_parallel=True)
    assert isinstance(result, dict)
    # SP-adjusted entries should be present
    assert "model.norm" in result


def test_not_registered_and_hf_fail_base_plan(monkeypatch):
    """No optimised plan and HF raises → base plan (with/without SP)."""

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


def test_nemotron_flash_remote_code_uses_registered_tp_plan():
    """Nemotron Flash must retain its validated TP2 path after custom-code fail-fast checks."""
    model_cls = type("NemotronFlashForCausalLM", (), {"config_class": SimpleNamespace(model_type="nemotron_flash")})
    model_cls.__module__ = "transformers_modules.nemotron_flash.modeling_nemotron_flash"
    model = model_cls()
    model.__class__ = _get_mixin_wrapped_class(model_cls)  # what NeMo Auto does for remote-code classes
    model.config = type(
        "Config",
        (),
        {"model_type": "nemotron_flash", "architectures": ["NemotronFlashForCausalLM"]},
    )()

    result = _get_parallel_plan(model, sequence_parallel=False, tp_size=2)

    assert "model.layers.*.self_attn.qkv_proj" in result
    assert "model.layers.*.mlp.gate_up_proj" in result
    assert any(isinstance(layout, Shard) for layout in result["lm_head"].output_layouts)
    assert result["lm_head"].use_local_output is False


def test_nemotron_flash_drops_replicated_output_lm_head_plan():
    """Nemotron Flash must not mix replicated logits with a sharded lm_head norm."""
    model_cls = type("NemotronFlashForCausalLM", (), {"config_class": SimpleNamespace(model_type="nemotron_flash")})
    model = model_cls()
    model.__class__ = _get_mixin_wrapped_class(model_cls)
    model.config = SimpleNamespace(model_type="nemotron_flash")
    custom_plan = {"lm_head": ColwiseParallel(output_layouts=Replicate())}

    result = _get_parallel_plan(model, tp_shard_plan=custom_plan, tp_size=2)

    assert "lm_head" not in result


def test_default_plan_fallthrough_raises_for_remote_code_at_tp_size_gt_1(monkeypatch):
    """tp_size > 1 + custom-code arch with no registered plan should raise a clear ValueError.

    The default base plan produces DTensor placements without ``shard_order`` metadata,
    which trips an internal assert in ``torch.distributed.tensor._redistribute`` on the
    first weight redistribute. We refuse early *only* for HF custom-code architectures
    (loaded with ``trust_remote_code=True``, i.e. living under
    ``transformers_modules.*``), so users get an actionable error instead of an opaque
    PyTorch assertion. See https://github.com/NVIDIA-NeMo/Automodel/issues/2243.
    """

    def _raise_hf(_model):
        raise RuntimeError("hf fail")

    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_hf, raising=True)
    _set_global_model_cls(monkeypatch, _RemoteCodeDummyModel)

    for sp in (False, True):
        with pytest.raises(ValueError) as excinfo:
            _get_parallel_plan(_RemoteCodeDummyModel(), sequence_parallel=sp, tp_size=2)

        msg = str(excinfo.value)
        # The error must name the offending class and the three supported registration paths.
        assert _RemoteCodeDummyModel.__name__ in msg
        assert "components/models/<model_type>/parallelization.py" in msg
        assert "_tp_plan" in msg
        assert "tp_shard_plan" in msg


def test_default_plan_fallthrough_known_hf_arch_warns_at_tp_size_gt_1(monkeypatch, caplog):
    """Known HF archs (not ``transformers_modules.*``) keep working at tp_size > 1.

    They have been working in practice on the default base plan, so the guard only
    logs a warning and still returns the base plan rather than raising.
    """
    import logging as _logging

    def _raise_hf(_model):
        raise RuntimeError("hf fail")

    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_hf, raising=True)
    _set_global_model_cls(monkeypatch, _DummyModel)

    with caplog.at_level(_logging.WARNING, logger=parallelizer.logger.name):
        result = _get_parallel_plan(_DummyModel(), sequence_parallel=False, tp_size=2)

    assert "model.embed_tokens" in result and "lm_head" in result
    assert any("No usable tensor-parallel plan is registered" in r.message for r in caplog.records)


def test_default_plan_fallthrough_remote_code_folds_translator_diagnostic(monkeypatch):
    """Remote-code archs whose ``_tp_plan`` failed to translate get a diagnostic in the error.

    Covers the case where the model author exposed a ``_tp_plan`` but
    ``get_hf_tp_shard_plan`` raised while translating it (e.g. because the styles are
    not recognized by nemo). The raised ``ValueError`` should fold the translator's
    error message in so the user can distinguish "no `_tp_plan` at all" from
    "`_tp_plan` defined but unusable". See
    https://github.com/NVIDIA-NeMo/Automodel/pull/2244 discussion.
    """

    def _raise_translator(_model):
        raise ValueError("Unknown parallel style: foo_bar")

    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", _raise_translator, raising=True)
    _set_global_model_cls(monkeypatch, _RemoteCodeDummyModel)

    with pytest.raises(ValueError) as excinfo:
        _get_parallel_plan(_RemoteCodeDummyModel(), sequence_parallel=False, tp_size=2)

    msg = str(excinfo.value)
    # Diagnostic from get_hf_tp_shard_plan must be folded into the user-facing error.
    assert "Unknown parallel style: foo_bar" in msg
    # And the registration guidance must still be there.
    assert "components/models/<model_type>/parallelization.py" in msg
    assert "_tp_plan" in msg
    assert "tp_shard_plan" in msg


def test_default_plan_fallthrough_tp_size_1_still_returns_base_plan(monkeypatch):
    """tp_size == 1 keeps the existing behavior: the default base plan is returned.

    At tp_size == 1 no sharding actually happens, so the missing ``shard_order``
    metadata never matters. This preserves backwards compatibility for callers that
    do not pass ``tp_size`` (default is 1), including for custom-code archs.
    """

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
    monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", lambda _m: hf_plan, raising=True)
    _set_global_model_cls(monkeypatch, _RemoteCodeDummyModel)

    result = _get_parallel_plan(_RemoteCodeDummyModel(), sequence_parallel=False, tp_size=4)
    assert result == hf_plan


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


def test_legacy_alias_is_just_an_invalid_import_path():
    """``llama_nemotron_super_tp_plan`` names no plan any more: the architectures it covered declare their own."""
    with pytest.raises(ValueError, match="is not valid"):
        _get_parallel_plan(_DummyModel(), sequence_parallel=False, tp_shard_plan="llama_nemotron_super_tp_plan")


class TestGetLlamaNemotronSuperTpPlan:
    def test_returns_expected_keys(self):
        plan = LLAMA_PARALLEL_SPEC.resolved_tp_plan(sequence_parallel=False)
        assert isinstance(plan, dict)
        assert "model.embed_tokens" in plan
        assert "model.layers.*.self_attn.qkv_proj" in plan
        assert "model.layers.*.self_attn.o_proj" in plan
        assert "model.layers.*.mlp.gate_up_proj" in plan
        assert "model.layers.*.mlp.down_proj" in plan
        assert "lm_head" in plan

    def test_sp_adds_norm_and_layernorm_entries(self):
        plan = LLAMA_PARALLEL_SPEC.resolved_tp_plan(sequence_parallel=True)
        assert "model.norm" in plan
        assert "model.layers.*.input_layernorm" in plan
        assert "model.layers.*.post_attention_layernorm" in plan


class TestGetDecilmNemotronTpPlan:
    def test_returns_separate_qkv_projections(self):
        plan = DeciLMDeclaration.parallel_spec.resolved_tp_plan(sequence_parallel=False)
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
        plan = DeciLMDeclaration.parallel_spec.resolved_tp_plan(sequence_parallel=True)
        assert "model.norm" in plan
        assert "model.layers.*.input_layernorm" in plan


# ---------------------------------------------------------------------------
# Named plan resolution inside _get_parallel_plan
# ---------------------------------------------------------------------------


def _bridged_decilm() -> object:
    """A trust_remote_code Nemotron-NAS stand-in after the HF bridge bound its ParallelSpec."""
    model_cls = type("DeciLMForCausalLM", (), {"config_class": SimpleNamespace(model_type="nemotron-nas")})
    model = model_cls()
    model.__class__ = _get_mixin_wrapped_class(model_cls)
    model.config = SimpleNamespace(architectures=["DeciLMForCausalLM"], model_type="nemotron-nas")
    return model


def test_decilm_remote_code_class_auto_selects_nemotron_plan():
    result = _get_parallel_plan(_bridged_decilm(), sequence_parallel=False, tp_size=2)
    assert "model.layers.*.self_attn.q_proj" in result
    assert "model.layers.*.self_attn.k_proj" in result
    assert "model.layers.*.self_attn.v_proj" in result
    assert "model.layers.*.self_attn.qkv_proj" not in result
