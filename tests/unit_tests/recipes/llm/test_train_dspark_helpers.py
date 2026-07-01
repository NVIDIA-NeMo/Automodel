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

"""CPU unit tests for the DSpark recipe's V4-Flash helper knobs.

Covers the recipe-level glue added for the DeepSeek-V4-Flash target:
- ``_apply_target_chat_template``: a target whose tokenizer ships no chat
  template (V4-Flash) must take one from ``recipe_args.chat_template`` or fail
  fast, and an explicit template overrides whatever the tokenizer carries.
- ``_resolve_reduced_target_layers``: the ``target_num_hidden_layers``
  diagnostic override is range-checked.
- ``_resolve_dspark_optimizer_spec``: the ``optimizer:`` config is normalized
  into a ``build_optimizer`` spec, honoring an explicit ``_target_`` (e.g. TE
  FusedAdam with ``master_weights``/``exp_avg_dtype``/...) instead of always
  hardcoding plain ``torch.optim.AdamW``.
- ``_resolve_warmup_steps``: the ratio-derived warmup length is floored for
  short / small-dataset runs, unless the caller opts out with ``warmup_ratio<=0``.

(target_layer_ids range/-1/ordering validation is covered by the shared
``common.validate_target_layer_ids``, which HFDSparkTargetModel already calls.)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_automodel.recipes.llm.train_dspark import (
    _apply_target_chat_template,
    _resolve_dspark_optimizer_spec,
    _resolve_reduced_target_layers,
    _resolve_warmup_steps,
)

JINJA = (
    "{{ bos_token }}{% for m in messages %}{% if m['role'] == 'assistant' %}"
    "{% generation %}{{ m['content'] }}{% endgeneration %}{% endif %}{% endfor %}"
)


def _tok(chat_template=None):
    """A minimal tokenizer stub: ``_has_chat_template`` needs a ``chat_template``
    attribute plus a callable ``apply_chat_template``."""
    return SimpleNamespace(chat_template=chat_template, apply_chat_template=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# _apply_target_chat_template
# ---------------------------------------------------------------------------


def test_chat_template_set_when_provided_on_templateless_tokenizer():
    tok = _tok(chat_template=None)
    _apply_target_chat_template(tok, JINJA)
    assert tok.chat_template == JINJA


def test_chat_template_override_replaces_existing():
    tok = _tok(chat_template="OLD")
    _apply_target_chat_template(tok, JINJA)
    assert tok.chat_template == JINJA


def test_chat_template_none_with_existing_template_is_noop():
    tok = _tok(chat_template="EXISTING")
    _apply_target_chat_template(tok, None)
    assert tok.chat_template == "EXISTING"


def test_chat_template_none_without_template_raises():
    tok = _tok(chat_template=None)
    with pytest.raises(ValueError, match="no chat template"):
        _apply_target_chat_template(tok, None)


def test_chat_template_non_string_is_coerced(tmp_path):
    # A path-like value is stringified; _resolve_chat_template loads file contents.
    f = tmp_path / "tmpl.jinja"
    f.write_text(JINJA, encoding="utf-8")
    tok = _tok(chat_template=None)
    _apply_target_chat_template(tok, f)  # PosixPath, not str
    assert tok.chat_template == JINJA


# ---------------------------------------------------------------------------
# _resolve_reduced_target_layers
# ---------------------------------------------------------------------------


def test_reduced_layers_none_passes_through():
    assert _resolve_reduced_target_layers(43, None) is None


def test_reduced_layers_valid():
    assert _resolve_reduced_target_layers(43, 4) == 4


def test_reduced_layers_string_coerced():
    assert _resolve_reduced_target_layers(43, "4") == 4


def test_reduced_layers_full_depth_allowed():
    assert _resolve_reduced_target_layers(43, 43) == 43


@pytest.mark.parametrize("bad", [0, -1, 44, 100])
def test_reduced_layers_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="target_num_hidden_layers"):
        _resolve_reduced_target_layers(43, bad)


# ---------------------------------------------------------------------------
# _resolve_dspark_optimizer_spec
# ---------------------------------------------------------------------------


def _opt_cfg(**fields):
    """A minimal ``optimizer:`` config-node stub: dict-like ``to_dict``/``get``."""
    return SimpleNamespace(to_dict=lambda: dict(fields), get=lambda k, default=None: fields.get(k, default))


def test_optimizer_spec_defaults_to_adamw_when_no_target():
    target, kwargs = _resolve_dspark_optimizer_spec(_opt_cfg(lr=6e-4, warmup_ratio=0.04, min_lr_ratio=0.1))
    assert target == "torch.optim.AdamW"
    assert kwargs["lr"] == 6e-4
    assert kwargs["betas"] == (0.9, 0.95)
    assert kwargs["weight_decay"] == 0.0
    assert "warmup_ratio" not in kwargs
    assert "min_lr_ratio" not in kwargs


def test_optimizer_spec_respects_explicit_target_and_extra_kwargs():
    target, kwargs = _resolve_dspark_optimizer_spec(
        _opt_cfg(
            _target_="transformer_engine.pytorch.optimizers.FusedAdam",
            lr=1e-5,
            master_weights=True,
            master_weight_dtype="float32",
            exp_avg_dtype="float32",
            exp_avg_sq_dtype="float32",
            store_param_remainders=True,
        )
    )
    assert target == "transformer_engine.pytorch.optimizers.FusedAdam"
    assert kwargs["lr"] == 1e-5
    assert kwargs["master_weights"] is True
    assert kwargs["master_weight_dtype"] == "float32"
    assert kwargs["exp_avg_dtype"] == "float32"
    assert kwargs["exp_avg_sq_dtype"] == "float32"
    assert kwargs["store_param_remainders"] is True


def test_optimizer_spec_preserves_explicit_betas_and_weight_decay():
    _target, kwargs = _resolve_dspark_optimizer_spec(_opt_cfg(lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01))
    assert kwargs["betas"] == (0.9, 0.999)
    assert kwargs["weight_decay"] == 0.01


def test_optimizer_spec_coerces_lr_to_float():
    _target, kwargs = _resolve_dspark_optimizer_spec(_opt_cfg(lr="6e-4"))
    assert kwargs["lr"] == pytest.approx(6e-4)
    assert isinstance(kwargs["lr"], float)


# ---------------------------------------------------------------------------
# _resolve_warmup_steps
# ---------------------------------------------------------------------------


def test_warmup_steps_floors_short_runs():
    # 4% of 100 steps is 4 -- far too little warmup for a freshly-initialized
    # draft; the floor should kick in.
    assert _resolve_warmup_steps(0.04, 100) == 20


def test_warmup_steps_ratio_dominates_for_long_runs():
    # 4% of 10,000 steps is 400, well above the floor -- the ratio wins.
    assert _resolve_warmup_steps(0.04, 10_000) == 400


def test_warmup_steps_zero_ratio_is_explicit_opt_out():
    # The smoke config sets warmup_ratio=0.0 on purpose ("see movement
    # immediately"); the floor must not override that opt-out.
    assert _resolve_warmup_steps(0.0, 100) == 1


def test_warmup_steps_negative_ratio_treated_as_opt_out():
    assert _resolve_warmup_steps(-1.0, 100) == 1


def test_warmup_steps_custom_floor():
    assert _resolve_warmup_steps(0.01, 100, min_warmup_steps=5) == 5
    assert _resolve_warmup_steps(0.5, 100, min_warmup_steps=5) == 50
