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

(target_layer_ids range/-1/ordering validation is covered by the shared
``common.validate_target_layer_ids``, which HFDSparkTargetModel already calls.)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_automodel.recipes.llm.train_dspark import (
    _apply_target_chat_template,
    _resolve_reduced_target_layers,
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
