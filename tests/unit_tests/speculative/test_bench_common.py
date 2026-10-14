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

"""Unit tests for the ``--prompt-column`` raw-text path added to ``bench_common._load_prompts``.

The ``--messages-column`` path is already covered by ``test_bench_sglang.py``
(``test_load_prompts_caps_and_drops_unusable``); this file covers the sibling
path bench_sweep relies on for datasets (GSM8K, HumanEval, ...) that are not
chat-messages-shaped.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_automodel.components.speculative.bench_common import _extract_prompt_text, _load_prompts

# ---------------------------------------------------------------------------
# _extract_prompt_text
# ---------------------------------------------------------------------------


def test_extract_prompt_text_plain_string():
    assert _extract_prompt_text("hello") == "hello"


def test_extract_prompt_text_list_uses_first_entry():
    assert _extract_prompt_text(["first turn", "second turn"]) == "first turn"


def test_extract_prompt_text_rejects_empty_and_blank():
    assert _extract_prompt_text("") is None
    assert _extract_prompt_text("   ") is None
    assert _extract_prompt_text([]) is None


def test_extract_prompt_text_rejects_non_string_types():
    assert _extract_prompt_text(None) is None
    assert _extract_prompt_text(42) is None
    assert _extract_prompt_text({"role": "user"}) is None


# ---------------------------------------------------------------------------
# _load_prompts: prompt_column path
# ---------------------------------------------------------------------------


def _args(**overrides):
    base = dict(
        input_data="data",
        split="train",
        dataset_name=None,
        shuffle_seed=None,
        messages_column="messages",
        prompt_column=None,
        num_prompts=10,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_load_prompts_prompt_column_wraps_single_turn_user_message(monkeypatch):
    rows = [{"question": "2+2=?"}, {"question": "3+3=?"}]
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.chat_dataset._load_openai_messages", lambda *a, **k: rows
    )
    prompts = _load_prompts(_args(prompt_column="question"))
    assert prompts == [
        [{"role": "user", "content": "2+2=?"}],
        [{"role": "user", "content": "3+3=?"}],
    ]


def test_load_prompts_prompt_column_takes_precedence_over_messages_column(monkeypatch):
    """A dataset can carry both columns; --prompt-column wins when set."""
    rows = [{"question": "raw text", "messages": [{"role": "user", "content": "chat text"}]}]
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.chat_dataset._load_openai_messages", lambda *a, **k: rows
    )
    prompts = _load_prompts(_args(prompt_column="question"))
    assert prompts == [[{"role": "user", "content": "raw text"}]]


def test_load_prompts_prompt_column_drops_unusable_rows(monkeypatch):
    rows = [{"question": "ok"}, {"question": ""}, {"question": None}, {"other": "x"}]
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.chat_dataset._load_openai_messages", lambda *a, **k: rows
    )
    prompts = _load_prompts(_args(prompt_column="question"))
    assert prompts == [[{"role": "user", "content": "ok"}]]


def test_load_prompts_prompt_column_list_field_uses_first_turn(monkeypatch):
    """MT-Bench-shaped rows: a list-of-turns column, first turn used."""
    rows = [{"turns": ["first", "second"]}]
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.chat_dataset._load_openai_messages", lambda *a, **k: rows
    )
    prompts = _load_prompts(_args(prompt_column="turns"))
    assert prompts == [[{"role": "user", "content": "first"}]]


def test_load_prompts_no_prompt_column_attribute_falls_back_to_messages(monkeypatch):
    """Callers whose Namespace never sets prompt_column (the two single-dataset
    benchmarks' existing test fixtures) keep working via getattr's default."""
    rows = [{"messages": [{"role": "user", "content": "hi"}]}]
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.chat_dataset._load_openai_messages", lambda *a, **k: rows
    )
    args = SimpleNamespace(
        input_data="d", split="train", dataset_name=None, shuffle_seed=None, messages_column="messages", num_prompts=10
    )  # no prompt_column attr at all
    prompts = _load_prompts(args)
    assert prompts == [[{"role": "user", "content": "hi"}]]


def test_load_prompts_prompt_column_respects_num_prompts_cap(monkeypatch):
    rows = [{"question": f"q{i}"} for i in range(5)]
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.chat_dataset._load_openai_messages", lambda *a, **k: rows
    )
    prompts = _load_prompts(_args(prompt_column="question", num_prompts=2))
    assert len(prompts) == 2


# ---------------------------------------------------------------------------
# _load_prompts: prompt_context_column (Alpaca-style secondary field)
# ---------------------------------------------------------------------------


def test_load_prompts_appends_context_column_when_present(monkeypatch):
    """Alpaca-shaped rows: instruction + non-empty input are joined into one prompt."""
    rows = [{"instruction": "Identify the odd one out.", "input": "Twitter, Instagram, Telegram."}]
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.chat_dataset._load_openai_messages", lambda *a, **k: rows
    )
    prompts = _load_prompts(_args(prompt_column="instruction", prompt_context_column="input"))
    assert prompts == [[{"role": "user", "content": "Identify the odd one out.\n\nTwitter, Instagram, Telegram."}]]


def test_load_prompts_omits_empty_context_column(monkeypatch):
    """A blank/missing context field leaves the bare instruction untouched."""
    rows = [{"instruction": "Name a color.", "input": ""}, {"instruction": "Name a fruit."}]
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.chat_dataset._load_openai_messages", lambda *a, **k: rows
    )
    prompts = _load_prompts(_args(prompt_column="instruction", prompt_context_column="input"))
    assert prompts == [
        [{"role": "user", "content": "Name a color."}],
        [{"role": "user", "content": "Name a fruit."}],
    ]


def test_load_prompts_context_column_attribute_optional(monkeypatch):
    """A Namespace that never sets prompt_context_column behaves as before (no append)."""
    rows = [{"instruction": "Name a color.", "input": "ignored"}]
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.chat_dataset._load_openai_messages", lambda *a, **k: rows
    )
    args = _args(prompt_column="instruction")  # _args never sets prompt_context_column
    assert not hasattr(args, "prompt_context_column")
    prompts = _load_prompts(args)
    assert prompts == [[{"role": "user", "content": "Name a color."}]]


def test_generation_config_defaults_top_k_to_disabled():
    """The engines spell "no top-k" as -1, so a literal 0 must never be sent.

    OpenAI's chat-completions schema has no ``top_k``; SGLang and vLLM accept it
    as an extra where ``-1`` disables it. Our default of 0 therefore means
    "omit the key", which ``_run_workload`` relies on.
    """
    from nemo_automodel.components.speculative.regenerate import GenerationConfig

    unset = GenerationConfig(model="m", max_new_tokens=8, temperature=1.0, top_p=0.95)
    assert unset.top_k == 0
    assert GenerationConfig(model="m", max_new_tokens=8, temperature=1.0, top_p=0.95, top_k=20).top_k == 20


def _bench_modules():
    from nemo_automodel.components.speculative import bench_sglang, bench_sweep, bench_vllm

    return (bench_sglang, bench_vllm, bench_sweep)


@pytest.mark.parametrize("field,expected_default", [("temperature", 0.0), ("top_p", 1.0), ("top_k", 0)])
def test_every_bench_entrypoint_exposes_the_same_sampling_knobs(field, expected_default):
    """The three entrypoints must not drift apart on sampling.

    ``top_k`` was missing from all of them, so the DFlash 2 evaluation setting
    (temperature 1.0, top-p 0.95, top-k 20) could not be expressed at all.
    Parametrising over the knob set means a knob added to one entrypoint and
    forgotten in another fails here instead of silently diverging.
    """
    for module in _bench_modules():
        assert module._build_parser().get_default(field) == expected_default, (
            f"{module.__name__} must expose --{field.replace('_', '-')} defaulting to {expected_default}"
        )


def test_generation_config_sampling_fields_are_all_reachable_from_the_bench_cli():
    """Every sampling field on the shared config must be settable per entrypoint.

    A field that exists on ``GenerationConfig`` but has no flag is dead weight: it
    records a constant while looking configurable, which is exactly what
    ``top_k`` was when it was first added.
    """
    import dataclasses

    from nemo_automodel.components.speculative.regenerate import GenerationConfig

    sampling_fields = {f.name for f in dataclasses.fields(GenerationConfig)} & {"temperature", "top_p", "top_k"}
    for module in _bench_modules():
        parser = module._build_parser()
        for field in sorted(sampling_fields):
            assert parser.get_default(field) is not None, f"{module.__name__} does not expose {field}"
