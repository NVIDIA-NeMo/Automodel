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

from typing import List

import pytest
import torch

from nemo_automodel.components.eval import tool_call_evaluator as ev_mod
from nemo_automodel.components.eval.tool_call_evaluator import ToolCallAccuracyEvaluator


# ---------- fakes ----------


class FakeTokenizer:
    """Minimal tokenizer that renders prompts as their string repr and
    decodes by looking up a per-sample response from a script."""

    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 0
        self._next_call = 0
        self.responses: List[str] = []
        self.prompts_seen: List[list] = []
        self.tools_seen: List = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, tools=None):
        # Record what we were asked to render.
        self.prompts_seen.append(messages)
        self.tools_seen.append(tools)
        # Encode prompt as 1-byte-per-char ASCII ids; values >= 1 so they
        # don't collide with the eos pad.
        text = repr((messages, tools))
        return [(ord(c) % 250) + 1 for c in text]

    def decode(self, ids, *, skip_special_tokens=False):
        # Pop the next scripted response. Index by the order generate()
        # was called.
        if self._next_call < len(self.responses):
            resp = self.responses[self._next_call]
            self._next_call += 1
            return resp
        return ""


class FakeModel:
    """Stub model that records generate() calls and emits a sentinel
    response sequence per call. The decoded text comes from the
    tokenizer's scripted ``responses`` list, not from the token ids."""

    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.generate_calls = 0

    def parameters(self):
        yield self._param

    def generate(self, *, input_ids, attention_mask, max_new_tokens, do_sample, pad_token_id):
        self.generate_calls += 1
        # Append max_new_tokens zeros so the evaluator can slice off the
        # new portion; the decoded text is scripted in the tokenizer.
        new = torch.zeros(
            (input_ids.shape[0], max_new_tokens), dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.cat([input_ids, new], dim=1)


def _stub_samples():
    return [
        {
            "prompt_messages": [{"role": "user", "content": "what's the weather?"}],
            "tools": [{"name": "get_weather"}],
            "gt_tool_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
            "example_id": 1,
            "turn_index": 0,
        },
        {
            "prompt_messages": [{"role": "user", "content": "math"}],
            "tools": [{"name": "calc"}],
            "gt_tool_calls": [{"name": "calc", "arguments": {"a": 1, "b": 2}}],
            "example_id": 2,
            "turn_index": 0,
        },
    ]


def _make_evaluator(monkeypatch, **overrides):
    monkeypatch.setattr(ev_mod, "make_agent_chat_eval_samples", lambda **_: _stub_samples())
    kwargs = dict(path="dummy")
    kwargs.update(overrides)
    return ToolCallAccuracyEvaluator(**kwargs)


# ---------- tests ----------


def test_constructor_requires_one_source():
    with pytest.raises(ValueError):
        ToolCallAccuracyEvaluator()
    with pytest.raises(ValueError):
        ToolCallAccuracyEvaluator(dataset_name="x", path="y")


def test_evaluate_perfect_predictions(monkeypatch):
    evaluator = _make_evaluator(monkeypatch)
    tok = FakeTokenizer()
    tok.responses = [
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>',
        '<tool_call>{"name": "calc", "arguments": {"a": 1, "b": 2}}</tool_call>',
    ]
    model = FakeModel()

    metrics = evaluator.evaluate(model, tok)

    assert metrics["tool_call/_count"] == 2.0
    assert metrics["tool_call/has_call"] == 1.0
    assert metrics["tool_call/name_correct"] == 1.0
    assert metrics["tool_call/args_json_valid"] == 1.0
    assert metrics["tool_call/args_exact_match"] == 1.0


def test_evaluate_partial_predictions(monkeypatch):
    evaluator = _make_evaluator(monkeypatch)
    tok = FakeTokenizer()
    tok.responses = [
        # right name, wrong args
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
        # wrong name, right args
        '<tool_call>{"name": "wrong", "arguments": {"a": 1, "b": 2}}</tool_call>',
    ]
    model = FakeModel()

    metrics = evaluator.evaluate(model, tok)

    assert metrics["tool_call/has_call"] == 1.0
    assert metrics["tool_call/name_correct"] == 0.5
    assert metrics["tool_call/args_exact_match"] == 0.5  # only second has args match


def test_evaluate_no_tool_call_emitted(monkeypatch):
    evaluator = _make_evaluator(monkeypatch)
    tok = FakeTokenizer()
    tok.responses = [
        "I think the weather is nice today.",
        "Sure, the answer is 3.",
    ]
    model = FakeModel()

    metrics = evaluator.evaluate(model, tok)

    assert metrics["tool_call/has_call"] == 0.0
    assert metrics["tool_call/name_correct"] == 0.0
    assert metrics["tool_call/_count"] == 2.0


def test_evaluate_malformed_json(monkeypatch):
    evaluator = _make_evaluator(monkeypatch)
    tok = FakeTokenizer()
    tok.responses = [
        '<tool_call>not json at all</tool_call>',
        '<tool_call>{"name": "calc", "arguments": {"a": 1, "b": 2}}</tool_call>',
    ]
    model = FakeModel()

    metrics = evaluator.evaluate(model, tok)

    # First has parsed wrapper (has_call=1) but invalid JSON; second perfect.
    assert metrics["tool_call/has_call"] == 1.0
    assert metrics["tool_call/args_json_valid"] == 0.5


def test_evaluate_sample_sharding(monkeypatch):
    evaluator_r0 = _make_evaluator(monkeypatch, sample_shard=(0, 2))
    evaluator_r1 = _make_evaluator(monkeypatch, sample_shard=(1, 2))

    tok0 = FakeTokenizer()
    tok0.responses = ['<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>']
    tok1 = FakeTokenizer()
    tok1.responses = ['<tool_call>{"name": "calc", "arguments": {"a": 1, "b": 2}}</tool_call>']

    m0 = evaluator_r0.evaluate(FakeModel(), tok0)
    m1 = evaluator_r1.evaluate(FakeModel(), tok1)

    # Two samples, sharded across two ranks → 1 per rank.
    assert m0["tool_call/_count"] == 1.0
    assert m1["tool_call/_count"] == 1.0


def test_evaluate_skips_long_prompts(monkeypatch):
    evaluator = _make_evaluator(monkeypatch, max_prompt_tokens=10)
    tok = FakeTokenizer()
    tok.responses = ["never reached"]
    model = FakeModel()

    metrics = evaluator.evaluate(model, tok)

    # Stub prompts render to long strings (well over 10 tokens), so both
    # samples are skipped.
    assert metrics["tool_call/_count"] == 0.0
    assert model.generate_calls == 0


def test_evaluate_handles_generate_exception(monkeypatch):
    evaluator = _make_evaluator(monkeypatch)
    tok = FakeTokenizer()
    tok.responses = ['<tool_call>{"name": "calc", "arguments": {"a": 1, "b": 2}}</tool_call>']

    class FlakyModel(FakeModel):
        def generate(self, **kwargs):
            self.generate_calls += 1
            if self.generate_calls == 1:
                raise RuntimeError("CUDA OOM (simulated)")
            return super().generate(**kwargs)

    model = FlakyModel()
    metrics = evaluator.evaluate(model, tok)

    # Sample 1: generate raises -> skipped; decode never advances.
    # Sample 2: generate succeeds; decode returns responses[0], which is
    # the calc tool_call matching the second sample's ground truth.
    assert metrics["tool_call/_count"] == 1.0
    assert metrics["tool_call/name_correct"] == 1.0


def test_evaluate_passes_tools_to_template(monkeypatch):
    evaluator = _make_evaluator(monkeypatch)
    tok = FakeTokenizer()
    tok.responses = ["", ""]
    evaluator.evaluate(FakeModel(), tok)

    assert tok.tools_seen[0] == [{"name": "get_weather"}]
    assert tok.tools_seen[1] == [{"name": "calc"}]


def test_evaluate_empty_eval_set(monkeypatch):
    monkeypatch.setattr(ev_mod, "make_agent_chat_eval_samples", lambda **_: [])
    evaluator = ToolCallAccuracyEvaluator(path="dummy")
    metrics = evaluator.evaluate(FakeModel(), FakeTokenizer())

    assert metrics["tool_call/_count"] == 0.0
    # All metrics default to 0 when nothing scored.
    assert metrics["tool_call/name_correct"] == 0.0


def test_metric_prefix_is_applied(monkeypatch):
    evaluator = _make_evaluator(monkeypatch, metric_prefix="eval/agent")
    tok = FakeTokenizer()
    tok.responses = ["", ""]
    metrics = evaluator.evaluate(FakeModel(), tok)
    assert "eval/agent/_count" in metrics
    assert "eval/agent/name_correct" in metrics
