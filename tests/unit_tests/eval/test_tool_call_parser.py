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

import pytest

from nemo_automodel.components.eval.tool_call_parser import (
    ParsedToolCall,
    compute_sample_metrics,
    parse_tool_calls,
)


# ---------- parse_tool_calls: wrapper formats ----------


def test_parse_qwen_single_call():
    text = (
        '<tool_call>\n'
        '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n'
        '</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Tokyo"}
    assert calls[0].arguments_valid_json is True


def test_parse_qwen_string_encoded_arguments():
    text = (
        '<tool_call>'
        '{"name": "calc", "arguments": "{\\"a\\": 1, \\"b\\": 2}"}'
        '</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].arguments == {"a": 1, "b": 2}
    assert calls[0].arguments_valid_json is True


def test_parse_qwen_multiple_calls():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]


def test_parse_qwen_malformed_json_records_invalid():
    text = '<tool_call>{not json}</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name is None
    assert calls[0].arguments_valid_json is False


def test_parse_llama_python_tag():
    text = (
        '<|python_tag|>{"name": "search", "parameters": {"q": "cats"}}<|eom_id|>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "cats"}
    assert calls[0].arguments_valid_json is True


def test_parse_llama_python_tag_no_closing():
    text = '<|python_tag|>{"name": "search", "parameters": {"q": "cats"}}'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "search"


def test_parse_mistral_tool_calls_array():
    text = (
        '[TOOL_CALLS]['
        '{"name": "a", "arguments": {"k": 1}},'
        '{"name": "b", "arguments": {}}'
        ']'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[0].arguments == {"k": 1}


def test_parse_harmony_gpt_oss():
    text = (
        '<|channel|>commentary to=functions.lookup '
        '<|message|>{"city": "Paris"}'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "lookup"
    assert calls[0].arguments == {"city": "Paris"}


def test_parse_generic_json_fallback():
    text = 'I think I should call {"name": "tool", "arguments": {"k": 1}} now.'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "tool"
    assert calls[0].arguments == {"k": 1}


def test_parse_braces_in_strings_are_ignored():
    text = 'noise {"name": "t", "arguments": {"q": "a { brace }"}} more'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].arguments == {"q": "a { brace }"}


def test_parse_no_call_returns_empty():
    assert parse_tool_calls("just a plain answer, no tools.") == []


def test_parse_empty_string():
    assert parse_tool_calls("") == []


def test_parse_qwen_wins_over_generic_fallback():
    text = (
        '<tool_call>{"name": "wrapped", "arguments": {}}</tool_call>\n'
        '{"name": "stray", "arguments": {}}'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["wrapped"]


# ---------- compute_sample_metrics ----------


def _call(name, args, valid=True):
    return ParsedToolCall(name=name, arguments=args, arguments_valid_json=valid, raw="")


def test_metrics_perfect_match():
    pred = [_call("f", {"a": 1, "b": 2})]
    gt = [{"name": "f", "arguments": {"a": 1, "b": 2}}]
    m = compute_sample_metrics(pred, gt)
    assert m["has_call"] == 1.0
    assert m["name_correct"] == 1.0
    assert m["args_json_valid"] == 1.0
    assert m["args_field_recall"] == 1.0
    assert m["args_field_precision"] == 1.0
    assert m["args_exact_match"] == 1.0


def test_metrics_wrong_name():
    pred = [_call("g", {"a": 1})]
    gt = [{"name": "f", "arguments": {"a": 1}}]
    m = compute_sample_metrics(pred, gt)
    assert m["name_correct"] == 0.0
    # args_exact_match is independent of name — args still match.
    assert m["args_exact_match"] == 1.0
    assert m["args_field_recall"] == 1.0


def test_metrics_partial_args_recall_precision():
    pred = [_call("f", {"a": 1, "x": 9})]
    gt = [{"name": "f", "arguments": {"a": 1, "b": 2}}]
    m = compute_sample_metrics(pred, gt)
    assert m["args_field_recall"] == pytest.approx(0.5)
    assert m["args_field_precision"] == pytest.approx(0.5)
    assert m["args_exact_match"] == 0.0


def test_metrics_invalid_json_args():
    pred = [_call(None, {}, valid=False)]
    gt = [{"name": "f", "arguments": {"a": 1}}]
    m = compute_sample_metrics(pred, gt)
    assert m["has_call"] == 1.0
    assert m["args_json_valid"] == 0.0
    assert m["args_field_recall"] == 0.0


def test_metrics_no_pred_call():
    pred = []
    gt = [{"name": "f", "arguments": {"a": 1}}]
    m = compute_sample_metrics(pred, gt)
    assert m["has_call"] == 0.0
    assert m["name_correct"] == 0.0
    assert m["args_field_recall"] == 0.0


def test_metrics_empty_gt_args():
    pred = [_call("f", {})]
    gt = [{"name": "f", "arguments": {}}]
    m = compute_sample_metrics(pred, gt)
    assert m["args_exact_match"] == 1.0
    assert m["args_field_recall"] == 1.0
    assert m["args_field_precision"] == 1.0


def test_metrics_gt_args_as_json_string():
    pred = [_call("f", {"a": 1})]
    gt = [{"name": "f", "arguments": '{"a": 1}'}]
    m = compute_sample_metrics(pred, gt)
    assert m["args_exact_match"] == 1.0


def test_metrics_no_gt():
    assert compute_sample_metrics([_call("f", {})], []) == {
        "has_call": 0.0,
        "name_correct": 0.0,
        "args_json_valid": 0.0,
        "args_field_recall": 0.0,
        "args_field_precision": 0.0,
        "args_exact_match": 0.0,
    }


# ---------- nested JSON arguments survive balanced extraction ----------


def test_parse_qwen_nested_object_args():
    text = (
        '<tool_call>'
        '{"name": "search", "arguments": {"query": {"text": "x", "lang": "en"}}}'
        '</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"query": {"text": "x", "lang": "en"}}
    assert calls[0].arguments_valid_json is True


def test_parse_harmony_nested_object_args():
    text = (
        '<|channel|>commentary to=functions.search '
        '<|message|>{"query": {"text": "x", "lang": "en"}, "k": 5}'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"query": {"text": "x", "lang": "en"}, "k": 5}


def test_parse_llama_nested_object_args():
    text = (
        '<|python_tag|>'
        '{"name": "search", "parameters": {"filter": {"a": 1, "b": [2, 3]}}}'
        '<|eom_id|>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].arguments == {"filter": {"a": 1, "b": [2, 3]}}


def test_parse_mistral_array_valued_arguments():
    text = (
        '[TOOL_CALLS]['
        '{"name": "f", "arguments": {"items": [1, 2, 3], "meta": {"k": "v"}}}'
        ']'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].arguments == {"items": [1, 2, 3], "meta": {"k": "v"}}


def test_parse_harmony_brace_in_string_does_not_truncate():
    text = (
        '<|channel|>commentary to=functions.echo '
        '<|message|>{"text": "value with } closing brace inside"}'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].arguments == {"text": "value with } closing brace inside"}


# ---------- parallel tool calls: multi-GT alignment ----------


def test_metrics_parallel_pred_matches_all_gt():
    pred = [_call("a", {"x": 1}), _call("b", {"y": 2})]
    gt = [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]
    m = compute_sample_metrics(pred, gt)
    assert m["has_call"] == 1.0
    assert m["name_correct"] == 1.0
    assert m["args_exact_match"] == 1.0


def test_metrics_missing_second_parallel_call_penalized():
    # Model emits only the first of two expected parallel calls.
    # Position 0 scores 1.0, position 1 is missing -> 0.0; mean is 0.5.
    pred = [_call("a", {"x": 1})]
    gt = [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]
    m = compute_sample_metrics(pred, gt)
    assert m["has_call"] == pytest.approx(0.5)
    assert m["name_correct"] == pytest.approx(0.5)
    assert m["args_exact_match"] == pytest.approx(0.5)


def test_metrics_extra_pred_calls_are_ignored():
    pred = [_call("a", {"x": 1}), _call("b", {"y": 2}), _call("c", {})]
    gt = [{"name": "a", "arguments": {"x": 1}}]
    m = compute_sample_metrics(pred, gt)
    # Only first position is scored; extras don't drag the mean down.
    assert m["name_correct"] == 1.0
    assert m["args_exact_match"] == 1.0


def test_metrics_first_wrong_second_right():
    pred = [_call("wrong", {}), _call("b", {"y": 2})]
    gt = [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]
    m = compute_sample_metrics(pred, gt)
    assert m["name_correct"] == pytest.approx(0.5)
    assert m["args_exact_match"] == pytest.approx(0.5)
