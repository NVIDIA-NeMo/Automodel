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
"""Generic tool-call parser for evaluating agent SFT outputs.

Different chat templates wrap tool calls in different syntax:

* Qwen / Hermes / FunctionGemma / Gemma 3 / GLM-4:
  ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``
* Llama 3.1+: ``<|python_tag|>{"name": ..., "parameters": ...}<|eom_id|>``
* Mistral: ``[TOOL_CALLS][{...}, {...}]``
* Harmony / GPT-OSS: ``<|channel|>commentary to=functions.NAME<|message|>{...}``

This parser tries each known wrapper, then falls back to a generic JSON
object scan. It is intentionally permissive: malformed JSON, missing
wrappers, or unknown formats degrade gracefully and never raise.

The companion :func:`compute_sample_metrics` compares parser output
against ground-truth tool calls and produces 0/1 (or fractional)
indicators that average cleanly across a dataset.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParsedToolCall:
    """One tool call extracted from generated text.

    Attributes:
        name: function name if extracted, otherwise ``None``.
        arguments: parsed arguments dict; empty when JSON is invalid.
        arguments_valid_json: ``True`` if ``arguments`` parsed cleanly.
        raw: the source substring this was parsed from.
    """

    name: Optional[str]
    arguments: Dict[str, Any]
    arguments_valid_json: bool
    raw: str


# Harmony / GPT-OSS prefix anchor; the JSON body is then extracted with a
# balanced-brace scanner so nested ``{...}`` arguments survive.
_HARMONY_ANCHOR_RE = re.compile(
    r"<\|channel\|>\s*commentary\s+to=functions\.(?P<name>[\w\-]+)[^<]*"
    r"<\|message\|>\s*",
)
# Qwen / Hermes / FunctionGemma / Gemma 3 / GLM-4. The closing ``</tool_call>``
# is a strong anchor so non-greedy capture is safe here.
_QWEN_RE = re.compile(r"<tool_call>\s*(.+?)\s*</tool_call>", re.DOTALL)
# Llama 3.1+: ``<|python_tag|>{...}<|eom_id|>`` (closing tag optional at EOS).
# The body is a balanced object: locate the start ``{`` and scan for its match
# so nested ``{...}`` arguments aren't truncated.
_LLAMA_ANCHOR_RE = re.compile(r"<\|python_tag\|>\s*")
# Mistral: ``[TOOL_CALLS][{...}, {...}]``. Balanced-bracket scanner finds the
# end of the array regardless of nesting.
_MISTRAL_ANCHOR_RE = re.compile(r"\[TOOL_CALLS\]\s*")


def _extract_balanced(text: str, start: int, opener: str, closer: str) -> Optional[str]:
    """Return the substring from ``text[start]`` (which must be ``opener``)
    through its matching ``closer``, skipping over chars inside JSON strings.

    Returns ``None`` if ``text[start]`` is not ``opener`` or the span is
    unbalanced.
    """
    if start >= len(text) or text[start] != opener:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_args(args_value: Any) -> Tuple[Dict[str, Any], bool]:
    """Normalize an ``arguments`` field to a dict.

    Accepts a dict (passthrough) or a JSON-encoded string. Returns the
    parsed dict alongside a flag indicating whether the source was a
    well-formed JSON object.
    """
    if isinstance(args_value, dict):
        return args_value, True
    if isinstance(args_value, str):
        try:
            parsed = json.loads(args_value)
        except json.JSONDecodeError:
            return {}, False
        if isinstance(parsed, dict):
            return parsed, True
        return {}, False
    return {}, False


def _from_call_object(obj: Dict[str, Any], raw: str) -> Optional[ParsedToolCall]:
    """Build a :class:`ParsedToolCall` from a ``{"name": ..., "arguments": ...}`` dict.

    Llama 3.1 emits ``parameters`` instead of ``arguments``; both are accepted.
    Returns ``None`` when ``name`` is missing or non-string.
    """
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args_raw = obj.get("arguments", obj.get("parameters", {}))
    args, valid = _coerce_args(args_raw)
    return ParsedToolCall(name=name, arguments=args, arguments_valid_json=valid, raw=raw)


def _iter_balanced_json_objects(text: str) -> Iterator[str]:
    """Yield substrings that look like balanced top-level JSON objects.

    Skips characters inside JSON string literals (so braces inside strings
    don't unbalance the count). Designed for fallback extraction when no
    known wrapper matches.
    """
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, c in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : i + 1]
                start = -1


def _parse_qwen_style(text: str) -> List[ParsedToolCall]:
    calls: List[ParsedToolCall] = []
    for m in _QWEN_RE.finditer(text):
        inner = m.group(1)
        try:
            obj = json.loads(inner)
        except json.JSONDecodeError:
            calls.append(ParsedToolCall(name=None, arguments={}, arguments_valid_json=False, raw=m.group(0)))
            continue
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    parsed = _from_call_object(item, m.group(0))
                    if parsed is not None:
                        calls.append(parsed)
        elif isinstance(obj, dict):
            parsed = _from_call_object(obj, m.group(0))
            if parsed is not None:
                calls.append(parsed)
    return calls


def _parse_llama_style(text: str) -> List[ParsedToolCall]:
    calls: List[ParsedToolCall] = []
    for m in _LLAMA_ANCHOR_RE.finditer(text):
        body = _extract_balanced(text, m.end(), "{", "}")
        if body is None:
            continue
        raw_span = text[m.start() : m.end() + len(body)]
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            calls.append(ParsedToolCall(name=None, arguments={}, arguments_valid_json=False, raw=raw_span))
            continue
        if isinstance(obj, dict):
            parsed = _from_call_object(obj, raw_span)
            if parsed is not None:
                calls.append(parsed)
    return calls


def _parse_mistral_style(text: str) -> List[ParsedToolCall]:
    calls: List[ParsedToolCall] = []
    for m in _MISTRAL_ANCHOR_RE.finditer(text):
        body = _extract_balanced(text, m.end(), "[", "]")
        if body is None:
            continue
        try:
            arr = json.loads(body)
        except json.JSONDecodeError:
            continue
        raw_span = text[m.start() : m.end() + len(body)]
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, dict):
                    parsed = _from_call_object(item, raw_span)
                    if parsed is not None:
                        calls.append(parsed)
    return calls


def _parse_harmony_style(text: str) -> List[ParsedToolCall]:
    calls: List[ParsedToolCall] = []
    for m in _HARMONY_ANCHOR_RE.finditer(text):
        body = _extract_balanced(text, m.end(), "{", "}")
        if body is None:
            continue
        args, valid = _coerce_args(body)
        raw_span = text[m.start() : m.end() + len(body)]
        calls.append(
            ParsedToolCall(
                name=m.group("name"),
                arguments=args,
                arguments_valid_json=valid,
                raw=raw_span,
            )
        )
    return calls


def _parse_generic_json(text: str) -> List[ParsedToolCall]:
    """Last-resort fallback: scan for any JSON object with a ``name`` field."""
    calls: List[ParsedToolCall] = []
    for raw_json in _iter_balanced_json_objects(text):
        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            parsed = _from_call_object(obj, raw_json)
            if parsed is not None:
                calls.append(parsed)
    return calls


def parse_tool_calls(text: str) -> List[ParsedToolCall]:
    """Extract every tool call from a generated model response.

    Wrappers are tried in order of specificity; the first wrapper that
    yields any match wins. If no wrapper matches, a generic JSON-object
    scan is used. Returns an empty list when no plausible tool call is
    present.

    Args:
        text: raw decoded text from ``model.generate()``.

    Returns:
        Parsed tool calls in document order. Possibly empty.
    """
    if not text:
        return []

    for parser in (
        _parse_harmony_style,
        _parse_qwen_style,
        _parse_llama_style,
        _parse_mistral_style,
    ):
        result = parser(text)
        if result:
            return result

    return _parse_generic_json(text)


def _coerce_gt_args(args_value: Any) -> Dict[str, Any]:
    """Normalize a ground-truth ``arguments`` field to a dict."""
    if isinstance(args_value, dict):
        return args_value
    if isinstance(args_value, str):
        try:
            parsed = json.loads(args_value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _score_one_pair(pred: Optional[ParsedToolCall], gt: Dict[str, Any]) -> Dict[str, float]:
    """Score a single (pred, gt) tool-call pair. ``pred`` may be ``None``."""
    metrics = {
        "has_call": 0.0,
        "name_correct": 0.0,
        "args_json_valid": 0.0,
        "args_field_recall": 0.0,
        "args_field_precision": 0.0,
        "args_exact_match": 0.0,
    }
    if pred is None:
        return metrics

    gt_name = gt.get("name")
    gt_args = _coerce_gt_args(gt.get("arguments", {}))

    metrics["has_call"] = 1.0
    metrics["name_correct"] = 1.0 if pred.name == gt_name else 0.0
    metrics["args_json_valid"] = 1.0 if pred.arguments_valid_json else 0.0

    pred_keys = set(pred.arguments.keys()) if pred.arguments_valid_json else set()
    gt_keys = set(gt_args.keys())

    if gt_keys:
        metrics["args_field_recall"] = len(pred_keys & gt_keys) / len(gt_keys)
    else:
        metrics["args_field_recall"] = 1.0 if not pred_keys else 0.0

    if pred_keys:
        metrics["args_field_precision"] = len(pred_keys & gt_keys) / len(pred_keys)
    else:
        metrics["args_field_precision"] = 1.0 if not gt_keys else 0.0

    if pred.arguments_valid_json and pred.arguments == gt_args:
        metrics["args_exact_match"] = 1.0
    return metrics


def compute_sample_metrics(
    pred_calls: List[ParsedToolCall],
    gt_calls: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute per-sample tool-call metrics across all GT positions.

    Predicted calls are aligned **positionally** against the ground-truth
    list: ``pred_calls[i]`` is scored against ``gt_calls[i]``. Missing
    predictions (``i >= len(pred_calls)``) contribute zeros across every
    metric for that position, so a model that emits only one of two
    parallel tool calls is correctly penalized on the missing call.

    Extra predictions beyond ``len(gt_calls)`` are ignored. All values
    are in ``[0.0, 1.0]`` so callers can ``mean()`` across a dataset.

    Returned keys:

    * ``has_call``: prediction exists at this position.
    * ``name_correct``: predicted call name equals GT name.
    * ``args_json_valid``: prediction had valid JSON arguments.
    * ``args_field_recall``: fraction of GT argument keys present in pred.
    * ``args_field_precision``: fraction of pred argument keys present in GT.
    * ``args_exact_match``: pred arguments dict equals GT arguments dict.

    Args:
        pred_calls: output of :func:`parse_tool_calls`.
        gt_calls: ground-truth list of ``{"name": str, "arguments": dict|str}``.
    """
    metric_keys = (
        "has_call",
        "name_correct",
        "args_json_valid",
        "args_field_recall",
        "args_field_precision",
        "args_exact_match",
    )
    if not gt_calls:
        return {k: 0.0 for k in metric_keys}

    sums = {k: 0.0 for k in metric_keys}
    for i, gt in enumerate(gt_calls):
        pred = pred_calls[i] if i < len(pred_calls) else None
        per = _score_one_pair(pred, gt)
        for k in metric_keys:
            sums[k] += per[k]

    n = len(gt_calls)
    return {k: sums[k] / n for k in metric_keys}
