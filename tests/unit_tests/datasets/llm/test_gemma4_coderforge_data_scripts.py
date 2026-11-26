# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import importlib.util
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parents[4] / "examples/long_context_validation/gemma4_31B/data"


def _load_script(name):
    """Load a CoderForge data script (the example directory is not a package)."""
    spec = importlib.util.spec_from_file_location(f"gemma4_coderforge_{name}", DATA_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prefilter():
    return _load_script("prefilter_dataset")


def test_clean_message_keeps_json_string_arguments_and_fills_empty_ones(prefilter):
    # Arguments stay JSON strings (what the JSONL cache and ChatDataset carry); NeMoAutoTokenizer
    # decodes JSON objects for the Gemma4 template, which raises on any other string, so a call
    # without arguments must become "{}" rather than "".
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}},
            {"function": {"name": "think", "arguments": None}},
            {"function": {"name": "finish", "arguments": ""}},
            {"function": {"name": "edit", "arguments": {"path": "a.py"}}},
        ],
        "function_call": None,
    }
    out = prefilter.clean_message(msg)
    assert out["content"] == ""
    assert [call["function"] for call in out["tool_calls"]] == [
        {"name": "bash", "arguments": '{"cmd": "ls"}'},
        {"name": "think", "arguments": "{}"},
        {"name": "finish", "arguments": "{}"},
        {"name": "edit", "arguments": '{"path": "a.py"}'},
    ]
    assert [call["id"] for call in out["tool_calls"]] == ["a", "call_1", "call_2", "call_3"]


def test_analyzed_cache_path_keys_on_template_contents(prefilter, tmp_path):
    # n_tokens depends on the rendering, so editing the template in place must miss the old cache,
    # while the same contents keep hitting it.
    template = tmp_path / "chat_template.jinja"
    template.write_text("v1")

    def path():
        return prefilter.analyzed_cache_path(str(tmp_path), "org/ds", "train", "/models/gemma", str(template))

    first = path()
    assert path() == first
    template.write_text("v2")
    assert path() != first
    # A template that is not a file (inline Jinja, or None for the checkpoint's own) keys on its value.
    assert prefilter.analyzed_cache_path(str(tmp_path), "org/ds", "train", "/models/gemma", None).endswith(
        "_analyzed.jsonl"
    )


def test_validate_accepts_either_gemma4_stop_token():
    # CoderForge trajectories end on a `finish` tool call, so the final supervised token is the
    # <|tool_response> stop token (50) rather than <turn|> (106); both are generation stop tokens.
    validate_data = _load_script("validate_data")
    ends_on_tool_call = {"input_ids": [2, 7, 8, 50], "labels": [-100, 7, 8, 50]}
    ends_on_turn = {"input_ids": [2, 7, 106, 9], "labels": [-100, 7, 106, 9]}
    truncated = {"input_ids": [2, 7, 8, 9], "labels": [-100, 7, 8, 9]}

    _, passed, failed, _, n = validate_data.validate(
        [ends_on_tool_call, ends_on_turn, truncated],
        pad_token_id=0,
        stop_token_ids=[106, 50],
        terminator_window=2,
        num_samples=0,
    )
    assert n == 3
    assert passed["final_turn_terminated"] == 2 and failed["final_turn_terminated"] == 1
    assert passed["stop_token_in_supervised"] == 2 and failed["stop_token_in_supervised"] == 1
