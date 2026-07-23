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

"""Tests for the ``_text_tokens`` precompute worker.

The counts written here feed LengthGroupedSampler and the packing estimate, so
the set of roles counted must match ``_convert_sharegpt_to_conversation``.
"""

import json

import pytest

from scripts import precompute_tokens


class _WordTokenizer:
    """Whitespace tokenizer: one token per word, so counts are readable."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


@pytest.fixture
def word_tokenizer(monkeypatch):
    monkeypatch.setattr(precompute_tokens, "_worker_tokenizer", _WordTokenizer())


def _run(samples, columns=None, tags=None):
    lines = [json.dumps(s) + "\n" for s in samples]
    results = precompute_tokens._process_chunk((0, lines, columns or {}, tags or {}))
    return [json.loads(line) for _, line in results]


def test_system_turn_is_counted(word_tokenizer):
    """A system turn contributes to ``_text_tokens`` like user/assistant turns."""
    with_system, without_system = _run(
        [
            {
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hello there"},
                    {"role": "assistant", "content": "hi"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "hello there"},
                    {"role": "assistant", "content": "hi"},
                ]
            },
        ],
    )

    assert without_system["_text_tokens"] == 3
    assert with_system["_text_tokens"] == 5


def test_unknown_role_is_not_counted(word_tokenizer):
    """Roles outside the mapped set stay excluded from the estimate."""
    (sample,) = _run(
        [
            {
                "messages": [
                    {"role": "tool", "content": "ignored payload here"},
                    {"role": "user", "content": "hello there"},
                    {"role": "assistant", "content": "hi"},
                ]
            }
        ],
    )

    assert sample["_text_tokens"] == 3


def test_custom_system_tag(word_tokenizer):
    """``system_tag`` remaps the system role like the other role tags."""
    (sample,) = _run(
        [
            {
                "conversations": [
                    {"from": "sys", "value": "be terse"},
                    {"from": "human", "value": "hello there"},
                    {"from": "gpt", "value": "hi"},
                ]
            }
        ],
        columns={"messages": "conversations"},
        tags={
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "sys",
        },
    )

    assert sample["_text_tokens"] == 5


def test_media_tags_stripped_only_from_user_turns(word_tokenizer):
    """Only user media placeholders are removed from the token-count estimate."""
    with_media, empty = _run(
        [
            {
                "messages": [
                    {"role": "system", "content": "system <image> prompt"},
                    {"role": "user", "content": "<image> describe this"},
                    {"role": "assistant", "content": "a <video> cat"},
                ]
            },
            {"messages": [{"role": "user", "content": "<image>"}]},
        ],
    )

    assert with_media["_text_tokens"] == 8
    assert empty["_text_tokens"] == 0
