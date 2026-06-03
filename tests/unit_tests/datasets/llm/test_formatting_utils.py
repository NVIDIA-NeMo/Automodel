#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from nemo_automodel.components.datasets.llm import formatting_utils


class _CountingTokenizer:
    """Minimal tokenizer stub: each message renders to a fixed token count.

    Records how many times ``apply_chat_template`` is called so a test can
    assert the mask builder skips the full-conversation re-tokenization.
    """

    def __init__(self, tokens_per_message=2):
        self.calls = 0
        self.tokens_per_message = tokens_per_message

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=False,
        padding=False,
        truncation=None,
        max_length=None,
    ):
        self.calls += 1
        n = len(messages) * self.tokens_per_message
        return {"input_ids": list(range(n)), "attention_mask": [1] * n}


def _conversation():
    return [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]


def test_multiturn_mask_full_length_matches_recompute():
    # Passing full_length must produce a byte-identical mask to the recompute path.
    formatted = _conversation()
    input_ids = list(range(2 * len(formatted)))

    baseline = formatting_utils._build_multiturn_assistant_mask(_CountingTokenizer(), formatted, input_ids)
    optimized = formatting_utils._build_multiturn_assistant_mask(
        _CountingTokenizer(), formatted, input_ids, full_length=len(input_ids)
    )
    assert optimized == baseline
    # assistant turns at idx 1 (tokens [2, 4)) and idx 3 (tokens [6, 8))
    assert baseline == [0, 0, 1, 1, 0, 0, 1, 1]


def test_multiturn_mask_full_length_skips_full_retokenize():
    # When the dialogue ends on an assistant turn, the closing boundary is the
    # full conversation and must be served from full_length, not a fresh call.
    formatted = _conversation()
    input_ids = list(range(2 * len(formatted)))

    baseline_tok = _CountingTokenizer()
    formatting_utils._build_multiturn_assistant_mask(baseline_tok, formatted, input_ids)

    optimized_tok = _CountingTokenizer()
    formatting_utils._build_multiturn_assistant_mask(optimized_tok, formatted, input_ids, full_length=len(input_ids))
    assert optimized_tok.calls == baseline_tok.calls - 1


def test_multiturn_mask_requires_assistant():
    # The no-assistant guard is preserved.
    formatted = [{"role": "user", "content": "a"}]
    with pytest.raises(AssertionError, match="At least one assistant message"):
        formatting_utils._build_multiturn_assistant_mask(_CountingTokenizer(), formatted, [0, 1])
