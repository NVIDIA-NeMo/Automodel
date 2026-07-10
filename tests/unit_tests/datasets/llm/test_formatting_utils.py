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


def test_multiturn_mask_unpadded_full_ids_matches_recompute():
    # Passing unpadded_full_ids must produce a byte-identical mask to the recompute path.
    formatted = _conversation()
    input_ids = list(range(2 * len(formatted)))

    baseline = formatting_utils._build_multiturn_assistant_mask(_CountingTokenizer(), formatted, input_ids)
    optimized = formatting_utils._build_multiturn_assistant_mask(
        _CountingTokenizer(), formatted, input_ids, unpadded_full_ids=list(input_ids)
    )
    assert optimized == baseline
    # assistant turns at idx 1 (tokens [2, 4)) and idx 3 (tokens [6, 8))
    assert baseline == [0, 0, 1, 1, 0, 0, 1, 1]


def test_multiturn_mask_unpadded_full_ids_skips_full_retokenize():
    # When the dialogue ends on an assistant turn, the closing boundary is the
    # full conversation and must be served from unpadded_full_ids, not a fresh call.
    formatted = _conversation()
    input_ids = list(range(2 * len(formatted)))

    baseline_tok = _CountingTokenizer()
    formatting_utils._build_multiturn_assistant_mask(baseline_tok, formatted, input_ids)

    optimized_tok = _CountingTokenizer()
    formatting_utils._build_multiturn_assistant_mask(
        optimized_tok, formatted, input_ids, unpadded_full_ids=list(input_ids)
    )
    assert optimized_tok.calls == baseline_tok.calls - 1


def test_multiturn_mask_requires_assistant():
    # The no-assistant guard is preserved.
    formatted = [{"role": "user", "content": "a"}]
    with pytest.raises(AssertionError, match="At least one assistant message"):
        formatting_utils._build_multiturn_assistant_mask(_CountingTokenizer(), formatted, [0, 1])


def test_mask_labels_to_last_turn_keeps_only_final_run():
    # Two supervised runs separated by an ignored run; only the last survives.
    labels = [-100, 1, 2, -100, -100, 3, 4, -100]
    formatting_utils._mask_labels_to_last_turn(labels)
    assert labels == [-100, -100, -100, -100, -100, 3, 4, -100]


def test_mask_labels_to_last_turn_single_run_unchanged():
    labels = [-100, -100, 5, 6, 7]
    formatting_utils._mask_labels_to_last_turn(labels)
    assert labels == [-100, -100, 5, 6, 7]


def test_mask_labels_to_last_turn_no_supervised_tokens_is_noop():
    labels = [-100, -100, -100]
    formatting_utils._mask_labels_to_last_turn(labels)
    assert labels == [-100, -100, -100]


def test_mask_labels_to_last_turn_on_binary_mask():
    # On a 0/1 assistant mask (ignore_index=0): keep only the last run of 1s.
    mask = [0, 1, 1, 0, 1, 1, 1, 0]
    formatting_utils._mask_labels_to_last_turn(mask, ignore_index=0)
    assert mask == [0, 0, 0, 0, 1, 1, 1, 0]


def test_is_consistent_render_prefix():
    is_prefix = formatting_utils._is_consistent_render_prefix
    assert is_prefix([], [1, 2, 3])  # empty prefix is trivially consistent
    assert is_prefix([1, 2], [1, 2, 3])  # exact token prefix
    assert is_prefix([1, 9], [1, 2, 3])  # divergence only at the final prefix token is tolerated
    assert not is_prefix([1, 9, 3], [1, 2, 3])  # divergence before the final token
    assert not is_prefix([1, 2, 3, 4], [1, 2, 3])  # prefix render longer than the full render


class _RewritingTokenizer:
    """Simulates Qwen3-style chat templates that rewrite earlier turns.

    Assistant reasoning tokens render only for turns after the last user
    message. Once a later user turn arrives, earlier assistant turns re-render
    without their reasoning tokens, so a prefix render is not a token prefix of
    the full-conversation render.
    """

    _ROLE_TOKEN = 90
    _REASONING_TOKEN = 91
    eos_token_id = 2
    chat_template = "<dummy reasoning_content template without generation keyword>"

    def __init__(self):
        self._vocab = {}
        self._cursor = 100

    def _id_for_token(self, tok):
        if tok not in self._vocab:
            self._vocab[tok] = self._cursor
            self._cursor += 1
        return self._vocab[tok]

    def apply_chat_template(self, messages, **kwargs):
        last_user = max((i for i, msg in enumerate(messages) if msg["role"] == "user"), default=-1)
        ids = []
        for i, msg in enumerate(messages):
            ids.append(self._ROLE_TOKEN)
            if msg["role"] == "assistant" and i > last_user and msg.get("reasoning_content"):
                ids.extend([self._REASONING_TOKEN] * len(msg["reasoning_content"].split()))
            ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _rewriting_conversation():
    return [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1 done", "reasoning_content": "step one two"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2 done", "reasoning_content": "step three"},
    ]


def test_multiturn_mask_raises_on_rewriting_template():
    # A prefix render that is not a token prefix of the full render would put
    # assistant spans at the wrong positions; the builder must refuse instead.
    tok = _RewritingTokenizer()
    formatted = _rewriting_conversation()
    full_ids = tok.apply_chat_template(formatted)["input_ids"]

    with pytest.raises(ValueError, match="rewrites earlier turns"):
        formatting_utils._build_multiturn_assistant_mask(tok, formatted, full_ids)


def test_format_chat_template_raises_on_rewriting_template():
    # End to end: the default answer-only path surfaces the template problem
    # instead of silently supervising tool/user tokens.
    tok = _RewritingTokenizer()
    with pytest.raises(ValueError, match="rewrites earlier turns"):
        formatting_utils.format_chat_template(
            tok,
            formatted_text=_rewriting_conversation(),
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            answer_only_loss_mask=True,
        )


def test_reasoning_mask_skips_rewritten_turns_and_masks_stable_ones(caplog):
    # The early assistant turn is re-rendered without reasoning in the full
    # conversation (nothing to mask there); the final turn is stable and its
    # reasoning tokens must still be masked.
    tok = _RewritingTokenizer()
    formatted = _rewriting_conversation()
    full_ids = tok.apply_chat_template(formatted)["input_ids"]

    with caplog.at_level("WARNING"):
        mask = formatting_utils._build_reasoning_mask(tok, formatted, full_ids)

    assert "Skipping reasoning_content masking for assistant message 1" in caplog.text
    reasoning_positions = [i for i, token in enumerate(full_ids) if token == tok._REASONING_TOKEN]
    assert reasoning_positions  # the final turn's reasoning is rendered
    assert [i for i, value in enumerate(mask) if value] == reasoning_positions

    # Passing the reference render explicitly must not change the result.
    explicit = formatting_utils._build_reasoning_mask(tok, formatted, full_ids, unpadded_full_ids=list(full_ids))
    assert explicit == mask
