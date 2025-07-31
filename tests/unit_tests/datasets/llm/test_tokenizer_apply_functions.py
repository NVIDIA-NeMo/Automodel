# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Unit tests for the tokenizer helper utilities in
``nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset``.

The helpers are pure functions so we employ *minimal* tokenizer stubs that
implement just the behaviour required by the utilities.  The goal is to verify
that the helpers correctly

1. build the *input_ids*, *labels* and *loss_mask* fields; and
2. apply the *answer-only* masking logic when requested.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import (
    _apply_tokenizer_plain,
    _apply_tokenizer_with_chat_template,
)


class _StubTokenizerPlain:  # noqa: D401 – minimal interface only
    """A trivial whitespace tokenizer with deterministic ids.

    The tokenizer maps *new* tokens to monotonically increasing integers.
    ``bos_token_id`` and ``eos_token_id`` are fixed to *1* and *2*
    respectively and are automatically added when ``add_special_tokens`` is
    *True* (default mirrors 🤗 *transformers* API).
    """

    bos_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}
        self._cursor: int = 3  # start after BOS/EOS
        # *chat_template* is intentionally **absent** so that the code path for
        # ``_apply_tokenizer_plain`` is exercised.

    def _id_for_token(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = self._cursor
            self._cursor += 1
        return self._vocab[tok]

    def __call__(self, text: str, *, add_special_tokens: bool = True):  # type: ignore[override]
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        ids.extend(self._id_for_token(tok) for tok in text.strip().split())
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return {"input_ids": ids}


class _StubTokenizerChat(_StubTokenizerPlain):  # noqa: D401
    """Extends :class:`_StubTokenizerPlain` with chat-template support."""

    chat_template = "<dummy>"
    _start_of_turn_token = "<sot>"
    _start_of_turn_token_id = 99

    def apply_chat_template(self, messages):  # type: ignore[override]
        """Very small surrogate that encodes ``messages`` as id sequence.

        Encoding scheme:
        ``[SOT] <user tokens> [SOT] <assistant tokens> <EOS>``
        where ``[SOT]`` is the *start-of-turn* marker (id=99).
        """
        ids: List[int] = [self._start_of_turn_token_id]
        ids.extend(self._id_for_token(tok) for tok in messages[0]["content"].split())
        ids.append(self._start_of_turn_token_id)
        ids.extend(self._id_for_token(tok) for tok in messages[1]["content"].split())
        ids.append(self.eos_token_id)
        return ids

    # ``_apply_tokenizer_with_chat_template`` will call the tokenizer on the
    # *start-of-turn* token with ``add_special_tokens=False`` to retrieve the id.
    def __call__(self, text: str, *, add_special_tokens: bool = False):  # type: ignore[override]
        if text == self._start_of_turn_token:
            return {"input_ids": [self._start_of_turn_token_id]}
        return super().__call__(text, add_special_tokens=add_special_tokens)


def test_apply_tokenizer_plain_answer_only_mask():
    tok = _StubTokenizerPlain()
    context = "Context"
    question = "Why?"
    answer = "Because."

    out = _apply_tokenizer_plain(tok, context, question, answer, answer_only_loss_mask=True)

    # Basic keys/length checks
    assert set(out) == {"input_ids", "labels", "loss_mask"}
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["loss_mask"])

    # Prompt/answer masking logic 
    prompt_text = f"{context} {question} "
    prompt_ids = tok(prompt_text)["input_ids"]
    # The helper strips the trailing EOS token from the prompt
    if prompt_ids and prompt_ids[-1] == tok.eos_token_id:
        prompt_ids = prompt_ids[:-1]

    answer_ids = tok(answer.strip())["input_ids"]
    # The helper strips the leading BOS from the answer
    if answer_ids and answer_ids[0] == tok.bos_token_id:
        answer_ids = answer_ids[1:]

    expected_zeros = len(prompt_ids) - 1  # as per implementation
    expected_ones = len(answer_ids)

    assert out["loss_mask"].count(0) == expected_zeros
    assert out["loss_mask"].count(1) == expected_ones


def test_apply_tokenizer_plain_full_loss_mask():
    tok = _StubTokenizerPlain()
    out = _apply_tokenizer_plain(tok, "ctx", "Q?", "A.", answer_only_loss_mask=False)

    # Loss mask should be *all ones*
    assert all(v == 1 for v in out["loss_mask"])


def test_apply_tokenizer_chat_template_answer_only_mask():
    tok = _StubTokenizerChat()
    ctx, qst, ans = "Some context", "Life?", "42"
    out = _apply_tokenizer_with_chat_template(
        tok, ctx, qst, ans, start_of_turn_token=tok._start_of_turn_token, answer_only_loss_mask=True
    )

    # Basic invariants
    assert set(out) == {"input_ids", "labels", "loss_mask"}
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["loss_mask"])

    # The first chunk (user prompt) should be masked out (zeros)
    first_one_idx = out["loss_mask"].index(1)
    assert first_one_idx > 0  # we expect at least one *zero*
    assert all(v == 0 for v in out["loss_mask"][:first_one_idx])
    assert all(v == 1 for v in out["loss_mask"][first_one_idx:])


def test_apply_tokenizer_chat_template_full_loss_mask():
    tok = _StubTokenizerChat()
    out = _apply_tokenizer_with_chat_template(
        tok,
        "ctx",
        "Q?",
        "A.",
        start_of_turn_token=tok._start_of_turn_token,
        answer_only_loss_mask=False,
    )

    assert all(v == 1 for v in out["loss_mask"]) 