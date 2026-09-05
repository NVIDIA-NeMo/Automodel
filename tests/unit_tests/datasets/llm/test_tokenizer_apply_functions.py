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

import unicodedata
from typing import Dict, List

import pytest

from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    _build_generation_prompt_mask,
    _match_generation_prompt,
    _package_tokenized_example,
    _pad_to_seq_length,
    _perturbed_assistant_message,
    _tokenize_chat,
    _warned_add_pad_token,
    _warned_generation_prompt,
    format_chat_template,
    format_prompt_completion,
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
    # Mirror HF behavior flag used by formatting utils when computing prompt length
    add_bos_token = True

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}
        self._cursor: int = 3  # start after BOS/EOS
        # *chat_template* is intentionally **absent** so that the code path for
        # ``format_prompt_completion`` is exercised.

    def _id_for_token(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = self._cursor
            self._cursor += 1
        return self._vocab[tok]

    def __call__(self, text: str, *, add_special_tokens: bool = True, padding=None, truncation=None, max_length=None):  # type: ignore[override]
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        ids.extend(self._id_for_token(tok) for tok in text.split())
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return {"input_ids": ids}


class _StubTokenizerChat(_StubTokenizerPlain):  # noqa: D401
    """Extends :class:`_StubTokenizerPlain` with chat-template support."""

    chat_template = "<dummy {% generation %} template>"
    _start_of_turn_token = "<sot>"
    _start_of_turn_token_id = 99

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        """Very small surrogate that encodes ``messages`` as id sequence.

        Encoding scheme:
        ``[SOT] <prompt tokens (system+user)> [SOT] <assistant tokens> <EOS>``
        where ``[SOT]`` is the *start-of-turn* marker (id=99).
        """
        # Separate prompt messages (system, user) from assistant messages
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]

        # Build ids: [SOT] + prompt tokens + [SOT] + assistant tokens + [EOS]
        ids: List[int] = [self._start_of_turn_token_id]

        # Add all prompt tokens (system + user)
        prompt_token_count = 0
        for msg in prompt_messages:
            tokens = msg["content"].split()
            ids.extend(self._id_for_token(tok) for tok in tokens)
            prompt_token_count += len(tokens)

        # Add second SOT and assistant tokens
        ids.append(self._start_of_turn_token_id)
        assistant_token_count = 0
        for msg in assistant_messages:
            tokens = msg["content"].split()
            ids.extend(self._id_for_token(tok) for tok in tokens)
            assistant_token_count += len(tokens)

        ids.append(self.eos_token_id)

        # Handle return_dict parameter
        if kwargs.get("return_dict", False):
            result = {"input_ids": ids}
            # Handle return_assistant_tokens_mask parameter
            if kwargs.get("return_assistant_tokens_mask", False):
                # Create mask: first SOT and prompt tokens are 0 (masked),
                # second SOT and assistant tokens are 1 (not masked)
                mask = [0] * (1 + prompt_token_count)  # first SOT + prompt tokens
                mask += [1] * (1 + assistant_token_count + 1)  # second SOT + assistant tokens + EOS
                result["assistant_masks"] = mask
            return result
        return ids

    # ``format_chat_template`` will call the tokenizer on the
    # *start-of-turn* token with ``add_special_tokens=False`` to retrieve the id.
    def __call__(self, text: str, *, add_special_tokens: bool = False):  # type: ignore[override]
        if text == self._start_of_turn_token:
            return {"input_ids": [self._start_of_turn_token_id]}
        return super().__call__(text, add_special_tokens=add_special_tokens)


class _StubTokenizerChatWithReasoning(_StubTokenizerPlain):  # noqa: D401
    """Chat-template tokenizer with generation blocks and explicit reasoning output."""

    chat_template = "<dummy reasoning_content {% generation %} template>"
    _start_of_turn_token_id = 99

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = [self._start_of_turn_token_id]
        assistant_mask: List[int] = [0]

        for msg in messages:
            role_token = self._id_for_token(f"<{msg['role']}>")
            ids.append(role_token)
            is_assistant = msg["role"] == "assistant"
            assistant_mask.append(1 if is_assistant else 0)

            if msg["role"] != "assistant":
                for tok in msg["content"].split():
                    ids.append(self._id_for_token(tok))
                    assistant_mask.append(0)
                continue

            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                ids.append(self._id_for_token("<think>"))
                assistant_mask.append(1)
                for tok in reasoning.split():
                    ids.append(self._id_for_token(tok))
                    assistant_mask.append(1)
                ids.append(self._id_for_token("</think>"))
                assistant_mask.append(1)

            for tok in msg["content"].split():
                ids.append(self._id_for_token(tok))
                assistant_mask.append(1)

        ids.append(self.eos_token_id)
        assistant_mask.append(1 if any(msg["role"] == "assistant" for msg in messages) else 0)

        if kwargs.get("return_dict", False):
            result = {"input_ids": ids}
            if kwargs.get("return_assistant_tokens_mask", False):
                result["assistant_masks"] = assistant_mask
            return result
        return ids


class _RecordingPaddingTokenizer(_StubTokenizerPlain):
    """Stub tokenizer that records ``padding_side`` during ``__call__``.

    Used to assert that ``format_prompt_completion`` flips the side to
    ``"right"`` for the duration of the tokenize call and restores the
    original value after — including when the original is ``"left"`` (the
    transformers v5.8 ``LlamaTokenizer`` class default).
    """

    padding_side = "right"

    def __call__(self, text, *, add_special_tokens=True, padding=None, truncation=None, max_length=None):
        self.padding_side_during_call = self.padding_side
        return super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )


@pytest.mark.parametrize("initial_side", ["left", "right"])
def test_format_prompt_completion_forces_right_padding_and_restores(initial_side):
    """Covers the padding_side save/restore wrapper for both initial sides.

    Each call goes through every line of the wrapper (save, set, try, finally,
    restore), so the parametrize ensures the inner ``if _saved_padding_side
    is not None`` branches are exercised regardless of which session codecov
    looks at.
    """
    tok = _RecordingPaddingTokenizer()
    tok.padding_side = initial_side
    out = format_prompt_completion(
        tok,
        "Context Q?",
        "A.",
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        answer_only_loss_mask=True,
    )
    assert tok.padding_side_during_call == "right"
    assert tok.padding_side == initial_side
    assert "input_ids" in out and "labels" in out


def test_format_prompt_completion_without_padding_side_attr_is_a_noop_for_the_wrapper():
    """Covers the False branches of the padding_side wrapper.

    When the tokenizer has no ``padding_side`` attribute (e.g.
    ``_StubTokenizerPlain``), ``getattr`` returns ``None`` and the
    ``if _saved_padding_side is not None`` set/restore branches must
    short-circuit without touching the tokenizer.
    """
    tok = _StubTokenizerPlain()
    assert not hasattr(tok, "padding_side")
    out = format_prompt_completion(
        tok,
        "Context Q?",
        "A.",
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        answer_only_loss_mask=True,
    )
    # No attribute was created on the tokenizer as a side effect.
    assert not hasattr(tok, "padding_side")
    assert "input_ids" in out and "labels" in out


def testformat_prompt_completion_answer_only_mask():
    tok = _StubTokenizerPlain()
    context = "Context"
    question = "Why?"
    answer = "Because."
    prompt = f"{context} {question} "
    out = format_prompt_completion(
        tok, prompt, answer, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id, answer_only_loss_mask=True
    )

    # Basic keys/length checks
    del out["___PAD_TOKEN_IDS___"]
    assert set(out) == {"input_ids", "labels", "attention_mask"}
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])

    # Prompt/answer masking logic
    prompt_text = f"{context} {question} "
    # The implementation tokenizes prompt without special tokens to calculate mask
    prompt_ids_no_special = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_text = f"{context} {question} {answer}"
    # @akoumparouli: remove the eos token
    full_text_ids = tok(full_text)["input_ids"][:-1]
    # bos + 3; eos has been removed
    assert len(full_text_ids) == 4
    assert len(full_text_ids) == len(out["input_ids"])

    # The format_prompt_completion adds BOS to len_prompt_ids, then shifts labels by 1
    # So expected masked tokens = len(prompt_ids_no_special) + 1 (BOS) - 1 (shift) = len(prompt_ids_no_special)
    expected_zeros = len(prompt_ids_no_special)
    expected_ones = len(out["labels"]) - expected_zeros

    num_ignore_labels = out["labels"].count(-100)
    assert num_ignore_labels == expected_zeros, (out, out["labels"][-4:], len(out["labels"]), num_ignore_labels)
    assert len(out["labels"]) - num_ignore_labels == expected_ones


def testformat_prompt_completion_full_loss_mask():
    tok = _StubTokenizerPlain()
    context, question, answer = "ctx", "Q?", "A."
    prompt = f"{context} {question} "
    out = format_prompt_completion(
        tok, prompt, answer, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id, answer_only_loss_mask=False
    )

    # Loss mask should be *all ones*
    del out["___PAD_TOKEN_IDS___"]
    assert set(out) == {"input_ids", "labels", "attention_mask"}
    assert len(out["labels"]) == len(out["input_ids"]) == len(out["attention_mask"])
    assert out["labels"].count(-100) == 0


def test_apply_tokenizer_chat_template_answer_only_mask():
    tok = _StubTokenizerChat()
    ctx, qst, ans = "Some context", "Life?", "42"
    out = format_chat_template(
        tok,
        formatted_text=[
            {"role": "system", "content": ctx},
            {"role": "user", "content": qst},
            {"role": "assistant", "content": ans},
        ],
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

    # Basic invariants
    del out["___PAD_TOKEN_IDS___"]
    assert set(out) == {"input_ids", "labels", "attention_mask"}
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])

    # The first chunk (user prompt) should be masked out (zeros)
    assert out["input_ids"][0] == tok._start_of_turn_token_id
    pos = out["input_ids"][1:].index(tok._start_of_turn_token_id)
    assert pos > 0
    # we assume first [first start_of_turn_token_id, second start_of_turn_token_id) to be all -100
    assert all(v == -100 for v in out["labels"][:pos])
    # and the rest to be != -100
    assert all(v != -100 for v in out["labels"][pos:])


def test_apply_tokenizer_chat_template_full_loss_mask():
    tok = _StubTokenizerChat()
    out = format_chat_template(
        tok,
        formatted_text=[
            {"role": "system", "content": "ctx"},
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "A."},
        ],
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    del out["___PAD_TOKEN_IDS___"]
    assert set(out) == {"input_ids", "labels", "attention_mask"}
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])
    assert all(v == 1 for v in out["attention_mask"])


def test_apply_chat_template_can_mask_reasoning_content_generation_kwd():
    tok = _StubTokenizerChatWithReasoning()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "reasoning_content": "think step", "content": "final answer"},
    ]

    unmasked = format_chat_template(
        tok,
        formatted_text=[m.copy() for m in messages],
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        mask_reasoning_content=False,
    )
    masked = format_chat_template(
        tok,
        formatted_text=[m.copy() for m in messages],
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        mask_reasoning_content=True,
    )

    think_id = tok._id_for_token("think")
    step_id = tok._id_for_token("step")
    final_id = tok._id_for_token("final")
    answer_id = tok._id_for_token("answer")

    assert think_id in masked["input_ids"] and step_id in masked["input_ids"]
    assert masked["labels"][masked["input_ids"].index(think_id)] == -100
    assert masked["labels"][masked["input_ids"].index(step_id)] == -100
    assert masked["labels"][masked["input_ids"].index(final_id)] != -100
    assert masked["labels"][masked["input_ids"].index(answer_id)] != -100
    assert sum(1 for value in masked["labels"] if value != -100) < sum(
        1 for value in unmasked["labels"] if value != -100
    )


class _StubTokenizerChatNoGen:
    """Chat-template tokenizer WITHOUT generation keyword; returns no assistant mask."""

    eos_token_id = 2
    chat_template = "<dummy template without generation keyword>"
    _start_of_turn_token_id = 99

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}
        self._cursor: int = 3  # start after BOS/EOS

    def _id_for_token(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = self._cursor
            self._cursor += 1
        return self._vocab[tok]

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        # Compose ids as:
        # [SOT] + <all non-assistant message tokens> + [SOT] + <assistant tokens (if any)>
        ids: List[int] = [self._start_of_turn_token_id]
        # prompt tokens (system + user, etc.)
        for msg in messages:
            if msg["role"] == "assistant":
                break
            ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
        # delimiter before assistant section
        ids.append(self._start_of_turn_token_id)
        # assistant tokens (if present)
        assistant_started = False
        for msg in messages:
            if msg["role"] == "assistant":
                assistant_started = True
            if assistant_started:
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
        ids.append(self.eos_token_id)
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


def test_apply_chat_template_manual_mask_without_generation_kwd():
    # Tokenizer without generation keyword in template
    tok = _StubTokenizerChatNoGen()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what now"},
        {"role": "assistant", "content": "answer goes here"},
    ]

    # Compute expected prompt length as used by the implementation
    prompt_only = messages[:-1]
    tokenized_prompt = tok.apply_chat_template(prompt_only, return_dict=True)
    len_prompt_ids = len(tokenized_prompt["input_ids"])

    out = format_chat_template(
        tok,
        formatted_text=[m.copy() for m in messages],
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        answer_only_loss_mask=True,
    )

    # Basic structure
    pad_info = out.pop("___PAD_TOKEN_IDS___")
    assert set(out) == {"input_ids", "labels", "attention_mask"}
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])
    assert pad_info["labels"] == -100

    # Since labels drop the first token (treated as BOS/SOT), expected ignored labels:
    expected_ignored = max(0, len_prompt_ids - 1)
    assert out["labels"].count(-100) == expected_ignored
    # Sanity: there must be supervised tokens (assistant section)
    assert expected_ignored < len(out["labels"])
    # Number of supervised tokens (exclude -100) should equal number of assistant tokens.
    assistant_tokens = sum(len(str(m["content"]).split()) for m in messages if m["role"] == "assistant")
    num_supervised = sum(1 for v in out["labels"] if v != -100)
    assert num_supervised == assistant_tokens


def test_apply_chat_template_manual_mask_raises_when_last_not_assistant():
    tok = _StubTokenizerChatNoGen()
    # Last message is not assistant → assertion should trigger
    bad_messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
    ]
    with pytest.raises(AssertionError):
        _ = format_chat_template(
            tok,
            formatted_text=[m.copy() for m in bad_messages],
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            answer_only_loss_mask=True,
        )


class _StubTokenizerChatNoGenMultiTurn(_StubTokenizerChatNoGen):
    """Tokenizer without generation blocks that preserves role boundaries."""

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = [self._start_of_turn_token_id]
        for msg in messages:
            ids.append(self._id_for_token(f"<{msg['role']}>"))
            ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
        ids.append(self.eos_token_id)
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


def test_apply_chat_template_manual_mask_without_generation_kwd_multiturn_tool_calling():
    tok = _StubTokenizerChatNoGenMultiTurn()
    messages = [
        {"role": "user", "content": "need weather"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "rain likely"},
        {"role": "assistant", "content": "bring umbrella"},
    ]

    out = format_chat_template(
        tok,
        formatted_text=[m.copy() for m in messages],
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        answer_only_loss_mask=True,
    )

    num_supervised = sum(1 for value in out["labels"] if value != -100)
    assert num_supervised == 4


def test_apply_chat_template_warns_when_reasoning_content_is_unused(caplog):
    tok = _StubTokenizerChatNoGen()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "reasoning_content": "hidden chain of thought"},
    ]

    with caplog.at_level("WARNING"):
        format_chat_template(
            tok,
            formatted_text=[m.copy() for m in messages],
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            answer_only_loss_mask=True,
        )

    assert "reasoning_content" in caplog.text


class _StubTokenizerChatNoGenWithReasoning(_StubTokenizerChatNoGen):
    """Non-generation tokenizer that renders reasoning_content before assistant content."""

    chat_template = "<dummy reasoning_content template>"

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = [self._start_of_turn_token_id]
        for msg in messages:
            ids.append(self._id_for_token(f"<{msg['role']}>"))
            if msg["role"] == "assistant":
                reasoning = msg.get("reasoning_content", "")
                if reasoning:
                    ids.append(self._id_for_token("<think>"))
                    ids.extend(self._id_for_token(tok) for tok in reasoning.split())
                    ids.append(self._id_for_token("</think>"))
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
            else:
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
        ids.append(self.eos_token_id)
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


def test_apply_chat_template_can_mask_reasoning_content_without_generation_kwd():
    tok = _StubTokenizerChatNoGenWithReasoning()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "reasoning_content": "think step", "content": "final answer"},
    ]

    masked = format_chat_template(
        tok,
        formatted_text=[m.copy() for m in messages],
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        answer_only_loss_mask=True,
        mask_reasoning_content=True,
    )

    think_id = tok._id_for_token("think")
    step_id = tok._id_for_token("step")
    final_id = tok._id_for_token("final")
    answer_id = tok._id_for_token("answer")

    assert masked["labels"][masked["input_ids"].index(think_id)] == -100
    assert masked["labels"][masked["input_ids"].index(step_id)] == -100
    assert masked["labels"][masked["input_ids"].index(final_id)] != -100
    assert masked["labels"][masked["input_ids"].index(answer_id)] != -100


def test_format_chat_template_train_on_last_turn_only_masks_earlier_turns():
    # Only the final assistant turn stays supervised; earlier turns are dropped.
    tok = _StubTokenizerChatNoGenMultiTurn()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "alpha beta"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "gamma delta"},
    ]
    common = dict(eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id, answer_only_loss_mask=True)

    all_turns = format_chat_template(tok, formatted_text=[m.copy() for m in messages], **common)
    last_only = format_chat_template(
        tok, formatted_text=[m.copy() for m in messages], train_on_last_turn_only=True, **common
    )

    all_supervised = {v for v in all_turns["labels"] if v != -100}
    last_supervised = {v for v in last_only["labels"] if v != -100}

    for word in ("alpha", "beta", "gamma", "delta"):
        assert tok._id_for_token(word) in all_supervised
    assert tok._id_for_token("gamma") in last_supervised
    assert tok._id_for_token("delta") in last_supervised
    assert tok._id_for_token("alpha") not in last_supervised
    assert tok._id_for_token("beta") not in last_supervised


class _StubTokenizerChatNoGenReasoningSplitsContent(_StubTokenizerChatNoGen):
    """Assistant content renders both BEFORE and AFTER the reasoning block.

    Mirrors templates that emit text, then a ``<think>`` block, then more text
    (or a tool call), so a masked reasoning span sits between two supervised
    content spans within the same turn.
    """

    chat_template = "<dummy reasoning_content template>"

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = [self._start_of_turn_token_id]
        for msg in messages:
            ids.append(self._id_for_token(f"<{msg['role']}>"))
            if msg["role"] == "assistant" and msg.get("reasoning_content"):
                words = str(msg["content"]).split()
                head, tail = words[:1], words[1:]
                ids.extend(self._id_for_token(t) for t in head)
                ids.append(self._id_for_token("<think>"))
                ids.extend(self._id_for_token(t) for t in str(msg["reasoning_content"]).split())
                ids.append(self._id_for_token("</think>"))
                ids.extend(self._id_for_token(t) for t in tail)
            else:
                ids.extend(self._id_for_token(t) for t in str(msg["content"]).split())
        ids.append(self.eos_token_id)
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


def test_format_chat_template_last_turn_only_keeps_content_around_masked_reasoning():
    # Regression: the final turn renders as [head][<think>why</think>][tail].
    # The last-turn restriction must run on the hole-free mask, so masking the
    # reasoning hole does not also drop the supervised "head" before it (the old
    # order kept only the last contiguous run, i.e. just "tail").
    tok = _StubTokenizerChatNoGenReasoningSplitsContent()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "earlierturn"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "head tail", "reasoning_content": "why"},
    ]

    out = format_chat_template(
        tok,
        formatted_text=[m.copy() for m in messages],
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        answer_only_loss_mask=True,
        mask_reasoning_content=True,
        train_on_last_turn_only=True,
    )

    supervised = {v for v in out["labels"] if v != -100}
    # Both content spans of the last turn stay supervised.
    assert tok._id_for_token("head") in supervised
    assert tok._id_for_token("tail") in supervised
    # The reasoning hole is masked, and the earlier assistant turn is dropped.
    assert tok._id_for_token("why") not in supervised
    assert tok._id_for_token("earlierturn") not in supervised


# pad_token_id == eos_token_id overlap tests


class _StubTokNoPad:
    """Tokenizer with NO pad_token_id — forces _add_pad_token to fall back."""

    eos_token_id = 2
    eos_token = "</s>"
    pad_token_id = None
    pad_token = None
    name_or_path = "stub-no-pad"


class _StubTokDistinctPad:
    """Tokenizer with a dedicated pad_token_id distinct from eos."""

    eos_token_id = 2
    eos_token = "</s>"
    pad_token_id = 0
    pad_token = "<pad>"
    name_or_path = "stub-distinct-pad"


class _StubTokPadEqualsEos:
    """Tokenizer where pad_token_id is explicitly equal to eos_token_id."""

    eos_token_id = 2
    eos_token = "</s>"
    pad_token_id = 2
    pad_token = "</s>"
    name_or_path = "stub-pad-equals-eos"


class TestAddPadToken:
    """Unit tests for _add_pad_token covering pad==eos scenarios."""

    @pytest.fixture(autouse=True)
    def _reset_warn_state(self):
        _warned_add_pad_token.clear()

    def test_no_pad_token_falls_back_to_eos(self):
        tok = _StubTokNoPad()
        result = _add_pad_token(tok)
        assert result is None
        assert tok.pad_token_id == tok.eos_token_id

    def test_no_pad_token_sets_pad_token_string(self):
        tok = _StubTokNoPad()
        _add_pad_token(tok)
        assert tok.pad_token == tok.eos_token

    def test_distinct_pad_returns_pad_token_id(self):
        tok = _StubTokDistinctPad()
        result = _add_pad_token(tok)
        assert result == 0
        assert tok.pad_token_id == 0

    def test_pad_equals_eos_returns_pad_token_id(self):
        tok = _StubTokPadEqualsEos()
        result = _add_pad_token(tok)
        assert result == 2

    def test_no_pad_warns_fallback(self, caplog):
        tok = _StubTokNoPad()
        with caplog.at_level("WARNING"):
            _add_pad_token(tok)
        assert any("falling back to eos_token_id" in r.message for r in caplog.records)

    def test_pad_equals_eos_warns_overlap(self, caplog):
        tok = _StubTokPadEqualsEos()
        with caplog.at_level("WARNING"):
            _add_pad_token(tok)
        assert any("pad_token_id" in r.message and "== eos_token_id" in r.message for r in caplog.records)

    def test_distinct_pad_no_overlap_warning(self, caplog):
        tok = _StubTokDistinctPad()
        with caplog.at_level("WARNING"):
            _add_pad_token(tok)
        assert not any("== eos_token_id" in r.message for r in caplog.records)

    def test_fallback_caller_pattern(self):
        """Verify the ``_add_pad_token(tok) or eos_token_id`` pattern used by all datasets."""
        tok = _StubTokNoPad()
        pad_token_id = _add_pad_token(tok) or tok.eos_token_id
        assert pad_token_id == tok.eos_token_id


class TestPadToSeqLength:
    """Edge-case tests for _pad_to_seq_length."""

    def test_no_padding_needed(self):
        assert _pad_to_seq_length([1, 2, 3], 0, 3) == [1, 2, 3]

    def test_pad_with_minus_100(self):
        result = _pad_to_seq_length([10, 20], -100, 5)
        assert result == [10, 20, -100, -100, -100]

    def test_pad_with_eos_token_id(self):
        result = _pad_to_seq_length([10, 20], 2, 4)
        assert result == [10, 20, 2, 2]

    def test_pad_single_element(self):
        result = _pad_to_seq_length([42], -100, 3)
        assert result == [42, -100, -100]

    def test_empty_list(self):
        result = _pad_to_seq_length([], -100, 3)
        assert result == [-100, -100, -100]


class _StubTokForPackage:
    """Minimal tokenizer for _package_tokenized_example tests."""

    eos_token_id = 2
    chat_template = None

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id


class TestPackageTokenizedExamplePadEos:
    """Verify _package_tokenized_example safety when pad_token_id == eos_token_id."""

    def _make_example(self, pad_token_id, seq_length=None, padding="do_not_pad"):
        tok = _StubTokForPackage(pad_token_id)
        eos = tok.eos_token_id
        # Simulate: [BOS=1, 10, 11, 12, EOS=2]
        input_ids = [1, 10, 11, 12, eos]
        assistant_masks = [0, 0, 0, 1, 1]
        return _package_tokenized_example(
            tokenizer=tok,
            input_ids=input_ids,
            assistant_masks=assistant_masks,
            eos_token_id=eos,
            pad_token_id=pad_token_id,
            seq_length=seq_length,
            padding=padding,
        )

    def test_labels_padded_with_minus_100_when_pad_equals_eos(self):
        out = self._make_example(pad_token_id=2, seq_length=8, padding="max_length")
        pad_region = out["labels"][4:]
        assert all(v == -100 for v in pad_region), f"Labels must be padded with -100, got {pad_region}"

    def test_labels_padded_with_minus_100_when_pad_distinct(self):
        out = self._make_example(pad_token_id=0, seq_length=8, padding="max_length")
        pad_region = out["labels"][4:]
        assert all(v == -100 for v in pad_region)

    def test_input_ids_padded_with_pad_token_id_when_overlap(self):
        out = self._make_example(pad_token_id=2, seq_length=8, padding="max_length")
        pad_region = out["input_ids"][4:]
        assert all(v == 2 for v in pad_region)

    def test_pad_token_ids_metadata_always_minus_100_for_labels(self):
        for pid in [0, 2]:
            out = self._make_example(pad_token_id=pid, seq_length=8, padding="max_length")
            assert out["___PAD_TOKEN_IDS___"]["labels"] == -100

    def test_attention_mask_zeros_in_padded_region(self):
        out = self._make_example(pad_token_id=2, seq_length=8, padding="max_length")
        content_mask = out["attention_mask"][:4]
        pad_mask = out["attention_mask"][4:]
        assert all(v == 1 for v in content_mask)
        assert all(v == 0 for v in pad_mask)

    def test_no_padding_when_do_not_pad(self):
        out = self._make_example(pad_token_id=2)
        assert len(out["input_ids"]) == 4
        assert len(out["labels"]) == 4
        assert -100 not in out["labels"][2:]

    def test_eos_in_supervised_labels_when_pad_equals_eos(self):
        out = self._make_example(pad_token_id=2, seq_length=8, padding="max_length")
        supervised = [v for v in out["labels"] if v != -100]
        assert 2 in supervised, "Real EOS must appear in supervised label region"

    def test_eos_not_in_padding_labels(self):
        out = self._make_example(pad_token_id=2, seq_length=8, padding="max_length")
        pad_region = out["labels"][4:]
        assert 2 not in pad_region, "EOS token id must not appear as label padding"


class TestPackageTokenizedExamplePrePaddedInput:
    """Tests for _package_tokenized_example when input_ids arrive already padded.

    This happens when a tokenizer's apply_chat_template is called with
    padding="max_length" — the returned input_ids already contain trailing
    pad tokens.  _package_tokenized_example must detect these and set
    attention_mask=0 at those positions.
    """

    def test_attention_mask_zeros_for_pre_padded_distinct_pad(self):
        """Pre-padded input with pad_token_id != eos_token_id."""
        tok = _StubTokForPackage(pad_token_id=0)
        eos = tok.eos_token_id  # 2
        # Simulate tokenizer output already padded to length 8:
        # [BOS=1, A=10, B=11, EOS=2, PAD=0, PAD=0, PAD=0, PAD=0]
        input_ids = [1, 10, 11, eos, 0, 0, 0, 0]
        assistant_masks = [0, 0, 1, 1, 0, 0, 0, 0]
        out = _package_tokenized_example(
            tokenizer=tok,
            input_ids=input_ids,
            assistant_masks=assistant_masks,
            eos_token_id=eos,
            pad_token_id=0,
            seq_length=None,
            padding="do_not_pad",
        )
        # Content length is computed on the original input (4 real tokens),
        # then reduced by 1 for the next-token shift → 3 ones.
        assert out["attention_mask"] == [
            1,
            1,
            1,
            0,
            0,
            0,
            0,
        ], f"Expected zeros at pre-padded positions, got {out['attention_mask']}"

    def test_attention_mask_zeros_for_pre_padded_pad_equals_eos(self):
        """Pre-padded input where pad_token_id == eos_token_id.

        The real trailing EOS should keep attention_mask=1, but subsequent
        pad tokens (same id) should get attention_mask=0.
        """
        tok = _StubTokForPackage(pad_token_id=2)
        eos = tok.eos_token_id  # 2
        # [BOS=1, A=10, B=11, EOS=2, PAD=2, PAD=2, PAD=2]
        input_ids = [1, 10, 11, eos, 2, 2, 2]
        assistant_masks = [0, 0, 1, 1, 0, 0, 0]
        out = _package_tokenized_example(
            tokenizer=tok,
            input_ids=input_ids,
            assistant_masks=assistant_masks,
            eos_token_id=eos,
            pad_token_id=2,
            seq_length=None,
            padding="do_not_pad",
        )
        # Content length is computed on the original input (4 real tokens
        # including one trailing EOS), then reduced by 1 for the shift → 3 ones.
        assert out["attention_mask"] == [
            1,
            1,
            1,
            0,
            0,
            0,
        ], f"Expected one trailing EOS kept + zeros for pad, got {out['attention_mask']}"

    def test_attention_mask_no_padding_present(self):
        """No pre-padding — attention_mask should be all ones (existing behavior)."""
        tok = _StubTokForPackage(pad_token_id=0)
        eos = tok.eos_token_id  # 2
        input_ids = [1, 10, 11, eos]
        assistant_masks = [0, 0, 1, 1]
        out = _package_tokenized_example(
            tokenizer=tok,
            input_ids=input_ids,
            assistant_masks=assistant_masks,
            eos_token_id=eos,
            pad_token_id=0,
            seq_length=None,
            padding="do_not_pad",
        )
        # After [:-1]: input_ids = [1, 10, 11], no pad tokens
        assert out["attention_mask"] == [1, 1, 1]

    def test_pre_padded_then_further_padded_by_seq_length(self):
        """Input already partially padded, then _pad_to_seq_length extends further."""
        tok = _StubTokForPackage(pad_token_id=0)
        eos = tok.eos_token_id  # 2
        # Pre-padded to 6, but seq_length=10
        input_ids = [1, 10, 11, eos, 0, 0]
        assistant_masks = [0, 0, 1, 1, 0, 0]
        out = _package_tokenized_example(
            tokenizer=tok,
            input_ids=input_ids,
            assistant_masks=assistant_masks,
            eos_token_id=eos,
            pad_token_id=0,
            seq_length=10,
            padding="max_length",
        )
        # Content length computed on original (4 real), minus 1 for shift → 3 ones.
        # _pad_to_seq_length extends to 10.
        assert len(out["attention_mask"]) == 10
        assert out["attention_mask"][:3] == [1, 1, 1]
        assert all(v == 0 for v in out["attention_mask"][3:])


class _StubTokPadEosPlain(_StubTokenizerPlain):
    """Plain tokenizer (no chat template) where pad_token_id == eos_token_id."""

    pad_token_id = 2
    pad_token = "<eos>"


class _StubTokPadEosChat(_StubTokenizerChat):
    """Chat tokenizer where pad_token_id == eos_token_id."""

    pad_token_id = 2
    pad_token = "<eos>"


class _StubTokPadEosChatNoGen(_StubTokenizerChatNoGen):
    """Chat tokenizer (no generation kwd) where pad_token_id == eos_token_id."""

    pad_token_id = 2
    pad_token = "<eos>"


class TestFormatPromptCompletionPadEos:
    """Tests for format_prompt_completion when pad_token_id == eos_token_id."""

    def _format(self, tok, prompt, answer, seq_length=None, padding="do_not_pad"):
        eos = tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos
        return format_prompt_completion(
            tok,
            prompt,
            answer,
            eos_token_id=eos,
            pad_token_id=pad,
            seq_length=seq_length,
            padding=padding,
            answer_only_loss_mask=True,
        )

    def test_labels_never_padded_with_eos(self):
        tok = _StubTokPadEosPlain()
        out = self._format(tok, "Question: ", "Answer.", seq_length=20, padding="max_length")
        last_supervised = max((i for i, v in enumerate(out["labels"]) if v != -100), default=-1)
        pad_region = out["labels"][last_supervised + 1 :]
        assert len(pad_region) > 0, "Test requires padding to be present"
        assert all(v == -100 for v in pad_region), f"Label padding must use -100, not eos_token_id, got {pad_region}"

    def test_eos_survives_in_supervised_labels(self):
        tok = _StubTokPadEosPlain()
        out = self._format(tok, "Q ", "A.", seq_length=20, padding="max_length")
        supervised = [v for v in out["labels"] if v != -100]
        assert tok.eos_token_id in supervised

    def test_attention_mask_correct(self):
        tok = _StubTokPadEosPlain()
        out = self._format(tok, "Q ", "A.", seq_length=20, padding="max_length")
        assert len(out["attention_mask"]) == 20
        ones = sum(1 for v in out["attention_mask"] if v == 1)
        zeros = sum(1 for v in out["attention_mask"] if v == 0)
        assert ones > 0 and zeros > 0
        seen_zero = False
        for v in out["attention_mask"]:
            if v == 0:
                seen_zero = True
            elif seen_zero:
                pytest.fail("Attention mask not right-padded")

    def test_pad_token_ids_metadata(self):
        tok = _StubTokPadEosPlain()
        out = self._format(tok, "Q ", "A.", seq_length=20, padding="max_length")
        meta = out["___PAD_TOKEN_IDS___"]
        assert meta["labels"] == -100
        assert meta["attention_mask"] == 0
        assert meta["input_ids"] == tok.eos_token_id

    def test_no_padding_unaffected(self):
        tok = _StubTokPadEosPlain()
        out = self._format(tok, "Q ", "A.")
        assert -100 not in out["labels"] or out["labels"][0] == -100
        assert all(v == 1 for v in out["attention_mask"])

    def test_prompt_masked_answer_supervised(self):
        tok = _StubTokPadEosPlain()
        out = self._format(tok, "Context Question ", "Answer.", seq_length=20, padding="max_length")
        assert out["labels"][0] == -100, "First label (prompt) must be masked"
        supervised = [v for v in out["labels"] if v != -100]
        assert len(supervised) > 0, "Must have supervised (answer) tokens"


class _StubTokenizerChatTruncating(_StubTokenizerChat):
    """Chat tokenizer that respects max_length truncation like HF tokenizers."""

    def apply_chat_template(self, messages, **kwargs):
        result = super().apply_chat_template(messages, **kwargs)
        max_length = kwargs.get("max_length")
        if max_length is not None:
            if kwargs.get("return_dict", False):
                ids = result["input_ids"][:max_length]
                result["input_ids"] = ids
                if "assistant_masks" in result:
                    result["assistant_masks"] = result["assistant_masks"][:max_length]
            else:
                result = result[:max_length]
        return result


class TestFormatChatTemplateNoEosAfterTruncation:
    """EOS must NOT be appended when the sequence was truncated to seq_length.

    When apply_chat_template returns seq_length tokens (i.e. the sequence was
    truncated), appending EOS makes the total seq_length+1 which after
    BOS-removal in _package_tokenized_example produces exactly seq_length
    labels with no room for -100 padding.  The last label becomes the
    spurious EOS instead of -100.
    """

    def _messages(self):
        # Long enough content to exceed any small seq_length
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a b c d e f g h i j k l m n o p q r s t"},
            {"role": "assistant", "content": "x y z w v u"},
        ]

    def test_no_eos_appended_when_truncated_generation_kwd(self):
        tok = _StubTokenizerChatTruncating()
        seq_length = 10  # Force truncation
        out = format_chat_template(
            tok,
            [m.copy() for m in self._messages()],
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            seq_length=seq_length,
            padding="max_length",
            truncation=True,
        )
        # All labels must be exactly seq_length
        assert len(out["labels"]) == seq_length
        # The last label must be -100 (padding), NOT eos_token_id
        assert out["labels"][-1] == -100, (
            f"Last label should be -100 (padding) after truncation, got {out['labels'][-1]}"
        )

    def test_eos_still_appended_when_not_truncated(self):
        tok = _StubTokenizerChatTruncating()
        seq_length = 100  # Large enough — no truncation
        out = format_chat_template(
            tok,
            [m.copy() for m in self._messages()],
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            seq_length=seq_length,
            padding="max_length",
            truncation=True,
        )
        assert len(out["labels"]) == seq_length
        # EOS should be in the supervised region (not truncated, so EOS was appended)
        supervised = [v for v in out["labels"] if v != -100]
        assert tok.eos_token_id in supervised


class TestFormatChatTemplatePadEos:
    """Tests for format_chat_template when pad_token_id == eos_token_id."""

    def _messages(self):
        return [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is AI"},
            {"role": "assistant", "content": "Artificial intelligence"},
        ]

    def _format(self, tok, messages, seq_length=None, padding="do_not_pad"):
        eos = tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos
        return format_chat_template(
            tok,
            [m.copy() for m in messages],
            eos_token_id=eos,
            pad_token_id=pad,
            seq_length=seq_length,
            padding=padding,
        )

    def test_labels_never_padded_with_eos_generation_kwd(self):
        tok = _StubTokPadEosChat()
        out = self._format(tok, self._messages(), seq_length=30, padding="max_length")
        last_supervised = max((i for i, v in enumerate(out["labels"]) if v != -100), default=-1)
        pad_region = out["labels"][last_supervised + 1 :]
        assert len(pad_region) > 0, "Test requires padding to be present"
        assert all(v == -100 for v in pad_region)

    def test_eos_survives_in_supervised_generation_kwd(self):
        tok = _StubTokPadEosChat()
        out = self._format(tok, self._messages(), seq_length=30, padding="max_length")
        supervised = [v for v in out["labels"] if v != -100]
        assert tok.eos_token_id in supervised

    def test_labels_never_padded_with_eos_no_generation_kwd(self):
        tok = _StubTokPadEosChatNoGen()
        out = self._format(tok, self._messages(), seq_length=30, padding="max_length")
        last_supervised = max((i for i, v in enumerate(out["labels"]) if v != -100), default=-1)
        pad_region = out["labels"][last_supervised + 1 :]
        assert len(pad_region) > 0, "Test requires padding to be present"
        assert all(v == -100 for v in pad_region)

    def test_eos_survives_in_supervised_no_generation_kwd(self):
        tok = _StubTokPadEosChatNoGen()
        out = self._format(tok, self._messages(), seq_length=30, padding="max_length")
        supervised = [v for v in out["labels"] if v != -100]
        assert tok.eos_token_id in supervised

    def test_pad_token_ids_metadata(self):
        tok = _StubTokPadEosChat()
        out = self._format(tok, self._messages(), seq_length=30, padding="max_length")
        assert out["___PAD_TOKEN_IDS___"]["labels"] == -100
        assert out["___PAD_TOKEN_IDS___"]["input_ids"] == tok.eos_token_id

    def test_attention_mask_right_padded(self):
        tok = _StubTokPadEosChat()
        out = self._format(tok, self._messages(), seq_length=30, padding="max_length")
        seen_zero = False
        for v in out["attention_mask"]:
            if v == 0:
                seen_zero = True
            elif seen_zero:
                pytest.fail("Attention mask not right-padded")

    def test_all_lengths_match(self):
        for tok_cls in [_StubTokPadEosChat, _StubTokPadEosChatNoGen]:
            tok = tok_cls()
            out = self._format(tok, self._messages(), seq_length=30, padding="max_length")
            n = len(out["input_ids"])
            assert n == 30
            assert len(out["labels"]) == n
            assert len(out["attention_mask"]) == n


class TestContentLengthBranches:
    """Tests covering all branches of the content_length logic in _package_tokenized_example."""

    def _run(
        self, pad_token_id, input_ids, assistant_masks=None, eos_token_id=2, seq_length=None, padding="do_not_pad"
    ):
        tok = _StubTokForPackage(pad_token_id)
        tok.eos_token_id = eos_token_id
        if assistant_masks is None:
            assistant_masks = [1] * len(input_ids)
        return _package_tokenized_example(
            tokenizer=tok,
            input_ids=input_ids,
            assistant_masks=assistant_masks,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            seq_length=seq_length,
            padding=padding,
        )

    def test_content_length_pad_token_id_none(self):
        """When pad_token_id is None, all tokens are attended."""
        out = self._run(pad_token_id=None, input_ids=[1, 10, 11, 2], eos_token_id=2)
        assert out["attention_mask"] == [1, 1, 1]

    def test_content_length_single_token_input(self):
        """Single-token input produces empty output after shift."""
        out = self._run(pad_token_id=0, input_ids=[10], eos_token_id=2)
        assert out["attention_mask"] == []

    def test_content_length_pad_eq_eos_no_trailing(self):
        """pad==eos but no trailing eos/pad tokens — all content attended."""
        out = self._run(pad_token_id=2, input_ids=[1, 10, 11, 99], eos_token_id=2)
        assert out["attention_mask"] == [1, 1, 1]

    def test_content_length_pad_eq_eos_all_same(self):
        """All tokens are eos/pad — end=0, content_length=min(1,4)=1, after shift=0."""
        out = self._run(pad_token_id=2, input_ids=[2, 2, 2, 2], eos_token_id=2)
        assert out["attention_mask"] == [0, 0, 0]

    def test_content_length_pad_ne_eos_all_pad(self):
        """All tokens are pad (distinct from eos) — none attended."""
        out = self._run(pad_token_id=0, input_ids=[0, 0, 0, 0], eos_token_id=2)
        assert out["attention_mask"] == [0, 0, 0]

    def test_content_length_identical_for_padded_and_nonpadded(self):
        """Padded and non-padded inputs produce the same content_length."""
        padded_out = self._run(pad_token_id=2, input_ids=[1, 10, 11, 2, 2, 2], eos_token_id=2)
        nonpadded_out = self._run(pad_token_id=2, input_ids=[1, 10, 11, 2], eos_token_id=2)
        padded_content = sum(padded_out["attention_mask"])
        nonpadded_content = sum(nonpadded_out["attention_mask"])
        assert padded_content == nonpadded_content

    def test_labels_at_nonattended_positions_prompt_completion(self):
        """Labels at attention_mask=0 positions are -100 when using max_length padding."""
        tok = _StubTokForPackage(pad_token_id=0)
        tok.eos_token_id = 2
        input_ids = [1, 10, 11, 2]
        assistant_masks = [0, 0, 1, 1]
        out = _package_tokenized_example(
            tokenizer=tok,
            input_ids=input_ids,
            assistant_masks=assistant_masks,
            eos_token_id=2,
            pad_token_id=0,
            seq_length=8,
            padding="max_length",
        )
        for i in range(len(out["labels"])):
            if out["attention_mask"][i] == 0:
                assert out["labels"][i] == -100, f"Position {i}: attention_mask=0 but labels={out['labels'][i]}"

    def test_labels_preserved_when_no_padding(self):
        """Without padding, all labels are attended and none masked by padding."""
        out = self._run(
            pad_token_id=0,
            input_ids=[1, 10, 11, 2],
            assistant_masks=[1, 1, 1, 1],
            eos_token_id=2,
        )
        assert all(v == 1 for v in out["attention_mask"])
        assert -100 not in out["labels"]


# ---------------------------------------------------------------------------
# mask_generation_prompt
# ---------------------------------------------------------------------------


class _StubTokenizerChatNoGenThinking(_StubTokenizerChatNoGen):
    """Nemotron-style template without a generation keyword.

    Assistant turns render ``<assistant> <think> [reasoning] </think> content EOS``:
    a turn without ``reasoning_content`` still gets an empty ``<think> </think>``
    block. ``add_generation_prompt=True`` appends ``<assistant> <think>`` and, when
    ``enable_thinking`` is False, the closing ``</think>`` as well.
    """

    chat_template = "<dummy reasoning_content enable_thinking template>"
    # When True, every render ends with EOS (even ones ending on a user turn),
    # which exercises the trailing-EOS tolerance of the prefix alignment.
    trailing_eos = False

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = [self._start_of_turn_token_id]
        for msg in messages:
            ids.append(self._id_for_token(f"<{msg['role']}>"))
            if msg["role"] == "assistant":
                ids.append(self._id_for_token("<think>"))
                ids.extend(self._id_for_token(tok) for tok in str(msg.get("reasoning_content") or "").split())
                ids.append(self._id_for_token("</think>"))
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
                ids.append(self.eos_token_id)
            else:
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
        if kwargs.get("add_generation_prompt"):
            ids.append(self._id_for_token("<assistant>"))
            ids.append(self._id_for_token("<think>"))
            if not kwargs.get("enable_thinking", True):
                ids.append(self._id_for_token("</think>"))
        if self.trailing_eos and ids[-1] != self.eos_token_id:
            ids.append(self.eos_token_id)
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


class _StubTokenizerChatGenThinking(_StubTokenizerChatNoGenThinking):
    """Same rendering, but with a ``{% generation %}`` template whose assistant mask
    excludes the role header and includes the think block, content, and EOS."""

    chat_template = "<dummy {% generation %} enable_thinking template>"

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids = super().apply_chat_template(messages, **{**kwargs, "return_dict": False})
        if not kwargs.get("return_dict", False):
            return ids
        result = {"input_ids": ids}
        if kwargs.get("return_assistant_tokens_mask", False):
            assistant_id = self._id_for_token("<assistant>")
            mask = [0] * len(ids)
            inside = False
            for pos, tok in enumerate(ids):
                if tok == assistant_id:
                    inside = True
                    continue  # header stays out of the generation block
                if inside:
                    mask[pos] = 1
                    if tok == self.eos_token_id:
                        inside = False
            result["assistant_masks"] = mask
        return result


class _StubTokenizerChatNoGenThinkingNoThinkSuffix(_StubTokenizerChatNoGenThinking):
    """GLM-4.5 style: ``enable_thinking=False`` also appends ``/nothink`` to the last user turn,
    so the generation prompt's added tokens start before the assistant header."""

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids = super().apply_chat_template(messages, **{**kwargs, "return_dict": False})
        if kwargs.get("add_generation_prompt") and not kwargs.get("enable_thinking", True):
            header = ids.index(self._id_for_token("<assistant>"), 1)
            ids = ids[:header] + [self._id_for_token("/nothink")] + ids[header:]
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


class _StubTokenizerChatNoGenThinkingPromptOnlyBlock(_StubTokenizerChatNoGen):
    """SmolLM3 style: the training render of a turn WITHOUT reasoning has no think block,
    but the ``enable_thinking=False`` generation prompt ends with an empty one (and also
    rewrites the system block, so its added tokens start before the assistant header)."""

    chat_template = "<dummy prompt-only think block template>"

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = [self._start_of_turn_token_id]
        thinking = kwargs.get("enable_thinking", True)
        if kwargs.get("add_generation_prompt") and not thinking:
            ids.append(self._id_for_token("<no_think_system>"))
        for msg in messages:
            ids.append(self._id_for_token(f"<{msg['role']}>"))
            if msg["role"] == "assistant":
                reasoning = str(msg.get("reasoning_content") or "")
                if reasoning:
                    ids.append(self._id_for_token("<think>"))
                    ids.extend(self._id_for_token(tok) for tok in reasoning.split())
                    ids.append(self._id_for_token("</think>"))
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
                ids.append(self.eos_token_id)
            else:
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
        if kwargs.get("add_generation_prompt"):
            ids.append(self._id_for_token("<assistant>"))
            if not thinking:
                ids.append(self._id_for_token("<think>"))
                ids.append(self._id_for_token("</think>"))
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


class _StubTokenizerChatNoGenThinkingSystemRewrite(_StubTokenizerChatNoGenThinking):
    """``enable_thinking=False`` rewrites the very first token of the render, so the
    generation prompt's added tokens include every earlier turn, assistant ones included."""

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids = super().apply_chat_template(messages, **{**kwargs, "return_dict": False})
        if kwargs.get("add_generation_prompt") and not kwargs.get("enable_thinking", True):
            ids = [ids[0], self._id_for_token("<no_think_system>")] + ids[1:]
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


def _format(tok, messages, **kw):
    return format_chat_template(
        tok,
        formatted_text=[m.copy() for m in messages],
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        answer_only_loss_mask=True,
        unshifted=True,
        **kw,
    )


_NO_REASONING = [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "final answer"},
]


def test_mask_generation_prompt_masks_template_inserted_empty_think_block():
    tok = _StubTokenizerChatNoGenThinking()

    default = _format(tok, _NO_REASONING)
    masked = _format(tok, _NO_REASONING, mask_generation_prompt=True)

    # Same tokens either way; only the loss mask differs.
    assert masked["input_ids"] == default["input_ids"]

    def supervised(sample):
        inv = {v: k for k, v in tok._vocab.items()}
        inv[tok.eos_token_id] = "<eos>"
        inv[tok._start_of_turn_token_id] = "<sot>"
        return [inv[i] for i, keep in zip(sample["input_ids"], sample["loss_mask"]) if keep]

    # Without the flag the empty think block (and the header) are supervised: the bug.
    assert supervised(default) == ["<assistant>", "<think>", "</think>", "final", "answer", "<eos>"]
    # With the flag only what the model actually generates at inference remains.
    assert supervised(masked) == ["final", "answer", "<eos>"]


def test_mask_generation_prompt_uses_thinking_prompt_for_reasoning_turns():
    tok = _StubTokenizerChatNoGenThinking()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "reasoning_content": "think step", "content": "final answer"},
    ]

    sample = _format(tok, messages, mask_generation_prompt=True, mask_reasoning_content=True)
    inv = {v: k for k, v in tok._vocab.items()}
    inv[tok.eos_token_id] = "<eos>"
    supervised = [inv.get(i, "<sot>") for i, keep in zip(sample["input_ids"], sample["loss_mask"]) if keep]

    # The thinking generation prompt is ``<assistant> <think>``; the reasoning text is
    # removed by mask_reasoning_content; the model still has to produce ``</think>``.
    assert supervised == ["</think>", "final", "answer", "<eos>"]


def test_mask_generation_prompt_multiturn_and_last_turn_only():
    tok = _StubTokenizerChatNoGenThinking()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "alpha"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "beta"},
    ]
    inv_ids = lambda sample: [  # noqa: E731
        {v: k for k, v in tok._vocab.items()}.get(i, "<eos>" if i == tok.eos_token_id else "<sot>")
        for i, keep in zip(sample["input_ids"], sample["loss_mask"])
        if keep
    ]

    both = _format(tok, messages, mask_generation_prompt=True)
    assert inv_ids(both) == ["alpha", "<eos>", "beta", "<eos>"]

    last = _format(tok, messages, mask_generation_prompt=True, train_on_last_turn_only=True)
    assert inv_ids(last) == ["beta", "<eos>"]


def test_mask_generation_prompt_with_generation_kwd_template():
    tok = _StubTokenizerChatGenThinking()

    default = _format(tok, _NO_REASONING)
    masked = _format(tok, _NO_REASONING, mask_generation_prompt=True)
    inv = {v: k for k, v in tok._vocab.items()}
    inv[tok.eos_token_id] = "<eos>"

    def supervised(sample):
        return [inv[i] for i, keep in zip(sample["input_ids"], sample["loss_mask"]) if keep]

    # The template's own assistant mask already excludes the header but keeps the tags.
    assert supervised(default) == ["<think>", "</think>", "final", "answer", "<eos>"]
    assert supervised(masked) == ["final", "answer", "<eos>"]


def test_mask_generation_prompt_tolerates_trailing_eos_in_prefix_render():
    tok = _StubTokenizerChatNoGenThinking()
    tok.trailing_eos = True

    masked = _format(tok, _NO_REASONING, mask_generation_prompt=True)
    inv = {v: k for k, v in tok._vocab.items()}
    inv[tok.eos_token_id] = "<eos>"
    inv[tok._start_of_turn_token_id] = "<sot>"
    supervised = [inv[i] for i, keep in zip(masked["input_ids"], masked["loss_mask"]) if keep]

    assert "<think>" not in supervised and "</think>" not in supervised
    assert supervised[-3:] == ["final", "answer", "<eos>"]


def test_mask_generation_prompt_accepts_generation_prompt_suffix():
    tok = _StubTokenizerChatNoGenThinkingNoThinkSuffix()

    masked = _format(tok, _NO_REASONING, mask_generation_prompt=True)
    inv = {v: k for k, v in tok._vocab.items()}
    inv[tok.eos_token_id] = "<eos>"
    inv[tok._start_of_turn_token_id] = "<sot>"
    supervised = [inv[i] for i, keep in zip(masked["input_ids"], masked["loss_mask"]) if keep]

    # ``/nothink`` never appears in the rendered turn, but the rest of the generation
    # prompt (``<assistant> <think> </think>``) does and is masked.
    assert tok._id_for_token("/nothink") not in masked["input_ids"]
    assert supervised == ["final", "answer", "<eos>"]


def test_mask_generation_prompt_matches_prompt_with_extra_leading_and_trailing_tokens():
    # The generation prompt both starts before the header (system rewrite) and ends with an
    # empty think block the training render does not contain; only the header can be matched.
    tok = _StubTokenizerChatNoGenThinkingPromptOnlyBlock()

    default = _format(tok, _NO_REASONING)
    masked = _format(tok, _NO_REASONING, mask_generation_prompt=True)
    inv = {v: k for k, v in tok._vocab.items()}
    inv[tok.eos_token_id] = "<eos>"
    inv[tok._start_of_turn_token_id] = "<sot>"

    def supervised(sample):
        return [inv[i] for i, keep in zip(sample["input_ids"], sample["loss_mask"]) if keep]

    assert supervised(default) == ["<assistant>", "final", "answer", "<eos>"]
    assert supervised(masked) == ["final", "answer", "<eos>"]


def test_mask_generation_prompt_does_not_anchor_on_earlier_rendered_turn():
    # The generation prompt's added tokens contain the FIRST assistant turn verbatim, and
    # both turns share the same content. Anchoring on that earlier copy would mask real
    # content, so the match must use the last occurrence (the prompt's own header).
    tok = _StubTokenizerChatNoGenThinkingSystemRewrite()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "same answer"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "same answer"},
    ]

    masked = _format(tok, messages, mask_generation_prompt=True)
    inv = {v: k for k, v in tok._vocab.items()}
    inv[tok.eos_token_id] = "<eos>"
    inv[tok._start_of_turn_token_id] = "<sot>"
    supervised = [inv[i] for i, keep in zip(masked["input_ids"], masked["loss_mask"]) if keep]

    assert supervised == ["same", "answer", "<eos>", "same", "answer", "<eos>"]


def test_mask_generation_prompt_masks_leading_assistant_turn_from_empty_prefix():
    # The template renders an empty conversation, so the generation prompt of a leading
    # assistant turn is just what it appends to nothing, and the turn is masked like any other.
    tok = _StubTokenizerChatNoGenThinking()
    messages = [{"role": "assistant", "content": "hello"}]

    masked = _format(tok, messages, mask_generation_prompt=True)
    assert masked["input_ids"] == _format(tok, messages)["input_ids"]
    assert _decode(tok, masked) == ["hello", "<eos>"]


def test_mask_generation_prompt_skips_leading_assistant_turn_when_template_rejects_empty_prefix(caplog):
    class _NeedsMessages(_StubTokenizerChatNoGenThinking):
        def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
            messages[0]  # like templates that read messages[0]['role']
            return super().apply_chat_template(messages, **kwargs)

    tok = _NeedsMessages()
    messages = [{"role": "assistant", "content": "hello"}]
    full = tok.apply_chat_template([m.copy() for m in messages])
    _warned_generation_prompt.clear()
    with caplog.at_level("WARNING"):
        mask = _build_generation_prompt_mask(tok, [m.copy() for m in messages], full, unpadded_full_ids=full)

    assert mask == [0] * len(full)
    assert "leading assistant turn" in caplog.text


def test_mask_generation_prompt_propagates_render_errors_on_non_empty_prefix():
    class _NoGenerationPrompt(_StubTokenizerChatNoGenThinking):
        def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
            if kwargs.get("add_generation_prompt"):
                raise RuntimeError("add_generation_prompt unsupported")
            return super().apply_chat_template(messages, **kwargs)

    tok = _NoGenerationPrompt()
    full = tok.apply_chat_template([m.copy() for m in _NO_REASONING])
    with pytest.raises(RuntimeError, match="add_generation_prompt"):
        _build_generation_prompt_mask(tok, [m.copy() for m in _NO_REASONING], full, unpadded_full_ids=full)


def test_mask_generation_prompt_warns_when_base_prefix_does_not_match_full_render(caplog):
    class _RewritesHistory(_StubTokenizerChatNoGenThinking):
        """Renders a user-only prefix differently from the same prefix inside a full conversation."""

        def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
            ids = super().apply_chat_template(messages, **{**kwargs, "return_dict": False})
            if messages and messages[-1]["role"] == "user" and not kwargs.get("add_generation_prompt"):
                ids = ids + [self._id_for_token("<pending>")]
            if kwargs.get("return_dict", False):
                return {"input_ids": ids}
            return ids

    tok = _RewritesHistory()
    full = tok.apply_chat_template([m.copy() for m in _NO_REASONING])
    _warned_generation_prompt.clear()  # the warning is emitted once per process
    with caplog.at_level("WARNING"):
        mask = _build_generation_prompt_mask(tok, [m.copy() for m in _NO_REASONING], full, unpadded_full_ids=full)

    assert mask == [0] * len(full)
    assert "Could not align the generation prompt" in caplog.text


def test_tokenize_chat_only_forwards_generation_prompt_kwargs_when_requested():
    seen = []

    class _Recording(_StubTokenizerChatNoGenThinking):
        def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
            seen.append(kwargs)
            return super().apply_chat_template(messages, **kwargs)

    tok = _Recording()
    _tokenize_chat(tok, _NO_REASONING[:1])
    _tokenize_chat(tok, _NO_REASONING[:1], add_generation_prompt=True, enable_thinking=False)

    assert "add_generation_prompt" not in seen[0] and "enable_thinking" not in seen[0]
    assert seen[1]["add_generation_prompt"] is True and seen[1]["enable_thinking"] is False


class _StubTokenizerChatNoGenQwen3Like(_StubTokenizerChatNoGen):
    """Qwen3 style: the thinking generation prompt is the bare header and the model emits
    ``<think>`` itself; the non-thinking prompt appends an empty ``<think> </think>``. A turn
    whose reasoning sits inline in ``content`` renders exactly like a ``reasoning_content``
    turn, which is the common distilled-data format."""

    chat_template = "<dummy qwen3-like enable_thinking template>"
    # Stock Qwen3 renders assistant turns that precede the last user turn without their
    # reasoning block, so a prefix render does not reproduce the full render.
    rewrites_history = False

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = [self._start_of_turn_token_id]
        last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
        for idx, msg in enumerate(messages):
            ids.append(self._id_for_token(f"<{msg['role']}>"))
            content = str(msg["content"])
            if msg["role"] == "assistant":
                reasoning = str(msg.get("reasoning_content") or "")
                if self.rewrites_history and idx < last_user:
                    pass
                elif reasoning:
                    ids.append(self._id_for_token("<think>"))
                    ids.extend(self._id_for_token(tok) for tok in reasoning.split())
                    ids.append(self._id_for_token("</think>"))
                elif "<think>" not in content:
                    ids.append(self._id_for_token("<think>"))
                    ids.append(self._id_for_token("</think>"))
                ids.extend(self._id_for_token(tok) for tok in content.split())
                ids.append(self.eos_token_id)
            else:
                ids.extend(self._id_for_token(tok) for tok in content.split())
        if kwargs.get("add_generation_prompt"):
            ids.append(self._id_for_token("<assistant>"))
            if not kwargs.get("enable_thinking", True):
                ids.append(self._id_for_token("<think>"))
                ids.append(self._id_for_token("</think>"))
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


class _StubTokenizerChatNoGenPlainHeader(_StubTokenizerChatNoGen):
    """Plain-text header whose last token equals its first (``nl ### Assistant: nl``)."""

    chat_template = "<dummy plain header template>"

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = [self._start_of_turn_token_id]
        for msg in messages:
            if msg["role"] == "assistant":
                ids.extend(self._id_for_token(tok) for tok in ("nl", "###", "Assistant:", "nl"))
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
                ids.append(self.eos_token_id)
            else:
                ids.extend(self._id_for_token(tok) for tok in ("nl", "###", "User:", "nl"))
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
        if kwargs.get("add_generation_prompt"):
            ids.extend(self._id_for_token(tok) for tok in ("nl", "###", "Assistant:", "nl"))
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


class _StubTokenizerChatNoGenRewriteNoHeader(_StubTokenizerChatNoGenThinking):
    """Pathological: ``enable_thinking=False`` rewrites the first token but the template
    ignores ``add_generation_prompt``, so the only thing the prompt render adds is the
    rewritten history, which contains earlier assistant turns verbatim."""

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids = super().apply_chat_template(messages, **{**kwargs, "return_dict": False, "add_generation_prompt": False})
        if kwargs.get("add_generation_prompt") and not kwargs.get("enable_thinking", True):
            ids = [ids[0], self._id_for_token("<no_think_system>")] + ids[1:]
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


def _decode(tok, sample):
    inv = {v: k for k, v in tok._vocab.items()}
    inv[tok.eos_token_id] = "<eos>"
    inv[tok._start_of_turn_token_id] = "<sot>"
    return [inv[i] for i, keep in zip(sample["input_ids"], sample["loss_mask"]) if keep]


def test_mask_generation_prompt_keeps_think_opener_when_reasoning_is_inline_in_content():
    # No reasoning_content key, but the content carries the trace. The thinking prompt (bare
    # header) reproduces the turn completely, the non-thinking prompt only partially, so the
    # thinking mode wins and the <think> the model must emit stays supervised.
    tok = _StubTokenizerChatNoGenQwen3Like()
    inline = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<think> deep thought </think> final answer"},
    ]
    assert _decode(tok, _format(tok, inline, mask_generation_prompt=True)) == [
        "<think>",
        "deep",
        "thought",
        "</think>",
        "final",
        "answer",
        "<eos>",
    ]
    # The plain non-thinking turn still gets its template-inserted empty block removed.
    assert _decode(tok, _format(tok, _NO_REASONING, mask_generation_prompt=True)) == ["final", "answer", "<eos>"]


def test_mask_generation_prompt_masks_plain_text_header_ending_with_its_first_token():
    tok = _StubTokenizerChatNoGenPlainHeader()
    masked = _format(tok, _NO_REASONING, mask_generation_prompt=True)
    assert _decode(tok, masked) == ["final", "answer", "<eos>"]


def test_mask_generation_prompt_rejects_anchor_inside_rendered_history():
    # The prompt render adds no header, only a rewritten history whose last assistant header
    # is followed by a real turn (and its EOS). Matching from there would mask content, so
    # that anchor is rejected and the turn is left as the assistant mask had it.
    tok = _StubTokenizerChatNoGenRewriteNoHeader()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "same answer"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "same answer"},
    ]
    default = _format(tok, messages)
    masked = _format(tok, messages, mask_generation_prompt=True)
    assert masked["loss_mask"] == default["loss_mask"]


def test_match_generation_prompt_prefers_complete_then_longer_match():
    eos = 99
    # Prompt starts at the turn: complete match of all 3 tokens.
    assert _match_generation_prompt([7, 8, 9], [7, 8, 9, 1, 2], eos) == (3, True)
    # Prompt extends past the turn (SmolLM3): partial, anchored at its start.
    assert _match_generation_prompt([7, 8, 9, 5, 6], [7, 8, 9, 1, 2], eos) == (3, False)
    # Prompt starts before the header (GLM's /nothink): anchored at the last occurrence.
    assert _match_generation_prompt([4, 4, 7, 8, 9], [7, 8, 9, 1, 2], eos) == (3, True)
    # Last occurrence followed by an end-of-turn token is rendered history, not a prompt.
    assert _match_generation_prompt([7, 8, eos, 3], [7, 8, 9], eos) == (2, False)
    assert _match_generation_prompt([4, 7, 8, eos, 3], [7, 8, 9], eos) == (0, False)
    assert _match_generation_prompt([], [7], eos) == (0, False)
    assert _match_generation_prompt([7], [], eos) == (0, False)


def test_mask_generation_prompt_handles_left_truncation_with_generation_kwd():
    # With a {% generation %} template the multiturn builder never runs, so the truncation
    # window must be mapped through the untruncated reference render (reference_offset).
    class _LeftTruncating(_StubTokenizerChatGenThinking):
        def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
            result = super().apply_chat_template(messages, **kwargs)
            max_length = kwargs.get("max_length")
            if kwargs.get("truncation") and max_length:
                if isinstance(result, dict):
                    result = {k: v[-max_length:] for k, v in result.items()}
                else:
                    result = result[-max_length:]
            return result

    tok = _LeftTruncating()
    full = tok.apply_chat_template([m.copy() for m in _NO_REASONING])
    keep = len(full) - 2  # drop the leading <sot> and <user>
    masked = _format(tok, _NO_REASONING, mask_generation_prompt=True, truncation=True, seq_length=keep)
    assert len(masked["input_ids"]) == keep
    assert _decode(tok, masked) == ["final", "answer", "<eos>"]


def test_mask_generation_prompt_rejects_non_contiguous_truncation():
    class _MiddleTruncating(_StubTokenizerChatGenThinking):
        def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
            result = super().apply_chat_template(messages, **kwargs)
            max_length = kwargs.get("max_length")
            if kwargs.get("truncation") and max_length:
                drop = len(result["input_ids"]) - max_length if isinstance(result, dict) else len(result) - max_length
                if isinstance(result, dict):
                    result = {k: v[1 : 1 + max_length] for k, v in result.items()}
                else:
                    result = result[1 : 1 + max_length]
                assert drop >= 1
            return result

    tok = _MiddleTruncating()
    full = tok.apply_chat_template([m.copy() for m in _NO_REASONING])
    with pytest.raises(ValueError, match="contiguous prefix or suffix"):
        _format(tok, _NO_REASONING, mask_generation_prompt=True, truncation=True, seq_length=len(full) - 2)


def test_mask_generation_prompt_reuses_multiturn_prefix_renders():
    calls = []

    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "alpha"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "beta"},
    ]

    class _Counting(_StubTokenizerChatNoGenThinking):
        def apply_chat_template(self, rendered, **kwargs):  # type: ignore[override]
            calls.append(
                (len(rendered), kwargs.get("add_generation_prompt", False), rendered == messages[: len(rendered)])
            )
            return super().apply_chat_template(rendered, **kwargs)

    tok = _Counting()
    _format(tok, messages, mask_generation_prompt=True)
    plain_prefix_renders = [n for n, gen, original in calls if not gen and original and n < len(messages)]
    # Each proper prefix (1, 2, 3 messages) is rendered once and shared by both builders.
    assert sorted(plain_prefix_renders) == [1, 2, 3]
    # Plus three bound renders per assistant turn (the turn with its text removed and swapped
    # for each of the two sentinels).
    assert sorted(n for n, gen, original in calls if not gen and not original) == [2, 2, 2, 4, 4, 4]

    # truncation=True on a sample that fits is not a truncation: the prefix renders are still
    # shared and the full conversation is not rendered again as an untruncated reference.
    plain_renders = sorted(n for n, gen, original in calls if not gen and original)
    calls.clear()
    _format(tok, messages, mask_generation_prompt=True, truncation=True, seq_length=512)
    assert sorted(n for n, gen, original in calls if not gen and original) == plain_renders


def test_multiturn_mask_retains_prefix_ids_only_when_generation_prompt_is_masked(monkeypatch):
    # Prefix renders are worth keeping only for the generation-prompt builder. With the option
    # off, format_chat_template passes no cache and the multiturn builder keeps prefix lengths
    # alone, so the per-sample temporary state stays O(turns) rather than O(turns x tokens).
    from nemo_automodel.components.datasets.llm import formatting_utils

    seen: List[Dict[int, List[int]] | None] = []
    original = formatting_utils._build_multiturn_assistant_mask

    def spy(*args, **kwargs):
        seen.append(kwargs.get("prefix_cache"))
        return original(*args, **kwargs)

    monkeypatch.setattr(formatting_utils, "_build_multiturn_assistant_mask", spy)
    tok = _StubTokenizerChatNoGenThinking()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "alpha"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "beta"},
    ]
    off = _format(tok, messages)
    on = _format(tok, messages, mask_generation_prompt=True)
    assert seen[0] is None
    assert sorted(seen[1]) == [1, 2, 3, 4]  # the shared cache holds every prefix render
    assert on["input_ids"] == off["input_ids"]

    # The length-only path and the shared-cache path place the assistant spans identically.
    full = _tokenize_chat(tok, messages)
    by_length = formatting_utils._build_multiturn_assistant_mask(tok, messages, full, unpadded_full_ids=full)
    cache: Dict[int, List[int]] = {}
    by_cache = formatting_utils._build_multiturn_assistant_mask(
        tok, messages, full, unpadded_full_ids=full, prefix_cache=cache
    )
    assert by_length == by_cache
    assert sorted(cache) == [1, 2, 3, 4]


class _StubTokenizerChatNoGenGemmaLike(_StubTokenizerChatNoGen):
    """Gemma-4 style. Turns close with ``<eot>``, a token distinct from ``eos_token_id``; an
    assistant turn that answers a tool response continues the previous model turn with no
    header; and ``enable_thinking=True`` rewrites the render from its first token (a thinking
    system block), so the generation prompt's added tokens contain the whole conversation."""

    chat_template = "<dummy gemma-like template>"

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = [self._start_of_turn_token_id]
        thinking = kwargs.get("add_generation_prompt") and kwargs.get("enable_thinking", False)
        if thinking:
            ids.append(self._id_for_token("<think_system>"))
        prev = None
        for msg in messages:
            role = msg["role"]
            if role == "tool":
                ids.append(self._id_for_token("<tool_response>"))
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
                ids.append(self._id_for_token("</tool_response>"))
                prev = role
                continue
            if not (role == "assistant" and prev == "tool"):
                ids.append(self._id_for_token(f"<{role}>"))
            for call in msg.get("tool_calls") or []:
                ids.append(self._id_for_token("<tool_call>"))
                ids.append(self._id_for_token(call["function"]["name"]))
                ids.append(self._id_for_token("</tool_call>"))
            ids.extend(self._id_for_token(tok) for tok in str(msg.get("content") or "").split())
            if not msg.get("tool_calls"):
                ids.append(self._id_for_token("<eot>"))
            prev = role
        if kwargs.get("add_generation_prompt"):
            if prev != "tool":
                ids.append(self._id_for_token("<assistant>"))
            elif thinking:
                ids.append(self._id_for_token("<think>"))
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


_TOOL_CALL = [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]


def test_mask_generation_prompt_never_reaches_content_when_eot_differs_from_eos():
    # Regression for a continuation turn (no header) whose text repeats earlier text. The
    # thinking prompt render rewrites history, so the last occurrence of the turn's first
    # token inside it is followed by that earlier copy and an <eot> the eos guard does not
    # know. The structural bound (a render with the content swapped out) must stop the
    # match at zero tokens; only the tool-call turn's header may be removed.
    tok = _StubTokenizerChatNoGenGemmaLike()
    text = "The weather in Paris is sunny."
    messages = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": "", "tool_calls": _TOOL_CALL},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "assistant", "content": text},
    ]

    default = _format(tok, messages)
    masked = _format(tok, messages, mask_generation_prompt=True)
    assert masked["input_ids"] == default["input_ids"]
    body = ["<tool_call>", "get_weather", "</tool_call>", *text.split(), "<eot>"]
    assert _decode(tok, default) == ["<assistant>", *body]
    assert _decode(tok, masked) == body


def test_mask_generation_prompt_bound_caps_offset_zero_anchor():
    # The prompt render both starts at the turn's first token and continues with the turn's
    # own text (an offset-0 anchor that reproduces real content). Without the bound the whole
    # turn would be masked.
    class _EchoesTurn(_StubTokenizerChatNoGen):
        chat_template = "<dummy echoing template>"

        def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
            ids: List[int] = [self._start_of_turn_token_id]
            for msg in messages:
                ids.append(self._id_for_token(f"<{msg['role']}>"))
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
                ids.append(self._id_for_token("<eot>"))
            if kwargs.get("add_generation_prompt"):
                ids.append(self._id_for_token("<assistant>"))
                ids.extend(self._id_for_token(tok) for tok in "final answer".split())
                ids.append(self._id_for_token("<eot>"))
            if kwargs.get("return_dict", False):
                return {"input_ids": ids}
            return ids

    tok = _EchoesTurn()
    default = _format(tok, _NO_REASONING)
    masked = _format(tok, _NO_REASONING, mask_generation_prompt=True)
    assert _decode(tok, default) == ["<assistant>", "final", "answer", "<eot>"]
    assert _decode(tok, masked) == ["final", "answer", "<eot>"]


def test_perturbed_assistant_message_swaps_generated_text_and_drops_tool_calls():
    message = {
        "role": "assistant",
        "name": "bot",
        "content": "0.5 kg",
        "reasoning_content": "weigh it",
        "tool_calls": _TOOL_CALL,
        "tool_responses": [{"name": "get_weather", "response": "sunny"}],
    }
    perturbed = _perturbed_assistant_message(message, "1")
    assert perturbed == {
        "role": "assistant",
        "name": "bot",
        "content": "1",
        "reasoning_content": "1",
        "tool_responses": message["tool_responses"],  # environment text, not model output
    }
    assert message["tool_calls"] is _TOOL_CALL  # the original is untouched

    parts = {"role": "assistant", "content": [{"type": "image"}, {"type": "text", "text": "0 apples"}]}
    assert _perturbed_assistant_message(parts, "1")["content"] == [{"type": "image"}, {"type": "text", "text": "1"}]
    assert _perturbed_assistant_message({"role": "assistant", "content": None, "reasoning": "r"}, "0") == {
        "role": "assistant",
        "content": "0",
        "reasoning": "0",
    }


class _StubTokenizerChatNoGenGemmaLikeNormalizing(_StubTokenizerChatNoGenGemmaLike):
    """Gemma-like stub whose vocabulary folds full-width digits onto ASCII (NFKC-style
    normalization), so two texts that differ as characters can share a token id."""

    chat_template = "<dummy gemma-like normalizing template>"

    def _id_for_token(self, tok: str) -> int:
        return super()._id_for_token(unicodedata.normalize("NFKC", tok))


def test_mask_generation_prompt_bound_is_token_level_under_normalization():
    # The answer starts with a full-width zero that the tokenizer normalizes onto the same
    # id as the ASCII "0" sentinel. A character-level sentinel would share the first content
    # token with the real render and let the bound reach it; the token-level bound uses a
    # second sentinel that cannot also collide, so the header-less continuation turn keeps
    # every generated token supervised.
    tok = _StubTokenizerChatNoGenGemmaLikeNormalizing()
    text = "０ degrees in Paris"
    messages = [
        {"role": "user", "content": "How cold is it?"},
        {"role": "assistant", "content": "", "tool_calls": _TOOL_CALL},
        {"role": "tool", "tool_call_id": "c1", "content": "0"},
        {"role": "assistant", "content": text},
    ]
    assert tok._id_for_token("０") == tok._id_for_token("0")

    default = _format(tok, messages)
    masked = _format(tok, messages, mask_generation_prompt=True)
    assert masked["input_ids"] == default["input_ids"]
    body = ["<tool_call>", "get_weather", "</tool_call>", "0", "degrees", "in", "Paris", "<eot>"]
    assert _decode(tok, default) == ["<assistant>", *body]
    assert _decode(tok, masked) == body


class _StubTokenizerChatNoGenGemmaLikeWordStart(_StubTokenizerChatNoGenGemmaLike):
    """Gemma-like stub whose tokenizer opens every non-empty message text with a
    ``<word_start>`` token: a content-owned prefix that is the same for any value, so every
    sentinel render shares its first content token with the real render."""

    chat_template = "<dummy gemma-like word-start template>"

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = [self._start_of_turn_token_id]
        thinking = kwargs.get("add_generation_prompt") and kwargs.get("enable_thinking", False)
        if thinking:
            ids.append(self._id_for_token("<think_system>"))
        prev = None
        for msg in messages:
            role = msg["role"]
            content = str(msg.get("content") or "")
            if role == "tool":
                ids.append(self._id_for_token("<tool_response>"))
                ids.extend(self._id_for_token(tok) for tok in content.split())
                ids.append(self._id_for_token("</tool_response>"))
                prev = role
                continue
            if not (role == "assistant" and prev == "tool"):
                ids.append(self._id_for_token(f"<{role}>"))
            for call in msg.get("tool_calls") or []:
                ids.append(self._id_for_token("<tool_call>"))
                ids.append(self._id_for_token(call["function"]["name"]))
                ids.append(self._id_for_token("</tool_call>"))
            if content:
                ids.append(self._id_for_token("<word_start>"))
                ids.extend(self._id_for_token(tok) for tok in content.split())
            if not msg.get("tool_calls"):
                ids.append(self._id_for_token("<eot>"))
            prev = role
        if kwargs.get("add_generation_prompt"):
            if prev != "tool":
                ids.append(self._id_for_token("<assistant>"))
            elif thinking:
                ids.append(self._id_for_token("<think>"))
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids


def test_mask_generation_prompt_bound_excludes_content_owned_prefix_token():
    # Every sentinel render shares a content-owned first token with the real render (the
    # tokenizer opens any value with the same token), so two diverging sentinels alone would
    # still prove nothing about that token. The bound is also capped by the render with the
    # generated text removed entirely, which has no such token: the header-less continuation
    # turn keeps every generated token, including <word_start>, supervised.
    tok = _StubTokenizerChatNoGenGemmaLikeWordStart()
    text = "The weather in Paris is sunny."
    messages = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": "", "tool_calls": _TOOL_CALL},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "assistant", "content": text},
    ]

    default = _format(tok, messages)
    masked = _format(tok, messages, mask_generation_prompt=True)
    assert masked["input_ids"] == default["input_ids"]
    body = ["<tool_call>", "get_weather", "</tool_call>", "<word_start>", *text.split(), "<eot>"]
    assert _decode(tok, default) == ["<assistant>", *body]
    assert _decode(tok, masked) == body


def test_mask_generation_prompt_bound_ignores_token_string_prefix_relation():
    # Byte-level tokenizers expose internal token strings that can stand in a prefix relation
    # while their decoded text does not (Nemotron 3.5: "iá»" -> "i�" and "iá»ģm" -> "iềm").
    # Here the generated <word_start> token's internal string is a strict prefix of the string of
    # the token the empty render has in its place (<eot>). That relation must not move the
    # content boundary: <word_start> is model output and stays supervised.
    class _InternalStrings(_StubTokenizerChatNoGenGemmaLikeWordStart):
        chat_template = "<dummy gemma-like internal-strings template>"

        def convert_ids_to_tokens(self, token_id: int) -> str:
            names = {v: k for k, v in self._vocab.items()}
            return {"<word_start>": "iá»", "<eot>": "iá»ģm"}.get(names.get(token_id), names.get(token_id, ""))

    tok = _InternalStrings()
    text = "The weather in Paris is sunny."
    messages = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": "", "tool_calls": _TOOL_CALL},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "assistant", "content": text},
    ]
    default = _format(tok, messages)
    masked = _format(tok, messages, mask_generation_prompt=True)
    assert tok.convert_ids_to_tokens(tok._id_for_token("<eot>")).startswith(
        tok.convert_ids_to_tokens(tok._id_for_token("<word_start>"))
    )
    assert masked["input_ids"] == default["input_ids"]
    body = ["<tool_call>", "get_weather", "</tool_call>", "<word_start>", *text.split(), "<eot>"]
    assert _decode(tok, masked) == body


class _StubTokenizerChatNoGenQwen3ThinkingLike(_StubTokenizerChatNoGen):
    """Qwen3-Thinking style: a reasoning turn renders ``<assistant> <think> \\n reasoning \\n
    </think> \\n\\n content EOS`` and the generation prompt is ``<assistant> <think> \\n``. The
    tokenizer keeps ``\\n`` and ``\\n\\n`` as distinct ids, so a turn with no reasoning text
    fuses the ``\\n`` after ``<think>`` with the one before ``</think>`` into ``\\n\\n``."""

    chat_template = "<dummy qwen3-thinking-like template>"

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = [self._start_of_turn_token_id]
        for msg in messages:
            ids.append(self._id_for_token(f"<{msg['role']}>"))
            if msg["role"] == "assistant":
                reasoning = str(msg.get("reasoning_content") or "").split()
                ids.append(self._id_for_token("<think>"))
                if reasoning:
                    ids.append(self._id_for_token("\n"))
                    ids.extend(self._id_for_token(tok) for tok in reasoning)
                    ids.append(self._id_for_token("\n"))
                else:
                    ids.append(self._id_for_token("\n\n"))
                ids.append(self._id_for_token("</think>"))
                ids.append(self._id_for_token("\n\n"))
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
                ids.append(self.eos_token_id)
            else:
                ids.extend(self._id_for_token(tok) for tok in str(msg["content"]).split())
        if kwargs.get("add_generation_prompt"):
            ids.extend(self._id_for_token(tok) for tok in ("<assistant>", "<think>", "\n"))
        if kwargs.get("return_dict", False):
            return {"input_ids": ids}
        return ids

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return {v: k for k, v in self._vocab.items()}.get(token_id, "<eos>")


_QWEN3_THINKING_REASONING = [
    {"role": "user", "content": "How many legs does a spider have?"},
    {"role": "assistant", "reasoning_content": "count them", "content": "eight"},
]


def test_mask_generation_prompt_empty_render_bound_stops_at_merged_template_token():
    # With the reasoning text removed the template's "\n" after <think> fuses with the "\n"
    # before </think> into one "\n\n" token, so the empty render diverges from the real turn one
    # token early. Nothing on the token ids proves that "\n" is template text (its token string
    # being a prefix of the merged token's is not a proof, see the test below), so the bound
    # fails closed there: <assistant> <think> leave the loss and the "\n" keeps its supervision.
    tok = _StubTokenizerChatNoGenQwen3ThinkingLike()
    default = _format(tok, _QWEN3_THINKING_REASONING)
    masked = _format(tok, _QWEN3_THINKING_REASONING, mask_generation_prompt=True)
    assert masked["input_ids"] == default["input_ids"]
    body = ["\n", "count", "them", "\n", "</think>", "\n\n", "eight", "<eos>"]
    assert _decode(tok, default) == ["<assistant>", "<think>", *body]
    assert _decode(tok, masked) == body


def test_mask_generation_prompt_fails_closed_when_sentinel_renders_do_not_diverge():
    # A tokenizer that maps every single-character text onto one UNK id renders both sentinels
    # identically, so nothing about the generated-text boundary is proven: the bound is zero and
    # even the real header stays in the loss rather than risking assistant content.
    class _UnkSingleChars(_StubTokenizerChatNoGenThinking):
        chat_template = "<dummy unk template>"

        def _id_for_token(self, tok: str) -> int:
            if len(tok) == 1:
                return 0  # UNK
            return super()._id_for_token(tok)

    tok = _UnkSingleChars()
    default = _format(tok, _NO_REASONING)
    masked = _format(tok, _NO_REASONING, mask_generation_prompt=True)
    assert masked["input_ids"] == default["input_ids"]
    assert masked["loss_mask"] == default["loss_mask"]
    assert _decode(tok, default) == ["<assistant>", "<think>", "</think>", "final", "answer", "<eos>"]


class _StubTokenizerChatGenSymmetricTurns(_StubTokenizerChatNoGen):
    """``{% generation %}`` template whose every turn renders as ``<turn> [reasoning] content``
    and whose generation prompt is a bare ``<turn>``, so a short truncation window can match
    both the prefix and the suffix of the untruncated render. Truncates on ``truncation_side``."""

    chat_template = "<dummy {% generation %} symmetric template>"
    truncation_side = "left"

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[override]
        ids: List[int] = []
        mask: List[int] = []
        for msg in messages:
            turn = [self._id_for_token("<turn>")]
            turn.extend(self._id_for_token(tok) for tok in str(msg.get("reasoning_content") or "").split())
            turn.append(self._id_for_token(str(msg["content"])))
            ids.extend(turn)
            mask.extend([int(msg["role"] == "assistant")] * len(turn))
        if kwargs.get("add_generation_prompt"):
            ids.append(self._id_for_token("<turn>"))
            mask.append(0)
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length:
            keep = slice(-max_length, None) if self.truncation_side == "left" else slice(0, max_length)
            ids, mask = ids[keep], mask[keep]
        if not kwargs.get("return_dict", False):
            return ids
        result = {"input_ids": ids}
        if kwargs.get("return_assistant_tokens_mask", False):
            result["assistant_masks"] = mask
        return result


def test_mask_generation_prompt_left_truncation_uses_truncation_side():
    # Untruncated: <turn> a <turn> a ; the two-token window equals both ends. The tokenizer
    # truncates on the left, so the window is the assistant turn and its header must be masked;
    # mapping it onto the prefix would put the turn outside the window and keep the header.
    tok = _StubTokenizerChatGenSymmetricTurns()
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "a"}]
    masked = _format(tok, messages, mask_generation_prompt=True, truncation=True, seq_length=2)
    assert masked["loss_mask"] == [0, 1]

    tok.truncation_side = "right"
    masked = _format(tok, messages, mask_generation_prompt=True, truncation=True, seq_length=2)
    # Right truncation keeps the user turn, which is not supervised to begin with.
    assert masked["loss_mask"] == [0, 0]


def test_mask_reasoning_content_left_truncation_uses_truncation_side():
    # Same ambiguity for the reasoning mask. Untruncated: <turn> a <turn> <turn>(reasoning) a ;
    # the two-token window equals both ends and the reasoning token sits in the suffix copy.
    tok = _StubTokenizerChatGenSymmetricTurns()
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "reasoning_content": "<turn>", "content": "a"},
    ]
    masked = _format(tok, messages, mask_reasoning_content=True, truncation=True, seq_length=2)
    assert masked["loss_mask"] == [0, 1]
    masked = _format(tok, messages, mask_reasoning_content=True, truncation=True, seq_length=3)
    # Window: <turn> <turn>(reasoning) a ; only the reasoning copy is removed.
    assert masked["loss_mask"] == [1, 0, 1]


@pytest.mark.parametrize("train_on_last_turn_only", [False, True])
def test_mask_generation_prompt_qwen3_style_history_rewrite_is_single_turn_only(train_on_last_turn_only):
    tok = _StubTokenizerChatNoGenQwen3Like()
    tok.rewrites_history = True
    assert _decode(tok, _format(tok, _NO_REASONING, mask_generation_prompt=True)) == ["final", "answer", "<eos>"]

    # Multi-turn fails before the option runs: answer-only masking itself cannot locate the
    # turns of a template that rewrites history without a {% generation %} block.
    multi_turn = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "alpha"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "beta"},
    ]
    with pytest.raises(ValueError, match="Cannot build an answer-only loss mask"):
        _format(tok, multi_turn, mask_generation_prompt=True, train_on_last_turn_only=train_on_last_turn_only)
