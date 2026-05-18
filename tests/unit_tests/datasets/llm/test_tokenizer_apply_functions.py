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

from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    _package_tokenized_example,
    _pad_to_seq_length,
    _warned_add_pad_token,
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
        assert (
            out["labels"][-1] == -100
        ), f"Last label should be -100 (padding) after truncation, got {out['labels'][-1]}"

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
