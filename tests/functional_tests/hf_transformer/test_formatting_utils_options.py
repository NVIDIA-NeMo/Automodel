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

from __future__ import annotations

import os

import pytest

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    format_chat_template,
    format_prompt_completion,
)


@pytest.mark.parametrize(
    "seq_length,padding,truncation",
    [
        (None, "do_not_pad", None),
        (128, "max_length", True),
    ],
)
def test_format_prompt_completion_options(seq_length, padding, truncation):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    TOKENIZER_DIR = f"{os.environ['TEST_DATA_DIR']}/hf_mixtral_2l"
    assert os.path.exists(TOKENIZER_DIR), "Tokenizer directory does not exist"
    tok = NeMoAutoTokenizer.from_pretrained(TOKENIZER_DIR, add_eos_token=True)
    # Only applicable when tokenizer lacks chat template
    assert getattr(tok, "chat_template", None) is None

    eos_token_id = getattr(tok, "eos_token_id", 0)
    pad_token_id = _add_pad_token(tok) or eos_token_id
    if padding != "do_not_pad":
        tok.pad_token = tok.eos_token

    # If using padding="max_length", seq_length must be an int
    if padding == "max_length" and not isinstance(seq_length, int):
        pytest.skip("padding='max_length' requires seq_length to be set.")

    context = "France is a country in Europe."
    question = "What is the capital of France?"
    answer = "Paris."
    prompt = f"{context} {question} "

    out = format_prompt_completion(
        tokenizer=tok,
        prompt=prompt,
        answer=answer,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
        padding=padding,
        truncation=truncation,
        answer_only_loss_mask=True,
    )

    input_ids = out["input_ids"]
    labels = out["labels"]
    attention_mask = out["attention_mask"]

    # Basic structure
    assert set(["input_ids", "labels", "attention_mask"]).issubset(out.keys())
    assert len(input_ids) == len(labels) == len(attention_mask) > 0

    # seq_length enforcement
    if isinstance(seq_length, int) and padding != "do_not_pad":
        assert len(input_ids) == seq_length
        assert len(labels) == seq_length
        assert labels[-1] == -100, "Trailing padding label must be masked"

    # EOS should be present in supervised labels when not truncated
    if getattr(tok, "eos_token_id", None) is not None and truncation is not True:
        assert tok.eos_token_id in labels, "EOS must appear in labels"

    # There should be masked (prompt) and supervised (answer) tokens
    assert any(v == -100 for v in labels), "Must have masked prompt tokens"
    if truncation is not True:
        assert any(v != -100 for v in labels), "Must have supervised answer tokens"

    # Where attention_mask=0, labels must be -100
    if padding == "do_not_pad":
        for i in range(len(labels)):
            if attention_mask[i] == 0:
                assert labels[i] == -100, f"Position {i}: attention_mask=0 but labels={labels[i]} (expected -100)"
    else:
        # The boundary position (content_end) has attention_mask=0 by design
        # (intentional equivalence with non-padded). Check true padding after it.
        content_end = sum(attention_mask)
        for i in range(content_end + 1, len(labels)):
            assert labels[i] == -100, f"Position {i}: label in padding region should be -100, got {labels[i]}"

    # Attention mask must be contiguous: ones then zeros (right padding)
    saw_zero = False
    for i, v in enumerate(attention_mask):
        if v == 0:
            saw_zero = True
        elif saw_zero:
            pytest.fail(f"attention_mask has 1 at position {i} after a 0 (not right-padded)")


@pytest.mark.parametrize(
    "seq_length,padding,truncation",
    [
        (None, "do_not_pad", None),
        (128, "max_length", True),
    ],
)
def test_format_chat_template_options(seq_length, padding, truncation):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    TOKENIZER_DIR = f"{os.environ['TEST_DATA_DIR']}/qwen3_4b_instruct_2407"
    assert os.path.exists(TOKENIZER_DIR), "Tokenizer directory does not exist"
    tok = NeMoAutoTokenizer.from_pretrained(TOKENIZER_DIR)
    # Only applicable when tokenizer DOES define a chat template
    if not getattr(tok, "chat_template", None):
        pytest.skip("Tokenizer qwen3_4b_instruct_2407 has no chat_template; skipping chat-template tests.")

    eos_token_id = getattr(tok, "eos_token_id", 0)
    pad_token_id = _add_pad_token(tok) or eos_token_id

    if padding == "max_length" and not isinstance(seq_length, int):
        pytest.skip("padding='max_length' requires seq_length to be set.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]

    out = format_chat_template(
        tokenizer=tok,
        formatted_text=messages,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
        padding=padding,
        truncation=truncation,
    )

    input_ids = out["input_ids"]
    labels = out["labels"]
    attention_mask = out["attention_mask"]

    # Basic structure
    assert set(["input_ids", "labels", "attention_mask"]).issubset(out.keys())
    assert len(input_ids) == len(labels) == len(attention_mask) > 0

    # seq_length enforcement
    if isinstance(seq_length, int):
        assert len(input_ids) == seq_length
        assert len(labels) == seq_length

    # There must be at least some supervised tokens in labels
    assert any(v != -100 for v in labels), "Must have supervised assistant tokens"

    # Where attention_mask=0, labels must be -100
    if padding == "do_not_pad":
        for i in range(len(labels)):
            if attention_mask[i] == 0:
                assert labels[i] == -100, f"Position {i}: attention_mask=0 but labels={labels[i]} (expected -100)"
    else:
        # The boundary position (content_end) has attention_mask=0 by design
        # (intentional equivalence with non-padded). Check true padding after it.
        content_end = sum(attention_mask)
        for i in range(content_end + 1, len(labels)):
            assert labels[i] == -100, f"Position {i}: label in padding region should be -100, got {labels[i]}"

    # Attention mask must be contiguous: ones then zeros (right padding)
    saw_zero = False
    for i, v in enumerate(attention_mask):
        if v == 0:
            saw_zero = True
        elif saw_zero:
            pytest.fail(f"attention_mask has 1 at position {i} after a 0 (not right-padded)")

    # Padded tail: all padding positions must have pad_token_id in input_ids
    if isinstance(seq_length, int) and padding == "max_length":
        content_end = sum(attention_mask)
        for i in range(content_end + 1, len(input_ids)):
            assert input_ids[i] == pad_token_id, (
                f"Position {i}: expected pad_token_id={pad_token_id} in padding region, got {input_ids[i]}"
            )


def test_format_chat_template_mask_generation_prompt_real_tokenizer():
    """``mask_generation_prompt`` on a real chat template removes exactly the generation prompt.

    The expected span is derived from the tokenizer itself (what ``add_generation_prompt``
    appends to the conversation prefix), so the test holds for any template version:
    the ids are unchanged, the tokens right after the prefix leave the loss, and every
    token the model generates (the answer and its terminator) stays supervised.
    """
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    TOKENIZER_DIR = f"{os.environ['TEST_DATA_DIR']}/qwen3_4b_instruct_2407"
    assert os.path.exists(TOKENIZER_DIR), "Tokenizer directory does not exist"
    tok = NeMoAutoTokenizer.from_pretrained(TOKENIZER_DIR)
    if not getattr(tok, "chat_template", None):
        pytest.skip("Tokenizer qwen3_4b_instruct_2407 has no chat_template; skipping chat-template tests.")

    eos_token_id = getattr(tok, "eos_token_id", 0)
    pad_token_id = _add_pad_token(tok) or eos_token_id
    # An answer that starts with a digit: the structural bound swaps generated text for the
    # digits "0" and "1", so this is the case a character-level sentinel would get wrong.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many legs does a spider have?"},
        {"role": "assistant", "content": "8 legs."},
    ]

    def run(mask_generation_prompt):
        return format_chat_template(
            tokenizer=tok,
            formatted_text=[dict(m) for m in messages],
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            answer_only_loss_mask=True,
            unshifted=True,
            mask_generation_prompt=mask_generation_prompt,
        )

    default = run(False)
    masked = run(True)
    assert masked["input_ids"] == default["input_ids"]

    def render(add_generation_prompt):
        text = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=add_generation_prompt)
        return tok(text, add_special_tokens=False)["input_ids"]

    prefix_ids = render(False)
    prompt_ids = render(True)
    assert prompt_ids[: len(prefix_ids)] == prefix_ids
    generation_prompt = prompt_ids[len(prefix_ids) :]
    assert generation_prompt, "template must add a generation prompt"
    assert default["input_ids"][len(prefix_ids) : len(prompt_ids)] == generation_prompt

    answer_ids = tok("8 legs.", add_special_tokens=False)["input_ids"]
    expected = list(default["loss_mask"])
    for pos in range(len(prefix_ids), len(prompt_ids)):
        expected[pos] = 0
    assert masked["loss_mask"] == expected
    # The generation prompt is the only difference, and the answer (starting with the digit)
    # and its terminator are still supervised.
    assert all(default["loss_mask"][pos] == 1 for pos in range(len(prefix_ids), len(prompt_ids)))
    answer_start = len(prompt_ids)
    assert masked["input_ids"][answer_start : answer_start + len(answer_ids)] == answer_ids
    assert all(masked["loss_mask"][answer_start : answer_start + len(answer_ids) + 1])
