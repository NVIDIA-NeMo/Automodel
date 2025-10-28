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
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest
from transformers import AutoTokenizer

from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    format_chat_template,
    format_prompt_completion,
)


@pytest.mark.parametrize(
    "seq_length,padding,truncation",
    [
        (None, "do_not_pad", None),
        (4, "max_length", True),
    ],
)
def test_format_prompt_completion_options(seq_length, padding, truncation):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    TOKENIZER_DIR = f"{os.environ['TEST_DATA_DIR']}/hf_mixtral_2l"
    assert os.path.exists(TOKENIZER_DIR), "Tokenizer directory does not exist"
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
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

    # Basic structure
    assert set(["input_ids", "labels", "attention_mask"]).issubset(out.keys())
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"]) > 0

    # seq_length enforcement (either by HF padding or our packager)
    if isinstance(seq_length, int) and padding != "do_not_pad":
        assert len(out["input_ids"]) == seq_length
        assert len(out["labels"]) == seq_length
        # Trailing padding label must be masked
        assert out["labels"][-1] == -100, (out, pad_token_id)

    # EOS should be present in labels (supervised area) but not as last input_id
    if getattr(tok, "eos_token_id", None) is not None and not truncation == True:
        assert tok.eos_token_id in out["labels"], "EOS must appear in labels"
        # find last non-pad input position and ensure it's not EOS
        last_non_pad = len(out["input_ids"]) - 1
        while last_non_pad >= 0 and out["input_ids"][last_non_pad] == pad_token_id:
            last_non_pad -= 1
        assert last_non_pad >= 0
        assert out["input_ids"][last_non_pad] != tok.eos_token_id

    # There should be masked (prompt) and supervised (answer) tokens
    assert any(l == -100 for l in out["labels"])  # masked prompt
    if not truncation == True:
        assert any(l != -100 for l in out["labels"])  # supervised answer

    # Attention mask should have zeros only in padded tail (if any)
    if isinstance(seq_length, int):
        # From the end, once we see a 0, the rest must be 0
        seen_zero = False
        for v in reversed(out["attention_mask"]):
            if v == 0:
                seen_zero = True
            else:
                if seen_zero:
                    pytest.fail("Non-zero attention_mask value after padded zeros.")


@pytest.mark.parametrize(
    "seq_length,padding,truncation",
    [
        (None, "do_not_pad", None),
        (4, "max_length", True),
    ],
)
def test_format_chat_template_options(seq_length, padding, truncation):

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    TOKENIZER_DIR = f"{os.environ['TEST_DATA_DIR']}/qwen3_4b_instruct_2407"
    assert os.path.exists(TOKENIZER_DIR), "Tokenizer directory does not exist"
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    # Only applicable when tokenizer DOES define a chat template
    if not getattr(tok, "chat_template", None):
        pytest.skip(f"Tokenizer qwen3_4b_instruct_2407 has no chat_template; skipping chat-template tests.")

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

    # Basic structure
    assert set(["input_ids", "labels", "attention_mask"]).issubset(out.keys())
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"]) > 0

    # seq_length enforcement
    if isinstance(seq_length, int):
        assert len(out["input_ids"]) == seq_length
        assert len(out["labels"]) == seq_length
        if truncation == False:
            assert out["labels"][-1] == -100

    # For chat templates, EOS should not be the last input id (unless it's all pad)
    if getattr(tok, "eos_token_id", None) is not None:
        last_non_pad = len(out["input_ids"]) - 1
        while last_non_pad >= 0 and out["input_ids"][last_non_pad] == pad_token_id:
            last_non_pad -= 1
        if last_non_pad >= 0:
            assert out["input_ids"][last_non_pad] != tok.eos_token_id

    # There must be at least some supervised tokens in labels
    assert any(l != -100 for l in out["labels"])  # assistant tokens

    # Attention mask padded tail zeros, if padded
    if isinstance(seq_length, int) and truncation == False:
        seen_zero = False
        for v in reversed(out["attention_mask"]):
            if v == 0:
                seen_zero = True
            else:
                if seen_zero:
                    pytest.fail("Non-zero attention_mask value after padded zeros.")


