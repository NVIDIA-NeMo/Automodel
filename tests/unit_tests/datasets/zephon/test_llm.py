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

"""Local token resolution and source normalization contracts."""

from dataclasses import replace

import numpy as np
import pytest

from nemo_automodel.components.datasets.zephon.config import (
    ZephonPretokenizedTextSourceConfig,
    ZephonInstructionSourceConfig,
)
from nemo_automodel.components.datasets.zephon.llm import (
    measure_pretokenized_document,
    normalize_pretokenized_document,
    normalize_instruction_chat,
    resolve_token_ids,
)
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer


def test_instruction_normalization_retains_native_messages_and_whitespace():
    config = ZephonInstructionSourceConfig(column_mapping={"context": "c", "question": "q", "answer": "a"})
    payload = {"c": "w0", "q": "w1", "a": "w2 w3"}
    assert normalize_instruction_chat(payload, config=config) == {
        "messages": [
            {"role": "user", "content": "w0 w1 "},
            {"role": "assistant", "content": "w2 w3"},
        ]
    }
    assert normalize_instruction_chat(payload, config=replace(config, use_hf_chat_template=True)) == {
        "messages": [
            {"role": "system", "content": "w0"},
            {"role": "user", "content": "w1"},
            {"role": "assistant", "content": "w2 w3"},
        ]
    }
    assert payload == {"c": "w0", "q": "w1", "a": "w2 w3"}


def test_instruction_normalization_rejects_nontext_columns():
    config = ZephonInstructionSourceConfig(column_mapping={"question": "q", "answer": "a"})
    with pytest.raises(ValueError, match="must contain strings"):
        normalize_instruction_chat({"q": ["w0"], "a": "w1"}, config=config)


def test_pretokenized_schema_preserves_unshifted_tokens_and_masks():
    config = ZephonPretokenizedTextSourceConfig(tokens_field="tokens")
    actual = normalize_pretokenized_document(
        {"tokens": [2, 3, 1], "loss_mask": [0, 1, 1]}, config=config, valid_token_ids=np.arange(6, dtype=np.int64)
    )
    assert actual["input_ids"].tolist() == [2, 3, 1]
    assert actual["loss_mask"].tolist() == [0, 1, 1]
    unmasked = normalize_pretokenized_document(
        {"tokens": [2, 3, 1]}, config=config, valid_token_ids=np.arange(6, dtype=np.int64)
    )
    assert unmasked["loss_mask"].tolist() == [1, 1, 1]
    for malformed in (
        {"tokens": [2.0, 3.0]},
        {"tokens": [2, 3], "labels": [3, 1]},
        {"tokens": [2, 3], "positions": [0, 1]},
        {"tokens": [2, 3], "loss_mask": [0]},
        {"tokens": [2, 3], "attention_mask": [1, 0]},
        {"tokens": [2, 3], "attention_mask": []},
        {"tokens": [2, 3], "attention_mask": [2, 1]},
        {"tokens": [2.0]},
    ):
        with pytest.raises(ValueError):
            normalize_pretokenized_document(malformed, config=config, valid_token_ids=np.arange(6, dtype=np.int64))


def test_pad_override_and_eos_collision(tokenizer):
    assert resolve_token_ids(tokenizer, pad_token_id=None).pad == tokenizer.eos_token_id
    with pytest.raises(ValueError, match="contradicts"):
        resolve_token_ids(tokenizer, pad_token_id=2)
    with pytest.raises(ValueError, match="vocabulary"):
        resolve_token_ids(tokenizer, pad_token_id=999)
    with pytest.raises(ValueError, match="vocabulary"):
        resolve_token_ids(tokenizer, pad_token_id=1.0)


@pytest.mark.parametrize("pad_id", [True, 1.0])
def test_padding_id_requires_an_integer(tokenizer, pad_id):
    with pytest.raises(ValueError, match="existing tokenizer vocabulary ID"):
        resolve_token_ids(tokenizer, pad_token_id=pad_id)


def test_pretokenized_validation_preserves_vocabulary_holes_and_signed_range():
    config = ZephonPretokenizedTextSourceConfig()
    valid = np.array([0, 2, 5, np.iinfo(np.int64).max], dtype=np.int64)
    for tokens in ([2, 5], np.array([2, 5], dtype=np.uint64), [2, np.iinfo(np.int64).max]):
        actual = normalize_pretokenized_document({"input_ids": tokens}, config=config, valid_token_ids=valid)
        assert actual["input_ids"].tolist() == list(tokens)
        assert measure_pretokenized_document({"input_ids": tokens}, tokens_field="input_ids") == 2
    for tokens in ([1, 2], [-1, 2], np.array([2**63, 2], dtype=np.uint64), [2.0], [True]):
        with pytest.raises(ValueError, match="integer tokenizer vocabulary IDs"):
            normalize_pretokenized_document({"input_ids": tokens}, config=config, valid_token_ids=valid)


def test_pretokenized_short_and_unsupervised_documents_are_retained():
    config = ZephonPretokenizedTextSourceConfig(tokens_field="tokens")
    valid = np.arange(6, dtype=np.int64)
    for payload in (
        {"tokens": []},
        {"tokens": [2]},
        {"tokens": [2, 3], "loss_mask": [0, 0]},
        {"tokens": [2, 3], "loss_mask": [1, 1]},
    ):
        actual = normalize_pretokenized_document(payload, config=config, valid_token_ids=valid)
        assert actual["input_ids"].tolist() == payload["tokens"]
        assert actual["loss_mask"].tolist() == payload.get("loss_mask", [1] * len(payload["tokens"]))
        assert actual["input_ids"].dtype == np.int64
        assert actual["loss_mask"].dtype == np.uint8
        assert measure_pretokenized_document(payload, tokens_field="tokens") == len(payload["tokens"])
    with pytest.raises(ValueError, match="binary"):
        normalize_pretokenized_document({"tokens": [2, 3], "loss_mask": [0, 2]}, config=config, valid_token_ids=valid)


def test_native_named_template_uses_selected_string_for_masks_and_every_render(tokenizer):
    from nemo_automodel.components.datasets.llm.formatting_utils import format_chat_template

    template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    messages = [{"role": "user", "content": "w0 w1"}, {"role": "assistant", "content": "w2 w3"}]
    tokenizer.chat_template = template
    expected = format_chat_template(tokenizer, messages, eos_token_id=1, pad_token_id=1, answer_only_loss_mask=True)
    tokenizer.chat_template = {"named_only": template}
    actual = format_chat_template(tokenizer, messages, eos_token_id=1, pad_token_id=1, answer_only_loss_mask=True)
    assert actual == expected
    # Tool-use selection must match what HF actually renders, including auxiliary masks.
    tokenizer.chat_template = {"default": "{% generation %}unused{% endgeneration %}", "tool_use": template}
    assert (
        format_chat_template(tokenizer, messages, eos_token_id=1, pad_token_id=1, tools=[], answer_only_loss_mask=True)
        == expected
    )
