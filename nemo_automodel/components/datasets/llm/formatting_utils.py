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

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def _pad_to_seq_length(sample, pad_token_id, seq_length):
    """Pad a sample to a specific sequence length."""
    n = seq_length - len(sample)
    if n == 0:
        return sample
    return sample + [pad_token_id] * n


def _add_pad_token(tokenizer):
    """Add pad token to tokenizer if not present."""
    pad_token_id = None
    if not hasattr(tokenizer, "pad_token_id"):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id
    if not hasattr(tokenizer, "pad_token") and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    return pad_token_id


def _has_chat_template(tokenizer: "PreTrainedTokenizer") -> bool:
    """
    Check if the tokenizer supports a chat template.

    Args:
        tokenizer: The tokenizer to check.

    Returns:
        True if the tokenizer supports a chat template, False otherwise.
    """
    return getattr(tokenizer, "chat_template", None) is not None and callable(
        getattr(tokenizer, "apply_chat_template", None)
    )


def _package_tokenized_example(has_chat_template, input_ids, eos_token_id, pad_token_id, seq_length, context_len):
    """
    Package a tokenized example with proper masking and padding.

    Args:
        has_chat_template: Whether the tokenizer has a chat template.
        input_ids: The tokenized input ids.
        eos_token_id: The end-of-sequence token id.
        pad_token_id: The padding token id.
        seq_length: Optional sequence length for padding.
        context_len: Length of the context/prompt (to mask in labels).

    Returns:
        A dictionary with input_ids, labels, and attention_mask.
    """
    # llama3 tokenizer does not add eos token
    # see: https://github.com/huggingface/transformers/issues/22794
    if not has_chat_template and eos_token_id != input_ids[-1]:
        input_ids += [eos_token_id]

    labels = input_ids.copy()
    # [a, b, EOS]
    input_ids = input_ids[:-1]
    # input_ids= [a, b] -> attention_mask = [1, 1]
    attention_mask = [1] * len(input_ids)

    # Labels: mask out prompt tokens
    labels[:context_len] = [-100] * context_len
    # remove BOS
    labels = labels[1:]
    if not has_chat_template:
        assert labels[-1] == eos_token_id, f"labels[-1]={labels[-1]} != eos_token_id={eos_token_id}"
        assert input_ids[-1] != eos_token_id, f"input_ids[-1]={input_ids[-1]} == eos_token_id={eos_token_id}"
    assert len(input_ids) == len(labels), f"len(input_ids)={len(input_ids)} != len(labels)={len(labels)}"

    if isinstance(seq_length, int):
        input_ids = _pad_to_seq_length(input_ids, pad_token_id, seq_length)
        labels = _pad_to_seq_length(labels, -100, seq_length)

    # the attention mask can also be extended in the collator with zeros.
    attention_mask += [0] * (len(labels) - len(attention_mask))
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "___PAD_TOKEN_IDS___": {
            "input_ids": pad_token_id,
            "labels": -100,
            "attention_mask": 0,
        },
    }


def format_prompt_completion(
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    answer: str,
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int] = None,
    answer_only_loss_mask: bool = True,
) -> Dict[str, List[int]]:
    """
    Format a prompt-completion style example (without chat template).

    Args:
        tokenizer: The tokenizer to use.
        prompt: The prompt string (e.g. context + question).
        answer: The answer string.
        eos_token_id: The end-of-sequence token id.
        pad_token_id: The padding token id.
        seq_length: Optional sequence length for padding.

    Returns:
        A dictionary with the formatted example.
    """
    full_text = prompt + answer

    # Tokenize separately to locate answer start
    if answer_only_loss_mask:
        prompt_ids = tokenizer(prompt)["input_ids"]
        len_prompt_ids = len(prompt_ids)
    else:
        len_prompt_ids = 0
    # Tokenize full text
    input_ids = tokenizer(full_text)["input_ids"]

    return _package_tokenized_example(False, input_ids, eos_token_id, pad_token_id, seq_length, len_prompt_ids)


def format_chat_template(
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    answer: str,
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int] = None,
    start_of_turn_token: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Format a chat template style example.

    Args:
        tokenizer: The tokenizer to use.
        prompt: The prompt string (e.g. context + question).
        answer: The answer string.
        eos_token_id: The end-of-sequence token id.
        pad_token_id: The padding token id.
        seq_length: Optional sequence length for padding.
        start_of_turn_token: The start of turn token string.

    Returns:
        A dictionary with the formatted example.
    """
    formatted_text = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    input_ids = tokenizer.apply_chat_template(formatted_text)
    if isinstance(start_of_turn_token, str):
        start_of_turn_token_id = tokenizer(start_of_turn_token, add_special_tokens=False)["input_ids"][0]
        first_start_of_turn_token_id = input_ids.index(start_of_turn_token_id)
        # Loss mask is starting with the second start of turn token.
        # labels    = [a b c S d e] ; S is the start of turn token.
        response_start = input_ids.index(start_of_turn_token_id, first_start_of_turn_token_id + 1)
    else:
        response_start = 0
    return _package_tokenized_example(True, input_ids, eos_token_id, pad_token_id, seq_length, response_start)
