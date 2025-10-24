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

import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

GENERATION_REGEX = re.compile(r"\{%-?\s+generation\s+-?%\}")


class NoContextLeftError(RuntimeError):
    """Raised when context must be fully removed to satisfy seq_length."""


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


def _package_tokenized_example(
    tokenizer,
    input_ids,
    assistant_masks,
    eos_token_id,
    pad_token_id,
    seq_length,
):
    """
    Package a tokenized example with proper masking and padding.

    Args:
        tokenizer: The tokenizer to use.
        input_ids: The tokenized input ids.
        assistant_masks: Boolean mask indicating which tokens are assistant/answer tokens (1) vs prompt tokens (0).
        eos_token_id: The end-of-sequence token id.
        pad_token_id: The padding token id.
        seq_length: Optional sequence length for padding.

    Returns:
        A dictionary with input_ids, labels, and attention_mask.
    """
    # llama3 tokenizer does not add eos token
    # see: https://github.com/huggingface/transformers/issues/22794
    if not _has_chat_template(tokenizer) and eos_token_id != input_ids[-1]:
        input_ids += [eos_token_id]
        assistant_masks += [1]

    labels = input_ids.copy()
    input_ids = input_ids[:-1]
    # input_ids= [a, b] -> attention_mask = [1, 1]
    attention_mask = [1] * len(input_ids)
    # Labels: mask out prompt tokens
    labels[:] = [label if bool(m) else -100 for label, m in zip(labels, assistant_masks)]
    # remove BOS
    labels = labels[1:]
    if not _has_chat_template(tokenizer):
        assert labels[-1] == eos_token_id, f"labels[-1]={labels[-1]} != eos_token_id={eos_token_id}"
        assert input_ids[-1] != eos_token_id, f"input_ids[-1]={input_ids[-1]} == eos_token_id={eos_token_id}"
    assert len(input_ids) == len(labels), f"len(input_ids)={len(input_ids)} != len(labels)={len(labels)}"

    # No padding is applied here. We only ensure mask matches length.
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


def _truncate_prompt_to_fit_plain(
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    answer: str,
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int],
    answer_only_loss_mask: bool,
) -> str:
    """Iteratively remove leading context words until packaged length fits.

    Splits on spaces to avoid mid-word truncation. Raises NoContextLeftError
    if the prompt must be fully removed to satisfy the constraint.
    """
    if not isinstance(seq_length, int):
        return prompt

    current_prompt = prompt
    while True:
        context_len = len(tokenizer(current_prompt)["input_ids"]) if answer_only_loss_mask else 0
        full_ids = tokenizer(current_prompt + answer)["input_ids"]
        packaged = _package_tokenized_example(False, full_ids, eos_token_id, pad_token_id, seq_length, context_len)
        if len(packaged["labels"]) <= seq_length:
            return current_prompt
        # remove up to the first space from the left
        cut = current_prompt.find(" ")
        if cut == -1:
            raise NoContextLeftError("Context fully removed but sequence still exceeds seq_length")
        current_prompt = current_prompt[cut + 1 :]
        # Skip any additional spaces
        while current_prompt.startswith(" "):
            current_prompt = current_prompt[1:]
        if not current_prompt:
            raise NoContextLeftError("No context left after truncation")


def _truncate_prompt_to_fit_chat(
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    answer: str,
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int],
    start_of_turn_token: Optional[str],
) -> str:
    """Iteratively remove leading context words for chat-template path."""
    if not isinstance(seq_length, int):
        return prompt

    current_prompt = prompt
    while True:
        messages = [
            {"role": "user", "content": current_prompt},
            {"role": "assistant", "content": answer},
        ]
        input_ids = tokenizer.apply_chat_template(messages)
        if isinstance(start_of_turn_token, str):
            start_of_turn_token_id = tokenizer(start_of_turn_token, add_special_tokens=False)["input_ids"][0]
            first_start_of_turn_token_id = input_ids.index(start_of_turn_token_id)
            response_start = input_ids.index(start_of_turn_token_id, first_start_of_turn_token_id + 1)
        else:
            response_start = 0
        packaged = _package_tokenized_example(
            True, input_ids, eos_token_id, pad_token_id, seq_length, response_start
        )
        if len(packaged["labels"]) <= seq_length:
            return current_prompt
        # remove up to the first space from the left
        cut = current_prompt.find(" ")
        if cut == -1:
            raise NoContextLeftError("Context fully removed but sequence still exceeds seq_length")
        current_prompt = current_prompt[cut + 1 :]
        while current_prompt.startswith(" "):
            current_prompt = current_prompt[1:]
        if not current_prompt:
            raise NoContextLeftError("No context left after truncation")


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
        seq_length: Optional maximum sequence length. If the packaged
            sequence exceeds this length, context is removed from the left at
            space boundaries. No padding is applied.

    Returns:
        A dictionary with the formatted example.
    """
    # Optionally truncate prompt to fit the requested maximum length
    truncated_prompt = _truncate_prompt_to_fit_plain(
        tokenizer, prompt, answer, eos_token_id, pad_token_id, seq_length, answer_only_loss_mask
    )

    # Tokenize separately to locate answer start
    if answer_only_loss_mask:
        prompt_ids = tokenizer(truncated_prompt)["input_ids"]
        len_prompt_ids = len(prompt_ids)
    else:
        len_prompt_ids = 0
    # Tokenize full text
    input_ids = tokenizer(truncated_prompt + answer)["input_ids"]

    # Create assistant_masks: 0 for prompt tokens, 1 for answer tokens
    assistant_masks = [0] * len_prompt_ids + [1] * (len(input_ids) - len_prompt_ids)

    return _package_tokenized_example(
        tokenizer=tokenizer,
        input_ids=input_ids,
        assistant_masks=assistant_masks,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
    )


def format_chat_template(
    tokenizer: "PreTrainedTokenizer",
    formatted_text: List[Dict[str, str]],
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int] = None,
    tools: Optional[List[Dict]] = None,
) -> Dict[str, List[int]]:
    """
    Format a chat template style example.

    Args:
        tokenizer: The tokenizer to use.
        formatted_text: The formatted text, with role tags embedded in the content.
        eos_token_id: The end-of-sequence token id.
        pad_token_id: The padding token id.
        seq_length: Optional sequence length for padding.
        tools: Optional list of tool definitions for function calling.

    Returns:
        A dictionary with the formatted example.
    """
    # Ensure we have a usable chat template
    if not _has_chat_template(tokenizer):
        raise ValueError("Tokenizer lacks a usable chat template (chat_template/apply_chat_template)")

    template_has_generation_kwd = GENERATION_REGEX.search(tokenizer.chat_template) is not None

    tokenized_chat = tokenizer.apply_chat_template(
        formatted_text,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=template_has_generation_kwd,
    )

    # Choose the last conversation as answer other history are context by finding the last masked token
    # which indicates end of context and beginning of answer
    input_ids = tokenized_chat.get("input_ids")
    if template_has_generation_kwd:
        mask = tokenized_chat["assistant_masks"]
    else:
        mask = [1] * len(input_ids)

    if getattr(tokenizer, "eos_token_id", None) and input_ids[-1] != tokenizer.eos_token_id:
        input_ids += [tokenizer.eos_token_id]
        mask += [1]

    return _package_tokenized_example(
        tokenizer=tokenizer,
        input_ids=input_ids,
        assistant_masks=mask,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
    )
