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
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

GENERATION_REGEX = re.compile(r"\{%-?\s+generation\s+-?%\}")


@torch.no_grad()
def _get_right_trailing_pad_mask(
    sequence: torch.Tensor,
    pad_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Boolean mask identifying right-trailing padding positions.

    When *pad_token_id != eos_token_id*, it is simply ``sequence == pad_token_id``.

    When the two IDs collide, a plain equality check would also match real EOS
    tokens inside the content.  In that case the function locates the trailing
    contiguous run of the shared token and treats all positions **after the
    first one** in that run as padding.  The first token in the trailing run is
    the real EOS and is kept unmasked so the model still learns to predict
    end-of-sequence.

    Args:
        sequence: 1-D token id tensor.
        pad_token_id: The token id used for padding.
        eos_token_id: The token id used for end-of-sequence.  When equal to
            *pad_token_id* the positional trailing-run logic is used.

    Returns:
        Boolean tensor (same shape as *sequence*) where ``True`` = padding.
    """
    if pad_token_id != eos_token_id:
        return sequence == pad_token_id

    mask = torch.zeros(sequence.shape, dtype=torch.bool, device=sequence.device)
    non_pad_positions = (sequence != pad_token_id).nonzero(as_tuple=True)[0]
    if non_pad_positions.numel() > 0:
        last_content_idx = non_pad_positions[-1].item()
        # last_content_idx + 1 → real EOS (keep), last_content_idx + 2 → padding
        mask[last_content_idx + 2 :] = True
    else:
        # Entire sequence is the pad/eos token; keep the first as real EOS.
        mask[1:] = True
    return mask


def _pad_to_seq_length(sample, pad_token_id, seq_length):
    """Pad a sample to a specific sequence length."""
    n = seq_length - len(sample)
    if n == 0:
        return sample
    return sample + [pad_token_id] * n


_warned_add_pad_token = set()


def _add_pad_token(tokenizer):
    """Add pad token to tokenizer if not present."""
    pad_token_id = None
    if getattr(tokenizer, "pad_token_id", None) is None and not _warned_add_pad_token.add("no_pad_id"):
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning(
            "Tokenizer has no pad_token_id; falling back to eos_token_id (%s). "
            "This may cause issues if downstream code masks padding by token ID.",
            tokenizer.eos_token_id,
        )
    else:
        pad_token_id = tokenizer.pad_token_id
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if (
        pad_token_id
        and pad_token_id == getattr(tokenizer, "eos_token_id", None)
        and not _warned_add_pad_token.add("pad_eq_eos")
    ):
        logger.warning(
            "pad_token_id (%s) == eos_token_id (%s) for tokenizer '%s'. "
            "Ensure loss masking uses positional logic rather than token-ID comparison.",
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            getattr(tokenizer, "name_or_path", "unknown"),
        )
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
    truncation="do_not_truncate",
    padding="do_not_pad",
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
        truncation: Optional truncation strategy.
        padding: Optional padding strategy.
    Returns:
        A dictionary with input_ids, labels, and attention_mask.
    """
    labels = input_ids.copy()
    input_ids = input_ids[:-1]
    # input_ids= [a, b] -> attention_mask = [1, 1]
    attention_mask = [1] * len(input_ids)
    # Labels: mask out prompt tokens
    labels[:] = [label if bool(m) else -100 for label, m in zip(labels, assistant_masks)]
    # remove BOS
    labels = labels[1:]
    if not _has_chat_template(tokenizer) and truncation is None:
        assert labels[-1] == eos_token_id, f"labels[-1]={labels[-1]} != eos_token_id={eos_token_id}"
        assert input_ids[-1] != eos_token_id, f"input_ids[-1]={input_ids[-1]} == eos_token_id={eos_token_id}"
    assert len(input_ids) == len(labels), f"len(input_ids)={len(input_ids)} != len(labels)={len(labels)}"

    if isinstance(seq_length, int) and padding not in [None, "do_not_pad", False]:
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
    padding: Union[str, bool] = "do_not_pad",
    truncation: Union[str, bool] = "do_not_truncate",
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
        # don't add eos token here. NOTE: this is only for calculating the length of the prompt.
        # we are not modifying the prompt to be returned here.
        prompt_ids = [tokenizer.bos_token_id] if getattr(tokenizer, "add_bos_token", False) else []
        prompt_ids += tokenizer(prompt, add_special_tokens=False)["input_ids"]
        len_prompt_ids = len(prompt_ids)
    else:
        len_prompt_ids = 0
    # Tokenize full text
    input_ids = tokenizer(
        full_text,
        padding=padding,
        truncation=truncation,
        max_length=seq_length,
    )["input_ids"]

    # Create assistant_masks: 0 for prompt tokens, 1 for answer tokens
    assistant_masks = [0] * len_prompt_ids + [1] * (len(input_ids) - len_prompt_ids)

    return _package_tokenized_example(
        tokenizer=tokenizer,
        input_ids=input_ids,
        assistant_masks=assistant_masks,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
        truncation=truncation,
        padding=padding,
    )


def format_chat_template(
    tokenizer: "PreTrainedTokenizer",
    formatted_text: List[Dict[str, str]],
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int] = None,
    padding: Union[str, bool] = "do_not_pad",
    truncation: Union[str, bool] = "do_not_truncate",
    tools: Optional[List[Dict]] = None,
    answer_only_loss_mask: bool = True,
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
        answer_only_loss_mask: Whether to compute the loss mask only on the answer tokens.

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
        padding=padding,
        truncation=truncation,
        max_length=seq_length,
    )

    # Choose the last conversation as answer other history are context by finding the last masked token
    # which indicates end of context and beginning of answer
    input_ids = tokenized_chat.get("input_ids")
    if template_has_generation_kwd:
        mask = tokenized_chat["assistant_masks"]
    elif not template_has_generation_kwd and answer_only_loss_mask:
        # in this case we need to manually split up the formatted_text. Only the final assistant turn should be considered as answer.
        answer_text = formatted_text.pop()
        assert answer_text["role"] == "assistant", "The last message in the formatted_text must be an assistant message"
        tokenized_prompt = tokenizer.apply_chat_template(
            formatted_text,
            tools=tools,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=template_has_generation_kwd,
            padding=padding,
            truncation=truncation,
            max_length=seq_length,
        )
        len_prompt_ids = len(tokenized_prompt.get("input_ids", []))
        mask = [0] * len_prompt_ids + [1] * (len(input_ids) - len_prompt_ids)
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
        truncation=truncation,
        padding=padding,
    )
