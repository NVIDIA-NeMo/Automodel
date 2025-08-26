# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from datasets import load_dataset

from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    format_chat_template,
    format_prompt_completion,
)


def _formatting_prompts_func(example, tokenizer, eos_token_id, pad_token_id, seq_length=None):
    question = example["question"]
    context = example["context"]
    answer = example["answers"]["text"][0].strip() if example["answers"]["text"] else ""
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    return format_prompt_completion(
        tokenizer=tokenizer,
        prompt=prompt,
        answer=answer,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
    )


def _formatting_prompts_func_with_chat_template(
    example, tokenizer, eos_token_id, pad_token_id, seq_length=None, start_of_turn_token=None
):
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0].strip()
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    return format_chat_template(
        tokenizer=tokenizer,
        prompt=prompt,
        answer=answer,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
        start_of_turn_token=start_of_turn_token,
    )


def make_squad_dataset(
    tokenizer,
    seq_length=None,
    limit_dataset_samples=None,
    start_of_turn_token=None,
    fp8=False,
    split="train",
    dataset_name="squad",
):
    """
    Load and preprocess a SQuAD-style QA dataset for model fine-tuning.

    This function retrieves the specified split of the SQuAD dataset, applies
    either a simple prompt–completion format or a chat‐template format
    (if `tokenizer.chat_template` is set), tokenizes each example,
    constructs `input_ids` and `labels`, and optionally pads
    all sequences to a fixed length.

    Args:
        tokenizer: A Hugging Face tokenizer with attributes
            `eos_token_id`, optional `bos_id`, optional `eos_id`, and
            optionally `chat_template`/`apply_chat_template`.
        seq_length (int, optional): If set, pad/truncate each example to this
            length.
        limit_dataset_samples (int, optional): If set, limit the number of
            examples loaded from the split.
        start_of_turn_token (str or None): If using a chat template, the
            token that marks the start of each turn. Used to compute the
            response offset for `labels`.
        fp8 (bool): Flag for future use (e.g., mixed precision). Currently
            unused.
        split (str): Which split of the dataset to load (e.g. 'train',
            'validation').
        dataset_name (str): Identifier for the Hugging Face dataset
            (default "rajpurkar/squad").

    Returns:
        A Hugginggth Face Dataset where each example is a dict with keys:
        - `input_ids`: List of token IDs for the prompt + answer.
        - `labels`: List of token IDs shifted for language modeling.
          to the loss (answers only).
    """

    if limit_dataset_samples is not None:
        assert isinstance(limit_dataset_samples, int), "Expected limit_dataset_samples to be an int"
        if not "[" in split:
            split = f"{split}[:{limit_dataset_samples}]"
        else:
            logging.warning(f"Dataset split {split} already has a slice, skipping limit_dataset_samples")
    dataset = load_dataset(dataset_name, split=split)

    # format the dataset
    chat_template = getattr(tokenizer, "chat_template", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", 0)
    # if pad_token_id is not set, use eos_token_id
    # therefore, pad_token can either [PAD] or [EOS]
    pad_token_id = _add_pad_token(tokenizer) or eos_token_id

    if chat_template is None:
        fmt_fn = lambda x: _formatting_prompts_func(x, tokenizer, eos_token_id, pad_token_id, seq_length)
    else:
        fmt_fn = lambda x: _formatting_prompts_func_with_chat_template(
            x, tokenizer, eos_token_id, pad_token_id, seq_length, start_of_turn_token
        )  # noqa: E731

    # map the dataset
    return dataset.map(
        fmt_fn,
        batched=False,
        remove_columns=["id", "title", "context", "question", "answers"],
    )
