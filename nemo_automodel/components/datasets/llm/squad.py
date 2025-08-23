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


def _pad_to_seq_length(sample, pad_token_id, seq_length):
    n = seq_length - len(sample)
    if n == 0:
        return sample
    return sample + [pad_token_id] * n


def _add_pad_token(tokenizer):
    pad_token_id = None
    if not hasattr(tokenizer, "pad_token_id"):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id
    if not hasattr(tokenizer, "pad_token") and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    return pad_token_id


def _package_tokenized_example(has_chat_template, input_ids, eos_token_id, pad_token_id, seq_length, context_len):
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


def _formatting_prompts_func(example, tokenizer, eos_token_id, pad_token_id, seq_length=None):
    question = example["question"]
    context = example["context"]
    answer = example["answers"]["text"][0].strip() if example["answers"]["text"] else ""

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    full_text = prompt + " " + answer

    # Tokenize separately to locate answer start
    prompt_ids = tokenizer(prompt)["input_ids"]
    # Tokenize full text
    input_ids = tokenizer(full_text)["input_ids"]
    return _package_tokenized_example(False, input_ids, eos_token_id, pad_token_id, seq_length, len(prompt_ids))


def _formatting_prompts_func_with_chat_template(
    example, tokenizer, eos_token_id, pad_token_id, seq_length=None, start_of_turn_token=None
):
    formatted_text = [
        {"role": "user", "content": f"{example['context']} {example['question']}"},
        {"role": "assistant", "content": example["answers"]["text"][0].strip()},
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
