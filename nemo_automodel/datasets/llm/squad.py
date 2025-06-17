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
from datasets import load_dataset


def make_squad_dataset(
    tokenizer,
    seq_length=None,
    limit_dataset_samples=None,
    start_of_turn_token=None,
    fp8=False,
    split="train",
    dataset_name="rajpurkar/squad",
):
    """
    Load and preprocess a SQuAD-style QA dataset for model fine-tuning.

    This function retrieves the specified split of the SQuAD dataset, applies
    either a simple prompt–completion format or a chat‐template format
    (if `tokenizer.chat_template` is set), tokenizes each example,
    constructs `input_ids`, `labels`, and `loss_mask`, and optionally pads
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
            response offset for `loss_mask`.
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
        - `loss_mask`: List of 0/1 flags indicating which tokens contribute
          to the loss (answers only).
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", 0)
    chat_template = getattr(tokenizer, "chat_template", None)

    def pad_to_seq_length(sample):
        seq_pad_len_ar = max(0, seq_length - len(next(iter(sample.values()))))
        return {k: v + [eos_token_id if v != "loss_mask" else 0] * seq_pad_len_ar for k, v in sample.items()}

    def formatting_prompts_func(example):
        formatted_text = [
            f"{example['context']} {example['question']} ",
            example["answers"]["text"][0].strip(),
        ]
        context_ids, answer_ids = list(map(lambda x: tokenizer(x)["input_ids"], formatted_text))
        bos_id = getattr(tokenizer, "bos_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        # Remove EOS token from context's end
        if len(context_ids) > 0 and context_ids[-1] == eos_id:
            context_ids = context_ids[:-1]
        # Remove BOS token from answer's start
        if len(answer_ids) > 0 and answer_ids[0] == bos_id:
            answer_ids = answer_ids[1:]

        input_ids = context_ids + answer_ids
        return dict(
            input_ids=input_ids,
            labels=input_ids[1:] + [eos_token_id or input_ids[-1]],
            loss_mask=[0] * len(context_ids) + [1] * len(answer_ids),
        )

    def formatting_prompts_func_with_chat_template(example, start_of_turn_token=None):
        formatted_text = [
            {"role": "user", "content": f"{example['context']} {example['question']}"},
            {"role": "assistant", "content": example["answers"]["text"][0].strip()},
        ]
        input_ids = tokenizer.apply_chat_template(formatted_text)
        if isinstance(start_of_turn_token, str):
            start_of_turn_token_id = tokenizer(start_of_turn_token, add_special_tokens=False)["input_ids"][0]
            first_start_of_turn_token_id = input_ids.index(start_of_turn_token_id)
            response_start = input_ids.index(start_of_turn_token_id, first_start_of_turn_token_id + 1) + 1
        else:
            response_start = 0
        loss_mask = [0] * response_start + [1] * (len(input_ids) - response_start)
        return dict(
            input_ids=input_ids,
            labels=input_ids[1:] + [getattr(tokenizer, "eos_token_id", None) or input_ids[-1]],
            loss_mask=loss_mask,
        )

    if limit_dataset_samples is not None:
        assert isinstance(limit_dataset_samples, int), "Expected limit_dataset_samples to be an int"
        split = f"{split}[:{limit_dataset_samples}]"

    dataset = load_dataset(dataset_name, split=split)

    fmt_fn = formatting_prompts_func
    if chat_template is not None:
        fmt_fn = lambda x: formatting_prompts_func_with_chat_template(x, start_of_turn_token)  # noqa: E731

    if isinstance(seq_length, int):
        fmt_fn_ = fmt_fn
        fmt_fn = lambda x: pad_to_seq_length(fmt_fn_(x))  # noqa: E731

    return dataset.map(
        fmt_fn,
        batched=False,
        remove_columns=["id", "title", "context", "question", "answers"],
    )
