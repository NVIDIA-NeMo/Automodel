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
# from nemo_automodel.datasets.llm.hf_dataset import HFDatasetBuilder
from datasets import Dataset, DatasetDict, load_dataset

def get_eos_token_id(tokenizer):
    return getattr(tokenizer, 'eos_token_id', None) or getattr(tokenizer, 'eos_id', None)

def get_chat_template(tokenizer):
    # attempt to unwrap NeMo's tokenizer wrapper and check if wrapped tokenizer has chat_template
    return getattr(tokenizer, 'chat_template', None)


def make_squad_dataset(
    tokenizer,
    seq_length=None,
    packed_sequence_size=None,
    limit_dataset_samples=None,
    start_of_turn_token=None,
    fp8=False,
    split='train',
    dataset_name="rajpurkar/squad",
):
    eos_token_id = get_eos_token_id(tokenizer)
    chat_template = get_chat_template(tokenizer)

    def pad_to_seq_length(sample):
        seq_pad_len_ar = max(0, seq_length - len(next(iter(sample.values()))))
        return {k: v + [eos_token_id if v != 'loss_mask' else 0] * seq_pad_len_ar for k, v in sample.items()}

    def formatting_prompts_func(example):
        formatted_text = [
            f"{example['context']} {example['question']} ",
            example['answers']['text'][0].strip(),
        ]
        context_ids, answer_ids = list(map(lambda x: tokenizer(x)['input_ids'], formatted_text))
        bos_id = getattr(tokenizer, 'bos_id', None)
        eos_id = getattr(tokenizer, 'eos_id', None)
        if len(context_ids) > 0 and bos_id is not None and context_ids[0] != bos_id:
            context_ids.insert(0, bos_id)
        if len(answer_ids) > 0 and eos_id is not None and answer_ids[-1] != eos_id:
            answer_ids.append(eos_id)

        input_ids = context_ids + answer_ids
        return dict(
            input_ids=input_ids,
            labels=input_ids[1:] + [eos_token_id or input_ids[-1]],
            loss_mask=[0] * len(context_ids) + [1] * len(answer_ids),
        )

    def formatting_prompts_func_with_chat_template(example, start_of_turn_token=None):
        formatted_text = [
            {'role': 'user', 'content': f"{example['context']} {example['question']}"},
            {'role': 'assistant', 'content': example['answers']['text'][0].strip()},
        ]
        input_ids = tokenizer.apply_chat_template(formatted_text)
        if isinstance(start_of_turn_token, str):
            start_of_turn_token_id = tokenizer(start_of_turn_token, add_special_tokens=False)['input_ids'][0]
            first_start_of_turn_token_id = input_ids.index(start_of_turn_token_id)
            response_start = input_ids.index(start_of_turn_token_id, first_start_of_turn_token_id + 1) + 1
        else:
            response_start = 0
        loss_mask = [0] * response_start + [1] * (len(input_ids) - response_start)
        return dict(
            input_ids=input_ids,
            labels=input_ids[1:] + [getattr(tokenizer, 'eos_token_id', None) or input_ids[-1]],
            loss_mask=loss_mask,
        )

    if limit_dataset_samples is not None:
        assert isinstance(limit_dataset_samples, int), "Expected limit_dataset_samples to be an int"
        split = f'{split}[:{limit_dataset_samples}]'

    if isinstance(packed_sequence_size, int) and packed_sequence_size > 0:
        raise NotImplemented("packed is WIP")
        # datamodule = llm.HFDatasetDataModulePacked(
        #     "rajpurkar/squad",
        #     packed_sequence_size=packed_sequence_size,
        #     split=splits,
        #     micro_batch_size=micro_batch_size,
        #     pad_token_id=tokenizer.eos_id if tokenizer.eos_id is not None else 0,
        #     pad_seq_len_divisible=16 if fp8 else None,  # FP8 training requires seq length to be divisible by 16.
        # )
    else:
        dataset = load_dataset(dataset_name, split=split)

    fmt_fn = formatting_prompts_func
    if chat_template is not None:
        fmt_fn = lambda x: formatting_prompts_func_with_chat_template(x, start_of_turn_token)
    if isinstance(seq_length, int):
        fmt_fn_ = fmt_fn
        fmt_fn = lambda x: pad_to_seq_length(fmt_fn_(x))

    return dataset.map(
        fmt_fn,
        batched=False,
        remove_columns=["id", "title", "context", "question", 'answers'],
    )

