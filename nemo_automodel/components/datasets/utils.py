# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import math
from typing import Optional

import torch


def batchify(tensor):
    """
    Ensures that the input tensor has at least two dimensions by adding an extra batch dimension if necessary.

    Args:
        tensor (torch.Tensor): The input tensor to be batchified.

    Returns:
        torch.Tensor:  The tensor with an extra dimension added if it was originally 1-dimensional.
        Otherwise, the tensor is returned as-is.
    """
    if tensor.ndim == 1:
        return tensor.unsqueeze_(0)
    return tensor


def extract_key_from_dicts(batch, key):
    """
    Extracts the value of the given key from each dictionary in a list of dictionaries.

    Args:
        batch (List[dict]): A list of dictionaries.
        key (str): The key whose values are to be extracted from each dictionary.

    Returns:
        List: A list of values associated with the specified key, in the same order as
        the dictionaries in the input batch.
    """
    return list(map(lambda x: x[key], batch))


def pad_within_micro(batch, pad_token_id, pad_seq_len_divisible=None):
    """
    Pads each list in a batch of lists to the same length with a specified token.

    Args:
        batch (List[List[int]]): A batch of sequences (e.g., token IDs), where each sequence
            is a list of integers.
        pad_token_id (int): The token ID to use for padding shorter sequences.
        pad_seq_len_divisible (int): The value to use for padding sequence length so that it is
            divisible by pad_seq_len_divisible.

    Returns:
        List[List[int]]: A batch of sequences where each inner list has been padded with the pad
        token to match the length of the longest sequence in the batch.
    """
    max_len = max(map(len, batch))
    if pad_seq_len_divisible:
        max_len = (pad_seq_len_divisible - max_len % pad_seq_len_divisible) + max_len
    if pad_token_id is None:
        # if it's none, extend the last token
        pad_token_id = batch[0][-1]
    return [item + [pad_token_id] * (max_len - len(item)) for item in batch]


def find_last_non_pad_token(lst: list[int], value: int) -> int | None:
    # lst = [optional-value .., non-value, ..., non-value, value, ...]
    # return the index of the last non-value token
    i = len(lst) - 1
    found = False
    while i >= 0:
        if lst[i] == value:
            i -= 1
            found = True
        else:
            if found:
                return i
            else:
                return None
    return None


def get_pad_token_from_key(val: str, pad_token_ids: Optional[dict[str, int]] = None) -> int | None:
    PAD_TOKEN_IDS = {
        "labels": -100,
        "attention_mask": 0,
        "loss_mask": 0,
    }
    if pad_token_ids is not None and val in pad_token_ids:
        return pad_token_ids[val]
    return PAD_TOKEN_IDS.get(val, None)


def make_attention_mask_from_labels(ids: list[int], ignore_token: int = -100) -> list[int]:
    # if the last token is not an ignore token, then the attention mask is all 1s
    if len(ids) == 0:
        return []
    if ids[-1] != ignore_token:
        ans = [1] * len(ids)
    else:
        # otherwise, find the last non-pad token and set the attention mask to 1s up to that point
        last_non_pad_token_pos = find_last_non_pad_token(ids, ignore_token)
        if last_non_pad_token_pos is None:
            ans = [1] * len(ids)
        else:
            ans = [1] * (last_non_pad_token_pos + 1)
        ans = ans + [0] * (len(ids) - len(ans))
    assert len(ans) == len(ids)
    return ans


def default_collater(batch, pad_seq_len_divisible=None):
    """
    Default batch collator that handles padding and batching.

    Args:
        batch: A batch of examples.
        pad_seq_len_divisible: If provided, pad sequence length to be divisible by this value.

    Returns:
        dict: A dictionary containing batched tensors.
    """
    pad_token_ids = batch[0].pop("___PAD_TOKEN_IDS___", None)
    # ans contains a dict with:
    # key: str (e.g., "input_ids", "attention_mask", "labels", "loss_mask")
    # value: list[list[int]] (e.g., [[1, 2, 3], [4, 5, 6]])
    ans = {
        key: pad_within_micro(
            extract_key_from_dicts(batch, key),
            get_pad_token_from_key(key, pad_token_ids),
            pad_seq_len_divisible,
        )
        for key in batch[0].keys()
    }

    # convert to tensors
    return {k: batchify(torch.LongTensor(v)) for k, v in ans.items()}


def packed_sequence_thd_collater(batch):
    """
    Collater for packed sequences in THD (total, hidden, depth) format without padding.

    This collater is designed for THD format, where multiple variable-length
    sequences are concatenated with/without padding tokens between them. The THD format represents
    sequences as (total_tokens, hidden_dim, depth) where total_tokens is the sum of all sequence
    lengths in the batch.

    Unlike traditional padding-based approaches (BSHD/SBHD formats), this THD format:
    - Concatenates sequences directly without padding: [a a a b b c c c c]
    - Uses seq_lens to identify sequence boundaries
    - Supports optional identifier or padding tokens between sequences via seq_lens_padded

    Args:
        batch (List[dict]): A list of dictionaries, where each dictionary represents one example
            with concatenated sequences. Each dictionary should contain:
            - 'input_ids': List of token IDs for all packed sequences
            - 'labels': List of labels for all packed sequences
            - 'position_ids': List of position IDs for all packed sequences
            - 'seq_lens': List of actual sequence lengths (used to compute cu_seqlens)
            - 'seq_lens_padded': List of sequence lengths including identifier tokens

            Example:
            [
                {'input_ids': [1,2,3,4,5,...], 'seq_lens': [3, 2, ...], 'seq_lens_padded': [4, 3, ...]},
                {'input_ids': [6,7,8,9,...], 'seq_lens': [2, 2, ...], 'seq_lens_padded': [3, 3, ...]},
            ]

            In this example, if seq_lens = [3, 2] and seq_lens_padded = [4, 3], it means:
            - First sequence has 3 real tokens followed by 1 identifier token
            - Second sequence has 2 real tokens followed by 1 identifier token
            - cumulative seq_lens would be [0, 3, 5]
            - cumulative seq_lens_padded would be [0, 4, 7] (for sequence boundaries)

    Returns:
        dict: A dictionary with concatenated tensors:
            - 'input_ids': tensor of shape [total_tokens] - all sequences concatenated
            - 'labels': tensor of shape [total_tokens] - all labels concatenated
            - 'position_ids': tensor of shape [total_tokens] - position IDs for each token
            - 'seq_lens': tensor of shape [total_sequences] - cumulative sequence lengths for attention
            - 'seq_lens_padded': tensor of shape [total_sequences] - cumulative lengths including identifiers
    """
    # Remove padding token IDs if present (not used in passthrough)
    if len(batch) > 0 and "___PAD_TOKEN_IDS___" in batch[0]:
        for item in batch:
            item.pop("___PAD_TOKEN_IDS___", None)

    # Extract all keys from the first batch item
    if len(batch) == 0:
        return {}

    tokens = torch.cat([torch.tensor(x["input_ids"]) for x in batch])
    labels = torch.cat([torch.tensor(x["labels"]) for x in batch])
    position_ids = torch.cat([torch.tensor(x["position_ids"]) for x in batch])
    seq_lens = torch.cat([torch.tensor(x["seq_lens"]) for x in batch])
    seq_lens_padded = torch.cat([torch.tensor(x["seq_lens_padded"]) for x in batch])

    return {
        "input_ids": tokens,
        "labels": labels,
        "position_ids": position_ids,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
    }


class SFTSingleTurnPreprocessor:
    """
    Generic single-turn text-to-text SFT (supervised-fine-tuning) pre-processor.

    Args:
        tokenizer: Pre-trained tokenizer (HF).
    """

    def __init__(self, tokenizer):
        """
        SFTSingleTurnPreprocessor constructor.

        Args:
            tokenizer: Pretrained tokenizer.
        """
        self.tokenizer = tokenizer
        self.block_size = None
        self.preprocessing_num_workers = 1
        self.overwrite_cache = False
        self.pad_to_max_length = True

    def _tokenize_function(self, examples, dataset):
        ctx = dataset.get_context(examples)
        tgt = dataset.get_target(examples)

        ctx_tok = self.tokenizer(ctx)
        tgt_tok = self.tokenizer(tgt)

        # strip trailing special token from context
        if len(ctx_tok["input_ids"][0]) > 0 and ctx_tok["input_ids"][0][-1] in self.tokenizer.all_special_ids:
            ctx_tok["input_ids"] = [ids[:-1] for ids in ctx_tok["input_ids"]]
            ctx_tok["attention_mask"] = [m[:-1] for m in ctx_tok["attention_mask"]]

        # strip leading special token from target
        if len(tgt_tok["input_ids"][0]) > 0 and tgt_tok["input_ids"][0][0] in self.tokenizer.all_special_ids:
            tgt_tok["input_ids"] = [ids[1:] for ids in tgt_tok["input_ids"]]
            tgt_tok["attention_mask"] = [m[1:] for m in tgt_tok["attention_mask"]]

        out = {}
        out["input_ids"] = [
            c_ids + t_ids for c_ids, t_ids in zip(ctx_tok["input_ids"], tgt_tok["input_ids"], strict=False)
        ]
        out["attention_mask"] = [
            c_m + t_m for c_m, t_m in zip(ctx_tok["attention_mask"], tgt_tok["attention_mask"], strict=False)
        ]
        # label: -100 for ctx, true ids for tgt
        out["labels"] = [
            [-100] * (len(c_ids) - 1) + t_ids + [-100]
            for c_ids, t_ids in zip(ctx_tok["input_ids"], tgt_tok["input_ids"], strict=False)
        ]

        out["loss_mask"] = [[1 if t != -100 else 0 for t in lbl] for lbl in out["labels"]]
        return out

    def _compute_dataset_max_len(self, tokenized_ds):
        max_len = max(map(lambda x: len(x["input_ids"]), tokenized_ds))
        # make multiple of 8
        max_len = math.ceil(max_len / 8) * 8
        # respect model block size
        if self.block_size is not None:
            max_len = min(max_len, self.block_size)
        return max_len

    def _pad_function(self, max_len):
        tk = self.tokenizer

        def _pad(examples):
            pad_id = tk.pad_token_id or 0
            examples["input_ids"] = [
                (ids[:max_len] + [pad_id] * max(0, max_len - len(ids))) for ids in examples["input_ids"]
            ]
            examples["attention_mask"] = [
                ([1] * min(len(ids), max_len) + [0] * max(0, max_len - len(ids))) for ids in examples["attention_mask"]
            ]
            examples["labels"] = [(lbl[:max_len] + [-100] * max(0, max_len - len(lbl))) for lbl in examples["labels"]]
            examples["loss_mask"] = [(lm[:max_len] + [0] * max(0, max_len - len(lm))) for lm in examples["loss_mask"]]
            # return dictionary with sequences all exactly `max_len` long
            return examples

        return _pad

    def process(self, raw_dataset, ds):
        """
        Main processor entry.

        Args:
            raw_dataset (datasets.DatasetDict): the dataset (e.g. returned by load_dataset)
            ds (dataset): the dataset with get_target method.

        Returns:
            datasets.DatasetDict: tokenized + optionally padded datasets (all splits preserved).
        """
        if not hasattr(self.tokenizer, "pad_token") and hasattr(self.tokenizer, "bos_token"):
            self.tokenizer.pad_token = self.tokenizer.bos_token

        # 1. tokenise ----------------------------------------------------------------
        tokenized = raw_dataset.map(
            lambda x: self._tokenize_function(x, dataset=ds),
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=raw_dataset.column_names,
            load_from_cache_file=not self.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        # 2. pad (optional) ----------------------------------------------------------
        if self.pad_to_max_length:
            # 2a. compute global max len
            max_len = self._compute_dataset_max_len(tokenized)

            # 2b. pad to max len
            pad_fn = self._pad_function(max_len)
            tokenized = tokenized.map(
                pad_fn,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                load_from_cache_file=not self.overwrite_cache,
                desc=f"Padding dataset to max length {max_len}",
            )

        return tokenized
