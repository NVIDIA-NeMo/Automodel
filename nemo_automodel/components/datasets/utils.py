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
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


def batchify(tensor, default_tensor_cls=torch.LongTensor):
    """
    Ensures that the input tensor has at least two dimensions by adding an extra batch dimension if necessary.

    Args:
        tensor (torch.Tensor): The input tensor to be batchified.

    Returns:
        torch.Tensor:  The tensor with an extra dimension added if it was originally 1-dimensional.
        Otherwise, the tensor is returned as-is.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = default_tensor_cls(tensor)
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
        "input_ids": 0,
    }
    if pad_token_ids is None:
        pad_token_ids = {}
    ans = pad_token_ids.get(val, PAD_TOKEN_IDS.get(val, None))
    return ans


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


def create_causal_mask_mapping(
    model_config,
    batch_size,
    seq_len,
    position_ids=None,
    attention_mask=None,
    device=None,
):
    """
    Create causal mask mapping for pipeline parallelism.

    This is the core mask creation logic that can be reused by different collate functions.
    Extracts common mask creation logic to avoid duplication between collate functions.

    Args:
        model_config: HuggingFace model config
        batch_size: Batch size
        seq_len: Sequence length
        position_ids: Optional position IDs tensor [batch_size, seq_len]
        attention_mask: Optional 2D attention mask tensor [batch_size, seq_len] for padding
        device: Device to create tensors on (defaults to cpu)

    Returns:
        dict: Mapping of mask types to 4D mask tensors
            - "full_attention": [batch_size, 1, seq_len, seq_len]
            - "sliding_attention": [batch_size, 1, seq_len, seq_len] (if model uses sliding window)
    """
    if device is None:
        device = torch.device("cpu")

    # Create position_ids if not provided
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Prepare mask creation kwargs
    mask_kwargs = {
        "config": model_config,
        "input_embeds": torch.empty((batch_size, seq_len), device=device),
        "attention_mask": attention_mask,
        "cache_position": position_ids[0],  # Use first row (all rows identical for non-padded data)
        "past_key_values": None,  # Training only
        "position_ids": position_ids,
    }

    # Create causal masks
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
    }

    # Add sliding window mask if model uses it
    if hasattr(model_config, "sliding_window") and model_config.sliding_window is not None:
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    return causal_mask_mapping


def add_causal_masks_to_batch(batch_dict, model_config):
    """
    Add precomputed causal masks to an already-batched data dict.

    This function is designed for datasets that yield complete batches (like MockIterableDataset),
    where we want to add mask precomputation as a separate processing step.

    Args:
        batch: A dict or list containing a single batched dict with tensors:
            - input_ids: [batch_size, seq_length]
            - position_ids: [batch_size, seq_length] (optional)
            - labels: [batch_size, seq_length]
        model_config: HuggingFace model config for creating causal masks
        precompute_masks: If False, skip mask creation (for compatibility with train_ft.py wrapper)

    Returns:
        dict: Same batch with added causal_mask_mapping field
    """
    # Extract info from batch
    batch_size = batch_dict["input_ids"].shape[0]
    seq_len = batch_dict["input_ids"].shape[1]
    position_ids = batch_dict.get("position_ids")
    attention_mask = batch_dict.get("attention_mask")  # May have padding info

    # Create causal masks using the shared helper function
    causal_mask_mapping = create_causal_mask_mapping(
        model_config=model_config,
        batch_size=batch_size,
        seq_len=seq_len,
        position_ids=position_ids,
        attention_mask=attention_mask,
        device=batch_dict["input_ids"].device,
    )

    batch_dict["causal_mask_mapping"] = causal_mask_mapping
    return batch_dict


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
    Collater for packed sequences in THD (total, hidden, depth) format.

    This collater is designed for THD format, where multiple variable-length
    sequences are concatenated with/without padding tokens between them. The THD format represents
    sequences as (total_tokens, hidden_dim, depth) where total_tokens is the sum of all sequence
    lengths in the batch.

    Unlike traditional padding-based approaches (BSHD/SBHD formats), this THD format:
    - Concatenates sequences directly: [a a a b b c c c c]
    - Uses seq_lens to identify sequence boundaries for attention computation
    - Supports optional identifier or padding tokens between sequences via seq_lens_padded

    This collater supports both pipeline parallelism (PP) and non-PP use cases by:
    - Stacking token-level tensors (input_ids, labels, position_ids) along batch dimension
    - Padding and stacking seq_lens and seq_lens_padded with sentinel value -1000
    - Including 'qkv_format': 'thd' in the output to indicate THD format

    IMPORTANT: All examples in the batch must have the same token sequence length for input_ids,
    labels, and position_ids. This is typically ensured by the dataset/packing logic that creates
    fixed-length packed sequences.

    Args:
        batch (List[dict]): A list of dictionaries, where each dictionary represents one packed example.
            Each dictionary should contain:
            - 'input_ids': List[int] - Token IDs for all packed sequences (must be same length across batch)
            - 'labels': List[int] - Labels for all packed sequences (must be same length across batch)
            - 'position_ids': List[int] - Position IDs for all tokens (must be same length across batch)
            - 'seq_lens': List[int] - Actual sequence lengths for each packed sequence
            - 'seq_lens_padded': List[int] - Sequence lengths including identifier/padding tokens

            Example batch with 2 examples, both with 6 total tokens:
            [
                {
                    'input_ids': [1, 2, 3, 99, 4, 5],  # Two sequences: [1,2,3] and [4,5] with sep token 99
                    'labels': [1, 2, 3, -100, 4, 5],
                    'position_ids': [0, 1, 2, 0, 0, 1],
                    'seq_lens': [3, 2],  # Actual sequence lengths (excluding separator)
                    'seq_lens_padded': [4, 2]  # Including separator token
                },
                {
                    'input_ids': [6, 7, 99, 8, 9, 10],  # Two sequences with separator
                    'labels': [6, 7, -100, 8, 9, 10],
                    'position_ids': [0, 1, 0, 0, 1, 2],
                    'seq_lens': [2, 3],
                    'seq_lens_padded': [3, 3]
                }
            ]

    Returns:
        dict: A dictionary with batched tensors:
            - 'input_ids': tensor of shape [batch_size, seq_len] - stacked token sequences
            - 'labels': tensor of shape [batch_size, seq_len] - stacked labels
            - 'position_ids': tensor of shape [batch_size, seq_len] - stacked position IDs
            - 'seq_lens': tensor of shape [batch_size, max_num_packs] - padded sequence lengths
            - 'seq_lens_padded': tensor of shape [batch_size, max_num_packs] - padded lengths with separators
            - 'qkv_format': str - Always 'thd' to indicate THD format

        Note: seq_lens and seq_lens_padded are padded with -1000 to handle variable number of
        packed sequences per example. These sentinel values should be filtered out before use.
    """
    # Remove padding token IDs if present (not used in passthrough)
    if len(batch) > 0 and "___PAD_TOKEN_IDS___" in batch[0]:
        for item in batch:
            item.pop("___PAD_TOKEN_IDS___", None)

    # Extract all keys from the first batch item
    if len(batch) == 0:
        return {}

    tokens = batchify(torch.stack([torch.tensor(x["input_ids"]) for x in batch]))
    labels = batchify(torch.stack([torch.tensor(x["labels"]) for x in batch]))
    position_ids = batchify(torch.stack([torch.tensor(x["position_ids"]) for x in batch]))

    seq_lens = batchify(torch.LongTensor(pad_within_micro([x["seq_lens"] for x in batch], -1000)))
    seq_lens_padded = batchify(torch.LongTensor(pad_within_micro([x["seq_lens_padded"] for x in batch], -1000)))

    return {
        "input_ids": tokens,
        "labels": labels,
        "position_ids": position_ids,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
        "qkv_format": "thd",
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

        # 1. tokenise
        tokenized = raw_dataset.map(
            lambda x: self._tokenize_function(x, dataset=ds),
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=raw_dataset.column_names,
            load_from_cache_file=not self.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        # 2. pad (optional)
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
