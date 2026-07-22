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
        max_len = math.ceil(max_len / pad_seq_len_divisible) * pad_seq_len_divisible
    if pad_token_id is None:
        # if it's none, extend the last token
        pad_token_id = batch[0][-1]
    return [item + [pad_token_id] * (max_len - len(item)) for item in batch]


def find_last_non_pad_token(lst: list[int], value: int) -> int | None:
    """Return the last non-padding index before a trailing padding run."""
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
    """Return the default pad token id for a batch field name."""
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
    """Build an attention mask from labels with trailing ignored positions."""
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
        "inputs_embeds": torch.empty((batch_size, seq_len), device=device),
        "attention_mask": attention_mask,
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


def default_collater(
    batch: list[dict[str, list[int] | torch.Tensor]],
    pad_seq_len_divisible: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Default batch collator that handles padding and batching.

    Sequence-list fields are padded and stacked. Pre-batched tensor fields are concatenated along their first
    axis, preserving all trailing axes, dtype, and device.

    Args:
        batch: Local examples. A list-valued field has shape ``[S_i]`` for example ``i``, where ``S_i`` is its
            sequence length, and becomes ``[B, S]`` after padding, where ``B`` is the number of examples and
            ``S`` is the padded maximum. A tensor-valued field is already batched as ``[B_i, ...]`` with
            arbitrary trailing axes and becomes ``[sum_i(B_i), ...]``. The optional
            ``___PAD_TOKEN_IDS___`` entry is removed from the first input mapping in place.
        pad_seq_len_divisible: If set, round padded ``S`` up to a multiple of this value.

    Returns:
        Mapping of fields to tensors. List-valued inputs become ``torch.int64`` tensors shaped ``[B, S]``;
        tensor-valued inputs preserve dtype/device and trailing layout. When ``attention_mask`` or
        ``input_ids`` is present with shape ``[B, S]``, ``padding_mask`` is a boolean ``[B, S]`` tensor.
    """
    pad_token_ids = batch[0].pop("___PAD_TOKEN_IDS___", None)
    # ans contains a dict with:
    # key: str (e.g., "input_ids", "attention_mask", "labels", "loss_mask")
    # value: list[list[int]] (e.g., [[1, 2, 3], [4, 5, 6]])
    ans = {}
    for key in batch[0].keys():
        values = extract_key_from_dicts(batch, key)
        if all(isinstance(v, torch.Tensor) for v in values):
            # Pre-batched fields: each value is a [batch_size, seq_len] tensor; concatenate along the
            # batch dim rather than treating it as a ragged list[int] to be padded.
            ans[key] = torch.cat([batchify(v) for v in values], dim=0)
        else:
            ans[key] = pad_within_micro(
                values,
                get_pad_token_from_key(key, pad_token_ids),
                pad_seq_len_divisible,
            )

    # convert to tensors (already-tensor fields are passed through batchify unchanged)
    result = {k: batchify(v if isinstance(v, torch.Tensor) else torch.LongTensor(v)) for k, v in ans.items()}

    # Add padding_mask. Prefer the real attention_mask: matching the pad token *value*
    # (input_ids == pad_token_id) misclassifies real tokens as padding whenever pad_token_id
    # collides with a content token (e.g. pad_token_id == eos_token_id), which masks real
    # eos/separator tokens out of the MoE experts -> near-random outputs on chat data.
    if "attention_mask" in result:
        result["padding_mask"] = ~result["attention_mask"].bool()
    elif "input_ids" in result:
        input_ids_pad_token = get_pad_token_from_key("input_ids", pad_token_ids) or 0
        result["padding_mask"] = (result["input_ids"] == input_ids_pad_token).bool()

    return result


def packed_sequence_thd_collater(batch: list[dict] | dict) -> dict:
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

    When batch items lack packed-sequence metadata (seq_lens, seq_lens_padded, position_ids),
    such as samples from ChatDataset, this collater synthesizes the missing fields so that each
    sample is treated as a single-sequence "pack". Variable-length sequences are padded to the
    longest length in the batch. This enables using THD format with TE context parallelism
    without requiring the dataset to perform actual sequence packing.

    Args:
        batch: A list of example dictionaries, or a pre-batched dictionary whose token tensors
            have shape ``[batch_size, seq_len]``. The latter supports iterable benchmark datasets
            used with ``DataLoader(batch_size=None)``.

            For pre-packed data, each dictionary should contain:
            - 'input_ids': List[int] - Token IDs for all packed sequences (must be same length across batch)
            - 'labels': List[int] - Labels for all packed sequences (must be same length across batch)
            - 'position_ids': List[int] - Position IDs for all tokens (must be same length across batch)
            - 'seq_lens': List[int] - Actual sequence lengths for each packed sequence
            - 'seq_lens_padded': List[int] - Sequence lengths including identifier/padding tokens

            For non-packed data (e.g. ChatDataset), each dictionary needs only:
            - 'input_ids': List[int] - Token IDs (variable length across batch)
            - 'labels': List[int] - Labels (same length as input_ids)
            - 'attention_mask': List[int] - (optional) 1 for real tokens, 0 for padding

            Example batch with 2 packed examples, both with 6 total tokens:
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
    # DataLoader passes a single item directly to collate_fn when batch_size=None. Benchmark
    # iterable datasets use that mode and already yield [B, S] tensors, so add THD metadata
    # without trying to stack the dictionary as a list of examples.
    if isinstance(batch, dict):
        if not batch:
            return {}

        input_ids = torch.as_tensor(batch["input_ids"])
        labels = torch.as_tensor(batch["labels"], device=input_ids.device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if input_ids.ndim != 2 or labels.shape != input_ids.shape:
            raise ValueError(
                "Pre-batched THD input_ids and labels must have matching [batch_size, seq_len] shapes; "
                f"got input_ids={tuple(input_ids.shape)}, labels={tuple(labels.shape)}."
            )

        batch_size, seq_len = input_ids.shape
        position_ids_value = batch.get("position_ids")
        if position_ids_value is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = torch.as_tensor(position_ids_value, device=input_ids.device)
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            if position_ids.shape != input_ids.shape:
                raise ValueError(
                    "Pre-batched THD position_ids must match input_ids shape; "
                    f"got position_ids={tuple(position_ids.shape)}, input_ids={tuple(input_ids.shape)}."
                )

        seq_lens_value = batch.get("seq_lens")
        if seq_lens_value is None:
            attention_mask = batch.get("attention_mask")
            if attention_mask is None:
                actual_seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=input_ids.device)
            else:
                attention_mask = torch.as_tensor(attention_mask, device=input_ids.device)
                if attention_mask.ndim == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                if attention_mask.shape != input_ids.shape:
                    raise ValueError(
                        "Pre-batched THD attention_mask must match input_ids shape; "
                        f"got attention_mask={tuple(attention_mask.shape)}, input_ids={tuple(input_ids.shape)}."
                    )
                actual_seq_lens = attention_mask.to(torch.long).sum(dim=1)
            seq_lens = actual_seq_lens.unsqueeze(1)
        else:
            seq_lens = torch.as_tensor(seq_lens_value, dtype=torch.long, device=input_ids.device)
            if seq_lens.ndim == 1:
                seq_lens = seq_lens.unsqueeze(1) if seq_lens.numel() == batch_size else seq_lens.unsqueeze(0)

        seq_lens_padded_value = batch.get("seq_lens_padded")
        if seq_lens_padded_value is None:
            seq_lens_padded = torch.full((batch_size, 1), seq_len, dtype=torch.long, device=input_ids.device)
        else:
            seq_lens_padded = torch.as_tensor(seq_lens_padded_value, dtype=torch.long, device=input_ids.device)
            if seq_lens_padded.ndim == 1:
                seq_lens_padded = (
                    seq_lens_padded.unsqueeze(1)
                    if seq_lens_padded.numel() == batch_size
                    else seq_lens_padded.unsqueeze(0)
                )

        result = {
            "input_ids": input_ids.to(torch.long),
            "labels": labels.to(torch.long),
            "position_ids": position_ids.to(torch.long),
            "seq_lens": seq_lens,
            "seq_lens_padded": seq_lens_padded,
            "qkv_format": "thd",
        }
        if "mock_data_fingerprint" in batch:
            result["mock_data_fingerprint"] = str(batch["mock_data_fingerprint"])
        return result

    # Extract and remove padding token metadata if present
    pad_token_ids = None
    if len(batch) > 0 and "___PAD_TOKEN_IDS___" in batch[0]:
        pad_token_ids = batch[0].get("___PAD_TOKEN_IDS___")
        for item in batch:
            item.pop("___PAD_TOKEN_IDS___", None)

    if len(batch) == 0:
        return {}

    # If batch items lack packed-sequence metadata (e.g. from ChatDataset),
    # synthesize seq_lens, seq_lens_padded, and position_ids so that each
    # sample is treated as a single-sequence "pack".
    if "seq_lens" not in batch[0]:
        input_ids_pad = get_pad_token_from_key("input_ids", pad_token_ids) or 0
        max_len = max(len(item["input_ids"]) for item in batch)

        for item in batch:
            cur_len = len(item["input_ids"])
            if "attention_mask" in item:
                actual_len = sum(item["attention_mask"])
                item.pop("attention_mask")
            else:
                actual_len = cur_len

            pad_amount = max_len - cur_len
            item["seq_lens"] = [actual_len]
            # seq_lens_padded must cover the full padded length so that
            # cu_seqlens_padded[-1] == total_tokens in the downstream THD pipeline.
            item["seq_lens_padded"] = [max_len]
            item["position_ids"] = list(range(max_len))

            if pad_amount > 0:
                item["input_ids"] = list(item["input_ids"]) + [input_ids_pad] * pad_amount
                item["labels"] = list(item["labels"]) + [-100] * pad_amount

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


def packed_sequence_thd_collater_vlm(examples, processor=None, **kwargs):
    """THD collater adapter for the VLM recipe's ``(examples, processor)`` call convention.

    ``recipes/vlm/finetune.py:build_dataloader`` invokes the configured collate as
    ``collate_cfg.instantiate(examples=examples, processor=processor)``, whereas
    :func:`packed_sequence_thd_collater` takes only ``batch``. This thin adapter accepts
    (and ignores) ``processor`` and any extra kwargs so pre-tokenized ``ChatDataset``
    samples can use THD / context-parallel batching inside the VLM recipe.

    Args:
        examples: Batch of samples, forwarded to :func:`packed_sequence_thd_collater`.
        processor: Unused; accepted only for VLM-recipe call-convention compatibility.

    Returns:
        dict: THD-format batch (see :func:`packed_sequence_thd_collater`).
    """
    return packed_sequence_thd_collater(examples)


def _indexed_mask_to_4d_block_causal(attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert an indexed attention mask to a 4D block-causal mask.

    Args:
        attention_mask: Integer tensor of shape ``[B, S]`` where each
            position contains the 1-based index of the sub-sequence it
            belongs to (0 = padding).

    Returns:
        Bool tensor of shape ``[B, 1, S, S]`` suitable for
        ``eager`` / ``sdpa`` attention backends.  ``True`` means the
        position is **allowed** to attend.
    """
    # attention_mask: [B, S]
    B, S = attention_mask.shape

    # same_doc[b, i, j] = True iff positions i and j belong to the same sub-sequence
    mask_q = attention_mask.unsqueeze(2)  # [B, S, 1]
    mask_k = attention_mask.unsqueeze(1)  # [B, 1, S]
    same_doc = mask_q == mask_k  # [B, S, S]

    # causal: position i can attend to position j only if j <= i
    causal = torch.ones(S, S, dtype=torch.bool, device=attention_mask.device).tril()  # [S, S]

    # not_padding: both positions must be non-padding (index > 0)
    not_padding_q = (attention_mask > 0).unsqueeze(2)  # [B, S, 1]
    not_padding_k = (attention_mask > 0).unsqueeze(1)  # [B, 1, S]

    mask_4d = same_doc & causal.unsqueeze(0) & not_padding_q & not_padding_k  # [B, S, S]

    return mask_4d.unsqueeze(1)  # [B, 1, S, S]


def neat_packed_collater(batch: list[dict], attn_implementation: str = "sdpa") -> dict:
    """Collater for neat-packed LLM sequences.

    Stacks ``input_ids``, ``labels``, ``position_ids`` and converts the
    indexed ``attention_mask`` to the format required by the attention backend.

    For flash attention (``flash_attention_2`` / ``flash_attention_3`` /
    ``flash_attention_4``): keeps the indexed 2D mask ``[B, S]``.
    For ``sdpa`` / ``eager``: converts to a 4D block-causal float mask.

    Args:
        batch: List of sample dicts produced by ``neat_pack_dataset``.
        attn_implementation: Attention backend (``"flash_attention_2"``,
            ``"sdpa"``, or ``"eager"``).

    Returns:
        Dict with batched tensors ready for model forward.
    """
    if not batch:
        return {}

    input_ids = batchify(torch.stack([torch.as_tensor(x["input_ids"]) for x in batch]))
    labels = batchify(torch.stack([torch.as_tensor(x["labels"]) for x in batch]))
    position_ids = batchify(torch.stack([torch.as_tensor(x["position_ids"]) for x in batch]))
    attention_mask = batchify(torch.stack([torch.as_tensor(x["attention_mask"]) for x in batch]))

    if attn_implementation in ("flash_attention_2", "flash_attention_3", "flash_attention_4"):
        mask_out = attention_mask
    else:
        mask_out = _indexed_mask_to_4d_block_causal(attention_mask)

    result = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "attention_mask": mask_out,
    }
    if attention_mask.max() > 1:
        result["_packed_seq_ids"] = attention_mask
    return result


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
