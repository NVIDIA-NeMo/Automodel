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

"""Neat packing (greedy knapsack) for LLM datasets.

This module provides an alternative packing strategy that uses a greedy
knapsack algorithm (min-heap) for better bin-packing utilization compared
to the sequential first-fit approach in ``packed_sequence.py``.

The packed output uses an **indexed attention mask** (1, 2, 3, ... per
sub-sequence, 0 for padding) and **reset position IDs** so that it works
with eager / SDPA attention backends — no Transformer Engine dependency.
"""

import heapq
import logging
import time

import torch
from datasets import Dataset

logger = logging.getLogger(__name__)

CROSS_ENTROPY_IGNORE_IDX = -100


def greedy_knapsack(lengths: list[int], max_length: int) -> list[list[int]]:
    """Bin-pack sample indices using a greedy knapsack (min-heap) algorithm.

    Samples are sorted by length in descending order.  Each sample is
    assigned to the bin with the smallest current total that can still
    accommodate it.  If no bin fits, a new bin is created.

    Args:
        lengths: Length of each sample.
        max_length: Maximum capacity of each bin.

    Returns:
        A list of bins, where each bin is a list of sample indices.
    """
    N = len(lengths)
    # Sort indices by descending length for better packing
    sorted_indices = sorted(range(N), key=lambda i: -lengths[i])

    # Min-heap of (current_fill, bin_index)
    heap: list[tuple[int, int]] = []
    bins: list[list[int]] = []

    log_interval = max(1, N // 10)
    t0 = time.perf_counter()

    for count, idx in enumerate(sorted_indices):
        length = lengths[idx]
        if length > max_length:
            continue  # handled by caller (drop or raise)

        # Min-heap: the least-filled bin is at the top.
        # If even that bin can't fit the sample, no bin can — open a new one.
        if heap and heap[0][0] + length <= max_length:
            fill, bin_idx = heapq.heappop(heap)
            bins[bin_idx].append(idx)
            heapq.heappush(heap, (fill + length, bin_idx))
        else:
            new_bin_idx = len(bins)
            bins.append([idx])
            heapq.heappush(heap, (length, new_bin_idx))

        if (count + 1) % log_interval == 0:
            elapsed = time.perf_counter() - t0
            logger.info(
                "  Greedy knapsack: %d/%d (%.0f%%) | %d bins | %.1fs",
                count + 1, N, 100.0 * (count + 1) / N, len(bins), elapsed,
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        "  Greedy knapsack: done %d samples -> %d bins in %.1fs",
        N, len(bins), elapsed,
    )

    return bins


def _build_packed_sample(
    samples: list[dict],
    pack_size: int,
    padding_idx: int,
) -> dict:
    """Concatenate multiple samples into a single packed sample.

    Args:
        samples: List of sample dicts, each with ``input_ids`` and ``labels``
            (already autoregressive-shifted by the dataset).
        pack_size: Target packed sequence length (pad to this).
        padding_idx: Token ID used for padding ``input_ids``.

    Returns:
        Dict with ``input_ids``, ``labels``, ``attention_mask``,
        ``position_ids`` — all tensors of shape ``[pack_size]``.
    """
    all_input_ids: list[int] = []
    all_labels: list[int] = []
    all_attention_mask: list[int] = []
    all_position_ids: list[int] = []

    for seq_idx, sample in enumerate(samples, start=1):
        ids = sample["input_ids"]
        labs = sample["labels"]
        # Ensure lists
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(labs, torch.Tensor):
            labs = labs.tolist()

        seq_len = len(ids)
        all_input_ids.extend(ids)
        all_labels.extend(labs)
        all_attention_mask.extend([seq_idx] * seq_len)
        all_position_ids.extend(range(seq_len))

    # Pad to pack_size
    current_len = len(all_input_ids)
    pad_len = pack_size - current_len

    if pad_len > 0:
        all_input_ids.extend([padding_idx] * pad_len)
        all_labels.extend([CROSS_ENTROPY_IGNORE_IDX] * pad_len)
        all_attention_mask.extend([0] * pad_len)
        all_position_ids.extend([0] * pad_len)

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "attention_mask": torch.tensor(all_attention_mask, dtype=torch.long),
        "position_ids": torch.tensor(all_position_ids, dtype=torch.long),
    }


def neat_pack_dataset(
    dataset,
    split: str,
    pack_size: int,
    max_packs: int | None = None,
    padding_idx: int = 0,
    drop_long_samples: bool = False,
) -> Dataset:
    """Pack a dataset using greedy knapsack for better utilization.

    Args:
        dataset: HuggingFace dataset or dataset dict.
        split: Dataset split key (e.g. ``"train"``).
        pack_size: Target packed sequence length.
        max_packs: Optional cap on number of packs to create.
        padding_idx: Token ID for padding.
        drop_long_samples: If ``True``, silently drop samples longer than
            ``pack_size``; otherwise raise ``ValueError``.

    Returns:
        A HuggingFace ``Dataset`` of packed samples.
    """
    try:
        dataset = dataset[split]
    except Exception:
        logger.warning("Dataset split '%s' not found. Using entire dataset.", split)

    # Collect all samples and their lengths
    samples: list[dict] = []
    lengths: list[int] = []
    n_dropped = 0

    for sample in dataset:
        ids = sample["input_ids"]
        seq_len = len(ids) if isinstance(ids, list) else ids.shape[0]

        if seq_len > pack_size:
            if drop_long_samples:
                n_dropped += 1
                continue
            raise ValueError(
                f"Sample is too long ({seq_len} > {pack_size}). "
                "Set drop_long_samples=True or increase pack_size."
            )

        # Apply loss_mask to labels if present
        if "loss_mask" in sample:
            labels = sample["labels"]
            loss_mask = sample["loss_mask"]
            if isinstance(labels, list):
                labels = [
                    lab if mask else CROSS_ENTROPY_IGNORE_IDX
                    for lab, mask in zip(labels, loss_mask)
                ]
                sample["labels"] = labels
            else:
                labels = labels.clone()
                labels[torch.tensor(loss_mask) == 0] = CROSS_ENTROPY_IGNORE_IDX
                sample["labels"] = labels

        samples.append(sample)
        lengths.append(seq_len)

    if n_dropped > 0:
        logger.info("Dropped %d samples exceeding pack_size=%d.", n_dropped, pack_size)

    if not samples:
        raise ValueError("No samples remaining after filtering.")

    # Run greedy knapsack
    bins = greedy_knapsack(lengths, pack_size)
    logger.info(
        "Greedy knapsack: %d samples -> %d packs (avg utilization: %.1f%%)",
        len(samples),
        len(bins),
        100.0 * sum(lengths) / (len(bins) * pack_size) if bins else 0,
    )

    # Build packed samples
    packs: list[dict] = []
    for bin_indices in bins:
        if max_packs is not None and len(packs) >= max_packs:
            break
        bin_samples = [samples[i] for i in bin_indices]
        packed = _build_packed_sample(bin_samples, pack_size, padding_idx)
        packs.append(packed)

    logger.info("Total number of packs created: %d", len(packs))

    return Dataset.from_dict(
        {key: [pack[key] for pack in packs] for key in packs[0].keys()}
    )
