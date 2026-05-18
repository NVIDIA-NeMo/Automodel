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

"""Data helpers for minimal EAGLE-3 training."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset


def _stack_batch(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack a batch of pre-padded unshifted chat samples."""
    batch = {}
    for key in ("input_ids", "loss_mask", "attention_mask"):
        batch[key] = torch.tensor([feature[key] for feature in features], dtype=torch.long)
    return batch


def build_eagle3_dataloader(
    *,
    data_path: str,
    tokenizer,
    seq_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    split: str | None = None,
    distributed: bool = False,
    shuffle_seed: int | None = 42,
) -> DataLoader:
    """Build a dataloader backed by the repo's chat formatting utilities."""
    dataset = ChatDataset(
        data_path,
        tokenizer=tokenizer,
        split=split,
        seq_length=seq_length,
        padding="max_length",
        truncation=True,
        shuffle_seed=shuffle_seed,
        unshifted=True,
    )
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_stack_batch,
        drop_last=False,
    )


def build_eagle3_token_mapping(
    dataloader: DataLoader,
    *,
    target_vocab_size: int,
    draft_vocab_size: int | None,
    special_token_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build draft-vocab mapping tensors from supervised token frequency.

    Counts are accumulated as a dense ``[target_vocab_size]`` tensor and
    ``all_reduce`` summed across ranks when ``torch.distributed`` is
    initialized, so every rank ends up with the same draft vocabulary.

    Returns:
        Tuple ``(selected_token_ids, selected_token_mask)`` where:
        - ``selected_token_ids`` has shape ``[draft_vocab_size]``
        - ``selected_token_mask`` has shape ``[target_vocab_size]``
    """
    if draft_vocab_size is None or draft_vocab_size >= target_vocab_size:
        selected_token_ids = torch.arange(target_vocab_size, dtype=torch.long)
        selected_token_mask = torch.ones(target_vocab_size, dtype=torch.bool)
        return selected_token_ids, selected_token_mask

    counts = torch.zeros(target_vocab_size, dtype=torch.long)
    for batch in dataloader:
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].bool()
        supervised_ids = input_ids[loss_mask].to(torch.long).flatten()
        if supervised_ids.numel() == 0:
            continue
        in_range = (supervised_ids >= 0) & (supervised_ids < target_vocab_size)
        supervised_ids = supervised_ids[in_range]
        counts.scatter_add_(0, supervised_ids, torch.ones_like(supervised_ids))

    if dist.is_available() and dist.is_initialized():
        # NCCL collectives require CUDA tensors; move counts onto the current
        # device for the reduction and bring it back to CPU for the Python-side
        # selection logic below.
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            counts_for_reduce = counts.to(torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(counts_for_reduce, op=dist.ReduceOp.SUM)
            counts = counts_for_reduce.cpu()
        else:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    selected: list[int] = []
    seen: set[int] = set()
    for token_id in special_token_ids or []:
        if token_id is None or token_id < 0 or token_id >= target_vocab_size or token_id in seen:
            continue
        selected.append(int(token_id))
        seen.add(int(token_id))

    sorted_token_ids = torch.argsort(counts, descending=True, stable=True).tolist()
    for token_id in sorted_token_ids:
        if len(selected) >= draft_vocab_size:
            break
        if token_id in seen or counts[token_id].item() == 0:
            continue
        selected.append(token_id)
        seen.add(token_id)

    for token_id in range(target_vocab_size):
        if len(selected) >= draft_vocab_size:
            break
        if token_id not in seen:
            selected.append(token_id)
            seen.add(token_id)

    selected_token_ids = torch.tensor(selected[:draft_vocab_size], dtype=torch.long)
    selected_token_mask = torch.zeros(target_vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True
    return selected_token_ids, selected_token_mask
