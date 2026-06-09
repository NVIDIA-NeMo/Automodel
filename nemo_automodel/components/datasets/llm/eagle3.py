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

import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset

logger = logging.getLogger(__name__)


def _stack_batch(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack a batch of pre-padded unshifted chat samples."""
    batch = {}
    for key in ("input_ids", "loss_mask", "attention_mask"):
        batch[key] = torch.tensor([feature[key] for feature in features], dtype=torch.long)
    return batch


def build_packed_eagle3_dataset(
    source_dataset,
    *,
    packed_sequence_size: int,
    pad_token_id: int,
) -> list[dict[str, list[int]]]:
    """Greedily pack variable-length chat samples into rows of ``packed_sequence_size``.

    Each source sample is one *document*; documents are concatenated into a
    fixed-width row with ``position_ids`` reset per document and trailing pad
    folded into the final document (so ``seq_lens`` sums to the row width).

    Cross-document leakage at TTT boundaries is handled by ``doc_remaining[t]``
    (real tokens after slot ``t`` within its document): the trainer supervises
    slot ``t`` at step ``k`` to predict ``k+1`` ahead, valid iff
    ``k < doc_remaining[t]``. This masks every cross-document / into-padding
    supervision -- packing creates many such boundaries per row.

    Returns a list of packed-row dicts with keys ``input_ids``, ``loss_mask``,
    ``attention_mask``, ``position_ids``, ``doc_remaining`` (length
    ``packed_sequence_size``) and ``seq_lens`` (per-document padded lengths).
    """
    packs: list[dict[str, list[int]]] = []
    cur_ids: list[int] = []
    cur_loss: list[int] = []
    cur_pos: list[int] = []
    cur_remaining: list[int] = []
    cur_seq_lens: list[int] = []

    def _flush() -> None:
        nonlocal cur_ids, cur_loss, cur_pos, cur_remaining, cur_seq_lens
        if not cur_ids:
            return
        valid_len = len(cur_ids)
        num_pad = packed_sequence_size - valid_len
        ids = cur_ids + [pad_token_id] * num_pad
        loss = cur_loss + [0] * num_pad
        attn = [1] * valid_len + [0] * num_pad
        remaining = cur_remaining + [0] * num_pad
        # Fold trailing pad into the final document (continue its position ids).
        last_pos = cur_pos[-1] if cur_pos else -1
        pos = cur_pos + [min(last_pos + 1 + j, packed_sequence_size - 1) for j in range(num_pad)]
        seq_lens = list(cur_seq_lens)
        if num_pad > 0:
            seq_lens[-1] += num_pad
        packs.append(
            {
                "input_ids": ids,
                "loss_mask": loss,
                "attention_mask": attn,
                "position_ids": pos,
                "doc_remaining": remaining,
                "seq_lens": seq_lens,
            }
        )
        cur_ids, cur_loss, cur_pos, cur_remaining, cur_seq_lens = [], [], [], [], []

    for sample in source_dataset:
        ids = list(sample["input_ids"])
        loss = [int(bool(m)) for m in sample["loss_mask"]]
        length = len(ids)
        if length == 0:
            continue
        if length > packed_sequence_size:
            # Guard: the source dataset's truncation should already cap at the row width.
            ids = ids[:packed_sequence_size]
            loss = loss[:packed_sequence_size]
            length = packed_sequence_size
        if cur_ids and len(cur_ids) + length > packed_sequence_size:
            _flush()
        cur_ids += ids
        cur_loss += loss
        cur_pos += list(range(length))
        # Real tokens remaining after each slot within this document.
        cur_remaining += list(range(length - 1, -1, -1))
        cur_seq_lens.append(length)
    _flush()

    if not packs:
        raise ValueError(
            f"No packs were produced from the source dataset for packed_sequence_size="
            f"{packed_sequence_size}. The dataset may be empty."
        )
    return packs


def _pack_collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate packed rows; ragged ``seq_lens`` is 0-padded to ``[B, max_docs]``."""
    batch = {}
    for key in ("input_ids", "loss_mask", "attention_mask", "position_ids", "doc_remaining"):
        batch[key] = torch.tensor([feature[key] for feature in features], dtype=torch.long)
    max_docs = max(len(feature["seq_lens"]) for feature in features)
    seq_lens = torch.zeros(len(features), max_docs, dtype=torch.long)
    for i, feature in enumerate(features):
        row = feature["seq_lens"]
        seq_lens[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    batch["seq_lens"] = seq_lens
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
    mask_reasoning_content: bool = False,
    packed_sequence_size: int = 0,
) -> DataLoader:
    """Build a dataloader backed by the repo's chat formatting utilities.

    ``packed_sequence_size > 0`` (EAGLE-3 only) enables sequence packing (see
    :func:`build_packed_eagle3_dataset`), removing the padding waste of the
    default ``padding="max_length"`` path; ``== 0`` keeps the original behavior.
    """
    collate_fn = _stack_batch
    if packed_sequence_size > 0:
        # Source samples are unpadded (one document each); packing pads the row.
        source = ChatDataset(
            data_path,
            tokenizer=tokenizer,
            split=split,
            seq_length=packed_sequence_size,
            padding="do_not_pad",
            truncation=True,
            shuffle_seed=shuffle_seed,
            unshifted=True,
            mask_reasoning_content=mask_reasoning_content,
        )
        dataset = build_packed_eagle3_dataset(
            source,
            packed_sequence_size=packed_sequence_size,
            pad_token_id=source.pad_token_id,
        )
        collate_fn = _pack_collate
    else:
        dataset = ChatDataset(
            data_path,
            tokenizer=tokenizer,
            split=split,
            seq_length=seq_length,
            padding="max_length",
            truncation=True,
            shuffle_seed=shuffle_seed,
            unshifted=True,
            mask_reasoning_content=mask_reasoning_content,
        )
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None

    # The EAGLE recipes load the target model onto CUDA before iterating, so the
    # CUDA context is already initialized by the time these workers spawn. Two
    # defaults are unsafe in that state and must be overridden when
    # ``num_workers > 0``:
    #   * ``fork`` (the Linux default start method) hands each worker a copy of
    #     the parent's live CUDA context, which is invalid in the child and
    #     aborts the worker with ``cudaErrorInitializationError`` -> SIGABRT.
    #     ``forkserver`` starts workers from a clean process with no inherited
    #     CUDA context.
    #   * Without ``persistent_workers`` the pool is torn down and re-forked at
    #     every epoch boundary -- exactly when the abort surfaced (the first
    #     epoch ran fine). Keeping workers alive across epochs removes that
    #     re-fork entirely.
    worker_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        worker_kwargs["persistent_workers"] = True
        if torch.cuda.is_available():
            worker_kwargs["multiprocessing_context"] = "forkserver"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False,
        **worker_kwargs,
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
    # Validate sizes up front. ``target_vocab_size`` must be positive (it
    # sizes the dense count tensor + mask), and ``draft_vocab_size`` -- when
    # supplied -- must be a positive integer. Without these guards a caller
    # passing ``draft_vocab_size=0`` quietly gets an empty selection, and
    # ``draft_vocab_size=-1`` interacts with the ``selected[:draft_vocab_size]``
    # slice to drop the last special token instead of erroring -- both
    # silently miscompile downstream rather than failing fast.
    if not isinstance(target_vocab_size, int) or target_vocab_size <= 0:
        raise ValueError(
            f"build_eagle3_token_mapping requires target_vocab_size to be a "
            f"positive integer, got target_vocab_size={target_vocab_size!r}."
        )
    if draft_vocab_size is not None and (not isinstance(draft_vocab_size, int) or draft_vocab_size <= 0):
        raise ValueError(
            f"build_eagle3_token_mapping requires draft_vocab_size to be a "
            f"positive integer or None (= use the full target vocab), got "
            f"draft_vocab_size={draft_vocab_size!r}."
        )

    if draft_vocab_size is None or draft_vocab_size >= target_vocab_size:
        selected_token_ids = torch.arange(target_vocab_size, dtype=torch.long)
        selected_token_mask = torch.ones(target_vocab_size, dtype=torch.bool)
        return selected_token_ids, selected_token_mask

    distributed = dist.is_available() and dist.is_initialized()
    is_rank0 = (not distributed) or dist.get_rank() == 0
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None

    counts = torch.zeros(target_vocab_size, dtype=torch.long)
    total_supervised_tokens = 0
    # Drive the progress bar manually and iterate ``dataloader`` directly: wrapping
    # the loader in ``tqdm(dataloader)`` triggers a second ``__iter__`` on loaders
    # without ``__len__``, which would double-scan a single-pass / streaming loader.
    progress = tqdm(
        total=total_batches,
        desc=f"Counting supervised tokens for draft vocab ({draft_vocab_size})",
        unit="batch",
        disable=not is_rank0,
        leave=True,
    )
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].bool()
        supervised_ids = input_ids[loss_mask].to(torch.long).flatten()
        progress.update(1)
        if supervised_ids.numel() == 0:
            continue
        in_range = (supervised_ids >= 0) & (supervised_ids < target_vocab_size)
        supervised_ids = supervised_ids[in_range]
        total_supervised_tokens += supervised_ids.numel()
        if is_rank0 and step % 100 == 0:
            progress.set_postfix(tokens=total_supervised_tokens)
        counts.scatter_add_(0, supervised_ids, torch.ones_like(supervised_ids))
    progress.close()

    if distributed:
        # NCCL collectives require CUDA tensors; move counts onto the current
        # device for the reduction and bring it back to CPU for the Python-side
        # selection logic below.
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            counts_for_reduce = counts.to(torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(counts_for_reduce, op=dist.ReduceOp.SUM)
            counts = counts_for_reduce.cpu()
        else:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    total_frequency = int(counts.sum().item())
    unique_tokens = int((counts > 0).sum().item())
    if is_rank0:
        logger.info(
            "Counted %d supervised tokens across %d unique target tokens",
            total_frequency,
            unique_tokens,
        )

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
    if is_rank0:
        selected_frequency = int(counts[selected_token_ids].sum().item())
        ratio = selected_frequency / total_frequency if total_frequency > 0 else 0.0
        logger.info("top %d token frequency ratio: %.2f%%", selected_token_ids.numel(), ratio * 100)
    return selected_token_ids, selected_token_mask


def _expected_draft_vocab_size(target_vocab_size: int, draft_vocab_size: int | None) -> int:
    """Return how many ids ``build_eagle3_token_mapping`` yields for this config.

    Mirrors its selection branch: a ``None`` or too-large ``draft_vocab_size``
    falls back to the full target vocab.
    """
    if draft_vocab_size is None or draft_vocab_size >= target_vocab_size:
        return target_vocab_size
    return draft_vocab_size


def save_eagle3_token_mapping(path: str, selected_token_ids: torch.Tensor, target_vocab_size: int) -> None:
    """Persist the draft-vocab selection so future runs skip the frequency scan.

    Written atomically (``.tmp`` + ``os.replace``) so a crash mid-write never
    leaves a half-written file a later run would load. Only ``selected_token_ids``
    is stored -- ``selected_token_mask`` is fully derivable from it plus
    ``target_vocab_size``.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(
        {"selected_token_ids": selected_token_ids.detach().cpu(), "target_vocab_size": int(target_vocab_size)},
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_eagle3_token_mapping(
    path: str,
    *,
    target_vocab_size: int,
    draft_vocab_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load a cached draft-vocab mapping, or ``None`` if absent / incompatible.

    The cache is keyed only on ``target_vocab_size`` and the resulting draft
    vocab size -- it does NOT fingerprint the dataset or tokenizer. A cache built
    from a different dataset still loads cleanly, so a caller that changes the
    training data must point ``selected_token_ids_path`` at a fresh location (or
    delete the file). Returns ``None`` -- so the caller rebuilds -- when the file
    is missing, unreadable, or its stored vocab sizes do not match the config.
    """
    if not os.path.exists(path):
        return None
    try:
        # The payload is only ``{Tensor, int}``, so the safe loader suffices and
        # avoids pickle's arbitrary-code-execution vector even though this path
        # is user-controlled. Any load failure (legacy/foreign file) falls
        # through to the rebuild below.
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - corrupted / foreign file -> rebuild
        logger.warning("Failed to load EAGLE-3 token map from %s (%s); rebuilding.", path, exc)
        return None
    ids = data.get("selected_token_ids") if isinstance(data, dict) else None
    saved_target = data.get("target_vocab_size") if isinstance(data, dict) else None
    expected = _expected_draft_vocab_size(target_vocab_size, draft_vocab_size)
    if not isinstance(ids, torch.Tensor) or saved_target != target_vocab_size:
        logger.warning(
            "Cached EAGLE-3 token map at %s is incompatible (target_vocab_size %s != %s); rebuilding.",
            path,
            saved_target,
            target_vocab_size,
        )
        return None
    ids = ids.to(torch.long)
    if ids.numel() != expected:
        logger.warning(
            "Cached EAGLE-3 token map at %s has %d ids, expected %d for draft_vocab_size=%s; rebuilding.",
            path,
            ids.numel(),
            expected,
            draft_vocab_size,
        )
        return None
    mask = torch.zeros(target_vocab_size, dtype=torch.bool)
    mask[ids] = True
    return ids, mask


def _broadcast_cached_ids(
    cache_path: str,
    *,
    target_vocab_size: int,
    draft_vocab_size: int | None,
) -> torch.Tensor | None:
    """Rank 0 loads (and validates) the cached ids; broadcast the result to all ranks.

    Only rank 0 touches the filesystem, so the load-vs-build decision is identical
    on every rank even when ``cache_path`` lives on a node-local (non-shared)
    filesystem. This matters because ``build_eagle3_token_mapping`` issues a
    collective ``all_reduce``: if some ranks loaded a cache while others rebuilt,
    that collective would mismatch and hang. Returns the ids (cpu, long) or
    ``None`` (rebuild on every rank).
    """
    distributed = dist.is_available() and dist.is_initialized()
    is_rank_0 = (not distributed) or dist.get_rank() == 0
    loaded: torch.Tensor | None = None
    if is_rank_0:
        result = load_eagle3_token_mapping(
            cache_path, target_vocab_size=target_vocab_size, draft_vocab_size=draft_vocab_size
        )
        loaded = result[0] if result is not None else None
    if not distributed:
        return loaded

    use_cuda = dist.get_backend() == "nccl" and torch.cuda.is_available()
    device = torch.device("cuda", torch.cuda.current_device()) if use_cuda else torch.device("cpu")
    # Broadcast the id count first (0 == "no usable cache, rebuild"), then the payload.
    count = torch.tensor([loaded.numel() if loaded is not None else 0], dtype=torch.long, device=device)
    dist.broadcast(count, src=0)
    num_ids = int(count.item())
    if num_ids == 0:
        return None
    payload = loaded.to(device) if is_rank_0 else torch.empty(num_ids, dtype=torch.long, device=device)
    dist.broadcast(payload, src=0)
    return payload.cpu()


def load_or_build_eagle3_token_mapping(
    dataloader: DataLoader,
    *,
    target_vocab_size: int,
    draft_vocab_size: int | None,
    special_token_ids: list[int] | None = None,
    cache_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the draft-vocab mapping, reusing a cached copy at ``cache_path``.

    With ``cache_path`` set, present, and compatible, loads the mapping and skips
    the full-dataset frequency scan ``build_eagle3_token_mapping`` performs.
    Otherwise builds the mapping and -- on rank 0 -- writes it to ``cache_path``
    for next time. With ``cache_path=None`` this is exactly
    ``build_eagle3_token_mapping``.
    """
    if cache_path is not None:
        cached_ids = _broadcast_cached_ids(
            cache_path, target_vocab_size=target_vocab_size, draft_vocab_size=draft_vocab_size
        )
        if cached_ids is not None:
            mask = torch.zeros(target_vocab_size, dtype=torch.bool)
            mask[cached_ids] = True
            logger.info("Loaded EAGLE-3 draft vocab (%d ids) from cache %s", cached_ids.numel(), cache_path)
            return cached_ids, mask

    selected_token_ids, selected_token_mask = build_eagle3_token_mapping(
        dataloader,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        special_token_ids=special_token_ids,
    )
    if cache_path is not None and ((not dist.is_initialized()) or dist.get_rank() == 0):
        save_eagle3_token_mapping(cache_path, selected_token_ids, target_vocab_size)
        logger.info("Cached EAGLE-3 draft vocab (%d ids) to %s", selected_token_ids.numel(), cache_path)
    return selected_token_ids, selected_token_mask
