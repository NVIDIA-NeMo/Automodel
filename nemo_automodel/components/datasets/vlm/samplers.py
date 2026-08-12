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
import time

import torch
from torch.utils.data import Sampler

from nemo_automodel.components.datasets.vlm.media_token_estimation import (
    DEFAULT_TOKENS_PER_MEDIA_ITEM,
    MediaTokenEstimator,
)

logger = logging.getLogger(__name__)


class LengthGroupedSampler(Sampler):
    """Sampler that groups samples by total token count for balanced
    distributed training.

    With ``shard_data=True`` each rank owns a different subset of data.
    This sampler sorts every rank's indices by **total tokens**
    (``text_tokens + media_tokens``, descending).  All ranks share the
    same ``seed + epoch`` so position *N* on every rank corresponds to a
    sample of similar length, keeping cross-rank padding minimal.

    Per-epoch randomness is achieved by rotating the sorted order by a
    deterministic random offset (same on every rank).

    Args:
        dataset: The dataset to sample from.
        seed: Base random seed (same value on every rank).
        processor: Optional HuggingFace processor (e.g. ``Qwen2VLProcessor``).
            Used by :class:`MediaTokenEstimator` for accurate media token
            estimation.
    """

    def __init__(self, dataset, seed=42, processor=None, max_length=None, batch_size=1):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.max_length = max_length
        self.batch_size = max(1, batch_size)

        # Resolves media token counts from the processor (see media_token_estimation)
        self._media_estimator = MediaTokenEstimator(processor)

        # Compute text and media tokens separately for two-level sorting
        self.text_lengths, self.media_lengths = self._compute_or_load_lengths(dataset)

        # Total token count per sample (text + media) — used for sorting
        self.lengths = [t + m for t, m in zip(self.text_lengths, self.media_lengths)]

        # Pre-filter overlong samples using estimated lengths.
        # PreTokenizedDatasetWrapper still acts as a safety-net retry for the
        # rare case where the heuristic underestimates the true tokenized length.
        all_indices = range(len(dataset))
        if max_length is not None:
            # Use 1.2x headroom: the estimated length is a heuristic, so only
            # drop samples that are clearly overlong.  Borderline samples are
            # left to PreTokenizedDatasetWrapper's precise tokenize-and-retry.
            filter_threshold = max_length + 512
            kept = [i for i in all_indices if self.lengths[i] <= filter_threshold]
            n_dropped = len(dataset) - len(kept)
            if n_dropped:
                logger.info(
                    "LengthGroupedSampler: pre-filtered %d/%d samples with "
                    "estimated length > %.0f tokens (1.2 * max_length %d).",
                    n_dropped,
                    len(dataset),
                    filter_threshold,
                    max_length,
                )
        else:
            kept = list(all_indices)

        # Sort by total tokens (descending), then shuffle within small
        # buckets each epoch to add randomness while preserving grouping.
        self.sorted_indices = sorted(
            kept,
            key=lambda i: self.lengths[i],
            reverse=True,
        )

        # Cross-rank count alignment: with shard_data=True each rank owns a
        # different slice of the corpus, so different ranks may filter out
        # different numbers of overlong samples.  An imbalanced sampler length
        # causes distributed deadlock.  All-reduce MIN so every rank uses the
        # same number of steps (we drop the tail — the shortest samples).
        if max_length is not None:
            import torch.distributed as dist

            if dist.is_initialized():
                count = torch.tensor(len(self.sorted_indices), dtype=torch.long).cuda()
                dist.all_reduce(count, op=dist.ReduceOp.MIN)
                min_count = count.item()
                if min_count < len(self.sorted_indices):
                    logger.info(
                        "LengthGroupedSampler: truncating from %d to %d samples "
                        "to align with the rank that filtered the most.",
                        len(self.sorted_indices),
                        min_count,
                    )
                    self.sorted_indices = self.sorted_indices[:min_count]

    # ------------------------------------------------------------------
    # Fast length computation with disk caching
    # ------------------------------------------------------------------

    @staticmethod
    def _get_raw_samples(dataset):
        """Unwrap dataset wrappers to get the underlying list for direct access."""
        raw = dataset
        while hasattr(raw, "dataset"):
            raw = raw.dataset
        if isinstance(raw, list):
            return raw
        return None

    def _compute_or_load_lengths(self, dataset):
        """Compute token lengths with direct list access for speed."""
        # Access underlying list directly, bypassing wrapper __getitem__ overhead
        raw_samples = self._get_raw_samples(dataset)
        if raw_samples is None:
            raw_samples = [dataset[i] for i in range(len(dataset))]

        n = len(raw_samples)

        # Compute lengths with progress logging
        logger.info("Estimating token lengths for %d samples...", n)
        t0 = time.monotonic()
        text_lengths = [0] * n
        media_lengths = [0] * n

        for i, example in enumerate(raw_samples):
            text_lengths[i], media_lengths[i] = self._estimate_tokens(example)
            if (i + 1) % 100_000 == 0 or i == n - 1:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                logger.info(
                    "  %d/%d samples (%.1fs elapsed, %.0f samples/s)",
                    i + 1,
                    n,
                    elapsed,
                    rate,
                )

        elapsed = time.monotonic() - t0
        logger.info("Token length estimation done in %.1fs (%.0f samples/s)", elapsed, n / max(elapsed, 1e-6))

        return text_lengths, media_lengths

    # ------------------------------------------------------------------
    # Length estimation
    # ------------------------------------------------------------------

    def _estimate_tokens(self, example):
        """Return ``(text_tokens, media_tokens)`` for one example.

        Uses pre-computed ``_text_tokens`` / ``_media_tokens`` when available
        (written by ``scripts/precompute_tokens.py``).  Otherwise falls back
        to heuristic estimation.
        """
        # --- text tokens ---
        precomputed_text = example.get("_text_tokens")
        if precomputed_text is not None:
            text_tokens = int(precomputed_text)
        else:
            # Fallback: heuristic ~1 token per 3 chars
            total_chars = 0
            for msg in example.get("conversation", []):
                content = msg.get("content", [])
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            total_chars += len(item.get("text", ""))
            text_tokens = total_chars // 3

        # --- media tokens (always computed from config, never cached) ---
        media_count = 0
        for msg in example.get("conversation", []):
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") in ("image", "video"):
                        media_count += 1

        mm_meta = example.get("mm_inputs_meta")
        if mm_meta is not None and self._media_estimator.can_estimate:
            media_tokens = self._media_estimator.estimate_media_tokens(
                images_meta=mm_meta.get("images_meta"),
                videos_meta=mm_meta.get("videos_meta"),
            )
        else:
            media_tokens = media_count * DEFAULT_TOKENS_PER_MEDIA_ITEM

        return text_tokens, media_tokens

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def set_epoch(self, epoch):
        """Set the epoch for deterministic shuffling (standard PyTorch pattern)."""
        self.epoch = epoch

    def __iter__(self):
        # Deterministic generator seeded identically on every rank.
        # All ranks share the same seed + epoch → same chunk permutation →
        # chunk K on every rank corresponds to similar total tokens
        # (because each rank's sorted_indices is ordered by length and
        # chunks are contiguous slices of that sorted order).
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Chunk sorted_indices into groups of batch_size so that samples
        # within a chunk have similar lengths (they are adjacent in the
        # sorted order).  Then shuffle at the chunk level to add
        # per-epoch randomness while preserving intra-batch length
        # similarity and cross-rank alignment.
        bs = self.batch_size
        chunks = [self.sorted_indices[i : i + bs] for i in range(0, len(self.sorted_indices), bs)]
        chunk_perm = torch.randperm(len(chunks), generator=g)
        indices = []
        for ci in chunk_perm:
            indices.extend(chunks[ci])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)
