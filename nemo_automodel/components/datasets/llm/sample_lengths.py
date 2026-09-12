# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Per-sample token counts, shared by the length-aware LLM samplers."""

from __future__ import annotations

import logging
import time

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def compute_sample_lengths(dataset: Dataset) -> list[int]:
    """Return the ``input_ids`` token count of every sample in ``dataset``.

    Samples without ``input_ids`` count as zero length so a malformed record sorts to
    the cheap end instead of raising in the middle of a long scan.

    Args:
        dataset: Map-style dataset whose samples are mappings carrying ``input_ids``.

    Returns:
        Token counts indexed by dataset index.
    """
    # Fast path: read the underlying list directly when the dataset just wraps one,
    # which skips per-item __getitem__ overhead on large datasets.
    n = len(dataset)
    raw = dataset
    while hasattr(raw, "dataset"):
        raw = raw.dataset
    # Only trust the unwrapped list when it lines up one-to-one with the outer dataset:
    # a wrapper that remaps indices (Subset, filtered or shuffled views) shares no index
    # space with its base, and reading through it would attribute lengths to the wrong
    # samples. That merely reordered batches before; it now sets the memory ceiling.
    if not isinstance(raw, list) or len(raw) != n:
        raw = None
    logger.info("Computing token lengths for %d samples...", n)
    start = time.monotonic()
    lengths = [0] * n

    for i in range(n):
        sample = raw[i] if raw is not None else dataset[i]
        ids = sample.get("input_ids")
        if ids is not None:
            lengths[i] = len(ids) if isinstance(ids, list) else ids.numel()
        if (i + 1) % 100_000 == 0 or i == n - 1:
            elapsed = time.monotonic() - start
            logger.info("  %d/%d samples (%.1fs, %.0f samples/s)", i + 1, n, elapsed, (i + 1) / max(elapsed, 1e-6))

    logger.info("Length computation done in %.1fs", time.monotonic() - start)
    return lengths


__all__ = ["compute_sample_lengths"]
