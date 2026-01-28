# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Iterator, Optional


class ReservoirSampler:
    """Streaming shuffle with a fixed-size buffer.

    This is a bounded-memory shuffling wrapper for streaming datasets/iterables.
    It maintains a buffer of ``buffer_size`` items. Once the buffer is filled,
    it repeatedly:

    - samples a random buffer slot
    - yields the evicted item
    - replaces it with the next item from the underlying iterator

    When the underlying iterator is exhausted, the remaining buffer items are
    yielded.
    """

    def __init__(self, iterator: Iterable[Dict[str, Any]], buffer_size: int, seed: Optional[int] = None):
        """
        Reservoir sampler is a sampler that samples items from an iterator using a buffer.
        It is used to sample items from an iterator in a way that is memory efficient.

        Args:
            iterator: Iterator to sample from.
            buffer_size: Size of the buffer.
            seed: Seed for the random number generator.
        """
        if iterator is None:
            raise ValueError("iterator must be provided")
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be > 0, got {buffer_size}")

        self._buffer_size = int(buffer_size)
        self._seed = seed
        self._iterable = iterator

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the iterator and sample items from the buffer.
        """
        rng = random.Random(self._seed)
        it = iter(self._iterable)

        buffer: list[Optional[Dict[str, Any]]] = []
        for item in it:
            buffer.append(item)
            if len(buffer) == self._buffer_size:
                break

        if not buffer:
            return

        rng.shuffle(buffer)
        while True:
            new_pos = rng.randrange(len(buffer))
            evicted_item = buffer[new_pos]
            try:
                buffer[new_pos] = next(it)
            except StopIteration:
                yield evicted_item
                buffer[new_pos] = None
                break
            else:
                yield evicted_item

        # handle tail
        yield from filter(lambda x: x is not None, buffer)

    def __len__(self) -> int:
        """
        No len methods is supported with ReservoirSampler.
        """
        raise RuntimeError("__len__ is not supported with ReservoirSampler.")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        No getitem method is supported with ReservoirSampler.
        """
        raise RuntimeError("__getitem__ is not supported with ReservoirSampler.")
