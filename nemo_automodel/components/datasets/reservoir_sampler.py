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

from random import randint, shuffle
from typing import Any, Dict, Iterator, Optional


class ReservoirSampler:
    """Reservoir sampler."""

    def __init__(self, iterator: Iterator[Dict[str, Any]], buffer_size: int, seed: Optional[int] = None):
        """
        Reservoir sampler is a sampler that samples items from an iterator using a buffer.
        It is used to sample items from an iterator in a way that is memory efficient.

        Args:
            iterator: Iterator to sample from.
            buffer_size: Size of the buffer.
            seed: Seed for the random number generator.
        """
        assert buffer_size > 0, "buffer_size must be > 0"
        assert iterator is not None, "iterator must be provided"
        self._buffer_size = buffer_size
        self._buffer = []
        self.iterator = iterator

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the iterator and sample items from the buffer.
        """
        for item in self.iterator:
            self._buffer.append(item)
            if len(self._buffer) == self._buffer_size:
                break
        shuffle(self._buffer)
        while True:
            new_pos = randint(0, len(self._buffer) - 1)
            evicted_item = self._buffer[new_pos]
            try:
                self._buffer[new_pos] = next(self.iterator)
            except StopIteration:
                yield evicted_item
                self._buffer[new_pos] = None
                break
            yield evicted_item

        # handle tail
        yield from filter(lambda x: x is not None, self._buffer)

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
