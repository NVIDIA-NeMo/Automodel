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

import gc
import logging
import time

logger = logging.getLogger(__name__)


class GarbageCollection:
    """Utility for periodic manual garbage collection during training."""

    # Reference:
    # https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py#L46
    def __init__(self, gc_every_steps: int = 1000):
        assert gc_every_steps > 0, "gc_every_steps must be a positive integer"
        self.gc_every_steps = gc_every_steps
        gc.disable()
        self.collect("Initial GC collection")

    def run(self, step_count: int) -> None:
        """Run periodic garbage collection based on step count."""
        if step_count > 1 and step_count % self.gc_every_steps == 0:
            self.collect("Performing periodic GC collection")

    @staticmethod
    def collect(reason: str, generation: int = 1) -> None:
        """Collect garbage and emit timing logs."""
        begin = time.monotonic()
        gc.collect(generation)
        logger.info("[GC] %s took %.2f seconds", reason, time.monotonic() - begin)
