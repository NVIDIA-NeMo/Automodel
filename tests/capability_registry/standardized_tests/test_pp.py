# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Pipeline-parallelism validation stub (not implemented in v1).

When implemented, this should:
  - Build a PP mesh with ``pp_size`` stages.
  - Train K-1 steps under :class:`AutoPipeline` with a ``1f1b`` (or
    ``interleaved1f1b``) schedule.
  - Capture rank-PP-last logits and reduce against the reference (no-PP) run.
"""

from __future__ import annotations

import torch

from tests.capability_registry.standardized_tests._base import CapabilityTestResult


class PPTest:
    """SKIP stub for pipeline parallelism validation."""

    name: str = "pp"
    implemented: bool = False
    world_size: int = 2

    def run(
        self,
        *,
        model_id: str,
        dtype: torch.dtype,
        kl_threshold: float,
        num_steps: int,
        local_batch_size: int,
    ) -> CapabilityTestResult:
        """Return a skipped result; real implementation is future work."""
        return CapabilityTestResult(
            capability=self.name,
            passed=True,
            skipped=True,
            max_kl=None,
            threshold=kl_threshold,
            variant_label="not implemented",
            error="PP validation harness not implemented in v1",
        )
