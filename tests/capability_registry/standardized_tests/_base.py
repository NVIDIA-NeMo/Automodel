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

"""Base types for capability validation tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable


@dataclass
class CapabilityTestResult:
    """Outcome of a single capability validation run.

    Attributes:
        capability: Capability name (``"tp"``, ``"cp"``, ...).
        passed: True when ``max_kl <= threshold``. ``True`` for skipped results
            (a skip is not a failure).
        skipped: True when no real validation ran (model unsupported, GPU
            insufficient, capability not yet implemented).
        max_kl: Max per-token KL divergence; ``None`` if skipped or errored.
        threshold: KL threshold the run was checked against.
        variant_label: Short label describing the variant configuration
            (e.g. ``"TP=2"``, ``"CP=2"``, ``"not implemented"``).
        error: Optional human-readable error or skip-reason string.
    """

    capability: str
    passed: bool
    skipped: bool
    max_kl: float | None
    threshold: float
    variant_label: str
    error: str | None = None

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict (used for parent <-> child IPC)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CapabilityTestResult":
        """Inverse of :meth:`to_dict`."""
        return cls(**data)


@runtime_checkable
class CapabilityTest(Protocol):
    """Contract every standardized capability test must obey.

    Attributes:
        name: Capability identifier matching the ``supports_<name>`` flag.
        implemented: ``False`` for SKIP-stubs (PP/EP in v1).
        world_size: Number of GPUs required by this test.
    """

    name: str
    implemented: bool
    world_size: int

    def run(
        self,
        *,
        model_id: str,
        dtype,  # noqa: ANN001 - torch.dtype, kept loose to avoid heavy import here
        kl_threshold: float,
        num_steps: int,
        local_batch_size: int,
    ) -> CapabilityTestResult:
        """Execute the validation and return a result."""
        ...
