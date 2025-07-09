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

"""Runtime back-ports for old PyTorch versions. Will be deleted in future stable PyTorch versions."""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)


def apply_patches() -> None:
    """
    Inject modified modules into an *old* ``torch.distributed.checkpoint``.
    """
    # -----------------------------------------------------------------
    # Ensure SavePlanner provides the _cached_metadata class attribute.
    # This is required by NeMo-Automodel's extended planners but may be
    # missing from older PyTorch versions (< 2.4).  Monkey-patch it here
    # so downstream code can rely on its existence independent of the
    # installed torch release.
    # -----------------------------------------------------------------
    try:
        planner_mod = importlib.import_module("torch.distributed.checkpoint.planner")
        SavePlanner = getattr(planner_mod, "SavePlanner", None)
        if SavePlanner is not None and not hasattr(SavePlanner, "_cached_metadata"):
            # Forward-declare attribute; note we don't import Metadata to
            # avoid circular deps – a forward reference string in the
            # annotation keeps static checkers happy while remaining
            # runtime-safe.
            SavePlanner._cached_metadata = {}

            # Update type annotations dynamically for better type hints
            anns = getattr(SavePlanner, "__annotations__", {})
            anns.setdefault("_cached_metadata", "dict[str, 'Metadata']")
            SavePlanner.__annotations__ = anns  # type: ignore[attr-defined]

            logger.debug("Added missing SavePlanner._cached_metadata back-port")
    except ModuleNotFoundError:
        # planner module unavailable – nothing to patch
        pass
