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

"""Runtime back-ports for old PyTorch versions.

This helper rewires selected functions/classes inside
``torch.distributed.checkpoint`` to newer implementations bundled with
NeMo-Automodel.  It must be imported *before* you construct any
``SavePlanner``/``LoadPlanner``/etc.  The project's ``__init__.py
automatically calls :pyfunc:`apply_patches` on import, so users usually
don't have to do anything manually.
"""
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Helper – make patching idempotent.
# ---------------------------------------------------------------------------
_SENTINEL = "__nemo_backport__"

def _already_patched(obj) -> bool:
    return getattr(obj, _SENTINEL, False)

def _mark_patched(obj) -> None:
    setattr(obj, _SENTINEL, True)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def apply_patches() -> None:
    """Inject newer helpers into an *old* ``torch.distributed.checkpoint``.

    Safe to call multiple times – only the first invocation does work.
    Subsequent calls exit early.
    """

    try:
        torch_fs_mod = importlib.import_module("torch.distributed.checkpoint.filesystem")
        torch_dp_mod = importlib.import_module("torch.distributed.checkpoint.default_planner")
    except ModuleNotFoundError as exc:  # pragma: no cover – exotic env
        logger.warning("torch.distributed.checkpoint not available – skipping back-ports (%s)", exc)
        return

    # Import NeMo replacement implementations lazily (avoids circular deps)
    nemo_fs_mod = importlib.import_module("nemo_automodel.checkpoint.filesystem")
    nemo_dp_mod = importlib.import_module("nemo_automodel.checkpoint.default_planner")

    # ---------------------------------------------------------------------
    # filesystem._write_files_from_queue
    # ---------------------------------------------------------------------
    if not _already_patched(torch_fs_mod._write_files_from_queue):
        torch_fs_mod._write_files_from_queue = nemo_fs_mod._write_files_from_queue
        _mark_patched(torch_fs_mod._write_files_from_queue)
        logger.debug("Patched torch.distributed.checkpoint.filesystem._write_files_from_queue")

    # ---------------------------------------------------------------------
    # default_planner.create_default_local_load_plan
    # ---------------------------------------------------------------------
    if not _already_patched(torch_dp_mod.create_default_local_load_plan):
        torch_dp_mod.create_default_local_load_plan = nemo_dp_mod.create_default_local_load_plan
        _mark_patched(torch_dp_mod.create_default_local_load_plan)
        logger.debug("Patched default_planner.create_default_local_load_plan")

    # ---------------------------------------------------------------------
    # _EmptyStateDictLoadPlanner.set_up_planner
    # ---------------------------------------------------------------------
    if hasattr(torch_dp_mod, "_EmptyStateDictLoadPlanner") and hasattr(nemo_dp_mod, "_EmptyStateDictLoadPlanner"):
        torch_cls = torch_dp_mod._EmptyStateDictLoadPlanner
        nemo_cls = nemo_dp_mod._EmptyStateDictLoadPlanner
        if not _already_patched(torch_cls.set_up_planner):
            torch_cls.set_up_planner = nemo_cls.set_up_planner
            _mark_patched(torch_cls.set_up_planner)
            logger.debug("Patched _EmptyStateDictLoadPlanner.set_up_planner")

    # Also alias our extended SerializationFormat so identity checks succeed
    if not hasattr(torch_fs_mod, "_nemo_enum_aliased"):
        torch_fs_mod.SerializationFormat = nemo_fs_mod.SerializationFormat
        torch_fs_mod._nemo_enum_aliased = True

    logger.info("NeMo-Automodel: torch.distributed.checkpoint back-ports applied") 