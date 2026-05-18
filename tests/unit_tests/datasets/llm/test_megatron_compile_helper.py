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

"""Tests for the Megatron C++ helpers compile flow.

The Makefile in `nemo_automodel/components/datasets/llm/megatron/` originally
hard-coded ``uv run python`` for `pybind11` header lookup, which breaks on any
environment activated through ``source .venv/bin/activate`` or ``pip install``
(both officially supported per CONTRIBUTING.md). This test verifies that
:func:`compile_helper` invokes ``make`` with the active interpreter via
``PYTHON=sys.executable``, so the build picks up the right pybind11 headers
regardless of how the user entered the environment.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

from nemo_automodel.components.datasets.llm.megatron.megatron_utils import compile_helper


def test_compile_helper_passes_active_python_to_make():
    """`compile_helper` must thread `PYTHON=sys.executable` into the make call.

    Without this, the bundled Makefile cannot find pybind11 headers when the
    environment was not activated via ``uv run``.
    """
    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value.returncode = 0
        compile_helper()

    mock_run.assert_called_once()
    args, _ = mock_run.call_args
    cmd = args[0]
    # First arg must be `make`, last arg must carry the active Python.
    assert cmd[0] == "make"
    assert f"PYTHON={sys.executable}" in cmd, f"Expected `PYTHON={sys.executable}` in make argv, got {cmd!r}"
