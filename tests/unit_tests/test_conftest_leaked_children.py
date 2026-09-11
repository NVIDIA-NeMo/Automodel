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
"""Behavioral check for the leaked child-process cleanup in ``tests/unit_tests/conftest.py``.

The inner pytest run is a subprocess because the property under test is that the pytest
process itself exits promptly after a test times out while its ``mp.spawn`` workers hang.
"""

import os
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

# The inner pytest subprocess imports torch from scratch and needs a few seconds beyond the 5s fallback.
pytestmark = pytest.mark.timeout(60)

_CONFTEST_SOURCE = Path(__file__).with_name("conftest.py").read_text()

_HANGING_WORKER = """
import time

import pytest
import torch.multiprocessing as mp


def _hang(rank):
    time.sleep(120)


@pytest.mark.timeout(1)
def test_hanging_worker():
    mp.spawn(_hang, nprocs=1, join=True)
"""


def test_pytest_exits_after_timed_out_spawn(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    # pytester relocates HOME for inner runs, which hides user-site installs such as torch from a fresh
    # interpreter, so hand the subprocess the outer interpreter's import path.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path for path in sys.path if path))
    pytester.makeconftest(_CONFTEST_SOURCE)
    pytester.makepyfile(test_hang=_HANGING_WORKER)
    # Without the cleanup, interpreter shutdown blocks on the sleeping worker and this raises TimeoutExpired.
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", timeout=40)
    result.assert_outcomes(failed=1, errors=1)
    result.stdout.fnmatch_lines(["*Timeout (>1.0s) from pytest-timeout*"])
    result.stdout.fnmatch_lines(["*left 1 child process(es) running*"])
