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
"""Behavioral check that ``tests/unit_tests/conftest.py`` undoes a test's multiprocessing start method.

The inner pytest run is a subprocess so that it starts, like a pytest-xdist worker, from a process
that has not chosen a start method yet.
"""

import os
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

# The inner pytest subprocess imports torch from scratch.
pytestmark = [pytest.mark.timeout(70), pytest.mark.cpu_only]

_CONFTEST_SOURCE = Path(__file__).with_name("conftest.py").read_text()

_START_METHOD_CHANGE = """
import multiprocessing


def test_switches_the_process_to_spawn() -> None:
    multiprocessing.set_start_method("spawn", force=True)


def test_next_test_finds_no_start_method_chosen() -> None:
    assert multiprocessing.get_start_method(allow_none=True) is None
    # Like the first worker started from the default context, this fixes the platform default.
    multiprocessing.get_start_method()


def test_fixed_default_does_not_reach_the_next_test_either() -> None:
    assert multiprocessing.get_start_method(allow_none=True) is None
"""


@pytest.mark.runtime_budget(
    30,
    hard_timeout=70,
    reason="starts a fresh pytest subprocess that imports torch",
)
def test_start_method_set_by_one_test_does_not_reach_the_next(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pytester relocates HOME for inner runs, which hides user-site installs such as torch from a fresh
    # interpreter, so hand the subprocess the outer interpreter's import path.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path for path in sys.path if path))
    # The generated run checks the conftest fixture, not the installed plugin set.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeconftest(_CONFTEST_SOURCE)
    pytester.makepyfile(test_start_method=_START_METHOD_CHANGE)

    result = pytester.runpytest_subprocess("-p", "pytest_timeout", "-p", "no:cacheprovider", "--tb=short", timeout=40)

    result.assert_outcomes(passed=3)
