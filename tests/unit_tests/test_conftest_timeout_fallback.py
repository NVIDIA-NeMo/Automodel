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
"""Behavioral checks for the 5s fallback timeout installed by ``tests/unit_tests/conftest.py``.

Each test copies the real conftest into an isolated pytester project, so the fallback's
directory scoping resolves to the temporary tree, then runs pytest in-process and asserts
on the observable outcome: which timeout, if any, actually fires.
"""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

# Inner runs must be able to sleep past the 5s fallback without the outer timer firing inside them.
pytestmark = pytest.mark.timeout(60)

_CONFTEST_SOURCE = Path(__file__).with_name("conftest.py").read_text()

_SLEEPER = """
import time


def test_sleep():
    time.sleep({seconds})
"""


def _run_sleeper(pytester: pytest.Pytester, seconds: float, *args: str, module_marker: str = "") -> pytest.RunResult:
    """Run one sleeping test under a copy of the unit-test conftest.

    Args:
        pytester: The pytester fixture providing the isolated project directory.
        seconds: How long the inner test sleeps.
        *args: Extra command-line arguments for the inner pytest invocation.
        module_marker: Optional ``pytestmark`` assignment prepended to the inner module.

    Returns:
        The result of the inner pytest run.
    """
    pytester.makeconftest(_CONFTEST_SOURCE)
    pytester.makepyfile(test_sleep=module_marker + _SLEEPER.format(seconds=seconds))
    return pytester.runpytest("-p", "no:cacheprovider", *args)


def test_unmarked_test_fails_at_fallback(pytester: pytest.Pytester):
    result = _run_sleeper(pytester, 5.5)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Timeout (>5.0s) from pytest-timeout*"])


def test_module_marker_is_preserved(pytester: pytest.Pytester):
    result = _run_sleeper(pytester, 1, module_marker="import pytest\npytestmark = pytest.mark.timeout(0.5)\n")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Timeout (>0.5s) from pytest-timeout*"])


def test_cli_timeout_zero_disables_fallback(pytester: pytest.Pytester):
    result = _run_sleeper(pytester, 5.5, "--timeout=0")
    result.assert_outcomes(passed=1)


def test_cli_timeout_overrides_fallback(pytester: pytest.Pytester):
    result = _run_sleeper(pytester, 1, "--timeout=0.5")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Timeout (>0.5s) from pytest-timeout*"])


def test_env_timeout_overrides_fallback(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PYTEST_TIMEOUT", "0.5")
    result = _run_sleeper(pytester, 1)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Timeout (>0.5s) from pytest-timeout*"])


def test_ini_timeout_overrides_fallback(pytester: pytest.Pytester):
    pytester.makeini("[pytest]\ntimeout = 0.5\n")
    result = _run_sleeper(pytester, 1)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Timeout (>0.5s) from pytest-timeout*"])


def test_fallback_is_scoped_to_the_conftest_tree(pytester: pytest.Pytester):
    pytester.makepyfile(
        **{
            "unit_tests/conftest.py": _CONFTEST_SOURCE,
            "unit_tests/test_inside.py": "def test_inside():\n    pass\n",
            "other/test_outside.py": "def test_outside():\n    pass\n",
        }
    )
    items, _ = pytester.inline_genitems("unit_tests", "other")
    markers = {item.name: item.get_closest_marker("timeout") for item in items}
    assert markers["test_inside"].args == (5,)
    assert markers["test_outside"] is None
