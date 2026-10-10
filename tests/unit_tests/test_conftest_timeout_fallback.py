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
"""Behavioral checks for the unit-test runtime policy in ``conftest.py``.

Each test copies the real conftest into an isolated pytester project. Git commits
model the base and pull-request revisions so the checks exercise the same
changed-test discovery used in CI.
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

pytestmark = pytest.mark.timeout(70)

_CONFTEST_SOURCE = Path(__file__).with_name("conftest.py").read_text()
_POLICY_ARGS = ("--unit-test-runtime-budget=0.05", "--unit-test-hard-timeout=0.5")
_TEST_MODULE = """
import time
from pathlib import Path

import pytest

{module_marker}
{function_marker}def test_sleep():
{body}
"""


def _git(pytester: pytest.Pytester, *args: str) -> None:
    subprocess.run(["git", *args], cwd=pytester.path, check=True, capture_output=True, text=True)


def _initialize_repository(pytester: pytest.Pytester) -> None:
    _git(pytester, "init", "-q")
    _git(pytester, "config", "user.email", "runtime-policy@example.com")
    _git(pytester, "config", "user.name", "Runtime Policy Test")
    _git(pytester, "add", ".")
    _git(pytester, "commit", "-q", "-m", "baseline")


@pytest.fixture(autouse=True)
def _isolate_runtime_base(monkeypatch: pytest.MonkeyPatch) -> None:
    # The inner projects have their own history, independent of the outer CI base.
    monkeypatch.delenv("AUTOMODEL_RUNTIME_BUDGET_BASE", raising=False)


def _module_source(*, body: str, module_marker: str = "", function_marker: str = "") -> str:
    indented_body = "\n".join(f"    {line}" for line in body.splitlines())
    return _TEST_MODULE.format(
        module_marker=module_marker,
        function_marker=function_marker,
        body=indented_body,
    )


def _run_sleeper(
    pytester: pytest.Pytester,
    seconds: float,
    *args: str,
    module_marker: str = "",
    function_marker: str = "",
    changed: bool = True,
    run_in_subprocess: bool = False,
    body: str | None = None,
) -> tuple[pytest.RunResult, Path]:
    """Run one sleeping test under a copy of the unit-test conftest."""
    pytester.makeconftest(_CONFTEST_SOURCE)
    baseline_body = (
        'Path("completed").write_text("yes")'
        if changed
        else f'time.sleep({seconds})\nPath("completed").write_text("yes")'
    )
    test_path = pytester.makepyfile(
        test_sleep=_module_source(
            body=baseline_body,
            module_marker=module_marker,
            function_marker=function_marker,
        )
    )
    _initialize_repository(pytester)

    if changed:
        test_path.write_text(
            _module_source(
                body=body or f'time.sleep({seconds})\nPath("completed").write_text("yes")',
                module_marker=module_marker,
                function_marker=function_marker,
            )
        )
        _git(pytester, "add", str(test_path.name))
        _git(pytester, "commit", "-q", "-m", "change test")

    if run_in_subprocess:
        # pytester relocates HOME, which can hide user-site packages such as torch
        # from the fresh interpreter used by local development environments.
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path for path in sys.path if path))
            result = pytester.runpytest_subprocess("-p", "no:cacheprovider", *_POLICY_ARGS, *args)
    else:
        result = pytester.runpytest("-p", "no:cacheprovider", *_POLICY_ARGS, *args)
    return result, pytester.path / "completed"


def test_changed_test_fails_soft_budget_after_completing(pytester: pytest.Pytester):
    result, completed = _run_sleeper(pytester, 0.12)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*exceeded its 0.05s runtime budget in both attempts*"])
    assert completed.read_text() == "yes"
    assert "Timeout (" not in result.stdout.str()


def test_inherited_timeout_does_not_exempt_changed_test(pytester: pytest.Pytester):
    result, completed = _run_sleeper(
        pytester,
        0.12,
        module_marker="pytestmark = pytest.mark.timeout(0.4)",
    )

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*exceeded its 0.05s runtime budget in both attempts*"])
    assert completed.exists()


def test_unchanged_test_preserves_existing_timeout_marker(pytester: pytest.Pytester):
    result, _ = _run_sleeper(
        pytester,
        0.12,
        module_marker="pytestmark = pytest.mark.timeout(0.4)",
        changed=False,
    )

    result.assert_outcomes(passed=1)


def test_exact_runtime_budget_cannot_raise_changed_test_limit(pytester: pytest.Pytester) -> None:
    result, completed = _run_sleeper(
        pytester,
        0.12,
        function_marker="@pytest.mark.runtime_budget(0.2, hard_timeout=0.5)\n",
    )

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*exceeded its 0.05s runtime budget in both attempts*"])
    assert completed.exists()


def test_unchanged_runtime_budget_keeps_only_hang_watchdog(pytester: pytest.Pytester) -> None:
    result, _ = _run_sleeper(
        pytester,
        0.12,
        function_marker="@pytest.mark.runtime_budget(0.05, hard_timeout=0.5)\n",
        changed=False,
    )
    result.assert_outcomes(passed=1)
    assert "confirming once" not in result.stdout.str()


def test_changed_test_cannot_request_more_than_thirty_seconds(pytester: pytest.Pytester) -> None:
    result, completed = _run_sleeper(
        pytester,
        0,
        function_marker='@pytest.mark.runtime_budget(60, reason="slow integration")\n',
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*cannot request a runtime_budget above 30s*"])
    assert not completed.exists()


_COUNT_ATTEMPTS = """attempts = Path("attempts")
count = int(attempts.read_text()) + 1 if attempts.exists() else 1
attempts.write_text(str(count))
"""


def test_transient_overrun_passes_after_one_confirmation(pytester: pytest.Pytester) -> None:
    # Leave room for pytest/coverage fixture overhead in the fast attempt.
    result, _ = _run_sleeper(
        pytester,
        0,
        "-x",
        "--junitxml=results.xml",
        "--unit-test-runtime-budget=0.5",
        "--unit-test-hard-timeout=2",
        body=_COUNT_ATTEMPTS + "time.sleep(1 if count == 1 else 0)",
    )
    result.assert_outcomes(passed=1)
    assert (pytester.path / "attempts").read_text() == "2"
    cases = ET.parse(pytester.path / "results.xml").findall(".//testcase")
    assert len(cases) == 1
    assert cases[0].find("failure") is None


def test_persistent_overrun_fails_after_exactly_two_attempts(pytester: pytest.Pytester) -> None:
    result, _ = _run_sleeper(pytester, 0, "-x", body=_COUNT_ATTEMPTS + "time.sleep(0.12)")
    result.assert_outcomes(failed=1)
    assert (pytester.path / "attempts").read_text() == "2"


def test_hard_timeout_is_not_retried(pytester: pytest.Pytester) -> None:
    result, _ = _run_sleeper(pytester, 0, run_in_subprocess=True, body=_COUNT_ATTEMPTS + "time.sleep(2)")
    result.assert_outcomes(failed=1)
    assert (pytester.path / "attempts").read_text() == "1"
    result.stdout.fnmatch_lines(["*Timeout (>0.5s) from pytest-timeout*"])


def test_confirmation_has_its_own_hard_watchdog(pytester: pytest.Pytester) -> None:
    result, _ = _run_sleeper(pytester, 0.3, run_in_subprocess=True)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*exceeded its 0.05s runtime budget in both attempts*"])
    assert "Timeout (" not in result.stdout.str()


@pytest.mark.parametrize("fail_on_attempt", [1, 2])
def test_assertion_failures_are_never_retried(pytester: pytest.Pytester, fail_on_attempt: int) -> None:
    result, _ = _run_sleeper(
        pytester,
        0,
        body=_COUNT_ATTEMPTS + f"time.sleep(0.12)\nassert count != {fail_on_attempt}, 'real failure'",
    )
    result.assert_outcomes(failed=1)
    assert (pytester.path / "attempts").read_text() == str(fail_on_attempt)
    result.stdout.fnmatch_lines(["*AssertionError: real failure*"])


@pytest.mark.parametrize("phase", ["setup", "teardown"])
def test_fixture_errors_are_never_retried(pytester: pytest.Pytester, phase: str) -> None:
    fixture = """
@pytest.fixture(autouse=True)
def broken_fixture():
    Path("fixture_attempts").open("a").write("x")
    {setup}
    yield
    {teardown}
""".format(
        setup="raise ValueError('broken setup')" if phase == "setup" else "pass",
        teardown="raise ValueError('broken teardown')" if phase == "teardown" else "pass",
    )
    result, _ = _run_sleeper(pytester, 0.12, module_marker=fixture)
    result.assert_outcomes(errors=1, passed=int(phase == "teardown"))
    assert (pytester.path / "fixture_attempts").read_text() == "x"


@pytest.mark.parametrize("phase", ["setup", "teardown"])
def test_fixture_time_counts_and_function_fixtures_are_recreated(pytester: pytest.Pytester, phase: str) -> None:
    fixture = """
@pytest.fixture(autouse=True)
def slow_fixture():
    Path("fixture_attempts").open("a").write("x")
    {setup}
    yield
    {teardown}
""".format(
        setup="time.sleep(0.12)" if phase == "setup" else "pass",
        teardown="time.sleep(0.12)" if phase == "teardown" else "pass",
    )
    result, _ = _run_sleeper(pytester, 0, module_marker=fixture)
    result.assert_outcomes(failed=1)
    assert (pytester.path / "fixture_attempts").read_text() == "xx"


@pytest.mark.parametrize("duration", ["float('nan')", "float('inf')", "-1", "0"])
def test_invalid_runtime_budget_is_rejected(pytester: pytest.Pytester, duration: str) -> None:
    result, _ = _run_sleeper(pytester, 0, function_marker=f"@pytest.mark.runtime_budget({duration})\n")
    assert result.ret == pytest.ExitCode.USAGE_ERROR


def test_confirmation_works_with_xdist_and_coverage(pytester: pytest.Pytester) -> None:
    result, _ = _run_sleeper(
        pytester,
        0,
        "-n",
        "2",
        "--dist=loadfile",
        "--cov=.",
        "--cov-report=",
        "--junitxml=results.xml",
        "--unit-test-runtime-budget=0.5",
        "--unit-test-hard-timeout=2",
        run_in_subprocess=True,
        body=_COUNT_ATTEMPTS + "time.sleep(1 if count == 1 else 0)",
    )
    result.assert_outcomes(passed=1)
    assert (pytester.path / "attempts").read_text() == "2"
    assert len(ET.parse(pytester.path / "results.xml").findall(".//testcase")) == 1


def test_module_runtime_budget_is_rejected(pytester: pytest.Pytester):
    result, _ = _run_sleeper(
        pytester,
        0,
        module_marker="pytestmark = pytest.mark.runtime_budget(0.2, hard_timeout=0.5)",
        changed=False,
    )

    assert result.ret != 0
    result.stderr.fnmatch_lines(["*runtime_budget must be applied directly to a test*"])


def test_cli_timeout_zero_disables_policy(pytester: pytest.Pytester):
    result, completed = _run_sleeper(pytester, 0.12, "--timeout=0")

    result.assert_outcomes(passed=1)
    assert completed.exists()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=70,
    reason="starts a fresh pytest subprocess to isolate the intentional timeout",
)
def test_cli_timeout_overrides_policy(pytester: pytest.Pytester):
    result, completed = _run_sleeper(pytester, 0.12, "--timeout=0.05", run_in_subprocess=True)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Timeout (>0.05s) from pytest-timeout*"])
    assert not completed.exists()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=70,
    reason="starts a fresh pytest subprocess to isolate the intentional timeout",
)
def test_env_timeout_overrides_policy(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PYTEST_TIMEOUT", "0.05")
    result, completed = _run_sleeper(pytester, 0.12, run_in_subprocess=True)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Timeout (>0.05s) from pytest-timeout*"])
    assert not completed.exists()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=70,
    reason="starts a fresh pytest subprocess to isolate the intentional timeout",
)
def test_ini_timeout_overrides_policy(pytester: pytest.Pytester):
    pytester.makeini("[pytest]\ntimeout = 0.05\n")
    result, completed = _run_sleeper(pytester, 0.12, run_in_subprocess=True)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Timeout (>0.05s) from pytest-timeout*"])
    assert not completed.exists()


def test_hard_watchdog_is_scoped_to_the_conftest_tree(pytester: pytest.Pytester):
    pytester.makepyfile(
        **{
            "unit_tests/conftest.py": _CONFTEST_SOURCE,
            "unit_tests/test_inside.py": "def test_inside():\n    pass\n",
            "other/test_outside.py": "def test_outside():\n    pass\n",
        }
    )
    items, _ = pytester.inline_genitems("unit_tests", "other")
    markers = {item.name: item.get_closest_marker("timeout") for item in items}

    assert markers["test_inside"].args == (70.0,)
    assert markers["test_outside"] is None
