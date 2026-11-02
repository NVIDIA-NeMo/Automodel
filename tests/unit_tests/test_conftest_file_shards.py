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

"""Whole-file GPU shards and ``cpu_only`` deselection under a copy of the unit-test conftest."""

import json
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.cpu_only

_CONFTEST_SOURCE = Path(__file__).with_name("conftest.py").read_text()
_TWO_TESTS = "def test_one():\n    pass\n\n\ndef test_two():\n    pass\n"


def _collected_files(result: pytest.RunResult) -> list[str]:
    """Return the file of every test node ID that ``--collect-only -q`` printed."""
    return [line.split("::")[0] for line in result.outlines if "::" in line]


def _collect(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    return pytester.runpytest("-p", "no:cacheprovider", "--collect-only", "-q", *args)


def test_file_shards_split_whole_files_balanced_by_recorded_duration(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(_CONFTEST_SOURCE)
    pytester.makepyfile(
        test_heavy=_TWO_TESTS, test_light_a=_TWO_TESTS, test_light_b=_TWO_TESTS, test_unrecorded=_TWO_TESTS
    )
    recorded = {"test_heavy.py": 30.0, "test_light_a.py": 10.0, "test_light_b.py": 10.0}
    (pytester.path / "gpu_shard_durations.json").write_text(json.dumps({"source": "test", "seconds": recorded}))

    shards = [
        _collected_files(_collect(pytester, f"--file-shard-id={shard_id}", "--file-num-shards=2"))
        for shard_id in (0, 1)
    ]

    # Each shard collects both tests of the files it owns, and no file runs twice.
    every_file = ["test_heavy.py", "test_light_a.py", "test_light_b.py", "test_unrecorded.py"]
    assert sorted(shards[0] + shards[1]) == sorted(2 * every_file)
    assert set(shards[0]).isdisjoint(shards[1])
    # The 30-second file balances the two 10-second files plus the unrecorded one.
    assert {"test_heavy.py"} in (set(shards[0]), set(shards[1]))


def test_file_shards_report_their_share_of_the_files(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(_CONFTEST_SOURCE)
    pytester.makepyfile(test_a=_TWO_TESTS, test_b=_TWO_TESTS, test_c=_TWO_TESTS)
    (pytester.path / "gpu_shard_durations.json").write_text(json.dumps({"source": "test", "seconds": {}}))

    result = pytester.runpytest("-p", "no:cacheprovider", "--collect-only", "--file-shard-id=1", "--file-num-shards=3")

    result.stdout.fnmatch_lines(["file shard 2/3: 1 of 3 test files"])


def test_file_shard_id_must_fit_the_shard_count(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(_CONFTEST_SOURCE)
    pytester.makepyfile(test_a=_TWO_TESTS)

    result = _collect(pytester, "--file-shard-id=2", "--file-num-shards=2")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "--file-shard-id in [0, --file-num-shards)" in result.stderr.str()


def test_cpu_only_marks_modules_and_whole_directories_for_deselection(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(_CONFTEST_SOURCE)
    pytester.makepyfile(
        test_cuda=_TWO_TESTS,
        test_gloo="import pytest\n\npytestmark = pytest.mark.cpu_only\n\n\n" + _TWO_TESTS,
    )
    pytester.mkdir("ci_tests").joinpath("test_workflow.py").write_text(_TWO_TESTS)

    gpu_shard = _collect(pytester, "-m", "not cpu_only")
    cpu_job = _collect(pytester)

    assert set(_collected_files(gpu_shard)) == {"test_cuda.py"}
    assert set(_collected_files(cpu_job)) == {"test_cuda.py", "test_gloo.py", "ci_tests/test_workflow.py"}
