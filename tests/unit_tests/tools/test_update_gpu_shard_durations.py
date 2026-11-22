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

"""Tests for rebuilding the GPU shard durations from pytest progress in job logs."""

import json

import pytest

from tools.update_gpu_shard_durations import main, parse_file_durations

# A first attempt that finished, then a retry whose progress must not be counted again.
_JOB_LOG = """\
2026-10-28T04:01:29.8000000Z + coverage run -m pytest tests/unit_tests
2026-10-28T04:03:24.0000000Z collected 6 items
2026-10-28T04:03:30.5000000Z \x1b[32mtests/unit_tests/models/test_slow.py \x1b[0m\x1b[32m.\x1b[0m\x1b[32m.\x1b[0m [ 33%]
2026-10-28T04:03:31.0000000Z tests/unit_tests/loss/test_wrapped.py ......................................... [ 50%]
2026-10-28T04:03:32.0000000Z unrelated output from a test
2026-10-28T04:03:33.0000000Z .......                                                                  [ 67%]
2026-10-28T04:03:33.2500000Z tests/unit_tests/test_fast.py ss                                           [100%]
2026-10-28T04:03:40.0000000Z ========= 4 passed, 2 skipped in 16.00s ==========
2026-10-28T04:04:00.0000000Z ##[group]Attempt 2
2026-10-28T04:06:00.0000000Z tests/unit_tests/models/test_slow.py ..                                    [ 33%]
"""


def test_parse_file_durations_measures_each_file_until_the_next_starts():
    durations = parse_file_durations(_JOB_LOG)

    assert durations == pytest.approx({"models/test_slow.py": 6.5, "loss/test_wrapped.py": 2.5, "test_fast.py": 0.25})


def test_parse_file_durations_reads_gh_run_view_logs():
    prefixed = "\n".join(f"L0_Unit_Tests_GPU_1_of_2\tmain\t{line}" for line in _JOB_LOG.splitlines())

    assert parse_file_durations(prefixed) == parse_file_durations(_JOB_LOG)


def test_parse_file_durations_rejects_logs_without_pytest_progress():
    with pytest.raises(ValueError, match="no pytest progress lines"):
        parse_file_durations("2026-10-28T04:01:29.8000000Z ERROR: the container did not start\n")


def test_main_sums_shard_logs_into_sorted_json(tmp_path):
    shard_logs = [tmp_path / "shard1.log", tmp_path / "shard2.log"]
    shard_logs[0].write_text(_JOB_LOG)
    shard_logs[1].write_text(
        "2026-10-28T04:03:24.0000000Z collected 1 item\n"
        "2026-10-28T04:03:26.0000000Z tests/unit_tests/a/test_other.py .              [100%]\n"
        "2026-10-28T04:03:27.0000000Z ===== 1 passed in 3.00s =====\n"
    )
    output = tmp_path / "durations.json"

    assert main([str(path) for path in shard_logs] + ["--source", "run 1", "--output", str(output)]) == 0

    document = json.loads(output.read_text())
    assert document["source"] == "run 1"
    expected_files = ["a/test_other.py", "loss/test_wrapped.py", "models/test_slow.py", "test_fast.py"]
    assert list(document["seconds"]) == expected_files
    assert document["seconds"]["a/test_other.py"] == 2.0


def test_main_fails_without_writing_when_a_log_has_no_progress(tmp_path, capsys):
    empty_log = tmp_path / "shard.log"
    empty_log.write_text("2026-10-28T04:01:29.8000000Z ERROR: the container did not start\n")
    output = tmp_path / "durations.json"

    assert main([str(empty_log), "--source", "run 1", "--output", str(output)]) == 1

    assert not output.exists()
    assert "no pytest progress lines" in capsys.readouterr().err
