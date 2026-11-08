# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import coverage
import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]


def _run_aggregation(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    workflow = yaml.safe_load((_ROOT / ".github/workflows/cicd-main.yml").read_text())
    step = next(
        step for step in workflow["jobs"]["Coverage"]["steps"] if step["name"] == "Get total coverage of current branch"
    )
    script = step["run"].replace("${{ matrix.flag }}", "unit-test")
    # Use the installed coverage package without invoking a package download.
    script = script.replace("uvx --from 'coverage[toml]' coverage", f"{shlex.quote(sys.executable)} -m coverage")
    return subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if not key.startswith("COVERAGE_")},
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.parametrize("artifact", [None, "empty", "xml-only", "corrupt"])
def test_missing_or_invalid_coverage_fails(tmp_path: Path, artifact: str | None) -> None:
    """A missing database must not silently turn into a successful upload job."""
    if artifact is not None:
        shard = tmp_path / "coverage-downloads/shard"
        shard.mkdir(parents=True)
        if artifact == "xml-only":
            (shard / "coverage.xml").write_text("<coverage />")
        elif artifact == "corrupt":
            (shard / ".coverage").write_text("not a coverage database")
    result = _run_aggregation(tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (tmp_path / "coverage.xml").exists()


def test_aggregation_preserves_hits_from_all_shards(tmp_path: Path) -> None:
    """The upload report contains the union of hits, including paths with spaces."""
    source = tmp_path / "sample.py"
    source.write_text("first = 1\nsecond = 2\nthird = 3\n")
    for name, lines in (("CPU shard", [1]), ("GPU shard", [2])):
        shard = tmp_path / "coverage-downloads" / name
        shard.mkdir(parents=True)
        data = coverage.CoverageData(basename=str(shard / ".coverage"))
        data.add_lines({str(source): lines})
        data.write()
    result = _run_aggregation(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = ET.parse(tmp_path / "coverage.xml").getroot()
    assert int(report.attrib["lines-covered"]) == 2
    assert int(report.attrib["lines-valid"]) == 3
    assert {int(line.attrib["number"]) for line in report.findall(".//line") if int(line.attrib["hits"])} == {1, 2}
