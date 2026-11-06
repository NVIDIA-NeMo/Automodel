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
"""The migrated performance settings reproduce the original benchmark inputs."""

import re
from pathlib import Path

import pytest
import yaml

from nemo_automodel.components.config import loader
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config


def test_migrated_performance_configs_match_source_benchmarks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only constructor imports are replaced: exercise the real CLI parser,
    # scalar conversion and ordered parent-reset semantics without GPU packages.
    monkeypatch.setattr(loader, "_resolve_target", lambda target: target)
    root = Path(__file__).resolve().parents[3]
    migrated = 0
    for path in sorted((root / "examples/llm_finetune").rglob("*.yaml")):
        text = path.read_text()
        source = re.search(r"# Synthetic performance settings migrated from (examples/llm_benchmark/\S+\.yaml)\.", text)
        if source is None:
            continue
        original = yaml.safe_load(text)
        benchmark_path = root / source.group(1)
        expected = parse_args_and_load_config(str(benchmark_path), argv=[]).to_dict()
        overrides = original["ci"]["perf_testing"]["config_overrides"]
        argv = [arg for key, value in overrides.items() for arg in (f"--{key}", str(value))]
        actual = parse_args_and_load_config(str(path), argv=argv).to_dict()
        for key, value in expected.items():
            if key != "ci":
                assert actual.get(key) == value, (path, key)
        for key in original.keys() - expected.keys() - {"ci"}:
            assert actual.get(key) is None, (path, key)
        packing = expected.get("packed_sequence") or {}
        if packing.get("packed_sequence_size", 0) > 0:
            assert original["packed_sequence"]["packed_sequence_size"] == packing["packed_sequence_size"]
        migrated += 1
    assert migrated > 0
