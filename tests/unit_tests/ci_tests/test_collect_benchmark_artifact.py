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

import json
import sys
from pathlib import Path

import pytest

from tests.ci_tests.scripts.collect_benchmark_artifact import main, parse_log


@pytest.mark.parametrize("prefix", ["", "0: 2026-10-13 12:00:00 | INFO | nemo_automodel.recipes.llm.benchmark | "])
@pytest.mark.parametrize(
    "flops_line,expected",
    [
        (
            "TFLOPs/global step: 2683.590000 | TFLOPs/GPU/step: 83.862188 | FLOPs scope: model",
            {"tflops_total": 2683.59, "tflops_per_gpu": 83.862188},
        ),
        ("TFLOPs/GPU: 2683.590000 | FLOPs scope: model", {"tflops_per_gpu": 2683.59}),
        ("FLOPs unavailable", {}),
    ],
)
def test_parse_flops_ignores_mtp_and_preserves_legacy_logs(
    tmp_path: Path, prefix: str, flops_line: str, expected: dict[str, float]
) -> None:
    log = tmp_path / "benchmark.log"
    log.write_text(
        "\n".join(
            prefix + line
            for line in (
                "MTP TFLOPs/GPU: 12.0 (backbone 2671.59, total 2683.59)",
                "MTP TFLOPs/global step: 12.0 (backbone 2671.59, total 2683.59)",
                "MTP TFLOPs/GPU/step: 0.375",
                flops_line,
                "Average iteration time: 2.000 seconds",
                "Average MFU: 4.000000% | FLOPs scope: model",
                "World size: 32",
            )
        )
    )
    parsed = parse_log(str(log))
    assert parsed["results"] == {**expected, "avg_iter_time_seconds": 2.0, "avg_mfu_percent": 4.0}
    assert parsed["metadata"] == {"world_size": 32}


@pytest.mark.parametrize(
    "peak_lines, expected_peak",
    [
        (
            [
                "MFU peak: 1979 TFLOPs/s per device (NVIDIA H100 80GB HBM3, float8_e4m3fn)",
                "MFU peak: 1979 TFLOPs/s per device",
            ],
            1979.0,
        ),
        (["MFU peak: 494.5 TFLOPs/s per device (NVIDIA H100 80GB HBM3, float32 with TF32)"], 494.5),
        (["MFU peak: unknown"], None),
    ],
)
def test_artifact_records_the_logged_peak_instead_of_the_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, peak_lines: list[str], expected_peak: float | None
) -> None:
    config = tmp_path / "recipe.yaml"
    # The recipe ignores benchmark.peak_tflops; the artifact must not report it.
    config.write_text("benchmark:\n  warmup_steps: 10\n  peak_tflops: 989\n")
    log = tmp_path / "benchmark.log"
    log.write_text("\n".join([*peak_lines, "Average MFU: 10.000000% | FLOPs scope: model"]))
    output = tmp_path / "benchmark_results.json"
    monkeypatch.setattr(sys, "argv", ["collect", "--config", str(config), "--log", str(log), "--output", str(output)])

    main()

    artifact = json.loads(output.read_text())
    assert artifact["config"]["peak_tflops"] == expected_peak
    assert artifact["results"] == {"avg_mfu_percent": 10.0}
