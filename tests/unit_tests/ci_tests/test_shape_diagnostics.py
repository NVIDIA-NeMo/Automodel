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

import json

import pytest
import torch

from tests.functional_tests.checkpoint_robustness.shape_diagnostics import (
    _build_shape_diagnostic_report,
    _normalize_shape_diagnostic_config,
    _persist_shape_diagnostic_report,
)


def test_shape_diagnostic_config_adds_standing_gate_and_deduplicates_sweep():
    config = _normalize_shape_diagnostic_config(
        {"enabled": True, "sweep_lengths": [512, 128, 512]},
        parity_sequence_length=2048,
    )

    assert config.sweep_lengths == (128, 512)
    assert config.lengths(parity_sequence_length=2048, gate_sequence_length=256) == (128, 256, 512)


@pytest.mark.parametrize(
    ("raw_config", "message"),
    [
        ({"enabled": True, "median_kl": True}, "Unknown shape_diagnostic fields"),
        ({"enabled": True, "sweep_lengths": [2048]}, "must be shorter than parity_sequence_length"),
        ({"enabled": False, "sweep_lengths": [128]}, "requires shape_diagnostic.enabled=true"),
    ],
)
def test_shape_diagnostic_config_rejects_invalid_or_median_fields(raw_config, message):
    with pytest.raises(ValueError, match=message):
        _normalize_shape_diagnostic_config(raw_config, parity_sequence_length=2048)


def test_shape_report_labels_bitwise_point_and_persists_concise_log(tmp_path, capsys):
    base_logits = torch.tensor([[[2.0, -2.0], [1.0, -1.0], [0.5, -0.5]]])
    standalone_logits = {
        1: base_logits[:, :1].clone(),
        2: base_logits[:, :2].clone() + 0.1,
    }
    report = _build_shape_diagnostic_report(
        base_logits,
        standalone_logits,
        parity_document_sha256="fixed-document",
        gate_sequence_length=1,
        sweep_lengths=(2,),
        router_diagnostics={
            2: {
                "tokens_with_any_flip_count": 1,
                "sustained_flip_final_token_kl": {"mean_kl": 0.25},
            }
        },
    )

    assert report["enforced"] is False
    assert report["points"][1]["purposes"] == ["standing_gate_context"]
    assert report["points"][1]["interpretation"] == "same_kernel_regime_not_probing"
    assert report["points"][2]["purposes"] == ["calibration_sweep"]
    assert report["points"][2]["interpretation"] == "shape_sensitive_numeric_path_observed"

    report_path = tmp_path / "shape.json"
    _persist_shape_diagnostic_report(report, report_path)
    assert json.loads(report_path.read_text())["parity_document_sha256"] == "fixed-document"
    concise_log = capsys.readouterr().out
    assert "CHECKPOINT_SHAPE_DIAGNOSTICS" in concise_log
    assert '"metrics"' not in concise_log
