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

from tests.functional_tests.checkpoint_robustness.router_diagnostics import compare_glm_router_captures


def _capture(router_logits, indices):
    return {
        "schema_version": 1,
        "framework": "test",
        "model_family": "glm4_moe_lite",
        "layers": {
            1: {
                "router_logits": torch.tensor(router_logits),
                "correction_bias": torch.zeros(3),
                "indices": torch.tensor(indices),
                "score_func": "sigmoid",
                "n_groups": 1,
            }
        },
    }


def test_router_report_separates_natural_floor_from_routed_tail(tmp_path):
    hf_path = tmp_path / "hf.pt"
    automodel_path = tmp_path / "automodel.pt"
    report_path = tmp_path / "report.json"
    torch.save(_capture([[4.0, 1.0, 0.0], [1.0, 0.99, 0.0]], [[0], [0]]), hf_path)
    torch.save(_capture([[3.9, 1.1, 0.0], [0.99, 1.0, 0.0]], [[0], [1]]), automodel_path)
    reference_logits = torch.tensor([[[2.0, -2.0], [20.0, -20.0]]])
    candidate_logits = reference_logits.clone()
    candidate_logits[:, 1, :] = -candidate_logits[:, 1, :]

    report = compare_glm_router_captures(
        hf_path,
        automodel_path,
        report_path,
        reference_logits=reference_logits,
        candidate_logits=candidate_logits,
    )

    assert report["route_flip_token_count"] == 1
    assert report["route_flip_token_fraction"] == pytest.approx(0.5)
    final_token_kl = report["final_token_kl"]
    assert final_token_kl["natural_agreement_floor_no_flipped_layers"]["token_count"] == 1
    assert final_token_kl["natural_agreement_floor_no_flipped_layers"]["mean_kl"] == pytest.approx(0.0)
    assert final_token_kl["routed_tail_one_or_more_flipped_layers"]["token_count"] == 1
    assert final_token_kl["routed_tail_one_or_more_flipped_layers"]["mean_kl"] > 1.0
    assert final_token_kl["by_flipped_layer_count"][0]["token_count"] == 1
    assert final_token_kl["by_flipped_layer_count"][1]["token_count"] == 1
    persisted = json.loads(report_path.read_text())
    assert persisted["schema_version"] == 1
    assert persisted["final_token_kl"]["by_flipped_layer_count"]["0"]["token_count"] == 1
