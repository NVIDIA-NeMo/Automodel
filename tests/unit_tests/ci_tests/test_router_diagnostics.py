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
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import Gate
from tests.functional_tests.checkpoint_robustness.router_diagnostics import (
    _flip_pair_bias_directions,
    capture_glm_automodel_routers,
    capture_glm_hf_routers,
    compare_glm_router_captures,
    summarize_glm_router_shape_captures,
)


class Glm4MoeLiteMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_group=1)
        self.gate = SimpleNamespace(e_score_correction_bias=torch.zeros(3))

    def route_tokens_to_experts(self, router_logits):
        return router_logits.topk(1, dim=-1).indices, torch.ones_like(router_logits[..., :1])


class _HfRouterModel(nn.Module):
    def __init__(self, *, include_invalid_name: bool = False):
        super().__init__()
        self.config = SimpleNamespace(model_type="glm4_moe_lite")
        self.layers = nn.ModuleDict({"0": Glm4MoeLiteMoE()})
        if include_invalid_name:
            self.orphan_router = Glm4MoeLiteMoE()


class _GateBlock(nn.Module):
    def __init__(self, gate: Gate):
        super().__init__()
        self.gate = gate


class _AutoModelRouterModel(nn.Module):
    def __init__(self, gate: Gate):
        super().__init__()
        self.config = SimpleNamespace(model_type="glm4_moe_lite")
        self.layers = nn.ModuleDict({"0": _GateBlock(gate)})


def _gate() -> Gate:
    config = MoEConfig(
        n_routed_experts=3,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        dim=4,
        inter_dim=8,
        moe_inter_dim=8,
        norm_topk_prob=False,
        force_e_score_correction_bias=True,
        dtype=torch.float32,
    )
    return Gate(config, gate_precision=torch.float32)


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


def test_hf_capture_validates_all_modules_before_patching(tmp_path):
    model = _HfRouterModel(include_invalid_name=True)
    router = model.layers["0"]

    with pytest.raises(ValueError, match="transformer layer index"):
        with capture_glm_hf_routers(model, tmp_path / "capture.pt"):
            pass

    assert "route_tokens_to_experts" not in router.__dict__


def test_hf_capture_rejects_missing_router_on_nonzero_rank_path(tmp_path):
    model = nn.Module()
    model.config = SimpleNamespace(model_type="glm4_moe_lite")

    with patch(
        "tests.functional_tests.checkpoint_robustness.router_diagnostics._rank0",
        return_value=False,
    ):
        with pytest.raises(ValueError, match="No vanilla-HF"):
            with capture_glm_hf_routers(model, tmp_path / "capture.pt"):
                pass


@pytest.mark.parametrize("nonfinite", [float("inf"), float("nan")])
def test_hf_capture_rejects_nonfinite_router_values_and_restores_patch(tmp_path, nonfinite):
    model = _HfRouterModel()
    router = model.layers["0"]

    with pytest.raises(ValueError, match="non-finite router_logits"):
        with capture_glm_hf_routers(model, tmp_path / "capture.pt"):
            router.route_tokens_to_experts(torch.tensor([[nonfinite, 0.0, 1.0]]))

    assert "route_tokens_to_experts" not in router.__dict__


def test_automodel_capture_wraps_cuda_graph_routing_core(tmp_path):
    gate = _gate()
    gate.use_routing_core = True
    model = _AutoModelRouterModel(gate)
    capture_path = tmp_path / "capture.pt"

    with capture_glm_automodel_routers(model, capture_path):
        gate(torch.randn(2, 4), torch.ones(2, dtype=torch.bool), None)

    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    assert capture["model_family"] == "glm4_moe_lite"
    assert capture["layers"][0]["indices"].shape == (2, 1)
    assert "forward" not in gate.routing_core.__dict__


def test_flip_bias_direction_vectorization_counts_added_dropped_pairs():
    summary = _flip_pair_bias_directions(
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([[0, 2], [0, 2]]),
        torch.tensor([0.5, 0.2, 0.8]),
        torch.tensor([True, True]),
    )

    assert summary["added_bias_greater_count"] == 2
    assert summary["added_bias_equal_count"] == 0
    assert summary["added_bias_less_count"] == 0


def test_router_report_separates_natural_floor_from_routed_tail(tmp_path, capsys):
    hf_path = tmp_path / "hf.pt"
    automodel_path = tmp_path / "automodel.pt"
    report_path = tmp_path / "report.json"
    torch.save(_capture([[4.0, 1.0, 0.0], [1.0, 0.99, 0.0]], [[0], [0]]), hf_path)
    torch.save(_capture([[3.9, 1.1, 0.0], [0.99, 1.0, 0.0]], [[0], [1]]), automodel_path)
    reference_logits = torch.tensor([[[2.0, -2.0], [20.0, -20.0]]])
    candidate_logits = reference_logits.clone()
    candidate_logits[:, 1, :] = -candidate_logits[:, 1, :]

    report = compare_glm_router_captures(
        hf_path=hf_path,
        automodel_path=automodel_path,
        report_path=report_path,
        reference_logits=reference_logits,
        candidate_logits=candidate_logits,
    )

    assert report["route_flip_token_count"] == 1
    assert report["route_flip_token_fraction"] == pytest.approx(0.5)
    assert report["early_layer_summary"]["route_flip_token_count"] == 1
    assert report["early_layer_summary"]["flips_above_score_perturbation_bound_fraction"] == 0.0
    sign_test = report["early_layer_summary"]["correction_bias_direction_sign_test"]
    assert sign_test["added_bias_equal_count"] == 1
    assert sign_test["added_bias_greater_fraction_with_ties_split"] == pytest.approx(0.5)
    final_token_kl = report["final_token_kl"]
    assert final_token_kl["natural_agreement_floor_no_flipped_layers"]["token_count"] == 1
    assert final_token_kl["natural_agreement_floor_no_flipped_layers"]["mean_kl"] == pytest.approx(0.0)
    assert final_token_kl["routed_tail_one_or_more_flipped_layers"]["token_count"] == 1
    assert final_token_kl["routed_tail_one_or_more_flipped_layers"]["mean_kl"] > 1.0
    assert final_token_kl["by_flipped_layer_count"][0]["token_count"] == 1
    assert final_token_kl["by_flipped_layer_count"][1]["token_count"] == 1
    persisted = json.loads(report_path.read_text())
    assert persisted["schema_version"] == 2
    assert persisted["final_token_kl"]["by_flipped_layer_count"]["0"]["token_count"] == 1
    concise_log = capsys.readouterr().out
    assert "CHECKPOINT_ROUTER_DIAGNOSTICS" in concise_log
    assert '"layer_metrics"' not in concise_log


def test_router_shape_report_attributes_kl_to_sustained_self_flips(tmp_path):
    base_path = tmp_path / "base.pt"
    standalone_path = tmp_path / "standalone.pt"
    base_capture = _capture([[4.0, 1.0, 0.0], [1.0, 0.99, 0.0]], [[0], [0]])
    standalone_capture = _capture([[3.9, 1.1, 0.0], [0.99, 1.0, 0.0]], [[0], [1]])
    # Repeat the same routed layer under distinct indices to model 11-layer persistence.
    base_capture["layers"] = {layer: base_capture["layers"][1] for layer in range(1, 12)}
    standalone_capture["layers"] = {layer: standalone_capture["layers"][1] for layer in range(1, 12)}
    torch.save(base_capture, base_path)
    torch.save(standalone_capture, standalone_path)
    reference_logits = torch.tensor([[[2.0, -2.0], [20.0, -20.0]]])
    candidate_logits = reference_logits.clone()
    candidate_logits[:, 1, :] = -candidate_logits[:, 1, :]

    report = summarize_glm_router_shape_captures(
        base_path=base_path,
        standalone_path=standalone_path,
        reference_logits=reference_logits,
        candidate_logits=candidate_logits,
    )

    assert report["tokens_with_any_flip_count"] == 1
    assert report["tokens_with_sustained_flips_count"] == 1
    assert report["sustained_flip_final_token_kl"]["token_count"] == 1
    assert report["sustained_flip_final_token_kl"]["mean_kl"] > 1.0


def test_router_report_handles_all_zero_logit_cosine(tmp_path):
    hf_path = tmp_path / "hf.pt"
    automodel_path = tmp_path / "automodel.pt"
    report_path = tmp_path / "report.json"
    capture = _capture([[0.0, 0.0, 0.0]], [[0]])
    torch.save(capture, hf_path)
    torch.save(capture, automodel_path)
    final_logits = torch.tensor([[[0.0, 0.0]]])

    report = compare_glm_router_captures(
        hf_path=hf_path,
        automodel_path=automodel_path,
        report_path=report_path,
        reference_logits=final_logits,
        candidate_logits=final_logits.clone(),
    )

    assert report["router_logit_cosine"] == 1.0
    assert report["layer_metrics"][1]["router_logit_cosine"] == 1.0


def test_router_report_handles_one_zero_logit_cosine(tmp_path):
    hf_path = tmp_path / "hf.pt"
    automodel_path = tmp_path / "automodel.pt"
    report_path = tmp_path / "report.json"
    torch.save(_capture([[0.0, 0.0, 0.0]], [[0]]), hf_path)
    torch.save(_capture([[1.0, 0.0, 0.0]], [[0]]), automodel_path)
    final_logits = torch.tensor([[[0.0, 0.0]]])

    report = compare_glm_router_captures(
        hf_path=hf_path,
        automodel_path=automodel_path,
        report_path=report_path,
        reference_logits=final_logits,
        candidate_logits=final_logits.clone(),
    )

    assert report["router_logit_cosine"] == 0.0
    assert report["layer_metrics"][1]["router_logit_cosine"] == 0.0


def test_router_report_rejects_equal_numel_shape_mismatch(tmp_path):
    hf_path = tmp_path / "hf.pt"
    automodel_path = tmp_path / "automodel.pt"
    report_path = tmp_path / "report.json"
    hf_capture = _capture([[4.0, 1.0, 0.0], [1.0, 0.99, 0.0]], [[0], [0]])
    automodel_capture = _capture([[4.0, 1.0, 0.0], [1.0, 0.99, 0.0]], [[0], [0]])
    automodel_capture["layers"][1]["router_logits"] = automodel_capture["layers"][1]["router_logits"].reshape(1, 2, 3)
    automodel_capture["layers"][1]["indices"] = automodel_capture["layers"][1]["indices"].reshape(1, 2, 1)
    torch.save(hf_capture, hf_path)
    torch.save(automodel_capture, automodel_path)
    final_logits = torch.zeros(1, 2, 2)

    with pytest.raises(ValueError, match="Router capture shape mismatch"):
        compare_glm_router_captures(
            hf_path=hf_path,
            automodel_path=automodel_path,
            report_path=report_path,
            reference_logits=final_logits,
            candidate_logits=final_logits.clone(),
        )
