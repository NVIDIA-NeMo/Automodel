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

"""Unit tests for MoE load balance metrics."""

import math

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.moe.load_balance_metrics import (
    collect_expert_loads,
    compute_brief_metrics,
    compute_detailed_metrics,
    enable_load_balance_tracking,
)


def _make_layer_loads(loads_list, aux_losses=None):
    """Helper to build a layer_loads dict from a list of 1-D tensors."""
    result = {}
    for i, load in enumerate(loads_list):
        al = None if aux_losses is None else aux_losses[i]
        result[f"layers.{i}.moe.gate"] = {
            "expert_load": torch.tensor(load, dtype=torch.float32),
            "aux_loss": torch.tensor(al) if al is not None else None,
            "n_experts": len(load),
        }
    return result


class TestComputeBriefMetrics:
    def test_uniform_load(self):
        """Uniform load should give CV=0, utilization=1.0 for all experts."""
        layer_loads = _make_layer_loads([[100.0, 100.0, 100.0, 100.0]])
        metrics = compute_brief_metrics(layer_loads, top_k=4)

        assert metrics["moe/cv_mean"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["moe/expert_utilization_p25"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["moe/expert_utilization_median"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["moe/expert_utilization_p75"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["moe/expert_utilization_min"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["moe/expert_utilization_max"] == pytest.approx(1.0, abs=1e-4)

    def test_skewed_load(self):
        """Skewed load should give CV > 0, utilization_max > 1."""
        layer_loads = _make_layer_loads([[400.0, 100.0, 100.0, 100.0]])
        metrics = compute_brief_metrics(layer_loads)

        assert metrics["moe/cv_mean"] > 0.5
        # expert_utilization_max = max(load) / mean(load) = 400 / 175
        assert metrics["moe/expert_utilization_max"] == pytest.approx(400.0 / 175.0, abs=1e-4)
        # expert_utilization_min = min(load) / mean(load) = 100 / 175
        assert metrics["moe/expert_utilization_min"] == pytest.approx(100.0 / 175.0, abs=1e-4)

    def test_only_aggregates_no_per_layer(self):
        """Brief mode should NOT produce per-layer keys."""
        layer_loads = _make_layer_loads([[100.0, 200.0], [150.0, 150.0]])
        metrics = compute_brief_metrics(layer_loads)

        assert "moe/cv_mean" in metrics
        assert "moe/layer_0/cv" not in metrics
        assert "moe/layer_1/cv" not in metrics
        assert "moe/layer_0/utilization_mean" not in metrics

    def test_cv_aggregate_stats(self):
        """Brief mode should produce mean, median, min, max for CV."""
        # Layer 0: skewed, Layer 1: uniform
        layer_loads = _make_layer_loads([[400.0, 100.0], [150.0, 150.0]])
        metrics = compute_brief_metrics(layer_loads)

        for suffix in ["_mean", "_median", "_min", "_max"]:
            assert f"moe/cv{suffix}" in metrics

        # Layer 1 is uniform -> cv=0, Layer 0 is skewed -> cv>0
        assert metrics["moe/cv_min"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["moe/cv_max"] > 0.0

    def test_no_imbalance_or_entropy_keys(self):
        """Brief mode should NOT have imbalance or entropy keys."""
        layer_loads = _make_layer_loads([[100.0, 200.0]])
        metrics = compute_brief_metrics(layer_loads)

        assert not any("imbalance" in k for k in metrics)
        assert not any("entropy" in k for k in metrics)

    def test_no_utilization_mean(self):
        """expert_utilization_mean should NOT be present (always 1.0)."""
        layer_loads = _make_layer_loads([[100.0, 200.0, 300.0]])
        metrics = compute_brief_metrics(layer_loads)

        assert "moe/expert_utilization_mean" not in metrics

    def test_expert_utilization_top_k_global(self):
        """Top-K filtering should emit top/bottom K experts globally across all layers."""
        layer_loads = _make_layer_loads([
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 900.0],
            [50.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        ])
        metrics = compute_brief_metrics(layer_loads, top_k=2)

        util_keys = [k for k in metrics if k.startswith("moe_expert_utilization/layer_")]
        assert len(util_keys) <= 4

        # Global top and bottom should be present
        assert "moe_expert_utilization/layer_0_expert_7" in metrics
        assert "moe_expert_utilization/layer_1_expert_0" in metrics

    def test_top_k_caps_total_keys(self):
        """With many layers, total expert utilization keys should be at most 2*top_k."""
        layer_loads = _make_layer_loads([
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
            [110.0, 210.0, 310.0, 410.0, 510.0, 610.0, 710.0, 810.0],
            [120.0, 220.0, 320.0, 420.0, 520.0, 620.0, 720.0, 820.0],
            [130.0, 230.0, 330.0, 430.0, 530.0, 630.0, 730.0, 830.0],
        ])
        metrics = compute_brief_metrics(layer_loads, top_k=3)

        util_keys = [k for k in metrics if k.startswith("moe_expert_utilization/layer_")]
        assert len(util_keys) <= 6

    def test_no_aux_loss(self):
        """When aux_loss is None, aux_loss_mean should be absent."""
        layer_loads = _make_layer_loads([[100.0, 100.0]])
        metrics = compute_brief_metrics(layer_loads)

        assert "moe/aux_loss_mean" not in metrics

    def test_with_aux_loss(self):
        """Aux loss mean should be computed when available."""
        layer_loads = _make_layer_loads(
            [[100.0, 100.0], [200.0, 100.0]],
            aux_losses=[0.1, 0.3],
        )
        metrics = compute_brief_metrics(layer_loads)

        assert metrics["moe/aux_loss_mean"] == pytest.approx(0.2, abs=1e-6)

    def test_zero_load_expert(self):
        """Zero-load expert should give utilization_min=0."""
        layer_loads = _make_layer_loads([[100.0, 0.0, 100.0, 100.0]])
        metrics = compute_brief_metrics(layer_loads)

        assert metrics["moe/expert_utilization_min"] == pytest.approx(0.0, abs=1e-6)
        # max = 100 / 75 (ideal = 300/4 = 75)
        assert metrics["moe/expert_utilization_max"] == pytest.approx(100.0 / 75.0, abs=1e-4)

    def test_percentiles(self):
        """p25 and p75 should reflect the utilization distribution."""
        # 8 experts: loads [0, 100, 200, 300, 400, 500, 600, 700]
        # total=2800, ideal=350, ratios sorted: [0, 2/7, 4/7, 6/7, 8/7, 10/7, 12/7, 14/7]
        # p25 = sorted[2] = 4/7, p75 = sorted[6] = 12/7
        layer_loads = _make_layer_loads([[0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0]])
        metrics = compute_brief_metrics(layer_loads)

        assert metrics["moe/expert_utilization_p25"] == pytest.approx(200.0 / 350.0, abs=1e-4)
        assert metrics["moe/expert_utilization_p75"] == pytest.approx(600.0 / 350.0, abs=1e-4)


class TestComputeDetailedMetrics:
    def test_includes_per_layer_keys(self):
        """Detailed mode should include per-layer CV breakdowns."""
        layer_loads = _make_layer_loads([[100.0, 200.0], [150.0, 150.0]])
        metrics = compute_detailed_metrics(layer_loads)

        assert "moe/layer_0/cv" in metrics
        assert "moe/layer_1/cv" in metrics

    def test_no_per_layer_imbalance_or_entropy(self):
        """Detailed mode should NOT have imbalance or entropy keys."""
        layer_loads = _make_layer_loads([[100.0, 200.0]])
        metrics = compute_detailed_metrics(layer_loads)

        assert not any("imbalance" in k for k in metrics)
        assert not any("entropy" in k for k in metrics)

    def test_includes_cv_aggregates(self):
        """Detailed mode should include mean, median, min, max for CV."""
        layer_loads = _make_layer_loads([[100.0, 200.0]])
        metrics = compute_detailed_metrics(layer_loads)

        for suffix in ["_mean", "_median", "_min", "_max"]:
            assert f"moe/cv{suffix}" in metrics

    def test_includes_utilization_aggregates(self):
        """Detailed mode should include utilization p25, median, p75, min, max."""
        layer_loads = _make_layer_loads([[100.0, 200.0]])
        metrics = compute_detailed_metrics(layer_loads)

        assert "moe/expert_utilization_p25" in metrics
        assert "moe/expert_utilization_median" in metrics
        assert "moe/expert_utilization_p75" in metrics
        assert "moe/expert_utilization_min" in metrics
        assert "moe/expert_utilization_max" in metrics
        assert "moe/expert_utilization_mean" not in metrics

    def test_includes_expert_utilization(self):
        """Detailed mode should include top-K/bottom-K expert utilization."""
        layer_loads = _make_layer_loads([[100.0, 200.0, 300.0, 800.0]])
        metrics = compute_detailed_metrics(layer_loads, top_k=2)

        util_keys = [k for k in metrics if k.startswith("moe_expert_utilization/layer_")]
        assert len(util_keys) <= 4
        assert "moe_expert_utilization/layer_0_expert_3" in metrics
        assert "moe_expert_utilization/layer_0_expert_0" in metrics

    def test_per_layer_values_correct(self):
        """Per-layer CV should match expected values."""
        layer_loads = _make_layer_loads(
            [[100.0, 100.0, 100.0, 100.0]],
            aux_losses=[0.5],
        )
        metrics = compute_detailed_metrics(layer_loads)

        assert metrics["moe/layer_0/cv"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["moe/layer_0/aux_loss"] == pytest.approx(0.5, abs=1e-6)

    def test_includes_per_layer_utilization_mean(self):
        """Detailed mode should include per-layer utilization means."""
        layer_loads = _make_layer_loads([[100.0, 300.0], [200.0, 200.0]])
        metrics = compute_detailed_metrics(layer_loads)

        # Layer 0: ideal=200, ratios=[0.5, 1.5] -> mean=1.0
        assert "moe/layer_0/utilization_mean" in metrics
        assert metrics["moe/layer_0/utilization_mean"] == pytest.approx(1.0, abs=1e-4)
        # Layer 1: ideal=200, ratios=[1.0, 1.0] -> mean=1.0
        assert "moe/layer_1/utilization_mean" in metrics
        assert metrics["moe/layer_1/utilization_mean"] == pytest.approx(1.0, abs=1e-4)


class TestEnableLoadBalanceTracking:
    def test_sets_flag_on_gate_modules(self):
        """enable_load_balance_tracking should set _track_load_balance on all Gates."""
        from nemo_automodel.components.moe.config import MoEConfig
        from nemo_automodel.components.moe.layers import Gate

        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=16,
            inter_dim=32,
            moe_inter_dim=32,
            norm_topk_prob=False,
        )
        gate = Gate(config)
        parent = nn.Module()
        parent.gate = gate
        assert gate._track_load_balance is False

        enable_load_balance_tracking(parent)
        assert gate._track_load_balance is True


class TestCollectExpertLoads:
    def test_returns_correct_structure(self):
        """collect_expert_loads should return data from Gates with stored loads."""
        from nemo_automodel.components.moe.config import MoEConfig
        from nemo_automodel.components.moe.layers import Gate

        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=16,
            inter_dim=32,
            moe_inter_dim=32,
            norm_topk_prob=False,
        )
        gate = Gate(config)
        gate._track_load_balance = True
        gate._last_expert_load = torch.tensor([10.0, 20.0, 30.0, 40.0])
        gate._last_aux_loss = torch.tensor(0.5)

        parent = nn.Module()
        parent.gate = gate

        loads = collect_expert_loads(parent)
        assert len(loads) == 1

        key = list(loads.keys())[0]
        assert "gate" in key
        assert loads[key]["n_experts"] == 4
        assert torch.equal(loads[key]["expert_load"], torch.tensor([10.0, 20.0, 30.0, 40.0]))
        assert loads[key]["aux_loss"].item() == pytest.approx(0.5)

    def test_skips_gates_without_load(self):
        """Gates without stored expert_load should be skipped."""
        from nemo_automodel.components.moe.config import MoEConfig
        from nemo_automodel.components.moe.layers import Gate

        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=16,
            inter_dim=32,
            moe_inter_dim=32,
            norm_topk_prob=False,
        )
        gate = Gate(config)
        gate._track_load_balance = True

        parent = nn.Module()
        parent.gate = gate

        loads = collect_expert_loads(parent)
        assert len(loads) == 0
