# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.afmoe.config import AfmoeConfig
from nemo_automodel.components.models.afmoe.state_dict_adapter import AfmoeStateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def config():
    cfg = Mock(spec=AfmoeConfig)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 32
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.num_shared_experts = 1
    cfg.num_dense_layers = 1
    return cfg


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=128,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.001,
        score_func="sigmoid",
        route_scale=2.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        force_e_score_correction_bias=True,
        shared_expert_inter_dim=32,
    )


@pytest.fixture
def backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
    )


@pytest.fixture
def adapter(config, moe_config, backend):
    return AfmoeStateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)


def _make_hf_expert_state_dict(n_layers=2, n_experts=4, hidden=64, moe_inter=32, num_dense=1, dtype=torch.bfloat16):
    """Create a minimal HF-format state dict with router, experts, and expert_bias."""
    sd = {}
    for layer_idx in range(n_layers):
        prefix = f"model.layers.{layer_idx}"
        if layer_idx >= num_dense:
            # Router gate
            sd[f"{prefix}.mlp.router.gate.weight"] = torch.randn(n_experts, hidden, dtype=dtype)
            # Expert bias
            sd[f"{prefix}.mlp.expert_bias"] = torch.zeros(n_experts)
            # Per-expert weights
            for e in range(n_experts):
                sd[f"{prefix}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(moe_inter, hidden, dtype=dtype)
                sd[f"{prefix}.mlp.experts.{e}.up_proj.weight"] = torch.randn(moe_inter, hidden, dtype=dtype)
                sd[f"{prefix}.mlp.experts.{e}.down_proj.weight"] = torch.randn(hidden, moe_inter, dtype=dtype)
            # Shared expert
            sd[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = torch.randn(moe_inter, hidden, dtype=dtype)
            sd[f"{prefix}.mlp.shared_experts.up_proj.weight"] = torch.randn(moe_inter, hidden, dtype=dtype)
            sd[f"{prefix}.mlp.shared_experts.down_proj.weight"] = torch.randn(hidden, moe_inter, dtype=dtype)
        else:
            # Dense MLP
            sd[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(128, hidden, dtype=dtype)
            sd[f"{prefix}.mlp.up_proj.weight"] = torch.randn(128, hidden, dtype=dtype)
            sd[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden, 128, dtype=dtype)
    return sd


class TestAfmoeStateDictAdapter:
    def test_router_key_renamed_from_hf(self, adapter):
        hf_sd = _make_hf_expert_state_dict()
        nemo_sd = adapter.from_hf(hf_sd)

        # Router gate should be renamed
        assert "model.layers.1.mlp.gate.weight" in nemo_sd
        assert "model.layers.1.mlp.router.gate.weight" not in nemo_sd

    def test_expert_bias_renamed_from_hf(self, adapter):
        hf_sd = _make_hf_expert_state_dict()
        nemo_sd = adapter.from_hf(hf_sd)

        assert "model.layers.1.mlp.gate.e_score_correction_bias" in nemo_sd
        assert "model.layers.1.mlp.expert_bias" not in nemo_sd

    def test_experts_merged_from_hf(self, adapter):
        hf_sd = _make_hf_expert_state_dict()
        nemo_sd = adapter.from_hf(hf_sd)

        # Per-expert keys should be merged into grouped format
        assert "model.layers.1.mlp.experts.gate_and_up_projs" in nemo_sd
        assert "model.layers.1.mlp.experts.down_projs" in nemo_sd
        # Individual expert keys should be gone
        assert "model.layers.1.mlp.experts.0.gate_proj.weight" not in nemo_sd

    def test_experts_merged_shape(self, adapter):
        hf_sd = _make_hf_expert_state_dict()
        nemo_sd = adapter.from_hf(hf_sd)

        gate_up = nemo_sd["model.layers.1.mlp.experts.gate_and_up_projs"]
        down = nemo_sd["model.layers.1.mlp.experts.down_projs"]
        # gate_and_up: [n_experts, dim, 2*moe_inter]
        assert gate_up.shape == (4, 64, 64)  # 4 experts, dim=64, 2*32=64
        # down: [n_experts, moe_inter, dim]
        assert down.shape == (4, 32, 64)

    def test_shared_experts_pass_through(self, adapter):
        hf_sd = _make_hf_expert_state_dict()
        nemo_sd = adapter.from_hf(hf_sd)

        # Shared experts should pass through unchanged
        assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in nemo_sd
        assert "model.layers.1.mlp.shared_experts.up_proj.weight" in nemo_sd
        assert "model.layers.1.mlp.shared_experts.down_proj.weight" in nemo_sd

    def test_dense_mlp_pass_through(self, adapter):
        hf_sd = _make_hf_expert_state_dict()
        nemo_sd = adapter.from_hf(hf_sd)

        # Dense layer (layer 0) MLP keys pass through
        assert "model.layers.0.mlp.gate_proj.weight" in nemo_sd
        assert "model.layers.0.mlp.up_proj.weight" in nemo_sd
        assert "model.layers.0.mlp.down_proj.weight" in nemo_sd

    def test_to_hf_reverses_router_rename(self, adapter):
        nemo_sd = {
            "model.layers.1.mlp.gate.weight": torch.randn(4, 64),
            "model.layers.1.mlp.gate.e_score_correction_bias": torch.zeros(4),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
        }
        hf_sd = adapter.to_hf(nemo_sd)

        assert "model.layers.1.mlp.router.gate.weight" in hf_sd
        assert "model.layers.1.mlp.expert_bias" in hf_sd
        assert "model.layers.0.mlp.gate_proj.weight" in hf_sd

    def test_to_hf_splits_experts(self, adapter):
        nemo_sd = {
            "model.layers.1.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 64),
            "model.layers.1.mlp.experts.down_projs": torch.randn(4, 32, 64),
        }
        hf_sd = adapter.to_hf(nemo_sd)

        for e in range(4):
            assert f"model.layers.1.mlp.experts.{e}.gate_proj.weight" in hf_sd
            assert f"model.layers.1.mlp.experts.{e}.up_proj.weight" in hf_sd
            assert f"model.layers.1.mlp.experts.{e}.down_proj.weight" in hf_sd

    def test_roundtrip_preserves_all_values(self, adapter):
        """HF -> NeMo -> HF round-trip must preserve exact tensor values."""
        torch.manual_seed(42)
        hf_sd = _make_hf_expert_state_dict()
        originals = {k: v.clone() for k, v in hf_sd.items()}

        nemo_sd = adapter.from_hf(hf_sd)
        roundtrip_sd = adapter.to_hf(nemo_sd)

        assert set(roundtrip_sd.keys()) == set(originals.keys()), (
            f"Missing: {set(originals.keys()) - set(roundtrip_sd.keys())}, "
            f"Extra: {set(roundtrip_sd.keys()) - set(originals.keys())}"
        )
        for key in originals:
            max_diff = (originals[key].float() - roundtrip_sd[key].float()).abs().max().item()
            assert max_diff == 0.0, f"Round-trip mismatch for {key}: max_diff={max_diff}"
