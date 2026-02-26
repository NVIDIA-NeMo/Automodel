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

from dataclasses import dataclass

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p5.state_dict_adapter import Step3p5StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


@dataclass
class MockStep3p5Config:
    """Mock configuration for Step3p5 model."""
    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    moe_num_experts: int = 4
    moe_intermediate_size: int = 32
    torch_dtype: str = "bfloat16"


@pytest.fixture
def config():
    return MockStep3p5Config()


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=128,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
    )


@pytest.fixture
def backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


@pytest.fixture
def adapter(config, moe_config, backend):
    return Step3p5StateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)


class TestStep3p5StateDictAdapterFromHF:
    def test_converts_grouped_gate_up_proj(self, adapter, moe_config):
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim
        inter_dim = moe_config.moe_inter_dim

        # HF format: [n_exp, inter, dim]
        hf_gate = torch.randn(n_experts, inter_dim, dim)
        hf_up = torch.randn(n_experts, inter_dim, dim)

        hf_state_dict = {
            "model.layers.0.moe.gate_proj.weight": hf_gate,
            "model.layers.0.moe.up_proj.weight": hf_up,
        }

        native_state_dict = adapter.from_hf(hf_state_dict)

        # Native format: [n_exp, dim, 2*inter]
        assert "model.layers.0.moe.experts.gate_and_up_projs" in native_state_dict
        gate_and_up = native_state_dict["model.layers.0.moe.experts.gate_and_up_projs"]
        assert gate_and_up.shape == (n_experts, dim, 2 * inter_dim)

        # Verify content: gate_t = hf_gate.T, up_t = hf_up.T, concat
        expected_gate_t = hf_gate.transpose(1, 2)  # [n_exp, dim, inter]
        expected_up_t = hf_up.transpose(1, 2)  # [n_exp, dim, inter]
        expected = torch.cat([expected_gate_t, expected_up_t], dim=-1)
        torch.testing.assert_close(gate_and_up, expected.to(torch.bfloat16))

    def test_router_gate_passes_through(self, adapter, moe_config):
        """Test that router gate weight passes through unchanged."""
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim

        router_weight = torch.randn(n_experts, dim)

        hf_state_dict = {
            "model.layers.0.moe.gate.weight": router_weight,
        }

        native_state_dict = adapter.from_hf(hf_state_dict)

        assert "model.layers.0.moe.gate.weight" in native_state_dict
        torch.testing.assert_close(
            native_state_dict["model.layers.0.moe.gate.weight"],
            router_weight,
        )

    def test_router_bias_mapping(self, adapter, moe_config):
        """Test that HF router_bias is mapped to native gate.bias."""
        n_experts = moe_config.n_routed_experts

        router_bias = torch.randn(n_experts)

        hf_state_dict = {
            "model.layers.0.moe.router_bias": router_bias,
        }

        native_state_dict = adapter.from_hf(hf_state_dict)

        # HF router_bias should be mapped to gate.bias
        assert "model.layers.0.moe.gate.bias" in native_state_dict
        torch.testing.assert_close(
            native_state_dict["model.layers.0.moe.gate.bias"],
            router_bias,
        )

    def test_shared_expert_passes_through(self, adapter):
        """Test that shared expert weights pass through unchanged."""
        gate_proj = torch.randn(128, 64)
        up_proj = torch.randn(128, 64)
        down_proj = torch.randn(64, 128)

        hf_state_dict = {
            "model.layers.0.share_expert.gate_proj.weight": gate_proj,
            "model.layers.0.share_expert.up_proj.weight": up_proj,
            "model.layers.0.share_expert.down_proj.weight": down_proj,
        }

        native_state_dict = adapter.from_hf(hf_state_dict)

        assert "model.layers.0.share_expert.gate_proj.weight" in native_state_dict
        assert "model.layers.0.share_expert.up_proj.weight" in native_state_dict
        assert "model.layers.0.share_expert.down_proj.weight" in native_state_dict
        torch.testing.assert_close(
            native_state_dict["model.layers.0.share_expert.gate_proj.weight"],
            gate_proj,
        )

    def test_converts_grouped_down_proj(self, adapter, moe_config):
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim
        inter_dim = moe_config.moe_inter_dim

        # HF format: [n_exp, dim, inter]
        hf_down = torch.randn(n_experts, dim, inter_dim)

        hf_state_dict = {
            "model.layers.0.moe.down_proj.weight": hf_down,
        }

        native_state_dict = adapter.from_hf(hf_state_dict)

        # Native format: [n_exp, inter, dim]
        assert "model.layers.0.moe.experts.down_projs" in native_state_dict
        down_projs = native_state_dict["model.layers.0.moe.experts.down_projs"]
        assert down_projs.shape == (n_experts, inter_dim, dim)

        # Verify content
        expected = hf_down.transpose(1, 2)
        torch.testing.assert_close(down_projs, expected.to(torch.bfloat16))

    def test_non_moe_weights_pass_through(self, adapter):
        other_weight = torch.randn(64, 64)
        hf_state_dict = {
            "model.layers.0.input_layernorm.weight": other_weight,
        }

        native_state_dict = adapter.from_hf(hf_state_dict)

        assert "model.layers.0.input_layernorm.weight" in native_state_dict
        torch.testing.assert_close(
            native_state_dict["model.layers.0.input_layernorm.weight"],
            other_weight,
        )

    def test_detects_model_prefix(self, adapter, moe_config):
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim
        inter_dim = moe_config.moe_inter_dim

        hf_state_dict = {
            "layers.0.moe.gate_proj.weight": torch.randn(n_experts, inter_dim, dim),
            "layers.0.moe.up_proj.weight": torch.randn(n_experts, inter_dim, dim),
        }

        native_state_dict = adapter.from_hf(hf_state_dict)

        # Should detect no model prefix and use that format
        assert "layers.0.moe.experts.gate_and_up_projs" in native_state_dict

    def test_handles_multiple_layers(self, adapter, moe_config):
        """Test from_hf correctly handles multiple MoE layers."""
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim
        inter_dim = moe_config.moe_inter_dim

        # Create HF state dict with two layers
        hf_state_dict = {
            "model.layers.0.moe.gate_proj.weight": torch.randn(n_experts, inter_dim, dim),
            "model.layers.0.moe.up_proj.weight": torch.randn(n_experts, inter_dim, dim),
            "model.layers.0.moe.down_proj.weight": torch.randn(n_experts, dim, inter_dim),
            "model.layers.1.moe.gate_proj.weight": torch.randn(n_experts, inter_dim, dim),
            "model.layers.1.moe.up_proj.weight": torch.randn(n_experts, inter_dim, dim),
            "model.layers.1.moe.down_proj.weight": torch.randn(n_experts, dim, inter_dim),
        }

        native_state_dict = adapter.from_hf(hf_state_dict)

        # Both layers should be converted
        assert "model.layers.0.moe.experts.gate_and_up_projs" in native_state_dict
        assert "model.layers.0.moe.experts.down_projs" in native_state_dict
        assert "model.layers.1.moe.experts.gate_and_up_projs" in native_state_dict
        assert "model.layers.1.moe.experts.down_projs" in native_state_dict

        # Verify shapes
        assert native_state_dict["model.layers.0.moe.experts.gate_and_up_projs"].shape == (n_experts, dim, 2 * inter_dim)
        assert native_state_dict["model.layers.1.moe.experts.gate_and_up_projs"].shape == (n_experts, dim, 2 * inter_dim)


class TestStep3p5StateDictAdapterToHF:
    def test_converts_gate_and_up_projs(self, adapter, moe_config):
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim
        inter_dim = moe_config.moe_inter_dim

        # Native format: [n_exp, dim, 2*inter]
        gate_and_up = torch.randn(n_experts, dim, 2 * inter_dim)
        native_state_dict = {
            "model.layers.0.moe.experts.gate_and_up_projs": gate_and_up,
        }

        hf_state_dict = adapter.to_hf(native_state_dict)

        # HF format: [n_exp, inter, dim] for gate and up
        assert "model.layers.0.moe.gate_proj.weight" in hf_state_dict
        assert "model.layers.0.moe.up_proj.weight" in hf_state_dict

        gate_weight = hf_state_dict["model.layers.0.moe.gate_proj.weight"]
        up_weight = hf_state_dict["model.layers.0.moe.up_proj.weight"]

        assert gate_weight.shape == (n_experts, inter_dim, dim)
        assert up_weight.shape == (n_experts, inter_dim, dim)

    def test_converts_down_projs(self, adapter, moe_config):
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim
        inter_dim = moe_config.moe_inter_dim

        # Native format: [n_exp, inter, dim]
        down_projs = torch.randn(n_experts, inter_dim, dim)
        native_state_dict = {
            "model.layers.0.moe.experts.down_projs": down_projs,
        }

        hf_state_dict = adapter.to_hf(native_state_dict)

        # HF format: [n_exp, dim, inter]
        assert "model.layers.0.moe.down_proj.weight" in hf_state_dict
        down_weight = hf_state_dict["model.layers.0.moe.down_proj.weight"]
        assert down_weight.shape == (n_experts, dim, inter_dim)

    def test_maps_gate_bias_to_router_bias(self, adapter, moe_config):
        """Test that native gate.bias is mapped to HF router_bias."""
        n_experts = moe_config.n_routed_experts

        gate_bias = torch.randn(n_experts)
        native_state_dict = {
            "model.layers.0.moe.gate.bias": gate_bias,
        }

        hf_state_dict = adapter.to_hf(native_state_dict)

        # Native gate.bias should be mapped to HF router_bias
        assert "model.layers.0.moe.router_bias" in hf_state_dict
        torch.testing.assert_close(
            hf_state_dict["model.layers.0.moe.router_bias"],
            gate_bias,
        )


class TestStep3p5StateDictAdapterRoundtrip:
    def test_roundtrip_preserves_weights(self, adapter, moe_config):
        """Test that HF -> Native -> HF preserves weights."""
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim
        inter_dim = moe_config.moe_inter_dim

        # Original HF format
        original_gate = torch.randn(n_experts, inter_dim, dim)
        original_up = torch.randn(n_experts, inter_dim, dim)
        original_down = torch.randn(n_experts, dim, inter_dim)

        hf_state_dict = {
            "model.layers.0.moe.gate_proj.weight": original_gate,
            "model.layers.0.moe.up_proj.weight": original_up,
            "model.layers.0.moe.down_proj.weight": original_down,
        }

        # HF -> Native
        native_state_dict = adapter.from_hf(hf_state_dict)

        # Native -> HF
        recovered_hf = adapter.to_hf(native_state_dict)

        # Verify recovered matches original (allowing for dtype conversion)
        torch.testing.assert_close(
            recovered_hf["model.layers.0.moe.gate_proj.weight"],
            original_gate.to(torch.bfloat16),
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(
            recovered_hf["model.layers.0.moe.up_proj.weight"],
            original_up.to(torch.bfloat16),
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(
            recovered_hf["model.layers.0.moe.down_proj.weight"],
            original_down.to(torch.bfloat16),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_roundtrip_preserves_router_bias(self, adapter, moe_config):
        """Test that HF -> Native -> HF preserves router bias."""
        n_experts = moe_config.n_routed_experts

        original_bias = torch.randn(n_experts)

        hf_state_dict = {
            "model.layers.0.moe.router_bias": original_bias,
        }

        # HF -> Native
        native_state_dict = adapter.from_hf(hf_state_dict)
        assert "model.layers.0.moe.gate.bias" in native_state_dict

        # Native -> HF
        recovered_hf = adapter.to_hf(native_state_dict)

        # Verify recovered matches original
        torch.testing.assert_close(
            recovered_hf["model.layers.0.moe.router_bias"],
            original_bias,
        )

    def test_roundtrip_preserves_non_moe(self, adapter):
        """Test that HF -> Native -> HF preserves non-MoE weights."""
        embed_weight = torch.randn(128, 64)
        norm_weight = torch.randn(64)
        attn_weight = torch.randn(64, 64)

        hf_state_dict = {
            "model.embed_tokens.weight": embed_weight,
            "model.layers.0.input_layernorm.weight": norm_weight,
            "model.layers.0.self_attn.q_proj.weight": attn_weight,
        }

        # HF -> Native
        native_state_dict = adapter.from_hf(hf_state_dict)

        # Native -> HF
        recovered_hf = adapter.to_hf(native_state_dict)

        # Verify all keys preserved
        torch.testing.assert_close(
            recovered_hf["model.embed_tokens.weight"],
            embed_weight,
        )
        torch.testing.assert_close(
            recovered_hf["model.layers.0.input_layernorm.weight"],
            norm_weight,
        )
        torch.testing.assert_close(
            recovered_hf["model.layers.0.self_attn.q_proj.weight"],
            attn_weight,
        )


class TestStep3p5StateDictAdapterDTensorAware:
    """Tests for DTensor-aware behavior of the adapter."""

    def test_to_hf_handles_regular_tensors(self, adapter, moe_config):
        """Test that to_hf works correctly with regular (non-DTensor) tensors."""
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim
        inter_dim = moe_config.moe_inter_dim

        # Native format tensors
        gate_and_up = torch.randn(n_experts, dim, 2 * inter_dim)
        down = torch.randn(n_experts, inter_dim, dim)

        native_state_dict = {
            "model.layers.0.moe.experts.gate_and_up_projs": gate_and_up,
            "model.layers.0.moe.experts.down_projs": down,
        }

        # Convert to HF
        hf_state_dict = adapter.to_hf(native_state_dict)

        # Verify shapes match HF format
        assert hf_state_dict["model.layers.0.moe.gate_proj.weight"].shape == (n_experts, inter_dim, dim)
        assert hf_state_dict["model.layers.0.moe.up_proj.weight"].shape == (n_experts, inter_dim, dim)
        assert hf_state_dict["model.layers.0.moe.down_proj.weight"].shape == (n_experts, dim, inter_dim)

    def test_from_hf_without_device_mesh(self, adapter, moe_config):
        """Test from_hf without device_mesh loads all experts."""
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim
        inter_dim = moe_config.moe_inter_dim

        # HF format tensors (full experts)
        hf_gate = torch.randn(n_experts, inter_dim, dim)
        hf_up = torch.randn(n_experts, inter_dim, dim)
        hf_down = torch.randn(n_experts, dim, inter_dim)

        hf_state_dict = {
            "model.layers.0.moe.gate_proj.weight": hf_gate,
            "model.layers.0.moe.up_proj.weight": hf_up,
            "model.layers.0.moe.down_proj.weight": hf_down,
        }

        # Convert without device_mesh (no EP)
        native_state_dict = adapter.from_hf(hf_state_dict, device_mesh=None)

        # Verify all experts are loaded
        gate_and_up = native_state_dict["model.layers.0.moe.experts.gate_and_up_projs"]
        down = native_state_dict["model.layers.0.moe.experts.down_projs"]

        assert gate_and_up.shape == (n_experts, dim, 2 * inter_dim)
        assert down.shape == (n_experts, inter_dim, dim)

    def test_passthrough_keys_preserved(self, adapter):
        """Test that non-MoE keys pass through unchanged."""
        embed_weight = torch.randn(128, 64)
        norm_weight = torch.randn(64)

        native_state_dict = {
            "model.embed_tokens.weight": embed_weight,
            "model.layers.0.input_layernorm.weight": norm_weight,
        }

        hf_state_dict = adapter.to_hf(native_state_dict)

        # Keys should be preserved
        assert "model.embed_tokens.weight" in hf_state_dict
        assert "model.layers.0.input_layernorm.weight" in hf_state_dict

        # Values should match
        torch.testing.assert_close(hf_state_dict["model.embed_tokens.weight"], embed_weight)
        torch.testing.assert_close(hf_state_dict["model.layers.0.input_layernorm.weight"], norm_weight)
