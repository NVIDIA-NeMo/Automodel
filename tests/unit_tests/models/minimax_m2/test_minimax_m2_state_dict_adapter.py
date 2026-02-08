# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m2.state_dict_adapter import MiniMaxM2StateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig


@dataclass
class MockMiniMaxM2Config:
    hidden_size: int = 64
    intermediate_size: int = 32
    num_local_experts: int = 4
    torch_dtype: str = "bfloat16"


@pytest.fixture
def config():
    return MockMiniMaxM2Config()


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=32,
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
        force_e_score_correction_bias=True,
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
    return MiniMaxM2StateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)


def _make_hf_moe_state(n_experts: int, inter_dim: int, dim: int) -> dict[str, torch.Tensor]:
    sd = {}
    for e in range(n_experts):
        sd[f"model.layers.0.block_sparse_moe.experts.{e}.w1.weight"] = torch.randn(inter_dim, dim)
        sd[f"model.layers.0.block_sparse_moe.experts.{e}.w3.weight"] = torch.randn(inter_dim, dim)
        sd[f"model.layers.0.block_sparse_moe.experts.{e}.w2.weight"] = torch.randn(dim, inter_dim)
    return sd


class TestMiniMaxM2StateDictAdapterFromHF:
    def test_converts_expert_weights(self, adapter, moe_config):
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim
        inter_dim = moe_config.moe_inter_dim

        hf_state_dict = _make_hf_moe_state(n_experts, inter_dim, dim)
        native_state_dict = adapter.from_hf(hf_state_dict)

        gate_up_key = "model.layers.0.mlp.experts.gate_and_up_projs"
        down_key = "model.layers.0.mlp.experts.down_projs"
        assert gate_up_key in native_state_dict
        assert down_key in native_state_dict

        gate_up = native_state_dict[gate_up_key]
        down = native_state_dict[down_key]
        assert gate_up.shape == (n_experts, dim, 2 * inter_dim)
        assert down.shape == (n_experts, inter_dim, dim)

    def test_maps_gate_and_bias_keys(self, adapter):
        gate_w = torch.randn(4, 64)
        corr_b = torch.randn(4)
        hf_state_dict = {
            "model.layers.0.block_sparse_moe.gate.weight": gate_w,
            "model.layers.0.block_sparse_moe.e_score_correction_bias": corr_b,
        }

        native_state_dict = adapter.from_hf(hf_state_dict)
        assert "model.layers.0.mlp.gate.weight" in native_state_dict
        assert "model.layers.0.mlp.gate.e_score_correction_bias" in native_state_dict

    def test_detects_no_model_prefix(self, adapter, moe_config):
        hf_state_dict = {
            "layers.0.block_sparse_moe.experts.0.w1.weight": torch.randn(moe_config.moe_inter_dim, moe_config.dim),
            "layers.0.block_sparse_moe.experts.0.w3.weight": torch.randn(moe_config.moe_inter_dim, moe_config.dim),
            "layers.0.block_sparse_moe.experts.0.w2.weight": torch.randn(moe_config.dim, moe_config.moe_inter_dim),
            "layers.0.block_sparse_moe.experts.1.w1.weight": torch.randn(moe_config.moe_inter_dim, moe_config.dim),
            "layers.0.block_sparse_moe.experts.1.w3.weight": torch.randn(moe_config.moe_inter_dim, moe_config.dim),
            "layers.0.block_sparse_moe.experts.1.w2.weight": torch.randn(moe_config.dim, moe_config.moe_inter_dim),
            "layers.0.block_sparse_moe.experts.2.w1.weight": torch.randn(moe_config.moe_inter_dim, moe_config.dim),
            "layers.0.block_sparse_moe.experts.2.w3.weight": torch.randn(moe_config.moe_inter_dim, moe_config.dim),
            "layers.0.block_sparse_moe.experts.2.w2.weight": torch.randn(moe_config.dim, moe_config.moe_inter_dim),
            "layers.0.block_sparse_moe.experts.3.w1.weight": torch.randn(moe_config.moe_inter_dim, moe_config.dim),
            "layers.0.block_sparse_moe.experts.3.w3.weight": torch.randn(moe_config.moe_inter_dim, moe_config.dim),
            "layers.0.block_sparse_moe.experts.3.w2.weight": torch.randn(moe_config.dim, moe_config.moe_inter_dim),
        }
        native_state_dict = adapter.from_hf(hf_state_dict)
        assert "layers.0.mlp.experts.gate_and_up_projs" in native_state_dict


class TestMiniMaxM2StateDictAdapterToHF:
    def test_converts_back_to_minimax_keys(self, adapter, moe_config):
        n_experts = moe_config.n_routed_experts
        dim = moe_config.dim
        inter_dim = moe_config.moe_inter_dim

        native_state_dict = {
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(n_experts, dim, 2 * inter_dim),
            "model.layers.0.mlp.experts.down_projs": torch.randn(n_experts, inter_dim, dim),
            "model.layers.0.mlp.gate.weight": torch.randn(n_experts, dim),
            "model.layers.0.mlp.gate.e_score_correction_bias": torch.randn(n_experts),
        }

        hf_state_dict = adapter.to_hf(native_state_dict)

        assert "model.layers.0.block_sparse_moe.gate.weight" in hf_state_dict
        assert "model.layers.0.block_sparse_moe.e_score_correction_bias" in hf_state_dict
        assert "model.layers.0.block_sparse_moe.experts.0.w1.weight" in hf_state_dict
        assert "model.layers.0.block_sparse_moe.experts.0.w3.weight" in hf_state_dict
        assert "model.layers.0.block_sparse_moe.experts.0.w2.weight" in hf_state_dict
