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

import sys
import types
import importlib.util

import pytest
import torch
from unittest.mock import Mock, patch

try:
    import fast_hadamard_transform  # noqa: F401
except ImportError:
    if "fast_hadamard_transform" not in sys.modules:
        mock_hadamard = types.ModuleType("fast_hadamard_transform")
        mock_hadamard.__spec__ = importlib.util.spec_from_loader("fast_hadamard_transform", loader=None)
        mock_hadamard.hadamard_transform = lambda x, scale: x
        sys.modules["fast_hadamard_transform"] = mock_hadamard

from nemo_automodel.components.models.glm_moe_dsa.state_dict_adapter import GlmMoeDsaStateDictAdapter
from nemo_automodel.components.models.glm4_moe.state_dict_adapter import Glm4MoeStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.models.common import BackendConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def config():
    cfg = Mock()
    cfg.num_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_experts = 4
    return cfg


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.001,
        score_func="sigmoid",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def adapter(config, moe_config, backend_config):
    return GlmMoeDsaStateDictAdapter(config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float32)


class TestGlmMoeDsaStateDictAdapterInheritance:
    def test_inherits_from_glm4_moe_adapter(self):
        assert issubclass(GlmMoeDsaStateDictAdapter, Glm4MoeStateDictAdapter)

    def test_has_indexer_non_quantized_keys(self):
        expected = [
            "indexer.k_norm.weight",
            "indexer.k_norm.bias",
            "indexer.weights_proj.weight",
        ]
        assert GlmMoeDsaStateDictAdapter._indexer_non_quantized_keys == expected


class TestConvertSingleTensorToHf:
    def test_expert_tensor_conversion(self, adapter):
        tensor = torch.randn(4, 64, 128)
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts") as mock_convert:
            mock_convert.return_value = [
                ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(64, 64)),
                ("model.layers.0.mlp.experts.0.up_proj.weight", torch.randn(64, 64)),
            ]

            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            mock_convert.assert_called_once_with(fqn, tensor)
            assert len(result) == 2
            assert result[0][0] == "model.layers.0.mlp.experts.0.gate_proj.weight"
            assert result[1][0] == "model.layers.0.mlp.experts.0.up_proj.weight"

    def test_non_expert_tensor_conversion(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.attention.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts") as mock_convert:
            mock_convert.return_value = None

            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert torch.equal(result[0][1], tensor)

    def test_preserves_tensor_identity_for_non_experts(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1] is tensor

    def test_exclude_key_regex(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "exclude_this.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

            assert len(result) == 0

    def test_expert_tensor_with_exclude_regex(self, adapter):
        tensor = torch.randn(4, 64, 128)
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts") as mock_convert:
            mock_convert.return_value = [
                ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(64, 64)),
                ("exclude_me.weight", torch.randn(64, 64)),
            ]

            result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

            assert len(result) == 1
            assert result[0][0] == "model.layers.0.mlp.experts.0.gate_proj.weight"
            assert "exclude_me.weight" not in [k for k, _ in result]

    def test_exclude_key_regex_no_match(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r".*kv_proj.*")

            assert len(result) == 1
            assert result[0][0] == fqn


class TestConvertSingleTensorToHfQuantization:
    def test_quantization_normal_weight(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_a_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == torch.float8_e4m3fn

    def test_quantization_skips_non_weight_keys(self, adapter):
        tensor = torch.randn(64)
        fqn = "model.layers.0.self_attn.q_proj.bias"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype

    def test_quantization_skips_indexer_k_norm_weight(self, adapter):
        tensor = torch.randn(64)
        fqn = "model.layers.0.self_attn.indexer.k_norm.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype

    def test_quantization_skips_indexer_k_norm_bias(self, adapter):
        tensor = torch.randn(64)
        fqn = "model.layers.0.self_attn.indexer.k_norm.bias"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype

    def test_quantization_skips_indexer_weights_proj(self, adapter):
        tensor = torch.randn(64, 128)
        fqn = "model.layers.0.self_attn.indexer.weights_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype

    def test_quantization_applies_to_indexer_linear_weights(self, adapter):
        tensor = torch.randn(64, 128)
        fqn = "model.layers.0.self_attn.indexer.wq_b.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == torch.float8_e4m3fn

    def test_without_quantization_preserves_dtype(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_a_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=False)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype

    def test_quantization_with_exclude_regex(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_a_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(
                fqn, tensor, quantization=True, exclude_key_regex=r".*q_a_proj.*"
            )

            assert len(result) == 0
