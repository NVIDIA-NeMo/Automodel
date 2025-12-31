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

from nemo_automodel.components.models.qwen3_vl_moe.state_dict_adapter import Qwen3VLMoeStateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig


@pytest.fixture
def config():
    cfg = Mock()
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 64
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
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
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def adapter(config, moe_config, backend_config):
    return Qwen3VLMoeStateDictAdapter(config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float32)


class TestInitialization:
    def test_sets_expected_attributes(self, config, moe_config, backend_config):
        adapter = Qwen3VLMoeStateDictAdapter(
            config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float16
        )

        assert adapter.config is config
        assert adapter.moe_config is moe_config
        assert adapter.backend is backend_config
        assert adapter.dtype == torch.float16
        assert adapter._uses_model_prefix is True


class TestToHF:
    def test_creates_expected_gate_and_down_placeholders(self, adapter):
        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_projs": torch.randn(4, 128, 64),
        }

        out = adapter.to_hf(state_dict)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        down_key = "model.language_model.layers.0.mlp.experts.down_proj"

        assert gate_key in out
        assert down_key in out
        assert out[gate_key].shape == (4, 64, 128)
        assert out[down_key].shape == (4, 128, 64)
        assert out[gate_key].dtype == adapter.dtype
        assert out[down_key].dtype == adapter.dtype

    def test_respects_exclude_regex(self, adapter):
        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 128),
            "exclude.me": torch.randn(1),
        }

        out = adapter.to_hf(state_dict, exclude_key_regex=r"^exclude")

        assert "exclude.me" not in out


    def test_aggregates_with_device_mesh_non_dtensor(self, adapter, monkeypatch):
        local_experts = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=adapter.dtype,
        )  # shape: [2, 2, 2]

        # Only experts 1 and 2 live on this rank
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh",
            lambda mesh, n_experts: (1, 3),
        )
        # No distributed init => skip all_gather branch
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["ep"]

        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": local_experts,
        }

        out = adapter.to_hf(state_dict, device_mesh=device_mesh)
        gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        global_gate = out[gate_key]

        assert global_gate.shape == (adapter.moe_config.n_routed_experts, 2, 2)
        # Experts 1 and 2 should be populated from local_experts; others remain zero
        torch.testing.assert_close(global_gate[1:3], local_experts)
        assert torch.all(global_gate[0] == 0)
        assert torch.all(global_gate[3] == 0)


    def test_aggregates_dtensor_path_uses_split_helper(self, adapter, monkeypatch):
        local_slice = torch.tensor([[9.0, 10.0]], dtype=adapter.dtype)  # shape: [1, 2]

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.is_dtensor", lambda tensor: True
        )
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.split_experts_weights_dtensor_aware",
            lambda weight, n_experts: ([local_slice], [2]),
        )
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["ep"]

        state_dict = {
            "model.language_model.layers.0.mlp.experts.down_projs": torch.empty(1, 1, 2),
        }

        out = adapter.to_hf(state_dict, device_mesh=device_mesh)
        down_key = "model.language_model.layers.0.mlp.experts.down_proj"
        global_down = out[down_key]

        assert global_down.shape[0] == adapter.moe_config.n_routed_experts
        torch.testing.assert_close(global_down[2], local_slice)


class TestFromHF:
    def test_detects_model_prefix(self, adapter):
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
        }

        adapter.from_hf(hf_state)

        assert adapter._uses_model_prefix is True

    def test_handles_missing_prefix(self, adapter):
        hf_state = {
            "language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
        }

        out = adapter.from_hf(hf_state)

        assert adapter._uses_model_prefix is False
        assert "language_model.layers.0.mlp.experts.gate_and_up_projs" in out
        assert "language_model.layers.0.mlp.experts.down_projs" in out

    def test_combines_expert_weights_into_native_layout(self, adapter):
        gate_up = torch.randn(4, 32, 64, dtype=torch.float16)
        down = torch.randn(4, 64, 32, dtype=torch.float16)

        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": down,
        }

        out = adapter.from_hf(hf_state)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"
        down_key = "model.language_model.layers.0.mlp.experts.down_projs"

        assert gate_key in out
        assert down_key in out
        torch.testing.assert_close(out[gate_key], gate_up.to(adapter.dtype))
        torch.testing.assert_close(out[down_key], down.to(adapter.dtype))

    def test_converts_dtensor_inputs_to_local(self, monkeypatch, adapter):
        gate_up = torch.randn(4, 16, 32, dtype=torch.float16)
        down = torch.randn(4, 32, 16, dtype=torch.float16)

        class FakeDTensor:
            def __init__(self, data):
                self._data = data

            def to_local(self):
                return self._data

            def __getitem__(self, idx):
                return self._data[idx]

        captured = {"locals": []}

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.is_dtensor",
            lambda tensor: isinstance(tensor, FakeDTensor),
        )

        def fake_create_dtensor(local_tensor, device_mesh, rank):
            captured["locals"].append(local_tensor)
            captured["device_mesh"] = device_mesh
            captured["rank"] = rank
            return local_tensor

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.create_dtensor_from_local",
            fake_create_dtensor,
        )

        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": FakeDTensor(gate_up),
            "model.language_model.layers.0.mlp.experts.down_proj": FakeDTensor(down),
        }

        out = adapter.from_hf(hf_state)

        assert len(captured["locals"]) == 2
        torch.testing.assert_close(captured["locals"][0], gate_up.to(adapter.dtype))
        torch.testing.assert_close(captured["locals"][1], down.to(adapter.dtype))
        assert captured["device_mesh"] is None
        assert captured["rank"] is None

        gate_key = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"
        down_key = "model.language_model.layers.0.mlp.experts.down_projs"
        torch.testing.assert_close(out[gate_key], gate_up.to(adapter.dtype))
        torch.testing.assert_close(out[down_key], down.to(adapter.dtype))


class TestConvertSingleTensorToHf:
    def test_expert_tensor_conversion(self, adapter):
        tensor = torch.randn(4, 16, 32)
        fqn = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        key, value = result[0]
        assert key == "model.language_model.layers.0.mlp.experts.gate_up_proj"
        torch.testing.assert_close(value, tensor.to(adapter.dtype))

    def test_non_expert_tensor_passthrough(self, adapter):
        tensor = torch.randn(16, 16)
        fqn = "model.language_model.layers.0.self_attn.q_proj.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        key, value = result[0]
        assert key == fqn
        assert value is tensor

    def test_exclude_regex_filters_results(self, adapter):
        tensor = torch.randn(16, 16)
        fqn = "exclude.me"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

        assert result == []


