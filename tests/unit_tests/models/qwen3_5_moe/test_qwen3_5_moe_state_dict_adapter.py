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

from nemo_automodel.components.models.qwen3_5_moe.state_dict_adapter import Qwen3_5MoeStateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.models.common import BackendConfig


@pytest.fixture
def config():
    cfg = Mock()
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 64
    cfg.shared_expert_intermediate_size = 64
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    return cfg


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=64,
        moe_inter_dim=64,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.001,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=64,
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
    return Qwen3_5MoeStateDictAdapter(config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
class TestInitialization:
    def test_sets_expected_attributes(self, config, moe_config, backend_config):
        adapter = Qwen3_5MoeStateDictAdapter(
            config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float16
        )

        assert adapter.config is config
        assert adapter.moe_config is moe_config
        assert adapter.backend is backend_config
        assert adapter.dtype == torch.float16
        assert adapter._uses_model_prefix is True

    def test_key_mappings_are_consistent(self, adapter):
        assert ".mlp.shared_expert." in adapter.hf_to_internal_map
        assert adapter.hf_to_internal_map[".mlp.shared_expert."] == ".mlp.shared_experts."
        # reverse mapping should be the inverse
        assert ".mlp.shared_experts." in adapter.internal_to_hf_map
        assert adapter.internal_to_hf_map[".mlp.shared_experts."] == ".mlp.shared_expert."


# ---------------------------------------------------------------------------
# _apply_key_mapping
# ---------------------------------------------------------------------------
class TestApplyKeyMapping:
    def test_shared_expert_to_shared_experts(self, adapter):
        state_dict = {
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(64, 32),
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(64, 32),
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(32, 64),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        }

        out = adapter._apply_key_mapping(state_dict, adapter.hf_to_internal_map)

        assert "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_experts.up_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_experts.down_proj.weight" in out
        # Original singular keys removed
        assert "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight" not in out
        # Unrelated keys untouched
        assert "model.language_model.layers.0.self_attn.q_proj.weight" in out

    def test_reverse_mapping(self, adapter):
        state_dict = {
            "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(64, 32),
        }

        out = adapter._apply_key_mapping(state_dict, adapter.internal_to_hf_map)

        assert "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight" not in out

    def test_multiple_layers(self, adapter):
        state_dict = {
            f"model.language_model.layers.{i}.mlp.shared_expert.gate_proj.weight": torch.randn(64, 32)
            for i in range(3)
        }

        out = adapter._apply_key_mapping(state_dict, adapter.hf_to_internal_map)

        for i in range(3):
            assert f"model.language_model.layers.{i}.mlp.shared_experts.gate_proj.weight" in out
            assert f"model.language_model.layers.{i}.mlp.shared_expert.gate_proj.weight" not in out


# ---------------------------------------------------------------------------
# to_hf
# ---------------------------------------------------------------------------
class TestToHF:
    def test_converts_aggregated_experts_with_transpose(self, adapter):
        # NeMo layout: [n_experts, hidden, moe_inter] (gate_and_up_projs)
        gate_up = torch.randn(4, 64, 128)
        down = torch.randn(4, 64, 64)

        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": gate_up,
            "model.language_model.layers.0.mlp.experts.down_projs": down,
        }

        out = adapter.to_hf(state_dict)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        down_key = "model.language_model.layers.0.mlp.experts.down_proj"

        assert gate_key in out
        assert down_key in out
        # Should be transposed(1,2) relative to input
        assert out[gate_key].shape == (4, 128, 64)
        assert out[down_key].shape == (4, 64, 64)

    def test_maps_shared_experts_to_shared_expert(self, adapter):
        state_dict = {
            "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(64, 32),
            "model.language_model.layers.0.mlp.shared_experts.up_proj.weight": torch.randn(64, 32),
        }

        out = adapter.to_hf(state_dict)

        assert "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_expert.up_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight" not in out

    def test_respects_exclude_regex(self, adapter):
        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 128),
            "exclude.me": torch.randn(1),
        }

        out = adapter.to_hf(state_dict, exclude_key_regex=r"^exclude")

        assert "exclude.me" not in out

    def test_passthrough_non_expert_keys(self, adapter):
        tensor = torch.randn(64, 64)
        state_dict = {
            "model.language_model.layers.0.self_attn.q_proj.weight": tensor,
        }

        out = adapter.to_hf(state_dict)

        assert "model.language_model.layers.0.self_attn.q_proj.weight" in out
        assert out["model.language_model.layers.0.self_attn.q_proj.weight"] is tensor

    def test_aggregates_with_device_mesh_non_dtensor(self, adapter, monkeypatch):
        local_experts = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=adapter.dtype,
        )  # shape: [2, 2, 2]

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh",
            lambda mesh, n_experts: (1, 3),
        )
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
        # Experts 1 and 2 should be populated (transposed from local_experts); others zero
        torch.testing.assert_close(global_gate[1:3], local_experts.transpose(1, 2))
        assert torch.all(global_gate[0] == 0)
        assert torch.all(global_gate[3] == 0)

    def test_aggregates_dtensor_path_uses_split_helper(self, adapter, monkeypatch):
        local_slice = torch.tensor([[9.0, 10.0]], dtype=adapter.dtype)

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
        # The global tensor is transposed(1,2) after gathering, so local_slice [1,2] becomes [2,1]
        torch.testing.assert_close(global_down[2], local_slice.T)

    def test_all_gather_path_populates_global_tensor(self, adapter, monkeypatch):
        local_experts = torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ],
            dtype=adapter.dtype,
        )  # shape: [2, 1, 1]

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["ep"]
        device_mesh.get_group = lambda dim: "ep_group" if dim == 0 else None

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh",
            lambda mesh, n_experts: (0, 2),
        )
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
        monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)

        def fake_all_gather_object(gathered, payload, group=None):
            gathered[0] = payload
            other_weights = [torch.tensor([[3.0]], dtype=adapter.dtype), torch.tensor([[4.0]], dtype=adapter.dtype)]
            gathered[1] = ([2, 3], other_weights)

        monkeypatch.setattr("torch.distributed.all_gather_object", fake_all_gather_object)

        state_dict = {"model.language_model.layers.0.mlp.experts.gate_and_up_projs": local_experts}
        out = adapter.to_hf(state_dict, device_mesh=device_mesh)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        global_gate = out[gate_key]

        assert global_gate.shape == (adapter.moe_config.n_routed_experts, 1, 1)
        # After transpose(1,2) and gather, all experts should be populated
        torch.testing.assert_close(global_gate[0], torch.tensor([[1.0]], dtype=adapter.dtype))
        torch.testing.assert_close(global_gate[1], torch.tensor([[2.0]], dtype=adapter.dtype))
        torch.testing.assert_close(global_gate[2], torch.tensor([[3.0]], dtype=adapter.dtype))
        torch.testing.assert_close(global_gate[3], torch.tensor([[4.0]], dtype=adapter.dtype))


# ---------------------------------------------------------------------------
# from_hf
# ---------------------------------------------------------------------------
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

    def test_combines_expert_weights_with_transpose(self, adapter):
        # HF layout: [n_experts, moe_inter, hidden]
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
        # Should be transposed(1,2) to NeMo layout
        torch.testing.assert_close(out[gate_key], gate_up.transpose(1, 2).to(adapter.dtype))
        torch.testing.assert_close(out[down_key], down.transpose(1, 2).to(adapter.dtype))

    def test_maps_shared_expert_to_shared_experts(self, adapter):
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(64, 32),
        }

        out = adapter.from_hf(hf_state)

        assert "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight" not in out

    def test_raises_when_no_expert_keys(self, adapter):
        hf_state = {
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        }

        with pytest.raises(RuntimeError, match="Expected aggregated expert weights"):
            adapter.from_hf(hf_state)

    def test_skips_scale_inv_keys(self, adapter):
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
            "model.language_model.layers.0.mlp.experts.gate_up_proj_scale_inv": torch.randn(4),
        }

        out = adapter.from_hf(hf_state)

        assert not any(k.endswith("_scale_inv") for k in out.keys())

    def test_passthrough_non_expert_keys(self, adapter):
        tensor = torch.randn(64, 64)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
            "model.language_model.layers.0.self_attn.q_proj.weight": tensor,
        }

        out = adapter.from_hf(hf_state)

        assert "model.language_model.layers.0.self_attn.q_proj.weight" in out

    def test_expert_parallel_sharding(self, adapter, monkeypatch):
        """When device_mesh is provided, from_hf should slice experts by rank."""
        gate_up = torch.randn(4, 32, 64)
        down = torch.randn(4, 64, 32)

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["ep"]

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh",
            lambda mesh, n_experts: (1, 3),
        )
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_submesh",
            lambda mesh, dims: Mock(get_rank=lambda: 0),
        )

        def fake_create_dtensor(local_tensor, mesh, rank):
            return local_tensor

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.create_dtensor_from_local",
            fake_create_dtensor,
        )

        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": down,
        }

        out = adapter.from_hf(hf_state, device_mesh=device_mesh)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"
        down_key = "model.language_model.layers.0.mlp.experts.down_projs"

        # Only experts 1 and 2 should be sliced
        assert out[gate_key].shape[0] == 2
        assert out[down_key].shape[0] == 2


# ---------------------------------------------------------------------------
# convert_single_tensor_to_hf
# ---------------------------------------------------------------------------
class TestConvertSingleTensorToHf:
    def test_gate_and_up_projs_conversion(self, adapter):
        tensor = torch.randn(4, 64, 128)
        fqn = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        key, value = result[0]
        assert key == "model.language_model.layers.0.mlp.experts.gate_up_proj"
        # Should be transposed(1,2)
        assert value.shape == (4, 128, 64)

    def test_down_projs_conversion(self, adapter):
        tensor = torch.randn(4, 64, 32)
        fqn = "model.language_model.layers.0.mlp.experts.down_projs"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        key, value = result[0]
        assert key == "model.language_model.layers.0.mlp.experts.down_proj"
        assert value.shape == (4, 32, 64)

    def test_shared_experts_key_mapping(self, adapter):
        tensor = torch.randn(64, 32)
        fqn = "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"
        assert torch.equal(result[0][1], tensor)

    def test_non_expert_tensor_passthrough(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.language_model.layers.0.self_attn.q_proj.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == fqn
        assert result[0][1] is tensor

    def test_exclude_regex_filters_results(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "exclude.me"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

        assert result == []

    def test_expert_key_with_no_model_prefix(self, adapter):
        adapter._uses_model_prefix = False
        tensor = torch.randn(4, 64, 128)
        fqn = "language_model.layers.0.mlp.experts.gate_and_up_projs"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        key, _ = result[0]
        assert key == "language_model.layers.0.mlp.experts.gate_up_proj"


# ---------------------------------------------------------------------------
# to_hf  –  ep_shard multi-node scenarios
# ---------------------------------------------------------------------------
class TestToHFEpShard:
    """Tests for to_hf with ep_shard > 1 (multi-node expert FSDP sharding)."""

    def _make_fake_dtensor(self, local_data, full_data):
        """Create a fake DTensor that records full_tensor() calls."""

        class _FakeDTensor:
            def __init__(self, local, full):
                self._local = local
                self._full = full
                self.shape = full.shape

            def full_tensor(self):
                return self._full

            def cpu(self):
                return _FakeDTensor(self._local.cpu(), self._full.cpu())

            def to(self, dtype):
                return _FakeDTensor(self._local.to(dtype), self._full.to(dtype))

        return _FakeDTensor(local_data, full_data)

    def test_to_hf_dtensor_full_tensor_is_used(self, adapter, monkeypatch):
        """full_tensor() must be called so ep_shard dim is all-gathered before all_gather_object."""
        n_experts = adapter.moe_config.n_routed_experts  # 4
        # NeMo native: [n_experts, hidden, inter]; HF: [n_experts, inter, hidden]
        hidden, inter = 4, 8
        ep_size = 2
        local_experts = n_experts // ep_size  # 2

        # Full expert weight per expert (native layout): [hidden, inter]
        full_weights = [torch.randn(hidden, inter, dtype=adapter.dtype) for _ in range(local_experts)]
        # Local ep_shard shard: [hidden/2, inter]
        local_weights = [w[: hidden // 2] for w in full_weights]

        fake_split_results = [self._make_fake_dtensor(l, f) for l, f in zip(local_weights, full_weights)]

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.is_dtensor", lambda t: True
        )
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.split_experts_weights_dtensor_aware",
            lambda weight, n: (fake_split_results, [0, 1]),
        )
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
        monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: ep_size)

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["ep"]
        device_mesh.get_group = lambda dim: "ep_group"

        def fake_all_gather_object(gathered, payload, group=None):
            gathered[0] = payload
            gathered[1] = ([2, 3], [torch.randn(hidden, inter, dtype=adapter.dtype) for _ in range(2)])

        monkeypatch.setattr("torch.distributed.all_gather_object", fake_all_gather_object)

        dummy = self._make_fake_dtensor(
            torch.empty(local_experts, hidden // 2, inter),
            torch.empty(n_experts, hidden, inter),
        )

        state_dict = {"model.language_model.layers.0.mlp.experts.gate_and_up_projs": dummy}
        out = adapter.to_hf(state_dict, device_mesh=device_mesh)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        global_gate = out[gate_key]

        # to_hf applies transpose(1,2): native [n, hidden, inter] → HF [n, inter, hidden]
        assert global_gate.shape == (n_experts, inter, hidden)
        # Experts 0,1 should have full weight (transposed)
        torch.testing.assert_close(global_gate[0], full_weights[0].T)
        torch.testing.assert_close(global_gate[1], full_weights[1].T)


# ---------------------------------------------------------------------------
# from_hf  –  ep_shard multi-node scenarios
# ---------------------------------------------------------------------------
class TestFromHFEpShard:
    """Tests for from_hf with ep_shard > 1 (multi-node expert FSDP sharding)."""

    def _setup_from_hf_mocks(self, monkeypatch, ep_range, ep_shard_size, ep_shard_rank):
        """Shared mock setup for from_hf ep_shard tests."""
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh",
            lambda mesh, n: ep_range,
        )

        mock_ep_sub = Mock()
        mock_ep_sub.get_rank.return_value = 0

        mock_ep_shard_sub = Mock()
        mock_ep_shard_sub.size.return_value = ep_shard_size
        mock_ep_shard_sub.get_local_rank.return_value = ep_shard_rank

        def fake_get_submesh(mesh, dims):
            if dims == ("ep",):
                return mock_ep_sub
            if dims == ("ep_shard",):
                return mock_ep_shard_sub
            return Mock()

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_submesh", fake_get_submesh
        )

        captured_list = []

        def fake_create_dtensor(local_tensor, mesh, rank):
            captured_list.append(local_tensor)
            return local_tensor

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.create_dtensor_from_local",
            fake_create_dtensor,
        )

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["ep_shard", "ep"]

        return device_mesh, captured_list

    def test_from_hf_slices_ep_shard_dim(self, adapter, monkeypatch):
        """With ep_shard_size=2, from_hf must slice dim 1 of the transposed tensor."""
        n_experts = adapter.moe_config.n_routed_experts  # 4
        # HF: [n_experts, inter, hidden]; native (after transpose): [n_experts, hidden, inter]
        inter, hidden = 8, 4
        ep_shard_size, ep_shard_rank = 2, 1

        device_mesh, captured_list = self._setup_from_hf_mocks(
            monkeypatch, ep_range=(0, n_experts), ep_shard_size=ep_shard_size, ep_shard_rank=ep_shard_rank
        )

        gate_up_hf = torch.arange(n_experts * inter * hidden, dtype=adapter.dtype).reshape(n_experts, inter, hidden)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up_hf,
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(n_experts, hidden, inter, dtype=adapter.dtype),
        }

        adapter.from_hf(hf_state, device_mesh=device_mesh)

        # First captured tensor is gate_and_up_projs
        local_gate = captured_list[0]
        # After transpose(1,2): [n_experts, hidden, inter]; ep_shard slices dim 1 (hidden)
        chunk = hidden // ep_shard_size
        native_full = gate_up_hf.transpose(1, 2).to(adapter.dtype)
        expected = native_full[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
        assert local_gate.shape == (n_experts, chunk, inter)
        torch.testing.assert_close(local_gate, expected)

    def test_from_hf_no_ep_shard_unchanged(self, adapter, monkeypatch):
        """With ep_shard_size=1 (single-node), from_hf must NOT slice dim 1."""
        n_experts = adapter.moe_config.n_routed_experts
        inter, hidden = 8, 4

        device_mesh, captured_list = self._setup_from_hf_mocks(
            monkeypatch, ep_range=(0, n_experts), ep_shard_size=1, ep_shard_rank=0
        )

        gate_up_hf = torch.randn(n_experts, inter, hidden, dtype=adapter.dtype)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up_hf,
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(n_experts, hidden, inter, dtype=adapter.dtype),
        }

        adapter.from_hf(hf_state, device_mesh=device_mesh)

        local_gate = captured_list[0]
        # No ep_shard slicing — full transposed tensor
        assert local_gate.shape == (n_experts, hidden, inter)
        torch.testing.assert_close(local_gate, gate_up_hf.transpose(1, 2).to(adapter.dtype))
