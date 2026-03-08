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
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.models.common import BackendConfig


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
        experts="torch",
        dispatcher="torch",
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

    def test_all_gather_path_populates_global_tensor(self, adapter, monkeypatch):
        # Local shard has experts 0 and 1; simulate another rank providing experts 2 and 3
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
            # payload from this rank for experts [0,1]; simulate other rank with [2,3]
            gathered[0] = payload
            other_weights = [torch.tensor([[3.0]], dtype=adapter.dtype), torch.tensor([[4.0]], dtype=adapter.dtype)]
            gathered[1] = ([2, 3], other_weights)

        monkeypatch.setattr("torch.distributed.all_gather_object", fake_all_gather_object)

        state_dict = {"model.language_model.layers.0.mlp.experts.gate_and_up_projs": local_experts}
        out = adapter.to_hf(state_dict, device_mesh=device_mesh)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        global_gate = out[gate_key]

        assert global_gate.shape == (adapter.moe_config.n_routed_experts, 1, 1)
        torch.testing.assert_close(global_gate[0], torch.tensor([[1.0]], dtype=adapter.dtype))
        torch.testing.assert_close(global_gate[1], torch.tensor([[2.0]], dtype=adapter.dtype))
        torch.testing.assert_close(global_gate[2], torch.tensor([[3.0]], dtype=adapter.dtype))
        torch.testing.assert_close(global_gate[3], torch.tensor([[4.0]], dtype=adapter.dtype))


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


# ---------------------------------------------------------------------------
# to_hf  –  ep_shard multi-node scenarios
# ---------------------------------------------------------------------------
class TestToHFEpShard:
    """Tests for to_hf with ep_shard > 1 (multi-node expert FSDP sharding)."""

    def _make_fake_dtensor(self, local_data, full_data):
        """Create a fake DTensor that records full_tensor() calls."""

        class _FakeDTensor:
            """Mimics a DTensor sharded on ep_shard with .full_tensor() support."""

            def __init__(self, local, full):
                self._local = local
                self._full = full
                self.shape = full.shape  # DTensor.shape returns global shape

            def full_tensor(self):
                return self._full

            def cpu(self):
                return _FakeDTensor(self._local.cpu(), self._full.cpu())

            def to(self, dtype):
                return _FakeDTensor(self._local.to(dtype), self._full.to(dtype))

        return _FakeDTensor(local_data, full_data)

    def test_to_hf_dtensor_full_tensor_is_used(self, adapter, monkeypatch):
        """full_tensor() must be called (not to_local/cpu) so the ep_shard dim is all-gathered."""
        n_experts = adapter.moe_config.n_routed_experts  # 4
        inter, hidden = 8, 4
        ep_size = 2
        local_experts = n_experts // ep_size  # 2

        # Full expert weight per expert: [inter, hidden]
        full_weights = [torch.randn(inter, hidden, dtype=adapter.dtype) for _ in range(local_experts)]
        # Local shard (ep_shard=2): [inter/2, hidden]
        local_weights = [w[: inter // 2] for w in full_weights]

        # split_experts_weights_dtensor_aware returns FakeDTensors
        fake_split_results = [self._make_fake_dtensor(l, f) for l, f in zip(local_weights, full_weights)]
        expert_ids = [0, 1]

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.is_dtensor", lambda t: True
        )
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.split_experts_weights_dtensor_aware",
            lambda weight, n: (fake_split_results, expert_ids),
        )
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
        monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: ep_size)

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["ep"]
        device_mesh.get_group = lambda dim: "ep_group"

        def fake_all_gather_object(gathered, payload, group=None):
            gathered[0] = payload
            # Other EP rank has experts 2, 3
            gathered[1] = ([2, 3], [torch.randn(inter, hidden, dtype=adapter.dtype) for _ in range(2)])

        monkeypatch.setattr("torch.distributed.all_gather_object", fake_all_gather_object)

        # Use a dummy tensor whose .shape returns global shape [n_experts, inter, hidden]
        dummy = self._make_fake_dtensor(
            torch.empty(local_experts, inter // 2, hidden),
            torch.empty(n_experts, inter, hidden),
        )

        state_dict = {"model.language_model.layers.0.mlp.experts.gate_and_up_projs": dummy}
        out = adapter.to_hf(state_dict, device_mesh=device_mesh)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        global_gate = out[gate_key]

        # The global tensor must have full inter dimension (not ep_shard-local)
        assert global_gate.shape == (n_experts, inter, hidden)
        # Experts 0 and 1 should contain the FULL weight (from full_tensor), not the local shard
        torch.testing.assert_close(global_gate[0], full_weights[0])
        torch.testing.assert_close(global_gate[1], full_weights[1])


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
        """With ep_shard_size=2, from_hf must slice dim 1 by ep_shard rank."""
        n_experts = adapter.moe_config.n_routed_experts  # 4
        inter, hidden = 8, 4
        ep_shard_size, ep_shard_rank = 2, 1
        local_experts = n_experts // 2  # 2

        device_mesh, captured_list = self._setup_from_hf_mocks(
            monkeypatch, ep_range=(0, local_experts), ep_shard_size=ep_shard_size, ep_shard_rank=ep_shard_rank
        )

        gate_up = torch.arange(n_experts * inter * hidden, dtype=adapter.dtype).reshape(n_experts, inter, hidden)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(n_experts, hidden, inter, dtype=adapter.dtype),
        }

        adapter.from_hf(hf_state, device_mesh=device_mesh)

        # First captured tensor is gate_and_up_projs (dict is insertion-ordered)
        local_gate = captured_list[0]
        chunk = inter // ep_shard_size
        assert local_gate.shape == (local_experts, chunk, hidden)
        expected = gate_up[:local_experts, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
        torch.testing.assert_close(local_gate, expected.to(adapter.dtype))

    def test_from_hf_no_ep_shard_unchanged(self, adapter, monkeypatch):
        """With ep_shard_size=1 (single-node), from_hf must NOT slice dim 1."""
        n_experts = adapter.moe_config.n_routed_experts  # 4
        inter, hidden = 8, 4

        device_mesh, captured_list = self._setup_from_hf_mocks(
            monkeypatch, ep_range=(0, n_experts), ep_shard_size=1, ep_shard_rank=0
        )

        gate_up = torch.randn(n_experts, inter, hidden, dtype=adapter.dtype)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(n_experts, hidden, inter, dtype=adapter.dtype),
        }

        adapter.from_hf(hf_state, device_mesh=device_mesh)

        local_gate = captured_list[0]
        assert local_gate.shape == (n_experts, inter, hidden)
        torch.testing.assert_close(local_gate, gate_up.to(adapter.dtype))

    def test_from_hf_ep_shard_roundtrip(self, adapter, monkeypatch):
        """to_hf → from_hf roundtrip: data at a specific ep_shard rank must be recoverable."""
        n_experts = adapter.moe_config.n_routed_experts  # 4
        inter, hidden = 8, 4
        ep_shard_size, ep_shard_rank = 2, 0

        original = torch.arange(n_experts * inter * hidden, dtype=adapter.dtype).reshape(n_experts, inter, hidden)

        device_mesh, captured_list = self._setup_from_hf_mocks(
            monkeypatch, ep_range=(0, n_experts), ep_shard_size=ep_shard_size, ep_shard_rank=ep_shard_rank
        )

        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": original.clone(),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(n_experts, hidden, inter, dtype=adapter.dtype),
        }

        adapter.from_hf(hf_state, device_mesh=device_mesh)

        local_gate = captured_list[0]
        chunk = inter // ep_shard_size
        expected_shard = original[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
        torch.testing.assert_close(local_gate, expected_shard)
