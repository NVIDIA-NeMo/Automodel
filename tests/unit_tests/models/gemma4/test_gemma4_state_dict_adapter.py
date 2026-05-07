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

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.state_dict_adapter import Gemma4MoEStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

N_EXPERTS = 4
HIDDEN = 64
EXPERT_INTER = 32


@pytest.fixture
def config():
    cfg = Mock()
    cfg.num_hidden_layers = 2
    cfg.hidden_size = HIDDEN
    cfg.intermediate_size = 128
    cfg.num_experts = N_EXPERTS
    cfg.top_k_experts = 2
    cfg.moe_intermediate_size = EXPERT_INTER
    cfg.expert_intermediate_size = EXPERT_INTER
    return cfg


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=HIDDEN,
        inter_dim=128,
        moe_inter_dim=EXPERT_INTER,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_activation="geglu",
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
    return Gemma4MoEStateDictAdapter(
        config=config,
        moe_config=moe_config,
        backend=backend_config,
        dtype=torch.float32,
    )


def _make_hf_state_dict(layer_idx=0, with_model_prefix=True):
    """Build a minimal HF-format Gemma4 MoE state dict for one layer (v5.5 layout)."""
    prefix = "model.language_model." if with_model_prefix else ""
    layer = f"{prefix}layers.{layer_idx}"
    return {
        f"{layer}.router.proj.weight": torch.randn(N_EXPERTS, HIDDEN),
        f"{layer}.router.scale": torch.randn(HIDDEN),
        f"{layer}.router.per_expert_scale": torch.ones(N_EXPERTS) * 2.0,
        f"{layer}.experts.gate_up_proj": torch.randn(N_EXPERTS, 2 * EXPERT_INTER, HIDDEN),
        f"{layer}.experts.down_proj": torch.randn(N_EXPERTS, HIDDEN, EXPERT_INTER),
        f"{layer}.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN),
    }


# ---------------------------------------------------------------------------
# from_hf tests
# ---------------------------------------------------------------------------
class TestFromHf:
    def test_router_keys_remapped(self, adapter):
        hf_sd = _make_hf_state_dict()

        nemo_sd = adapter.from_hf(hf_sd)

        assert "model.language_model.layers.0.moe.gate.proj.weight" in nemo_sd
        assert "model.language_model.layers.0.moe.gate.scale" in nemo_sd

    def test_router_original_keys_removed(self, adapter):
        hf_sd = _make_hf_state_dict()

        nemo_sd = adapter.from_hf(hf_sd)

        for key in nemo_sd:
            assert ".router." not in key

    def test_expert_gate_up_concatenated(self, adapter):
        hf_sd = _make_hf_state_dict()
        gate_up_proj = hf_sd["model.language_model.layers.0.experts.gate_up_proj"]

        nemo_sd = adapter.from_hf(hf_sd)

        gate_and_up = nemo_sd["model.language_model.layers.0.moe.experts.gate_and_up_projs"]
        # HF [E, 2*inter, hidden] transposed to NeMo [E, hidden, 2*inter]
        assert gate_and_up.shape == (N_EXPERTS, HIDDEN, 2 * EXPERT_INTER)
        torch.testing.assert_close(gate_and_up, gate_up_proj.transpose(-2, -1))

    def test_per_expert_scale_absorbed_into_down_projs(self, adapter):
        hf_sd = _make_hf_state_dict()
        down_proj = hf_sd["model.language_model.layers.0.experts.down_proj"]
        per_expert_scale = hf_sd["model.language_model.layers.0.router.per_expert_scale"]

        nemo_sd = adapter.from_hf(hf_sd)

        down_projs = nemo_sd["model.language_model.layers.0.moe.experts.down_projs"]
        # HF [E, hidden, inter] transposed to NeMo [E, inter, hidden], scaled by per_expert_scale
        expected = down_proj.transpose(-2, -1) * per_expert_scale[:, None, None]
        torch.testing.assert_close(down_projs, expected)

    def test_passthrough_keys_preserved(self, adapter):
        hf_sd = _make_hf_state_dict()
        original_attn = hf_sd["model.language_model.layers.0.self_attn.q_proj.weight"].clone()

        nemo_sd = adapter.from_hf(hf_sd)

        assert "model.language_model.layers.0.self_attn.q_proj.weight" in nemo_sd
        torch.testing.assert_close(
            nemo_sd["model.language_model.layers.0.self_attn.q_proj.weight"],
            original_attn,
        )

    def test_hf_expert_keys_not_in_output(self, adapter):
        hf_sd = _make_hf_state_dict()

        nemo_sd = adapter.from_hf(hf_sd)

        for key in nemo_sd:
            assert ".experts.gate_up_proj" not in key or "gate_and_up_projs" in key
            assert ".experts.down_proj" not in key or "experts.down_projs" in key
            assert ".router.per_expert_scale" not in key

    def test_incomplete_expert_keys_raises(self, adapter):
        hf_sd = _make_hf_state_dict()
        del hf_sd["model.language_model.layers.0.experts.gate_up_proj"]

        with pytest.raises(RuntimeError, match="Incomplete expert weights"):
            adapter.from_hf(hf_sd)

    def test_from_hf_slices_ep_shard_feature_dimension(self, adapter, monkeypatch):
        class FakeShardMesh:
            def size(self):
                return 2

            def get_local_rank(self):
                return 1

            def get_rank(self):
                return 0

        class FakeDeviceMesh:
            mesh_dim_names = ("ep_shard", "ep")

        hf_sd = _make_hf_state_dict()
        gate_up_proj = hf_sd["model.language_model.layers.0.experts.gate_up_proj"]
        down_proj = hf_sd["model.language_model.layers.0.experts.down_proj"]
        per_expert_scale = hf_sd["model.language_model.layers.0.router.per_expert_scale"]

        monkeypatch.setattr(
            "nemo_automodel.components.models.gemma4_moe.state_dict_adapter.state_dict_utils."
            "get_expert_range_for_rank_from_mesh",
            lambda device_mesh, n_experts: (0, n_experts),
        )
        monkeypatch.setattr(
            "nemo_automodel.components.models.gemma4_moe.state_dict_adapter.state_dict_utils.get_submesh",
            lambda device_mesh, dims: FakeShardMesh(),
        )
        monkeypatch.setattr(
            "nemo_automodel.components.models.gemma4_moe.state_dict_adapter.state_dict_utils.create_dtensor_from_local",
            lambda local_tensor, device_mesh, rank: local_tensor,
        )

        nemo_sd = adapter.from_hf(hf_sd, device_mesh=FakeDeviceMesh())

        expected_gate_and_up = gate_up_proj.transpose(-2, -1)[:, HIDDEN // 2 :, :]
        expected_down = (down_proj.transpose(-2, -1) * per_expert_scale[:, None, None])[:, EXPERT_INTER // 2 :, :]
        torch.testing.assert_close(
            nemo_sd["model.language_model.layers.0.moe.experts.gate_and_up_projs"],
            expected_gate_and_up,
        )
        torch.testing.assert_close(
            nemo_sd["model.language_model.layers.0.moe.experts.down_projs"],
            expected_down,
        )

    def test_without_model_prefix(self, adapter):
        hf_sd = _make_hf_state_dict(with_model_prefix=False)

        nemo_sd = adapter.from_hf(hf_sd)

        assert any("moe.gate.proj.weight" in k for k in nemo_sd)
        assert any("moe.experts.gate_and_up_projs" in k for k in nemo_sd)

    def test_multiple_layers(self, adapter):
        hf_sd = {}
        for layer_idx in range(2):
            hf_sd.update(_make_hf_state_dict(layer_idx=layer_idx))

        nemo_sd = adapter.from_hf(hf_sd)

        for layer_idx in range(2):
            assert f"model.language_model.layers.{layer_idx}.moe.experts.gate_and_up_projs" in nemo_sd
            assert f"model.language_model.layers.{layer_idx}.moe.experts.down_projs" in nemo_sd
            assert f"model.language_model.layers.{layer_idx}.moe.gate.proj.weight" in nemo_sd
            assert f"model.language_model.layers.{layer_idx}.moe.gate.scale" in nemo_sd


# ---------------------------------------------------------------------------
# to_hf tests
# ---------------------------------------------------------------------------
class TestToHf:
    def _make_nemo_state_dict(self, layer_idx=0):
        """Build a minimal NeMo-format state dict for one layer."""
        prefix = f"model.language_model.layers.{layer_idx}"
        return {
            f"{prefix}.moe.gate.proj.weight": torch.randn(N_EXPERTS, HIDDEN),
            f"{prefix}.moe.gate.scale": torch.randn(HIDDEN),
            f"{prefix}.moe.experts.gate_and_up_projs": torch.randn(N_EXPERTS, HIDDEN, 2 * EXPERT_INTER),
            f"{prefix}.moe.experts.down_projs": torch.randn(N_EXPERTS, EXPERT_INTER, HIDDEN),
            f"{prefix}.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN),
        }

    def test_gate_keys_remapped_to_router(self, adapter):
        nemo_sd = self._make_nemo_state_dict()

        hf_sd = adapter.to_hf(nemo_sd)

        assert "model.language_model.layers.0.router.proj.weight" in hf_sd
        assert "model.language_model.layers.0.router.scale" in hf_sd

    def test_gate_and_up_split_correctly(self, adapter):
        nemo_sd = self._make_nemo_state_dict()
        gate_and_up = nemo_sd["model.language_model.layers.0.moe.experts.gate_and_up_projs"]

        hf_sd = adapter.to_hf(nemo_sd)

        gate_up_proj = hf_sd["model.language_model.layers.0.experts.gate_up_proj"]
        # NeMo [E, hidden, 2*inter] transposed to HF [E, 2*inter, hidden]
        assert gate_up_proj.shape == (N_EXPERTS, 2 * EXPERT_INTER, HIDDEN)
        torch.testing.assert_close(gate_up_proj, gate_and_up.transpose(-2, -1))

    def test_down_projs_output_and_per_expert_scale(self, adapter):
        nemo_sd = self._make_nemo_state_dict()
        original_down = nemo_sd["model.language_model.layers.0.moe.experts.down_projs"]

        hf_sd = adapter.to_hf(nemo_sd)

        down_proj = hf_sd["model.language_model.layers.0.experts.down_proj"]
        per_expert_scale = hf_sd["model.language_model.layers.0.router.per_expert_scale"]

        # NeMo [E, inter, hidden] transposed to HF [E, hidden, inter]
        torch.testing.assert_close(down_proj, original_down.transpose(-2, -1))
        torch.testing.assert_close(per_expert_scale, torch.ones(N_EXPERTS, dtype=torch.float32))

    def test_passthrough_keys_preserved(self, adapter):
        nemo_sd = self._make_nemo_state_dict()
        original_attn = nemo_sd["model.language_model.layers.0.self_attn.q_proj.weight"].clone()

        hf_sd = adapter.to_hf(nemo_sd)

        assert "model.language_model.layers.0.self_attn.q_proj.weight" in hf_sd
        torch.testing.assert_close(
            hf_sd["model.language_model.layers.0.self_attn.q_proj.weight"],
            original_attn,
        )

    def test_nemo_expert_keys_not_in_output(self, adapter):
        nemo_sd = self._make_nemo_state_dict()

        hf_sd = adapter.to_hf(nemo_sd)

        for key in hf_sd:
            assert "gate_and_up_projs" not in key
            assert "experts.down_projs" not in key

    def test_gather_expert_tensor_materializes_dtensor_slices(self, adapter, monkeypatch):
        class FakeDTensor:
            def __init__(self, tensor):
                self.tensor = tensor
                self.full_tensor_called = False

            def full_tensor(self):
                self.full_tensor_called = True
                return self.tensor

        class FakeDeviceMesh:
            mesh_dim_names = ("ep",)

        source_tensor = FakeDTensor(torch.empty(0))
        split_weights = [
            FakeDTensor(torch.randn(HIDDEN, 2 * EXPERT_INTER)),
            FakeDTensor(torch.randn(HIDDEN, 2 * EXPERT_INTER)),
        ]

        monkeypatch.setattr(
            "nemo_automodel.components.models.gemma4_moe.state_dict_adapter.state_dict_utils.is_dtensor",
            lambda tensor: isinstance(tensor, FakeDTensor),
        )
        monkeypatch.setattr(
            "nemo_automodel.components.models.gemma4_moe.state_dict_adapter.state_dict_utils."
            "split_experts_weights_dtensor_aware",
            lambda tensor, n_experts: (split_weights, [0, 1]),
        )

        gathered = adapter._gather_expert_tensor(source_tensor, FakeDeviceMesh(), N_EXPERTS)

        assert gathered.shape == (N_EXPERTS, HIDDEN, 2 * EXPERT_INTER)
        torch.testing.assert_close(gathered[0], split_weights[0].tensor)
        torch.testing.assert_close(gathered[1], split_weights[1].tensor)
        assert all(weight.full_tensor_called for weight in split_weights)

    def test_exclude_key_regex(self, adapter):
        nemo_sd = self._make_nemo_state_dict()
        nemo_sd["model.language_model.layers.0.exclude_me.weight"] = torch.randn(10)

        hf_sd = adapter.to_hf(nemo_sd, exclude_key_regex=r".*exclude_me.*")

        assert not any("exclude_me" in k for k in hf_sd)

    def test_multiple_layers(self, adapter):
        nemo_sd = {}
        for layer_idx in range(2):
            nemo_sd.update(self._make_nemo_state_dict(layer_idx=layer_idx))

        hf_sd = adapter.to_hf(nemo_sd)

        for layer_idx in range(2):
            assert f"model.language_model.layers.{layer_idx}.router.proj.weight" in hf_sd
            assert f"model.language_model.layers.{layer_idx}.experts.gate_up_proj" in hf_sd
            assert f"model.language_model.layers.{layer_idx}.experts.down_proj" in hf_sd
            assert f"model.language_model.layers.{layer_idx}.router.per_expert_scale" in hf_sd


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_hf_to_nemo_to_hf_preserves_shapes(self, adapter):
        hf_sd = _make_hf_state_dict()

        nemo_sd = adapter.from_hf(hf_sd)
        hf_sd_rt = adapter.to_hf(nemo_sd)

        for key in [
            "model.language_model.layers.0.experts.gate_up_proj",
            "model.language_model.layers.0.experts.down_proj",
            "model.language_model.layers.0.router.per_expert_scale",
            "model.language_model.layers.0.router.proj.weight",
            "model.language_model.layers.0.router.scale",
        ]:
            assert key in hf_sd_rt, f"Missing key after round-trip: {key}"
            assert hf_sd[key].shape == hf_sd_rt[key].shape, f"Shape mismatch for {key}"

    def test_hf_to_nemo_to_hf_preserves_gate_up_values(self, adapter):
        hf_sd = _make_hf_state_dict()
        hf_sd["model.language_model.layers.0.router.per_expert_scale"] = torch.ones(N_EXPERTS)

        nemo_sd = adapter.from_hf(hf_sd)
        hf_sd_rt = adapter.to_hf(nemo_sd)

        torch.testing.assert_close(
            hf_sd_rt["model.language_model.layers.0.experts.gate_up_proj"],
            hf_sd["model.language_model.layers.0.experts.gate_up_proj"],
        )

    def test_hf_to_nemo_to_hf_preserves_down_proj_with_unit_scale(self, adapter):
        hf_sd = _make_hf_state_dict()
        hf_sd["model.language_model.layers.0.router.per_expert_scale"] = torch.ones(N_EXPERTS)

        nemo_sd = adapter.from_hf(hf_sd)
        hf_sd_rt = adapter.to_hf(nemo_sd)

        torch.testing.assert_close(
            hf_sd_rt["model.language_model.layers.0.experts.down_proj"],
            hf_sd["model.language_model.layers.0.experts.down_proj"],
        )

    def test_router_keys_round_trip(self, adapter):
        hf_sd = _make_hf_state_dict()

        nemo_sd = adapter.from_hf(hf_sd)
        hf_sd_rt = adapter.to_hf(nemo_sd)

        torch.testing.assert_close(
            hf_sd_rt["model.language_model.layers.0.router.proj.weight"],
            hf_sd["model.language_model.layers.0.router.proj.weight"],
        )
        torch.testing.assert_close(
            hf_sd_rt["model.language_model.layers.0.router.scale"],
            hf_sd["model.language_model.layers.0.router.scale"],
        )


# ---------------------------------------------------------------------------
# convert_single_tensor_to_hf tests
# ---------------------------------------------------------------------------
class TestConvertSingleTensorToHf:
    def test_passthrough_returns_same_fqn_and_tensor(self, adapter):
        tensor = torch.randn(HIDDEN, HIDDEN)
        fqn = "model.language_model.layers.0.self_attn.q_proj.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == fqn
        assert result[0][1] is tensor

    def test_router_proj_weight_remapped(self, adapter):
        tensor = torch.randn(HIDDEN, HIDDEN)
        fqn = "model.language_model.layers.0.moe.gate.proj.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "model.language_model.layers.0.router.proj.weight"
        assert result[0][1] is tensor

    def test_router_scale_remapped(self, adapter):
        tensor = torch.randn(HIDDEN)
        fqn = "model.language_model.layers.1.moe.gate.scale"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "model.language_model.layers.1.router.scale"
        assert result[0][1] is tensor

    def test_gate_and_up_projs_transposed_and_renamed(self, adapter):
        tensor = torch.randn(N_EXPERTS, HIDDEN, 2 * EXPERT_INTER)
        fqn = "model.language_model.layers.0.moe.experts.gate_and_up_projs"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "model.language_model.layers.0.experts.gate_up_proj"
        assert result[0][1].shape == (N_EXPERTS, 2 * EXPERT_INTER, HIDDEN)
        torch.testing.assert_close(result[0][1], tensor.transpose(-2, -1).contiguous())

    def test_down_projs_transposed_renamed_and_emits_scale(self, adapter):
        tensor = torch.randn(N_EXPERTS, EXPERT_INTER, HIDDEN)
        fqn = "model.language_model.layers.0.moe.experts.down_projs"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 2
        hf_key, transposed = result[0]
        scale_key, scale_tensor = result[1]

        assert hf_key == "model.language_model.layers.0.experts.down_proj"
        assert transposed.shape == (N_EXPERTS, HIDDEN, EXPERT_INTER)
        torch.testing.assert_close(transposed, tensor.transpose(-2, -1).contiguous())

        assert scale_key == "model.language_model.layers.0.router.per_expert_scale"
        assert scale_tensor.shape == (N_EXPERTS,)
        torch.testing.assert_close(scale_tensor, torch.ones(N_EXPERTS, dtype=tensor.dtype))

    def test_exclude_key_regex_filters_key(self, adapter):
        tensor = torch.randn(HIDDEN, HIDDEN)
        fqn = "model.language_model.layers.0.moe.gate.proj.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r".*moe\.gate.*")

        assert result == []

    def test_exclude_key_regex_does_not_filter_non_matching(self, adapter):
        tensor = torch.randn(HIDDEN, HIDDEN)
        fqn = "model.language_model.layers.0.self_attn.q_proj.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r".*moe\.gate.*")

        assert len(result) == 1
        assert result[0][0] == fqn
