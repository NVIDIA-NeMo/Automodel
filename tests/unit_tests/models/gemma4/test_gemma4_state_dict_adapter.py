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

pytest.importorskip("transformers.models.gemma4", reason="Gemma4 not available in this transformers version")

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
    """Build a minimal HF-format Gemma4 MoE state dict for one layer.

    Uses the HF transformers >= 5.5 layout where gate and up projections are
    stored as a single combined ``gate_up_proj`` tensor.
    """
    prefix = "model.language_model." if with_model_prefix else ""
    layer = f"{prefix}layers.{layer_idx}"
    # Build gate_up_proj from two independently generated halves so that tests
    # can slice it deterministically.
    gate_proj = torch.randn(N_EXPERTS, EXPERT_INTER, HIDDEN)
    up_proj = torch.randn(N_EXPERTS, EXPERT_INTER, HIDDEN)
    gate_up_proj = torch.cat([gate_proj, up_proj], dim=1)  # [E, 2*inter, hidden]
    return {
        f"{layer}.router.proj.weight": torch.randn(N_EXPERTS, HIDDEN),
        f"{layer}.router.scale": torch.randn(HIDDEN),
        f"{layer}.experts.gate_up_proj": gate_up_proj,
        f"{layer}.experts.down_proj": torch.randn(N_EXPERTS, HIDDEN, EXPERT_INTER),
        f"{layer}.router.per_expert_scale": torch.ones(N_EXPERTS) * 2.0,
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
        # gate_up_proj is [E, 2*inter, hidden]; after from_hf it becomes
        # [E, hidden, 2*inter] (transposed)

        nemo_sd = adapter.from_hf(hf_sd)

        gate_and_up = nemo_sd["model.language_model.layers.0.moe.experts.gate_and_up_projs"]
        assert gate_and_up.shape == (N_EXPERTS, HIDDEN, 2 * EXPERT_INTER)
        torch.testing.assert_close(gate_and_up, gate_up_proj.transpose(-2, -1))

    def test_per_expert_scale_absorbed_into_down_projs(self, adapter):
        hf_sd = _make_hf_state_dict()
        down_proj = hf_sd["model.language_model.layers.0.experts.down_proj"]  # [E, hidden, inter]
        per_expert_scale = hf_sd["model.language_model.layers.0.router.per_expert_scale"]

        nemo_sd = adapter.from_hf(hf_sd)

        down_projs = nemo_sd["model.language_model.layers.0.moe.experts.down_projs"]
        # from_hf transposes [E, hidden, inter] -> [E, inter, hidden] and multiplies by scale
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
            assert ".experts.gate_up_proj" not in key
            assert ".experts.down_proj" not in key or "experts.down_projs" in key
            assert ".router.per_expert_scale" not in key

    def test_incomplete_expert_keys_raises(self, adapter):
        hf_sd = _make_hf_state_dict()
        del hf_sd["model.language_model.layers.0.experts.gate_up_proj"]

        with pytest.raises(RuntimeError, match="Incomplete expert weights"):
            adapter.from_hf(hf_sd)

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

    def test_gate_and_up_combined_correctly(self, adapter):
        nemo_sd = self._make_nemo_state_dict()
        gate_and_up = nemo_sd["model.language_model.layers.0.moe.experts.gate_and_up_projs"]

        hf_sd = adapter.to_hf(nemo_sd)

        gate_up_proj = hf_sd["model.language_model.layers.0.experts.gate_up_proj"]
        # to_hf transposes NeMo [E, hidden, 2*inter] -> HF [E, 2*inter, hidden]
        assert gate_up_proj.shape == (N_EXPERTS, 2 * EXPERT_INTER, HIDDEN)
        torch.testing.assert_close(gate_up_proj, gate_and_up.transpose(-2, -1))

    def test_down_projs_output_and_per_expert_scale(self, adapter):
        nemo_sd = self._make_nemo_state_dict()
        original_down = nemo_sd["model.language_model.layers.0.moe.experts.down_projs"]

        hf_sd = adapter.to_hf(nemo_sd)

        down_proj = hf_sd["model.language_model.layers.0.experts.down_proj"]
        per_expert_scale = hf_sd["model.language_model.layers.0.router.per_expert_scale"]

        # to_hf transposes NeMo [E, inter, hidden] -> HF [E, hidden, inter]
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
            if key in hf_sd:
                assert hf_sd[key].shape == hf_sd_rt[key].shape, f"Shape mismatch for {key}"

    def test_hf_to_nemo_to_hf_preserves_gate_up_values(self, adapter):
        hf_sd = _make_hf_state_dict()
        # Use per_expert_scale=1 so round-trip is exact
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
