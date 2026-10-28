# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import pytest
import torch

from nemo_automodel.components.models.beam.model import beam_moe_config
from nemo_automodel.components.models.beam.state_dict_adapter import BeamStateDictAdapter

from .utils import hf_model_and_checkpoint


@pytest.fixture
def adapter(beam_config, backend_config):
    moe_config = beam_moe_config(beam_config, torch.float32)
    return BeamStateDictAdapter(beam_config, moe_config, backend_config, dtype=torch.float32)


class TestBeamStateDictAdapter:
    def test_hf_checkpoint_is_per_expert(self, beam_config):
        sd = hf_model_and_checkpoint(beam_config, torch.device("cpu"))[1]
        assert "model.layers.1.mlp.experts.0.gate_proj.weight" in sd
        assert not any(k.endswith("experts.gate_up_proj") for k in sd)

    def test_from_hf_groups_experts_and_passes_everything_else(self, adapter, beam_config):
        sd = hf_model_and_checkpoint(beam_config, torch.device("cpu"))[1]
        native = adapter.from_hf(dict(sd))
        E, H, intermediate_size = 8, beam_config.hidden_size, beam_config.moe_intermediate_size
        for layer in (1, 2, 3):
            gu = native[f"model.layers.{layer}.mlp.experts.gate_and_up_projs"]
            dn = native[f"model.layers.{layer}.mlp.experts.down_projs"]
            assert gu.shape == (E, H, 2 * intermediate_size) and dn.shape == (E, intermediate_size, H)
            for e in range(E):
                torch.testing.assert_close(
                    gu[e, :, :intermediate_size], sd[f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight"].T
                )
                torch.testing.assert_close(
                    gu[e, :, intermediate_size:], sd[f"model.layers.{layer}.mlp.experts.{e}.up_proj.weight"].T
                )
                torch.testing.assert_close(dn[e], sd[f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight"].T)
        assert "model.layers.0.mlp.experts.gate_and_up_projs" not in native  # dense layer
        for key in (
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.mlp.gate.weight",
            "model.layers.1.mlp.gate.e_score_correction_bias",
            "model.layers.2.self_attn.gate_bias",
            "model.layers.2.self_attn.gate_proj.weight",
            "model.layers.3.post_attention_layernorm.weight",
            "model.layers.3.pre_feedforward_layernorm.weight",
            "model.layers.3.post_feedforward_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ):
            assert torch.equal(native[key], sd[key]), key

    def test_round_trip_is_exact(self, adapter, beam_config):
        sd = hf_model_and_checkpoint(beam_config, torch.device("cpu"))[1]
        back = adapter.to_hf(adapter.from_hf(dict(sd)))
        torch.testing.assert_close(back, sd, rtol=0, atol=0)

    @pytest.mark.parametrize("v4_compatible", [False, True])
    @pytest.mark.parametrize("prefix", ["model.", ""])
    def test_lora_round_trip(self, adapter: BeamStateDictAdapter, v4_compatible: bool, prefix: str) -> None:
        """Beam's adapter preserves expert LoRA weights through v4 and v5 exports."""
        adapter._uses_model_prefix = bool(prefix)
        base = f"{prefix}layers.1.mlp.experts"
        experts = adapter.moe_config.n_routed_experts
        hidden = adapter.moe_config.dim
        intermediate = adapter.moe_config.moe_inter_dim
        rank = 8
        native = {
            f"{base}.lora_gate_and_up_A": torch.randn(experts, hidden, rank),
            f"{base}.lora_gate_and_up_B": torch.randn(experts, rank, 2 * intermediate),
            f"{base}.lora_down_A": torch.randn(experts, intermediate, rank),
            f"{base}.lora_down_B": torch.randn(experts, rank, hidden),
        }
        exported = adapter.to_hf(native, v4_compatible=v4_compatible)
        if v4_compatible:
            assert f"{base}.0.gate_proj.lora_A.weight" in exported
            assert len(exported) == experts * 6
        else:
            assert f"{base}.base_layer.lora_A.weight" in exported
            assert len(exported) == 4
        torch.testing.assert_close(adapter.from_hf(dict(exported)), native, atol=0, rtol=0)
        assert not adapter.to_hf(native, v4_compatible=v4_compatible, exclude_key_regex=r".*lora_[AB]\.weight$")

    def test_to_hf_respects_exclude_regex(self, adapter, beam_config):
        sd = hf_model_and_checkpoint(beam_config, torch.device("cpu"))[1]
        back = adapter.to_hf(adapter.from_hf(dict(sd)), exclude_key_regex=r".*lm_head.*")
        assert "lm_head.weight" not in back and "model.embed_tokens.weight" in back


def test_low_memory_dcp_load_enabled(adapter):
    """Beam inherits the Qwen3-MoE opt-in: its non-expert tensors pass through, so DCP loads into model storage."""
    assert adapter.supports_low_memory_dcp_load


def test_checkpoint_load_destinations_write_through_to_beam_weights(beam_config, backend_config):
    """DCP writes through Beam's HF destinations must update the owning native parameters."""
    from nemo_automodel.components.models.beam.model import BeamForCausalLM

    model = BeamForCausalLM(beam_config, backend=backend_config)
    state = model.state_dict()
    destinations = model.state_dict_adapter.to_hf(state, for_checkpoint_load=True)
    assert model.state_dict_adapter.supports_low_memory_dcp_load
    assert destinations
    expected = {}
    with torch.no_grad():
        for index, (name, destination) in enumerate(destinations.items(), 1):
            assert any(
                destination.untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr()
                for parameter in state.values()
            ), name
            destination.fill_(index / 100)
            expected[name] = destination.clone()
    reexported = model.state_dict_adapter.to_hf(model.state_dict())
    torch.testing.assert_close(reexported, expected, rtol=0, atol=0)
