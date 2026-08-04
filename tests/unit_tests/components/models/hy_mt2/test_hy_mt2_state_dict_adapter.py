# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for ``HyMT2StateDictAdapter``.

Covers the rename tables, per-expert split/merge inherited from
``MoESplitExpertsStateDictMixin``, and the defensive MTP-layer filter.
"""

from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_mt2.state_dict_adapter import (
    _HF_TO_NATIVE_RENAMES,
    _NATIVE_TO_HF_RENAMES,
    HyMT2StateDictAdapter,
)
from nemo_automodel.components.moe.config import MoEConfig

N_EXPERTS = 4
HIDDEN = 16
MOE_INTER = 8
NUM_LAYERS = 2  # layer 0 dense, layer 1 MoE


@pytest.fixture
def config():
    cfg = Mock()
    cfg.num_hidden_layers = NUM_LAYERS
    cfg.hidden_size = HIDDEN
    cfg.intermediate_size = 32
    cfg.moe_intermediate_size = MOE_INTER
    cfg.expert_hidden_dim = MOE_INTER
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_experts = N_EXPERTS
    cfg.num_experts_per_tok = 2
    cfg.num_shared_experts = 1
    cfg.first_k_dense_replace = 1
    return cfg


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=HIDDEN,
        inter_dim=32,
        moe_inter_dim=MOE_INTER,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="sigmoid",
        route_scale=2.826,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
        force_e_score_correction_bias=True,
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
    return HyMT2StateDictAdapter(config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float32)


def _make_disk_state_dict(*, with_mtp: bool = False):
    """Synthesize an on-disk Hy-MT2 (== Hy3-preview key layout) state dict."""
    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(32, HIDDEN),
        "model.norm.weight": torch.randn(HIDDEN),
        "lm_head.weight": torch.randn(32, HIDDEN),
        # Layer 0: dense
        "model.layers.0.input_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(32, HIDDEN),
        "model.layers.0.mlp.up_proj.weight": torch.randn(32, HIDDEN),
        "model.layers.0.mlp.down_proj.weight": torch.randn(HIDDEN, 32),
        # Layer 1: MoE with on-disk Tencent-internal names
        "model.layers.1.input_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.1.post_attention_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.layers.1.self_attn.k_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
        "model.layers.1.self_attn.v_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.layers.1.mlp.router.gate.weight": torch.randn(N_EXPERTS, HIDDEN),
        "model.layers.1.mlp.expert_bias": torch.randn(N_EXPERTS),
        "model.layers.1.mlp.shared_mlp.gate_proj.weight": torch.randn(MOE_INTER, HIDDEN),
        "model.layers.1.mlp.shared_mlp.up_proj.weight": torch.randn(MOE_INTER, HIDDEN),
        "model.layers.1.mlp.shared_mlp.down_proj.weight": torch.randn(HIDDEN, MOE_INTER),
    }
    for e in range(N_EXPERTS):
        sd[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = torch.randn(MOE_INTER, HIDDEN)
        sd[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = torch.randn(MOE_INTER, HIDDEN)
        sd[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = torch.randn(HIDDEN, MOE_INTER)
    if with_mtp:
        sd[f"model.layers.{NUM_LAYERS}.input_layernorm.weight"] = torch.randn(HIDDEN)
        sd[f"model.layers.{NUM_LAYERS}.mlp.expert_bias"] = torch.randn(N_EXPERTS)
    return sd


class TestInitialization:
    def test_attributes_set(self, config, moe_config, backend_config):
        a = HyMT2StateDictAdapter(config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float16)
        assert a.config is config
        assert a.moe_config is moe_config
        assert a.backend is backend_config
        assert a.dtype == torch.float16
        assert a._uses_model_prefix is True

    def test_default_dtype_is_bfloat16(self, config, moe_config, backend_config):
        a = HyMT2StateDictAdapter(config=config, moe_config=moe_config, backend=backend_config)
        assert a.dtype == torch.bfloat16

    def test_inherits_split_experts_mixin(self, adapter):
        from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

        assert isinstance(adapter, MoESplitExpertsStateDictMixin)


class TestRenameTables:
    """Each rename pattern must be reversible: native -> hf -> native."""

    @pytest.mark.parametrize(
        "native, hf",
        [
            ("model.layers.5.mlp.gate.e_score_correction_bias", "model.layers.5.mlp.expert_bias"),
            ("model.layers.5.mlp.gate.weight", "model.layers.5.mlp.router.gate.weight"),
            ("model.layers.5.mlp.shared_experts.gate_proj.weight", "model.layers.5.mlp.shared_mlp.gate_proj.weight"),
            ("model.layers.5.mlp.shared_experts.up_proj.weight", "model.layers.5.mlp.shared_mlp.up_proj.weight"),
            ("model.layers.5.mlp.shared_experts.down_proj.weight", "model.layers.5.mlp.shared_mlp.down_proj.weight"),
        ],
    )
    def test_round_trip(self, native, hf):
        nk = native
        for pat, repl in _NATIVE_TO_HF_RENAMES:
            nk, n = pat.subn(repl, nk)
            if n:
                break
        assert nk == hf

        hk = hf
        for pat, repl in _HF_TO_NATIVE_RENAMES:
            hk, n = pat.subn(repl, hk)
            if n:
                break
        assert hk == native

    def test_unrelated_keys_pass_through(self):
        """Renames must not touch attention, embed, lm_head, layernorm, or dense MLP keys."""
        for k in (
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",  # dense MLP gate_proj must NOT match
            "model.norm.weight",
        ):
            for tab in (_NATIVE_TO_HF_RENAMES, _HF_TO_NATIVE_RENAMES):
                v = k
                for pat, repl in tab:
                    v, n = pat.subn(repl, v)
                    if n:
                        break
                assert v == k, f"{k} unexpectedly renamed to {v}"


class TestFromHF:
    def test_renames_router_gate(self, adapter):
        native = adapter.from_hf(_make_disk_state_dict(), device_mesh=None)
        assert "model.layers.1.mlp.gate.weight" in native
        assert "model.layers.1.mlp.router.gate.weight" not in native

    def test_renames_expert_bias_to_gate_bias(self, adapter):
        native = adapter.from_hf(_make_disk_state_dict(), device_mesh=None)
        assert "model.layers.1.mlp.gate.e_score_correction_bias" in native
        assert "model.layers.1.mlp.expert_bias" not in native

    def test_renames_shared_mlp_to_shared_experts(self, adapter):
        native = adapter.from_hf(_make_disk_state_dict(), device_mesh=None)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"model.layers.1.mlp.shared_experts.{proj}.weight" in native
            assert f"model.layers.1.mlp.shared_mlp.{proj}.weight" not in native

    def test_merges_experts_into_grouped_form(self, adapter):
        native = adapter.from_hf(_make_disk_state_dict(), device_mesh=None)
        for e in range(N_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert f"model.layers.1.mlp.experts.{e}.{proj}.weight" not in native
        assert "model.layers.1.mlp.experts.gate_and_up_projs" in native
        assert "model.layers.1.mlp.experts.down_projs" in native

    def test_merged_shapes_are_native_layout(self, adapter):
        native = adapter.from_hf(_make_disk_state_dict(), device_mesh=None)
        assert tuple(native["model.layers.1.mlp.experts.gate_and_up_projs"].shape) == (
            N_EXPERTS,
            HIDDEN,
            2 * MOE_INTER,
        )
        assert tuple(native["model.layers.1.mlp.experts.down_projs"].shape) == (
            N_EXPERTS,
            MOE_INTER,
            HIDDEN,
        )

    def test_drops_mtp_layer_keys(self, adapter):
        hf = _make_disk_state_dict(with_mtp=True)
        assert any(k.startswith(f"model.layers.{NUM_LAYERS}.") for k in hf)
        native = adapter.from_hf(hf, device_mesh=None)
        assert not any(k.startswith(f"model.layers.{NUM_LAYERS}.") for k in native)


class TestToHF:
    def test_renames_native_back_to_on_disk(self, adapter):
        # Build a minimal native state dict; reuse from_hf to produce it.
        native = adapter.from_hf(_make_disk_state_dict(), device_mesh=None)
        hf = adapter.to_hf(native)
        # On-disk renames present after to_hf
        assert "model.layers.1.mlp.router.gate.weight" in hf
        assert "model.layers.1.mlp.expert_bias" in hf
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"model.layers.1.mlp.shared_mlp.{proj}.weight" in hf
        # Native-only names must be gone.
        assert "model.layers.1.mlp.gate.weight" not in hf
        assert "model.layers.1.mlp.gate.e_score_correction_bias" not in hf

    def test_splits_grouped_experts_to_per_expert(self, adapter):
        native = adapter.from_hf(_make_disk_state_dict(), device_mesh=None)
        hf = adapter.to_hf(native)
        # Per-expert keys re-appear after splitting.
        for e in range(N_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert f"model.layers.1.mlp.experts.{e}.{proj}.weight" in hf
        # Grouped keys gone.
        assert "model.layers.1.mlp.experts.gate_and_up_projs" not in hf
        assert "model.layers.1.mlp.experts.down_projs" not in hf

    def test_round_trip_preserves_per_expert_weights(self, adapter):
        """A full disk -> native -> disk round-trip preserves expert weights."""
        disk = _make_disk_state_dict()
        native = adapter.from_hf(disk, device_mesh=None)
        round_tripped = adapter.to_hf(native)
        for e in range(N_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.1.mlp.experts.{e}.{proj}.weight"
                assert key in round_tripped
                assert torch.allclose(round_tripped[key].to(disk[key].dtype), disk[key])


class TestMTPFilter:
    def test_filters_layer_at_num_hidden(self, adapter):
        assert adapter._is_mtp_key(f"model.layers.{NUM_LAYERS}.foo") is True
        assert adapter._is_mtp_key(f"layers.{NUM_LAYERS}.foo") is True

    def test_does_not_filter_in_range_layers(self, adapter):
        assert adapter._is_mtp_key("model.layers.0.foo") is False
        assert adapter._is_mtp_key(f"model.layers.{NUM_LAYERS - 1}.foo") is False

    def test_does_not_filter_non_layer_keys(self, adapter):
        assert adapter._is_mtp_key("model.embed_tokens.weight") is False
        assert adapter._is_mtp_key("lm_head.weight") is False
        assert adapter._is_mtp_key("model.norm.weight") is False
