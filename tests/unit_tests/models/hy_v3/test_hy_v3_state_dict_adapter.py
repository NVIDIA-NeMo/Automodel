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

"""Unit tests for ``HYV3StateDictAdapter``.

Covers the four behaviors that distinguish HYV3 from the shared
``MoESplitExpertsStateDictMixin``:

  1. HYV3-specific name renames (router.gate, expert_bias, shared_mlp.).
  2. Per-expert split / merge inherited from the mixin.
  3. MTP-layer filtering (drops keys for layer index >= num_hidden_layers).
  4. Round-trip integrity: ``from_hf(to_hf(x))`` recovers ``x``.
"""

from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_v3.state_dict_adapter import (
    _HF_TO_NATIVE_RENAMES,
    _NATIVE_TO_HF_RENAMES,
    HYV3StateDictAdapter,
)
from nemo_automodel.components.moe.config import MoEConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


N_EXPERTS = 4
HIDDEN = 16
MOE_INTER = 8
NUM_LAYERS = 2  # layer 0 dense, layer 1 MoE
NUM_MTP = 1


@pytest.fixture
def config():
    cfg = Mock()
    cfg.num_hidden_layers = NUM_LAYERS
    cfg.hidden_size = HIDDEN
    cfg.intermediate_size = 32
    cfg.moe_intermediate_size = MOE_INTER
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_experts = N_EXPERTS
    cfg.num_experts_per_tok = 2
    cfg.num_shared_experts = 1
    cfg.first_k_dense_replace = 1
    cfg.num_nextn_predict_layers = NUM_MTP
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
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
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
    return HYV3StateDictAdapter(
        config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float32
    )


def _make_disk_state_dict(*, with_mtp: bool = True):
    """Synthesize a state dict matching the on-disk Tencent Hy3-preview format.

    Layer 0: dense MLP. Layer 1: MoE with N_EXPERTS experts + shared MLP +
    router gate + expert_bias. Optionally include one MTP layer at index
    NUM_LAYERS that should be filtered out by ``from_hf``.
    """
    sd: dict[str, torch.Tensor] = {
        # Top-level (passes through unchanged on both directions).
        "model.embed_tokens.weight": torch.randn(32, HIDDEN),
        "model.norm.weight": torch.randn(HIDDEN),
        "lm_head.weight": torch.randn(32, HIDDEN),
        # Layer 0: dense MLP + attention.
        "model.layers.0.input_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(32, HIDDEN),
        "model.layers.0.mlp.up_proj.weight": torch.randn(32, HIDDEN),
        "model.layers.0.mlp.down_proj.weight": torch.randn(HIDDEN, 32),
        # Layer 1: MoE -- on-disk format with Tencent-internal names.
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
        # MTP layer: layer index NUM_LAYERS, must be dropped on from_hf.
        sd[f"model.layers.{NUM_LAYERS}.input_layernorm.weight"] = torch.randn(HIDDEN)
        sd[f"model.layers.{NUM_LAYERS}.mlp.expert_bias"] = torch.randn(N_EXPERTS)
    return sd


def _make_native_state_dict():
    """Synthesize a state dict matching the Automodel native HYV3 format."""
    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(32, HIDDEN),
        "model.norm.weight": torch.randn(HIDDEN),
        "lm_head.weight": torch.randn(32, HIDDEN),
        "model.layers.0.input_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(32, HIDDEN),
        "model.layers.0.mlp.up_proj.weight": torch.randn(32, HIDDEN),
        "model.layers.0.mlp.down_proj.weight": torch.randn(HIDDEN, 32),
        "model.layers.1.input_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.1.post_attention_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.layers.1.self_attn.k_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
        "model.layers.1.self_attn.v_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
        "model.layers.1.self_attn.v_proj.weight": torch.randn(HIDDEN // 2, HIDDEN),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(HIDDEN, HIDDEN),
        # Native MoE keys (post-rename + post-merge):
        "model.layers.1.mlp.gate.weight": torch.randn(N_EXPERTS, HIDDEN),
        "model.layers.1.mlp.gate.e_score_correction_bias": torch.randn(N_EXPERTS),
        "model.layers.1.mlp.experts.gate_and_up_projs": torch.randn(N_EXPERTS, HIDDEN, 2 * MOE_INTER),
        "model.layers.1.mlp.experts.down_projs": torch.randn(N_EXPERTS, MOE_INTER, HIDDEN),
        "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.randn(MOE_INTER, HIDDEN),
        "model.layers.1.mlp.shared_experts.up_proj.weight": torch.randn(MOE_INTER, HIDDEN),
        "model.layers.1.mlp.shared_experts.down_proj.weight": torch.randn(HIDDEN, MOE_INTER),
    }
    return sd


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_attributes_set(self, config, moe_config, backend_config):
        a = HYV3StateDictAdapter(config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float16)
        assert a.config is config
        assert a.moe_config is moe_config
        assert a.backend is backend_config
        assert a.dtype == torch.float16
        assert a._uses_model_prefix is True

    def test_default_dtype_is_bfloat16(self, config, moe_config, backend_config):
        a = HYV3StateDictAdapter(config=config, moe_config=moe_config, backend=backend_config)
        assert a.dtype == torch.bfloat16

    def test_inherits_mixin(self, adapter):
        from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

        assert isinstance(adapter, MoESplitExpertsStateDictMixin)


# ---------------------------------------------------------------------------
# Rename tables
# ---------------------------------------------------------------------------


class TestRenameTables:
    """Sanity-check the static rename tables: each native pattern must round-trip."""

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
    def test_native_to_hf_round_trip(self, native, hf):
        # Apply native->HF
        nk = native
        for pat, repl in _NATIVE_TO_HF_RENAMES:
            nk, n = pat.subn(repl, nk)
            if n:
                break
        assert nk == hf

        # Apply HF->native
        hk = hf
        for pat, repl in _HF_TO_NATIVE_RENAMES:
            hk, n = pat.subn(repl, hk)
            if n:
                break
        assert hk == native

    def test_unrelated_keys_pass_through(self):
        """Renames must not touch attention, embed, lm_head, layernorm keys."""
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


# ---------------------------------------------------------------------------
# from_hf: on-disk -> native
# ---------------------------------------------------------------------------


class TestFromHF:
    def test_renames_router_gate(self, adapter):
        hf = _make_disk_state_dict(with_mtp=False)
        native = adapter.from_hf(hf, device_mesh=None)
        assert "model.layers.1.mlp.gate.weight" in native
        assert "model.layers.1.mlp.router.gate.weight" not in native

    def test_renames_expert_bias_to_gate_bias(self, adapter):
        hf = _make_disk_state_dict(with_mtp=False)
        native = adapter.from_hf(hf, device_mesh=None)
        assert "model.layers.1.mlp.gate.e_score_correction_bias" in native
        assert "model.layers.1.mlp.expert_bias" not in native

    def test_renames_shared_mlp_to_shared_experts(self, adapter):
        hf = _make_disk_state_dict(with_mtp=False)
        native = adapter.from_hf(hf, device_mesh=None)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"model.layers.1.mlp.shared_experts.{proj}.weight" in native
            assert f"model.layers.1.mlp.shared_mlp.{proj}.weight" not in native

    def test_merges_experts_into_grouped_form(self, adapter):
        hf = _make_disk_state_dict(with_mtp=False)
        native = adapter.from_hf(hf, device_mesh=None)
        # Per-expert split keys must be gone.
        for e in range(N_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert f"model.layers.1.mlp.experts.{e}.{proj}.weight" not in native
        # Grouped tensors present.
        assert "model.layers.1.mlp.experts.gate_and_up_projs" in native
        assert "model.layers.1.mlp.experts.down_projs" in native

    def test_merged_shapes_are_native_layout(self, adapter):
        hf = _make_disk_state_dict(with_mtp=False)
        native = adapter.from_hf(hf, device_mesh=None)
        # Native gate_and_up_projs: [E, hidden, 2*moe_inter]
        assert tuple(native["model.layers.1.mlp.experts.gate_and_up_projs"].shape) == (
            N_EXPERTS,
            HIDDEN,
            2 * MOE_INTER,
        )
        # Native down_projs: [E, moe_inter, hidden]
        assert tuple(native["model.layers.1.mlp.experts.down_projs"].shape) == (
            N_EXPERTS,
            MOE_INTER,
            HIDDEN,
        )

    def test_merged_values_match_per_expert_inputs(self, adapter):
        """The stacked native tensors must contain the per-expert tensors transposed
        and concatenated in the well-defined gate-then-up order."""
        hf = _make_disk_state_dict(with_mtp=False)
        native = adapter.from_hf(hf, device_mesh=None)

        gate_up = native["model.layers.1.mlp.experts.gate_and_up_projs"]
        down = native["model.layers.1.mlp.experts.down_projs"]
        for e in range(N_EXPERTS):
            g_hf = hf[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"]  # [moe_inter, hidden]
            u_hf = hf[f"model.layers.1.mlp.experts.{e}.up_proj.weight"]
            d_hf = hf[f"model.layers.1.mlp.experts.{e}.down_proj.weight"]  # [hidden, moe_inter]

            # gate half: [hidden, moe_inter] -> first MOE_INTER columns of gate_and_up_projs
            assert torch.allclose(gate_up[e, :, :MOE_INTER], g_hf.transpose(0, 1).to(adapter.dtype))
            # up half: last MOE_INTER columns
            assert torch.allclose(gate_up[e, :, MOE_INTER:], u_hf.transpose(0, 1).to(adapter.dtype))
            # down: [hidden, moe_inter] HF -> [moe_inter, hidden] native
            assert torch.allclose(down[e], d_hf.transpose(0, 1).to(adapter.dtype))

    def test_drops_mtp_layer_keys(self, adapter):
        hf = _make_disk_state_dict(with_mtp=True)
        # MTP keys are present in input.
        assert any(k.startswith(f"model.layers.{NUM_LAYERS}.") for k in hf)
        native = adapter.from_hf(hf, device_mesh=None)
        # MTP keys must not survive.
        assert not any(k.startswith(f"model.layers.{NUM_LAYERS}.") for k in native)
        assert not any(k.startswith(f"layers.{NUM_LAYERS}.") for k in native)

    def test_passes_through_unrelated_keys(self, adapter):
        hf = _make_disk_state_dict(with_mtp=False)
        native = adapter.from_hf(hf, device_mesh=None)
        for k in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "model.layers.1.self_attn.q_proj.weight",
        ):
            assert k in native
            assert torch.equal(native[k], hf[k])


# ---------------------------------------------------------------------------
# to_hf: native -> on-disk
# ---------------------------------------------------------------------------


class TestToHF:
    def test_renames_native_back_to_on_disk(self, adapter):
        native = _make_native_state_dict()
        hf = adapter.to_hf(native)
        # On-disk keys present.
        assert "model.layers.1.mlp.router.gate.weight" in hf
        assert "model.layers.1.mlp.expert_bias" in hf
        assert "model.layers.1.mlp.shared_mlp.gate_proj.weight" in hf
        # Native keys gone.
        assert "model.layers.1.mlp.gate.weight" not in hf
        assert "model.layers.1.mlp.gate.e_score_correction_bias" not in hf
        assert "model.layers.1.mlp.shared_experts.gate_proj.weight" not in hf

    def test_splits_experts_into_per_expert_keys(self, adapter):
        native = _make_native_state_dict()
        hf = adapter.to_hf(native)
        # Grouped keys gone.
        assert "model.layers.1.mlp.experts.gate_and_up_projs" not in hf
        assert "model.layers.1.mlp.experts.down_projs" not in hf
        # Per-expert split keys present.
        for e in range(N_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert f"model.layers.1.mlp.experts.{e}.{proj}.weight" in hf

    def test_per_expert_shapes_match_disk_layout(self, adapter):
        native = _make_native_state_dict()
        hf = adapter.to_hf(native)
        for e in range(N_EXPERTS):
            assert tuple(hf[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"].shape) == (MOE_INTER, HIDDEN)
            assert tuple(hf[f"model.layers.1.mlp.experts.{e}.up_proj.weight"].shape) == (MOE_INTER, HIDDEN)
            assert tuple(hf[f"model.layers.1.mlp.experts.{e}.down_proj.weight"].shape) == (HIDDEN, MOE_INTER)

    def test_exclude_key_regex(self, adapter):
        native = _make_native_state_dict()
        native["custom.exclude.weight"] = torch.randn(2, 2)
        hf = adapter.to_hf(native, exclude_key_regex=r"^custom\.exclude")
        assert "custom.exclude.weight" not in hf
        # Renames still applied.
        assert "model.layers.1.mlp.router.gate.weight" in hf


# ---------------------------------------------------------------------------
# convert_single_tensor_to_hf
# ---------------------------------------------------------------------------


class TestConvertSingleTensorToHF:
    def test_non_expert_key_renamed(self, adapter):
        t = torch.randn(N_EXPERTS, HIDDEN)
        out = adapter.convert_single_tensor_to_hf("model.layers.1.mlp.gate.weight", t)
        assert len(out) == 1
        assert out[0][0] == "model.layers.1.mlp.router.gate.weight"
        assert torch.equal(out[0][1], t)

    def test_non_expert_key_pass_through(self, adapter):
        t = torch.randn(HIDDEN, HIDDEN)
        out = adapter.convert_single_tensor_to_hf("model.layers.0.self_attn.q_proj.weight", t)
        assert len(out) == 1
        assert out[0][0] == "model.layers.0.self_attn.q_proj.weight"
        assert torch.equal(out[0][1], t)

    def test_expert_tensor_split_and_renamed(self, adapter):
        # gate_and_up_projs -> per-expert gate_proj.weight + up_proj.weight (split + transposed)
        t = torch.randn(N_EXPERTS, HIDDEN, 2 * MOE_INTER)
        pairs = adapter.convert_single_tensor_to_hf("model.layers.1.mlp.experts.gate_and_up_projs", t)
        keys = {k for k, _ in pairs}
        assert len(pairs) == 2 * N_EXPERTS
        for e in range(N_EXPERTS):
            assert f"model.layers.1.mlp.experts.{e}.gate_proj.weight" in keys
            assert f"model.layers.1.mlp.experts.{e}.up_proj.weight" in keys

    def test_expert_down_proj_split_and_transposed(self, adapter):
        t = torch.randn(N_EXPERTS, MOE_INTER, HIDDEN)
        pairs = adapter.convert_single_tensor_to_hf("model.layers.1.mlp.experts.down_projs", t)
        keys = {k for k, _ in pairs}
        assert len(pairs) == N_EXPERTS
        for e in range(N_EXPERTS):
            assert f"model.layers.1.mlp.experts.{e}.down_proj.weight" in keys
        # Shape per expert: [hidden, moe_inter]
        for k, v in pairs:
            assert tuple(v.shape) == (HIDDEN, MOE_INTER)

    def test_exclude_regex_applied_after_rename(self, adapter):
        t = torch.randn(N_EXPERTS)
        out = adapter.convert_single_tensor_to_hf(
            "model.layers.1.mlp.gate.e_score_correction_bias",
            t,
            exclude_key_regex=r".*\.expert_bias$",  # matches the renamed-to name
        )
        assert out == []

    def test_exclude_regex_when_no_match(self, adapter):
        t = torch.randn(2, 2)
        out = adapter.convert_single_tensor_to_hf("custom.weight", t, exclude_key_regex=r"^never")
        assert len(out) == 1
        assert out[0][0] == "custom.weight"


# ---------------------------------------------------------------------------
# Round-trip: from_hf(to_hf(native)) == native
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_native_to_hf_to_native(self, adapter):
        native = _make_native_state_dict()
        hf = adapter.to_hf(native)
        recovered = adapter.from_hf(hf, device_mesh=None)

        assert set(recovered.keys()) == set(native.keys())
        for k in native:
            a, b = native[k].float(), recovered[k].float()
            assert a.shape == b.shape, f"{k}: shape {a.shape} != {b.shape}"
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"{k} differs after round-trip"

    def test_disk_to_native_to_disk(self, adapter):
        """Loading from disk then re-saving must round-trip every per-expert key.
        MTP keys (which from_hf drops) are excluded from the comparison."""
        hf = _make_disk_state_dict(with_mtp=True)
        native = adapter.from_hf(hf, device_mesh=None)
        re_hf = adapter.to_hf(native)

        expected_keys = {k for k in hf if not adapter._is_mtp_key(k)}
        # MTP layer keys must not show up in re_hf either.
        assert set(re_hf.keys()) == expected_keys

        for k in expected_keys:
            a, b = hf[k].float(), re_hf[k].float()
            assert a.shape == b.shape, f"{k}: shape mismatch"
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"{k} differs after round-trip"


# ---------------------------------------------------------------------------
# _is_mtp_key
# ---------------------------------------------------------------------------


class TestIsMTPKey:
    @pytest.mark.parametrize(
        "key, expected",
        [
            ("model.layers.0.self_attn.q_proj.weight", False),
            ("model.layers.1.mlp.expert_bias", False),
            (f"model.layers.{NUM_LAYERS}.input_layernorm.weight", True),
            (f"model.layers.{NUM_LAYERS}.mlp.expert_bias", True),
            (f"model.layers.{NUM_LAYERS + 5}.self_attn.q_proj.weight", True),
            ("model.embed_tokens.weight", False),
            ("model.norm.weight", False),
            ("lm_head.weight", False),
            # Without the model. prefix
            ("layers.0.foo", False),
            (f"layers.{NUM_LAYERS}.foo", True),
        ],
    )
    def test_layer_index_classification(self, adapter, key, expected):
        assert adapter._is_mtp_key(key) is expected

    def test_threshold_uses_config_num_hidden_layers(self, config, moe_config, backend_config):
        config.num_hidden_layers = 80
        a = HYV3StateDictAdapter(config=config, moe_config=moe_config, backend=backend_config)
        assert a._is_mtp_key("model.layers.79.foo") is False
        assert a._is_mtp_key("model.layers.80.foo") is True
