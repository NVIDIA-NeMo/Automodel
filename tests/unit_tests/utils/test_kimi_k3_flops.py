# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the Kimi K3 FLOPs formula (hybrid KDA / gated-MLA attention + latent MoE)."""

import pytest

from nemo_automodel.components.models.kimi_k3.config import KimiK3Config, KimiK3TextConfig
from nemo_automodel.components.utils import flops_utils

# moonshotai/Kimi-K3 model card: 2.8T total / 104B activated; tech report Table 1: 2.78T / 104.2B.
# The formula counts matmul weights only (no router GEMM, norms or gate biases) and lands within 0.2% of Table 1.
K3_TOTAL_PARAMS = 2.78e12
K3_ACTIVE_PARAMS = 104e9

# Released checkpoint values (config.json text_config), passed explicitly so these tests do not
# depend on the KimiK3TextConfig defaults.
CHECKPOINT_OVERRIDES = dict(num_attention_heads=96, num_key_value_heads=96, intermediate_size=33792)
# The pre-fix from_config shape (56-head MLA, 18432-wide dense FFN) that earlier K3 benchmark
# numbers were measured with; kept as a second pinned value.
LEGACY_OVERRIDES = dict(num_attention_heads=56, num_key_value_heads=56, intermediate_size=18432)


def _checkpoint_config(**overrides) -> KimiK3TextConfig:
    return KimiK3TextConfig(**{**CHECKPOINT_OVERRIDES, **overrides})


def _kda_layer_counts(config):
    kda = sum(1 for i in range(config.num_hidden_layers) if config.is_kda_layer(i))
    return kda, config.num_hidden_layers - kda


def _implied_total_params(cfg) -> float:
    """Total checkpoint parameters implied by the formula: swap top-k experts for all experts, add router + input embedding."""
    hs, lat = cfg.hidden_size, cfg.routed_expert_hidden_size
    per_expert = 3 * lat * cfg.moe_intermediate_size
    moe_layers = cfg.num_hidden_layers - cfg.first_k_dense_replace
    active = flops_utils.kimi_k3_flops(cfg, gbs=1, seq_len=1) / 6
    return (
        active
        - moe_layers * cfg.num_experts_per_token * per_expert
        + moe_layers * (cfg.num_experts * per_expert + hs * cfg.num_experts)
        + cfg.vocab_size * hs
    )


def test_registered_for_text_and_multimodal_configs():
    assert flops_utils.get_flops_formula_for_hf_config(KimiK3TextConfig()) is flops_utils.kimi_k3_flops
    assert flops_utils.get_flops_formula_for_hf_config(KimiK3Config()) is flops_utils.kimi_k3_flops


def test_multimodal_wrapper_uses_text_config():
    text = _checkpoint_config()
    wrapper = KimiK3Config(text_config=text)
    assert flops_utils.kimi_k3_flops(wrapper, gbs=2, seq_len=512) == flops_utils.kimi_k3_flops(text, gbs=2, seq_len=512)


def test_layer_pattern_matches_the_model_card():
    cfg = _checkpoint_config()
    assert _kda_layer_counts(cfg) == (69, 24)  # "69 KDA + 24 Gated MLA"
    assert cfg.first_k_dense_replace == 1 and cfg.moe_layer_freq == 1  # "Number of Dense Layers: 1"


def test_active_and_total_params_match_the_model_card():
    """Per-token FLOPs / 6 at seq_len=1 is the active matmul parameter count."""
    cfg = _checkpoint_config()
    active = flops_utils.kimi_k3_flops(cfg, gbs=1, seq_len=1) / 6
    assert active == pytest.approx(K3_ACTIVE_PARAMS, rel=0.005)
    assert _implied_total_params(cfg) == pytest.approx(K3_TOTAL_PARAMS, rel=0.005)


def test_precomputed_values():
    assert int(flops_utils.kimi_k3_flops(_checkpoint_config(), gbs=1, seq_len=1024)) == 641380921638912
    assert int(flops_utils.kimi_k3_flops(KimiK3TextConfig(**LEGACY_OVERRIDES), gbs=1, seq_len=1024)) == 625049308495872


def test_linear_in_global_batch_size():
    cfg = _checkpoint_config()
    one = flops_utils.kimi_k3_flops(cfg, gbs=1, seq_len=2048)
    assert flops_utils.kimi_k3_flops(cfg, gbs=4096, seq_len=2048) == pytest.approx(4096 * one)


def test_wall_clock_basis_and_dense_fallback():
    """Published K3 numbers use 6 * 104e9 FLOPs/token; the generic fallback under-counts by >2x."""
    cfg = _checkpoint_config()
    for seq_len in (1024, 2048):
        k3 = flops_utils.kimi_k3_flops(cfg, gbs=1, seq_len=seq_len)
        assert 1.0 < k3 / (6 * K3_ACTIVE_PARAMS * seq_len) < 1.01  # attention BMM + KDA kernel on top of the params
        assert flops_utils.transformer_flops(cfg, gbs=1, seq_len=seq_len) < 0.65 * k3


def test_attention_split_is_a_convex_combination():
    """A depth-8 hybrid equals the per-layer mix of the all-KDA and all-MLA variants."""
    base = _checkpoint_config(num_hidden_layers=8)
    all_mla = _checkpoint_config(
        num_hidden_layers=8,
        linear_attn_config={**base.linear_attn_config, "kda_layers": [], "full_attn_layers": list(range(1, 9))},
    )
    all_kda = _checkpoint_config(
        num_hidden_layers=8,
        linear_attn_config={**base.linear_attn_config, "kda_layers": list(range(1, 9)), "full_attn_layers": []},
    )
    assert _kda_layer_counts(base) == (6, 2)
    f_base = flops_utils.kimi_k3_flops(base, gbs=1, seq_len=1024)
    f_mla = flops_utils.kimi_k3_flops(all_mla, gbs=1, seq_len=1024)
    f_kda = flops_utils.kimi_k3_flops(all_kda, gbs=1, seq_len=1024)
    assert f_base == pytest.approx((6 * f_kda + 2 * f_mla) / 8)
    assert f_kda != f_mla


def test_active_flops_do_not_depend_on_expert_count():
    small = _checkpoint_config(num_hidden_layers=24, num_experts=64)
    big = _checkpoint_config(num_hidden_layers=24, num_experts=896)
    assert flops_utils.kimi_k3_flops(small, gbs=1, seq_len=1024) == flops_utils.kimi_k3_flops(big, gbs=1, seq_len=1024)
    assert flops_utils.kimi_k3_flops(small, gbs=1, seq_len=1024) < flops_utils.kimi_k3_flops(
        _checkpoint_config(), gbs=1, seq_len=1024
    )


def test_optional_blocks_are_costed_exactly():
    """Latent projections, MLA output gate and shared experts each add their GEMM weights."""
    ref = _checkpoint_config()
    hs = ref.hidden_size
    moe_layers = ref.num_hidden_layers - ref.first_k_dense_replace
    mla_layers = _kda_layer_counts(ref)[1]

    def params(cfg):
        return flops_utils.kimi_k3_flops(cfg, gbs=1, seq_len=1) / 6

    no_latent = _checkpoint_config(routed_expert_hidden_size=None)
    expected = moe_layers * (
        ref.num_experts_per_token * 3 * ref.routed_expert_hidden_size * ref.moe_intermediate_size
        + 2 * hs * ref.routed_expert_hidden_size
        - ref.num_experts_per_token * 3 * hs * ref.moe_intermediate_size
    )
    assert params(ref) - params(no_latent) == pytest.approx(expected)

    no_gate = _checkpoint_config(mla_use_output_gate=False)
    assert params(ref) - params(no_gate) == pytest.approx(mla_layers * hs * ref.num_attention_heads * ref.v_head_dim)

    no_shared = _checkpoint_config(num_shared_experts=0)
    assert params(ref) - params(no_shared) == pytest.approx(moe_layers * 3 * hs * ref.moe_intermediate_size * 2)
