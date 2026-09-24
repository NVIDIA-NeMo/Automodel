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

"""Check MiMo model FLOPs against module weights and enumerated causal pairs."""

import json

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2FlashConfig
from nemo_automodel.components.models.mimo_v2_flash.flops import mimo_v2_flops
from nemo_automodel.components.models.mimo_v2_flash.model import MiMoV2FlashForCausalLM
from nemo_automodel.components.utils.flops_utils import calculate_mfu, get_flops_formula_for_hf_config


def _tiny_config(**overrides):
    values = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        v_head_dim=4,
        swa_num_attention_heads=6,
        swa_num_key_value_heads=3,
        swa_head_dim=4,
        swa_v_head_dim=2,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 1],
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        partial_rotary_factor=1.0,
        sliding_window=3,
        max_position_embeddings=8,
        torch_dtype="float32",
    )
    return MiMoV2FlashConfig(**(values | overrides))


def test_config_registration_survives_round_trip(tmp_path):
    config = _tiny_config()
    assert get_flops_formula_for_hf_config(config) is mimo_v2_flops
    config.save_pretrained(tmp_path)
    serialized = json.loads((tmp_path / "config.json").read_text())
    assert "flops_formula" not in serialized
    restored = MiMoV2FlashConfig.from_pretrained(tmp_path)
    assert get_flops_formula_for_hf_config(restored) is mimo_v2_flops
    assert mimo_v2_flops(restored) == mimo_v2_flops(config, seq_len=8)


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("n_routed_experts", [None, 4, 8])
@pytest.mark.parametrize("n_shared_experts", [None, 1])
def test_flops_match_model_weight_shapes_and_visible_pairs(seq_len, n_routed_experts, n_shared_experts):
    """The reference counts actual model matrices and explicitly visits visible query/key pairs."""
    config = _tiny_config(n_routed_experts=n_routed_experts, n_shared_experts=n_shared_experts)
    backend = BackendConfig(
        linear="torch",
        attn="eager",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )
    with torch.device("meta"):
        model = MiMoV2FlashForCausalLM(config, backend=backend)

    active_weights = 0
    for name, param in model.named_parameters():
        if param.ndim < 2 or name == "model.embed_tokens.weight":
            continue
        if ".experts." in name:
            assert param.shape[0] == n_routed_experts
            active_weights += param.numel() // n_routed_experts * config.num_experts_per_tok
        else:
            active_weights += param.numel()

    # Derive BMM dimensions from the attention modules rather than config arithmetic.
    attention_macs = 0
    for layer in model.model.layers.values():
        attn = layer.self_attn
        for query in range(seq_len):
            for key in range(seq_len):
                visible = key <= query and (not attn.is_swa or query - key < config.sliding_window)
                if visible:
                    attention_macs += attn.num_attention_heads * (attn.head_dim + attn.v_head_dim)

    expected = 3 * 2 * (seq_len * active_weights + attention_macs)
    assert mimo_v2_flops(config, gbs=3, seq_len=seq_len) == 3 * expected


@pytest.mark.parametrize(
    ("seq_len", "expected"),
    [(4096, 48_125_875_807_322_112), (8192, 98_631_682_583_691_264)],
)
def test_full_v26_checkpoint_flops(seq_len, expected):
    # XiaomiMiMo/MiMo-V2.6-Flash-RL config.json, revision
    # 5711b268169967567844e1e560e8a3966da959b1. MTP/encoders are not used by the benchmark.
    config = MiMoV2FlashConfig(
        vocab_size=152576,
        hidden_size=4096,
        num_hidden_layers=48,
        intermediate_size=16384,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        num_experts_per_tok=8,
        n_shared_experts=None,
        moe_layer_freq=[0] + [1] * 47,
        hybrid_layer_pattern=[0 if i in (0, 5, 11, 17, 23, 29, 35, 41, 47) else 1 for i in range(48)],
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=192,
        v_head_dim=128,
        swa_num_attention_heads=64,
        swa_num_key_value_heads=8,
        swa_head_dim=192,
        swa_v_head_dim=128,
        sliding_window=128,
    )
    formula = get_flops_formula_for_hf_config(config)
    assert formula(config, gbs=128, seq_len=seq_len) == expected
    assert formula(config, gbs=1, seq_len=seq_len) * 128 == expected
    if seq_len == 4096:
        # Slowest-rank iteration durations, mean of steps 10..29 of the 4K run.
        assert calculate_mfu(expected / 1e12, 64, 3.3088171, reference_mfu=989) == pytest.approx(22.978920302894988)


def test_layer_patterns_are_truncated_to_executed_depth():
    full_pattern = _tiny_config(hybrid_layer_pattern=[0, 1, 0, 1], moe_layer_freq=[0, 1, 1, 1])
    assert mimo_v2_flops(full_pattern, seq_len=8) == mimo_v2_flops(_tiny_config(), seq_len=8)


@pytest.mark.parametrize(("gbs", "seq_len"), [(0, 8), (1, 0), (-1, 8), (1, -1)])
def test_invalid_batch_or_sequence_length(gbs, seq_len):
    with pytest.raises(ValueError, match="must be positive"):
        mimo_v2_flops(_tiny_config(), gbs=gbs, seq_len=seq_len)


@pytest.mark.parametrize("window", [None, 0, -1])
def test_invalid_sliding_window(window):
    with pytest.raises(ValueError, match="sliding_window must be positive"):
        mimo_v2_flops(_tiny_config(sliding_window=window), seq_len=8)
