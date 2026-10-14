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

"""Check MiMo FLOPs against explicit projection shapes and visible key pairs."""

import pytest

from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2Config, MiMoV2FlashConfig
from nemo_automodel.components.models.mimo_v2_flash.flops import model_flops
from nemo_automodel.components.utils.flops_utils import get_flops_formula_for_hf_config, transformer_flops


def _config(config_class: type[MiMoV2FlashConfig] = MiMoV2Config) -> MiMoV2FlashConfig:
    return config_class(
        hidden_size=12,
        vocab_size=17,
        intermediate_size=11,
        moe_intermediate_size=5,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=4,
        v_head_dim=2,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=3,
        swa_v_head_dim=2,
        n_routed_experts=7,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_layer_freq=[0, 1],
        hybrid_layer_pattern=[0, 1],
        sliding_window=3,
        max_position_embeddings=7,
    )


def _oracle(length: int) -> int:
    # Full-attention Q/K/V/O, sliding-attention Q/K/V/O, dense MLP,
    # two selected + one shared expert, router, and LM head weight shapes.
    shapes = [(12, 12), (4, 12), (2, 12), (12, 6)]
    shapes += [(12, 12), (6, 12), (4, 12), (12, 8)]
    shapes += [(11, 12), (11, 12), (12, 11)]
    shapes += [(5, 12), (5, 12), (12, 5)] * 3
    shapes += [(7, 12), (17, 12)]
    result = 6 * length * sum(rows * columns for rows, columns in shapes)
    for query in range(length):
        for key in range(length):
            if key <= query:
                result += 6 * 3 * (4 + 2)
            if key <= query and query - key < 3:
                result += 6 * 4 * (3 + 2)
    return result


@pytest.mark.parametrize("config_class", [MiMoV2FlashConfig, MiMoV2Config])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 7, 11])
def test_matches_projection_shape_and_visible_pair_oracle(config_class: type[MiMoV2FlashConfig], length: int) -> None:
    config = _config(config_class)
    formula = get_flops_formula_for_hf_config(config)
    assert formula is model_flops
    assert formula(config, gbs=3, seq_len=length) == 3 * _oracle(length)


def test_dense_fallback_does_not_count_mimo_architecture() -> None:
    config = _config()
    assert transformer_flops(config, gbs=1, seq_len=7) != _oracle(7)
    assert model_flops(config) == _oracle(7)


@pytest.mark.parametrize("gbs,length", [(0, 7), (1, 0), (1, -1)])
def test_invalid_dimensions(gbs: int, length: int) -> None:
    with pytest.raises(ValueError, match="positive seq_len and gbs"):
        model_flops(_config(), gbs=gbs, seq_len=length)


def test_mtp_is_not_silently_omitted() -> None:
    config = MiMoV2Config(num_hidden_layers=2, num_nextn_predict_layers=1)
    with pytest.raises(ValueError, match="num_nextn_predict_layers=0"):
        model_flops(config, seq_len=7)


@pytest.mark.parametrize("window", [None, 0, -1])
def test_invalid_sliding_window(window: int | None) -> None:
    config = MiMoV2Config(num_hidden_layers=2, hybrid_layer_pattern=[1, 1], sliding_window=window)
    with pytest.raises(ValueError, match="positive sliding_window"):
        model_flops(config, seq_len=7)
