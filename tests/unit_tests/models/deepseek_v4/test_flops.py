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

"""Regression and independent arithmetic checks for DeepSeek V4 benchmark FLOPs."""

import math
from pathlib import Path

import pytest
import yaml

from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.flops import model_flops
from nemo_automodel.components.utils.flops_utils import get_flops_formula_for_hf_config, transformer_flops


def _config() -> DeepseekV4Config:
    return DeepseekV4Config(
        hidden_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=2,
        o_lora_rank=3,
        o_groups=2,
        moe_intermediate_size=5,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        hc_mult=2,
        vocab_size=19,
        compress_ratios=[0, 4, 128],
        sliding_window=3,
        index_n_heads=2,
        index_head_dim=2,
        index_topk=2,
        max_position_embeddings=9,
        num_nextn_predict_layers=0,
    )


def _oracle(length: int) -> int:
    # Actual matrix shapes for Q-A, Q-B, KV, grouped O-A and O-B per layer.
    shapes = [(2, 8), (8, 2), (4, 8), (6, 4), (8, 6)]
    # Two active routed experts, one shared expert, router, two HC projections.
    shapes += [(5, 8), (5, 8), (8, 5)] * 3 + [(4, 8), (8, 16), (8, 16)]
    shapes *= 3
    # HyperHead, LM head, overlapping and non-overlapping KV/gate compressors.
    shapes += [(2, 16), (19, 8), (8, 8), (8, 8), (4, 8), (4, 8)]
    result = 6 * length * sum(rows * columns for rows, columns in shapes)
    # Each of two residual mixers per layer multiplies [2, 2] by [2, 8].
    result += 6 * length * 3 * 2 * (2 * 2 * 8)
    # Frozen indexer: independent wkv/wgate, Q and head-weight projections.
    result += 2 * length * sum(rows * columns for rows, columns in [(4, 8), (4, 8), (4, 2), (2, 8)])
    for query in range(length):
        local_keys = [key for key in range(query + 1) if query - key < 3]
        for ratio in (0, 4, 128):
            compressed_keys = [] if ratio == 0 else list(range(ratio - 1, query + 1, ratio))
            if ratio == 4:
                # Score all causal pooled keys, then attend to at most two.
                result += 2 * 2 * 2 * len(compressed_keys)
                compressed_keys = compressed_keys[:2]
            # QK and PV: two attention heads, four elements per head.
            result += 6 * 2 * 2 * 4 * (len(local_keys) + len(compressed_keys))
    return result


@pytest.mark.parametrize("length", [1, 3, 4, 5, 9, 127, 128, 129])
def test_formula_matches_matrix_shapes_and_enumerated_causal_keys(length: int) -> None:
    config = _config()
    formula = get_flops_formula_for_hf_config(config)
    assert formula is model_flops
    assert formula(config, gbs=3, seq_len=length) == 3 * _oracle(length)


def test_failing_benchmark_recipe_uses_model_formula() -> None:
    recipe_path = (
        Path(__file__).resolve().parents[4] / "examples/llm_benchmark/deepseek/deepseek_v4_flash_mok_32gpu.yaml"
    )
    recipe = yaml.safe_load(recipe_path.read_text())
    config = DeepseekV4Config(**{key: value for key, value in recipe["model"]["config"].items() if key != "_target_"})
    with pytest.raises(AttributeError, match="intermediate_size"):
        transformer_flops(config, gbs=32, seq_len=1024)
    formula = get_flops_formula_for_hf_config(config)
    assert formula is model_flops
    result = formula(config, gbs=recipe["step_scheduler"]["global_batch_size"], seq_len=recipe["dataset"]["seq_len"])
    assert math.isfinite(result) and result > 0
    assert not hasattr(config, "intermediate_size")


def test_default_length_and_config_roundtrip() -> None:
    config = _config()
    assert model_flops(config) == _oracle(9)
    serialized = config.to_json_string()
    assert "flops_formula" not in serialized
    restored = DeepseekV4Config.from_dict(config.to_dict())
    assert get_flops_formula_for_hf_config(restored)(restored) == _oracle(9)


@pytest.mark.parametrize("gbs,length", [(0, 9), (1, 0), (1, -1)])
def test_invalid_batch_or_sequence_is_rejected(gbs: int, length: int) -> None:
    with pytest.raises(ValueError, match="positive seq_len and gbs"):
        model_flops(_config(), gbs=gbs, seq_len=length)


@pytest.mark.parametrize(
    "config", [DeepseekV4Config(), DeepseekV4Config(num_nextn_predict_layers=0, vision_n_layers=1)]
)
def test_unsupported_mtp_and_vision_are_explicit(config: DeepseekV4Config) -> None:
    with pytest.raises(ValueError, match="vision_n_layers=0 and num_nextn_predict_layers=0"):
        model_flops(config, seq_len=9)
