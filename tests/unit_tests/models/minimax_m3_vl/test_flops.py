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

"""Independent useful-FLOPs oracles for MiniMax M3's benchmark counter."""

import pytest

from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig, MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.flops import model_flops
from nemo_automodel.components.utils.flops_utils import get_flops_formula_for_hf_config, transformer_flops


def _config(*, sparse: bool = True, local_blocks: int = 1, mtp: int = 0) -> MiniMaxM3VLTextConfig:
    return MiniMaxM3VLTextConfig(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=5,
        dense_intermediate_size=11,
        shared_intermediate_size=7,
        num_local_experts=5,
        num_experts_per_tok=2,
        n_shared_experts=1,
        vocab_size=19,
        max_position_embeddings=12,
        moe_layer_freq=[0, 1],
        num_mtp_modules=mtp,
        sparse_attention_config={
            "use_sparse_attention": sparse,
            "sparse_attention_freq": [0, 1],
            "sparse_index_dim": 3,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 2,
            "sparse_local_block": local_blocks,
        },
    )


def _oracle(length: int, *, sparse: bool = True) -> int:
    # List the actual linear tensor shapes: every decoder has Q/K/V/O;
    # layer 0 has a dense gate/up/down; layer 1 has two routed experts,
    # one shared expert and a router. The LM head is separate.
    shapes = [(12, 8), (4, 8), (4, 8), (8, 12)] * 2
    shapes += [(11, 8), (11, 8), (8, 11)]
    shapes += [(5, 8), (5, 8), (8, 5)] * 2
    shapes += [(7, 8), (7, 8), (8, 7), (5, 8), (19, 8)]
    result = sum(rows * columns for rows, columns in shapes) * length * 6
    for query in range(length):
        valid_keys = list(range(query + 1))
        # Dense layer: QK^T and PV, each across three heads of dimension 4.
        result += len(valid_keys) * 3 * 4 * 2 * 6
        if not sparse:
            result += len(valid_keys) * 3 * 4 * 2 * 6
            continue
        # Independent selection oracle: enumerate causal blocks and give earlier
        # blocks decreasing scores. The current block is forced into top-2.
        blocks = {}
        for key in valid_keys:
            blocks.setdefault(key // 2, []).append(key)
        current = query // 2
        selected = [current] + [block for block in blocks if block != current][:1]
        selected_keys = [key for block in selected for key in blocks[block]]
        result += len(selected_keys) * 3 * 4 * 2 * 6
        # Selection-only projections (Q: 6 x 8, K: 3 x 8), and two
        # index heads of dimension 3 score all causal keys, forward only.
        result += ((6 * 8 + 3 * 8) + 2 * 3 * len(valid_keys)) * 2
    return result


@pytest.mark.parametrize("length", [1, 3, 4, 5, 6, 7, 12, 13])
@pytest.mark.parametrize("sparse", [False, True])
def test_counter_matches_independent_parameter_and_selected_key_oracle(length: int, sparse: bool) -> None:
    config = _config(sparse=sparse)
    formula = get_flops_formula_for_hf_config(config)
    assert formula is model_flops
    assert formula(config, gbs=3, seq_len=length) == 3 * _oracle(length, sparse=sparse)


def test_original_fallback_does_not_certify_m3_flops() -> None:
    config = _config()
    expected = _oracle(12)
    assert transformer_flops(config, gbs=1, seq_len=12) != expected
    assert get_flops_formula_for_hf_config(config)(config, gbs=1, seq_len=12) == expected


def test_sequence_default_and_composite_scope() -> None:
    config = _config()
    formula = get_flops_formula_for_hf_config(config)
    assert formula(config) == _oracle(12)
    # A text formula must not silently certify vision/projector work.
    wrapper = MiniMaxM3VLConfig(text_config=config)
    assert get_flops_formula_for_hf_config(wrapper) is None
    text = wrapper.get_text_config()
    assert get_flops_formula_for_hf_config(text)(text) == _oracle(12)


@pytest.mark.parametrize("gbs,length", [(0, 12), (1, 0), (1, -1)])
def test_invalid_batch_or_sequence_is_rejected(gbs: int, length: int) -> None:
    with pytest.raises(ValueError, match="positive seq_len and gbs"):
        model_flops(_config(), gbs=gbs, seq_len=length)


def test_mtp_and_nonlocal_sparse_budget_are_explicitly_unsupported() -> None:
    with pytest.raises(ValueError, match="num_mtp_modules=0"):
        model_flops(_config(mtp=1))
    with pytest.raises(ValueError, match="local block"):
        model_flops(_config(local_blocks=0))


def test_short_layer_patterns_and_unsupported_output_gate_are_rejected() -> None:
    config = _config()
    config.moe_layer_freq = [0]
    with pytest.raises(ValueError, match="moe_layer_freq"):
        model_flops(config)
    config = _config()
    config.sparse_attention_config["sparse_attention_freq"] = [0]
    with pytest.raises(ValueError, match="sparse_attention_freq"):
        model_flops(config)
    config = _config()
    config.attention_output_gate = True
    with pytest.raises(ValueError, match="attention_output_gate"):
        model_flops(config)


def test_forced_initial_blocks_must_leave_room_for_partial_local_block() -> None:
    config = _config()
    config.sparse_attention_config["sparse_init_block"] = 2
    with pytest.raises(ValueError, match="leave room"):
        model_flops(config)


def test_shared_expert_count_scales_only_shared_projection_work() -> None:
    config = _config()
    original = model_flops(config, seq_len=12)
    config.n_shared_experts = 2
    # Three additional hidden8/intermediate7 matrices, forward + backward.
    assert model_flops(config, seq_len=12) - original == 3 * 8 * 7 * 12 * 6


@pytest.mark.parametrize("local_blocks,initial_blocks", [(3, 0), (2, 1)])
def test_forced_blocks_cannot_overfill_topk_budget(local_blocks: int, initial_blocks: int) -> None:
    config = _config(local_blocks=local_blocks)
    config.sparse_attention_config["sparse_init_block"] = initial_blocks
    with pytest.raises(ValueError, match="fit the top-k budget"):
        model_flops(config)


@pytest.mark.parametrize("local_blocks,initial_blocks", [(2, 0), (1, 1)])
def test_valid_forced_block_mix_preserves_selected_key_count(local_blocks: int, initial_blocks: int) -> None:
    config = _config(local_blocks=local_blocks)
    config.sparse_attention_config["sparse_init_block"] = initial_blocks
    # The actual old full block may differ, but every selected old block has two
    # keys and the forced current block has the same causal partial length.
    assert model_flops(config, seq_len=13) == _oracle(13)
