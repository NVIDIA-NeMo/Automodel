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

"""Independent useful-FLOPs oracle for Mistral Large 4's benchmark counter."""

import pytest

from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config
from nemo_automodel.components.models.mistral_large4.flops import model_flops
from nemo_automodel.components.utils.flops_utils import get_flops_formula_for_hf_config

HIDDEN = 8
HEADS = 2
LATENT = 3
ROPE = 2
NOPE = 2
VALUE = 3
TOPK = 4
INDEX_HEADS = 2
INDEX_DIM = 3


def _config() -> MistralLarge4Config:
    return MistralLarge4Config(
        hidden_size=HIDDEN,
        intermediate_size=5,
        num_hidden_layers=6,
        num_attention_heads=HEADS,
        vocab_size=19,
        kv_lora_rank=LATENT,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=VALUE,
        max_position_embeddings=12,
        nope_layer_flags=[False, False, False, True, False, True],
        kv_source_layers=[0, 1, 2, 3, 4, 3],
        index_sharing=["full", "full", "shared", "full", "shared", "full"],
        index_topk=TOPK,
        index_n_heads=INDEX_HEADS,
        index_head_dim=INDEX_DIM,
        first_k_dense_replace=1,
        n_routed_experts=5,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
    )


def _layer_shapes() -> list[tuple[int, int]]:
    """List the (rows, columns) of every trainable GEMM actually present in the model."""
    head_dim = NOPE + ROPE
    shapes: list[tuple[int, int]] = []
    for layer in range(6):
        nope_layer = layer in (3, 5)
        shares_kv = layer == 5
        shapes.append((HEADS * head_dim, HIDDEN))  # q_proj
        if not shares_kv:
            shapes.append((LATENT + (0 if nope_layer else ROPE), HIDDEN))  # kv_a
        shapes.append((HEADS * ((head_dim if nope_layer else NOPE) + VALUE), LATENT))  # kv_b
        shapes.append((HIDDEN, HEADS * VALUE))  # o_proj
        if layer == 0:
            shapes += [(5, HIDDEN), (5, HIDDEN), (HIDDEN, 5)]  # dense gate/up/down
        else:
            shapes += [(4, HIDDEN), (4, HIDDEN), (HIDDEN, 4)] * 2  # two routed experts
            shapes += [(4, HIDDEN), (4, HIDDEN), (HIDDEN, 4)]  # one shared expert
            shapes.append((5, HIDDEN))  # router
    shapes.append((19, HIDDEN))  # lm_head
    return shapes


def _oracle(length: int) -> int:
    head_dim = NOPE + ROPE
    result = sum(rows * columns for rows, columns in _layer_shapes()) * length * 6
    full_indexer_layers = (0, 1, 3, 5)
    for query in range(length):
        causal_keys = list(range(query + 1))
        # The hard selector keeps the TOPK most recent keys; any set of equal size is equivalent
        # for counting, and shorter prefixes keep every key.
        selected = causal_keys[-TOPK:]
        result += 6 * 6 * len(selected) * HEADS * (head_dim + VALUE)  # six layers, QK^T and PV
        # Frozen selector, forward only: wq_b, wk and weights_proj, then scores over all causal keys.
        selector = INDEX_HEADS * INDEX_DIM * HIDDEN + INDEX_DIM * HIDDEN + INDEX_HEADS * HIDDEN
        selector += INDEX_HEADS * INDEX_DIM * len(causal_keys)
        result += 2 * len(full_indexer_layers) * selector
    return result


@pytest.mark.parametrize("length", [1, 3, 4, 5, 8, 12, 13])
def test_counter_matches_independent_shape_and_selected_key_oracle(length: int) -> None:
    config = _config()
    formula = get_flops_formula_for_hf_config(config)
    assert formula is model_flops
    assert formula(config, gbs=3, seq_len=length) == 3 * _oracle(length)


def test_sequence_defaults_to_max_position_embeddings() -> None:
    config = _config()
    assert model_flops(config) == _oracle(12)


@pytest.mark.parametrize("gbs,length", [(0, 12), (1, 0), (1, -1)])
def test_invalid_batch_or_sequence_is_rejected(gbs: int, length: int) -> None:
    with pytest.raises(ValueError, match="positive seq_len and gbs"):
        model_flops(_config(), gbs=gbs, seq_len=length)
