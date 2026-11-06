# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent shape and causal-edge oracles for GLM-5.3 text model FLOPs."""

import pytest

from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.flops import glm5_next_flops
from nemo_automodel.components.utils.flops_utils import get_flops_formula_for_hf_config


def _tiny_config(attention: str, *, select_tail: bool = True, mlp: str = "dense") -> Glm5NextTextConfig:
    """Build a one-layer graph small enough to count each matrix by hand."""
    return Glm5NextTextConfig(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=3,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        vocab_size=17,
        hc_mult=2,
        linear_num_heads=3,
        linear_head_dim=2,
        linear_conv_kernel_dim=4,
        index_n_heads=3,
        index_head_dim=2,
        index_topk=4,
        index_kpool=2,
        index_kpool_always_select_tail=select_tail,
        layer_types=[attention],
        mlp_layer_types=[mlp],
    )


def test_text_registration_requires_explicit_multimodal_scope() -> None:
    """The benchmark must opt into text FLOPs instead of claiming vision coverage."""
    config = Glm5NextTextConfig(index_kpool=4)
    formula = get_flops_formula_for_hf_config(config)
    assert formula is glm5_next_flops
    assert get_flops_formula_for_hf_config(Glm5NextConfig(text_config=config)) is None


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 5, 6, 9, 12, 33])
@pytest.mark.parametrize("select_tail", [False, True])
def test_sparse_graph_matches_enumerated_causal_edges(seq_len: int, select_tail: bool) -> None:
    """Count actual visible pools/keys, including incomplete and saturated pools."""
    config = _tiny_config("deepseek_sparse_attention", select_tail=select_tail)
    formula = get_flops_formula_for_hf_config(config)
    assert formula is not None
    selected_edges = 0
    score_edges = 0
    for query in range(seq_len):
        pools = [tuple(range(start, start + 2)) for start in range(0, query, 2)]
        score_edges += len(pools)
        selected_edges += sum(len(pool) for pool in pools[:2])
        if select_tail:
            covered = {key for pool in pools for key in pool}
            selected_edges += len(set(range(query + 1)) - covered)
    # Trainable matrix MACs per token, counted from the toy graph's shapes:
    # MLA: 8x3 + 3x8 + 8x4 + 4x16 + 8x8 = 208.
    # Dense SwiGLU: 3x8x12 = 288; LM head: 8x17 = 136.
    # Two mHCs: 2 x (16x8 coefficient + 2x8 collapse + 2x2x8 mix) = 352.
    # Forward-only indexer: 3x6 + 8x2 + 8x2 + 8x3 = 74 MACs.
    # Each selected attention edge: train 6x2headsx(4+4) = 96 FLOPs.
    # Each visible indexer pool edge: forward 2x3headsx(2+1) = 18 FLOPs.
    expected = seq_len * (6 * (208 + 288 + 136 + 352) + 2 * 74)
    expected += 96 * selected_edges + 18 * score_edges
    assert formula(config, gbs=1, seq_len=seq_len) == expected


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 5, 64, 65])
def test_kda_counts_recurrent_math_and_causal_convolution(seq_len: int) -> None:
    """Chunk size and padded convolution taps cannot increase model FLOPs."""
    config = _tiny_config("linear_attention")
    formula = get_flops_formula_for_hf_config(config)
    assert formula is not None
    # KDA projections: four 8x6, two (8x2 + 2x6), and one 8x3 = 272 MACs.
    # Other matrices: dense=288, head=136, mHC=352 MACs per token.
    # Three recurrent heads each perform three 2x2 contractions in forward;
    # forward + backward = 3 x 3 x 2 x 2 x 2 x 3 = 216 FLOPs per token.
    expected = seq_len * (6 * (272 + 288 + 136 + 352) + 216)
    # Three depthwise convolutions of width 6, with only causal taps counted.
    valid_taps = sum(len(range(max(0, token - 3), token + 1)) for token in range(seq_len))
    expected += 6 * 3 * 6 * valid_taps
    assert formula(config, gbs=1, seq_len=seq_len) == expected


@pytest.mark.parametrize(
    ("seq_len", "expected"),
    [(4096, 426711186014208), (8192, 858343226671104), (262144, 28363446805659648)],
)
def test_released_model_matches_independently_audited_totals(seq_len: int, expected: int) -> None:
    """Freeze the audit's recurrent-core convention before the first benchmark."""
    config = Glm5NextTextConfig(index_kpool=4)
    formula = get_flops_formula_for_hf_config(config)
    assert formula is not None
    assert formula(config, gbs=1, seq_len=seq_len) == expected
    assert formula(config, gbs=8, seq_len=seq_len) == 8 * expected


def test_layer_pattern_uses_real_dense_and_moe_layers() -> None:
    """Replacing one dense MLP adds top-k plus shared experts and its router."""
    dense = _tiny_config("linear_attention")
    sparse = _tiny_config("linear_attention", mlp="sparse")
    formula = get_flops_formula_for_hf_config(dense)
    assert formula is not None
    seq_len = 7
    # The attention, indexer, mHC, head, and all other settings cancel.
    expected_delta = (
        6
        * seq_len
        * (
            3 * 8 * sparse.moe_intermediate_size * (sparse.num_experts_per_tok + sparse.n_shared_experts)
            + 8 * sparse.n_routed_experts
            - 3 * 8 * 12
        )
    )
    assert formula(sparse, gbs=1, seq_len=seq_len) - formula(dense, gbs=1, seq_len=seq_len) == expected_delta


def test_checkpoint_mtp_metadata_does_not_count_an_unexecuted_head() -> None:
    """This backbone does not execute the checkpoint's next-token prediction layer."""
    config = Glm5NextTextConfig(index_kpool=4, num_nextn_predict_layers=0)
    unused_mtp = Glm5NextTextConfig(index_kpool=4, num_nextn_predict_layers=1)
    formula = get_flops_formula_for_hf_config(config)
    assert formula is not None
    assert formula(config, gbs=1, seq_len=4096) == formula(unused_mtp, gbs=1, seq_len=4096)
