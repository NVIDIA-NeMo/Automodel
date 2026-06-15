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

"""Unit tests for the Multi-Token-Prediction (MTP) FLOPs helper used in MFU.

Expected values are hand-derived from the FLOPs definitions below (independent of the
implementation), so the tests assert what is *correct*, not merely what the code emits.

Fixtures use ``gbs=2, seq=4`` -> ``num_tokens = gbs * seq = 8``, ``hidden_size H = 8``,
``vocab_size V = 32``. From the formulas in ``flops_utils``:

  * one MLP block      = 6 * gbs * seq * H * intermediate * 3   (intermediate=16)  = 18432
  * eh_proj / depth    = 6 * num_tokens * (2H) * H              = 6*8*16*8          = 6144
  * lm_head / depth    = 6 * num_tokens * H * V                 = 6*8*8*32          = 12288

``_nemotronh_mtp_flops`` returns, for N depths:
    block * (N if use_repeated_layer else 1) + N * eh_proj + N * lm_head
where ``block`` is the summed per-sublayer FLOPs of ``mtp_block_types``.
"""

import types

from nemo_automodel.components.utils.flops_utils import (
    _nemotronh_mlp_layer_flops,
    _nemotronh_moe_layer_flops,
    _nemotronh_mtp_flops,
)

GBS, SEQ = 2, 4  # num_tokens = 8

_ONE_MLP = 18432  # 6*2*4*8*16*3
_EH_PER_DEPTH = 6144  # 6*8*(2*8)*8
_LM_PER_DEPTH = 12288  # 6*8*8*32


def _mlp_cfg():
    return types.SimpleNamespace(hidden_size=8, intermediate_size=16, vocab_size=32)


def _moe_cfg():
    # Latent MoE (Ultra/Super-V3): experts operate in moe_latent_size space + fc1/fc2 latent proj.
    return types.SimpleNamespace(
        hidden_size=8,
        vocab_size=32,
        moe_latent_size=4,
        moe_intermediate_size=6,
        num_experts_per_tok=2,
        moe_shared_expert_intermediate_size=10,
        n_routed_experts=16,
    )


def test_per_block_and_fusion_baselines():
    """Sanity-check the building blocks the hand-derivation relies on."""
    assert _nemotronh_mlp_layer_flops(_mlp_cfg(), GBS, SEQ) == _ONE_MLP


def test_mtp_repeated_single_block():
    # ['mlp'], N=2, repeated: block = 18432*2 = 36864; +eh*2 (12288) +lm*2 (24576) = 73728
    cfg = _mlp_cfg()
    assert _nemotronh_mtp_flops(cfg, GBS, SEQ, 2, ["mlp"], True) == 73728


def test_repeated_equals_unrolled():
    """The repeated layer's compute is N x: 1 physical depth run N times == N physical depths run once."""
    cfg = _mlp_cfg()
    repeated = _nemotronh_mtp_flops(cfg, GBS, SEQ, 2, ["mlp"], True)
    unrolled = _nemotronh_mtp_flops(cfg, GBS, SEQ, 2, ["mlp", "mlp"], False)
    assert repeated == unrolled == 73728


def test_mtp_repeated_two_sublayer_depth():
    # ['mlp','mlp'] (2 sublayers/depth), N=2, repeated: block = (2*18432)*2 = 73728; +eh*2 +lm*2 = 110592
    cfg = _mlp_cfg()
    assert _nemotronh_mtp_flops(cfg, GBS, SEQ, 2, ["mlp", "mlp"], True) == 110592


def test_eh_proj_and_lm_head_scale_with_n():
    """eh_proj + lm_head are per-depth: dropping the block isolates them at N*(eh+lm)."""
    cfg = _mlp_cfg()
    # N=1, no block (empty) -> 0 (guarded); use a zero-flop check via difference instead:
    n1 = _nemotronh_mtp_flops(cfg, GBS, SEQ, 1, ["mlp"], True)
    n2 = _nemotronh_mtp_flops(cfg, GBS, SEQ, 2, ["mlp"], True)
    # going 1 -> 2 depths (repeated) adds exactly one more (block + eh + lm)
    assert n2 - n1 == _ONE_MLP + _EH_PER_DEPTH + _LM_PER_DEPTH


def test_mtp_moe_block_counts_latent():
    cfg = _moe_cfg()
    # latent MoE block (hand-derived): routed 4608 + shared 7680 + gate 6144 + latent_proj 3072 = 21504
    assert _nemotronh_moe_layer_flops(cfg, GBS, SEQ) == 21504
    # ['moe'], N=2, repeated: 21504*2 + eh*2 + lm*2 = 43008 + 12288 + 24576 = 79872
    assert _nemotronh_mtp_flops(cfg, GBS, SEQ, 2, ["moe"], True) == 79872


def test_mtp_disabled_returns_zero():
    cfg = _mlp_cfg()
    assert _nemotronh_mtp_flops(cfg, GBS, SEQ, 0, ["mlp"], True) == 0
    assert _nemotronh_mtp_flops(cfg, GBS, SEQ, 2, [], True) == 0
