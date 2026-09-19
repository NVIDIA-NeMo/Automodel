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

"""Tests for deepseekv4_flops (DeepSeek-V4: shared-KV MQA, hybrid SWA/CSA/HCA attention, all-MoE, mHC)."""

from types import SimpleNamespace

from nemo_automodel.components.utils import flops_utils


def _v4_cfg(**overrides) -> SimpleNamespace:
    """Small DeepSeek-V4-style config: 4 layers [SWA, SWA, CSA, HCA]."""
    cfg = SimpleNamespace(
        hidden_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        head_dim=256,
        q_lora_rank=256,
        o_lora_rank=512,
        o_groups=1,
        vocab_size=4096,
        sliding_window=128,
        hc_mult=4,
        moe_intermediate_size=384,
        num_experts_per_tok=6,
        n_shared_experts=1,
        n_routed_experts=32,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        num_nextn_predict_layers=0,
        compress_ratios=[0, 0, 4, 128],
        max_position_embeddings=4096,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _hand_computed(cfg, seq_len):
    """Independent re-derivation of the formula for the 4-layer config above."""
    hs, H, hd, S = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim, seq_len
    win = cfg.sliding_window
    window_pairs = win * (win + 1) / 2 + (S - win) * win
    ratios = cfg.compress_ratios
    bmm = 0.0
    for r in ratios:
        pairs = window_pairs
        idx = 0.0
        if r == 4:
            pairs += sum(min((t + 1) // 4, cfg.index_topk) for t in range(S))
            n = S // 4
            idx = cfg.index_n_heads * cfg.index_head_dim * (n * (n + 1) / 2) * 4
        elif r:
            n = S // r
            pairs += n * (n + 1) / 2 * r
        bmm += 2 * H * hd * pairs + idx
    attn_core = hs * cfg.q_lora_rank + cfg.q_lora_rank * H * hd + hs * hd + H * hd * cfg.o_lora_rank
    attn_core += cfg.o_groups * cfg.o_lora_rank * hs
    mhc = 2 * (2 + cfg.hc_mult) * cfg.hc_mult * cfg.hc_mult * hs
    ffn = (cfg.num_experts_per_tok + cfg.n_shared_experts) * 3 * hs * cfg.moe_intermediate_size
    params = 0
    for r in ratios:
        p = attn_core + mhc + ffn
        if r:
            p += 2 * hs * (2 if r == 4 else 1) * hd
            if r == 4:
                p += (
                    cfg.q_lora_rank * cfg.index_n_heads * cfg.index_head_dim
                    + hs * cfg.index_n_heads
                    + 4 * hs * cfg.index_head_dim
                )
        params += p
    return 6 * bmm + 6 * params * S + 6 * cfg.vocab_size * hs * S


class TestDeepseekV4Flops:
    def test_matches_hand_computation(self):
        cfg = _v4_cfg()
        assert flops_utils.deepseekv4_flops(cfg, gbs=1, seq_len=1024) == _hand_computed(cfg, 1024)

    def test_scales_with_batch(self):
        cfg = _v4_cfg()
        one = flops_utils.deepseekv4_flops(cfg, gbs=1, seq_len=512)
        assert flops_utils.deepseekv4_flops(cfg, gbs=3, seq_len=512) == 3 * one

    def test_layer_kinds_order_costs(self):
        """A CSA layer (compressor + indexer) costs more than HCA, which costs more than sliding-window only."""
        swa = flops_utils.deepseekv4_flops(_v4_cfg(num_hidden_layers=1, compress_ratios=[0]), seq_len=2048)
        hca = flops_utils.deepseekv4_flops(_v4_cfg(num_hidden_layers=1, compress_ratios=[128]), seq_len=2048)
        csa = flops_utils.deepseekv4_flops(_v4_cfg(num_hidden_layers=1, compress_ratios=[4]), seq_len=2048)
        assert swa < hca < csa

    def test_transformers_layer_types_flavour(self):
        """transformers' config exposes layer_types + compress_rates instead of compress_ratios."""
        am = _v4_cfg()
        hf = _v4_cfg()
        del hf.compress_ratios
        hf.layer_types = [
            "sliding_attention",
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ]
        hf.compress_rates = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
        assert flops_utils.deepseekv4_flops(hf, seq_len=1024) == flops_utils.deepseekv4_flops(am, seq_len=1024)

    def test_mtp_adds_layer_and_head(self):
        base = flops_utils.deepseekv4_flops(_v4_cfg(), seq_len=1024)
        with_mtp = flops_utils.deepseekv4_flops(_v4_cfg(num_nextn_predict_layers=1), seq_len=1024)
        assert with_mtp > base

    def test_dispatch(self):
        cfg = type("DeepseekV4Config", (), {})()
        assert flops_utils.get_flops_formula_for_hf_config(cfg) == flops_utils.deepseekv4_flops
