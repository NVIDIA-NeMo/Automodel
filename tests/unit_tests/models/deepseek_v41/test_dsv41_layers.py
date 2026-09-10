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

"""CPU unit tests for the DeepSeek V4.1 layer building blocks."""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v41.layers import (
    DeepseekV41Attention,
    DeepseekV41Compressor,
    DeepseekV41Indexer,
    DeepseekV41RotaryEmbedding,
    DeepseekV41SharedState,
    build_compressed_visibility,
    build_window_topk_indices,
    fake_quant_fp4,
    fake_quant_fp8,
    hc_collapse,
    hc_expand,
    make_identity_pre_mix,
    select_candidate_blocks,
)
from tests.unit_tests.models.deepseek_v41.conftest import tiny_backend, tiny_config

_E2M1_VALUES = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}


class TestFakeQuant:
    def test_fp4_values_land_on_the_e2m1_grid(self):
        torch.manual_seed(0)
        x = torch.randn(3, 64) * 4
        for scale_format, block in (("e8m0", 32), ("e4m3", 16)):
            q = fake_quant_fp4(x, block, scale_format)
            blocks = q.unflatten(-1, (-1, block))
            amax = x.unflatten(-1, (-1, block)).abs().amax(-1, keepdim=True)
            if scale_format == "e8m0":
                scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
            else:
                scale = (amax / 6.0).to(torch.float8_e4m3fn).float()
            grid_values = (blocks / scale).abs().unique().tolist()
            assert set(round(v, 6) for v in grid_values) <= _E2M1_VALUES
            assert (q - x).abs().max() <= scale.max() + 1e-6  # the widest grid step is 2 * scale

    def test_fp4_e8m0_scale_is_power_of_two_and_rounds_to_nearest_even(self):
        x = torch.tensor([[0.3, -5.0, 1.1, 2.6, 0.75, 1.75, 0.25, 3.5] * 4])
        q = fake_quant_fp4(x, 32, "e8m0")
        # amax 5.0 -> scale 2^ceil(log2(5/6)) = 1.0 -> values rounded on the raw grid.
        # Ties: 5.0 -> 4.0 (even mantissa), 0.75 -> 1.0, 1.75 -> 2.0, 0.25 -> 0.0, 3.5 -> 4.0.
        assert torch.equal(q, torch.tensor([[0.5, -4.0, 1.0, 3.0, 1.0, 2.0, 0.0, 4.0] * 4]))

    def test_fp8_roundtrip_matches_manual(self):
        torch.manual_seed(1)
        x = torch.randn(2, 5, 64, dtype=torch.bfloat16) * 3
        q = fake_quant_fp8(x, 32)
        blocks = x.float().unflatten(-1, (-1, 32))
        amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        expected = ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).flatten(-2).to(torch.bfloat16)
        assert torch.equal(q, expected)
        assert q.dtype == torch.bfloat16

    def test_straight_through_gradient(self):
        x = torch.randn(4, 32, requires_grad=True)
        fake_quant_fp4(x, 32, "e8m0").sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))
        x.grad = None
        (fake_quant_fp8(x, 32) * 2).sum().backward()
        assert torch.equal(x.grad, torch.full_like(x, 2.0))

    def test_block_size_mismatch_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            fake_quant_fp4(torch.zeros(2, 20), 16, "e4m3")
        with pytest.raises(ValueError, match="scale format"):
            fake_quant_fp4(torch.zeros(2, 32), 32, "int8")


class TestHyperConnectionMixing:
    def test_identity_pre_mix_reads_stream_zero(self):
        x = torch.randn(2, 3, 4, 8)
        pre_mix = make_identity_pre_mix(x, 4)
        assert torch.allclose(hc_collapse(x, pre_mix), x[:, :, 0])

    def test_expand_uses_transposed_comb(self):
        y = torch.randn(1, 2, 8)
        residual = torch.randn(1, 2, 4, 8)
        post = torch.rand(1, 2, 4)
        comb = torch.rand(1, 2, 4, 4)
        out = hc_expand(y, residual, post, comb)
        expected = post.unsqueeze(-1) * y.unsqueeze(-2) + torch.einsum("bsjh,bsjd->bshd", comb, residual)
        assert torch.allclose(out, expected, atol=1e-6)


class TestSparseIndexConstruction:
    def test_window_indices_single_document(self):
        seq_ids = torch.ones(1, 6, dtype=torch.long)
        idx = build_window_topk_indices(seq_ids, window_size=3)
        assert idx.shape == (1, 6, 3)
        assert idx[0, 0].tolist() == [0, -1, -1]
        assert idx[0, 1].tolist() == [0, 1, -1]
        assert idx[0, 5].tolist() == [3, 4, 5]

    def test_window_indices_respect_packed_documents_and_padding(self):
        seq_ids = torch.tensor([[1, 1, 2, 2, 2, 0]])
        idx = build_window_topk_indices(seq_ids, window_size=4)
        assert idx[0, 2].tolist() == [-1, -1, 2, -1]  # first token of doc 2 sees only itself
        assert idx[0, 4].tolist() == [-1, 2, 3, 4]
        assert (idx[0, 5] == -1).all()  # padding query attends nowhere

    def test_compressed_visibility(self):
        q_positions = torch.tensor([[0, 1, 2, 3, 0, 1]])
        q_seq_ids = torch.tensor([[1, 1, 1, 1, 2, 2]])
        pool_seq_ids = torch.tensor([[1, 1, 2]])
        pool_positions = torch.tensor([[0, 1, 0]])
        allowed = build_compressed_visibility(q_positions, q_seq_ids, pool_seq_ids, pool_positions, compress_ratio=2)
        expected = torch.tensor(
            [
                [
                    [False, False, False],  # pos 0: no complete group yet
                    [True, False, False],  # pos 1: group 0 complete
                    [True, False, False],
                    [True, True, False],  # pos 3: groups 0 and 1
                    [False, False, False],  # doc 2 pos 0
                    [False, False, True],  # doc 2 pos 1 sees its own group only
                ]
            ]
        )
        assert torch.equal(allowed, expected)

    def test_candidate_blocks_pin_newest_block_and_keep_top(self):
        width = 12
        scores = torch.zeros(1, 1, width)
        scores[0, 0, 1] = 9.0  # block 0 strongest
        scores[0, 0, 5] = 3.0  # block 1
        allowed = torch.zeros(1, 1, width, dtype=torch.bool)
        allowed[0, 0, :10] = True  # newest visible position 9 -> block 2 (positions 8..11)
        scores = scores.masked_fill(~allowed, float("-inf"))
        keep = select_candidate_blocks(scores, allowed, topk_blocks=2, block_size=4)
        assert keep[0, 0].tolist() == [True] * 4 + [False] * 4 + [True] * 4

    def test_candidate_blocks_with_nothing_visible(self):
        scores = torch.full((1, 1, 8), float("-inf"))
        allowed = torch.zeros(1, 1, 8, dtype=torch.bool)
        keep = select_candidate_blocks(scores, allowed, topk_blocks=1, block_size=4)
        assert not keep.any()


class TestCompressor:
    def test_ratio_two_pools_with_softmax_gate(self):
        config = tiny_config()
        comp = DeepseekV41Compressor(config, compress_ratio=2).float()
        x = torch.randn(1, 5, config.hidden_size)
        out = comp(x)
        assert out.shape == (1, 2, config.head_dim)  # trailing partial group dropped
        kv = comp.wkv(x[:, :4]).unflatten(1, (-1, 2))
        gate = comp.wgate(x[:, :4]).unflatten(1, (-1, 2)).softmax(dim=2)
        expected = comp.norm((kv * gate).sum(2))
        assert torch.allclose(out, expected, atol=1e-6)
        assert comp.wkv.weight.dtype == torch.float32 and comp.wgate.weight.dtype == torch.float32

    def test_ratio_one_is_a_projection(self):
        config = tiny_config()
        comp = DeepseekV41Compressor(config, compress_ratio=1).float()
        assert comp.wgate is None
        x = torch.randn(2, 3, config.hidden_size)
        assert torch.allclose(comp(x), comp.norm(comp.wkv(x)))

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            DeepseekV41Compressor(tiny_config(), compress_ratio=0)


def _rope_tables(config, positions):
    rotary = DeepseekV41RotaryEmbedding(
        rope_theta=config.compress_rope_theta,
        head_dim=config.head_dim,
        partial_rotary_factor=config.qk_rope_head_dim / config.head_dim,
        rope_scaling=config.rope_scaling,
    )
    return rotary, rotary(torch.zeros(1), positions)


class TestIndexer:
    def test_reindex_layer_uses_candidate_pool_and_masks(self):
        config = tiny_config(index_topk=3)
        indexer = DeepseekV41Indexer(config, layer_idx=5, backend=tiny_backend()).float()
        assert indexer.uses_candidates and not indexer.owns_k and indexer.wk is None
        batch, seq_len, pool = 1, 4, 8
        x = torch.randn(batch, seq_len, config.hidden_size)
        qr = torch.randn(batch, seq_len, config.q_lora_rank)
        _, (cos, sin) = _rope_tables(config, torch.arange(seq_len).unsqueeze(0))
        index_k = torch.randn(batch, pool, config.index_head_dim)
        allowed = torch.ones(batch, seq_len, pool, dtype=torch.bool)
        allowed[:, :, 6:] = False
        state = DeepseekV41SharedState(compress_ratio=1)
        state.candidates = torch.zeros(batch, seq_len, pool, dtype=torch.bool)
        state.candidates[:, :, [0, 3]] = True
        topk = indexer(x, qr, cos, sin, index_k, allowed, state)
        assert topk.shape == (batch, seq_len, 3)
        valid = topk[topk >= 0]
        assert set(valid.tolist()) <= {0, 3}
        assert (topk[:, :, 2] == -1).all()  # only two candidates are ever selectable

    def test_full_layer_publishes_candidates_and_keys(self):
        config = tiny_config(index_topk=4)
        indexer = DeepseekV41Indexer(config, layer_idx=4, backend=tiny_backend()).float()
        assert indexer.is_candidate_source and indexer.owns_k
        pool = 8
        latent = torch.randn(1, pool, config.head_dim)
        rotary, (cos_p, sin_p) = _rope_tables(config, torch.arange(pool).unsqueeze(0))
        keys = indexer.build_keys(latent, cos_p, sin_p)
        assert keys.shape == (1, pool, config.index_head_dim)
        seq_len = 3
        x = torch.randn(1, seq_len, config.hidden_size)
        qr = torch.randn(1, seq_len, config.q_lora_rank)
        _, (cos, sin) = _rope_tables(config, torch.arange(seq_len).unsqueeze(0))
        allowed = torch.ones(1, seq_len, pool, dtype=torch.bool)
        state = DeepseekV41SharedState(compress_ratio=1)
        topk = indexer(x, qr, cos, sin, keys, allowed, state)
        assert state.candidates is not None and state.candidates.shape == (1, seq_len, pool)
        # candidate_topk_blocks=2 blocks of 4 positions cover the whole pool here
        assert state.candidates.all()
        assert (topk >= 0).all()

    def test_missing_candidates_raise(self):
        config = tiny_config()
        indexer = DeepseekV41Indexer(config, layer_idx=5, backend=tiny_backend()).float()
        _, (cos, sin) = _rope_tables(config, torch.arange(2).unsqueeze(0))
        with pytest.raises(RuntimeError, match="candidate pool"):
            indexer(
                torch.randn(1, 2, config.hidden_size),
                torch.randn(1, 2, config.q_lora_rank),
                cos,
                sin,
                torch.randn(1, 4, config.index_head_dim),
                torch.ones(1, 2, 4, dtype=torch.bool),
                DeepseekV41SharedState(compress_ratio=1),
            )


class TestAttentionStateSharing:
    @staticmethod
    def _attention_inputs(config, seq_len):
        position_ids = torch.arange(seq_len).unsqueeze(0)
        rotary_main = DeepseekV41RotaryEmbedding(
            rope_theta=config.rope_theta,
            head_dim=config.head_dim,
            partial_rotary_factor=config.qk_rope_head_dim / config.head_dim,
        )
        rotary_compress, compress_tables = _rope_tables(config, position_ids)
        return dict(
            position_embeddings=rotary_main(torch.zeros(1), position_ids),
            position_embeddings_compress=compress_tables,
            rotary_compress=rotary_compress,
            position_ids=position_ids,
            seq_ids=torch.ones(1, seq_len, dtype=torch.long),
        )

    def test_full_then_reuse_share_compressed_kv(self):
        config = tiny_config()
        seq_len = 9
        full = DeepseekV41Attention(config, layer_idx=2, backend=tiny_backend()).float()
        reuse = DeepseekV41Attention(config, layer_idx=3, backend=tiny_backend()).float()
        assert full.compressor is not None and full.indexer is not None
        assert reuse.compressor is None and reuse.indexer is None
        state = DeepseekV41SharedState()
        kwargs = self._attention_inputs(config, seq_len)
        x = torch.randn(1, seq_len, config.hidden_size)
        out_full = full(x, state=state, **kwargs)
        assert out_full.shape == (1, seq_len, config.hidden_size)
        assert state.compress_ratio == 2
        assert state.compress_kv.shape == (1, seq_len // 2, config.head_dim)
        assert state.index_k.shape == (1, seq_len // 2, config.index_head_dim)
        assert state.topk_idxs.shape[:2] == (1, seq_len)
        # Query 0 has no complete group yet, so every compressed slot is empty.
        assert (state.topk_idxs[0, 0] == -1).all()
        out_reuse = reuse(x, state=state, **kwargs)
        assert out_reuse.shape == (1, seq_len, config.hidden_size)
        assert torch.isfinite(out_reuse).all()

    def test_reuse_without_source_raises(self):
        config = tiny_config()
        reuse = DeepseekV41Attention(config, layer_idx=3, backend=tiny_backend()).float()
        kwargs = self._attention_inputs(config, 4)
        with pytest.raises(RuntimeError, match="no matching compressed KV"):
            reuse(torch.randn(1, 4, config.hidden_size), state=DeepseekV41SharedState(), **kwargs)

    def test_ratio_mismatch_raises(self):
        config = tiny_config()
        full_ratio2 = DeepseekV41Attention(config, layer_idx=2, backend=tiny_backend()).float()
        reindex_ratio1 = DeepseekV41Attention(config, layer_idx=5, backend=tiny_backend()).float()
        kwargs = self._attention_inputs(config, 6)
        state = DeepseekV41SharedState()
        full_ratio2(torch.randn(1, 6, config.hidden_size), state=state, **kwargs)
        with pytest.raises(RuntimeError, match="no matching compressed KV"):
            reindex_ratio1(torch.randn(1, 6, config.hidden_size), state=state, **kwargs)

    def test_sliding_window_layer_is_causal(self):
        config = tiny_config()
        attn = DeepseekV41Attention(config, layer_idx=0, backend=tiny_backend()).float()
        seq_len = 6
        kwargs = self._attention_inputs(config, seq_len)
        x = torch.randn(1, seq_len, config.hidden_size)
        out = attn(x, state=DeepseekV41SharedState(), **kwargs)
        x2 = x.clone()
        x2[:, -1] += 1.0  # perturb the last token only
        out2 = attn(x2, state=DeepseekV41SharedState(), **kwargs)
        assert torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-5)
        assert not torch.allclose(out[:, -1], out2[:, -1])
