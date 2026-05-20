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

"""Unit tests for the standalone helpers in
``nemo_automodel.components.models.deepseek_v4.layers``.

Pieces here are easy to construct in isolation (grouped output projection,
the Hyper-Connections weight builder, and the partial-RoPE helper).
Full-model behaviour is covered by ``test_dsv4_model_smoke.py``.
"""

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4Attention,
    DeepseekV4GroupedLinear,
    DeepseekV4HyperConnection,
    DeepseekV4RotaryEmbedding,
    _apply_partial_rope_interleaved,
    _build_indexer_topk_compressed_mask,
    _yarn_correction_dim,
    _yarn_correction_range,
    _yarn_linear_ramp,
    build_causal_padding_mask,
)


class TestDeepseekV4AttentionMask:
    def test_indexer_topk_mask_preserves_pool_zero_with_clamped_invalid_entries(self):
        """A valid pool index 0 must survive duplicate writes from clamped ``-1`` slots."""
        attention_mask = torch.zeros(1, 1, 1, 1)
        indexer_topk = torch.tensor([[[2, 0, -1, -1]]])

        min_val = torch.finfo(attention_mask.dtype).min
        expected_compressed_mask = torch.tensor([[[[0.0, min_val, 0.0, min_val, min_val]]]])
        compressed_mask = _build_indexer_topk_compressed_mask(attention_mask, indexer_topk, n_pooled=5).unsqueeze(1)
        torch.testing.assert_close(compressed_mask, expected_compressed_mask)

    def test_short_hca_training_window_stays_disabled_without_group_hca(self):
        """All-short groups should keep the original no-HCA path and grad=None semantics."""
        torch.manual_seed(1234)
        cfg = self._tiny_hca_config()
        seq_len = 7
        hidden_states, position_embeddings, position_embeddings_compress, rotary_compress, attention_mask = (
            self._hca_inputs(cfg, seq_len)
        )

        attention = DeepseekV4Attention(cfg, layer_idx=0)
        attention.init_weights(torch.device("cpu"))
        attention_ref = DeepseekV4Attention(cfg, layer_idx=0)
        attention_ref.load_state_dict(attention.state_dict())

        attention_ref.eval()
        with torch.no_grad():
            expected, expected_weights = attention_ref(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_embeddings_compress=position_embeddings_compress,
                rotary_compress=rotary_compress,
            )

        attention.train()
        actual, actual_weights = attention(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_embeddings_compress=position_embeddings_compress,
            rotary_compress=rotary_compress,
        )

        assert expected_weights.shape[-1] == seq_len
        assert actual_weights.shape[-1] == seq_len
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        actual.square().sum().backward()

        compressor = attention.compressor
        assert compressor is not None
        params = list(compressor.named_parameters())
        before = {name: param.detach().clone() for name, param in params}
        for name, param in params:
            assert param.grad is None, name

        optimizer = torch.optim.AdamW([param for _, param in params], lr=1e-3, weight_decay=0.1)
        optimizer.step()
        for name, param in params:
            torch.testing.assert_close(param, before[name], atol=0.0, rtol=0.0)

    def test_short_hca_training_window_is_fully_masked_when_group_has_hca(self, monkeypatch):
        """Mixed short/long groups should mask the synthetic HCA position completely."""
        torch.manual_seed(1234)
        cfg = self._tiny_hca_config()
        seq_len = 7
        hidden_states, position_embeddings, position_embeddings_compress, rotary_compress, attention_mask = (
            self._hca_inputs(cfg, seq_len)
        )

        attention = DeepseekV4Attention(cfg, layer_idx=0)
        attention.init_weights(torch.device("cpu"))
        attention_ref = DeepseekV4Attention(cfg, layer_idx=0)
        attention_ref.load_state_dict(attention.state_dict())

        attention_ref.eval()
        with torch.no_grad():
            expected, expected_weights = attention_ref(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_embeddings_compress=position_embeddings_compress,
                rotary_compress=rotary_compress,
            )

        attention.train()
        compressor = attention.compressor
        assert compressor is not None
        monkeypatch.setattr(
            compressor,
            "_compute_fsdp_group_has_complete_hca_window",
            lambda local_has_complete_hca_window, device: True,
        )
        actual, actual_weights = attention(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_embeddings_compress=position_embeddings_compress,
            rotary_compress=rotary_compress,
        )

        assert expected_weights.shape[-1] == seq_len
        assert actual_weights.shape[-1] == seq_len + 1
        torch.testing.assert_close(
            actual_weights[..., -1], torch.zeros_like(actual_weights[..., -1]), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_short_hca_training_window_keeps_compressor_in_backward(self, monkeypatch):
        """Mixed-group short HCA ranks should produce zero-valued compressor gradients."""
        torch.manual_seed(1234)
        cfg = self._tiny_hca_config()
        seq_len = 7
        hidden_states, position_embeddings, position_embeddings_compress, rotary_compress, attention_mask = (
            self._hca_inputs(cfg, seq_len)
        )
        attention = DeepseekV4Attention(cfg, layer_idx=0)
        attention.init_weights(torch.device("cpu"))
        attention.train()
        compressor = attention.compressor
        assert compressor is not None
        monkeypatch.setattr(
            compressor,
            "_compute_fsdp_group_has_complete_hca_window",
            lambda local_has_complete_hca_window, device: True,
        )

        output, _ = attention(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_embeddings_compress=position_embeddings_compress,
            rotary_compress=rotary_compress,
        )
        output.square().sum().backward()

        for name, param in compressor.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
            torch.testing.assert_close(param.grad, torch.zeros_like(param.grad), atol=0.0, rtol=0.0)

    def test_short_hca_training_window_stays_disabled_without_attention_mask(self, monkeypatch):
        """Synthetic HCA alignment requires a mask so the extra position can be hidden."""
        torch.manual_seed(1234)
        cfg = self._tiny_hca_config()
        seq_len = 7
        hidden_states, position_embeddings, position_embeddings_compress, rotary_compress, _ = self._hca_inputs(
            cfg, seq_len
        )
        attention = DeepseekV4Attention(cfg, layer_idx=0)
        attention.init_weights(torch.device("cpu"))
        attention.train()
        compressor = attention.compressor
        assert compressor is not None
        monkeypatch.setattr(
            compressor,
            "_compute_fsdp_group_has_complete_hca_window",
            lambda local_has_complete_hca_window, device: True,
        )

        _, actual_weights = attention(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_embeddings_compress=position_embeddings_compress,
            rotary_compress=rotary_compress,
        )

        assert actual_weights.shape[-1] == seq_len

    @pytest.mark.parametrize(
        ("seq_len", "group_has_hca", "expected_attention_width"),
        (
            (127, False, 127),
            (127, True, 128),
            (128, False, 129),
            (129, False, 130),
        ),
    )
    def test_hca_window_boundary_paths(self, monkeypatch, seq_len, group_has_hca, expected_attention_width):
        torch.manual_seed(1234)
        cfg = self._tiny_hca_config()
        hidden_states, position_embeddings, position_embeddings_compress, rotary_compress, attention_mask = (
            self._hca_inputs(cfg, seq_len)
        )
        attention = DeepseekV4Attention(cfg, layer_idx=0)
        attention.init_weights(torch.device("cpu"))
        attention.train()
        if group_has_hca:
            compressor = attention.compressor
            assert compressor is not None
            monkeypatch.setattr(
                compressor,
                "_compute_fsdp_group_has_complete_hca_window",
                lambda local_has_complete_hca_window, device: True,
            )

        _, actual_weights = attention(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_embeddings_compress=position_embeddings_compress,
            rotary_compress=rotary_compress,
        )

        assert actual_weights.shape[-1] == expected_attention_width

    @staticmethod
    def _tiny_hca_config() -> DeepseekV4Config:
        return DeepseekV4Config(
            vocab_size=32,
            hidden_size=16,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            qk_rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=8,
            o_groups=1,
            n_routed_experts=2,
            n_shared_experts=0,
            num_experts_per_tok=1,
            max_position_embeddings=256,
            compress_ratios=[128],
            sliding_window=128,
            attention_dropout=0.0,
            num_hash_layers=0,
            hc_mult=1,
            num_nextn_predict_layers=0,
            rms_norm_eps=1e-6,
            torch_dtype="float32",
        )

    @staticmethod
    def _hca_inputs(
        cfg: DeepseekV4Config,
        seq_len: int,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        nn.Module,
        torch.Tensor,
    ]:
        hidden_states = torch.randn(1, seq_len, cfg.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        partial_rotary_factor = float(cfg.qk_rope_head_dim) / float(cfg.head_dim)
        rotary = DeepseekV4RotaryEmbedding(
            rope_theta=float(cfg.rope_theta),
            head_dim=int(cfg.head_dim),
            partial_rotary_factor=partial_rotary_factor,
        )
        rotary_compress = DeepseekV4RotaryEmbedding(
            rope_theta=float(cfg.compress_rope_theta),
            head_dim=int(cfg.head_dim),
            partial_rotary_factor=partial_rotary_factor,
        )
        position_embeddings = rotary(hidden_states, position_ids)
        position_embeddings_compress = rotary_compress(hidden_states, position_ids)
        attention_mask = build_causal_padding_mask(
            None,
            seq_len=seq_len,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            batch_size=hidden_states.shape[0],
            sliding_window=cfg.sliding_window,
        )
        return hidden_states, position_embeddings, position_embeddings_compress, rotary_compress, attention_mask


class TestDeepseekV4GroupedLinear:
    """Block-diagonal grouped linear backing the V4 attention output
    projection ``wo_a`` (n_groups=8 in DSV4-Flash).
    """

    def test_weight_shape(self):
        # n_heads=64, head_dim=512, n_groups=8 -> in_per_group=4096, out_total=8192
        proj = DeepseekV4GroupedLinear(in_features_per_group=4096, out_features=8192, n_groups=8)
        assert proj.weight.shape == (8192, 4096)
        assert proj.bias is None  # bias=False default

    def test_n_groups_attribute(self):
        proj = DeepseekV4GroupedLinear(64, 256, n_groups=4)
        assert proj.n_groups == 4

    def test_forward_shape(self):
        bsz, seq, n_groups, in_per = 2, 4, 8, 64
        out_total = 256
        proj = DeepseekV4GroupedLinear(in_per, out_total, n_groups=n_groups)
        x = torch.randn(bsz, seq, n_groups, in_per)
        out = proj(x)
        assert out.shape == (bsz, seq, n_groups, out_total // n_groups)

    def test_forward_matches_per_group_matmul(self):
        """Output should equal a per-group matmul: y[g] = x[g] @ W[g].T."""
        n_groups, in_per = 4, 8
        out_total = 16
        proj = DeepseekV4GroupedLinear(in_per, out_total, n_groups=n_groups)
        with torch.no_grad():
            proj.weight.normal_()
        x = torch.randn(3, n_groups, in_per)
        out = proj(x)
        w_ref = proj.weight.view(n_groups, out_total // n_groups, in_per)
        out_ref = torch.stack([x[:, g, :] @ w_ref[g].t() for g in range(n_groups)], dim=1)
        torch.testing.assert_close(out, out_ref)


class TestDeepseekV4HyperConnection:
    """``compute_weights`` returns the (pre, post, comb) tensors used at
    each HC site.  Shapes are deterministic given ``hc_mult`` and
    ``hidden_size``; values change with the learned parameters.
    """

    @pytest.fixture
    def hc(self):
        # ``DeepseekV4HyperConnection`` allocates ``fn``/``base``/``scale``
        # via ``torch.empty(...)``; those are uninitialized memory and may
        # contain NaN bit patterns.  Zero them so the Sinkhorn-row test has
        # a well-defined starting point (real model loads init from the
        # checkpoint via the state-dict adapter, not via ``empty``).
        m = DeepseekV4HyperConnection(
            hc_mult=4,
            hidden_size=16,
            hc_sinkhorn_iters=4,
            hc_eps=1e-6,
            rms_norm_eps=1e-6,
        )
        with torch.no_grad():
            m.fn.zero_()
            m.base.zero_()
            m.scale.zero_()
        return m

    def test_parameter_dtypes_are_fp32(self, hc):
        # HC params must stay fp32 even when the surrounding model is bf16.
        assert hc.fn.dtype == torch.float32
        assert hc.base.dtype == torch.float32
        assert hc.scale.dtype == torch.float32

    def test_compute_weights_output_shapes(self, hc):
        bsz, seq, hc_mult, hidden = 2, 5, 4, 16
        x = torch.randn(bsz, seq, hc_mult, hidden)
        pre, post, comb = hc.compute_weights(x)
        assert pre.shape == (bsz, seq, hc_mult)
        assert post.shape == (bsz, seq, hc_mult)
        assert comb.shape == (bsz, seq, hc_mult, hc_mult)

    def test_post_uses_2x_sigmoid(self, hc):
        """``post`` is ``2 * sigmoid(...)``, so it can exceed 1."""
        x = torch.randn(2, 3, 4, 16)
        with torch.no_grad():
            hc.scale.data[1].fill_(50.0)
            hc.base.data[hc.hc_mult : 2 * hc.hc_mult].fill_(50.0)
        _, post, _ = hc.compute_weights(x)
        assert post.max().item() > 1.5

    def test_comb_rows_sum_close_to_one(self, hc):
        """After softmax+sinkhorn, ``comb`` is doubly-(near-)stochastic."""
        x = torch.randn(1, 2, 4, 16)
        _, _, comb = hc.compute_weights(x)
        row_sums = comb.sum(dim=-1)
        col_sums = comb.sum(dim=-2)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=0, atol=1e-2)
        torch.testing.assert_close(col_sums, torch.ones_like(col_sums), rtol=0, atol=1e-2)


class TestApplyPartialRopeInterleaved:
    """Released DSV4-Flash uses INTERLEAVED RoPE pairs ``(2k, 2k+1)``."""

    def _make_cos_sin(self, batch, seq, rd):
        """Build the Llama-style ``cat([f, f], -1)`` cos/sin tensors that
        ``_apply_partial_rope_interleaved`` consumes (it uses only the
        first half).
        """
        half = rd // 2
        freqs = torch.arange(half, dtype=torch.float32) * 0.1
        pos = torch.arange(seq, dtype=torch.float32).unsqueeze(-1) * freqs
        cos_h = pos.cos()
        sin_h = pos.sin()
        cos = torch.cat([cos_h, cos_h], dim=-1).unsqueeze(0).expand(batch, -1, -1)
        sin = torch.cat([sin_h, sin_h], dim=-1).unsqueeze(0).expand(batch, -1, -1)
        return cos, sin

    def test_preserves_nope_prefix(self):
        bsz, heads, seq, rd, nope = 1, 2, 4, 8, 16
        x = torch.randn(bsz, heads, seq, nope + rd)
        cos, sin = self._make_cos_sin(bsz, seq, rd)
        y = _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim=rd)
        torch.testing.assert_close(y[..., :nope], x[..., :nope])

    def test_inverse_with_negated_sin_round_trips(self):
        """Rotating with ``sin`` then ``-sin`` recovers the input."""
        bsz, heads, seq, rd = 2, 4, 3, 8
        x = torch.randn(bsz, heads, seq, rd + 4)
        cos, sin = self._make_cos_sin(bsz, seq, rd)
        rotated = _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim=rd)
        unrotated = _apply_partial_rope_interleaved(rotated, cos, -sin, rope_head_dim=rd)
        torch.testing.assert_close(unrotated, x, rtol=1e-4, atol=1e-5)

    def test_zero_angles_is_identity(self):
        bsz, heads, seq, rd = 1, 1, 2, 4
        x = torch.randn(bsz, heads, seq, rd)
        cos = torch.ones(bsz, seq, rd)
        sin = torch.zeros(bsz, seq, rd)
        y = _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim=rd)
        torch.testing.assert_close(y, x)


class TestYaRNHelpers:
    """Sanity checks on the three pure-math helpers that build the YaRN
    correction ramp.  Reference math: ``dsv4flash/inference/model.py:
    precompute_freqs_cis`` (the inner ``find_correction_dim`` /
    ``find_correction_range`` / ``linear_ramp_factor`` helpers).
    """

    def test_correction_dim_monotonic_in_rotations(self):
        """``find_correction_dim`` is monotonically *decreasing* in
        ``num_rotations`` (more rotations ⇒ lower dim index, since the
        higher-frequency dims rotate more often within a fixed window).
        """
        # DSV4-Flash compress-rope: dim=64, base=160000, max_seq_len=65536
        d_low = _yarn_correction_dim(num_rotations=1, dim=64, base=160000, max_seq_len=65536)
        d_high = _yarn_correction_dim(num_rotations=32, dim=64, base=160000, max_seq_len=65536)
        assert d_high < d_low

    def test_correction_range_clamped(self):
        """``find_correction_range`` clamps to ``[0, dim-1]``."""
        low, high = _yarn_correction_range(low_rot=32, high_rot=1, dim=64, base=160000, max_seq_len=65536)
        assert 0 <= low <= high <= 63

    def test_linear_ramp_endpoints_and_clamp(self):
        """Ramp is ``0`` below ``min_v``, ``1`` above ``max_v``, linear in
        between.  ``dim=32`` matches DSV4 inv_freq length.
        """
        ramp = _yarn_linear_ramp(min_v=15.0, max_v=25.0, dim=32)
        # Below min: 0
        assert torch.all(ramp[:16] == 0.0)
        # Above max: 1
        assert torch.all(ramp[26:] == 1.0)
        # In between: in [0, 1]
        assert torch.all((ramp >= 0.0) & (ramp <= 1.0))

    def test_linear_ramp_handles_min_eq_max(self):
        """When ``min == max`` the helper bumps ``max`` by 1e-3 to avoid
        division by zero; the ramp should still be a valid step function."""
        ramp = _yarn_linear_ramp(min_v=5.0, max_v=5.0, dim=10)
        assert torch.all((ramp >= 0.0) & (ramp <= 1.0))


class TestDeepseekV4RotaryEmbeddingYaRN:
    """``DeepseekV4RotaryEmbedding`` with and without ``rope_scaling``.

    DSV4-Flash uses YaRN on the compress-rope path only:
        rope_theta=160000, factor=16, original_max_position_embeddings=65536,
        beta_fast=32, beta_slow=1.
    """

    def _yarn_kwargs(self, **overrides):
        kwargs = dict(
            rope_theta=160000.0,
            head_dim=128,
            partial_rotary_factor=0.5,  # qk_rope_head_dim=64
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            },
        )
        kwargs.update(overrides)
        return kwargs

    def test_no_rope_scaling_is_plain_rope(self):
        """``rope_scaling=None`` ⇒ plain ``1 / theta^(2i/d)`` inv_freq."""
        rope = DeepseekV4RotaryEmbedding(rope_theta=10000.0, head_dim=128, partial_rotary_factor=0.5, rope_scaling=None)
        dim = 64  # head_dim * partial_rotary_factor
        expected = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        torch.testing.assert_close(rope.inv_freq, expected)

    def test_yarn_attenuates_high_freq_dims_by_factor(self):
        """The highest-frequency dims (above the correction range) are
        divided by ``factor`` exactly; lowest-frequency dims (below the
        correction range) are unchanged.
        """
        plain = DeepseekV4RotaryEmbedding(
            rope_theta=160000.0, head_dim=128, partial_rotary_factor=0.5, rope_scaling=None
        )
        yarn = DeepseekV4RotaryEmbedding(**self._yarn_kwargs())

        ratio = yarn.inv_freq / plain.inv_freq

        # Low-frequency dims (small i, low rotation rate) are below the
        # correction-range floor — ramp=0, smooth=1, so inv_freq is unchanged.
        torch.testing.assert_close(ratio[:8], torch.ones(8), rtol=0, atol=1e-5)

        # High-frequency dims (large i, fast rotation) are above the ceiling —
        # ramp=1, smooth=0, so inv_freq /= factor (exactly 1/16).
        torch.testing.assert_close(ratio[-4:], torch.full((4,), 1.0 / 16.0), rtol=0, atol=1e-5)

        # Middle band is monotonically interpolating between 1.0 and 1/16.
        mid = ratio[8:-4]
        assert torch.all(mid <= ratio[7:-4][:-1] + 1e-6)
        assert torch.all(mid >= 1.0 / 16.0 - 1e-6)

    def test_yarn_factor_one_is_no_op(self):
        """``factor=1`` ⇒ ``inv_freq / 1 * (1-smooth) + inv_freq*smooth`` = ``inv_freq``."""
        plain = DeepseekV4RotaryEmbedding(
            rope_theta=160000.0, head_dim=128, partial_rotary_factor=0.5, rope_scaling=None
        )
        yarn = DeepseekV4RotaryEmbedding(
            **self._yarn_kwargs(
                rope_scaling={
                    "type": "yarn",
                    "factor": 1,
                    "original_max_position_embeddings": 65536,
                    "beta_fast": 32,
                    "beta_slow": 1,
                }
            )
        )
        torch.testing.assert_close(yarn.inv_freq, plain.inv_freq, rtol=0, atol=1e-7)

    def test_yarn_zero_original_max_pos_is_no_op(self):
        """``original_max_position_embeddings=0`` short-circuits YaRN
        (matches reference's ``if original_seq_len > 0`` gate)."""
        plain = DeepseekV4RotaryEmbedding(
            rope_theta=160000.0, head_dim=128, partial_rotary_factor=0.5, rope_scaling=None
        )
        yarn = DeepseekV4RotaryEmbedding(
            **self._yarn_kwargs(
                rope_scaling={
                    "type": "yarn",
                    "factor": 16,
                    "original_max_position_embeddings": 0,
                    "beta_fast": 32,
                    "beta_slow": 1,
                }
            )
        )
        torch.testing.assert_close(yarn.inv_freq, plain.inv_freq, rtol=0, atol=1e-7)

    def test_yarn_unrecognized_type_is_no_op(self):
        """A ``rope_scaling`` dict whose ``type`` is not ``"yarn"`` should
        be ignored (the gate is exact-string-match insensitive only to case).
        """
        plain = DeepseekV4RotaryEmbedding(
            rope_theta=160000.0, head_dim=128, partial_rotary_factor=0.5, rope_scaling=None
        )
        yarn = DeepseekV4RotaryEmbedding(
            rope_theta=160000.0,
            head_dim=128,
            partial_rotary_factor=0.5,
            rope_scaling={"type": "linear", "factor": 16},
        )
        torch.testing.assert_close(yarn.inv_freq, plain.inv_freq)

    def test_yarn_matches_reference_math_pointwise(self):
        """Recompute YaRN's ``inv_freq`` from the reference formula and
        check pointwise equality (catches any drift in the helper port).
        """
        rope_theta = 160000.0
        dim = 64
        factor = 16
        orig = 65536
        beta_fast = 32
        beta_slow = 1

        # Reference formula from dsv4flash/inference/model.py:
        #   freqs = 1.0 / (base ** (arange(0, dim, 2) / dim))
        #   low, high = find_correction_range(beta_fast, beta_slow, dim, base, orig)
        #   smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        #   freqs = freqs / factor * (1 - smooth) + freqs * smooth
        plain_freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        low, high = _yarn_correction_range(beta_fast, beta_slow, dim, rope_theta, orig)
        smooth = 1.0 - _yarn_linear_ramp(low, high, dim // 2)
        expected = plain_freqs / factor * (1.0 - smooth) + plain_freqs * smooth

        yarn = DeepseekV4RotaryEmbedding(**self._yarn_kwargs())
        torch.testing.assert_close(yarn.inv_freq, expected, rtol=0, atol=1e-7)

    def test_yarn_forward_returns_correct_shape_and_dtype(self):
        """Smoke check: the YaRN-modified rotary still produces ``(cos, sin)``
        sized to ``qk_rope_head_dim`` and downcasts to ``x.dtype`` if the
        forward returns BF16-casted tensors.
        """
        rope = DeepseekV4RotaryEmbedding(**self._yarn_kwargs())
        bsz, seq = 2, 16
        x = torch.zeros(bsz, seq, dtype=torch.bfloat16)
        position_ids = torch.arange(seq).unsqueeze(0).expand(bsz, -1)
        cos, sin = rope(x, position_ids)
        # rope_head_dim = head_dim * partial_rotary_factor = 64
        assert cos.shape == (bsz, seq, 64)
        assert sin.shape == (bsz, seq, 64)
        assert not torch.isnan(cos).any() and not torch.isnan(sin).any()
