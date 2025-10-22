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

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    preprocess_args_and_kwargs_for_attn,
    postprocess_output_for_attn,
)


class TestInitializeAttnModuleAndFunc:
    """Tests for initialize_attn_module_and_func function."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_attention_initialization(self):
        """Test Transformer Engine attention initialization."""
        pytest.importorskip("transformer_engine")

        num_attention_heads = 8
        num_qk_channels = 64
        num_v_channels = 64
        softmax_scale = 0.125
        num_gqa_groups = 4

        attn_module, attn_func = initialize_attn_module_and_func(
            attn_impl="te",
            num_attention_heads=num_attention_heads,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            softmax_scale=softmax_scale,
            attn_mask_type="causal",
            qkv_format="bshd",
            num_gqa_groups=num_gqa_groups,
        )

        assert attn_module is not None
        assert callable(attn_func)
        assert isinstance(attn_module, nn.Module)

    def test_sdpa_attention_initialization(self):
        """Test SDPA attention initialization."""
        num_attention_heads = 8
        num_qk_channels = 64
        num_v_channels = 64
        softmax_scale = 0.125
        num_gqa_groups = 4

        attn_module, attn_func = initialize_attn_module_and_func(
            attn_impl="sdpa",
            num_attention_heads=num_attention_heads,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            softmax_scale=softmax_scale,
            attn_mask_type="causal",
            num_gqa_groups=num_gqa_groups,
        )

        assert attn_module is None
        assert callable(attn_func)

    def test_flex_attention_initialization(self):
        """Test Flex attention initialization."""
        num_attention_heads = 8
        num_qk_channels = 64
        num_v_channels = 64
        softmax_scale = 0.125

        attn_module, attn_func = initialize_attn_module_and_func(
            attn_impl="flex",
            num_attention_heads=num_attention_heads,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            softmax_scale=softmax_scale,
            attn_mask_type="causal",
            qkv_format="bshd",
        )

        assert attn_module is not None
        assert callable(attn_func)
        assert isinstance(attn_module, nn.Module)

    def test_unsupported_attention_implementation(self):
        """Test that unsupported attention implementation raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported attention implementation"):
            initialize_attn_module_and_func(
                attn_impl="unsupported",
                num_attention_heads=8,
                num_qk_channels=64,
                num_v_channels=64,
                softmax_scale=0.125,
            )


class TestPreprocessArgsAndKwargsForAttn:
    """Tests for preprocess_args_and_kwargs_for_attn function."""

    def setup_method(self):
        """Setup common test tensors."""
        self.batch_size = 2
        self.seq_len = 128
        self.num_heads = 8
        self.head_dim = 64

        self.q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self.k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self.v = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_with_attention_mask(self):
        """Test TE preprocessing with attention mask."""
        pytest.importorskip("transformer_engine")

        attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        attention_mask[:, self.seq_len // 2 :] = False

        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            self.q, self.k, self.v, attention_mask, attn_impl="te"
        )

        assert q_out.shape == self.q.shape
        assert k_out.shape == self.k.shape
        assert v_out.shape == self.v.shape
        assert "attn_mask_type" in attn_kwargs
        assert attn_kwargs["attn_mask_type"] == "padding_causal"
        assert "attention_mask" in attn_kwargs
        assert "window_size" in attn_kwargs

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_with_seq_lens(self):
        """Test TE preprocessing with seq_lens for packed sequences."""
        pytest.importorskip("transformer_engine")

        device = torch.device("cuda")
        seq_lens = torch.tensor([50, 78], device=device)

        q_gpu = self.q.to(device)
        k_gpu = self.k.to(device)
        v_gpu = self.v.to(device)

        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q_gpu, k_gpu, v_gpu, attention_mask=None, attn_impl="te", seq_lens=seq_lens
        )

        assert q_out.shape == q_gpu.shape
        assert "qkv_format" in attn_kwargs
        assert attn_kwargs["qkv_format"] == "thd"
        assert "cu_seqlens_q" in attn_kwargs
        assert "cu_seqlens_kv" in attn_kwargs
        assert attn_kwargs["cu_seqlens_q"].shape == (3,)  # [0, 50, 128]
        assert attn_kwargs["cu_seqlens_q"][0] == 0
        assert attn_kwargs["cu_seqlens_q"][1] == 50
        assert attn_kwargs["cu_seqlens_q"][2] == 128

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_with_seq_lens_and_seq_lens_padded(self):
        """Test TE preprocessing with both seq_lens and seq_lens_padded."""
        pytest.importorskip("transformer_engine")

        device = torch.device("cuda")
        seq_lens = torch.tensor([50, 78], device=device)
        seq_lens_padded = torch.tensor([52, 80], device=device)

        q_gpu = self.q.to(device)
        k_gpu = self.k.to(device)
        v_gpu = self.v.to(device)

        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q_gpu,
            k_gpu,
            v_gpu,
            attention_mask=None,
            attn_impl="te",
            seq_lens=seq_lens,
            seq_lens_padded=seq_lens_padded,
        )

        assert "cu_seqlens_q_padded" in attn_kwargs
        assert "cu_seqlens_kv_padded" in attn_kwargs
        assert attn_kwargs["cu_seqlens_q_padded"].shape == (3,)
        assert attn_kwargs["cu_seqlens_q_padded"][0] == 0
        assert attn_kwargs["cu_seqlens_q_padded"][1] == 52
        assert attn_kwargs["cu_seqlens_q_padded"][2] == 132

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_with_cu_seqlens(self):
        """Test TE preprocessing with cu_seqlens."""
        pytest.importorskip("transformer_engine")

        device = torch.device("cuda")
        cu_seqlens = torch.tensor([0, 50, 128], dtype=torch.int32, device=device)

        q_gpu = self.q.to(device)
        k_gpu = self.k.to(device)
        v_gpu = self.v.to(device)

        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q_gpu, k_gpu, v_gpu, attention_mask=None, attn_impl="te", cu_seqlens=cu_seqlens
        )

        assert "cu_seqlens_q" in attn_kwargs
        assert "cu_seqlens_kv" in attn_kwargs
        assert torch.equal(attn_kwargs["cu_seqlens_q"], cu_seqlens)
        assert torch.equal(attn_kwargs["cu_seqlens_kv"], cu_seqlens)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_with_cu_seqlens_q_and_kv(self):
        """Test TE preprocessing with separate cu_seqlens_q and cu_seqlens_kv."""
        pytest.importorskip("transformer_engine")

        device = torch.device("cuda")
        cu_seqlens_q = torch.tensor([0, 50, 128], dtype=torch.int32, device=device)
        cu_seqlens_kv = torch.tensor([0, 60, 128], dtype=torch.int32, device=device)

        q_gpu = self.q.to(device)
        k_gpu = self.k.to(device)
        v_gpu = self.v.to(device)

        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q_gpu, k_gpu, v_gpu, attention_mask=None, attn_impl="te", cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv
        )

        assert "cu_seqlens_q" in attn_kwargs
        assert "cu_seqlens_kv" in attn_kwargs
        assert torch.equal(attn_kwargs["cu_seqlens_q"], cu_seqlens_q)
        assert torch.equal(attn_kwargs["cu_seqlens_kv"], cu_seqlens_kv)

    def test_sdpa_preprocessing(self):
        """Test SDPA preprocessing (transposes tensors)."""
        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            self.q, self.k, self.v, attention_mask=None, attn_impl="sdpa"
        )

        # SDPA expects [B, H, S, D] -> [B, S, H, D]
        expected_shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        assert q_out.shape == expected_shape
        assert k_out.shape == expected_shape
        assert v_out.shape == expected_shape
        assert attn_kwargs["is_causal"] is True

    def test_sdpa_preprocessing_with_mask(self):
        """Test SDPA preprocessing with attention mask."""
        attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)

        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            self.q, self.k, self.v, attention_mask=attention_mask, attn_impl="sdpa"
        )

        # SDPA should still transpose even with mask
        expected_shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        assert q_out.shape == expected_shape


class TestPostprocessOutputForAttn:
    """Tests for postprocess_output_for_attn function."""

    def setup_method(self):
        """Setup common test tensors."""
        self.batch_size = 2
        self.seq_len = 128
        self.num_heads = 8
        self.head_dim = 64

    def test_sdpa_postprocessing(self):
        """Test SDPA postprocessing (transposes back)."""
        # SDPA output is [B, S, H, D]
        x = torch.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim)

        x_out = postprocess_output_for_attn(x, attn_impl="sdpa")

        # Should transpose back to [B, H, S, D]
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        assert x_out.shape == expected_shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_postprocessing_no_change(self):
        """Test TE postprocessing (no change)."""
        pytest.importorskip("transformer_engine")

        x = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)

        x_out = postprocess_output_for_attn(x, attn_impl="te")

        # TE doesn't transpose, so output should be identical
        assert x_out.shape == x.shape
        assert torch.equal(x, x_out)

    def test_flex_postprocessing_no_change(self):
        """Test Flex postprocessing (no change)."""
        x = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)

        x_out = postprocess_output_for_attn(x, attn_impl="flex")

        # Flex doesn't transpose, so output should be identical
        assert x_out.shape == x.shape
        assert torch.equal(x, x_out)


class TestEndToEndWorkflow:
    """Integration tests for complete preprocessing -> postprocessing workflow."""

    def setup_method(self):
        """Setup common test tensors."""
        self.batch_size = 2
        self.seq_len = 128
        self.num_heads = 8
        self.head_dim = 64

        self.q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self.k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self.v = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)

    def test_sdpa_round_trip(self):
        """Test that SDPA preprocessing and postprocessing are inverses."""
        original_shape = self.q.shape

        # Preprocess
        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            self.q, self.k, self.v, attention_mask=None, attn_impl="sdpa"
        )

        # Simulate attention output (same shape as q after preprocessing)
        attn_output = torch.randn_like(q_out)

        # Postprocess
        final_output = postprocess_output_for_attn(attn_output, attn_impl="sdpa")

        # Should be back to original shape
        assert final_output.shape == original_shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_round_trip_with_seq_lens(self):
        """Test TE preprocessing and postprocessing with seq_lens."""
        pytest.importorskip("transformer_engine")

        device = torch.device("cuda")
        seq_lens = torch.tensor([50, 78], device=device)

        q_gpu = self.q.to(device)
        k_gpu = self.k.to(device)
        v_gpu = self.v.to(device)

        original_shape = q_gpu.shape

        # Preprocess
        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q_gpu, k_gpu, v_gpu, attention_mask=None, attn_impl="te", seq_lens=seq_lens
        )

        # Simulate attention output (same shape as q after preprocessing)
        attn_output = torch.randn_like(q_out)

        # Postprocess
        final_output = postprocess_output_for_attn(attn_output, attn_impl="te")

        # Should remain the same shape
        assert final_output.shape == original_shape
