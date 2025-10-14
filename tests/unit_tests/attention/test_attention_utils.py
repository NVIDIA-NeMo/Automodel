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
    process_input_for_thd,
    split_batch_into_thd_chunks,
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


class TestProcessInputForTHD:
    """Tests for process_input_for_thd function."""

    def test_basic_conversion(self):
        """Test basic conversion from BSHD to THD format with 2D token IDs."""
        batch_size, seq_len = 2, 6
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
            "labels": torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
            "seq_lens": torch.tensor([[6], [6]]),
            "seq_lens_padded": torch.tensor([[6], [6]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - for 2D input [batch, seq], output is [batch*seq] (squeezed)
        assert result["input_ids"].shape == (12,)
        assert result["position_ids"].shape == (12,)
        assert result["labels"].shape == (12,)

        # Check values are preserved
        assert torch.equal(result["input_ids"], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

        # Check cu_seqlens
        assert "cu_seqlens" in result
        assert "cu_seqlens_padded" in result
        assert torch.equal(result["cu_seqlens"], torch.tensor([0, 6, 12], dtype=torch.int32))

    def test_with_multiple_packed_sequences(self):
        """Test with multiple packed sequences per example."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 99, 4, 5], [6, 7, 99, 8, 9, 10]]),
            "labels": torch.tensor([[2, 3, 99, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 0, 1], [0, 1, 0, 0, 1, 2]]),
            "seq_lens": torch.tensor([[3, 2], [2, 3]]),
            "seq_lens_padded": torch.tensor([[4, 2], [3, 3]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - 2D input [batch, seq] becomes [batch*seq] (squeezed)
        assert result["input_ids"].shape == (12,)
        assert result["position_ids"].shape == (12,)

        # Check cu_seqlens: [0, 3, 5, 7, 10]
        # First batch: seq 1 (len 3), seq 2 (len 2) -> cumsum [0, 3, 5]
        # Second batch: seq 1 (len 2), seq 2 (len 3) -> cumsum [5, 7, 10]
        expected_cu_seqlens = torch.tensor([0, 3, 5, 7, 10], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"], expected_cu_seqlens)

        # Check cu_seqlens_padded: [0, 4, 6, 9, 12]
        expected_cu_seqlens_padded = torch.tensor([0, 4, 6, 9, 12], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens_padded"], expected_cu_seqlens_padded)

    def test_with_variable_num_sequences_and_padding(self):
        """Test with variable number of sequences per example (seq_lens padding with -1000)."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 99, 4, 5], [6, 7, 8, 9, 10, 11]]),
            "labels": torch.tensor([[2, 3, 99, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 0, 1], [0, 1, 2, 3, 4, 5]]),
            "seq_lens": torch.tensor([[3, 2], [6, -1000]]),  # -1000 is padding
            "seq_lens_padded": torch.tensor([[4, 2], [6, -1000]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - 2D input [batch, seq] becomes [batch*seq] (squeezed)
        assert result["input_ids"].shape == (12,)
        assert result["position_ids"].shape == (12,)

        # Check cu_seqlens: [0, 3, 5, 11] (filters out -1000)
        expected_cu_seqlens = torch.tensor([0, 3, 5, 11], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"], expected_cu_seqlens)

        # Check cu_seqlens_padded: [0, 4, 6, 12]
        expected_cu_seqlens_padded = torch.tensor([0, 4, 6, 12], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens_padded"], expected_cu_seqlens_padded)

    def test_with_qkv_format_preservation(self):
        """Test that non-tensor keys like qkv_format are preserved."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3]]),
            "qkv_format": "thd",  # Non-tensor key
        }

        result = process_input_for_thd(batch)

        # Check that qkv_format is preserved
        assert "qkv_format" in result
        assert result["qkv_format"] == "thd"

    def test_dtype_preservation(self):
        """Test that dtypes are preserved correctly."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
            "seq_lens": torch.tensor([[3], [3]], dtype=torch.long),
            "seq_lens_padded": torch.tensor([[3], [3]], dtype=torch.long),
        }

        result = process_input_for_thd(batch)

        assert result["input_ids"].dtype == torch.long
        assert result["position_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long
        assert result["cu_seqlens"].dtype == torch.int32
        assert result["cu_seqlens_padded"].dtype == torch.int32

    def test_with_embeddings_3d_input(self):
        """Test with 3D embeddings input (pipeline parallelism scenario)."""
        batch_size, seq_len, hidden_dim = 2, 6, 128
        batch = {
            "input_ids": torch.randn(batch_size, seq_len, hidden_dim),  # 3D embeddings
            "labels": torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
            "seq_lens": torch.tensor([[6], [6]]),
            "seq_lens_padded": torch.tensor([[6], [6]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - 3D input [batch, seq, hidden] becomes [batch*seq, hidden]
        assert result["input_ids"].shape == (12, hidden_dim)
        assert result["position_ids"].shape == (12,)
        assert result["cu_seqlens"].shape == (3,)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, batch_size):
        """Test with different batch sizes."""
        seq_len = 16
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = process_input_for_thd(batch)

        # Check output shapes
        expected_total_tokens = batch_size * seq_len
        assert result["input_ids"].shape == (expected_total_tokens,)
        assert result["position_ids"].shape == (expected_total_tokens,)
        assert result["labels"].shape == (expected_total_tokens,)
        assert result["cu_seqlens"].shape == (batch_size + 1,)

    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256, 512, 1024])
    def test_different_sequence_lengths(self, seq_len):
        """Test with different sequence lengths."""
        batch_size = 4
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = process_input_for_thd(batch)

        # Check output shapes
        expected_total_tokens = batch_size * seq_len
        assert result["input_ids"].shape == (expected_total_tokens,)
        assert result["cu_seqlens"][-1].item() == expected_total_tokens

    def test_single_batch_single_sequence(self):
        """Test edge case: single batch with single sequence."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[2, 3, 4, 5]]),
            "position_ids": torch.tensor([[0, 1, 2, 3]]),
            "seq_lens": torch.tensor([[4]]),
            "seq_lens_padded": torch.tensor([[4]]),
        }

        result = process_input_for_thd(batch)

        assert result["input_ids"].shape == (4,)
        assert torch.equal(result["cu_seqlens"], torch.tensor([0, 4], dtype=torch.int32))

    def test_large_batch_with_packing(self):
        """Test with large batch size and multiple packed sequences."""
        batch_size = 16
        num_packs = 3
        seq_len_per_pack = 128
        total_seq_len = num_packs * seq_len_per_pack

        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, total_seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, total_seq_len)),
            "position_ids": torch.arange(total_seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, num_packs), seq_len_per_pack),
            "seq_lens_padded": torch.full((batch_size, num_packs), seq_len_per_pack),
        }

        result = process_input_for_thd(batch)

        expected_total_tokens = batch_size * total_seq_len
        expected_num_sequences = batch_size * num_packs

        assert result["input_ids"].shape == (expected_total_tokens,)
        assert result["cu_seqlens"].shape == (expected_num_sequences + 1,)
        assert result["cu_seqlens"][-1].item() == expected_total_tokens

    def test_chunking_basic(self):
        """Test basic chunking functionality."""
        batch_size, seq_len = 4, 6
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]),
            "labels": torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
            "seq_lens": torch.tensor([[6], [6], [6], [6]]),
            "seq_lens_padded": torch.tensor([[6], [6], [6], [6]]),
        }

        # Process with 2 chunks (2 batch items per chunk)
        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # Check shapes - should be [num_chunks, tokens_per_chunk]
        assert result["input_ids"].shape == (2, 12), f"Expected shape (2, 12), got {result['input_ids'].shape}"
        assert result["labels"].shape == (2, 12)
        assert result["position_ids"].shape == (2, 12)

        # Check cu_seqlens has correct shape [num_chunks, seqs_per_chunk+1]
        assert result["cu_seqlens"].shape[0] == 2

        # Each chunk should have cumulative lengths [0, 6, 12]
        assert torch.equal(result["cu_seqlens"][0], torch.tensor([0, 6, 12], dtype=torch.int32))
        assert torch.equal(result["cu_seqlens"][1], torch.tensor([0, 6, 12], dtype=torch.int32))

    def test_chunking_with_packed_sequences(self):
        """Test chunking with multiple packed sequences per example."""
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 99, 4, 5],
                [6, 7, 99, 8, 9, 10],
                [11, 12, 13, 99, 14, 15],
                [16, 17, 99, 18, 19, 20]
            ]),
            "labels": torch.tensor([
                [2, 3, 99, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [12, 13, 99, 14, 15, 16],
                [17, 18, 19, 20, 21, 22]
            ]),
            "position_ids": torch.tensor([
                [0, 1, 2, 0, 0, 1],
                [0, 1, 0, 0, 1, 2],
                [0, 1, 2, 0, 0, 1],
                [0, 1, 0, 0, 1, 2]
            ]),
            "seq_lens": torch.tensor([[3, 2], [2, 3], [3, 2], [2, 3]]),
            "seq_lens_padded": torch.tensor([[4, 2], [3, 3], [4, 2], [3, 3]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # Check shapes
        assert result["input_ids"].shape == (2, 12)
        assert result["labels"].shape == (2, 12)

        # Check cu_seqlens for first chunk: [0, 3, 5, 7, 10]
        expected_cu_seqlens_0 = torch.tensor([0, 3, 5, 7, 10], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"][0], expected_cu_seqlens_0)

        # Check cu_seqlens for second chunk: [0, 3, 5, 7, 10]
        expected_cu_seqlens_1 = torch.tensor([0, 3, 5, 7, 10], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"][1], expected_cu_seqlens_1)

    def test_chunking_with_embeddings(self):
        """Test chunking with 3D embeddings input."""
        batch_size, seq_len, hidden_dim = 4, 6, 128
        batch = {
            "input_ids": torch.randn(batch_size, seq_len, hidden_dim),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # Check shapes - should be [num_chunks, tokens_per_chunk, hidden_dim]
        assert result["input_ids"].shape == (2, 12, hidden_dim)
        assert result["position_ids"].shape == (2, 12)
        assert result["cu_seqlens"].shape[0] == 2


class TestProcessInputForTHDWithChunks:
    """Comprehensive tests for process_input_for_thd_with_chunks function."""

    def test_variable_length_cu_seqlens_padding(self):
        """Test that cu_seqlens with different lengths are padded correctly."""
        # Create a batch where different chunks will have different cu_seqlens lengths
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 99, 4, 5],  # 2 sequences (lengths 3, 2)
                [6, 7, 8, 9, 10, 11],  # 1 sequence (length 6)
                [12, 13, 14, 15, 99, 99],  # 1 sequence (length 4)
                [16, 17, 99, 18, 19, 20]  # 2 sequences (lengths 2, 3)
            ]),
            "labels": torch.randint(0, 100, (4, 6)),
            "position_ids": torch.tensor([
                [0, 1, 2, 0, 0, 1],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 0, 0],
                [0, 1, 0, 0, 1, 2]
            ]),
            "seq_lens": torch.tensor([[3, 2, -1000], [6, -1000, -1000], [4, -1000, -1000], [2, 3, -1000]]),
            "seq_lens_padded": torch.tensor([[4, 2, -1000], [6, -1000, -1000], [4, -1000, -1000], [3, 3, -1000]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # First chunk has 2+1=3 sequences, second chunk has 1+2=3 sequences
        # cu_seqlens should be [num_chunks, max_seqs_across_chunks+1]
        assert result["cu_seqlens"].shape[0] == 2

        # First chunk: [0, 3, 5, 11] (2 seqs from first batch, 1 seq from second batch)
        expected_cu_seqlens_0 = torch.tensor([0, 3, 5, 11], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"][0], expected_cu_seqlens_0)

        # Second chunk: [0, 4, 6, 9] (1 seq from third batch, 2 seqs from fourth batch)
        expected_cu_seqlens_1 = torch.tensor([0, 4, 6, 9], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"][1], expected_cu_seqlens_1)

    def test_single_chunk(self):
        """Test with num_chunks=1 (no actual chunking)."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=1)

        # With 1 chunk, output should have batch dimension = 1
        assert result["input_ids"].shape == (1, 6)
        assert result["labels"].shape == (1, 6)
        assert result["position_ids"].shape == (1, 6)
        assert result["cu_seqlens"].shape == (1, 3)
        assert torch.equal(result["cu_seqlens"][0], torch.tensor([0, 3, 6], dtype=torch.int32))

    def test_many_chunks(self):
        """Test with many chunks (num_chunks=8)."""
        batch_size = 16
        seq_len = 32
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=8)

        # With 8 chunks, each chunk processes 2 batch items (16 / 8)
        tokens_per_chunk = 2 * seq_len
        assert result["input_ids"].shape == (8, tokens_per_chunk)
        assert result["labels"].shape == (8, tokens_per_chunk)
        assert result["cu_seqlens"].shape[0] == 8

        # Each chunk should have 3 cu_seqlens values: [0, 32, 64]
        for i in range(8):
            assert torch.equal(result["cu_seqlens"][i], torch.tensor([0, 32, 64], dtype=torch.int32))

    def test_non_tensor_key_preservation(self):
        """Test that non-tensor keys are preserved from the batch."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.full((4, 1), 3),
            "seq_lens_padded": torch.full((4, 1), 3),
            "qkv_format": "thd",
            "metadata": {"batch_id": 123},
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # Non-tensor keys should be preserved
        assert "qkv_format" in result
        assert result["qkv_format"] == "thd"
        assert "metadata" in result
        assert result["metadata"] == {"batch_id": 123}

    def test_custom_seq_lens_padding_value(self):
        """Test with custom seq_lens_padding_value."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3, -999], [3, -999], [3, -999], [3, -999]]),
            "seq_lens_padded": torch.tensor([[3, -999], [3, -999], [3, -999], [3, -999]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, seq_lens_padding_value=-999)

        # Check that -999 is used for padding in cu_seqlens
        # cu_seqlens for each chunk should be [0, 3, 6] initially
        # After padding they remain [0, 3, 6]
        assert result["cu_seqlens"].shape == (2, 3)

    def test_cu_seqlens_padded_handling(self):
        """Test that cu_seqlens_padded is properly handled when present."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 99, 4, 5], [6, 7, 8, 9, 10, 11]] * 2),
            "labels": torch.tensor([[2, 3, 99, 4, 5, 6], [7, 8, 9, 10, 11, 12]] * 2),
            "position_ids": torch.tensor([[0, 1, 2, 0, 0, 1], [0, 1, 2, 3, 4, 5]] * 2),
            "seq_lens": torch.tensor([[3, 2], [6, -1000]] * 2),
            "seq_lens_padded": torch.tensor([[4, 2], [6, -1000]] * 2),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # Both cu_seqlens and cu_seqlens_padded should be present
        assert "cu_seqlens" in result
        assert "cu_seqlens_padded" in result
        assert result["cu_seqlens_padded"] is not None

        # They should have the same first dimension (num_chunks)
        assert result["cu_seqlens"].shape[0] == 2
        assert result["cu_seqlens_padded"].shape[0] == 2

    def test_chunks_equivalence_to_no_chunks(self):
        """Test that chunking with 1 chunk is equivalent to process_input_for_thd."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3]]),
        }

        result_no_chunk = process_input_for_thd(batch)
        result_with_chunk = split_batch_into_thd_chunks(batch, num_chunks=1)

        # Results should match (with extra batch dimension in chunked version)
        assert torch.equal(result_with_chunk["input_ids"][0], result_no_chunk["input_ids"])
        assert torch.equal(result_with_chunk["labels"][0], result_no_chunk["labels"])
        assert torch.equal(result_with_chunk["position_ids"][0], result_no_chunk["position_ids"])
        assert torch.equal(result_with_chunk["cu_seqlens"][0], result_no_chunk["cu_seqlens"])

    def test_dtype_preservation_in_chunks(self):
        """Test that dtypes are preserved correctly through chunking."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=torch.long),
            "seq_lens": torch.tensor([[3], [3], [3], [3]], dtype=torch.long),
            "seq_lens_padded": torch.tensor([[3], [3], [3], [3]], dtype=torch.long),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        assert result["input_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long
        assert result["position_ids"].dtype == torch.long
        assert result["cu_seqlens"].dtype == torch.int32
        assert result["cu_seqlens_padded"].dtype == torch.int32

    def test_position_ids_none_handling(self):
        """Test handling when position_ids is None."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]),
            "position_ids": None,
            "seq_lens": torch.tensor([[3], [3], [3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3], [3], [3]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # position_ids should be None in result
        assert result["position_ids"] is None

    def test_padding_mask_correctness(self):
        """Test that padding_mask is correctly generated."""
        batch = {
            "input_ids": torch.tensor([[0, 1, 2], [3, 0, 5], [0, 0, 8], [9, 10, 11]]),
            "labels": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3], [3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3], [3], [3]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, padding_token_id=0)

        # Check padding_mask shape
        assert result["padding_mask"].shape == (2, 6)

        # Verify padding mask identifies 0s as padding
        # First chunk: [0, 1, 2, 3, 0, 5] -> mask: [T, F, F, F, T, F]
        expected_mask_0 = torch.tensor([True, False, False, False, True, False])
        assert torch.equal(result["padding_mask"][0], expected_mask_0)

    @pytest.mark.parametrize("num_chunks", [2, 4, 8])
    def test_different_chunk_sizes(self, num_chunks):
        """Test with different numbers of chunks."""
        batch_size = 16
        seq_len = 8
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=num_chunks)

        items_per_chunk = batch_size // num_chunks
        tokens_per_chunk = items_per_chunk * seq_len

        assert result["input_ids"].shape == (num_chunks, tokens_per_chunk)
        assert result["cu_seqlens"].shape[0] == num_chunks
