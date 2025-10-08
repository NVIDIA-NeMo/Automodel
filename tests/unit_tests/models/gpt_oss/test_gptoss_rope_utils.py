import math
import pytest
import torch

from nemo_automodel.components.models.gpt_oss.rope_utils import (
    apply_rotary_emb,
    RotaryEmbedding,
)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


class TestApplyRotaryEmbedding:
    """Test _apply_rotary_emb function."""

    def test_apply_rotary_emb_shape_preservation(self, device):
        """Test that rotary embedding preserves tensor shapes."""
        batch_size, seq_len, n_heads, head_dim = 2, 4, 2, 8
        x = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
        cos = torch.randn(seq_len, head_dim // 2, device=device)
        sin = torch.randn(seq_len, head_dim // 2, device=device)

        result = apply_rotary_emb(x, cos, sin)

        assert result.shape == x.shape
        assert result.device == x.device
        assert result.dtype == x.dtype

    def test_apply_rotary_emb_correctness(self, device):
        """Test rotary embedding computation correctness."""
        # Simple test case with known values
        x = torch.ones(1, 2, 1, 4, device=device)
        cos = torch.ones(2, 2, device=device)
        sin = torch.zeros(2, 2, device=device)

        result = apply_rotary_emb(x, cos, sin)

        # With sin=0 and cos=1, result should be [x1, x2] -> [x1*1-x2*0, x2*1+x1*0] = [x1, x2]
        expected = x.clone()
        torch.testing.assert_close(result, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_apply_rotary_emb_gpu_dtype_handling(self):
        """Test GPU-specific dtype handling."""
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        x = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16, device=device)
        cos = torch.randn(2, 2, dtype=torch.float32, device=device)
        sin = torch.randn(2, 2, dtype=torch.float32, device=device)

        result = apply_rotary_emb(x, cos, sin)

        assert result.dtype == torch.bfloat16
        assert result.device == device


class TestRotaryEmbedding:
    """Test RotaryEmbedding module."""

    def test_rotary_embedding_init(self, device):
        """Test RotaryEmbedding initialization."""
        head_dim = 32
        base = 10000
        rope = RotaryEmbedding(
            head_dim=head_dim,
            base=base,
            dtype=torch.float32,
            device=device,
        )

        assert rope.head_dim == head_dim
        assert rope.base == base
        assert rope.dtype == torch.float32
        assert rope.device == device
        assert rope.scaling_factor == 1.0
        assert rope.ntk_alpha == 1.0
        assert rope.ntk_beta == 32.0

    def test_compute_concentration_and_inv_freq_no_scaling(self, device):
        """Test concentration and inverse frequency computation without scaling."""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
            scaling_factor=1.0,
            device=device,
        )

        concentration, inv_freq = rope._compute_concentration_and_inv_freq()

        assert concentration == 1.0
        assert inv_freq.shape == (16,)  # head_dim // 2
        assert inv_freq.device == device

    def test_compute_concentration_and_inv_freq_with_scaling(self, device):
        """Test concentration and inverse frequency computation with scaling."""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
            scaling_factor=2.0,
            initial_context_length=512,
            device=device,
        )

        concentration, inv_freq = rope._compute_concentration_and_inv_freq()

        expected_concentration = 0.1 * math.log(2.0) + 1.0
        assert abs(concentration - expected_concentration) < 1e-6
        assert inv_freq.shape == (16,)

    def test_compute_cos_sin(self, device):
        """Test cosine and sine computation."""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
            device=device,
        )

        num_tokens = 8
        cos, sin = rope._compute_cos_sin(num_tokens)

        assert cos.shape == (num_tokens, 16)
        assert sin.shape == (num_tokens, 16)
        assert cos.device == device
        assert sin.device == device

    def test_forward_shape_preservation(self, device):
        """Test that forward pass preserves query and key shapes."""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
            device=device,
        )

        batch_size, seq_len, n_heads, head_dim = 2, 4, 4, 32
        query = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
        key = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)

        q_rot, k_rot = rope(query, key)

        assert q_rot.shape == query.shape
        assert k_rot.shape == key.shape
        assert q_rot.device == query.device
        assert k_rot.device == key.device

    def test_forward_different_seq_lengths(self, device):
        """Test forward pass with different sequence lengths."""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
            device=device,
        )

        for seq_len in [1, 8, 16, 32]:
            query = torch.randn(1, seq_len, 4, 32, device=device)
            key = torch.randn(1, seq_len, 4, 32, device=device)

            q_rot, k_rot = rope(query, key)

            assert q_rot.shape == (1, seq_len, 4, 32)
            assert k_rot.shape == (1, seq_len, 4, 32)
