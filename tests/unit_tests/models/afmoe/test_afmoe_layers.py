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

from nemo_automodel.components.models.afmoe.config import AfmoeConfig
from nemo_automodel.components.models.afmoe.layers import AfmoeAttention
from nemo_automodel.components.models.common import BackendConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    return torch.device(f"cuda:{torch.cuda.current_device()}")


@pytest.fixture
def tiny_config():
    return AfmoeConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        num_dense_layers=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        global_attn_every_n_layers=2,
        sliding_window=64,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
    )


class TestAfmoeAttention:
    def test_local_attention_has_sliding_window(self, tiny_config, backend_config):
        attn = AfmoeAttention(tiny_config, layer_idx=0, backend=backend_config)
        assert attn.is_local_attention is True
        assert attn.sliding_window == tiny_config.sliding_window

    def test_global_attention_no_sliding_window(self, tiny_config, backend_config):
        attn = AfmoeAttention(tiny_config, layer_idx=1, backend=backend_config)
        assert attn.is_local_attention is False
        assert attn.sliding_window is None

    def test_has_gate_proj(self, tiny_config, backend_config):
        attn = AfmoeAttention(tiny_config, layer_idx=0, backend=backend_config)
        assert hasattr(attn, "gate_proj")

    def test_has_qk_norm(self, tiny_config, backend_config):
        attn = AfmoeAttention(tiny_config, layer_idx=0, backend=backend_config)
        assert hasattr(attn, "q_norm")
        assert hasattr(attn, "k_norm")

    def test_forward_shape(self, tiny_config, backend_config, device):
        attn = AfmoeAttention(tiny_config, layer_idx=0, backend=backend_config).to(device)

        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, tiny_config.hidden_size, device=device, dtype=torch.bfloat16)
        freqs_cis = torch.randn(batch, seq_len, tiny_config.head_dim, device=device)

        out = attn(x, freqs_cis=freqs_cis)
        assert out.shape == (batch, seq_len, tiny_config.hidden_size)

    def test_global_attention_forward_shape(self, tiny_config, backend_config, device):
        attn = AfmoeAttention(tiny_config, layer_idx=1, backend=backend_config).to(device)

        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, tiny_config.hidden_size, device=device, dtype=torch.bfloat16)
        freqs_cis = torch.randn(batch, seq_len, tiny_config.head_dim, device=device)

        out = attn(x, freqs_cis=freqs_cis)
        assert out.shape == (batch, seq_len, tiny_config.hidden_size)


class TestAfmoeAttentionParity:
    def test_rope_conditional_local_vs_global(self, tiny_config, backend_config, device):
        """Local attention (with RoPE) and global attention (without) must diverge given shared weights."""
        torch.manual_seed(42)
        local_attn = AfmoeAttention(tiny_config, layer_idx=0, backend=backend_config).to(device)
        global_attn = AfmoeAttention(tiny_config, layer_idx=1, backend=backend_config).to(device)
        global_attn.load_state_dict(local_attn.state_dict())

        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, tiny_config.hidden_size, device=device, dtype=torch.bfloat16)
        freqs_cis = torch.randn(batch, seq_len, tiny_config.head_dim, device=device)

        with torch.no_grad():
            local_out = local_attn(x, freqs_cis=freqs_cis)
            global_out = global_attn(x, freqs_cis=freqs_cis)

        max_diff = (local_out - global_out).abs().max().item()
        assert max_diff > 0.01, f"RoPE should cause divergence, but max_diff={max_diff}"

    def test_qk_norm_reduces_head_variance(self, tiny_config, backend_config, device):
        """Per-head QK RMSNorm should equalize magnitudes across heads."""
        attn = AfmoeAttention(tiny_config, layer_idx=0, backend=backend_config).to(device)

        batch, seq_len = 1, 4
        q = torch.randn(
            batch, seq_len, tiny_config.num_attention_heads, tiny_config.head_dim, device=device, dtype=torch.bfloat16
        )
        q[:, :, 0, :] *= 10.0  # Make first head 10x larger

        with torch.no_grad():
            q_normed = attn.q_norm(q)

        pre_var = q.norm(dim=-1).var(dim=-1).mean().item()
        post_var = q_normed.norm(dim=-1).var(dim=-1).mean().item()
        assert post_var < pre_var, "QK norm should reduce variance across heads"
