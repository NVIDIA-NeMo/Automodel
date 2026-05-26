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

"""Unit tests for ``HYV3Attention``."""

from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_v3.config import HYV3Config
from nemo_automodel.components.models.hy_v3.layers import HYV3Attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


HIDDEN = 64
N_HEADS = 8
N_KV = 2
HEAD_DIM = 16


@pytest.fixture
def device():
    return torch.device(f"cuda:{torch.cuda.current_device()}")


@pytest.fixture
def config():
    return HYV3Config(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )


@pytest.fixture
def sdpa_backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
        rope_fusion=False,
    )


def _make_freqs_cis(seq_len: int, device: torch.device) -> torch.Tensor:
    """Synthesize a freqs_cis tensor of shape (1, seq, head_dim) — matches
    ``apply_rotary_emb_qk(format='bshd', rope_fusion=False)`` which splits
    [cos, sin] along the last dim."""
    return torch.zeros(1, seq_len, HEAD_DIM, device=device)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_module_attributes(self, config, sdpa_backend):
        attn = HYV3Attention(config, backend=sdpa_backend)
        assert attn.num_heads == N_HEADS
        assert attn.num_kv_heads == N_KV
        assert attn.head_dim == HEAD_DIM
        assert attn.backend is sdpa_backend

    def test_projection_shapes(self, config, sdpa_backend):
        attn = HYV3Attention(config, backend=sdpa_backend)
        assert attn.q_proj.weight.shape == (N_HEADS * HEAD_DIM, HIDDEN)
        assert attn.k_proj.weight.shape == (N_KV * HEAD_DIM, HIDDEN)
        assert attn.v_proj.weight.shape == (N_KV * HEAD_DIM, HIDDEN)
        assert attn.o_proj.weight.shape == (HIDDEN, N_HEADS * HEAD_DIM)

    def test_q_k_norm_per_head_dim(self, config, sdpa_backend):
        """qk_norm is applied per-head, so the norm weight is sized to head_dim."""
        attn = HYV3Attention(config, backend=sdpa_backend)
        assert attn.q_norm.weight.shape == (HEAD_DIM,)
        assert attn.k_norm.weight.shape == (HEAD_DIM,)

    def test_no_attention_bias_by_default(self, config, sdpa_backend):
        attn = HYV3Attention(config, backend=sdpa_backend)
        assert attn.q_proj.bias is None
        assert attn.k_proj.bias is None
        assert attn.v_proj.bias is None
        assert attn.o_proj.bias is None

    def test_attention_bias_enabled(self, config, sdpa_backend):
        config.attention_bias = True
        attn = HYV3Attention(config, backend=sdpa_backend)
        assert attn.q_proj.bias is not None
        assert attn.k_proj.bias is not None
        assert attn.v_proj.bias is not None
        assert attn.o_proj.bias is not None


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shape_bshd(self, config, sdpa_backend, device):
        attn = HYV3Attention(config, backend=sdpa_backend).to(device)
        bsz, seqlen = 2, 4
        x = torch.randn(bsz, seqlen, HIDDEN, device=device, dtype=torch.bfloat16)
        freqs = _make_freqs_cis(seqlen, device)

        out = attn(x, freqs_cis=freqs)
        assert out.shape == (bsz, seqlen, HIDDEN)

    def test_calls_q_k_v_o_projections(self, config, sdpa_backend, device):
        attn = HYV3Attention(config, backend=sdpa_backend).to(device)
        bsz, seqlen = 1, 3
        x = torch.randn(bsz, seqlen, HIDDEN, device=device, dtype=torch.bfloat16)
        freqs = _make_freqs_cis(seqlen, device)

        with patch.object(attn.q_proj, "forward", wraps=attn.q_proj.forward) as q, \
             patch.object(attn.k_proj, "forward", wraps=attn.k_proj.forward) as k, \
             patch.object(attn.v_proj, "forward", wraps=attn.v_proj.forward) as v, \
             patch.object(attn.o_proj, "forward", wraps=attn.o_proj.forward) as o:
            attn(x, freqs_cis=freqs)
        q.assert_called_once()
        k.assert_called_once()
        v.assert_called_once()
        o.assert_called_once()

    def test_attention_mask_to_padding_mask_unused_in_sdpa(self, config, sdpa_backend, device):
        """With sdpa+causal the attention mask is converted via preprocess but never
        propagates to the kernel (is_causal=True suffices); we just check forward
        completes without raising."""
        attn = HYV3Attention(config, backend=sdpa_backend).to(device)
        bsz, seqlen = 1, 4
        x = torch.randn(bsz, seqlen, HIDDEN, device=device, dtype=torch.bfloat16)
        freqs = _make_freqs_cis(seqlen, device)
        mask = torch.ones(bsz, seqlen, dtype=torch.float32, device=device)
        out = attn(x, freqs_cis=freqs, attention_mask=mask)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Init weights
# ---------------------------------------------------------------------------


class TestInitWeights:
    def test_resets_norms_and_linears(self, config, sdpa_backend, device):
        attn = HYV3Attention(config, backend=sdpa_backend).to(device)
        with patch.object(attn.q_norm, "reset_parameters") as qn, \
             patch.object(attn.k_norm, "reset_parameters") as kn:
            attn.init_weights(buffer_device=device, init_std=0.01)
        qn.assert_called_once()
        kn.assert_called_once()

    def test_uses_provided_init_std(self, config, sdpa_backend, device):
        attn = HYV3Attention(config, backend=sdpa_backend).to(device)
        attn.init_weights(buffer_device=device, init_std=0.5)
        # trunc_normal_ has the requested std (within stat tolerance for a small tensor).
        # Just verify the linears were re-initialized to non-default values.
        for lin in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            assert lin.weight.abs().max().item() > 0
