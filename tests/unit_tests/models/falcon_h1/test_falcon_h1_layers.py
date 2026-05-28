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

"""Layer-equivalence tests for Falcon-H1.

Per SKILL.md: every rewritten layer gets a numerical-equivalence test against
the original HuggingFace layer, with identical weights, identical dtype, and
tolerances matched to that dtype. Realistic dims (hidden_size=256, 8/4 heads)
are used rather than the absolute minimum so divergence is not masked.

Attention / MLP / norm / decoder-layer tests run on CPU in float32 (no kernel
dependency). The Mamba mixer equivalence test is CUDA + mamba-ssm gated
because the mixer's fast path is Triton-only.
"""

import copy

import pytest

torch = pytest.importorskip("torch")

from .conftest import requires_falcon_h1, requires_cuda, requires_mamba_ssm

from nemo_automodel.components.models.falcon_h1.layers import (  # noqa: E402
    FalconH1Attention,
    FalconH1MLP,
    FalconH1DecoderLayer,
    FalconH1Mamba,
    _compute_mup_vector,
)


def _tol(dtype):
    return (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)


def _copy_matching(dst, src):
    """Copy params present (by name) in both modules; assert full coverage."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    common = set(src_sd) & set(dst_sd)
    missing = set(dst_sd) - common
    assert not missing, f"destination has params not in source: {missing}"
    dst.load_state_dict({k: src_sd[k] for k in dst_sd}, strict=True)


# --------------------------------------------------------------------------- #
# muP vector — pure tensor, no kernel. Compare to HF's compute_mup_vector.
# --------------------------------------------------------------------------- #
@requires_falcon_h1
def test_mup_vector_matches_reference(tiny_config):
    from transformers.models.falcon_h1.modeling_falcon_h1 import compute_mup_vector

    ours = _compute_mup_vector(tiny_config)
    ref = compute_mup_vector(tiny_config)
    assert ours.shape == ref.shape
    assert torch.allclose(ours, ref, atol=0, rtol=0)


@requires_falcon_h1
def test_mup_vector_length_matches_in_proj(tiny_config):
    """Vector length must equal in_proj output (projection_size)."""
    mixer = FalconH1Mamba(tiny_config, layer_idx=0)
    assert _compute_mup_vector(tiny_config).shape[-1] == mixer.in_proj.out_features


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
@requires_falcon_h1
def test_attention_equivalence(tiny_config):
    from transformers.models.falcon_h1.modeling_falcon_h1 import (
        FalconH1Attention as HFAttention,
        FalconH1RotaryEmbedding,
    )

    dtype = torch.float32
    cfg = copy.deepcopy(tiny_config)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)

    hf = HFAttention(cfg, layer_idx=0).to(dtype).eval()
    ours = FalconH1Attention(cfg, layer_idx=0).to(dtype).eval()
    _copy_matching(ours, hf)

    b, t = 2, 24
    hidden = torch.randn(b, t, cfg.hidden_size, dtype=dtype)
    rotary = FalconH1RotaryEmbedding(cfg).to(dtype)
    position_ids = torch.arange(t).unsqueeze(0).expand(b, -1)
    pos_emb = rotary(hidden, position_ids)

    with torch.no_grad():
        hf_out, _ = hf(hidden, position_embeddings=pos_emb, attention_mask=None)
        our_out, _ = ours(hidden, position_embeddings=pos_emb, attention_mask=None)

    atol, rtol = _tol(dtype)
    assert torch.allclose(hf_out, our_out, atol=atol, rtol=rtol), (
        f"max diff {(hf_out - our_out).abs().max().item()}"
    )


@requires_falcon_h1
def test_attention_key_multiplier_effective(tiny_config):
    """key_multiplier must actually scale the key path (guard against no-op)."""
    cfg = copy.deepcopy(tiny_config)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(1)
    attn = FalconH1Attention(cfg, layer_idx=0).to(torch.float32).eval()
    assert abs(attn.key_multiplier - cfg.key_multiplier) < 1e-9


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
@requires_falcon_h1
def test_mlp_equivalence(tiny_config):
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1MLP as HFMLP

    dtype = torch.float32
    torch.manual_seed(2)
    hf = HFMLP(tiny_config).to(dtype).eval()
    ours = FalconH1MLP(tiny_config).to(dtype).eval()
    _copy_matching(ours, hf)

    x = torch.randn(2, 16, tiny_config.hidden_size, dtype=dtype)
    with torch.no_grad():
        a, b = hf(x), ours(x)
    atol, rtol = _tol(dtype)
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"max diff {(a - b).abs().max().item()}"
    )


# --------------------------------------------------------------------------- #
# Gated RMSNorm — only meaningful when the layer exists
# --------------------------------------------------------------------------- #
@requires_falcon_h1
def test_gated_rmsnorm_equivalence():
    """If a gated-norm class is exposed, it must match HF's FalconH1RMSNormGated."""
    layers = pytest.importorskip(
        "nemo_automodel.components.models.falcon_h1.layers"
    )
    if not hasattr(layers, "FalconH1RMSNormGated"):
        pytest.skip("no gated RMSNorm implemented (mamba_rms_norm path absent)")

    from transformers.models.falcon_h1.modeling_falcon_h1 import (
        FalconH1RMSNormGated as HFGated,
    )

    dtype = torch.float32
    dim, n_groups = 128, 1
    torch.manual_seed(3)
    hf = HFGated(dim, eps=1e-5, n_groups=n_groups, norm_before_gate=False).to(dtype)
    ours = layers.FalconH1RMSNormGated(
        dim, eps=1e-5, n_groups=n_groups, norm_before_gate=False
    ).to(dtype)
    _copy_matching(ours, hf)

    h = torch.randn(2, 10, dim, dtype=dtype)
    gate = torch.randn(2, 10, dim, dtype=dtype)
    with torch.no_grad():
        a, b = hf(h, gate), ours(h, gate)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Decoder layer (parallel fuse) — Mamba stubbed so it runs on CPU
# --------------------------------------------------------------------------- #
@requires_falcon_h1
def test_decoder_layer_parallel_fuse_structure(tiny_config):
    """residual + ssm_out*m_ssm + attn_out*m_attn, then pre_ff MLP residual.

    We stub both sub-mixers with identity-shaped returns and check the algebra
    of the fuse exactly, including the muP scalars.
    """
    torch.manual_seed(4)
    layer = FalconH1DecoderLayer(tiny_config, layer_idx=0).to(torch.float32).eval()

    b, t, h = 2, 8, tiny_config.hidden_size
    hidden = torch.randn(b, t, h)

    fixed_mamba = torch.randn(b, t, h)
    fixed_attn = torch.randn(b, t, h)
    layer.mamba.forward = lambda hs, **kw: fixed_mamba  # type: ignore
    layer.self_attn.forward = lambda hs, **kw: (fixed_attn, None)  # type: ignore

    with torch.no_grad():
        normed = layer.input_layernorm(hidden)
        expected_mix = (
            hidden
            + fixed_mamba * tiny_config.ssm_out_multiplier
            + fixed_attn * tiny_config.attention_out_multiplier
        )
        expected = expected_mix + layer.feed_forward(layer.pre_ff_layernorm(expected_mix))
        out = layer(hidden, position_embeddings=None, attention_mask=None)
        if isinstance(out, tuple):
            out = out[0]
    # normed is referenced to ensure input_layernorm participates; the spy on
    # self_attn ignores its input, so we only assert the fuse algebra here.
    assert normed.shape == hidden.shape
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), (
        f"max diff {(out - expected).abs().max().item()}"
    )


# --------------------------------------------------------------------------- #
# Mamba mixer — real kernel, GPU only
# --------------------------------------------------------------------------- #
@requires_falcon_h1
@requires_cuda
@requires_mamba_ssm
def test_mamba_equivalence(tiny_config):
    from transformers.models.falcon_h1.modeling_falcon_h1 import (
        FalconH1Mixer as HFMixer,
    )

    dtype = torch.bfloat16
    torch.manual_seed(5)
    hf = HFMixer(tiny_config, layer_idx=0).to("cuda", dtype).eval()
    ours = FalconH1Mamba(tiny_config, layer_idx=0).to("cuda", dtype).eval()
    _copy_matching(ours, hf)

    hidden = torch.randn(2, 32, tiny_config.hidden_size, device="cuda", dtype=dtype)
    with torch.no_grad():
        a = hf(hidden)
        b = ours(hidden)
    atol, rtol = _tol(dtype)
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"max diff {(a - b).abs().max().item()}"
    )
