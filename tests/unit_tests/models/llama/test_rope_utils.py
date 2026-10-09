# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for ``LlamaRotaryEmbedding`` position handling.

``forward`` must return cos/sin for the *values* in ``position_ids``, not merely
for ``arange(seq_len)``. A regression here makes any non-contiguous position --
EAGLE TTT depth offsets (``arange + step_idx``), packed sequences, context
parallelism -- silently receive the wrong rotary phase.
"""

import logging
from unittest.mock import patch

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.llama.rope_utils import LlamaRotaryEmbedding, apply_rotary_pos_emb


def _build_rope(
    *,
    head_dim: int = 8,
    heads: int = 4,
    max_pos: int = 128,
    rope_fusion: bool = False,
) -> LlamaRotaryEmbedding:
    config = LlamaConfig(
        hidden_size=head_dim * heads,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=max_pos,
    )
    return LlamaRotaryEmbedding(config, rope_fusion=rope_fusion)


def test_rope_arange_is_per_position_and_unchanged():
    """``arange`` position_ids reproduce cos/sin evaluated per absolute position.

    This pins the legacy contiguous behavior as a special case of the
    position-value gather (the common training/inference path must not change).
    """
    rope = _build_rope()
    x = torch.zeros(1, 1, 8)
    n = 12
    cos, sin = rope(x, torch.arange(n).unsqueeze(0))
    assert cos.shape == (1, n, 8)
    cos_each = torch.stack([rope(x, torch.tensor([[i]]))[0][0, 0] for i in range(n)])
    sin_each = torch.stack([rope(x, torch.tensor([[i]]))[1][0, 0] for i in range(n)])
    torch.testing.assert_close(cos[0], cos_each)
    torch.testing.assert_close(sin[0], sin_each)


def test_rope_honors_position_offset():
    """``position_ids = arange(n) + k`` must shift the phase by ``k``.

    Regression for a bug where ``forward`` keyed only on ``seq_len`` and ignored
    the position values, turning EAGLE's ``position_ids + step_idx`` into a
    no-op (drafts trained without the intended per-depth rotary offset).
    """
    rope = _build_rope()
    x = torch.zeros(1, 1, 8)
    n, k = 6, 3
    base = torch.arange(n).unsqueeze(0)
    cos0, _ = rope(x, base)
    cosk, sink = rope(x, base + k)
    # The offset must actually change the embedding...
    assert (cos0 - cosk).abs().max().item() > 1e-3
    # ...and must equal cos/sin at the absolute shifted positions.
    cos_ref = torch.stack([rope(x, torch.tensor([[i + k]]))[0][0, 0] for i in range(n)])
    sin_ref = torch.stack([rope(x, torch.tensor([[i + k]]))[1][0, 0] for i in range(n)])
    torch.testing.assert_close(cosk[0], cos_ref)
    torch.testing.assert_close(sink[0], sin_ref)


def test_rope_gathers_non_contiguous_positions():
    """Arbitrary (packed / context-parallel) position_ids gather per-position."""
    rope = _build_rope()
    x = torch.zeros(1, 1, 8)
    positions = [0, 5, 2, 9]
    cos, sin = rope(x, torch.tensor([positions]))
    for i, p in enumerate(positions):
        cos_p, sin_p = rope(x, torch.tensor([[p]]))
        torch.testing.assert_close(cos[0, i], cos_p[0, 0])
        torch.testing.assert_close(sin[0, i], sin_p[0, 0])


def test_quack_backend_disables_fusion_and_gathers_non_contiguous_positions(caplog):
    """QuACK must not inherit the CUDA/TE fused-RoPE default.

    Passing ``rope_fusion=True`` reproduces the default on CUDA builds with
    Transformer Engine. QuACK must override it so arbitrary position IDs select
    their absolute rotary phases instead of a contiguous ``[0, seq_len)`` slice.
    """
    with caplog.at_level(logging.WARNING):
        backend = BackendConfig(rope="quack", rope_fusion=True)

    assert backend.rope_fusion is False
    assert "rope='quack' is incompatible with rope_fusion=True" in caplog.text

    rope = _build_rope(rope_fusion=backend.rope_fusion)
    x = torch.zeros(1, 1, 8)
    positions = torch.tensor([[0, 5, 2, 9]])
    cos, sin = rope(x, positions)
    contiguous_cos, _ = rope(x, torch.arange(positions.shape[-1]).unsqueeze(0))

    assert not torch.equal(cos, contiguous_cos)
    for sequence_index, position in enumerate(positions[0]):
        cos_at_position, sin_at_position = rope(x, position.reshape(1, 1))
        torch.testing.assert_close(cos[0, sequence_index], cos_at_position[0, 0])
        torch.testing.assert_close(sin[0, sequence_index], sin_at_position[0, 0])


def test_rope_position_exceeding_seq_len_grows_cache():
    """A single position past ``seq_len`` must not index out of the cache."""
    rope = _build_rope(max_pos=128)
    x = torch.zeros(1, 1, 8)
    # seq_len 2 but positions up to 40 (e.g. a deep EAGLE TTT offset).
    cos, sin = rope(x, torch.tensor([[39, 40]]))
    assert cos.shape == (1, 2, 8)
    cos_ref = rope(x, torch.tensor([[40]]))[0][0, 0]
    torch.testing.assert_close(cos[0, 1], cos_ref)


def test_rope_fused_path_uses_contiguous_slice_and_returns_freqs():
    """The fused TE path returns ``(cos, sin, freqs)`` from the contiguous slice.

    The fused kernel indexes raw angles by sequence position and assumes
    contiguous ``[0, seq_len)`` positions. It therefore keeps the legacy slice
    and -- by design -- does NOT honor a non-contiguous ``position_ids`` offset
    (packed sequences / context parallelism are not corrected on this path).
    """
    rope = _build_rope()
    rope.rope_fusion = True
    rope._cos_cache = rope._sin_cache = rope._freqs_cache = None
    rope.max_seq_len_cached = 0
    x = torch.zeros(1, 1, 8)
    n, k = 6, 3
    base = torch.arange(n).unsqueeze(0)

    out0 = rope(x, base)
    outk = rope(x, base + k)
    assert len(out0) == 3 and len(outk) == 3
    cos0, sin0, freqs0 = out0
    cosk, _, freqsk = outk
    assert cos0.shape == (1, n, 8)
    assert freqs0.shape == (n, 1, 1, 8)
    # The offset is intentionally ignored on the fused path: same slice [:n].
    torch.testing.assert_close(cos0, cosk)
    torch.testing.assert_close(freqs0, freqsk)


def test_rope_fused_path_does_not_sync_on_position_values():
    """The fused path must size the cache by ``seq_len``, never by ``position_ids.max()``.

    Calling ``.max()/.item()`` on ``position_ids`` forces a host-device sync (and
    a ``torch.compile`` graph break) on every step of the default GPU+TE training
    path. The fused branch only needs ``seq_len``, so it must not touch the
    position values' ``.max()``.
    """
    rope = _build_rope()
    rope.rope_fusion = True
    rope._cos_cache = rope._sin_cache = rope._freqs_cache = None
    rope.max_seq_len_cached = 0
    x = torch.zeros(1, 1, 8)

    def _no_max(*args, **kwargs):
        raise AssertionError("fused path must not call position_ids.max()")

    with patch.object(torch.Tensor, "max", _no_max):
        cos, sin, freqs = rope(x, torch.arange(6).unsqueeze(0))
    assert cos.shape == (1, 6, 8)


def test_bf16_model_cast_does_not_degrade_inv_freq():
    """A model-wide ``.to(bfloat16)`` must not degrade RoPE precision.

    ``LlamaForCausalLM.__init__`` casts the whole model via
    ``self.to(config.torch_dtype)``, and ``nn.Module.to`` rounds floating-point
    buffers -- so the ``inv_freq`` buffer is downcast to bf16. Building the cos/sin
    tables from that bf16-rounded buffer (then upcasting) loses precision relative to
    HF, which keeps ``inv_freq`` in float32; the gap shows up as a large logit/KL
    divergence when a checkpoint is reloaded in vanilla HF. The tables must therefore
    be identical whether or not the module was cast to bf16.

    Uses real Llama-3.2 rope params (``rope_theta=5e5`` + ``llama3`` scaling): the
    low-frequency components are where the bf16 rounding error is largest.
    """
    config = LlamaConfig(
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        },
        torch_dtype=torch.bfloat16,
    )
    x = torch.zeros(1, 9, config.hidden_size, dtype=torch.bfloat16)
    pos = torch.arange(9).unsqueeze(0)

    rope_ref = LlamaRotaryEmbedding(config)  # inv_freq stays float32
    rope_cast = LlamaRotaryEmbedding(config).to(torch.bfloat16)  # mimics the model-wide cast
    assert rope_cast.inv_freq.dtype == torch.bfloat16  # buffer is rounded by .to()

    cos_ref, sin_ref = rope_ref(x, pos)
    cos_cast, sin_cast = rope_cast(x, pos)
    # Bit-for-bit: the cast module must still build its tables from float32 inv_freq.
    torch.testing.assert_close(cos_cast, cos_ref, rtol=0, atol=0)
    torch.testing.assert_close(sin_cast, sin_ref, rtol=0, atol=0)


def _config_with_rope_scaling(rope_scaling: dict) -> LlamaConfig:
    """Build a config carrying ``rope_scaling``, mirroring how the EAGLE recipe
    seeds the draft config from ``target_config.to_dict()``."""
    base = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=2048,
    )
    config_dict = base.to_dict()
    config_dict["rope_scaling"] = rope_scaling
    return LlamaConfig.from_dict(config_dict)


def test_rope_yarn_matches_transformers_not_llama3_fallback():
    """A ``yarn`` rope_type must use transformers' YaRN schedule, not silently
    fall back to llama3 (the latent bug an EAGLE dense draft inherited from a
    YaRN target's ``rope_scaling``)."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from nemo_automodel.components.models.llama.rope_utils import _compute_llama3_inv_freq

    config = _config_with_rope_scaling({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048})
    rope = LlamaRotaryEmbedding(config)

    ref_inv_freq, ref_scaling = ROPE_INIT_FUNCTIONS["yarn"](config, torch.device("cpu"))
    torch.testing.assert_close(rope.inv_freq, ref_inv_freq)
    assert rope.attention_scaling == ref_scaling
    assert rope.attention_scaling != 1.0  # YaRN applies an mscale; proves it ran

    # The old behavior fell back to the llama3 NTK schedule; the fix must differ.
    llama3_inv_freq, _ = _compute_llama3_inv_freq(config, torch.device("cpu"))
    assert not torch.allclose(rope.inv_freq, llama3_inv_freq)


def test_rope_linear_and_dynamic_resolve_via_transformers():
    """``linear`` and ``dynamic`` schedules resolve through transformers without error."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    for rope_type in ("linear", "dynamic"):
        config = _config_with_rope_scaling({"rope_type": rope_type, "factor": 4.0})
        rope = LlamaRotaryEmbedding(config)
        ref_inv_freq, _ = ROPE_INIT_FUNCTIONS[rope_type](config, torch.device("cpu"))
        torch.testing.assert_close(rope.inv_freq, ref_inv_freq)


def test_rope_unknown_type_raises():
    """An unrecognised rope_type fails loudly instead of guessing a schedule."""
    config = _config_with_rope_scaling({"rope_type": "default", "rope_theta": 10000.0})
    # Inject a bogus type after construction to bypass HF config validation and
    # reach the resolver. ``_get_rope_config`` reads ``rope_parameters`` first.
    bogus = {"rope_type": "does_not_exist"}
    config.rope_parameters = bogus
    config.rope_scaling = bogus
    with pytest.raises(ValueError, match="Unsupported RoPE rope_type"):
        LlamaRotaryEmbedding(config)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("config_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("fused", "layout", "cp_size"),
    [(False, "bshd", 1), (False, "thd", 1), (True, "bshd", 1), (True, "bshd", 2), (True, "thd", 1), (True, "thd", 2)],
)
def test_rope_keeps_coefficients_and_raw_angles_in_fp32(dtype, config_dtype, fused, layout, cp_size):
    """Retain FP32 coefficients and raw angles independently of activation/config dtype."""
    config = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        dtype=config_dtype,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    rope = LlamaRotaryEmbedding(config, rope_fusion=fused)
    length = 263  # Includes angle 257, which BF16 cannot represent exactly.
    offset = 257 if not fused else (length if layout == "thd" and cp_size == 2 else 0)
    positions = torch.arange(length) + offset
    shape = (length, 32) if layout == "thd" else (1, length, 32)
    if layout == "bshd":
        positions = positions.unsqueeze(0)
    x = torch.zeros(shape, dtype=dtype)
    result = rope(x, positions, qkv_format=layout, cp_size=cp_size)

    # Standard base-10000, dimension-8 RoPE: frequency_i = 10000 ** (-2*i/8).
    cache_length = max(length + offset, length * (cp_size if fused and layout == "thd" else 1))
    frequencies = torch.tensor([1.0, 0.1, 0.01, 0.001], dtype=torch.float32)
    angles = torch.outer(torch.arange(cache_length, dtype=torch.float32), frequencies).repeat(1, 2)
    expected_angles = angles[positions]
    assert result[0].dtype == result[1].dtype == torch.float32
    torch.testing.assert_close(result[0], expected_angles.cos(), rtol=0, atol=0)
    torch.testing.assert_close(result[1], expected_angles.sin(), rtol=0, atol=0)
    assert rope._cos_cache.dtype == rope._sin_cache.dtype == torch.float32
    torch.testing.assert_close(rope._cos_cache, angles.cos(), rtol=0, atol=0)
    torch.testing.assert_close(rope._sin_cache, angles.sin(), rtol=0, atol=0)
    if fused:
        assert result[2].dtype == torch.float32
        torch.testing.assert_close(result[2], angles[:, None, None, :], rtol=0, atol=0)
        assert result[2][257, 0, 0, 0].item() == 257
        assert not torch.equal(result[2], result[2].to(torch.bfloat16).float())
    else:
        assert len(result) == 2


@pytest.mark.parametrize("fused", [False, True])
def test_rope_rebuilds_fp32_cache_after_module_cast(fused):
    """A warm cache rounded by Module.to must be recomputed, not upcast."""
    rope = _build_rope(rope_fusion=fused)
    x = torch.zeros(1, 263, 32, dtype=torch.bfloat16)
    positions = torch.arange(263).unsqueeze(0)
    original = tuple(tensor.clone() for tensor in rope(x, positions))
    rope.to(torch.bfloat16)
    assert rope._cos_cache.dtype == torch.bfloat16

    rebuilt = rope(x, positions)
    for actual, expected in zip(rebuilt, original):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == torch.float32


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    ("q_dtype", "k_dtype"),
    [(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32), (torch.float32, torch.float16)],
)
def test_rope_fp32_rotation_outputs_and_gradients(packed, q_dtype, k_dtype):
    """Compare rotation and its VJP with an independent complex-number oracle."""
    torch.manual_seed(4150)
    batch, length, heads, kv_heads, dim = 2, 11, 4, 2, 8
    q_shape = (batch * length, heads, dim) if packed else (batch, heads, length, dim)
    k_shape = (batch * length, kv_heads, dim) if packed else (batch, kv_heads, length, dim)
    q = torch.randn(q_shape, dtype=q_dtype, requires_grad=True)
    k = torch.randn(k_shape, dtype=k_dtype, requires_grad=True)
    # Nonzero long positions and nontrivial frequencies expose early coefficient rounding.
    positions = torch.arange(batch * length).reshape(batch, length) + 257
    frequencies = torch.tensor([1.0, 0.1, 0.01, 0.001], dtype=torch.float32)
    angles = positions.float().unsqueeze(-1) * frequencies
    config = LlamaConfig(
        hidden_size=heads * dim,
        num_attention_heads=heads,
        head_dim=dim,
        dtype=torch.bfloat16,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    rope = LlamaRotaryEmbedding(config)
    if packed:
        angles = angles.flatten(0, 1)
        positions = positions.flatten()
    x_shape = (batch * length, heads * dim) if packed else (batch, length, heads * dim)
    cos, sin = rope(torch.zeros(x_shape, dtype=torch.bfloat16), positions, qkv_format="thd" if packed else "bshd")
    actual = apply_rotary_pos_emb(q, k, cos, sin)
    phase = torch.complex(angles.cos(), angles.sin()).unsqueeze(1)

    for source, output in zip((q, k), actual):
        source_real, source_imag = source.detach().float().chunk(2, dim=-1)
        expected_complex = torch.complex(source_real, source_imag) * phase
        expected = torch.cat((expected_complex.real, expected_complex.imag), dim=-1).to(source.dtype)
        assert output.dtype == source.dtype
        upstream = torch.randn_like(output)
        grad_real, grad_imag = upstream.float().chunk(2, dim=-1)
        expected_grad_complex = torch.complex(grad_real, grad_imag) * phase.conj()
        expected_grad = torch.cat((expected_grad_complex.real, expected_grad_complex.imag), -1).to(source.dtype)
        output.backward(upstream)
        # Complex multiplication may contract FP32 operations differently;
        # reduced-precision outputs/gradients must still agree bit for bit.
        tolerance = 2 * torch.finfo(torch.float32).eps if source.dtype == torch.float32 else 0
        torch.testing.assert_close(source.grad, expected_grad, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)

    # BF16 Q/K must still be consumable with a BF16 V by attention.
    if not packed and q_dtype == k_dtype == torch.bfloat16:
        value = torch.randn(k_shape, dtype=k_dtype)
        attention = torch.nn.functional.scaled_dot_product_attention(*actual, value, enable_gqa=True)
        assert attention.dtype == q_dtype
        assert torch.isfinite(attention).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fusion parity")
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "q_dtype,k_dtype",
    [(torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16), (torch.float32, torch.float16)],
)
@pytest.mark.runtime_budget(60, hard_timeout=70, reason="compiles CUDA RoPE forward and backward")
def test_rope_cuda_fusion_parity(packed, q_dtype, k_dtype):
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(4150)
    batch, length, heads, kv_heads, dim = 2, 17, 4, 2, 128
    q = torch.randn(batch, length, heads, dim, device="cuda", dtype=q_dtype)
    k = torch.randn(batch, length, kv_heads, dim, device="cuda", dtype=k_dtype)
    q = (q.flatten(0, 1) if packed else q.transpose(1, 2)).requires_grad_()
    k = (k.flatten(0, 1) if packed else k.transpose(1, 2)).requires_grad_()
    angle = torch.randn(batch, length, dim // 2, device="cuda")
    angle = torch.cat((angle, angle), -1)
    if packed:
        angle = angle.flatten(0, 1)
    cos, sin = angle.cos(), angle.sin()
    ref_q = q.detach().clone().requires_grad_()
    ref_k = k.detach().clone().requires_grad_()
    originals = (q.detach().clone(), k.detach().clone())

    def reference(value):
        value_fp32 = value.float()
        first, second = value_fp32.chunk(2, -1)
        rotated = torch.cat((-second, first), -1)
        return (value_fp32 * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)).to(value.dtype)

    expected = (reference(ref_q), reference(ref_k))
    actual = checkpoint(apply_rotary_pos_emb, q, k, cos, sin, use_reentrant=False)
    with torch.no_grad():
        no_grad_output = apply_rotary_pos_emb(q, k, cos, sin)
    for out, ref, no_grad in zip(actual, expected, no_grad_output):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        torch.testing.assert_close(out, no_grad, rtol=0, atol=0)
        assert out.dtype == ref.dtype
    upstream = tuple(torch.randn_like(out) for out in actual)
    torch.autograd.backward(actual, upstream)
    torch.autograd.backward(expected, upstream)
    torch.testing.assert_close(q.grad, ref_q.grad, rtol=0, atol=0)
    torch.testing.assert_close(k.grad, ref_k.grad, rtol=0, atol=0)
    torch.testing.assert_close(q, originals[0], rtol=0, atol=0)
    torch.testing.assert_close(k, originals[1], rtol=0, atol=0)
