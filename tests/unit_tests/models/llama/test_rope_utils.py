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

from unittest.mock import patch

import torch
from transformers import LlamaConfig

from nemo_automodel.components.models.llama.rope_utils import LlamaRotaryEmbedding


def _build_rope(*, head_dim: int = 8, heads: int = 4, max_pos: int = 128) -> LlamaRotaryEmbedding:
    config = LlamaConfig(
        hidden_size=head_dim * heads,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=max_pos,
    )
    return LlamaRotaryEmbedding(config)


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
