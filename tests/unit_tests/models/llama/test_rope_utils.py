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
