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

"""Tests for the ``magi`` branch of the custom-model attention factory.

The head-dim guards (MLA / head_dim > 128) reject the config *before* importing
``magi_attention``, so they run on a CPU CI runner with magi absent. The valid
path that actually constructs the FFA attn_func is gated behind the package.
"""

from __future__ import annotations

import pytest

from nemo_automodel.components.attention.utils import initialize_attn_module_and_func


def test_magi_rejects_mla_unequal_head_dim():
    """MLA (qk_head_dim != v_head_dim, e.g. DeepSeek-V3/Moonlight) is unsupported."""
    with pytest.raises(ValueError, match="MLA-style"):
        initialize_attn_module_and_func(
            attn_impl="magi",
            num_attention_heads=16,
            num_qk_channels=192,  # 128 nope + 64 rope
            num_v_channels=128,
            softmax_scale=192**-0.5,
        )


def test_magi_rejects_head_dim_over_128():
    """head_dim > 128 (e.g. Gemma3 / Qwen3.5 full-attention = 256) is unsupported."""
    with pytest.raises(ValueError, match="head_dim <= 128"):
        initialize_attn_module_and_func(
            attn_impl="magi",
            num_attention_heads=16,
            num_qk_channels=256,
            num_v_channels=256,
            softmax_scale=256**-0.5,
        )


def test_magi_valid_head_dim_returns_callable():
    """A supported head_dim returns ``(None, attn_func)`` (needs magi installed)."""
    pytest.importorskip("magi_attention")
    attn_module, attn_func = initialize_attn_module_and_func(
        attn_impl="magi",
        num_attention_heads=16,
        num_qk_channels=128,
        num_v_channels=128,
        softmax_scale=128**-0.5,
    )
    assert attn_module is None
    assert callable(attn_func)
