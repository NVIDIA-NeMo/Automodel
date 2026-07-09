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
"""Preload-override behavior for the magi attention backend.

magi runs its own context-parallel dispatch + masking (including sequence packing),
so it must keep its registered backend rather than being forced to SDPA/flash at
cp_size>1 (which is the TE/SDPA-only CP assumption for other backends).
"""
import pytest

from nemo_automodel._transformers.kernel_patches import _apply_preload_overrides


@pytest.mark.parametrize("cp_size", [1, 2, 8])
@pytest.mark.parametrize("has_packed_sequence", [False, True])
def test_magi_is_never_overridden(cp_size, has_packed_sequence):
    """magi keeps its backend regardless of cp_size / packing (no SDPA/flash override)."""
    attn, use_liger = _apply_preload_overrides(
        tp_size=1,
        cp_size=cp_size,
        has_packed_sequence=has_packed_sequence,
        attn_implementation="magi",
        use_liger_kernel=True,
    )
    assert attn == "magi"
    # Liger is still disabled whenever TP/CP is on (it is incompatible).
    assert use_liger is (cp_size == 1)


def test_non_magi_still_forced_to_sdpa_at_cp_gt_1():
    """Regression: non-magi backends keep the existing cp>1 -> sdpa override."""
    attn, use_liger = _apply_preload_overrides(
        tp_size=1, cp_size=2, has_packed_sequence=False,
        attn_implementation="flash_attention_2", use_liger_kernel=True,
    )
    assert attn == "sdpa"
    assert use_liger is False


def test_magi_cp1_passthrough_keeps_liger():
    """At cp_size==1 with no TP, magi passes through and Liger is untouched."""
    attn, use_liger = _apply_preload_overrides(
        tp_size=1, cp_size=1, has_packed_sequence=False,
        attn_implementation="magi", use_liger_kernel=True,
    )
    assert attn == "magi"
    assert use_liger is True
