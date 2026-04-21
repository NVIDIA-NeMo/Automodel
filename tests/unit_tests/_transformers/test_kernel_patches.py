# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for kernel_patches helpers."""

import pytest

from nemo_automodel._transformers.kernel_patches import _apply_preload_overrides


class TestApplyPreloadOverridesPackedSequenceGuard:
    """Guard prevents the FA2-retry loop for models that don't support FA2."""

    def test_forces_fa2_on_first_call(self):
        attn, use_liger = _apply_preload_overrides(
            tp_size=1,
            cp_size=1,
            has_packed_sequence=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
        )
        assert attn == "flash_attention_2"
        assert use_liger is True

    def test_forces_fa2_when_attn_is_none(self):
        attn, _ = _apply_preload_overrides(
            tp_size=1,
            cp_size=1,
            has_packed_sequence=True,
            attn_implementation=None,
            use_liger_kernel=False,
        )
        assert attn == "flash_attention_2"

    def test_raises_on_retry_with_sdpa(self):
        # Simulates the fallback path: FA2 failed, retry entered with attn=sdpa.
        # Without the guard, the override would force FA2 again and loop.
        with pytest.raises(ValueError, match="Packed sequences require Flash Attention"):
            _apply_preload_overrides(
                tp_size=1,
                cp_size=1,
                has_packed_sequence=True,
                attn_implementation="sdpa",
                use_liger_kernel=False,
            )

    def test_raises_on_retry_with_eager(self):
        with pytest.raises(ValueError, match="Packed sequences require Flash Attention"):
            _apply_preload_overrides(
                tp_size=1,
                cp_size=1,
                has_packed_sequence=True,
                attn_implementation="eager",
                use_liger_kernel=False,
            )

    def test_error_message_includes_model_name(self):
        with pytest.raises(ValueError, match=r"nvidia/NemotronH"):
            _apply_preload_overrides(
                tp_size=1,
                cp_size=1,
                has_packed_sequence=True,
                attn_implementation="sdpa",
                use_liger_kernel=False,
                model_name_or_path="nvidia/NemotronH",
            )

    def test_fa3_still_allowed(self):
        attn, _ = _apply_preload_overrides(
            tp_size=1,
            cp_size=1,
            has_packed_sequence=True,
            attn_implementation="flash_attention_3",
            use_liger_kernel=False,
        )
        # FA3 passes the guard; override still normalizes to FA2.
        assert attn == "flash_attention_2"


class TestApplyPreloadOverridesUnchangedPaths:
    """Non-packed-sequence and CP paths are unaffected by the new guard."""

    def test_no_packed_sequence_passthrough(self):
        attn, use_liger = _apply_preload_overrides(
            tp_size=1,
            cp_size=1,
            has_packed_sequence=False,
            attn_implementation="sdpa",
            use_liger_kernel=True,
        )
        assert attn == "sdpa"
        assert use_liger is True

    def test_cp_forces_sdpa_and_disables_liger(self):
        attn, use_liger = _apply_preload_overrides(
            tp_size=1,
            cp_size=2,
            has_packed_sequence=False,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
        )
        assert attn == "sdpa"
        assert use_liger is False

    def test_tp_disables_liger(self):
        _, use_liger = _apply_preload_overrides(
            tp_size=2,
            cp_size=1,
            has_packed_sequence=False,
            attn_implementation="sdpa",
            use_liger_kernel=True,
        )
        assert use_liger is False

    def test_packed_sequence_with_cp_errors(self):
        with pytest.raises(ValueError, match="CP size 1"):
            _apply_preload_overrides(
                tp_size=1,
                cp_size=2,
                has_packed_sequence=True,
                attn_implementation="flash_attention_2",
                use_liger_kernel=False,
            )
