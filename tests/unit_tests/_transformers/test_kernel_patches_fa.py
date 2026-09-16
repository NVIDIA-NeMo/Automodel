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

"""FA2/FA3/FA4 selection logic in kernel_patches (fallback ladder, packed-sequence overrides)."""

import pytest

from nemo_automodel._transformers import kernel_patches
from nemo_automodel._transformers.kernel_patches import (
    FLASH_ATTN_IMPLEMENTATIONS,
    _apply_preload_overrides,
    _device_supports_fa3,
    _get_next_fallback_attn,
)


class TestFlashAttnDetection:
    def test_flag_constants_exist(self):
        assert isinstance(kernel_patches.HAS_FA3, bool)
        assert isinstance(kernel_patches.HAS_FA4, bool)

    def test_flash_attn_implementations_tuple(self):
        assert FLASH_ATTN_IMPLEMENTATIONS == (
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
        )


class TestGetNextFallbackAttn:
    @pytest.mark.parametrize(
        "current,expected",
        [
            ("flash_attention_4", "flash_attention_3"),
            ("flash_attention_3", "flash_attention_2"),
            ("flash_attention_2", "sdpa"),
            ("sdpa", "eager"),
            ("eager", "eager"),
            ("unknown_impl", "eager"),
        ],
    )
    def test_fallback_ladder(self, current, expected):
        assert _get_next_fallback_attn(current) == expected


class TestApplyPreloadOverridesPacked:
    @pytest.mark.parametrize("impl", FLASH_ATTN_IMPLEMENTATIONS)
    def test_packed_keeps_requested_flash_version(self, impl):
        """An explicitly requested FA2/FA3/FA4 must survive the packed-sequence override."""
        attn, liger = _apply_preload_overrides(
            tp_size=1, cp_size=1, has_packed_sequence=True, attn_implementation=impl, use_liger_kernel=False
        )
        assert attn == impl

    @pytest.mark.parametrize(
        "repo_id",
        [
            "kernels-community/flash-attn2",
            "kernels-community/flash-attn3",
            "kernels-community/vllm-flash-attn3",
            "kernels-community/aiter-flash-attn",
            "kernels-community/flash-attn4",
            "kernels-community/flash-attn4@main",
        ],
    )
    def test_packed_keeps_requested_hub_flash_without_pip_flash(self, monkeypatch, repo_id):
        """An explicit Hub kernel remains selected on a Hub-only host."""
        monkeypatch.setattr(kernel_patches, "HAS_FA", False)
        monkeypatch.setattr(kernel_patches, "HAS_FA3", False)
        monkeypatch.setattr(kernel_patches, "HAS_FA4", False)

        attn, _ = _apply_preload_overrides(
            tp_size=1,
            cp_size=1,
            has_packed_sequence=True,
            attn_implementation=repo_id,
            use_liger_kernel=False,
        )

        assert attn == repo_id

    def test_packed_non_flash_falls_back_to_available_flash(self, monkeypatch):
        monkeypatch.setattr(kernel_patches, "HAS_FA", True)
        attn, _ = _apply_preload_overrides(
            tp_size=1, cp_size=1, has_packed_sequence=True, attn_implementation="sdpa", use_liger_kernel=False
        )
        assert attn == "flash_attention_2"

    def test_packed_non_flash_prefers_fa3_when_fa2_missing(self, monkeypatch):
        monkeypatch.setattr(kernel_patches, "HAS_FA", False)
        monkeypatch.setattr(kernel_patches, "HAS_FA3", True)
        monkeypatch.setattr(kernel_patches, "HAS_FA4", False)
        attn, _ = _apply_preload_overrides(
            tp_size=1, cp_size=1, has_packed_sequence=True, attn_implementation="sdpa", use_liger_kernel=False
        )
        assert attn == "flash_attention_3"

    def test_packed_non_flash_uses_fa4_as_last_resort(self, monkeypatch):
        monkeypatch.setattr(kernel_patches, "HAS_FA", False)
        monkeypatch.setattr(kernel_patches, "HAS_FA3", False)
        monkeypatch.setattr(kernel_patches, "HAS_FA4", True)
        attn, _ = _apply_preload_overrides(
            tp_size=1, cp_size=1, has_packed_sequence=True, attn_implementation="sdpa", use_liger_kernel=False
        )
        assert attn == "flash_attention_4"

    def test_packed_no_flash_available_asserts(self, monkeypatch):
        monkeypatch.setattr(kernel_patches, "HAS_FA", False)
        monkeypatch.setattr(kernel_patches, "HAS_FA3", False)
        monkeypatch.setattr(kernel_patches, "HAS_FA4", False)
        with pytest.raises(AssertionError):
            _apply_preload_overrides(
                tp_size=1, cp_size=1, has_packed_sequence=True, attn_implementation="sdpa", use_liger_kernel=False
            )

    def test_cp_still_forces_sdpa(self):
        attn, _ = _apply_preload_overrides(
            tp_size=1,
            cp_size=2,
            has_packed_sequence=False,
            attn_implementation="flash_attention_3",
            use_liger_kernel=False,
        )
        assert attn == "sdpa"


class TestDeviceSupportsFA3:
    """FA3 is SM90a-only; the device gate must reject non-Hopper capabilities."""

    @pytest.mark.parametrize(
        "capability,expected",
        [((9, 0), True), ((10, 0), False), ((12, 0), False), ((8, 0), False)],
    )
    def test_capability_gate(self, monkeypatch, capability, expected):
        monkeypatch.setattr(kernel_patches.torch.cuda, "get_device_capability", lambda: capability)
        assert _device_supports_fa3() is expected
