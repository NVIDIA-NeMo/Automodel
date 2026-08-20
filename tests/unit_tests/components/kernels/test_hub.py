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

"""Unit tests for Hub kernel helper utilities."""

import types

import pytest

from nemo_automodel.components.kernels import hub as hub_kernels


class TestIsHubAttnImplementation:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("kernels-community/flash-attn2", True),
            ("kernels-community/flash-attn2:FlashAttention2", True),
            ("kernels-community/flash-attn2@v2.1.0", True),
            ("flash_attention_2", False),
            ("sdpa", False),
            ("https://huggingface.co/kernels-community/flash-attn2", False),
            ("", False),
        ],
    )
    def test_hub_repo_detection(self, value, expected):
        assert hub_kernels.is_hub_attn_implementation(value) is expected


class TestHasFlashAttnAvailable:
    def test_compiled_package_short_circuits(self, monkeypatch):
        monkeypatch.setattr(hub_kernels, "HAS_COMPILED_FA", True)
        monkeypatch.setattr(hub_kernels, "_hub_flash_attn_module", lambda *args, **kwargs: None)
        assert hub_kernels.has_flash_attn_available() is True

    def test_hub_fallback_when_no_pip_package(self, monkeypatch):
        monkeypatch.setattr(hub_kernels, "HAS_COMPILED_FA", False)
        fake_mod = types.ModuleType("fake_fa")
        monkeypatch.setattr(hub_kernels, "_hub_flash_attn_module", lambda *args, **kwargs: fake_mod)
        assert hub_kernels.has_flash_attn_available() is True


class TestGetFlashAttnVarlenFunc:
    def test_prefers_compiled_package(self, monkeypatch):
        sentinel = object()

        def fake_safe_import_from(module, name, **kwargs):
            if module == "flash_attn" and name == "flash_attn_varlen_func":
                return True, sentinel
            return False, None

        monkeypatch.setattr(hub_kernels, "safe_import_from", fake_safe_import_from)
        assert hub_kernels.get_flash_attn_varlen_func() is sentinel

    def test_returns_hub_varlen_when_pip_missing(self, monkeypatch):
        sentinel = object()
        fake_mod = types.SimpleNamespace(flash_attn_varlen_func=sentinel)

        monkeypatch.setattr(
            hub_kernels,
            "safe_import_from",
            lambda module, name, **kwargs: (False, None),
        )
        monkeypatch.setattr(hub_kernels, "_hub_flash_attn_module", lambda *args, **kwargs: fake_mod)
        assert hub_kernels.get_flash_attn_varlen_func() is sentinel
