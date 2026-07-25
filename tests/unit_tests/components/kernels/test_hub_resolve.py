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

"""Tests for Hub kernel config and attn resolution helpers."""

import types

import pytest

from nemo_automodel.components.kernels import hub as hub_kernels
from nemo_automodel.components.kernels.config import HubKernelConfig


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


class TestResolveAttnImplementation:
    def test_hub_alias_resolves_to_default_repo(self):
        assert hub_kernels.resolve_attn_implementation("hub") == hub_kernels.HUB_FLASH_ATTN2

    def test_backend_hub_kernels_override(self):
        cfg = HubKernelConfig(attn_repo="kernels-community/flash-attn3")
        assert (
            hub_kernels.resolve_attn_implementation("flash_attention_2", hub_kernels=cfg)
            == "kernels-community/flash-attn3"
        )

    def test_passthrough_for_standard_impl(self):
        assert hub_kernels.resolve_attn_implementation("sdpa") == "sdpa"


class TestExtractHubKernelsConfig:
    def test_reads_from_backend_config(self):
        from nemo_automodel.components.models.common.utils import BackendConfig

        backend = BackendConfig(hub_kernels=HubKernelConfig(attn_repo="kernels-community/flash-attn2"))
        cfg = hub_kernels.extract_hub_kernels_config({"backend": backend})
        assert cfg is not None
        assert cfg.attn_repo == "kernels-community/flash-attn2"

    def test_reads_from_backend_dict(self):
        cfg = hub_kernels.extract_hub_kernels_config(
            {"backend": {"hub_kernels": {"attn_repo": "kernels-community/flash-attn4"}}}
        )
        assert cfg is not None
        assert cfg.attn_repo == "kernels-community/flash-attn4"
