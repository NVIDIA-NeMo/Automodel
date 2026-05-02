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

"""Unit tests for the mistral3_vlm package-level resolver hook.

Importing the package installs a hook on
``_resolve_custom_model_cls_for_config`` (model_init.py) that claims FP8
Mistral3 VLM configs and dispatches to ``Mistral3FP8VLMForConditionalGeneration``.
"""

from types import SimpleNamespace
from unittest.mock import patch

# Importing the package installs the hook (idempotent).
import nemo_automodel.components.models.mistral3_vlm  # noqa: F401
from nemo_automodel._transformers import model_init as _mi
from nemo_automodel.components.models.mistral3_vlm.model import (
    Mistral3FP8VLMForConditionalGeneration,
)


def _make_fp8_mistral3_cfg():
    return SimpleNamespace(
        text_config=SimpleNamespace(model_type="ministral3"),
        quantization_config={"quant_method": "fp8"},
    )


def _make_non_mistral3_cfg():
    return SimpleNamespace(
        text_config=SimpleNamespace(model_type="llama"),
        quantization_config={"quant_method": "fp8"},
    )


class TestHookInstallation:
    def test_hook_installed_marker(self):
        # Idempotent guard: re-importing must not stack hooks.
        assert getattr(
            _mi._resolve_custom_model_cls_for_config,
            "_mistral3_vlm_hook_installed",
            False,
        ) is True

    def test_reimport_is_idempotent(self):
        before = _mi._resolve_custom_model_cls_for_config
        # The package's _install_resolver_hook short-circuits if already installed.
        from nemo_automodel.components.models.mistral3_vlm import _install_resolver_hook
        _install_resolver_hook()
        assert _mi._resolve_custom_model_cls_for_config is before


class TestHookDispatch:
    def test_claims_fp8_mistral3_vlm(self):
        cfg = _make_fp8_mistral3_cfg()
        # Bypass the registry lookup that the inner resolver would otherwise do.
        # Our hook short-circuits before calling the original.
        assert _mi._resolve_custom_model_cls_for_config(cfg) is Mistral3FP8VLMForConditionalGeneration

    def test_passes_through_non_mistral3_to_original(self):
        cfg = _make_non_mistral3_cfg()
        sentinel = object()
        # Patch the original (un-hooked) resolver that our hook delegates to.
        # The hook closes over the *original* resolver at install time; we
        # replicate that path by stubbing ModelRegistry interactions.
        with patch.object(_mi, "get_architectures", return_value=[]):
            # When original resolver returns None (no architectures), we get None.
            assert _mi._resolve_custom_model_cls_for_config(cfg) is None

    def test_supports_config_failure_falls_back(self):
        # If supports_config raises for some reason, the hook must catch and
        # delegate to the original resolver rather than propagating.
        cfg = _make_fp8_mistral3_cfg()
        with patch.object(
            Mistral3FP8VLMForConditionalGeneration,
            "supports_config",
            side_effect=RuntimeError("boom"),
        ):
            with patch.object(_mi, "get_architectures", return_value=[]):
                # Original resolver returns None; hook swallowed the exception.
                assert _mi._resolve_custom_model_cls_for_config(cfg) is None
