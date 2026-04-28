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

"""Verify HYV3 model + config are registered in nemo_automodel._transformers.registry."""

import pytest


class TestArchMapping:
    def test_hyv3_arch_registered(self):
        from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

        assert "HYV3ForCausalLM" in MODEL_ARCH_MAPPING

    def test_hyv3_arch_points_at_correct_module(self):
        from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

        entry = MODEL_ARCH_MAPPING["HYV3ForCausalLM"]
        assert entry[0] == "nemo_automodel.components.models.hy_v3.model"
        assert entry[1] == "HYV3ForCausalLM"

    def test_hyv3_arch_resolves_to_class(self):
        """Walk the mapping path -- importable + the named class exists."""
        import importlib

        from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

        mod_path, cls_name, *_ = MODEL_ARCH_MAPPING["HYV3ForCausalLM"]
        mod = importlib.import_module(mod_path)
        assert hasattr(mod, cls_name)


class TestCustomConfigRegistration:
    def test_hy_v3_config_registered(self):
        from nemo_automodel._transformers.registry import _CUSTOM_CONFIG_REGISTRATIONS

        assert "hy_v3" in _CUSTOM_CONFIG_REGISTRATIONS

    def test_hy_v3_config_resolves_to_class(self):
        import importlib

        from nemo_automodel._transformers.registry import _CUSTOM_CONFIG_REGISTRATIONS

        mod_path, cls_name = _CUSTOM_CONFIG_REGISTRATIONS["hy_v3"]
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        assert cls.__name__ == "HYV3Config"
        assert cls.model_type == "hy_v3"


class TestRetrievalMappingSync:
    """Sync-state tests: the PR removed the Ministral3 bidirectional retrieval
    backbone; ensure those keys are no longer in SUPPORTED_BACKBONES."""

    def test_ministral3_not_in_supported_backbones(self):
        from nemo_automodel._transformers.retrieval import SUPPORTED_BACKBONES

        assert "ministral3" not in SUPPORTED_BACKBONES
        assert "ministral3_bidirec" not in SUPPORTED_BACKBONES
        assert "mistral3" not in SUPPORTED_BACKBONES
