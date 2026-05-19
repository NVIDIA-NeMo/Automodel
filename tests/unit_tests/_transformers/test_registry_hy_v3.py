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


class TestArchMapping:
    def test_hyv3_arch_registered(self):
        from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

        assert "HYV3ForCausalLM" in MODEL_ARCH_MAPPING

    def test_hyv3_arch_points_at_correct_module(self):
        from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

        entry = MODEL_ARCH_MAPPING["HYV3ForCausalLM"]
        assert entry.module_path == "nemo_automodel.components.models.hy_v3.model"
        assert entry.class_name == "HYV3ForCausalLM"

    def test_hyv3_arch_resolves_to_class(self):
        """Walk the mapping path -- importable + the named class exists."""
        import importlib

        from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

        spec = MODEL_ARCH_MAPPING["HYV3ForCausalLM"]
        mod_path = spec.module_path
        cls_name = spec.class_name
        mod = importlib.import_module(mod_path)
        assert hasattr(mod, cls_name)


class TestCustomConfigRegistration:
    def test_hy_v3_config_registered(self):
        from nemo_automodel._transformers.registry import _CUSTOM_CONFIG_SPECS

        assert any("hy_v3" in spec.model_types for spec in _CUSTOM_CONFIG_SPECS)

    def test_hy_v3_config_resolves_to_class(self):
        import importlib

        from nemo_automodel._transformers.registry import _CUSTOM_CONFIG_SPECS

        spec = next(spec for spec in _CUSTOM_CONFIG_SPECS if "hy_v3" in spec.model_types)
        mod_path = spec.config_module_path
        cls_name = spec.config_class_name
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        assert cls.__name__ == "HYV3Config"
        assert cls.model_type == "hy_v3"


class TestSupportedBackbonesIntact:
    """Sanity check that hy_v3 registration didn't disturb existing backbones."""

    def test_llama_still_in_supported_backbones(self):
        from nemo_automodel._transformers.retrieval import SUPPORTED_BACKBONES

        assert "llama" in SUPPORTED_BACKBONES
