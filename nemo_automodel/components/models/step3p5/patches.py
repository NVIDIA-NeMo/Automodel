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

"""Compatibility patches for Step-3.5 model configs."""

import logging

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


def load_config_with_layer_types_fix(
    pretrained_model_name_or_path,
    attn_implementation,
    trust_remote_code,
    **kwargs,
):
    """Load an HF config after truncating ``layer_types`` to ``num_hidden_layers``."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
    n = config_dict.get("num_hidden_layers")
    lt = config_dict.get("layer_types")
    if isinstance(n, int) and isinstance(lt, list) and len(lt) > n:
        logger.warning(
            "Truncating layer_types (len=%d) to num_hidden_layers=%d for %s",
            len(lt),
            n,
            pretrained_model_name_or_path,
        )
        config_dict["layer_types"] = lt[:n]

    config_cls = None
    auto_map = config_dict.get("auto_map") or {}
    if trust_remote_code and "AutoConfig" in auto_map:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config_cls = get_class_from_dynamic_module(auto_map["AutoConfig"], pretrained_model_name_or_path)
    if config_cls is None:
        model_type = config_dict.get("model_type")
        config_cls = CONFIG_MAPPING.get(model_type)
    if config_cls is None:
        raise ValueError(
            f"Could not resolve config class for {pretrained_model_name_or_path} "
            f"(model_type={config_dict.get('model_type')!r})"
        )
    return config_cls.from_dict(config_dict, attn_implementation=attn_implementation)
