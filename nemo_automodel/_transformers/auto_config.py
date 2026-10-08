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

"""Hugging Face config loading through Transformers' native revision resolution."""

import os
from typing import Any

from transformers import AutoConfig, PretrainedConfig


class NeMoAutoConfig(AutoConfig):
    """Keep the NeMo config target while delegating revision resolution to Transformers.

    For a config followed by a separate model or tokenizer load, resolve the
    revision once with ``transformers.utils.hub.resolve_revision`` and pass it
    to every load as ``revision``. Config objects do not carry Hub loading state.
    """

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read the config dictionary for Automodel's custom-config registry.

        Args:
            pretrained_model_name_or_path: Hub repository, local directory, or config file.
            **kwargs: Transformers loading options and config overrides.

        Returns:
            The config dictionary and unused loading options.
        """
        return PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
