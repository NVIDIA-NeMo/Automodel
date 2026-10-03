# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Deprecated Qwen3.8-Flash-Next backend config, kept so existing YAML ``_target_`` paths still load."""

import logging
from dataclasses import dataclass
from typing import Literal

from nemo_automodel.components.models.common import BackendConfig

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Qwen3_8_FlashNextBackendConfig(BackendConfig):
    """Deprecated alias of :class:`BackendConfig` for Qwen3.8-Flash-Next.

    FA4 QSA is now selected with the shared ``BackendConfig(attn="fa4")``. This class only keeps YAMLs that target
    ``nemo_automodel.components.models.qwen3_8_flash_next.backend.Qwen3_8_FlashNextBackendConfig`` working: the
    former ``attn="cute"`` is translated to ``"fa4"`` with a migration warning. All other settings are inherited.
    """

    attn: Literal["te", "sdpa", "flex", "eager", "tilelang", "cudnn", "fa4", "cute"] = BackendConfig.attn

    def __post_init__(self) -> None:
        logger.warning(
            "Qwen3_8_FlashNextBackendConfig is deprecated; use nemo_automodel.components.models.common.BackendConfig."
        )
        if self.attn == "cute":
            logger.warning("backend.attn='cute' is deprecated for Qwen3.8-Flash-Next; use backend.attn='fa4'.")
            self.attn = "fa4"
        super().__post_init__()
