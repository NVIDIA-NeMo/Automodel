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

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

from huggingface_hub import constants as hf_constants

from nemo_automodel.components.checkpoint.config import CheckpointConfig

if TYPE_CHECKING:
    from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig

logger = logging.getLogger(__name__)


def build_checkpoint_config(
    config: CheckpointConfig | None,
    cache_dir: str | None,
    model_repo_id: str | None,
    is_peft: bool,
) -> "CheckpointingConfig":
    """Build a checkpoint configuration.

    Args:
        config: User-facing checkpoint config.  ``None`` uses all defaults.
        cache_dir: HF cache directory for the model.
        model_repo_id: Model repository ID.
        is_peft: Whether the model uses PEFT.

    Returns:
        Instantiated ``CheckpointingConfig`` ready for the checkpointer.
    """
    from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig

    if config is None:
        config = CheckpointConfig()

    if is_peft and config.model_save_format == "torch_save":
        logger.warning(
            "PEFT checkpointing is not supported for `torch_save` format; "
            "discarding user checkpoint config and using safetensors defaults "
            "(preserving `checkpoint_dir` if set)."
        )
        ckpt_dir = config.checkpoint_dir
        config = CheckpointConfig(checkpoint_dir=ckpt_dir)

    user_kwargs = asdict(config)
    user_kwargs.pop("restore_from", None)

    return CheckpointingConfig(
        enabled=True,
        model_repo_id=model_repo_id,
        model_cache_dir=cache_dir if cache_dir is not None else hf_constants.HF_HUB_CACHE,
        is_peft=is_peft,
        **user_kwargs,
    )
