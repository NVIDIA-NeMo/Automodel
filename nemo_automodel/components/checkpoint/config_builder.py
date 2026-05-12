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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from huggingface_hub import constants as hf_constants

if TYPE_CHECKING:
    from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig

logger = logging.getLogger(__name__)


def build_checkpoint_config(
    checkpoint_kwargs: Mapping[str, Any] | None,
    cache_dir: str | None,
    model_repo_id: str | None,
    is_peft: bool,
) -> "CheckpointingConfig":
    """Build a checkpoint configuration.

    Args:
        checkpoint_kwargs: Optional keyword overrides for checkpointing.
        cache_dir: Cache directory for the model.
        model_repo_id: Model repository ID.
        is_peft: Whether the model is PEFT.

    Returns:
        Instantiated checkpoint configuration.
    """
    from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig

    ckpt_kwargs = dict(
        enabled=True,
        checkpoint_dir="checkpoints/",
        model_save_format="safetensors",
        model_repo_id=model_repo_id,
        model_cache_dir=cache_dir if cache_dir is not None else hf_constants.HF_HUB_CACHE,
        save_consolidated=True,
        is_peft=is_peft,
    )
    user_cfg = {}
    if checkpoint_kwargs is not None:
        user_cfg = dict(checkpoint_kwargs)
        user_cfg.pop("restore_from", None)
    if is_peft and user_cfg.get("model_save_format") == "torch_save":
        logger.warning(
            "PEFT checkpointing is not supported for `torch_save` format; "
            "discarding user checkpoint config and using safetensors defaults "
            "(preserving `checkpoint_dir` if set)."
        )
        if "checkpoint_dir" in user_cfg:
            ckpt_kwargs["checkpoint_dir"] = user_cfg["checkpoint_dir"]
    else:
        ckpt_kwargs |= user_cfg
    checkpoint_config = CheckpointingConfig(**ckpt_kwargs)
    return checkpoint_config
