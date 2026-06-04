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

"""Public config surface for the checkpoint component.

``CheckpointingConfig`` holds the typed parameters that drive checkpointing
behaviour and exposes ``.build()`` to construct the :class:`Checkpointer`
engine (defined in ``checkpointing.py``).  Every field has a sensible default
so the recipe layer can construct it directly from the YAML ``checkpoint:``
block plus the model-derived ``model_repo_id`` / ``model_cache_dir`` /
``is_peft`` arguments — there is no separate builder/adapter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import constants as hf_constants
from packaging.version import parse

from nemo_automodel.components.checkpoint._backports.filesystem import SerializationFormat

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer


def _is_geq_torch_2_9() -> bool:
    """Check if the current torch version is greater than or equal to 2.9.0."""
    return parse(torch.__version__).base_version >= "2.9.0"


class SaveConsolidatedMode(str, Enum):
    """Controls when consolidated HF safetensors are exported."""

    FALSE = "false"
    FINAL = "final"
    EVERY = "every"


def _normalize_save_consolidated(value: bool | str | SaveConsolidatedMode) -> SaveConsolidatedMode:
    """Normalize legacy bools and string aliases to a consolidated export mode."""
    if isinstance(value, SaveConsolidatedMode):
        return value
    if isinstance(value, bool):
        return SaveConsolidatedMode.EVERY if value else SaveConsolidatedMode.FALSE
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "every"}:
            return SaveConsolidatedMode.EVERY
        if normalized == "final":
            return SaveConsolidatedMode.FINAL
        if normalized == "false":
            return SaveConsolidatedMode.FALSE
    raise ValueError(
        "Unsupported save_consolidated value: {}. Supported values are false, final, every, and legacy booleans.".format(
            value
        )
    )


@dataclass
class CheckpointingConfig:
    """Configuration for checkpointing.

    Every field has a default so the recipe layer can construct this directly
    from the YAML ``checkpoint:`` block merged with the model-derived
    ``model_repo_id`` / ``model_cache_dir`` / ``is_peft`` values.  When
    ``model_cache_dir`` is ``None`` it falls back to the HF hub cache.
    """

    enabled: bool = True
    checkpoint_dir: str | Path = "checkpoints/"
    model_save_format: str = "safetensors"
    model_cache_dir: str | Path | None = None
    model_repo_id: str | None = None
    save_consolidated: bool | str | SaveConsolidatedMode = "final"
    is_peft: bool = False
    model_state_dict_keys: list[str] | None = (
        None  # copy of the model state dict keys before any parallelization. Kept for BW compatibility.
    )
    is_async: bool = False
    dequantize_base_checkpoint: bool | None = None
    original_model_root_dir: str | None = None
    skip_task_head_prefixes_for_base_model: list[str] | None = (
        None  # Parameter prefixes to skip when loading base model
    )
    single_rank_consolidation: bool = False  # If True, only rank 0 performs consolidation.
    # This should be used for remote storage systems that don't support direct-append or non-sequential writes.
    staging_dir: str | None = None  # Optional directory for staging files during consolidation.
    # If provided, temp files will be created here instead of system temp. Useful when system temp has limited space.
    v4_compatible: bool = False  # If True, save the original pretrained config.json (with quantization_config removed)
    # instead of the in-memory v5 config.  Useful when downstream consumers (e.g. vLLM) expect a v4-format config.
    diffusers_compatible: bool = False  # If True, use diffusers-compatible index filename
    # (diffusion_pytorch_model.safetensors.index.json) so checkpoints are loadable via diffusers from_pretrained().
    best_metric_key: str = "default"  # Validation metric key used to select the best checkpoint.

    def __post_init__(self):
        """Resolve the cache dir, enforce PEFT constraints, and coerce the save format/mode."""
        if self.model_cache_dir is None:
            self.model_cache_dir = hf_constants.HF_HUB_CACHE

        # PEFT checkpointing is not supported for `torch_save`; flip only the two
        # incompatible fields (format -> safetensors, save_consolidated -> FINAL).
        # All other user-set fields (is_async, staging_dir, v4_compatible,
        # single_rank_consolidation, ...) are preserved.
        if self.is_peft and self.model_save_format == "torch_save":
            logging.warning(
                "PEFT checkpointing is not supported for `torch_save` format; "
                "falling back to `safetensors` (all other checkpoint settings are preserved)."
            )
            self.model_save_format = "safetensors"
            self.save_consolidated = SaveConsolidatedMode.FINAL

        # Convert a raw string such as "safetensors" into the right Enum.
        formats = [v.value for v in SerializationFormat]
        assert self.model_save_format in formats, (
            f"Unsupported model save format: {self.model_save_format}. Supported formats: {formats}"
        )
        self.model_save_format = SerializationFormat[self.model_save_format.upper()]

        # Normalize legacy bools and string aliases to a consolidated export mode.
        self.save_consolidated = _normalize_save_consolidated(self.save_consolidated)
        if self.save_consolidated != SaveConsolidatedMode.FALSE and not self.is_peft:
            if not self.v4_compatible:
                logging.warning(
                    "save_consolidated=%s but v4_compatible=False; "
                    "checkpoint assets may be not compatible with transformers v4; "
                    "[experimental] set --checkpoint.v4_compatible=True to enable",
                    self.save_consolidated.value,
                )
            else:
                logging.warning("[experimental] v4_compatible=True enables transformers v4 compatibility")

        # Async is only enabled for torch >= 2.9.0 currently because of large API changes in async DCP from 2.8.0 to 2.9.0
        if self.is_async and not _is_geq_torch_2_9():
            logging.error("Async mode is only supported for torch >= 2.9.0, disabling async mode")
            self.is_async = False

    def build(
        self,
        dp_rank: int,
        tp_rank: int,
        pp_rank: int,
        moe_mesh: DeviceMesh | None = None,
    ) -> Checkpointer:
        """Build the :class:`Checkpointer` engine for this config.

        ``Checkpointer`` is imported lazily to avoid a circular import
        (``checkpointing.py`` imports ``CheckpointingConfig`` from this module)
        and to keep the heavy DCP/safetensors deps out of module load.

        Args:
            dp_rank: Data-parallel rank.
            tp_rank: Tensor-parallel rank.
            pp_rank: Pipeline-parallel rank.
            moe_mesh: Optional device mesh for MoE checkpointing.

        Returns:
            Configured :class:`Checkpointer`.
        """
        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

        return Checkpointer(config=self, dp_rank=dp_rank, tp_rank=tp_rank, pp_rank=pp_rank, moe_mesh=moe_mesh)


__all__ = ["CheckpointingConfig", "SaveConsolidatedMode"]
