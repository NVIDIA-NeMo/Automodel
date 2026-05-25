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

Look here for the typed parameters that drive checkpointing behaviour.
Look at ``api.py`` for the builder that consumes this config.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckpointConfig:
    """User-facing checkpoint configuration (maps to the YAML ``checkpoint:`` block).

    Runtime-only values (``model_repo_id``, ``cache_dir``, ``is_peft``)
    are passed separately to ``build_checkpoint_config`` in ``api.py``.

    Attributes:
        checkpoint_dir: Directory for saving / loading checkpoints.
        model_save_format: Serialization format — ``"safetensors"`` or ``"torch_save"``.
        save_consolidated: Consolidate FSDP shards into a single file on save.
        is_async: Use async DCP writes (requires torch >= 2.9.0).
        restore_from: Path to restore weights from before training.
            Consumed by the recipe; stripped before building ``CheckpointingConfig``.
        single_rank_consolidation: Only rank 0 performs consolidation
            (needed for remote storage without append support).
        staging_dir: Temp directory for consolidation files.
        v4_compatible: Write HF-transformers v4 compatible ``config.json``.
        diffusers_compatible: Use diffusers-compatible index filename.
        best_metric_key: Validation metric key used to select the best checkpoint.
        dequantize_base_checkpoint: Dequantize quantized weights on save.
        original_model_root_dir: Root dir of the original model (for delta saves).
        skip_task_head_prefixes_for_base_model: Parameter prefixes to skip
            when loading the base model.
    """

    checkpoint_dir: str = "checkpoints/"
    model_save_format: str = "safetensors"
    save_consolidated: bool = True
    is_async: bool = False
    restore_from: str | None = None
    single_rank_consolidation: bool = False
    staging_dir: str | None = None
    v4_compatible: bool = False
    diffusers_compatible: bool = False
    best_metric_key: str = "default"
    dequantize_base_checkpoint: bool | None = None
    original_model_root_dir: str | None = None
    skip_task_head_prefixes_for_base_model: list[str] | None = field(default=None)


__all__ = ["CheckpointConfig"]
