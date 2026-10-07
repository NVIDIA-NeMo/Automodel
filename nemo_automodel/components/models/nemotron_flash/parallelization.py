# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Nemotron-Flash tensor-parallel policy and checkpoint identification."""

from __future__ import annotations

import logging

from torch import nn
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Shard

from nemo_automodel.components.models.llama.parallelization import LlamaParallelizer

logger = logging.getLogger(__name__)


def is_nemotron_flash_config(config) -> bool:
    """Return whether a Transformers config identifies Nemotron-Flash."""
    if config is None:
        return False
    if getattr(config, "model_type", None) == "nemotron_flash":
        return True
    if "NemotronFlashForCausalLM" in (getattr(config, "architectures", None) or ()):
        return True
    return "nemotron-flash" in (getattr(config, "name_or_path", "") or "").lower()


def adjust_flash_tp_plan(plan: dict[str, ParallelStyle]) -> dict[str, ParallelStyle]:
    """Keep logits and their weight norm in compatible vocabulary layouts."""
    for name in ("lm_head", "language_model.lm_head"):
        style = plan.get(name)
        layouts = getattr(style, "output_layouts", ())
        if not isinstance(layouts, (tuple, list)):
            layouts = (layouts,)
        if any(isinstance(layout, Shard) for layout in layouts):
            logger.info("Nemotron-Flash: retaining vocab-sharded %s TP plan.", name)
        elif plan.pop(name, None) is not None:
            logger.info("Nemotron-Flash: excluding replicated-output %s from TP plan.", name)
    return plan


class NemotronFlashParallelizer(LlamaParallelizer):
    """Preserve the remote-code model's normalized vocabulary projection."""

    def _finalize_tp_plan(self, model: nn.Module, plan: dict[str, ParallelStyle]) -> dict[str, ParallelStyle]:
        return adjust_flash_tp_plan(plan) if is_nemotron_flash_config(getattr(model, "config", None)) else plan


PARALLELIZER = NemotronFlashParallelizer()

__all__ = ["PARALLELIZER"]
