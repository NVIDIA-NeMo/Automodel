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

"""Training utilities shared across recipes."""

import importlib as _importlib
from typing import TYPE_CHECKING

from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig

if TYPE_CHECKING:
    from .domain_mixture import WEIGHTED_AGGREGATE_NAME, DomainMixtureConfig, DomainWeightConfig
    from .ema import EMAManager, ShardedModelEMAManager
    from .embedding_row_repair import EmbeddingRowRepairConfig
    from .garbage_collection import GarbageCollection
    from .model_output_utils import get_final_hidden_states
    from .neftune import NEFTune
    from .prewarm import PrewarmConfig
    from .rng import ScopedRNG, StatefulRNG, init_all_rng
    from .signal_handler import DistributedSignalHandler
    from .step_scheduler import StepScheduler
    from .timers import Timers
    from .utils import (
        ScopedModuleOffloading,
        clip_grad_norm,
        count_tail_padding,
        get_expert_tp_replication_factor,
        prepare_after_first_microbatch,
        prepare_for_final_backward,
        prepare_for_grad_accumulation,
        scale_grads_and_clip_grad_norm,
    )

__all__ = ["StepSchedulerConfig"]

_LAZY_ATTRS = {
    "DistributedSignalHandler": (".signal_handler", "DistributedSignalHandler"),
    "DomainMixtureConfig": (".domain_mixture", "DomainMixtureConfig"),
    "DomainWeightConfig": (".domain_mixture", "DomainWeightConfig"),
    "EMAManager": (".ema", "EMAManager"),
    "EmbeddingRowRepairConfig": (".embedding_row_repair", "EmbeddingRowRepairConfig"),
    "GarbageCollection": (".garbage_collection", "GarbageCollection"),
    "NEFTune": (".neftune", "NEFTune"),
    "PrewarmConfig": (".prewarm", "PrewarmConfig"),
    "ScopedModuleOffloading": (".utils", "ScopedModuleOffloading"),
    "ScopedRNG": (".rng", "ScopedRNG"),
    "ShardedModelEMAManager": (".ema", "ShardedModelEMAManager"),
    "StatefulRNG": (".rng", "StatefulRNG"),
    "StepScheduler": (".step_scheduler", "StepScheduler"),
    "Timers": (".timers", "Timers"),
    "WEIGHTED_AGGREGATE_NAME": (".domain_mixture", "WEIGHTED_AGGREGATE_NAME"),
    "clip_grad_norm": (".utils", "clip_grad_norm"),
    "count_tail_padding": (".utils", "count_tail_padding"),
    "get_expert_tp_replication_factor": (".utils", "get_expert_tp_replication_factor"),
    "get_final_hidden_states": (".model_output_utils", "get_final_hidden_states"),
    "init_all_rng": (".rng", "init_all_rng"),
    "prepare_after_first_microbatch": (".utils", "prepare_after_first_microbatch"),
    "prepare_for_final_backward": (".utils", "prepare_for_final_backward"),
    "prepare_for_grad_accumulation": (".utils", "prepare_for_grad_accumulation"),
    "scale_grads_and_clip_grad_norm": (".utils", "scale_grads_and_clip_grad_norm"),
}

__all__ += sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> object:
    """Load an exported component symbol on first access."""
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the component's exported symbols."""
    return sorted(__all__)
