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

"""DeepSeek V4 integration for MXFP4-resident experts under PEFT."""

import logging

import torch.nn as nn

from nemo_automodel.components.distributed.init_utils import get_world_size_safe

logger = logging.getLogger(__name__)


def prepare_mxfp4_peft_checkpoint_load(model: nn.Module, peft_config) -> None:
    """Prepare a DeepSeek V4 model to load routed experts directly in MXFP4."""
    if getattr(peft_config, "expert_weight_format", "bf16") != "mxfp4":
        return

    from nemo_automodel.components._peft.lora import convert_frozen_experts_to_mxfp4

    # MXFP4-resident experts rely on the MoE parallelizer to shard both packed
    # values and scales. Without it, the scale tensors are not applied correctly.
    if get_world_size_safe() == 1:
        raise ValueError(
            "peft.expert_weight_format='mxfp4' requires expert parallelism "
            "(multi-GPU with distributed.ep_size>1); it is not supported on a single GPU "
            "(the packed expert scales are not applied without the MoE parallelizer). "
            "Use ep_size>1, or set expert_weight_format='bf16'."
        )

    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is not None and hasattr(adapter, "expert_storage_format"):
        adapter.expert_storage_format = "mxfp4"

    num_converted = convert_frozen_experts_to_mxfp4(model, passthrough=True)
    logger.info("Converted %d frozen expert module(s) to mxfp4-resident storage (passthrough)", num_converted)


def finalize_mxfp4_peft_checkpoint_load(model: nn.Module, peft_config) -> None:
    """Pack any deferred DeepSeek V4 expert weights after checkpoint loading."""
    if getattr(peft_config, "expert_weight_format", "bf16") != "mxfp4":
        return

    from nemo_automodel.components._peft.lora import pack_mxfp4_expert_base_weights

    num_packed = pack_mxfp4_expert_base_weights(model)
    if num_packed:
        logger.info("Packed %d MoE expert modules to mxfp4-resident storage", num_packed)
