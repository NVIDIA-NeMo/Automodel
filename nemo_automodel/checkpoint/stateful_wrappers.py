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

from typing import Any, Optional

import torch
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful

from nemo_automodel.checkpoint.checkpointing import SerializationFormat

# modified from pytorch tutorial https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
class ModelState(Stateful):
    """Helper class for tracking model state in distributed checkpointing.

    This class is compliant with the Stateful protocol, allowing DCP to automatically
    call state_dict/load_state_dict as needed in the dcp.save/load APIs.

    Args:
        model: The PyTorch model to track.
    """

    def __init__(self, model: torch.nn.Module, serialization_format: SerializationFormat):
        self.model = model
        self.is_tied_lm_head = getattr(getattr(model, 'config', {}), 'tie_word_embeddings', False)
        self.serialization_format = serialization_format

    def state_dict(self) -> dict[str, Any]:
        """Get the model's state dictionary.

        Returns:
            dict: Dictionary containing the model's state dict with CPU offloading enabled.
        """
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type
        # to FSDP.SHARDED_STATE_DICT
        model_state_dict = get_model_state_dict(
            self.model,
            options=StateDictOptions(
                cpu_offload=True,
                full_state_dict=True if self.serialization_format == SerializationFormat.SAFETENSORS else False,
            ),
        )
        if self.is_tied_lm_head:
            model_state_dict.pop("lm_head.weight", None)
        return model_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dictionary into the model.

        Args:
            state_dict (dict): State dictionary to load.
        """
        # If we intentionally skipped saving "lm_head.weight" (tied embeddings)
        # PyTorch will complain during load even with strict=False.
        # To be fully compatible we inject a reference tensor so the key exists.
        if self.is_tied_lm_head and "lm_head.weight" not in state_dict:
            # weight tying guarantees this is identical to the embedding weight
            state_dict["lm_head.weight"] = self.model.lm_head.weight.detach()

        set_model_state_dict(
            self.model,
            state_dict,
            options=StateDictOptions(
                full_state_dict=True if self.serialization_format == SerializationFormat.SAFETENSORS else False,
                broadcast_from_rank0=True if self.serialization_format == SerializationFormat.SAFETENSORS else False,
            ),
        )


class OptimizerState(Stateful):
    """Helper class for tracking optimizer state in distributed checkpointing.

    This class is compliant with the Stateful protocol, allowing DCP to automatically
    call state_dict/load_state_dict as needed in the dcp.save/load APIs.

    Args:
        model: The PyTorch model associated with the optimizer.
        optimizer: The optimizer to track.
        scheduler: Optional learning rate scheduler.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self) -> dict[str, Any]:
        """Get the optimizer and scheduler state dictionaries.

        Returns:
            dict: Dictionary containing the optimizer and scheduler state dicts with CPU offloading enabled.
        """
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type
        # to FSDP.SHARDED_STATE_DICT
        optimizer_state_dict = get_optimizer_state_dict(
            self.model,
            self.optimizer,
            options=torch.distributed.checkpoint.state_dict.StateDictOptions(
                cpu_offload=True
            ),
        )

        state_dict = {
            "optim": optimizer_state_dict,
        }
        if self.scheduler is not None:
            state_dict["sched"] = self.scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dictionaries into the optimizer and scheduler.

        Args:
            state_dict (dict): State dictionary containing optimizer and scheduler states to load.
        """
        # sets our state dicts on the optimizer, now that we've loaded
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            state_dict["optim"],
        )

        # load the scheduler state if it exists
        if "sched" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["sched"])