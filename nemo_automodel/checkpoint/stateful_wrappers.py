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
)
from torch.distributed.checkpoint.stateful import Stateful

from nemo_automodel.checkpoint._backports.filesystem import SerializationFormat


# modified from pytorch tutorial https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
class ModelState(Stateful):
    """
    Helper class for tracking model state in distributed checkpointing.

    This class is compliant with the Stateful protocol, allowing DCP to automatically
    call state_dict/load_state_dict as needed in the dcp.save/load APIs.

    Args:
        model: The PyTorch model to track.
    """

    def __init__(self, model: torch.nn.Module, serialization_format: SerializationFormat):
        """
        Initialize a ModelState instance for distributed checkpointing.

        The constructor records the model reference, detects whether the model
        ties its language-model head to the input embeddings, and stores the
        desired serialization backend so that DCP can correctly save and restore
        the model’s parameters and buffers.

        Args:
            model (torch.nn.Module): The PyTorch model whose state should be
                captured during checkpointing.
            serialization_format (SerializationFormat): Backend/format to use when
                persisting the model state (e.g., torch, safetensors).
        """
        self.model = model
        self.is_tied_lm_head = getattr(getattr(model, 'config', {}), 'tie_word_embeddings', False)
        self.serialization_format = serialization_format

    def state_dict(self) -> dict[str, Any]:
        """
        Get the model's state dictionary.

        Returns:
            dict: Dictionary containing the model's state dict with CPU offloading enabled.
        """
        # this line automatically manages FSDP FQN's
        model_state_dict = get_model_state_dict(self.model)

        # This is a hack to fix the issue with the model state dict being saved with the "model.model." prefix.
        # This is necessary when saving consolidated safetensors. This is because calling HF's
        # .from_pretrained() requires the model to be saved with a single "model." prefix.
        if self.serialization_format == SerializationFormat.SAFETENSORS:
            keys_to_fix = [k for k in model_state_dict if k.startswith("model.")]
            for old_key in keys_to_fix:
                new_key = old_key[len("model."):]
                # avoid overwriting if new_key already exists (shouldn't happen, but be safe)
                if new_key not in model_state_dict:
                    model_state_dict[new_key] = model_state_dict[old_key]
                # delete the old, over-prefixed key
                del model_state_dict[old_key]

        if self.is_tied_lm_head:
            model_state_dict.pop("lm_head.weight", None)
        return model_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state dictionary into the model.

        Args:
            state_dict (dict): State dictionary to load.
        """
        # Undo the prefix-stripping that happened at save-time: DCP removes the
        # container name ("model") when it dispatches the dict to this
        # ModelState, so every key now lacks the leading "model." segment that
        # HuggingFace modules normally carry.  Re-add it so that
        # set_model_state_dict can match parameters correctly.
        if self.serialization_format == SerializationFormat.SAFETENSORS:
            keys_to_fix = [k for k in state_dict if not k.startswith("model.") and k != "lm_head.weight"]
            for old_key in keys_to_fix:
                new_key = f"model.{old_key}"
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict[old_key]
                del state_dict[old_key]

        # If we intentionally skipped saving "lm_head.weight" (tied embeddings)
        # PyTorch will complain during load even with strict=False.
        # To be fully compatible we inject a reference tensor so the key exists.
        if self.is_tied_lm_head and "lm_head.weight" not in state_dict:
            # weight tying guarantees this is identical to the embedding weight
            state_dict["lm_head.weight"] = self.model.lm_head.weight.detach()

        set_model_state_dict(
            self.model,
            state_dict,
        )


class OptimizerState(Stateful):
    """
    Helper class for tracking optimizer state in distributed checkpointing.

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
        """
        Initialize an OptimizerState instance.

        The constructor simply stores references to the model, optimizer, and
        (optionally) learning-rate scheduler so that their state can be captured
        and restored by the Distributed Checkpointing (DCP) framework.

        Args:
            model (torch.nn.Module): The neural-network model whose parameters the
                optimizer updates. Keeping the reference allows DCP to re-establish
                the model–optimizer relationship when loading a checkpoint.
            optimizer (torch.optim.Optimizer): Optimizer whose internal buffers
                (e.g., momentum, Adam moments, step counters) need to be saved and
                restored.
            scheduler (Optional[Any], optional): Learning-rate scheduler to track
                alongside the optimizer. Pass ``None`` if no scheduler is used.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self) -> dict[str, Any]:
        """
        Get the optimizer and scheduler state dictionaries.

        Returns:
            dict: Dictionary containing the optimizer and scheduler state dicts with CPU offloading enabled.
        """
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type
        # to FSDP.SHARDED_STATE_DICT
        optimizer_state_dict = get_optimizer_state_dict(
            self.model,
            self.optimizer,
        )

        state_dict = {
            "optim": optimizer_state_dict,
        }
        if self.scheduler is not None:
            state_dict["sched"] = self.scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state dictionaries into the optimizer and scheduler.

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