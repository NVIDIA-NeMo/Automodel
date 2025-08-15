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
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

_PREFIX = "model."


def _drop_outer_prefix(sd: dict[str, Any], prefix: str = _PREFIX) -> None:
    """
    Remove the *first* occurrence of `prefix` on every key in-place.
    """
    for k in list(sd.keys()):
        if k.startswith(prefix):
            sd[k[len(prefix) :]] = sd.pop(k)


def _add_outer_prefix(sd: dict[str, Any], prefix: str = _PREFIX, skip_keys: list[str] = []) -> None:
    """
    Prepend `prefix` once to every key in-place (inverse of `_drop_outer_prefix`).
    """
    for k in list(sd.keys()):
        if not k.startswith(prefix) and k not in skip_keys:
            sd[prefix + k] = sd.pop(k)


def _get_lm_head_weight_and_name(model: torch.nn.Module) -> Optional[tuple[torch.Tensor, str]]:
    for name, param in model.named_parameters(remove_duplicate=False):
        if "lm_head" in name and name.endswith(".weight"):
            return param, name

    return None, None


# modified from pytorch tutorial https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
class ModelState:
    """
    Helper class for tracking model state in distributed checkpointing.

    This class is compliant with the Stateful protocol, allowing DCP to automatically
    call state_dict/load_state_dict as needed in the dcp.save/load APIs.

    Args:
        model: The PyTorch model to track.
    """

    def __init__(self, model: torch.nn.Module, is_peft: bool = False, is_init_step: bool = False):
        """
        Initialize a ModelState instance for distributed checkpointing.

        The constructor records the model reference, detects whether the model
        ties its language-model head to the input embeddings, and stores the
        desired serialization backend so that DCP can correctly save and restore
        the model’s parameters and buffers.

        Args:
            model (torch.nn.Module): The PyTorch model whose state should be
                captured during checkpointing.
            is_peft (bool): Whether the model is PEFT.
            is_init_step (bool): Whether the model is being initialized.
        """
        self.model = model
        self.is_tied_lm_head = getattr(getattr(model, "config", {}), "tie_word_embeddings", False)
        self.is_peft = is_peft
        self.is_init_step = is_init_step

    def state_dict(self) -> dict[str, Any]:
        """
        Get the model's state dictionary.

        Returns:
            dict: Dictionary containing the model's state dict with CPU offloading enabled.
        """
        if self.is_init_step:
            return self._get_base_model_state_dict()

        options = None
        if self.is_peft:
            options = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)

        model_state_dict = get_model_state_dict(self.model, options=options)

        if self.is_tied_lm_head:
            _, lm_head_param_name = _get_lm_head_weight_and_name(self.model)
            model_state_dict.pop(lm_head_param_name, None)

        if self.is_peft:
            # HF PEFT models are saved with a "base.model." prefix. This is so they can be loaded
            # correctly with the HF PEFT API.
            _add_outer_prefix(model_state_dict, "base_model.model.")

        return model_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state dictionary into the model.

        Args:
            state_dict (dict): State dictionary to load.
        """
        if self.is_init_step:
            self._set_base_model_state_dict(state_dict)
            return

        options = None
        if self.is_peft:
            _drop_outer_prefix(state_dict, "base_model.model.")
            options = StateDictOptions(strict=False, broadcast_from_rank0=True, full_state_dict=True)

        # If we intentionally skipped saving "lm_head.weight" (tied embeddings)
        # PyTorch will complain during load even with strict=False.
        # To be fully compatible we inject a reference tensor so the key exists.
        if self.is_tied_lm_head and not self.is_peft:
            lm_head_weight, lm_head_param_name = _get_lm_head_weight_and_name(self.model)
            if lm_head_param_name not in state_dict:
                # weight tying guarantees this is identical to the embedding weight
                state_dict[lm_head_param_name] = lm_head_weight.detach()

        set_model_state_dict(
            self.model,
            state_dict,
            options=options,
        )

    def _get_base_model_state_dict(self) -> dict[str, Any]:
        model_state_dict = get_model_state_dict(self.model)
        if self.is_tied_lm_head:
            _, lm_head_param_name = _get_lm_head_weight_and_name(self.model)
            model_state_dict.pop(lm_head_param_name, None)
        if self.is_peft:
            keys_to_remove = [k for k in model_state_dict.keys() if "lora" in k]
            for k in keys_to_remove:
                model_state_dict.pop(k)

        return model_state_dict

    def _set_base_model_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(self.model, state_dict, options=StateDictOptions(strict=False))


class OptimizerState:
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
