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

import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer

from nemo_automodel.checkpoint.checkpointing import (
    load_model,
    load_optimizer,
    save_model,
    save_optimizer,
)


def has_load_restore_state(object):
    """
    Checks whether object has load_state_dict and state_dict functions.

    TODO: also need to check function signatures.

    Args:
        object (any): the object to check.

    Returns:
        bool: returns True if has callable load_state_dict and state_dict
    """
    return all(callable(getattr(object, attr, None)) for attr in ("load_state_dict", "state_dict"))


class BaseRecipe:
    """
    BaseRecipe provides checkpoint load/save functionality for recipes.
    """

    def __setattr__(self, key, value):
        """
        Overriden __setattr__ to keep track of stateful classes.

        Args:
            key (str): attribute named.
            value (Any): Value assigned

        Raises:
            ValueError: if __state_tracked is attemped to be overwriten.

        """
        # assuming no one will do recipe.__dict__['__state_tracked'] = None
        if key == "__state_tracked":
            raise ValueError("cannot set __state_tracked")
        if "__state_tracked" not in self.__dict__:
            self.__dict__["__state_tracked"] = set()
        # Track stateful objects unless they are validation/eval components.
        should_track = isinstance(value, (nn.Module, Optimizer)) or has_load_restore_state(value)

        if should_track and not any(substr in key.lower() for substr in ("val", "eval", "test")):
            assert key not in self.__dict__["__state_tracked"]
            self.__dict__["__state_tracked"].add(key)
        super().__setattr__(key, value)

    def save_checkpoint(self, epoch: int, step: int):
        """
        Save the current training state as a checkpoint.

        As long as the object has a 'load_state_dict' and 'state_dict' function, it will be saved.

        Args:
            epoch (int): The current epoch.
            step (int): The current step.
        """
        if not self.checkpoint_config.enabled:
            return

        path = self.checkpoint_config.checkpoint_dir
        path = os.path.join(path, f"epoch_{epoch}_step_{step}")
        os.makedirs(path, exist_ok=True)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Saving checkpoint to {path}", flush=True)

        # TODO(@adil-a): Change this when we create a LR scheduler class
        model, optimizer = None, None

        for key in self.__dict__["__state_tracked"]:
            if isinstance(getattr(self, key), nn.Module):
                model = getattr(self, key)
            elif isinstance(getattr(self, key), Optimizer):
                optimizer = getattr(self, key)
            else:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    torch.save(
                        getattr(self, key).state_dict(),
                        os.path.join(path, f"{key}.pt"),
                    )
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

        save_model(model, path, self.checkpoint_config, peft_config=self.peft_config)
        save_optimizer(optimizer, model, path)

    def load_checkpoint(self, restore_from: str | None = None):
        """
        Loads the latest checkpoint.
        """
        if not self.checkpoint_config.enabled:
            if (
                not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            ) and restore_from is not None:
                print("Enable checkpointing to resume from a checkpoint, skipping...", flush=True)
            return

        if restore_from:
            ckpt_dir = restore_from
        else:
            # Determine the latest checkpoint directory (e.g. ".../step_42").
            ckpt_dir = _find_latest_checkpoint(self.checkpoint_config.checkpoint_dir)
            if ckpt_dir is None:
                return

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Loading checkpoint from {ckpt_dir}", flush=True)

        # TODO(@adil-a): Change this when we create a LR scheduler class
        model, optimizer = None, None

        for key in self.__dict__["__state_tracked"]:
            if isinstance(getattr(self, key), nn.Module):
                model = getattr(self, key)
            elif isinstance(getattr(self, key), Optimizer):
                optimizer = getattr(self, key)
            else:
                getattr(self, key).load_state_dict(torch.load(os.path.join(ckpt_dir, f"{key}.pt"), weights_only=False))

        load_model(model, ckpt_dir, self.checkpoint_config)
        load_optimizer(optimizer, model, ckpt_dir)


def _find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the checkpoint directory and return it.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    # Accept checkpoints saved as either `step_<num>` or `epoch_<epoch>_step_<num>`
    # (or any other pattern that contains the substring `step_`).
    # This makes the checkpoint loading logic compatible with the naming scheme
    # used in `save_checkpoint`, which currently saves to `epoch_{epoch}_step_{step}`.
    checkpoint_files = list(checkpoint_dir.glob("*step_*"))
    if not checkpoint_files:
        return

    def _step_num(path: Path):
        """Return the numeric step from a path stem of the form step_<int>."""
        m = re.search(r"step_(\d+)$", path.stem)
        return int(m.group(1)) if m else -1

    latest = max(checkpoint_files, key=_step_num)

    # If no directory followed the expected "step_<int>" pattern, _step_num would be -1 for all of them.
    # Treat that as "no valid checkpoint".
    if _step_num(latest) == -1:
        return

    return latest
