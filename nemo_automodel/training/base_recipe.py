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

import torch
import torch.nn as nn
from torch.optim import Optimizer
import os
from nemo_automodel.checkpoint.checkpointing import save_model, save_optimizer

def has_load_restore_state(object):
    """ Checks whether object has load_state_dict and state_dict functions, ie whether the object
    follows the nn.Module API.

    TODO: also need to check function signatures.

    Args:
        object (any): the object to check.

    Returns:
        bool: returns True if has callable load_state_dict and state_dict
    """
    return all(
        callable(getattr(object, attr, None))
        for attr in ('load_state_dict', 'state_dict')
    )

class BaseRecipe:
    """
    Checkpoint registry
    """
    def __setattr__(self, key, value):
        """ Overriden __setattr__ to keep track of stateful classes.

        Args:
            key (str): attribute named.
            value (Any): Value assigned

        Raises:
            ValueError: if __state_tracked is attemped to be overwriten.

        """
        # assuming no one will do recipe.__dict__['__state_tracked'] = None
        if key == '__state_tracked':
            raise ValueError("cannot set __state_tracked")
        if not '__state_tracked' in self.__dict__:
            self.__dict__['__state_tracked'] = set()
        # Track stateful objects unless they are validation/eval components.
        should_track = (
            isinstance(value, (nn.Module, Optimizer)) or has_load_restore_state(value)
        )

        if should_track and not any(substr in key.lower() for substr in ("val", "eval", "test")):
            assert key not in self.__dict__['__state_tracked']
            self.__dict__['__state_tracked'].add(key)
        super().__setattr__(key, value)

    def save_checkpoint(self):
        """
        Save the current training state as a checkpoint.

        As long as the object has a 'load_state_dict' and 'state_dict' function, it will be saved.
        """
        if not self.checkpoint_config.enabled:
            return

        path = self.checkpoint_config.checkpoint_dir
        path = os.path.join(path, f"step_{self.step_scheduler.step}")
        os.makedirs(path, exist_ok=True)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Saving checkpoint to {path}", flush=True)

        # TODO(@adil-a): Change this when we create a LR scheduler class
        model, optimizer = None, None

        for key in self.__dict__['__state_tracked']:
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

        save_model(model, path, self.checkpoint_config)
        save_optimizer(optimizer, model, path)