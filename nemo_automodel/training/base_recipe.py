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
from torch.distributed.checkpoint.stateful import Stateful
import os

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

class BaseRecipe(Stateful):
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
        if isinstance(value, (nn.Module, Optimizer)) or has_load_restore_state(value):
            assert not key in self.__dict__['__state_tracked']
            self.__dict__['__state_tracked'].add(key)
        super().__setattr__(key, value)


    def _save_checkpoint(self):
        """
        Save the current training state as a checkpoint.

        Currently iterates over state-tracked attributes and saves their state_dict.
        """
        path = self.cfg.get("ckpt_path", "latest/")

        # Create the checkpoint directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        for key in self.__dict__['__state_tracked']:
            torch.save(getattr(self, key).state_dict(),
                os.path.join(path, key),
            )
        print(f"[ckpt] saved to {path}", flush=True)