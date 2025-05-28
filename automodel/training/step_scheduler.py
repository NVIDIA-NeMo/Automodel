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

from torch.distributed.checkpoint.stateful import Stateful
from typing import Any, Dict, Optional, Tuple

class StepScheduler(Stateful):
    """
    Scheduler for managing gradient accumulation and checkpointing steps.

    Attributes:
        grad_acc_steps (int): Steps to accumulate gradients.
        ckpt_every_steps (int): Interval steps for checkpointing.
        epoch_len (Optional[int]): Length of an epoch (number of batches).
        step (int): Global step counter.
        epoch (int): Current epoch counter.
        num_epochs (int): Total number of epochs.
    """
    def __init__(self,
                 grad_acc_steps: int,
                 ckpt_every_steps: int,
                 epoch_len: Optional[int],
                 start_step: int = 0,
                 start_epoch: int = 0,
                 num_epochs: int = 10):
        """
        Initialize the StepScheduler.

        Args:
            grad_acc_steps (int): Number of steps for gradient accumulation.
            ckpt_every_steps (int): Frequency of checkpoint steps.
            epoch_len (Optional[int]): Number of batches per epoch.
            start_step (int): Initial global step.
            start_epoch (int): Initial epoch.
            num_epochs (int): Total number of epochs.
        """
        self.grad_acc_steps   = grad_acc_steps
        self.ckpt_every_steps = ckpt_every_steps
        self.epoch_len        = epoch_len
        self.step   = start_step
        self.epoch  = start_epoch
        self.num_epochs = num_epochs

    def update(self, batch_idx: int) -> Tuple[bool, bool]:
        """
        Update the scheduler for the next batch.

        Args:
            batch_idx (int): Index of the current batch.

        Returns:
            Tuple[bool, bool]: A tuple of (is_optim_step, is_ckpt_step) indicating if a gradient
            step and/or checkpoint step should be performed.
        """
        self.step += 1
        return self.is_optim_step, self.is_ckpt_step(batch_idx)

    @property
    def is_optim_step(self):
        """whether this step needs to call the optimizer step

        Returns:
            bool: if true, the optimizer should run.
        """
        return (self.step % self.grad_acc_steps) == 0

    def is_ckpt_step(self, batch_idx):
        """whether this step needs to call the checkpoint saving.

        Returns:
            bool: if true, the checkpoint should run.
        """
        last_batch = self.epoch_len is not None and batch_idx == self.epoch_len - 1
        return (self.step % self.ckpt_every_steps) == 0 or last_batch

    # (optional) persistence
    def state_dict(self):
        """
        Get the current state of the scheduler.

        Returns:
            dict: Current state with 'step' and 'epoch' keys.
        """
        return {"step": self.step, "epoch": self.epoch}

    def load_state_dict(self, s):
        """
        Load the scheduler state from a dictionary.

        Args:
            s (dict): Dictionary containing 'step' and 'epoch'.
        """
        self.step, self.epoch = s["step"], s["epoch"]
