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
from typing import Optional, Tuple

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
                 dataloader: Optional[int],
                 val_every_steps: Optional[int] = None,
                 start_step: int = 0,
                 start_epoch: int = 0,
                 num_epochs: int = 10,
                 max_steps: Optional[int] = None):
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
        self.dataloader        = dataloader
        self.step   = start_step
        self.epoch  = start_epoch
        self.num_epochs = num_epochs
        self.epoch_len = len(dataloader)
        self.grad_step = 0        # number of optimizer steps taken
        self.val_every_steps = val_every_steps
        self.max_steps = max_steps


    def __iter__(self):
        """Iterates over dataloader while keeping track of counters

        Raises:
            StopIteration: If the dataloader was exhausted or max_steps was reached.

        Yields:
            dict: batch
        """
        for batch in self.dataloader:
            self.step += 1
            if isinstance(self.max_steps, int) and self.step > self.max_steps:
                return
            yield batch

    @property
    def is_optim_step(self):
        """whether this step needs to call the optimizer step

        Returns:
            bool: if true, the optimizer should run.
        """
        is_grad = (self.step % self.grad_acc_steps) == 0
        self.grad_step += int(is_grad)
        return is_grad

    @property
    def is_val_step(self):
        """whether this step needs to call the validation
        """
        is_val = False
        if self.val_every_steps and self.val_every_steps > 0 and self.is_optim_step:
            is_val = (self.grad_step % self.val_every_steps) == 0
        return is_val

    @property
    def is_ckpt_step(self):
        """whether this step needs to call the checkpoint saving.

        Returns:
            bool: if true, the checkpoint should run.
        """
        batch_idx = self.step % self.epoch_len
        last_batch = self.epoch_len is not None and batch_idx == self.epoch_len - 1
        return ((self.step % self.ckpt_every_steps) == 0 and self.step != 0) or last_batch

    @property
    def epochs(self):
        """Epoch iterator

        Yields:
            iterator: over epochs
        """
        yield from range(self.epoch, self.num_epochs)

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
