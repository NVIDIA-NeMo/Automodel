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

from typing import Optional

from torch.distributed.checkpoint.stateful import Stateful


class StepScheduler(Stateful):
    """
    Scheduler for managing gradient accumulation and checkpointing steps.
    """

    def __init__(
        self,
        grad_acc_steps: int,
        ckpt_every_steps: int,
        dataloader: Optional[int],
        val_every_steps: Optional[int] = None,
        start_step: int = 0,
        start_epoch: int = 0,
        num_epochs: int = 10,
        max_steps: int = 9223372036854775807,
    ):
        """
        Initialize the StepScheduler.

        Args:
            grad_acc_steps (int): Number of steps for gradient accumulation.
            ckpt_every_steps (int): Frequency of checkpoint steps.
            dataloader (Optional[int]): The training dataloader.
            val_every_steps (int): Number of training steps between validation.
            start_step (int): Initial global step.
            start_epoch (int): Initial epoch.
            num_epochs (int): Total number of epochs.
            max_steps (int): Total number of steps to run. Default is 2^63-1.
        """
        self.grad_acc_steps = grad_acc_steps
        assert grad_acc_steps > 0, "grad_acc_steps must be greater than 0"
        self.ckpt_every_steps = ckpt_every_steps
        assert ckpt_every_steps > 0, "ckpt_every_steps must be greater than 0"
        self.dataloader = dataloader
        self.step = start_step
        assert start_step >= 0, "start_step must be greater than or equal to 0"
        self.epoch = start_epoch
        assert start_epoch >= 0, "start_epoch must be greater than or equal to 0"
        self.num_epochs = num_epochs
        assert num_epochs > 0, "num_epochs must be greater than 0"
        self.epoch_len = len(dataloader)
        self.val_every_steps = val_every_steps
        assert val_every_steps is None or val_every_steps > 0, "val_every_steps must be greater than 0 if not None"
        self.max_steps = max_steps
        assert max_steps > 0, "max_steps must be greater than 0"

    def __iter__(self):
        """
        Iterates over dataloader while keeping track of counters.

        Raises:
            StopIteration: If the dataloader was exhausted or max_steps was reached.

        Yields:
            dict: batch
        """
        if self.step >= self.max_steps:
            return
        batch_buffer = []
        for batch in self.dataloader:
            batch_buffer.append(batch)
            if len(batch_buffer) == self.grad_acc_steps:
                self.step += 1
                yield batch_buffer
                batch_buffer = []
                if self.step >= self.max_steps:
                    return
        if batch_buffer:
            self.step += 1
            yield batch_buffer
        self.epoch += 1

    def set_epoch(self, epoch: int):
        """
        Set the epoch for the sampler.
        """
        self.epoch = epoch
        if hasattr(getattr(self.dataloader, "sampler", None), "set_epoch"):
            self.dataloader.sampler.set_epoch(epoch)

    @property
    def is_val_step(self):
        """
        Returns whether this step needs to call the validation.
        """
        is_val = False
        if self.val_every_steps and self.val_every_steps > 0:
            is_val = (self.step % self.val_every_steps) == 0
        return is_val

    @property
    def is_ckpt_step(self):
        """
        Returns whether this step needs to call the checkpoint saving.

        Returns:
            bool: if true, the checkpoint should run.
        """
        batch_idx = self.step % self.epoch_len
        last_batch = self.epoch_len is not None and batch_idx == self.epoch_len - 1
        finished = self.step >= self.max_steps
        return ((self.step % self.ckpt_every_steps) == 0 and self.step != 0) or last_batch or finished

    @property
    def epochs(self):
        """
        Epoch iterator.

        Yields:
            iterator: over epochs
        """
        epoch = self.epoch
        for e in range(epoch, self.num_epochs):
            if self.step >= self.max_steps:
                return
            yield e

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
