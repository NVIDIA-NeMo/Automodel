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

import logging
from math import ceil
from typing import Optional

from torch.distributed.checkpoint.stateful import Stateful

from nemo_automodel.components.training.signal_handler import DistributedSignalHandler

logger = logging.getLogger(__name__)


def _calculate_max_steps(
    num_epochs: int, epoch_len: Optional[int], default_max_steps: int = 9223372036854775807
) -> int:
    """
    Calculate the maximum number of steps.
    """
    if epoch_len is None:
        return default_max_steps
    return num_epochs * epoch_len


class StepScheduler(Stateful):
    """
    Scheduler for managing gradient accumulation and checkpointing steps.
    """

    def __init__(
        self,
        global_batch_size: int,
        local_batch_size: int,
        dp_size: int,
        dataloader: Optional[int],
        ckpt_every_steps: Optional[int] = None,
        val_every_steps: Optional[int] = None,
        start_step: int = 0,
        start_epoch: int = 0,
        num_epochs: int = 10,
        max_steps: int = None,
    ):
        """
        Initialize the StepScheduler.

        Args:
            global_batch_size (int): Total number of samples processed per optimizer step across all GPUs. This is the effective batch size for the entire training step.
            local_batch_size (int): Number of samples per micro-batch per GPU. This is the batch size for a single forward/backward pass on one GPU.
            dp_size (int): Number of GPUs for data parallelism.
            dataloader: The training dataloader.
            ckpt_every_steps (Optional[int]): Frequency of checkpoint steps.
            val_every_steps (Optional[int]): Number of training steps between validation.
            start_step (int): Initial global step. Used when resuming from checkpoint. Default: 0.
            start_epoch (int): Initial epoch. Used when resuming from checkpoint. Default: 0.
            num_epochs (int): Total number of epochs. Default: 10.
            max_steps (Optional[int]): Maximum number of steps to run. If None, calculated from num_epochs.
        """
        assert global_batch_size % (local_batch_size * dp_size) == 0, (
            f"global_batch_size ({global_batch_size}) must be divisible by local_batch_size * dp_size ({local_batch_size} * {dp_size})"
        )
        self.grad_acc_steps = global_batch_size // (local_batch_size * dp_size)
        assert self.grad_acc_steps >= 1, (
            f"grad_acc_steps ({self.grad_acc_steps}) must be greater than or equal to 1. Please ensure that global_batch_size >= (local_batch_size * dp_size)"
        )
        self.dataloader = dataloader
        self.step = start_step
        assert start_step >= 0, "start_step must be greater than or equal to 0"
        self.epoch = start_epoch
        assert start_epoch >= 0, "start_epoch must be greater than or equal to 0"
        self.num_epochs = num_epochs
        assert num_epochs > 0, "num_epochs must be greater than 0"
        # Throws with IterableDataset.
        try:
            self.epoch_len = ceil(len(dataloader) / self.grad_acc_steps)
        except:
            self.epoch_len = None
        self.val_every_steps = val_every_steps
        assert val_every_steps is None or val_every_steps > 0, "val_every_steps must be greater than 0 if not None"
        if max_steps is None:
            assert self.epoch_len is not None, "epoch_len must be provided if max_steps is not provided"
            max_steps = _calculate_max_steps(self.num_epochs, self.epoch_len)
            logger.info("max_steps not provided; will run for up to {} steps".format(max_steps))
        self.max_steps = max_steps
        assert max_steps > 0, "max_steps must be greater than 0"

        if ckpt_every_steps is None:
            if self.epoch_len is None:
                ckpt_every_steps = self.max_steps // 2
            else:
                ckpt_every_steps = self.epoch_len
            logger.info("ckpt_every_steps not provided; will save checkpoint every {} steps".format(ckpt_every_steps))
        self.ckpt_every_steps = ckpt_every_steps

        self.sig_handler = DistributedSignalHandler().__enter__()
        self.sigterm_flag = False

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
                yield batch_buffer
                self.step += 1
                batch_buffer = []
                if self.step >= self.max_steps or self.sigterm_flag:
                    return
        if batch_buffer:
            yield batch_buffer
            self.step += 1
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
            is_val = self.step % self.val_every_steps == self.val_every_steps - 1
        return (is_val or self.is_ckpt_step) and not self.sigterm_flag

    @property
    def is_ckpt_step(self):
        """
        Returns whether this step needs to call the checkpoint saving.

        Returns:
            bool: if true, the checkpoint should run.
        """
        is_ckpt_step = (self.step % self.ckpt_every_steps) == self.ckpt_every_steps - 1
        return is_ckpt_step or self.is_last_batch or self.is_last_step or self.sigterm_received

    @property
    def is_last_step(self):
        """
        Returns whether the training is finished.
        """
        # we +1 here because the step is incremented after
        # the batch is yielded in the tail handling of __iter__
        return self.step + 1 >= self.max_steps

    @property
    def is_last_batch(self):
        """
        Returns whether this is the last batch for this epoch.
        """
        if self.epoch_len is None:
            return False
        return (self.step % self.epoch_len) == self.epoch_len - 1

    @property
    def sigterm_received(self):
        """
        Returns whether SIGTERM was received.
        """
        self.sigterm_flag = self.sigterm_flag or any(self.sig_handler.signals_received())
        return self.sigterm_flag

    @property
    def epochs(self):
        """
        Epoch iterator.

        Yields:
            iterator: over epochs
        """
        epoch = self.epoch
        for e in range(epoch, self.num_epochs):
            if self.step >= self.max_steps or self.sigterm_received:
                return
            yield e

    def state_dict(self):
        """
        Get the current state of the scheduler.

        Returns:
            dict: Current state with 'step' and 'epoch' keys.
        """
        # At checkpoint time, we need to save step + 1 because we yield before incrementing the step
        # and the checkpointing happens after the yield but before the increment.
        # Added min(self.max_steps, self.step + 1) to ensure that the step is not greater than max_steps.
        # for example, if state_dict is called outside the for loop that increments step scheduler.
        return {"step": min(self.max_steps, self.step + 1), "epoch": self.epoch}

    def load_state_dict(self, s):
        """
        Load the scheduler state from a dictionary.

        Args:
            s (dict): Dictionary containing 'step' and 'epoch'.
        """
        self.step, self.epoch = s["step"], s["epoch"]
