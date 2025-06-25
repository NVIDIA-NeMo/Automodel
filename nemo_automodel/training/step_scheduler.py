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

from typing import Iterable, Iterator, Any

from torch.distributed.checkpoint.stateful import Stateful


class StepScheduler(Stateful):
    """
    Scheduler for managing gradient accumulation and checkpointing steps, it tracks:
      - batch iterations        (internal: batch_step)
      - optimizer steps         (public : step)
      - validation / checkpoint intervals (optimizer-step domain)
    """
    def __init__(
        self,
        *,
        dataloader: Iterable[Any] | None,
        grad_acc_steps: int,
        ckpt_every_steps: int,
        num_epochs: int = 10,
        val_every_steps: int | None = None,         # optimizer-step cadence
        start_step: int = 0,                        # starting optimizer step
        start_epoch: int = 0,
        max_steps: int | None = None,               # optimizer-step budget
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
            max_steps (int): Total number of steps to run.
        """
        if grad_acc_steps < 1:
            raise ValueError("grad_acc_steps must be ≥ 1")

        # external handles
        self.dataloader = dataloader

        # cadence config
        self.grad_acc_steps   = grad_acc_steps
        self.ckpt_every_steps = ckpt_every_steps
        self.val_every_steps  = val_every_steps

        # counters
        self.step: int        = start_step     # <- OPTIMIZER steps
        self.batch_step: int  = start_step * grad_acc_steps  # raw batches so far
        self.epoch: int       = start_epoch
        self.num_epochs: int  = num_epochs
        self.max_steps: int | None = max_steps

        # len() may be missing for IterableDataset
        self.epoch_len: int | None = (
            len(dataloader) if dataloader is not None and hasattr(dataloader, "__len__") else None
        )

        # per-batch cached flags
        self._optim_flag = False
        self._val_flag   = False
        self._ckpt_flag  = False

    def __iter__(self) -> Iterator[Any]:
        """
        Iterates over dataloader while keeping track of counters.

        Raises:
            StopIteration: If the dataloader was exhausted or max_steps was reached.

        Yields:
            dict: batch
        """
        if self.dataloader is None:
            return

        for self.epoch in range(self.epoch, self.num_epochs):
            self._set_epoch_for_sampler(self.epoch)

            for batch in self.dataloader:
                self.batch_step += 1
                self._update_flags()

                yield batch

                if self.max_steps is not None and self.step >= self.max_steps:
                    return

    def set_epoch(self, epoch: int):
        """
        Set the epoch for the dataloader.
        """
        self.epoch = epoch
        self._set_epoch_for_sampler(epoch)

    @property
    def is_optim_step(self) -> bool:
        """
        Returns whether this step needs to call the optimizer step.

        Returns:
            bool: if true, the optimizer should run.
        """
        return self._optim_flag

    @property
    def is_val_step(self) -> bool:
        """
        Returns whether this step needs to call the validation.
        """
        return self._val_flag

    @property
    def is_ckpt_step(self) -> bool:
        """
        Returns whether this step needs to call the checkpoint saving.

        Returns:
            bool: if true, the checkpoint should run.
        """
        return self._ckpt_flag

    def _set_epoch_for_sampler(self, epoch) -> None:
        """
        helper to set epoch to sampler
        """
        sampler = getattr(self.dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            try:
                sampler.set_epoch(epoch)
            except Exception:          # best-effort
                pass

    def _update_flags(self) -> None:
        """
        Compute flags once per *batch*.
        step is incremented only when an optimizer step occurs.
        """
        # optimizer step
        self._optim_flag = (self.batch_step % self.grad_acc_steps) == 0
        if self._optim_flag:
            self.step += 1

        # validation flag
        self._val_flag = (
            self.val_every_steps is not None
            and self.val_every_steps > 0
            and self._optim_flag
            and (self.step % self.val_every_steps) == 0
        )

        # checkpoint flag
        last_batch_of_epoch = (
            self.epoch_len is not None
            and (self.batch_step % self.epoch_len) == (self.epoch_len - 1)
        )
        by_cadence = (
            self.ckpt_every_steps > 0 and self._optim_flag and (self.step % self.ckpt_every_steps) == 0
        )
        self._ckpt_flag = by_cadence or last_batch_of_epoch

    def state_dict(self) -> dict[str, int]:
        """
        Get the current state of the scheduler.

        Returns:
            dict: Current state with 'step' and 'epoch' keys.
        """
        return {
            "step": self.step,
            "batch_step": self.batch_step,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        """
        Load the scheduler state from a dictionary.

        Args:
            state (dict): Dictionary containing 'step' and 'epoch'.
        """
        self.step       = int(state["step"])
        self.batch_step = int(state.get("batch_step", self.step * self.grad_acc_steps))
        self.epoch      = int(state["epoch"])

    def __repr__(self) -> str:  # pragma: no cover
        """
        debug friendly message

        Returns:
            str: message with step/batch-step/epoch
        """
        return (
            f"{self.__class__.__name__}(step={self.step}, batch_step={self.batch_step}, "
            f"epoch={self.epoch})"
        )
