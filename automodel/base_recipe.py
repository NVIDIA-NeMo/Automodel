import torch.nn as nn
from torch.optim import Optimizer
from typing import Any, Dict, Optional, Tuple
from torch.distributed.checkpoint.stateful import Stateful

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
        if isinstance(value, (nn.Module, Optimizer)) or has_load_restore_state(value):
            assert not key in self.__dict__['__state_tracked']
            self.__dict__['__state_tracked'].add(key)
        super().__setattr__(key, value)


    def _save_checkpoint(self):
        """
        Save the current training state as a checkpoint.

        Currently iterates over state-tracked attributes and saves their state_dict.
        """
        path = self.cfg.get("ckpt_path", "latest.pt")
        for key in self.__dict__['__state_tracked']:
            torch.save(getattr(self, key).state_dict(),
                path + "_key"
            )
        print(f"[ckpt] saved to {path}", flush=True)

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
            Tuple[bool, bool]: A tuple of (is_optim_step_step, is_ckpt_step) indicating if a gradient
            step and/or checkpoint step should be performed.
        """
        self.step += 1
        is_optim_step = (self.step % self.grad_acc_steps) == 0
        last_batch = self.epoch_len is not None and batch_idx == self.epoch_len - 1
        is_ckpt = (self.step % self.ckpt_every_steps) == 0 or last_batch
        return is_optim_step, is_ckpt

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
