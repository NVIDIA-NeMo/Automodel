from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from automodel.config.loader import load_yaml_config
from automodel.training.init_utils import initialize_distributed
from automodel.base_recipe import BaseRecipe


# ---------------------------
#  Stateless helper functions
# ---------------------------

def build_model(device, model_wrapper, cfg_model) -> nn.Module:
    """
    Build and initialize a model.

    Args:
        device: The target device.
        model_wrapper: A potential wrapper providing parallelism.
        cfg_model: Configuration for model instantiation.

    Returns:
        The instantiated model on the specified device.
    """
    model = cfg_model.instantiate()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.weight.requires_grad_(False)
    if model_wrapper is not None and callable(getattr(model_wrapper, 'parallelize', None)):
        model = model_wrapper.parallelize(model)
    return model.to(device)

def build_optimizer(device, cfg_opt, model) -> 'Optimizer':  # noqa: F821
    """
    Build an optimizer for the model.

    Args:
        device: The target device.
        cfg_opt: Configuration for optimizer instantiation.
        model: The model whose parameters will be optimized.

    Returns:
        The instantiated optimizer.
    """
    return cfg_opt.instantiate(params=model.parameters())

def build_loss_fn(device, cfg_loss):
    """
    Build a loss function.

    Args:
        device: The target device.
        cfg_loss: Loss function configuration or a callable loss function.

    Returns:
        The instantiated loss function on the specified device.
    """
    if callable(cfg_loss):
        return cfg_loss
    else:
        return cfg_loss.instantiate().to(device)

def build_dataloader(device, cfg_ds, cfg_dl) -> DataLoader:
    """
    Build a DataLoader for the dataset.

    Args:
        device: The target device.
        cfg_ds: Dataset configuration.
        cfg_dl: DataLoader configuration.

    Returns:
        The instantiated DataLoader.
    """
    ds = cfg_ds.instantiate()
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    return cfg_dl.instantiate(dataset=ds, sampler=sampler)

def build_distributed(cfg_dist: Dict[str, Any]) -> 'DistInfo':  # noqa: F821
    """
    Build and initialize distributed training resources.

    Args:
        cfg_dist: Configuration dictionary for distributed training.

    Returns:
        Distributed training information from initialize_distributed.
    """
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)

class StepScheduler:
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
                 num_epochs: int = 0):
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
            Tuple[bool, bool]: A tuple of (is_grad_step, is_ckpt_step) indicating if a gradient
            step and/or checkpoint step should be performed.
        """
        self.step += 1
        is_grad = (self.step % self.grad_acc_steps) == 0
        last_batch = self.epoch_len is not None and batch_idx == self.epoch_len - 1
        is_ckpt = (self.step % self.ckpt_every_steps) == 0 or last_batch
        return is_grad, is_ckpt

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


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------

class FinetuneRecipeForNextTokenPrediction(BaseRecipe):
    """
    Recipe for fine-tuning a model for next-token prediction.

    This class orchestrates training, from setup to main training loop.
    """
    def __init__(self, cfg):
        """
        Initialize the recipe with configuration.

        Args:
            cfg: Configuration dictionary/object for training.
        """
        self.cfg = cfg

    # ------------------ build phase ------------------
    def setup(self):
        """ Builds all components needed for training/validation/logging/checkpointing/etc.

        This is the last place where self.cfg should be referenced.

        Raises:
            NotImplemented: Raises if it tries to restore a checkpoint; will be removed.
        """
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        model_wrapper = None
        if 'distributed' in self.cfg:
            model_wrapper = self.cfg.distributed.instantiate(world_size=self.dist_env.world_size)
            print(model_wrapper)
        torch.manual_seed(self.cfg.get("seed", 42) + self.dist_env.rank)

        # Build components
        self.model = build_model(self.dist_env.device, model_wrapper, self.cfg.model)
        self.optimizer = build_optimizer(self.dist_env.device, self.cfg.optimizer, self.model)
        self.loss_fn   = build_loss_fn(self.dist_env.device, self.cfg.loss_fn)
        self.dataloader = build_dataloader(self.dist_env.device, self.cfg.dataset, self.cfg.dataloader)

        # Scheduler
        self.scheduler = StepScheduler(
            num_epochs = self.cfg.get("epochs", 10),
            grad_acc_steps   = self.cfg.get("grad_acc_steps", 10),
            ckpt_every_steps = self.cfg.get("ckpt_every_steps", 100),
            epoch_len        = len(self.dataloader),
        )

        # Optionally resume
        if (path := self.cfg.get("restore_from")) is not None:
            raise NotImplemented("TODO resume from {}".format(path))

    # ------------------ main loop ------------------
    def run_train_validation_loop(self):
        """
        Run the training loop over all epochs and batches.

        For each batch, perform a forward pass, compute loss, backpropagate,
        and update model parameters when necessary. Also prints loss every gradient step.
        """
        self.model.train()
        for self.scheduler.epoch in range(self.scheduler.epoch, self.scheduler.num_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                is_grad, is_ckpt = self.scheduler.update(batch_idx)
                loss = self._run_train_step(batch, is_grad)
                # if self.dist_env.is_main and is_ckpt:
                #     self._save_checkpoint()
                if self.dist_env.is_main and is_grad:
                    print(f"step {self.scheduler.step} | loss {loss.item():.6f}", flush=True)


    # ------------------ helpers ------------------
    def _run_train_step(self, batch, is_grad):
        """
        Execute a single training step.

        Args:
            batch: Batch of training data.
            is_grad: Flag indicating if a gradient step should be applied.

        Returns:
            Detached loss from the training step.
        """
        batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        mask   = batch.pop("loss_mask", None)

        out  = self.model(**batch)
        loss = self.loss_fn(out.logits.view(-1, out.logits.size(-1)),
                            labels.view(-1), mask=mask)
        loss.backward()

        if is_grad:
            #         if self.cfg.training.get("calculate_per_token_loss", False):
            # world_size = get_world_size_safe()
            # num_tokens_for_grad_scaling = self.total_num_tokens.clone().detach()
            # dist.all_reduce(num_tokens_for_grad_scaling)
                # DDP reduces across ranks, so we need to scale by the world size to inverse it
            scaling_factor = 109 #world_size / num_tokens_for_grad_scaling
            # print('scaling_factor= ' + str(scaling_factor))
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(scaling_factor)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, foreach=True)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.detach()

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

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the trainer, and initiates the training loop.
    """
    cfg = load_yaml_config("llama_3_2_1b_hellaswag.yaml")
    trainer = FinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

if __name__ == "__main__":
    main()
