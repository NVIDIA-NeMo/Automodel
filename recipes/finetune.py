from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from megatron.core.distributed.custom_fsdp import FSDP


from nemo_automodel.config.loader import load_yaml_config
from nemo_automodel.distributed.init_utils import initialize_distributed
from nemo_automodel.training.base_recipe import BaseRecipe
from nemo_automodel.training.step_scheduler import StepScheduler


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

        # FSDP2 and nvFSDP should already be on the correct device
        return model
    else:
        return model.to(device)

def build_optimizer(device, cfg_opt, model, tp_size) -> 'Optimizer':  # noqa: F821
    """
    Build an optimizer for the model.

    Args:
        device: The target device.
        cfg_opt: Configuration for optimizer instantiation.
        model: The model whose parameters will be optimized.

    Returns:
        The instantiated optimizer.
    """
    trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
    assert len(trainable_params) > 0, "trainable_params cannot be empty"
    if tp_size > 1:
        # TP does not support foreach
        cfg_opt.foreach = False
    return cfg_opt.instantiate(params=trainable_params)

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


def build_dataloader(device, cfg_ds, cfg_dl, distributed_sampler_kwargs) -> DataLoader:
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
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds,
        num_replicas=distributed_sampler_kwargs["num_replicas"],
        rank=distributed_sampler_kwargs["rank"],
    )
    return cfg_dl.instantiate(dataset=ds, sampler=sampler)


def build_distributed(cfg_dist: Dict[str, Any]) -> 'DistInfo':  # noqa: F821
    """
    Build and initialize distributed training resources.

    Args:
        cfg_dist: Configuration for distributed training.

    Returns:
        Distributed training information from initialize_distributed.
    """
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)

def build_step_scheduler(cfg, dataloader):
    """
    Build the step scheduler.

    Args:
        cfg: configuration for the StepScheduler class.
        dataloader: the training dataloader, used for extracting the epoch_len (in batches).

    Returns:
        StepScheduler: the configured StepScheduler.
    """
    assert not '_target_' in cfg, "_target_ not permitted in step scheduler"
    default_kwargs = dict(
        num_epochs = 10,
        grad_acc_steps = 10,
        ckpt_every_steps = 100,
        epoch_len = len(dataloader),
    )
    if cfg is not None:
        default_kwargs |= cfg.to_dict()
    return StepScheduler(**default_kwargs)


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

        self.model_wrapper = None
        distributed_sampler_kwargs = {}
        if "distributed" in self.cfg:
            self.model_wrapper = self.cfg.distributed.instantiate(
                world_size=self.dist_env.world_size
            )
            distributed_sampler_kwargs = {
                "num_replicas": self.model_wrapper.device_mesh["data_parallel"].size(),
                "rank": self.model_wrapper.device_mesh["data_parallel"].get_local_rank(),
            }

        torch.manual_seed(self.cfg.get("seed", 42) + self.dist_env.rank)

        # Build components
        self.model = build_model(self.dist_env.device, self.model_wrapper, self.cfg.model)
        self.optimizer = build_optimizer(
            self.dist_env.device, 
            self.cfg.optimizer, 
            self.model, 
            self.cfg.get("distributed.tp_size", 1),
        )
        self.loss_fn   = build_loss_fn(self.dist_env.device, self.cfg.loss_fn)
        self.dataloader = build_dataloader(
            self.dist_env.device,
            self.cfg.dataset,
            self.cfg.dataloader,
            distributed_sampler_kwargs,
        )

        # Scheduler
        self.step_scheduler = build_step_scheduler(self.cfg.get('step_scheduler', None), self.dataloader)

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
        for epoch in self.step_scheduler.epochs:
            for batch_idx, batch in enumerate(self.dataloader):
                is_optim_step, is_ckpt_step = self.step_scheduler.update(batch_idx)
                loss = self._run_train_step(batch, is_optim_step, 1.0,
                    num_grad_acc_steps=self.step_scheduler.grad_acc_steps)

                if self.dist_env.is_main and is_ckpt_step:
                    self._save_checkpoint()

                if self.dist_env.is_main and is_optim_step:
                    print(f"step {self.step_scheduler.step} | loss {loss.item():.6f}", flush=True)


    # ------------------ helpers ------------------
    def _run_train_step(self, batch, is_optim_step, clip_norm=1.0, num_grad_acc_steps=10):
        """
        Execute a single training step.

        Args:
            batch: Batch of training data.
            is_optim_step: Flag indicating if a gradient step should be applied.

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

        if is_optim_step:
            grad_params = []
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(1/num_grad_acc_steps)
                    grad_params.append(param)
            # TP does not support grad_clip yet
            if (
                isinstance(clip_norm, float)
                and self.cfg.get("distributed.tp_size", 1) == 1
            ):
                torch.nn.utils.clip_grad_norm_(grad_params, clip_norm, foreach=True)

            if isinstance(self.model, FSDP):
                # If the model uses nvFSDP, wait for all sharded gradients to be reduced and unsharded.
                # Necessary because the post-backward reduce-scatter is asynchronous, so gradients and backward
                # computations are concurrent, but the gradients of the final layer may not be available yet.
                self.model.finish_grad_sync()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if isinstance(self.model, FSDP) and self.model.ddp_config.preserve_fp32_weights:
                # If custom FSDP2 is configured with "optim" (optimizer state / high-precision model weight sharding),
                # then the optimizer step will be applied to the main high-precision model weights. Update the model
                # weights after the optimizer step.
                self.model.param_and_grad_buffer.copy_main_weights_to_model_weights()

        return loss.detach()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the trainer, and initiates the training loop.
    """
    cfg = load_yaml_config("llama_3_2_1b_hellaswag_nvfsdp.yaml")
    trainer = FinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

if __name__ == "__main__":
    main()
