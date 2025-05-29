from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import torch.distributed as dist

from nemo_automodel.config.loader import load_yaml_config
from nemo_automodel.distributed.init_utils import initialize_distributed, get_world_size_safe
from nemo_automodel.utils.dist_utils import reduce_loss
from nemo_automodel.training.base_recipe import BaseRecipe
from nemo_automodel.training.step_scheduler import StepScheduler
import contextlib


# ---------------------------
#  Stateless helper functions
# ---------------------------

def get_sync_ctx(model, is_optim_step):
    """
    Get the synchronization context for the model.

    Args:
        model: The model to synchronize.
        is_optim_step: Whether the current step is an optimizer step.

    Returns:
        A context manager that synchronizes the model.
    """
    # Use `no_sync` on DDP models when we are *not* on the final micro-batch for
    # this gradient update (i.e., when `is_grad` is False). This avoids an
    # all-reduce for every micro-batch and greatly improves throughput.
    if isinstance(model, dist.fsdp._fully_shard._fully_shard.FSDPModule):
        model.set_requires_gradient_sync(is_optim_step)
        sync_ctx = contextlib.nullcontext()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel) and not is_optim_step:
        sync_ctx = model.no_sync()
    else:
        sync_ctx = contextlib.nullcontext()
    return sync_ctx

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
    trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
    assert len(trainable_params) > 0, "trainable_params cannot be empty"
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

def build_dataloader(cfg_ds, cfg_dl) -> DataLoader:
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
    # Handle optional sampler
    sampler_cfg = None
    sampler_obj = None
    if hasattr(cfg_dl, "sampler"):
        sampler_cfg = cfg_dl.sampler
        # Remove it from kwargs so that ``instantiate`` doesn't try to build it
        # on its own (which fails because ``dataset`` would be missing).
        del cfg_dl.__dict__["sampler"]

        # Instantiate the sampler with the actual dataset.
        if sampler_cfg is not None:
            sampler_obj = sampler_cfg.instantiate(dataset=ds)

    return cfg_dl.instantiate(dataset=ds, sampler=sampler_obj)

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
        model_wrapper = None
        if 'distributed' in self.cfg:
            model_wrapper = self.cfg.distributed.instantiate(world_size=self.dist_env.world_size)
            if self.dist_env.is_main:
                print(model_wrapper)
        torch.manual_seed(self.cfg.get("seed", 42) + self.dist_env.rank)

        if self.dist_env.is_main and hasattr(self.cfg, 'logger'):
            wandb.init(
                project=self.cfg.logger.get("wandb_project", "default_project"),
                entity=self.cfg.logger.get("wandb_entity"),
                name=self.cfg.logger.get("wandb_exp_name"),
                dir=self.cfg.logger.get("wandb_save_dir"),
                config=self.cfg,
            )

        # Build components
        self.model = build_model(self.dist_env.device, model_wrapper, self.cfg.model)
        self.optimizer = build_optimizer(self.dist_env.device, self.cfg.optimizer, self.model)
        self.loss_fn   = build_loss_fn(self.dist_env.device, self.cfg.loss_fn)
        self.dataloader = build_dataloader(self.cfg.dataset, self.cfg.dataloader)

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        val_ds_cfg = self.cfg.get("validation_dataset")
        val_dl_cfg = self.cfg.get("validation_dataloader")
        if val_ds_cfg is not None and val_dl_cfg is not None:
            self.val_dataloader = build_dataloader(val_ds_cfg, val_dl_cfg)
        
        # Initialize metrics required for calculating loss
        self.total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        self.forward_data_store = []

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
        starting_epoch = self.step_scheduler.epoch
        num_epochs = self.step_scheduler.num_epochs
        for self.step_scheduler.epoch in range(starting_epoch, num_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                is_optim_step, is_ckpt_step, is_val_step = self.step_scheduler.update(batch_idx)
                grad_norm = self._run_train_step(batch, is_optim_step, 1.0)
                if is_optim_step:
                    reporting_loss = self.log_train_metrics(grad_norm)

                # if self.dist_env.is_main and is_ckpt_step:
                #     self._save_checkpoint()
                if self.dist_env.is_main and is_optim_step:
                    print(
                        f"step {self.step_scheduler.step} | epoch {self.step_scheduler.epoch} | loss {reporting_loss:.6f} | grad_norm {grad_norm:.6f}"
                    )
                
                if is_val_step and self.val_dataloader is not None:
                    val_loss = self._run_validation_epoch()
                    if self.dist_env.is_main:
                        if wandb.run is not None:
                            wandb.log({"val_loss": val_loss, "step": self.step_scheduler.step, "epoch": self.step_scheduler.epoch})
                        print(
                            f"[val] step {self.step_scheduler.step} | epoch {self.step_scheduler.epoch} | loss {val_loss:.4f}",
                        )


    # ------------------ helpers ------------------
    def _run_train_step(self, batch, is_optim_step, clip_norm=1.0):
        """
        Execute a single training step.

        Args:
            batch: Batch of training data.
            is_optim_step: Flag indicating if a gradient step should be applied.

        Returns:
            Grad norm from the training step.
        """
        batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        mask   = batch.pop("loss_mask", None)
        if mask is None:
            mask = (labels.detach() != -100).to(torch.int)

        out  = self.model(**batch)
        local_loss = self.loss_fn(out.logits.view(-1, out.logits.size(-1)),
                            labels.view(-1), mask=mask, reduction="sum")
        local_num_tokens = mask.sum().detach().to(torch.int)
        self.total_num_tokens += local_num_tokens
        self.forward_data_store.append(local_loss.detach())

        with get_sync_ctx(self.model, is_optim_step):
            local_loss.backward()

        grad_norm = None
        if is_optim_step:
            world_size = get_world_size_safe()
            num_tokens_for_grad_scaling = self.total_num_tokens.clone().detach()
            dist.all_reduce(num_tokens_for_grad_scaling)
            # DDP/FSDP reduces gradients across ranks, so we need to scale by the world size to inverse it
            scaling_factor = world_size / num_tokens_for_grad_scaling
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(scaling_factor)
            
            # Clip gradients **after** any optional rescaling.
            with torch.no_grad():
                grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                grad_norm = torch.nn.utils.get_total_norm(grads)
                if isinstance(grad_norm, torch.distributed.tensor.DTensor):
                    grad_norm = grad_norm.full_tensor()
                torch.nn.utils.clip_grads_with_norm_([p for p in self.model.parameters()], clip_norm, grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
        return grad_norm
    
    def _run_validation_epoch(self) -> float:
        """Run one pass over `self.val_dataloader` and return average loss per token."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
                labels = batch.pop("labels")
                mask = batch.pop("loss_mask", None)
                if mask is None:
                    mask = (labels.detach() != -100).to(torch.int)

                out = self.model(**batch)
                local_loss = self.loss_fn(
                    out.logits.view(-1, out.logits.size(-1)),
                    labels.view(-1),
                    mask=mask,
                    reduction="sum"
                )
                total_loss += local_loss.item()
                total_tokens += mask.sum().item()

        # Aggregate across ranks if distributed is initialized
        if dist.is_initialized():
            tensor = torch.tensor([total_loss, total_tokens], device=self.dist_env.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = tensor.tolist()

        self.model.train()
        return total_loss / max(total_tokens, 1e-8)
    
    def log_train_metrics(self, grad_norm):
        """
        Log metrics to wandb.

        Args:
            grad_norm: Grad norm from the training step.

        Returns:
            Reporting loss.
        """
        total_loss, total_num_tokens = reduce_loss(
            self.forward_data_store, self.total_num_tokens, per_token_loss=True
        )
        reporting_loss = (total_loss / total_num_tokens).item()
        grad_norm = grad_norm.item()
        self.total_num_tokens.zero_()
        self.forward_data_store = []
        log_data = {
            "train_loss": reporting_loss,
            "loss_sum": total_loss,
            "step": self.step_scheduler.step,
            "epoch": self.step_scheduler.epoch,
            "grad_norm": grad_norm,
            "num_tokens_per_step": total_num_tokens,
        }
        if self.optimizer.param_groups:
            log_data["learning_rate"] = self.optimizer.param_groups[0]['lr']

        if wandb.run is not None:
            wandb.log(log_data)
        return reporting_loss

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
