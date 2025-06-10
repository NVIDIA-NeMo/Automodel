from __future__ import annotations

import wandb
from wandb import Settings
from nemo_automodel.loggers.wandb_utils import suppress_wandb_log_messages

import torch.distributed as dist
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pathlib
from torch.distributed.device_mesh import _mesh_resources

try:
    from nvfsdp import nvFSDP
except ImportError:
    from nemo_automodel.distributed.nvfsdp.nvfsdp import nvFSDP

from nemo_automodel.config.cli import parse_args_and_load_config
from nemo_automodel.distributed.init_utils import initialize_distributed
from nemo_automodel.distributed.parallelizer import create_context_parallel_ctx, get_train_context
from nemo_automodel.training.base_recipe import BaseRecipe
from nemo_automodel.training.step_scheduler import StepScheduler
from nemo_automodel.utils.dist_utils import reduce_loss, get_sync_ctx, rescale_gradients, clip_gradients
from transformers import AutoTokenizer
from nemo_automodel.loggers.log_utils import setup_logging

import logging
logger = logging.getLogger(__name__)

# ---------------------------
#  Stateless helper functions
# ---------------------------

def build_model(device, cfg_model, cfg_peft, model_wrapper) -> nn.Module:
    """
    Build and initialize a model.

    Args:
        device: The target device.
        model_wrapper: Optional parallelism wrapper.
        cfg_model: Configuration for model instantiation.

    Returns:
        The instantiated model on the specified device.
    """
    model = cfg_model.instantiate()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.weight.requires_grad_(False)
    # Optionally apply PEFT (e.g., LoRA/DoRA, etc)
    if cfg_peft is not None:
        opts = cfg_peft.to_dict()
        peft_fn = opts.pop('peft_fn')
        peft_fn(model, **opts)

    if callable(getattr(model_wrapper, 'parallelize', None)):
        model = model_wrapper.parallelize(model)

        # FSDP2 and nvFSDP should already be on the correct device
        return model
    else:
        return model.to(device)

def build_optimizer(cfg_opt, model, tp_size) -> 'Optimizer':  # noqa: F821
    """
    Build an optimizer for the model.

    Args:
        cfg_opt: Configuration for optimizer instantiation.
        model: The model whose parameters will be optimized.
        tp_size: The size of the tensor parallel group.

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


def build_dataloader(cfg_ds, cfg_dl, cfg_model, distributed_sampler_kwargs) -> DataLoader:
    """
    Build a DataLoader for the dataset.

    Args:
        cfg_ds: Dataset configuration.
        cfg_dl: DataLoader configuration.
        distributed_sampler_kwargs: Additional arguments for the DistributedSampler.

    Returns:
        The instantiated DataLoader.
    """
    if not 'tokenizer' in cfg_ds:
        tokenizer = AutoTokenizer.from_pretrained(cfg_model.pretrained_model_name_or_path)
    elif not '_target_' in cfg_ds.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(**cfg_ds.tokenizer.to_dict())
    else:
        tokenizer = cfg_ds.tokenizer.instantiate()
    ds = cfg_ds.instantiate(tokenizer=tokenizer)
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds,
        num_replicas=distributed_sampler_kwargs.get("num_replicas", 1),
        rank=distributed_sampler_kwargs.get("rank", 0),
        shuffle=distributed_sampler_kwargs.get("shuffle", False),
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
        dataloader = dataloader,
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
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        # setups logging and adds the rankfilter to logging
        setup_logging()

        self.device_mesh = None
        self.model_wrapper = None
        distributed_sampler_kwargs = {}
        if "distributed" in self.cfg:
            self.model_wrapper = self.cfg.distributed.instantiate(
                world_size=self.dist_env.world_size
            )
            distributed_sampler_kwargs = {
                "num_replicas": self.model_wrapper.device_mesh["data_parallel"].size(),
                "rank": self.model_wrapper.device_mesh["data_parallel"].get_local_rank(),
                "shuffle": self.cfg.dataloader.get("shuffle", True),
            }
            if "shuffle" in self.cfg.dataloader:
                del self.cfg.dataloader.shuffle
            if hasattr(self.model_wrapper, 'device_mesh'):
                self.device_mesh = self.model_wrapper.device_mesh

        torch.manual_seed(self.cfg.get("seed", 42) + self.dist_env.rank)

        if self.dist_env.is_main and hasattr(self.cfg, 'logger'):
            suppress_wandb_log_messages()
            run = wandb.init(
                project=self.cfg.logger.get("wandb_project", "default_project"),
                entity=self.cfg.logger.get("wandb_entity"),
                name=self.cfg.logger.get("wandb_exp_name"),
                dir=self.cfg.logger.get("wandb_save_dir"),
                config=self.cfg,
                settings=Settings(silent=True),
            )
            logging.info("🚀 View run at {}".format(run.url))

        # Build components
        self.model = build_model(self.dist_env.device, self.cfg.model, self.cfg.get('peft', None), self.model_wrapper)
        self.optimizer = build_optimizer(
            self.cfg.optimizer,
            self.model,
            self.cfg.get("distributed.tp_size", 1),
        )
        self.loss_fn   = build_loss_fn(self.dist_env.device, self.cfg.loss_fn)
        self.dataloader = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            self.cfg.model,
            distributed_sampler_kwargs,
        )

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        if 'validation_dataset' in self.cfg:
            self.val_dataloader = build_dataloader(
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                self.cfg.model,
                distributed_sampler_kwargs,
            )

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
        for epoch in self.step_scheduler.epochs:
            for batch_idx, batch in enumerate(self.step_scheduler):
                self._run_train_step(batch, self.step_scheduler.is_optim_step, 1.0)
                if self.step_scheduler.is_ckpt_step and self.dist_env.is_main:
                #     self._save_checkpoint()
                    pass

                if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                    self._run_validation_epoch()


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
        self.model.train()

        batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        loss_mask = batch.pop("loss_mask", None)
        if loss_mask is None:
            loss_mask = (labels.detach() != -100).to(torch.int)

        # TODO(@boxiangw): Refractor. Needed for SP support
        # If 'position_ids' does not exist in batch already then override it. batch in case of Packed sequence
        # contains 'position_ids' and we don't want to override it.
        if (
            'position_ids' not in batch and
            (
                self.device_mesh["context_parallel"].size() > 1 or
                self.device_mesh["tensor_parallel"].size() > 1
            )
        ):
            batch["position_ids"] = torch.arange(0, batch['input_ids'].shape[1]).unsqueeze(0).to(self.model.device)

        # based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/train.py#L336
        if self.device_mesh["context_parallel"].size() > 1:

            input_ids = batch["input_ids"].to(self.model.device)
            position_ids = batch["position_ids"].to(self.model.device)

            if loss_mask is not None:
                cp_buffers = [input_ids, labels, position_ids, loss_mask]
                cp_seq_dims = [1, 1, 1, 1]
                cp_no_restore_buffers = {input_ids, labels, loss_mask}
            else:
                cp_buffers = [input_ids, labels, position_ids]
                cp_seq_dims = [1, 1, 1]
                cp_no_restore_buffers = {input_ids, labels}

            context_parallel_ctx = create_context_parallel_ctx(
                cp_mesh=self.model_wrapper.device_mesh["context_parallel"],
                cp_buffers=cp_buffers,
                cp_seq_dims=cp_seq_dims,
                cp_no_restore_buffers=cp_no_restore_buffers,
                cp_rotate_method="allgather",  # TODO add "alltoall" option
            )
            train_context = get_train_context(
                False,
                False,
            )

            with train_context(context_parallel_ctx):
                out  = self.model(**batch)

                # Prepare for loss calculation
                logits = out.logits.float()
                n_cls = logits.shape[-1]
                logits = logits.view(-1, n_cls)
                labels = labels.view(-1)
                assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
                local_loss = self.loss_fn(logits, labels, loss_mask)

            # In the case where all labels are masked, the loss should be 0.
            if loss_mask is not None and loss_mask.bool().sum() == 0:
                local_loss.detach().copy_(torch.zeros_like(local_loss))

        else:
            out  = self.model(**batch)
            local_loss = self.loss_fn(
                out.logits.view(-1, out.logits.size(-1)),
                labels.view(-1),
                mask=loss_mask,
                reduction="sum"
            )

        local_num_tokens = loss_mask.sum().detach().to(torch.int)
        self.total_num_tokens += local_num_tokens
        self.forward_data_store.append(local_loss.detach())

        with get_sync_ctx(self.model, is_optim_step):
            local_loss.backward()

        grad_norm = None
        if is_optim_step:
            rescale_gradients(
                self.model,
                self.total_num_tokens,
                self.device_mesh[(
                    "dp_cp"
                    if "dp_cp" in _mesh_resources.root_to_flatten_mapping.get(self.device_mesh, {})
                    else "data_parallel"
                )].get_group(),
                self.device_mesh[(
                    "dp_cp"
                    if "dp_cp" in _mesh_resources.root_to_flatten_mapping.get(self.device_mesh, {})
                    else "data_parallel"
                )].size()
            )

            # Clip gradients **after** any rescaling.
            # TODO(@boxiangw): Fix TP gradient clipping
            if self.device_mesh["tensor_parallel"].size() == 1:
                grad_norm = clip_gradients(self.model, clip_norm)
            else:
                # TODO: TP WAR
                grad_norm = 0.

            if isinstance(self.model, nvFSDP):
                # If the model uses nvFSDP, wait for all sharded gradients to be reduced and unsharded.
                # Necessary because the post-backward reduce-scatter is asynchronous, so gradients and backward
                # computations are concurrent, but the gradients of the final layer may not be available yet.
                self.model.finish_grad_sync()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if isinstance(self.model, nvFSDP):
                # If custom FSDP2 is configured with "optim" (optimizer state / high-precision model weight sharding),
                # then the optimizer step will be applied to the main high-precision model weights. Update the model
                # weights after the optimizer step.
                self.model.install_optimized_model_weights()
                self.model.zero_grad_buffer()

            # log
            reporting_loss = self.log_train_metrics(grad_norm)
            logging.info("step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | mem: {:.2f} GiB".format(
                    self.step_scheduler.step, self.step_scheduler.epoch, reporting_loss, grad_norm,
                    torch.cuda.max_memory_allocated() / 1024 ** 3
                )
            )
            torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def _run_validation_epoch(self) -> float:
        """Run one pass over `self.val_dataloader` and return average loss per token."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for batch in self.val_dataloader:
            batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss_mask = batch.pop("loss_mask", None)
            if loss_mask is None:
                loss_mask = (labels.detach() != -100).to(torch.int)

            if (
                'position_ids' not in batch and
                (
                    self.device_mesh["context_parallel"].size() > 1 or
                    self.device_mesh["tensor_parallel"].size() > 1
                )
            ):
                batch["position_ids"] = torch.arange(0, batch['input_ids'].shape[1]).unsqueeze(0).to(self.model.device)

            if self.device_mesh["context_parallel"].size() > 1:

                input_ids = batch["input_ids"].to(self.model.device)
                position_ids = batch["position_ids"].to(self.model.device)

                if loss_mask is not None:
                    cp_buffers = [input_ids, labels, position_ids, loss_mask]
                    cp_seq_dims = [1, 1, 1, 1]
                    cp_no_restore_buffers = {input_ids, labels, loss_mask}
                else:
                    cp_buffers = [input_ids, labels, position_ids]
                    cp_seq_dims = [1, 1, 1]
                    cp_no_restore_buffers = {input_ids, labels}

                context_parallel_ctx = create_context_parallel_ctx(
                    cp_mesh=self.model_wrapper.device_mesh["context_parallel"],
                    cp_buffers=cp_buffers,
                    cp_seq_dims=cp_seq_dims,
                    cp_no_restore_buffers=cp_no_restore_buffers,
                    cp_rotate_method="allgather",  # TODO add "alltoall" option
                )
                train_context = get_train_context(
                    False,
                    False,
                )
                with train_context(context_parallel_ctx):
                    out  = self.model(**batch)
                    # Prepare for loss calculation
                    logits = out.logits.float()
                    n_cls = logits.shape[-1]
                    logits = logits.view(-1, n_cls)
                    labels = labels.view(-1)
                    assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
                    local_loss = self.loss_fn(logits, labels, loss_mask)

                # In the case where all labels are masked, the loss should be 0.
                if loss_mask is not None and loss_mask.bool().sum() == 0:
                    local_loss.detach().copy_(torch.zeros_like(local_loss))
            else:
                out = self.model(**batch)
                local_loss = self.loss_fn(
                    out.logits.view(-1, out.logits.size(-1)),
                    labels.view(-1),
                    mask=loss_mask,
                    reduction="sum"
                )

            total_loss += local_loss.item()
            total_tokens += loss_mask.sum().item()

        # Aggregate across ranks if distributed is initialized
        if dist.is_initialized():
            tensor = torch.tensor([total_loss, total_tokens], device=self.dist_env.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = tensor.tolist()

        val_loss = total_loss / max(total_tokens, 1e-8)
        if self.dist_env.is_main:
            if wandb.run is not None:
                wandb.log(
                    {
                        "val_loss": val_loss,
                        "step": self.step_scheduler.step,
                        "epoch": self.step_scheduler.epoch
                    }
                )
        logging.info("[val] step {} | epoch {} | loss {:.4f}".format(
                self.step_scheduler.step, self.step_scheduler.epoch, val_loss
            )
        )

    def log_train_metrics(self, grad_norm):
        """
        Log metrics to wandb.

        Args:
            grad_norm: Grad norm from the training step.

        Returns:
            Reporting loss.
        """
        if self.device_mesh["context_parallel"].size() > 1:
            dp_group = self.device_mesh["dp_cp"].get_group()
        else:
            dp_group = self.device_mesh["data_parallel"].get_group()

        total_loss, total_num_tokens = reduce_loss(
            self.forward_data_store, self.total_num_tokens, per_token_loss=True, dp_group=dp_group
        )
        reporting_loss = (total_loss / total_num_tokens).item()
        grad_norm = grad_norm.item() if not isinstance(grad_norm, float) else grad_norm # TP WAR
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
    script_path = pathlib.Path(__file__).parent.resolve()
    cfg = parse_args_and_load_config(script_path / "llama_3_2_1b_hellaswag.yaml")
    trainer = FinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

if __name__ == "__main__":
    main()
