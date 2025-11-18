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

from __future__ import annotations

import logging
import pathlib
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict

import torch
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from transformers.utils.hub import TRANSFORMERS_CACHE

from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.utils import FirstRankPerNode
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricLoggerDist, MetricsSample
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.recipes.base_recipe import BaseRecipe

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.init_utils import DistInfo

import wandb
from wandb import Settings

logger = logging.getLogger(__name__)


def _unpack_qp(inputs: Dict[str, torch.Tensor]) -> tuple:
    """Unpack query and passage inputs from batch dictionary.

    Args:
        inputs: Dictionary containing query (q_*) and passage (d_*) tensors

    Returns:
        Tuple of (query_batch_dict, doc_batch_dict)
    """
    q_prefix, d_prefix, kd_labels_key = "q_", "d_", "kd_labels"
    query_batch_dict = {k[len(q_prefix) :]: v for k, v in inputs.items() if k.startswith(q_prefix)}
    doc_batch_dict = {k[len(d_prefix) :]: v for k, v in inputs.items() if k.startswith(d_prefix)}

    if kd_labels_key in inputs:
        assert len(query_batch_dict) > 0
        query_batch_dict[kd_labels_key] = inputs[kd_labels_key]

    if not query_batch_dict:
        query_batch_dict = None
    if not doc_batch_dict:
        doc_batch_dict = None

    return query_batch_dict, doc_batch_dict


def build_distributed(cfg_dist: Dict[str, Any]) -> "DistInfo":  # noqa: F821
    """Build and initialize distributed training resources.

    Args:
        cfg_dist: Configuration for distributed training.

    Returns:
        Distributed training information from initialize_distributed.
    """
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)


def build_checkpoint_config(cfg_ckpt, cache_dir, model_repo_id, is_peft) -> CheckpointingConfig:
    """Build a checkpoint configuration.

    Args:
        cfg_ckpt: Configuration for checkpointing.
        cache_dir: Cache directory for the model.
        model_repo_id: Model repository ID.
        is_peft: Whether the model is PEFT.

    Returns:
        The instantiated checkpoint configuration.
    """

    ckpt_kwargs = dict(
        enabled=False,
        checkpoint_dir="checkpoints/",
        model_save_format="safetensors",
        model_repo_id=model_repo_id,
        model_cache_dir=cache_dir if cache_dir is not None else TRANSFORMERS_CACHE,
        save_consolidated=False,
        is_peft=is_peft,
    )
    if cfg_ckpt is not None:
        cfg_ckpt = cfg_ckpt.to_dict()
        cfg_ckpt.pop("restore_from", None)
        cfg_ckpt.pop("load_base_model", None)
        ckpt_kwargs |= cfg_ckpt
    if ckpt_kwargs.get("is_peft", False) and ckpt_kwargs.get("model_save_format") == "torch_save":
        raise ValueError(
            "PEFT checkpointing is not supported for torch_save format. Save using `safetensors` format instead."
        )
    checkpoint_config = CheckpointingConfig(**ckpt_kwargs)
    return checkpoint_config


def build_step_scheduler(cfg, dataloader, dp_group_size, local_batch_size):
    """Build the step scheduler.

    Args:
        cfg: configuration for the StepScheduler class.
        dataloader: the training dataloader, used for extracting the epoch_len (in batches).
        dp_group_size: the size of the data parallel group.
        local_batch_size: the size of the local batch.

    Returns:
        StepScheduler: the configured StepScheduler.
    """
    assert "_target_" not in cfg, "_target_ not permitted in step scheduler"
    default_kwargs = dict(
        num_epochs=10,
        global_batch_size=32,
        local_batch_size=local_batch_size,
        dp_size=dp_group_size,
        ckpt_every_steps=100,
        dataloader=dataloader,
    )
    if cfg is not None:
        default_kwargs |= cfg.to_dict()
    return StepScheduler(**default_kwargs)


def build_lr_scheduler(cfg, optimizer, step_scheduler) -> list[OptimizerParamScheduler] | None:  # noqa: F821
    """Build the learning rate scheduler.

    Args:
        cfg: Configuration for the OptimizerParamScheduler.
        optimizer: The optimizer to be scheduled.
        step_scheduler: The step scheduler to extract training parameters.

    Returns:
        OptimizerParamScheduler: The configured learning rate scheduler, or None if not configured.
    """
    if cfg is None:
        return None

    # Calculate total steps for the training run
    total_epochs = step_scheduler.num_epochs
    epoch_len = len(step_scheduler.dataloader)
    grad_acc_steps = step_scheduler.grad_acc_steps

    # Total optimizer steps (accounting for gradient accumulation)
    total_steps = (total_epochs * epoch_len) // grad_acc_steps
    if step_scheduler.max_steps is not None:
        total_steps = min(total_steps, step_scheduler.max_steps)

    # Set defaults for scheduler parameters
    optimizer_param_schedulers = []
    user_kwargs = cfg.to_dict()
    default_kwargs = dict(
        lr_warmup_steps=min(1000, total_steps // 10),  # 10% warmup or max 1000 steps
        lr_decay_steps=total_steps,
        lr_decay_style="linear",
        wd_incr_steps=total_steps,
        wd_incr_style="constant",
    )

    if not isinstance(optimizer, list):
        optimizer = [optimizer]

    for opt in optimizer:
        base_lr = opt.param_groups[0]["lr"]
        default_kwargs.update(
            dict(
                optimizer=opt,
                init_lr=0.0,
                max_lr=base_lr,
                min_lr=0.0,
                start_wd=opt.param_groups[0].get("weight_decay", 0.0),
                end_wd=opt.param_groups[0].get("weight_decay", 0.0),
            )
        )
        default_kwargs.update(user_kwargs)
        optimizer_param_schedulers.append(OptimizerParamScheduler(**default_kwargs))

    logger.info(
        f"Building LR scheduler with total_steps={total_steps}, "
        f"warmup_steps={default_kwargs['lr_warmup_steps']}, "
        f"decay_style={default_kwargs['lr_decay_style']}"
    )

    return optimizer_param_schedulers


def build_wandb(cfg) -> wandb.Run:
    """Instantiates wandb and returns the instance. If no name is given, it will use the model name.

    Args:
        cfg: Configuration for wandb.

    Returns:
        The wandb instance.
    """
    assert cfg.get("wandb", None) is not None
    kwargs = cfg.wandb.to_dict()
    if kwargs.get("name", "") == "":
        model_name_or_path = cfg.model.get("pretrained_model_name_or_path", "biencoder_model")
        kwargs["name"] = "_".join(model_name_or_path.split("/")[-2:])
    run = wandb.init(
        **kwargs,
        config=cfg.to_dict(),
        settings=Settings(silent=True),
    )
    return run


def get_sync_ctx(model, sync_this_step):
    """Get synchronization context for gradient accumulation."""
    if hasattr(model, "no_sync") and not sync_this_step:
        return model.no_sync()
    return nullcontext()


def build_dataloader(cfg_dl, tokenizer, seed, batch_size=None, dp_rank=0, dp_world_size=1):
    """Build a DataLoader for biencoder training.

    Args:
        cfg_dl: DataLoader configuration.
        tokenizer: The tokenizer to use for collate_fn.
        seed: Random seed.
        batch_size: Batch size for the dataloader. Optional.
        dp_rank: Data parallel rank.
        dp_world_size: Data parallel world size.

    Returns:
        The instantiated DataLoader.
    """
    with ScopedRNG(seed=seed, ranked=True):
        # Build dataset
        with FirstRankPerNode():
            dataset = cfg_dl.dataset.instantiate()

        # Build collate_fn if it's a ConfigNode with _target_
        collate_fn = None
        if hasattr(cfg_dl, "collate_fn") and hasattr(cfg_dl.collate_fn, "_target_"):
            collate_fn = cfg_dl.collate_fn.instantiate(tokenizer=tokenizer)

        # Build dataloader with instantiated components
        if not isinstance(dataset, IterableDataset):
            shuffle = cfg_dl.get("shuffle", True)
            if "shuffle" in cfg_dl:
                del cfg_dl.shuffle

            dist_sampler_kwargs = {
                "num_replicas": dp_world_size,
                "rank": dp_rank,
                "shuffle": shuffle,
            }
            sampler = StatefulDistributedSampler(
                dataset,
                seed=seed,
                drop_last=True,
                **dist_sampler_kwargs,
            )
            dl_kwargs = {"sampler": sampler, "batch_size": batch_size}
        else:
            logging.info("Using IterableDataset; skipping sampler.")
            dl_kwargs = {"dataset": dataset, "batch_size": batch_size}

        dl_kwargs["dataset"] = dataset
        if collate_fn is not None:
            dl_kwargs["collate_fn"] = collate_fn

        return cfg_dl.instantiate(**dl_kwargs)


class TrainBiencoderRecipe(BaseRecipe):
    """Recipe for training biencoder models.

    This class orchestrates biencoder training, from setup to main training loop.
    It handles the unique aspects of biencoder training including dual encoders
    and contrastive learning.
    """

    def __init__(self, cfg):
        """Initialize the recipe with configuration.

        Args:
            cfg: Configuration dictionary/object for training.
        """
        self.cfg = cfg

    def setup(self):
        """Build all components needed for training/validation/logging/checkpointing."""
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        # setups logging and adds the rankfilter to logging
        setup_logging()

        apply_cache_compatibility_patches()
        # Set up the stateful random number generator
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

        self.device_mesh = None
        self.moe_mesh = None
        self.model_wrapper = None
        if "distributed" in self.cfg:
            self.model_wrapper = self.cfg.distributed.instantiate(world_size=self.dist_env.world_size)
            self.device_mesh = getattr(self.model_wrapper, "device_mesh", None)
            self.moe_mesh = getattr(self.model_wrapper, "moe_mesh", None)

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info("🚀 View run at {}".format(run.url))

        # Log experiment details on main rank
        self._log_experiment_details()
        self._log_library_versions()

        self.pp_enabled: bool = (
            True if hasattr(self.model_wrapper, "pp_size") and self.model_wrapper.pp_size > 1 else False
        )
        autopipeline_cfg = self.cfg.get("autopipeline", None)
        if self.pp_enabled:
            pp_batch_size = self.cfg.step_scheduler.local_batch_size
            pp_microbatch_size = self.cfg.autopipeline.pp_microbatch_size
            assert pp_batch_size // self.cfg.autopipeline.pp_microbatch_size >= self.model_wrapper.pp_size, (
                f"pp_batch_size {pp_batch_size} // pp_microbatch_size {self.cfg.autopipeline.pp_microbatch_size} must be greater than or equal to pp_size {self.model_wrapper.pp_size}"
            )

            assert autopipeline_cfg is not None, (
                "AutoPipeline configuration is required when pipeline parallelism is enabled"
            )
            assert not isinstance(self.model_wrapper, MegatronFSDPManager), (
                "MegatronFSDPManager is not supported when pipeline parallelism is enabled"
            )
            # Create AutoPipeline from config
            autopipeline = autopipeline_cfg.instantiate(
                world_mesh=self.device_mesh,
                moe_mesh=self.moe_mesh,
                pp_axis_name="pp",
                dp_axis_names=(
                    ("dp_replicate", "dp_shard_cp")
                    if "dp_replicate" in self.device_mesh.mesh_dim_names
                    and "dp_shard_cp" in self.device_mesh.mesh_dim_names
                    else ("dp_shard_cp",)
                ),
                cp_axis_name="cp" if "cp" in self.device_mesh.mesh_dim_names else None,
                tp_axis_name="tp" if "tp" in self.device_mesh.mesh_dim_names else None,
                ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
                ep_shard_axis_names=(
                    ("ep_shard",) if self.moe_mesh is not None and "ep_shard" in self.moe_mesh.mesh_dim_names else None
                ),
                pp_batch_size=pp_batch_size,
                pp_microbatch_size=pp_microbatch_size,
                patch_stage_backward_maybe_with_nosync=self.cfg.get("model.backend.enable_fsdp_optimizations", False),
                device=torch.cuda.current_device(),
            )
            assert isinstance(autopipeline, AutoPipeline), (
                f"autopipeline {autopipeline.__class__} is not an instance of AutoPipeline"
            )
            logger.warning(
                "Pipeline parallelism is enabled for biencoder training. "
                "Note that biencoder models typically do not benefit from PP and this is experimental."
            )
        else:
            autopipeline = None

        # Build components
        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()

        # Build checkpoint config
        checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            self.cfg.model.pretrained_model_name_or_path,
            True if self.cfg.get("peft", None) else False,
        )

        # Create Checkpointer instance
        self.checkpointer = Checkpointer(
            config=checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
        )

        # Build biencoder model
        logger.info("Building biencoder model...")
        if self.pp_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not yet supported for biencoder models. "
                "Please disable pipeline parallelism in the distributed config."
            )
        model = self.cfg.model.instantiate()

        # Apply parallelism wrapper if needed
        if self.model_wrapper is not None:
            model = self.model_wrapper.parallelize(model)

        # Ensure the model is on the correct device
        model = model.to(self.dist_env.device)

        # Setup model_parts for consistency with train_ft.py
        if isinstance(model, AutoPipeline):
            self.model_parts = model.parts
            self.pp = model
        else:
            self.model_parts = [model]
            self.pp = None

        # Build optimizer
        logger.info("Building optimizer...")
        trainable_params = list(filter(lambda x: x.requires_grad, self.model_parts[0].parameters()))
        assert len(trainable_params) > 0, "trainable_params cannot be empty"
        self.optimizer = [self.cfg.optimizer.instantiate(params=trainable_params)]

        # Build tokenizer
        self.tokenizer = self.cfg.tokenizer.instantiate()

        # Set up padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        # Build dataloader
        logger.info("Building dataloader...")
        self.dataloader = build_dataloader(
            self.cfg.dataloader,
            self.tokenizer,
            seed=self.cfg.get("seed", 42),
            batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
        )

        # Build validation dataloader if provided
        self.val_dataloader = None
        if "validation_dataloader" in self.cfg:
            logger.info("Building validation dataloader...")
            val_batch_size = self.cfg.get(
                "validation_dataloader.batch_size", self.cfg.get("step_scheduler.local_batch_size", 1)
            )
            self.val_dataloader = build_dataloader(
                self.cfg.validation_dataloader,
                self.tokenizer,
                seed=self.cfg.get("seed", 42),
                batch_size=val_batch_size,
                dp_rank=self._get_dp_rank(),
                dp_world_size=self._get_dp_group_size(),
            )

        # Build step scheduler
        self.step_scheduler = build_step_scheduler(
            self.cfg.get("step_scheduler", None),
            self.dataloader,
            self._get_dp_group_size(),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )

        # Build learning rate scheduler
        self.lr_scheduler = build_lr_scheduler(self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler)

        # Log model and optimizer details
        self._log_model_and_optimizer_details(self.model_parts, self.optimizer, self.lr_scheduler)

        # Initialize JSONL loggers
        self.metric_logger_train = MetricLoggerDist(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = MetricLoggerDist(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "validation.jsonl"
        )

        # Optionally resume from checkpoint
        restore_from = self.cfg.get("checkpoint.restore_from", None)
        self.load_checkpoint(restore_from)

        # Log step scheduler details
        self._log_step_scheduler_details(self.step_scheduler)

    def run_train_validation_loop(self):
        """Run the training loop over all epochs and batches."""
        for mp in self.model_parts:
            mp.train()
        self.timestamp = time.perf_counter()

        for epoch in self.step_scheduler.epochs:
            self.step_scheduler.set_epoch(epoch)
            # The step scheduler yields a list of batches for gradient accumulation
            for batches in self.step_scheduler:
                train_log_data = self._run_train_optim_step(batches, 1.0)

                # Log metrics
                self.log_train_metrics(train_log_data)

                # Save checkpoint every ckpt_every_steps
                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(epoch, self.step_scheduler.step)

                # Run validation every val_every_steps
                if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                    val_log_data = self._run_validation_epoch(self.val_dataloader)
                    self.log_val_metrics(val_log_data)
                    for mp in self.model_parts:
                        mp.train()

        # Close JSONL loggers after training loop completes
        self.metric_logger_train.close()
        self.metric_logger_valid.close()
        self.checkpointer.close()

    def _forward_backward_step(self, idx, batch, *, loss_buffer, num_batches, is_train: bool = True):
        """Forward and backward pass for a single batch.

        Args:
            idx: Index of the batch in gradient accumulation steps
            batch: Input batch containing query and document tensors
            loss_buffer: List to accumulate losses
            num_batches: Total number of batches in gradient accumulation
            is_train: Whether this is a training step
        """
        # Move batch to device
        batch = {
            k: v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Unpack query and passage inputs using the same logic as biencoder_trainer.py
        query, passage = _unpack_qp(batch)

        # Forward pass
        model = self.model_parts[0]
        train_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
        sync_ctx = get_sync_ctx(model, idx == num_batches - 1) if is_train else nullcontext()

        with train_ctx, sync_ctx:
            outputs = model(query=query, passage=passage)
            loss = outputs.loss

            loss_buffer.append(loss.clone().detach())

            if is_train:
                # Scale loss by number of gradient accumulation steps to get correct average gradients
                # FSDP/DDP will handle averaging across DP ranks automatically
                scaled_loss = loss / num_batches
                scaled_loss.backward()

    def _run_train_optim_step(self, batches, max_grad_norm=None):
        """Run one optimization step with gradient accumulation.

        Args:
            batches: List of batches for gradient accumulation
            max_grad_norm: Gradient clipping norm. Optional, if None will not clip gradients.

        Returns:
            MetricsSample with training metrics
        """
        loss_buffer = []

        # Gradient accumulation
        for idx, batch in enumerate(batches):
            self._forward_backward_step(idx, batch, loss_buffer=loss_buffer, num_batches=len(batches), is_train=True)

        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm,
            self.model_parts,
            norm_type=2.0,
            pp_enabled=self.pp_enabled,
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
            pp_axis_name="pp" if self.pp_enabled else None,
            foreach=True,
            num_label_tokens=None,  # Not applicable for biencoder
            dp_group_size=self._get_dp_group_size(include_cp=True),
        )

        # Optimizer step
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()

        # LR scheduler step
        if self.lr_scheduler is not None:
            for scheduler in self.lr_scheduler:
                scheduler.step(1)

        # Compute average loss across gradient accumulation and DP ranks
        reporting_loss = torch.sum(torch.stack(loss_buffer))
        if torch.distributed.is_initialized():
            reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
            # Divide by DP group size to get average across all ranks
            reporting_loss = reporting_loss / self._get_dp_group_size(include_cp=True)
        reporting_loss = reporting_loss.cpu().item()

        # Get current learning rate
        lr = self.optimizer[0].param_groups[0]["lr"]

        # Compute throughput
        elapsed = time.perf_counter() - self.timestamp
        self.timestamp = time.perf_counter()

        # Memory stats
        mem_allocated = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        metrics = {
            "loss": reporting_loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "mem": mem_allocated,
            "time_per_step": elapsed,
        }

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics=metrics,
        )

    def _run_validation_epoch(self, val_dataloader):
        """Run validation for one epoch.

        Args:
            val_dataloader: Validation data loader

        Returns:
            MetricsSample with validation metrics
        """
        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()
            loss_buffer = []

            # Metrics buffers
            all_scores = []
            all_labels = []

            with torch.no_grad():
                for batch in val_dataloader:
                    # Move batch to device
                    batch = {
                        k: v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    # Unpack query and passage inputs using the same logic as biencoder_trainer.py
                    query, passage = _unpack_qp(batch)

                    # Forward pass
                    outputs = self.model_parts[0](query=query, passage=passage)
                    loss_buffer.append(outputs.loss.clone().detach())

                    # Store scores and labels for metrics
                    all_scores.append(outputs.scores.detach().cpu())
                    all_labels.append(outputs.labels.detach().cpu())

            # Compute average loss
            avg_loss = torch.stack(loss_buffer).mean()
            if torch.distributed.is_initialized():
                avg_loss = self._dp_allreduce(avg_loss, include_cp=True)

            # Compute accuracy and MRR
            scores = torch.cat(all_scores, dim=0)
            labels = torch.cat(all_labels, dim=0)

            # Accuracy@1
            _, predicted_indices = torch.topk(scores, k=1, dim=1)
            correct = (predicted_indices.squeeze(-1) == labels).float()
            acc1 = correct.mean().item()

            # MRR
            _, sorted_indices = torch.sort(scores, dim=1, descending=True)
            ranks = (sorted_indices == labels.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
            mrr = (1.0 / ranks.float()).mean().item()

            metrics = {
                "val_loss": avg_loss.item(),
                "val_acc1": acc1,
                "val_mrr": mrr,
            }

            return MetricsSample(
                step=self.step_scheduler.step,
                epoch=self.step_scheduler.epoch,
                metrics=metrics,
            )

    def log_train_metrics(self, log_data: MetricsSample):
        """Log training metrics.

        Args:
            log_data: MetricsSample containing training metrics
        """
        if not self.dist_env.is_main:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=self.step_scheduler.step)

        # JSONL training log
        self.metric_logger_train.log(log_data)

        logging.info(
            "step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | time {:.2f}s".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["loss"],
                log_data.metrics["grad_norm"],
                log_data.metrics["lr"],
                log_data.metrics["mem"],
                log_data.metrics["time_per_step"],
            )
        )

        torch.cuda.reset_peak_memory_stats()

    def log_val_metrics(self, log_data: MetricsSample):
        """Log validation metrics.

        Args:
            log_data: MetricsSample containing validation metrics
        """
        if not self.dist_env.is_main:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=self.step_scheduler.step)

        # JSONL validation log
        self.metric_logger_valid.log(log_data)

        logging.info(
            "step {} | epoch {} | val_loss {:.4f} | val_acc1 {:.4f} | val_mrr {:.4f}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["val_loss"],
                log_data.metrics["val_acc1"],
                log_data.metrics["val_mrr"],
            )
        )

        torch.cuda.reset_peak_memory_stats()


def main(default_config_path="examples/biencoder/llama3_2_1b_biencoder.yaml"):
    """Main entry point for the biencoder fine-tuning recipe.

    Loads the configuration, sets up the recipe, and initiates the training loop.

    Args:
        default_config_path: Path to the default configuration file
    """
    cfg = parse_args_and_load_config(default_config_path)
    recipe = TrainBiencoderRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
