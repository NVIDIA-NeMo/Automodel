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
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import wandb
from huggingface_hub import constants as hf_constants
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from wandb import Settings

from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.device_mesh import create_device_mesh
from nemo_automodel.components.distributed.utils import FirstRankPerNode, get_sync_ctx
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.recipes.llm.train_ft import build_distributed, build_optimizer, build_step_scheduler

logger = logging.getLogger(__name__)


def contrastive_scores_and_labels(
    query: torch.Tensor, key: torch.Tensor, current_train_n_passages: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute contrastive scores and labels without in-batch negatives.

    Args:
        query: Query embeddings [batch_size, hidden_dim]
        key: Key/passage embeddings [batch_size * n_passages, hidden_dim]
        current_train_n_passages: Number of passages per query

    Returns:
        Tuple of (scores, labels) where scores is [batch_size, n_passages]
        and labels is [batch_size] of zeros (positive is first passage)
    """
    assert key.shape[0] % query.shape[0] == 0, "{} % {} > 0".format(key.shape[0], query.shape[0])
    query_shape = query.shape
    repeated_query = query.repeat(1, 1, current_train_n_passages).reshape(
        query_shape[0] * current_train_n_passages, query_shape[1]
    )
    qk = torch.sum(repeated_query * key, dim=-1).reshape(query_shape[0], current_train_n_passages)
    labels = torch.zeros(query_shape[0], dtype=torch.long, device=query.device)
    return qk, labels


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
        model_cache_dir=cache_dir if cache_dir is not None else hf_constants.HF_HUB_CACHE,
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

        # Recipe-level training configuration
        self.train_n_passages = self.cfg.get("train_n_passages", 1)
        self.eval_negative_size = self.cfg.get("eval_negative_size", 0)
        self.temperature = self.cfg.get("temperature", 1.0)

    def setup(self):
        """Build all components needed for training/validation/logging/checkpointing."""
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        setup_logging()

        apply_cache_compatibility_patches()
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

        self.device_mesh = None
        self.moe_mesh = None
        self.distributed_config = None
        if "distributed_config" in self.cfg:
            self.distributed_config = self.cfg.distributed_config.instantiate()
            self.device_mesh, self.moe_mesh = create_device_mesh(
                self.distributed_config,
                dp_size=self.cfg.get("distributed.dp_size", None),
                dp_replicate_size=self.cfg.get("distributed.dp_replicate_size", None),
                tp_size=self.cfg.get("distributed.tp_size", 1),
                pp_size=self.cfg.get("distributed.pp_size", 1),
                cp_size=self.cfg.get("distributed.cp_size", 1),
                ep_size=self.cfg.get("distributed.ep_size", 1),
                world_size=self.dist_env.world_size,
            )

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info("🚀 View run at {}".format(run.url))

        self._log_experiment_details()
        self._log_library_versions()

        self.pp_enabled: bool = self.cfg.get("distributed.pp_size", 1) > 1
        if self.pp_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not yet supported for biencoder models. Please set distributed.pp_size to 1."
            )

        checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            self.cfg.model.pretrained_model_name_or_path,
            is_peft=False,
        )

        if self.cfg.get("clip_grad_norm.max_norm", None) is not None:
            self.max_grad_norm = float(self.cfg.clip_grad_norm.max_norm)
        else:
            logging.info("No clip_grad_norm.max_norm specified in config, using default value of 1.0")
            self.max_grad_norm = 1.0

        self.checkpointer = Checkpointer(
            config=checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
        )

        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()

        logger.info("Building biencoder model...")
        with ScopedRNG(seed=self.cfg.get("seed", 42), ranked=True):
            model = self.cfg.model.instantiate(
                device_mesh=self.device_mesh,
                moe_mesh=self.moe_mesh,
                distributed_config=self.distributed_config,
                peft_config=self.peft_config,
            )

        self.model_parts = [model]
        self.pp = None

        logger.info("Building optimizer...")
        # Apply weight decay only to non-bias/non-norm params
        decay_params = []
        no_decay_params = []
        for name, param in self.model_parts[0].named_parameters():
            if not param.requires_grad:
                continue
            name_l = name.lower()
            if name.endswith(".bias") or ("norm" in name_l):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        trainable_params = decay_params + no_decay_params
        assert len(trainable_params) > 0, "trainable_params cannot be empty"

        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

        logger.info("Optimizer param groups: decay=%d, no_decay=%d", len(decay_params), len(no_decay_params))
        self.optimizer = [self.cfg.optimizer.instantiate(params=param_groups)]

        self.tokenizer = self.cfg.tokenizer.instantiate()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        logger.info("Building dataloader...")
        self.dataloader = build_dataloader(
            self.cfg.dataloader,
            self.tokenizer,
            seed=self.cfg.get("seed", 42),
            batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
        )

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

        self.step_scheduler = build_step_scheduler(
            self.cfg.get("step_scheduler", None),
            self.dataloader,
            self._get_dp_group_size(),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )

        self.lr_scheduler = build_lr_scheduler(self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler)
        self._log_model_and_optimizer_details(self.model_parts, self.optimizer, self.lr_scheduler)

        self.metric_logger_train = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "validation.jsonl"
        )

        restore_from = self.cfg.get("checkpoint.restore_from", None)
        self.load_checkpoint(restore_from)
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
                train_log_data = self._run_train_optim_step(batches, self.max_grad_norm)
                self.log_train_metrics(train_log_data)

                val_loss = None
                if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                    val_log_data = self._run_validation_epoch(self.val_dataloader)
                    self.log_val_metrics(val_log_data)
                    val_loss = {"val_loss": val_log_data.metrics["val_loss"]}
                    for mp in self.model_parts:
                        mp.train()

                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(
                        epoch,
                        self.step_scheduler.step,
                        train_loss=train_log_data.metrics["loss"],
                        val_loss=val_loss,
                    )

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
        batch = {
            k: v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        query, passage = _unpack_qp(batch)

        model = self.model_parts[0]
        train_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
        sync_ctx = (
            get_sync_ctx(
                model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )

        with train_ctx, sync_ctx:
            q_reps = model(query, encoder="query")
            p_reps = model(passage, encoder="passage")

            n_passages = self.train_n_passages
            scores, labels = contrastive_scores_and_labels(q_reps, p_reps, n_passages)
            if model.l2_normalize:
                scores = scores / self.temperature
            loss = F.cross_entropy(scores, labels)

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

        self.checkpointer.maybe_wait_for_staging()
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()

        if self.lr_scheduler is not None:
            for scheduler in self.lr_scheduler:
                scheduler.step(1)

        # Average loss across gradient accumulation steps and DP ranks
        reporting_loss = torch.mean(torch.stack(loss_buffer))
        if torch.distributed.is_initialized():
            reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
            reporting_loss = reporting_loss / self._get_dp_group_size(include_cp=True)
        reporting_loss = reporting_loss.cpu().item()

        lr = self.optimizer[0].param_groups[0]["lr"]
        elapsed = time.perf_counter() - self.timestamp
        self.timestamp = time.perf_counter()
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
            all_scores = []
            all_labels = []

            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {
                        k: v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    query, passage = _unpack_qp(batch)

                    model = self.model_parts[0]
                    q_reps = model(query, encoder="query")
                    p_reps = model(passage, encoder="passage")

                    n_passages = self.eval_negative_size + 1
                    scores, labels = contrastive_scores_and_labels(q_reps, p_reps, n_passages)
                    if model.l2_normalize:
                        scores = scores / self.temperature
                    loss = F.cross_entropy(scores, labels)

                    loss_buffer.append(loss.clone().detach())
                    all_scores.append(scores.detach().cpu())
                    all_labels.append(labels.detach().cpu())

            avg_loss = torch.stack(loss_buffer).mean()
            if torch.distributed.is_initialized():
                avg_loss = self._dp_allreduce(avg_loss, include_cp=True)

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

        if self.step_scheduler.is_remote_logging_step:
            if wandb.run is not None:
                wandb.log(log_data.to_dict(), step=self.step_scheduler.step)

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
