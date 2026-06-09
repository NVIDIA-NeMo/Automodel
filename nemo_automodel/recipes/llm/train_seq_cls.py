# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import torch
import wandb

from nemo_automodel._transformers.mfu import AutoMFU
from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.utils import FirstRankPerNode
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.optim.precision_warnings import resolve_storage_dtype
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.utils import clip_grad_norm
from nemo_automodel.components.utils.flops_utils import calculate_mfu
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config, shard_optimizers_for_megatron_fsdp
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.recipes.llm.train_ft import (
    _build_tokenizer,
    _get_model_name,
    build_model,
)

logger = logging.getLogger(__name__)


class TrainFinetuneRecipeForSequenceClassification(BaseRecipe):
    """Recipe for fine-tuning a model for sequence classification."""

    def __init__(self, cfg):
        self.cfg = cfg if isinstance(cfg, RecipeConfig) else RecipeConfig(cfg)

    def setup(self):
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = initialize_distributed(
            backend=self.cfg.get("dist_env", {}).get("backend", "nccl"),
            timeout_minutes=self.cfg.get("dist_env", {}).get("timeout_minutes", 1),
        )
        setup_logging()
        apply_cache_compatibility_patches()
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

        (
            self.distributed_setup,
            self.mesh_context,
            self.distributed_config,
            self.device_mesh,
            self.moe_mesh,
            self.pp_enabled,
            self.pipeline_config,
            self.moe_parallel_config,
            self.activation_checkpointing,
        ) = self._distributed_setup_attributes(
            create_distributed_setup_from_config(self.cfg, world_size=self.dist_env.world_size)
        )

        if self.dist_env.is_main and self.cfg.wandb is not None:
            suppress_wandb_log_messages()
            run = self.cfg.wandb.build(run_config=self.cfg.to_dict(), model_name=_get_model_name(self.cfg.model))
            logging.info("🚀 View run at {}".format(run.url))

        self._log_experiment_details()
        self._log_library_versions()

        # For classification, use standard attention implementation
        use_hf_fa2 = False

        # loss function: standard CE on logits
        self.loss_fn = torch.nn.CrossEntropyLoss()

        checkpoint_config = self.cfg.checkpoint

        if self.cfg.get("clip_grad_norm.max_norm", None) is not None:
            self.max_grad_norm = float(self.cfg.clip_grad_norm.max_norm)
        else:
            logging.info("No clip_grad_norm.max_norm specified in config, using default value of 1.0")
            self.max_grad_norm = 1.0

        self.checkpointer = checkpoint_config.build(
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
        )

        self.peft_config = self.cfg.instantiate_path("peft")
        # Default storage dtype to fp32 for full-parameter torch.optim training so the
        # parameters serve as the fp32 master copy (no-op for PEFT / TE FusedAdam /
        # explicit model.torch_dtype). Must run before build_model.
        resolve_storage_dtype(
            self.cfg.get("model"),
            self.cfg.get("optimizer"),
            is_peft=self.peft_config is not None,
            context="llm-seq-cls",
            logger=logger,
        )
        model = build_model(
            cfg_model=self.cfg.model,
            cfg_peft=self.peft_config,
            seed=self.cfg.get("seed", 42),
            has_packed_sequence=use_hf_fa2,
            cfg_compile=self.cfg.get("compile", None),
            cfg_quantization=self.cfg.get("quantization", None),
            distributed_setup=self.distributed_setup,
            unfreeze_modules=["classifier"] if self.peft_config is not None else None,
        )
        optimizer = self.cfg.optimizer.build(model, device_mesh=self.device_mesh, is_peft=self.peft_config is not None)
        allow_megatron_fsdp_sharding = getattr(self.cfg.optimizer, "supports_megatron_fsdp_sharding", True)
        self.optimizer = shard_optimizers_for_megatron_fsdp(
            model, optimizer, self.distributed_config, allow=allow_megatron_fsdp_sharding
        )

        self.model_parts = [model]
        self.mfu_calculator = AutoMFU.from_config(self.model_parts[0])

        _, self.tokenizer = _build_tokenizer(self.cfg.model, self.cfg.dataset)

        def materialize_loader(config):
            build_context = nullcontext() if config.dataset_builds_on_all_ranks else FirstRankPerNode()
            with ScopedRNG(seed=config.seed, ranked=True):
                return config.build(
                    tokenizer=self.tokenizer,
                    dataset_build_context=build_context,
                    dp_rank=self._get_dp_rank(),
                    dp_world_size=self._get_dp_group_size(),
                    pp_enabled=False,
                    cp_size=self.cfg.get("distributed.cp_size", 1),
                )

        self.dataloader = materialize_loader(self.cfg.dataloader)

        self.val_dataloader = None
        val_configs = self.cfg.validation_dataloaders
        if val_configs:
            self.val_dataloader = materialize_loader(next(iter(val_configs.values())))

        self.best_metric_key = self.cfg.get("checkpoint.best_metric_key", "default")
        self.step_scheduler = self.cfg.step_scheduler.build(
            self.dataloader,
            self._get_dp_group_size(),
            self.cfg.get("step_scheduler.local_batch_size", 1),
        )
        self._setup_garbage_collection(self.step_scheduler)

        self.lr_scheduler = (
            self.cfg.lr_scheduler.build(self.optimizer, self.step_scheduler)
            if self.cfg.lr_scheduler is not None
            else None
        )

        self._log_model_and_optimizer_details(self.model_parts, self.optimizer, self.lr_scheduler)

        restore_from = self.cfg.get("checkpoint.restore_from", None)
        self.metric_logger_train = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "validation.jsonl"
        )
        self.load_checkpoint(restore_from)
        self._log_step_scheduler_details(self.step_scheduler)

    def run_train_validation_loop(self):
        for mp in self.model_parts:
            mp.train()
        self.timestamp = time.perf_counter()

        pbar = self._make_progress_bar()
        try:
            for epoch in self.step_scheduler.epochs:
                self.step_scheduler.set_epoch(epoch)
                for batches in self.step_scheduler:
                    train_log_data = self._run_train_optim_step(batches)
                    self.log_train_metrics(train_log_data)
                    self._update_progress_bar(pbar, train_log_data.metrics)

                    val_loss = {}
                    if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                        val_log_data = self._validate_one_epoch(self.val_dataloader)
                        val_loss["val_loss"] = val_log_data.metrics["val_loss"]
                        self.log_val_metrics(val_log_data)
                        for mp in self.model_parts:
                            mp.train()

                    if self.step_scheduler.is_ckpt_step:
                        self.save_checkpoint(
                            epoch,
                            self.step_scheduler.step,
                            train_log_data.metrics["loss"],
                            val_loss,
                            best_metric_key=self.best_metric_key,
                        )
                    self._maybe_collect_garbage()
        finally:
            if pbar is not None:
                pbar.close()

        self.metric_logger_train.close()
        self.metric_logger_valid.close()
        self._finalize_and_close_checkpointer()

    def _run_train_optim_step(self, batches):
        model = self.model_parts[0]
        losses = []
        all_preds = []
        all_labels = []

        # Count input tokens for throughput calculation (excluding padding)
        num_tokens_in_batch = torch.tensor(
            sum(batch["attention_mask"].sum().item() for batch in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()

        for batch in batches:
            batch = {
                k: (v.to(self.dist_env.device, non_blocking=True) if v is not None else None) for k, v in batch.items()
            }
            labels = batch.pop("labels")
            batch = filter_forward_kwargs(model, batch)
            out = model(**batch)
            logits = getattr(out, "logits", out)
            loss = self.loss_fn(logits, labels.view(-1))
            losses.append(loss.detach().clone())

            # Collect predictions for accuracy calculation
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.detach())
            all_labels.append(labels.view(-1).detach())
            (loss * self._get_dp_group_size(include_cp=True)).backward()

        # Calculate gradient norm (distributed-aware)
        grad_norm = clip_grad_norm(
            max_grad_norm=self.max_grad_norm,
            model_parts=self.model_parts,
            norm_type=2.0,
            pp_enabled=self._get_pp_rank() != 0 if hasattr(self, "_get_pp_rank") else False,
            device_mesh=self.device_mesh,
        )

        # Calculate accuracy
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        correct = (all_preds == all_labels).sum()
        total = all_labels.numel()
        accuracy = correct.float() / total

        # Sync accuracy across distributed ranks if needed
        if self._get_dp_group_size(include_cp=True) > 1:
            correct = self._dp_allreduce(correct.float(), include_cp=True)
            total_across_ranks = self._dp_allreduce(
                torch.tensor(total, device=correct.device, dtype=torch.float), include_cp=True
            )
            accuracy = correct / total_across_ranks

        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()
        if self.lr_scheduler is not None:
            for scheduler in self.lr_scheduler:
                scheduler.step(1)

        # Calculate throughput (tokens per second)
        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta

        mfu = None
        mfu_calculator = getattr(self, "mfu_calculator", None)
        if batches and mfu_calculator is not None:
            step_flops = 0.0
            flops_supported = True
            for batch in batches:
                input_ids = batch.get("input_ids")
                if input_ids is None:
                    flops_supported = False
                    break
                batch_flops = mfu_calculator.get_flops(input_ids)
                if batch_flops is None:
                    flops_supported = False
                    break
                step_flops += float(batch_flops)

            if flops_supported:
                step_flops = self._dp_allreduce(
                    torch.tensor(step_flops, dtype=torch.float64, device=self.dist_env.device), include_cp=True
                ).item()
                mfu = calculate_mfu(step_flops / 1e12, self.dist_env.world_size, time_delta)

        total_loss = torch.sum(torch.stack(losses))
        total_loss = self._dp_allreduce(total_loss, include_cp=True).detach()
        loss = total_loss / len(batches)

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": loss,
                "accuracy": accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy,
                "grad_norm": grad_norm,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / self._get_cp_group_size() / max(self._get_dp_group_size(), 1),
                "mfu": mfu,
            },
        )

    @torch.no_grad()
    def _validate_one_epoch(self, dataloader):
        model = self.model_parts[0]
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        count = 0

        for batch in dataloader:
            batch = {
                k: (v.to(self.dist_env.device, non_blocking=True) if v is not None else None) for k, v in batch.items()
            }
            labels = batch.pop("labels")
            batch = filter_forward_kwargs(model, batch)
            out = model(**batch)
            logits = getattr(out, "logits", out)
            loss = self.loss_fn(logits, labels.view(-1))
            total_loss += loss.detach()

            # Collect predictions for accuracy
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds)
            all_labels.append(labels.view(-1))
            count += 1

        total_loss = total_loss if count == 0 else total_loss / count

        # Calculate accuracy
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            correct = (all_preds == all_labels).sum()
            total = all_labels.numel()
            accuracy = correct.float() / total

            # Sync across distributed ranks if needed
            if self._get_dp_group_size(include_cp=True) > 1:
                correct = self._dp_allreduce(correct.float(), include_cp=True)
                total_across_ranks = self._dp_allreduce(
                    torch.tensor(total, device=correct.device, dtype=torch.float), include_cp=True
                )
                accuracy = correct / total_across_ranks
        else:
            accuracy = 0.0

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": total_loss,
                "val_accuracy": accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )

    def log_val_metrics(self, log_data):
        """Log metrics to wandb and other loggers
        Args:
            log_data: MetricsSample object, containing:
                step: int, the current step.
                epoch: int, the current epoch.
                metrics: Dict[str, float], containing:
                    "val_loss": Validation loss.
                    "lr": Learning rate.
                    "num_label_tokens": Number of label tokens.
                    "mem": Memory allocated.
        """

        # Pipeline parallelism does not support validation -> log_data is None
        if not self.dist_env.is_main or log_data is None:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=log_data.step)

        # JSONL validation log
        self.metric_logger_valid.log(log_data)

        logging.info(
            "[val] step {} | epoch {} | loss {:.4f} | accuracy {:.4f} | lr {:.2e}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["val_loss"],
                log_data.metrics["val_accuracy"],
                log_data.metrics["lr"],
            )
        )

    def log_train_metrics(self, log_data):
        """Log metrics to wandb and other loggers.

        Args:
            log_data: MetricsSample object, containing:
                step: int, the current step.
                epoch: int, the current epoch.
                metrics: Dict[str, float], containing:
                    "loss": Training loss.
                    "accuracy": Training accuracy.
                    "grad_norm": Gradient norm from the training step.
                    "lr": Learning rate.
                    "mem": Memory allocated.
                    "tps": Tokens per second (throughput).
                    "tps_per_gpu": Tokens per second per GPU.
        """
        if not self.dist_env.is_main:
            return

        # Log to remote services (WandB) according to step_scheduler frequency
        if self.step_scheduler.is_remote_logging_step:
            if wandb.run is not None:
                wandb.log(log_data.to_dict(), step=self.step_scheduler.step)

        # JSONL training log (always log for detailed local records)
        self.metric_logger_train.log(log_data)
        logging.info(
            "step {} | epoch {} | loss {:.4f} | accuracy {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu)".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["loss"],
                log_data.metrics["accuracy"],
                log_data.metrics["grad_norm"],
                log_data.metrics["lr"],
                log_data.metrics["mem"],
                log_data.metrics["tps"],
                log_data.metrics["tps_per_gpu"],
            )
        )
        torch.cuda.reset_peak_memory_stats()


def main(config_path: str | None = None):
    """Run the sequence-classification fine-tuning recipe."""
    if config_path is None:
        config_path = (
            pathlib.Path(__file__).parent.resolve()
            / "../.."
            / "examples/llm_sequence_classification/yelp/yelp_bert.yaml"
        )
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainFinetuneRecipeForSequenceClassification(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
