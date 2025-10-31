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

import torch

from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricLoggerDist, MetricsSample
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.recipes.llm.train_ft import (
    build_checkpoint_config,
    build_dataloader,
    build_distributed,
    build_lr_scheduler,
    build_model_and_optimizer,
    build_step_scheduler,
)

logger = logging.getLogger(__name__)


class TrainFinetuneRecipeForSequenceClassification(BaseRecipe):
    """Recipe for fine-tuning a model for sequence classification."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        setup_logging()
        apply_cache_compatibility_patches()
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

        self.model_wrapper = None
        self.device_mesh = None
        self.moe_mesh = None
        if "distributed" in self.cfg:
            self.model_wrapper = self.cfg.distributed.instantiate(world_size=self.dist_env.world_size)
            self.device_mesh = getattr(self.model_wrapper, "device_mesh", None)
            self.moe_mesh = getattr(self.model_wrapper, "moe_mesh", None)

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            # Reuse helper from NTP recipe
            from nemo_automodel.recipes.llm.train_ft import build_wandb

            run = build_wandb(self.cfg)
            logging.info("🚀 View run at {}".format(run.url))

        self._log_experiment_details()
        self._log_library_versions()

        # For classification, use standard attention implementation
        use_hf_fa2 = False

        # loss function: standard CE on logits
        self.loss_fn = torch.nn.CrossEntropyLoss()

        checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            self.cfg.model.pretrained_model_name_or_path,
            False,
        )

        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

        self.checkpointer = Checkpointer(
            config=checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
        )

        model, model_state_dict_keys, self.optimizer, _ = build_model_and_optimizer(
            device=self.dist_env.device,
            cfg_model=self.cfg.model,
            cfg_opt=self.cfg.optimizer,
            cfg_peft=None,
            has_packed_sequence=use_hf_fa2,
            model_wrapper=self.model_wrapper,
            seed=self.cfg.get("seed", 42),
            tp_size=self.cfg.get("distributed.tp_size", 1),
            cp_size=self.cfg.get("distributed.cp_size", 1),
            cfg_fp8=None,
            cfg_compile=self.cfg.get("compile", None),
            cfg_quantization=self.cfg.get("quantization", None),
            autopipeline=None,
            loss_fn=self.loss_fn,
            parallelize_fn=None,
            load_base_model=self.cfg.get("checkpoint.load_base_model", True),
            checkpointer=self.checkpointer,
        )
        self.checkpointer.config.model_state_dict_keys = model_state_dict_keys

        self.model_parts = [model]

        self.dataloader, self.tokenizer = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            self.cfg.model,
            cfg_ps=None,
            seed=self.cfg.get("seed", 42),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            global_batch_size=self.cfg.get("step_scheduler.global_batch_size", 1),
            max_steps=self.cfg.get("step_scheduler.max_steps", None),
            val_check_interval=self.cfg.get("step_scheduler.val_every_steps", None),
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
            pp_enabled=False,
            supports_seq_lens=False,
        )

        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            self.val_dataloader, _ = build_dataloader(
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                self.cfg.model,
                cfg_ps=None,
                seed=self.cfg.get("seed", 42),
                local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
                global_batch_size=self.cfg.get("step_scheduler.global_batch_size", 1),
                max_steps=self.cfg.get("step_scheduler.max_steps", None),
                val_check_interval=self.cfg.get("step_scheduler.val_every_steps", None),
                dp_rank=self._get_dp_rank(),
                dp_world_size=self._get_dp_group_size(),
                pp_enabled=False,
            )

        self.step_scheduler = build_step_scheduler(
            self.cfg.get("step_scheduler", None),
            self.dataloader,
            self._get_dp_group_size(),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )

        self.lr_scheduler = build_lr_scheduler(self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler)

        self._log_model_and_optimizer_details(self.model_parts, self.optimizer, self.lr_scheduler)

        restore_from = self.cfg.get("checkpoint.restore_from", None)
        self.metric_logger_train = MetricLoggerDist(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = MetricLoggerDist(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "validation.jsonl"
        )
        self.load_checkpoint(restore_from)
        self._log_step_scheduler_details(self.step_scheduler)

    def run_train_validation_loop(self):
        for mp in self.model_parts:
            mp.train()

        for epoch in self.step_scheduler.epochs:
            self.step_scheduler.set_epoch(epoch)
            for batches in self.step_scheduler:
                train_log_data = self._run_train_optim_step(batches)
                self.log_train_metrics(train_log_data)

                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(epoch, self.step_scheduler.step)

                if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                    val_log_data = self._validate_one_epoch(self.val_dataloader)
                    self.log_val_metrics(val_log_data)
                    for mp in self.model_parts:
                        mp.train()

        self.metric_logger_train.close()
        self.metric_logger_valid.close()
        self.checkpointer.close()

    def _run_train_optim_step(self, batches):
        model = self.model_parts[0]
        losses = []
        for i, batch in enumerate(batches):
            batch = {
                k: (v.to(self.dist_env.device, non_blocking=True) if v is not None else None) for k, v in batch.items()
            }
            labels = batch.pop("labels")
            out = model(**batch)
            logits = getattr(out, "logits", out)
            loss = (
                self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                if logits.dim() == 3
                else self.loss_fn(logits, labels)
            )
            losses.append(loss.detach().clone())
            (loss * self._get_dp_group_size(include_cp=True)).backward()
            for opt in self.optimizer:
                opt.step()
                opt.zero_grad()
            if self.lr_scheduler is not None:
                for scheduler in self.lr_scheduler:
                    scheduler.step(1)

        total_loss = torch.sum(torch.stack(losses))
        total_loss = self._dp_allreduce(total_loss, include_cp=True).detach()
        loss = total_loss / len(batches)
        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": loss,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": 0.0,
                "tps_per_gpu": 0.0,
                "num_tokens_per_step": 0,
                "num_label_tokens": 0,
            },
        )
    @torch.no_grad()
    def _validate_one_epoch(self, dataloader):
        model = self.model_parts[0]
        model.eval()
        total_loss = 0.0
        count = 0
        for batch in dataloader:
            batch = {
                k: (v.to(self.dist_env.device, non_blocking=True) if v is not None else None) for k, v in batch.items()
            }
            labels = batch.pop("labels")
            out = model(**batch)
            logits = getattr(out, "logits", out)
            loss = (
                self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                if logits.dim() == 3
                else self.loss_fn(logits, labels)
            )
            total_loss += loss.detach()
            count += 1
        total_loss = total_loss if count == 0 else total_loss / count
        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": total_loss,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "num_label_tokens": 0,
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

        # if wandb.run is not None:
        #     wandb.log(log_data.to_dict(), step=log_data.step)

        # JSONL validation log
        self.metric_logger_valid.log(log_data)

        logging.info(
            "[val] step {} | epoch {} | loss {:.4f} | lr {:.2e} | num_label_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["val_loss"],
                log_data.metrics["lr"],
                log_data.metrics["num_label_tokens"],
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
                    "grad_norm": Grad norm from the training step.
                    "lr": Learning rate.
                    "mem": Memory allocated.
                    "tps": Tokens per second.
                    "tps_per_gpu": Tokens per second per GPU.
                    "num_label_tokens": Number of label tokens.
        """
        if not self.dist_env.is_main:
            return

        # if wandb.run is not None:
        #     wandb.log(log_data.to_dict(), step=self.step_scheduler.step)
        # JSONL training log
        self.metric_logger_train.log(log_data)
        logging.info(
            "step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | num_label_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["loss"],
                0,
                # log_data.metrics["grad_norm"],
                log_data.metrics["lr"],
                log_data.metrics["mem"],
                log_data.metrics["tps"],
                log_data.metrics["tps_per_gpu"],
                log_data.metrics["num_label_tokens"],
            )
        )
        torch.cuda.reset_peak_memory_stats()


def main(config_path: str | None = None):
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
