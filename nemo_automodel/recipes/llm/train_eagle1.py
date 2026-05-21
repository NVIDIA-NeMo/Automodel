# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Minimal Llama-only EAGLE-1 training recipe."""

from __future__ import annotations

import logging
import math
import pathlib
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig, LlamaConfig

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.eagle3 import build_eagle3_dataloader
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.speculative.eagle.core_v12 import EagleTrainerModule
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import LlamaEagleDraftModel
from nemo_automodel.components.speculative.eagle.target_v12 import HFEagleTargetModel
from nemo_automodel.recipes.base_recipe import BaseRecipe

logger = logging.getLogger(__name__)


def _all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


class TrainEagle1Recipe(BaseRecipe):
    """Recipe for minimal Llama-only EAGLE-1 training."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        """Build target model, draft model, data, optimizer, and trainer module."""
        self.dist_env = initialize_distributed(
            backend=self.cfg.get("dist_env", {}).get("backend", "nccl"),
            timeout_minutes=self.cfg.get("dist_env", {}).get("timeout_minutes", 30),
        )
        setup_logging()

        recipe_cfg = self.cfg.recipe_args
        self.device = self.dist_env.device or torch.device("cpu")

        target_path = recipe_cfg.target_model_name_or_path
        target_config = AutoConfig.from_pretrained(
            target_path, trust_remote_code=recipe_cfg.get("trust_remote_code", False)
        )
        architectures = getattr(target_config, "architectures", []) or []
        if "LlamaForCausalLM" not in architectures:
            raise ValueError(f"TrainEagle1Recipe currently supports only LlamaForCausalLM, got {architectures}")
        if not isinstance(target_config, LlamaConfig):
            raise ValueError(f"Expected LlamaConfig for EAGLE-1 training, got {type(target_config).__name__}")

        self.tokenizer = NeMoAutoTokenizer.from_pretrained(
            target_path,
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
        )
        self.compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.target_model = NeMoAutoModelForCausalLM.from_pretrained(
            target_path,
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
            torch_dtype=self.compute_dtype,
            force_hf=True,
        ).to(self.device)
        self.target_model.requires_grad_(False)
        self.target_wrapper = HFEagleTargetModel(self.target_model)

        self.train_dataloader = build_eagle3_dataloader(
            data_path=recipe_cfg.train_data_path,
            tokenizer=self.tokenizer,
            seq_length=recipe_cfg.seq_length,
            batch_size=recipe_cfg.micro_batch_size,
            shuffle=True,
            num_workers=recipe_cfg.get("num_workers", 0),
            split=recipe_cfg.get("train_split", None),
            distributed=self.dist_env.world_size > 1,
            shuffle_seed=recipe_cfg.get("shuffle_seed", 42),
        )
        self.val_dataloader = None
        if recipe_cfg.get("val_data_path", None):
            self.val_dataloader = build_eagle3_dataloader(
                data_path=recipe_cfg.val_data_path,
                tokenizer=self.tokenizer,
                seq_length=recipe_cfg.seq_length,
                batch_size=recipe_cfg.micro_batch_size,
                shuffle=False,
                num_workers=recipe_cfg.get("num_workers", 0),
                split=recipe_cfg.get("val_split", None),
                distributed=self.dist_env.world_size > 1,
                shuffle_seed=recipe_cfg.get("shuffle_seed", 42),
            )

        draft_config = target_config.to_dict()
        draft_config["architectures"] = ["LlamaEagleDraftModel"]
        draft_config["draft_num_hidden_layers"] = int(recipe_cfg.get("draft_num_hidden_layers", 1))
        self.draft_model = LlamaEagleDraftModel(LlamaConfig.from_dict(draft_config)).to(
            device=self.device, dtype=self.compute_dtype
        )
        self.draft_model.copy_embeddings_from_target(self.target_wrapper.get_input_embeddings())
        if recipe_cfg.get("freeze_embeddings", True):
            self.draft_model.freeze_embeddings()

        trainer_module = EagleTrainerModule(
            self.draft_model,
            target_lm_head=self.target_wrapper.get_lm_head(),
            hidden_loss_weight=float(recipe_cfg.get("hidden_loss_weight", 1.0)),
            token_loss_weight=float(recipe_cfg.get("token_loss_weight", 0.1)),
        ).to(self.device)
        if self.dist_env.world_size > 1:
            trainer_module = DistributedDataParallel(
                trainer_module,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        self.trainer_module = trainer_module

        opt_cfg = self.cfg.optimizer
        self.peak_lr = float(opt_cfg.lr)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.trainer_module.parameters() if p.requires_grad],
            lr=self.peak_lr,
            betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
        self.grad_accumulation_steps = recipe_cfg.get("grad_accumulation_steps", 1)
        self.max_grad_norm = recipe_cfg.get("max_grad_norm", 1.0)
        self.num_epochs = recipe_cfg.num_epochs
        self.log_every_steps = recipe_cfg.get("log_every_steps", 10)
        self.output_dir = pathlib.Path(recipe_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            num_batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            num_batches_per_epoch = 0
        total_optim_steps = max(1, (self.num_epochs * num_batches_per_epoch) // self.grad_accumulation_steps)
        warmup_ratio = float(opt_cfg.get("warmup_ratio", 0.05))
        min_lr_ratio = float(opt_cfg.get("min_lr_ratio", 0.1))
        warmup_steps = max(1, int(warmup_ratio * total_optim_steps))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
        self.total_optim_steps = total_optim_steps
        self.runtime = SimpleNamespace(global_step=0)

    def _module(self):
        return (
            self.trainer_module.module
            if isinstance(self.trainer_module, DistributedDataParallel)
            else self.trainer_module
        )

    def _save_checkpoint(self, name: str):
        if not self.dist_env.is_main:
            return
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._module().draft_model.state_dict(), save_dir / "draft_model.pt")
        self._module().draft_model.config.save_pretrained(save_dir)
        torch.save({"global_step": self.runtime.global_step}, save_dir / "eagle1_meta.pt")

    def _run_eval(self):
        if self.val_dataloader is None:
            return None
        self.trainer_module.eval()
        total_loss = torch.zeros((), device=self.device)
        total_acc = torch.zeros((), device=self.device)
        total_batches = torch.zeros((), device=self.device)
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                target_batch = self.target_wrapper.generate_batch(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    loss_mask=batch["loss_mask"],
                )
                metrics = self.trainer_module(
                    input_ids=target_batch.input_ids,
                    attention_mask=target_batch.attention_mask,
                    loss_mask=target_batch.loss_mask,
                    input_hidden_states=target_batch.input_hidden_states,
                    target_hidden_states=target_batch.target_hidden_states,
                    target_logits=target_batch.target_logits,
                )
                total_loss += metrics.loss.detach()
                total_acc += metrics.accuracy.detach()
                total_batches += 1

        total_loss = _all_reduce_mean(total_loss)
        total_acc = _all_reduce_mean(total_acc)
        total_batches = _all_reduce_mean(total_batches)
        self.trainer_module.train()
        return {
            "val_loss": (total_loss / total_batches.clamp_min(1)).item(),
            "val_accuracy": (total_acc / total_batches.clamp_min(1)).item(),
        }

    def run_train_validation_loop(self):
        """Run the training loop."""
        self.trainer_module.train()
        for epoch_idx in range(self.num_epochs):
            if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch_idx)

            running_loss = 0.0
            running_acc = 0.0
            micro_step = 0
            completed_steps = 0
            last_batch_idx = -1
            for batch_idx, batch in enumerate(self.train_dataloader):
                last_batch_idx = batch_idx
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                target_batch = self.target_wrapper.generate_batch(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    loss_mask=batch["loss_mask"],
                )
                metrics = self.trainer_module(
                    input_ids=target_batch.input_ids,
                    attention_mask=target_batch.attention_mask,
                    loss_mask=target_batch.loss_mask,
                    input_hidden_states=target_batch.input_hidden_states,
                    target_hidden_states=target_batch.target_hidden_states,
                    target_logits=target_batch.target_logits,
                )
                loss = metrics.loss / self.grad_accumulation_steps
                loss.backward()

                running_loss += metrics.loss.detach().item()
                running_acc += metrics.accuracy.detach().item()
                micro_step += 1

                if micro_step % self.grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()
                    self.runtime.global_step += 1
                    completed_steps += 1

                    if self.dist_env.is_main and self.runtime.global_step % self.log_every_steps == 0:
                        avg_loss = running_loss / self.log_every_steps
                        avg_acc = running_acc / self.log_every_steps
                        logger.info(
                            "epoch=%d step=%d loss=%.4f acc=%.4f lr=%.6g",
                            epoch_idx,
                            self.runtime.global_step,
                            avg_loss,
                            avg_acc,
                            self.lr_scheduler.get_last_lr()[0],
                        )
                        running_loss = 0.0
                        running_acc = 0.0

            if micro_step % self.grad_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()
                self.runtime.global_step += 1
                completed_steps += 1

            eval_metrics = self._run_eval()
            if self.dist_env.is_main:
                msg = f"Finished epoch {epoch_idx + 1}/{self.num_epochs} completed_steps={completed_steps}"
                if eval_metrics is not None:
                    msg += f" val_loss={eval_metrics['val_loss']:.4f} val_accuracy={eval_metrics['val_accuracy']:.4f}"
                logger.info(msg)

            checkpoint_name = f"epoch_{epoch_idx + 1}"
            if last_batch_idx >= 0:
                self._save_checkpoint(checkpoint_name)


def main(config_path: str | None = None):
    """Entrypoint for ``TrainEagle1Recipe``."""
    if config_path is None:
        raise ValueError("config_path is required for TrainEagle1Recipe")
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainEagle1Recipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()
