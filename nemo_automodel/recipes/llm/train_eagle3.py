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

"""Minimal Llama-only EAGLE-3 training recipe."""

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
from nemo_automodel.components.datasets.llm.eagle3 import (
    build_eagle3_dataloader,
    build_eagle3_token_mapping,
)
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.speculative.eagle import (
    Eagle3TrainerModule,
    HFEagle3TargetModel,
    LlamaEagle3DraftModel,
)
from nemo_automodel.recipes.base_recipe import BaseRecipe

logger = logging.getLogger(__name__)


def _all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


class TrainEagle3Recipe(BaseRecipe):
    """Recipe for the minimal Llama-only EAGLE-3 training MVP."""

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
            raise ValueError(f"TrainEagle3Recipe currently supports only LlamaForCausalLM, got {architectures}")
        if not isinstance(target_config, LlamaConfig):
            raise ValueError(f"Expected LlamaConfig for MVP EAGLE-3 training, got {type(target_config).__name__}")

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
        self.target_wrapper = HFEagle3TargetModel(
            self.target_model,
            aux_layer_ids=recipe_cfg.get("aux_layer_ids", None),
        )

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

        special_token_ids = [
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "unk_token_id", None),
        ]
        selected_token_ids, selected_token_mask = build_eagle3_token_mapping(
            self.train_dataloader,
            target_vocab_size=target_config.vocab_size,
            draft_vocab_size=recipe_cfg.get("draft_vocab_size", None),
            special_token_ids=special_token_ids,
        )

        draft_config = target_config.to_dict()
        draft_config["draft_vocab_size"] = int(selected_token_ids.numel())
        draft_config["target_hidden_size"] = target_config.hidden_size
        draft_config["architectures"] = ["LlamaEagle3DraftModel"]
        # Draft attention backend. Defaults to ``eager`` to preserve the
        # pre-FA2 numerics. Set ``recipe_args.draft_attn_implementation:
        # flash_attention_2`` in YAML to opt into FlashAttention for the
        # T x T causal block (Eagle3LlamaAttention merges FA's softmax_lse
        # with the diagonal-extension columns in log space).
        draft_config["attn_implementation"] = recipe_cfg.get("draft_attn_implementation", "eager")
        # Cast to the target's compute dtype so every linear / embedding / norm
        # in the draft matches the bf16 (cuda) or fp32 (cpu) hidden states fed
        # in from the target. Without this, ``initialize_rms_norm_module`` defaults
        # to bf16 while ``nn.Linear`` defaults to fp32, and ``model.fc`` errors
        # with ``expected mat1 and mat2 to have the same dtype``.
        self.draft_model = LlamaEagle3DraftModel(LlamaConfig.from_dict(draft_config)).to(
            device=self.device, dtype=self.compute_dtype
        )
        self.draft_model.copy_embeddings_from_target(self.target_wrapper.get_input_embeddings())
        if recipe_cfg.get("freeze_embeddings", True):
            self.draft_model.freeze_embeddings()

        trainer_module = Eagle3TrainerModule(
            self.draft_model,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            ttt_steps=recipe_cfg.ttt_steps,
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

        # Warmup + cosine LR schedule. EAGLE-3 from-scratch training with a
        # flat LR diverges after the first epoch (loss climbs back up); a
        # cosine schedule keeps the optimizer step size small enough once
        # AdamW's second-moment estimates have settled.
        try:
            num_batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            num_batches_per_epoch = 0
        total_optim_steps = max(
            1,
            (self.num_epochs * num_batches_per_epoch) // self.grad_accumulation_steps,
        )
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
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

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
        torch.save(
            {
                "selected_token_ids": self._module().selected_token_ids.cpu(),
                "selected_token_mask": self._module().selected_token_mask.cpu(),
                "global_step": self.runtime.global_step,
            },
            save_dir / "eagle3_meta.pt",
        )

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
                    aux_hidden_states=target_batch.aux_hidden_states,
                    target_logits=target_batch.logits,
                )
                total_loss = total_loss + metrics.loss.detach()
                total_acc = total_acc + metrics.accuracy.detach()
                total_batches = total_batches + 1
        total_loss = _all_reduce_mean(total_loss)
        total_acc = _all_reduce_mean(total_acc)
        total_batches = _all_reduce_mean(total_batches)
        self.trainer_module.train()
        return (total_loss / total_batches.clamp_min(1.0), total_acc / total_batches.clamp_min(1.0))

    def run_train_validation_loop(self):
        """Run the minimal EAGLE-3 train loop."""
        self.trainer_module.train()
        try:
            batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            batches_per_epoch = None
        if self.dist_env.is_main:
            logger.info(
                "Training start: num_epochs=%s batches_per_epoch=%s grad_accum=%s log_every=%s "
                "total_optim_steps=%s warmup_steps=%s peak_lr=%.3e min_lr_ratio=%s",
                self.num_epochs,
                batches_per_epoch,
                self.grad_accumulation_steps,
                self.log_every_steps,
                self.total_optim_steps,
                self.warmup_steps,
                self.peak_lr,
                self.min_lr_ratio,
            )

        for epoch in range(self.num_epochs):
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)

            running_loss = torch.zeros((), device=self.device)
            running_acc = torch.zeros((), device=self.device)
            running_steps = 0
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(self.train_dataloader):
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
                    aux_hidden_states=target_batch.aux_hidden_states,
                    target_logits=target_batch.logits,
                )
                loss = metrics.loss / float(self.grad_accumulation_steps)
                loss.backward()

                running_loss = running_loss + metrics.loss.detach()
                running_acc = running_acc + metrics.accuracy.detach()
                running_steps += 1

                if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.runtime.global_step += 1

                    if self.runtime.global_step % self.log_every_steps == 0:
                        mean_loss = _all_reduce_mean(running_loss / max(running_steps, 1))
                        mean_acc = _all_reduce_mean(running_acc / max(running_steps, 1))
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        if self.dist_env.is_main:
                            logger.info(
                                "epoch=%s step=%s train_loss=%.6f train_acc=%.6f lr=%.3e",
                                epoch,
                                self.runtime.global_step,
                                mean_loss.item(),
                                mean_acc.item(),
                                current_lr,
                            )
                        running_loss.zero_()
                        running_acc.zero_()
                        running_steps = 0

            if self.dist_env.is_main:
                logger.info(
                    "Epoch %s done: total_batches_seen=%s global_step=%s",
                    epoch,
                    batch_idx + 1,
                    self.runtime.global_step,
                )

            eval_metrics = self._run_eval()
            if eval_metrics is not None and self.dist_env.is_main:
                logger.info(
                    "epoch=%s val_loss=%.6f val_acc=%.6f",
                    epoch,
                    eval_metrics[0].item(),
                    eval_metrics[1].item(),
                )
            ckpt_name = f"epoch_{epoch:02d}_step_{self.runtime.global_step:06d}"
            self._save_checkpoint(ckpt_name)
            if self.dist_env.is_main:
                logger.info("Saved checkpoint: %s", self.output_dir / ckpt_name)

        if self.dist_env.is_main:
            logger.info("Training complete: global_step=%s", self.runtime.global_step)


def main(config_path=None):
    """Main entry point for the EAGLE-3 recipe."""
    if config_path is None:
        raise ValueError("config_path is required for TrainEagle3Recipe")
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainEagle3Recipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
