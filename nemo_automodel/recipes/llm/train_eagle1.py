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

import json
import logging
import math
import os
import pathlib
from types import SimpleNamespace

import torch
import torch.distributed as dist
from huggingface_hub import constants as hf_constants
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig, LlamaConfig

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    save_config,
)
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.eagle3 import build_eagle3_dataloader
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.speculative.eagle.core_v12 import EagleTrainerModule
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import LlamaEagleDraftModel
from nemo_automodel.components.speculative.eagle.target_v12 import HFEagleTargetModel
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.recipes.base_recipe import (
    BaseRecipe,
    _find_latest_checkpoint,
    _is_checkpoint_model_config_compatible,
    _resolve_restore_from_to_ckpt_dir,
)

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
        self._resume_epoch = 0

        self.rng = StatefulRNG(
            seed=int(recipe_cfg.get("shuffle_seed", 42)),
            ranked=self.dist_env.world_size > 1,
        )
        self._build_checkpointer(target_path)
        self.load_checkpoint(self.cfg.get("checkpoint.restore_from", None))

    def _build_checkpointer(self, target_path: str) -> None:
        """Build the checkpointer using the same plumbing as the standard recipes."""
        ckpt_cfg = self.cfg.get("checkpoint", None)
        default_dir = str(self.output_dir / "checkpoints")
        # EAGLE recipes construct the draft model directly and bypass
        # `apply_model_infrastructure`, which is where `_pre_shard_hf_state_dict_keys`
        # would normally be attached. Capture the pre-shard keys here so the
        # consolidated-safetensors path in `_maybe_build_consolidated_index`
        # has something to diff against instead of `None`.
        draft_state_dict_keys = list(self.draft_model.state_dict().keys())
        ckpt_kwargs = dict(
            enabled=True,
            checkpoint_dir=default_dir,
            model_save_format="safetensors",
            model_repo_id=str(target_path),
            model_cache_dir=hf_constants.HF_HUB_CACHE,
            save_consolidated=True,
            is_peft=False,
            model_state_dict_keys=draft_state_dict_keys,
        )
        if ckpt_cfg is not None:
            user_cfg = ckpt_cfg.to_dict() if hasattr(ckpt_cfg, "to_dict") else dict(ckpt_cfg)
            user_cfg.pop("restore_from", None)
            ckpt_kwargs.update(user_cfg)
        if ckpt_kwargs.get("model_state_dict_keys") is None:
            ckpt_kwargs["model_state_dict_keys"] = draft_state_dict_keys

        self.checkpoint_config = CheckpointingConfig(**ckpt_kwargs)
        dp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.checkpointer = Checkpointer(
            config=self.checkpoint_config,
            dp_rank=dp_rank,
            tp_rank=0,
            pp_rank=0,
            moe_mesh=None,
        )

    def _module(self):
        return (
            self.trainer_module.module
            if isinstance(self.trainer_module, DistributedDataParallel)
            else self.trainer_module
        )

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        train_loss: float | None = None,
        val_loss: dict[str, float] | None = None,
        best_metric_key: str = "default",
    ) -> None:
        """Persist draft model, optimizer, scheduler, RNG, and EAGLE meta.

        Overrides ``BaseRecipe.save_checkpoint`` because EAGLE recipes hold multiple
        ``nn.Module`` attributes (frozen target, target wrapper, trainer module wrapping
        the draft) — only ``draft_model`` should be persisted as the main model.
        """
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None or not checkpointer.config.enabled:
            return
        self.checkpointer.async_wait()

        prev_pending = getattr(self, "_last_pending_checkpoint_dir", None)
        prev_best_pending = getattr(self, "_last_pending_best_checkpoint_info", None)

        ckpt_root = self.checkpoint_config.checkpoint_dir
        path = os.path.join(str(ckpt_root), f"epoch_{epoch}_step_{step}")
        is_dist_initialized = dist.is_initialized()
        is_rank_0 = (not is_dist_initialized) or dist.get_rank() == 0
        best_val_metric = (
            val_loss.get(next(iter(val_loss.keys())) if len(val_loss) == 1 else best_metric_key) if val_loss else None
        )

        if prev_pending is not None:
            if is_rank_0:
                self._update_latest_symlink(prev_pending)
            setattr(self, "_last_pending_checkpoint_dir", None)
            if is_dist_initialized:
                dist.barrier()

        if prev_best_pending is not None:
            if is_rank_0 and prev_best_pending.get("val") is not None:
                self._update_best_symlink(prev_best_pending["path"], float(prev_best_pending["val"]))
            setattr(self, "_last_pending_best_checkpoint_info", None)
            if is_dist_initialized:
                dist.barrier()

        if is_rank_0:
            if os.path.exists(path):
                raise FileExistsError(f"Checkpoint directory {path} already exists")
            os.makedirs(path, exist_ok=True)
            loss_dict: dict[str, float] = {}
            if train_loss is not None:
                loss_dict["train_loss"] = float(train_loss)
            if val_loss:
                for k, v in val_loss.items():
                    loss_dict[k] = float(v)
            if loss_dict:
                with open(os.path.join(path, "losses.json"), "w") as f:
                    json.dump(loss_dict, f)
        if is_dist_initialized:
            dist.barrier()

        draft_model = self._module().draft_model
        self.checkpointer.save_model(draft_model, path, tokenizer=self.tokenizer)
        self.checkpointer.save_optimizer(self.optimizer, draft_model, path, self.lr_scheduler)
        self.checkpointer.save_on_dp_ranks(self.rng, "rng", path)

        if is_rank_0:
            self._save_extra_state(path, epoch=epoch)
            try:
                save_config(self.cfg.raw_config, path)
            except (AttributeError, OSError) as e:
                logger.warning("Failed to save config snapshot: %s", e)
        if is_dist_initialized:
            dist.barrier()

        if getattr(self.checkpointer.config, "is_async", False):
            setattr(self, "_last_pending_checkpoint_dir", path)
            if best_val_metric is not None:
                setattr(self, "_last_pending_best_checkpoint_info", {"path": path, "val": float(best_val_metric)})
        else:
            if is_rank_0:
                self._update_latest_symlink(path)
                if best_val_metric is not None:
                    self._update_best_symlink(path, float(best_val_metric))
            if is_dist_initialized:
                dist.barrier()

    def _save_extra_state(self, path: str, epoch: int) -> None:
        """Persist EAGLE-recipe-specific scalars. Subclasses extend this."""
        torch.save(
            {"global_step": self.runtime.global_step, "epoch": int(epoch)},
            os.path.join(path, "eagle_meta.pt"),
        )

    def load_checkpoint(self, restore_from: str | None = None) -> None:
        """Resolve and restore a checkpoint produced by ``save_checkpoint``.

        Restores the draft model, optimizer, LR scheduler, RNG, and ``global_step``.
        Target model weights are NOT restored — they are re-loaded from the HF hub on
        each run because the target is frozen.
        """
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None or not checkpointer.config.enabled:
            return
        is_rank_0 = (not dist.is_initialized()) or dist.get_rank() == 0
        ckpt_root = self.checkpoint_config.checkpoint_dir

        if restore_from:
            ckpt_dir = _resolve_restore_from_to_ckpt_dir(ckpt_root, restore_from)
            if ckpt_dir is None:
                if is_rank_0:
                    logger.warning("restore_from='LATEST' but no checkpoint found in %s", ckpt_root)
                return
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")
        else:
            auto = _find_latest_checkpoint(ckpt_root)
            if auto is None:
                return
            ckpt_dir = str(auto)

        ok, reason = _is_checkpoint_model_config_compatible(self.cfg, ckpt_dir)
        if not ok:
            if not restore_from:
                if is_rank_0:
                    logger.warning(
                        "Auto-detected checkpoint at %s is incompatible with current model configuration: %s. "
                        "Skipping restore.",
                        ckpt_dir,
                        reason,
                    )
                return
            if is_rank_0:
                logger.warning(
                    "Checkpoint at %s may be incompatible with current model configuration: %s. "
                    "Proceeding with restore anyway.",
                    ckpt_dir,
                    reason,
                )

        if is_rank_0:
            logger.info("Resuming from checkpoint: %s", ckpt_dir)

        draft_model = self._module().draft_model
        self.checkpointer.load_model(draft_model, os.path.join(ckpt_dir, "model"))
        self.checkpointer.load_optimizer(self.optimizer, draft_model, ckpt_dir, self.lr_scheduler)
        try:
            self.checkpointer.load_on_dp_ranks(self.rng, "rng", ckpt_dir)
        except FileNotFoundError:
            logger.warning("RNG state not found in %s; continuing without restoring RNG.", ckpt_dir)

        self._load_extra_state(ckpt_dir)

    def _load_extra_state(self, ckpt_dir: str) -> None:
        """Restore EAGLE-recipe-specific scalars. Subclasses extend this."""
        meta_path = os.path.join(ckpt_dir, "eagle_meta.pt")
        if not os.path.exists(meta_path):
            legacy = os.path.join(ckpt_dir, "eagle1_meta.pt")
            meta_path = legacy if os.path.exists(legacy) else meta_path
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, weights_only=False, map_location="cpu")
            self.runtime.global_step = int(meta.get("global_step", 0))
            self._resume_epoch = int(meta.get("epoch", 0))

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
        start_epoch = max(0, int(getattr(self, "_resume_epoch", 0)))
        if start_epoch >= self.num_epochs:
            if self.dist_env.is_main:
                logger.info("All %d epochs already completed; nothing to do.", self.num_epochs)
            return
        for epoch_idx in range(start_epoch, self.num_epochs):
            if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch_idx)

            running_loss = 0.0
            running_acc = 0.0
            epoch_loss = 0.0
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
                epoch_loss += metrics.loss.detach().item()
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

            if last_batch_idx >= 0:
                avg_loss = epoch_loss / max(1, micro_step) if micro_step else None
                self.save_checkpoint(
                    epoch=epoch_idx + 1,
                    step=self.runtime.global_step,
                    train_loss=avg_loss,
                    val_loss=eval_metrics,
                    best_metric_key="val_loss",
                )


def main(config_path: str | None = None):
    """Entrypoint for ``TrainEagle1Recipe``."""
    if config_path is None:
        raise ValueError("config_path is required for TrainEagle1Recipe")
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainEagle1Recipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()
