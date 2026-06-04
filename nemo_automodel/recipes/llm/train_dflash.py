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

"""DFlash draft-model training recipe (Qwen3-style targets).

DFlash drafts a whole block of tokens in parallel via MASK-token denoising
conditioned on the frozen target's hidden states (see
``nemo_automodel.components.speculative.dflash``). This recipe mirrors the EAGLE
recipes' scaffolding -- online target hidden-state capture, gradient
accumulation with a trailing-window flush, and the same checkpointer plumbing --
but trains the DFlash draft with its block-wise cross-entropy objective.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
from huggingface_hub import constants as hf_constants
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

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
from nemo_automodel.components.speculative.dflash.core import DFlashTrainerModule, NoValidAnchorsError
from nemo_automodel.components.speculative.dflash.draft_qwen3 import build_target_layer_ids
from nemo_automodel.components.speculative.dflash.registry import resolve_dflash_draft_spec
from nemo_automodel.components.speculative.dflash.target import HFDFlashTargetModel
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.recipes.base_recipe import (
    BaseRecipe,
    _find_latest_checkpoint,
    _is_checkpoint_model_config_compatible,
    _resolve_restore_from_to_ckpt_dir,
)

logger = logging.getLogger(__name__)


def _optim_steps_per_epoch(num_batches_per_epoch: int, grad_accumulation_steps: int) -> int:
    """Return ceil(num_batches / accum), the actual number of optimizer steps per epoch."""
    if num_batches_per_epoch <= 0 or grad_accumulation_steps <= 0:
        return 0
    return -(-num_batches_per_epoch // grad_accumulation_steps)


def _all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


class TrainDFlashRecipe(BaseRecipe):
    """Recipe for DFlash draft-model training on Qwen3-style dense / MoE targets."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        """Build the target model, DFlash draft, data, optimizer, and trainer module."""
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
        draft_spec = resolve_dflash_draft_spec(architectures)

        self.tokenizer = NeMoAutoTokenizer.from_pretrained(
            target_path, trust_remote_code=recipe_cfg.get("trust_remote_code", False)
        )
        self.compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.target_model = NeMoAutoModelForCausalLM.from_pretrained(
            target_path,
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
            torch_dtype=self.compute_dtype,
            force_hf=bool(recipe_cfg.get("target_force_hf", False)),
        )
        self.target_model.to(self.device)
        self.target_model.requires_grad_(False)

        # Resolve the captured target layers once and share them between the
        # target wrapper (what to capture) and the draft config (the ``fc`` input
        # width) so the two never disagree.
        num_target_layers = int(target_config.num_hidden_layers)
        draft_num_hidden_layers = int(recipe_cfg.get("draft_num_hidden_layers", 5))
        target_layer_ids = list(
            recipe_cfg.get("target_layer_ids", None)
            or build_target_layer_ids(num_target_layers, draft_num_hidden_layers)
        )
        self.target_wrapper = HFDFlashTargetModel(self.target_model, target_layer_ids=target_layer_ids)

        self.block_size = int(recipe_cfg.get("block_size", 16))
        self.mask_token_id = self._resolve_mask_token_id(recipe_cfg)

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

        # DFlash draft config: a small non-causal Qwen3 stack that reuses the
        # target's architecture defaults (head_dim, rope_theta, rms_norm_eps, ...).
        draft_config = target_config.to_dict()
        draft_config["architectures"] = ["Qwen3DFlashDraftModel"]
        draft_config["num_hidden_layers"] = draft_num_hidden_layers
        # ``layer_types``/``max_window_layers`` are sized to the target's depth;
        # rebuild them for the (shallower) draft. The DFlash attention never uses
        # sliding windows, so every draft layer is full attention.
        draft_config["layer_types"] = ["full_attention"] * draft_num_hidden_layers
        draft_config["max_window_layers"] = draft_num_hidden_layers
        draft_config["num_target_layers"] = num_target_layers
        draft_config["block_size"] = self.block_size
        draft_config["dflash_config"] = {
            "mask_token_id": self.mask_token_id,
            "target_layer_ids": target_layer_ids,
        }
        # A single knob drives both the trainer's mask format and the draft's
        # attention function -- they must agree (a flex BlockMask only works with
        # the flex attention fn, a dense bool mask only with sdpa/eager).
        attention_backend = recipe_cfg.get("attention_backend", "flex_attention")
        draft_config_obj = Qwen3Config.from_dict(draft_config)
        draft_config_obj._attn_implementation = attention_backend
        self.draft_model = draft_spec.draft_cls(draft_config_obj).to(device=self.device, dtype=self.compute_dtype)

        trainer_module = DFlashTrainerModule(
            draft_model=self.draft_model,
            target_lm_head=self.target_model.get_output_embeddings(),
            target_embed_tokens=self.target_model.get_input_embeddings(),
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            attention_backend=attention_backend,
            num_anchors=int(recipe_cfg.get("num_anchors", 512)),
            loss_decay_gamma=recipe_cfg.get("loss_decay_gamma", None),
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
        # Checkpoint cadence (independent knobs; the fully-trained model is always
        # saved at the end). ``ckpt_every_steps`` saves every N optimizer steps;
        # ``save_checkpoint_every_epoch`` saves at each epoch boundary (off by default).
        self.ckpt_every_steps = recipe_cfg.get("ckpt_every_steps", None)
        self.save_checkpoint_every_epoch = recipe_cfg.get("save_checkpoint_every_epoch", False)
        self.output_dir = pathlib.Path(recipe_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            num_batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            num_batches_per_epoch = 0
        total_optim_steps = max(
            1, self.num_epochs * _optim_steps_per_epoch(num_batches_per_epoch, self.grad_accumulation_steps)
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
        self.runtime = SimpleNamespace(global_step=0)
        self._resume_epoch = 0
        self._skipped_micro_batches = 0

        self.rng = StatefulRNG(seed=int(recipe_cfg.get("shuffle_seed", 42)), ranked=self.dist_env.world_size > 1)
        self._build_checkpointer(target_path)
        self.load_checkpoint(self.cfg.get("checkpoint.restore_from", None))

    def _resolve_mask_token_id(self, recipe_cfg) -> int:
        """Resolve the MASK token id used to fill non-anchor block positions."""
        mask_token_id = recipe_cfg.get("mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if mask_token_id is not None:
                logger.warning(
                    "recipe_args.mask_token_id not set; falling back to tokenizer.pad_token_id=%d. "
                    "Set it explicitly (e.g. a dedicated reserved token) if this is not intended.",
                    mask_token_id,
                )
        if mask_token_id is None:
            raise ValueError(
                "DFlash requires a mask_token_id: set recipe_args.mask_token_id (the token used for "
                "non-anchor block positions), or ensure the tokenizer defines a pad_token_id."
            )
        return int(mask_token_id)

    def _build_checkpointer(self, target_path: str) -> None:
        """Build the checkpointer using the same plumbing as the EAGLE recipes."""
        ckpt_cfg = self.cfg.get("checkpoint", None)
        default_dir = str(self.output_dir / "checkpoints")
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
            config=self.checkpoint_config, dp_rank=dp_rank, tp_rank=0, pp_rank=0, moe_mesh=None
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
        """Persist the DFlash draft model, optimizer, scheduler, RNG, and meta."""
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
        self.checkpointer.save_model(draft_model, path, tokenizer=self.tokenizer, is_final_checkpoint=False)
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
        """Persist DFlash meta: global_step, epoch, block_size, and target layers."""
        torch.save(
            {
                "global_step": self.runtime.global_step,
                "epoch": int(epoch),
                "block_size": self.block_size,
                "mask_token_id": self.mask_token_id,
                "target_layer_ids": list(self.target_wrapper.target_layer_ids),
            },
            os.path.join(path, "dflash_meta.pt"),
        )

    def load_checkpoint(self, restore_from: str | None = None) -> None:
        """Restore the DFlash draft model, optimizer, scheduler, RNG, and global_step."""
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
        if not ok and not restore_from:
            if is_rank_0:
                logger.warning(
                    "Auto-detected checkpoint at %s is incompatible: %s. Skipping restore.", ckpt_dir, reason
                )
            return

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
        """Restore DFlash meta: global_step and epoch."""
        meta_path = os.path.join(ckpt_dir, "dflash_meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, weights_only=False, map_location="cpu")
            self.runtime.global_step = int(meta.get("global_step", 0))
            self._resume_epoch = int(meta.get("epoch", 0))

    def _log_saved_checkpoint(self, kind: str, epoch: int, step: int) -> None:
        """Log a saved checkpoint on rank 0 when checkpointing is enabled."""
        ckpt_cfg = getattr(self, "checkpoint_config", None)
        if self.dist_env.is_main and ckpt_cfg is not None and ckpt_cfg.enabled:
            logger.info("Saved %s checkpoint to %s/epoch_%d_step_%d", kind, ckpt_cfg.checkpoint_dir, epoch, step)

    def _maybe_save_step_checkpoint(self, epoch: int) -> bool:
        """Save a checkpoint mid-epoch when ``ckpt_every_steps`` is configured."""
        every = getattr(self, "ckpt_every_steps", None)
        if every is None or every <= 0 or self.runtime.global_step % every != 0:
            return False
        self.save_checkpoint(epoch=epoch, step=self.runtime.global_step, best_metric_key="val_loss")
        self._log_saved_checkpoint("step", epoch, self.runtime.global_step)
        return True

    def _maybe_save_final_checkpoint(self, completed_epochs: int) -> bool:
        """Always save the fully-trained model at the end, unless a cadence already saved the final step."""
        gs = self.runtime.global_step
        if gs <= 0:
            return False
        every = getattr(self, "ckpt_every_steps", None)
        saved_by_step = bool(every and every > 0 and gs % every == 0)
        saved_by_epoch = bool(getattr(self, "save_checkpoint_every_epoch", False))
        if saved_by_step or saved_by_epoch:
            return False
        self.save_checkpoint(epoch=completed_epochs, step=gs, best_metric_key="val_loss")
        self._log_saved_checkpoint("final", completed_epochs, gs)
        return True

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
                    hidden_states=target_batch.hidden_states,
                    loss_mask=target_batch.loss_mask,
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
        """Run the DFlash training loop."""
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
            pending_micro_batches = 0
            completed_steps = 0
            last_batch_idx = -1
            num_batches = len(self.train_dataloader)
            is_ddp = isinstance(self.trainer_module, DistributedDataParallel)
            for batch_idx, batch in enumerate(self.train_dataloader):
                last_batch_idx = batch_idx
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                target_batch = self.target_wrapper.generate_batch(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    loss_mask=batch["loss_mask"],
                )
                is_window_close = (pending_micro_batches + 1 == self.grad_accumulation_steps) or (
                    batch_idx + 1 == num_batches
                )
                sync_ctx = nullcontext() if (not is_ddp or is_window_close) else self.trainer_module.no_sync()
                try:
                    with sync_ctx:
                        metrics = self.trainer_module(
                            input_ids=target_batch.input_ids,
                            hidden_states=target_batch.hidden_states,
                            loss_mask=target_batch.loss_mask,
                        )
                        loss = metrics.loss / self.grad_accumulation_steps
                        loss.backward()
                except NoValidAnchorsError:
                    # Every sample in this micro-batch is too short to form a block;
                    # nothing to learn from it. Skip without touching the
                    # accumulation counters. (No backward ran, so grads are intact.)
                    # NOTE: under DDP this skip is per-rank and data-dependent; pre-filter
                    # short samples for multi-rank runs to keep optimizer steps in lockstep.
                    self._skipped_micro_batches += 1
                    continue

                running_loss += metrics.loss.detach().item()
                running_acc += metrics.accuracy.detach().item()
                epoch_loss += metrics.loss.detach().item()
                micro_step += 1
                pending_micro_batches += 1

                if pending_micro_batches == self.grad_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()
                    self.runtime.global_step += 1
                    completed_steps += 1
                    pending_micro_batches = 0
                    self._maybe_save_step_checkpoint(epoch_idx)

                    if self.dist_env.is_main and self.runtime.global_step % self.log_every_steps == 0:
                        logger.info(
                            "epoch=%d step=%d loss=%.4f acc=%.4f lr=%.6g",
                            epoch_idx,
                            self.runtime.global_step,
                            running_loss / self.log_every_steps,
                            running_acc / self.log_every_steps,
                            self.lr_scheduler.get_last_lr()[0],
                        )
                        running_loss = 0.0
                        running_acc = 0.0

            # Flush the trailing partial accumulation window (see EAGLE recipes
            # for the rescale rationale).
            if pending_micro_batches > 0:
                scale = float(self.grad_accumulation_steps) / float(pending_micro_batches)
                for p in self.trainer_module.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()
                self.runtime.global_step += 1
                completed_steps += 1
                pending_micro_batches = 0
                self._maybe_save_step_checkpoint(epoch_idx)

            eval_metrics = self._run_eval()
            if self.dist_env.is_main:
                msg = (
                    f"Finished epoch {epoch_idx + 1}/{self.num_epochs} completed_steps={completed_steps} "
                    f"skipped_short_micro_batches={self._skipped_micro_batches}"
                )
                if eval_metrics is not None:
                    msg += f" val_loss={eval_metrics['val_loss']:.4f} val_accuracy={eval_metrics['val_accuracy']:.4f}"
                logger.info(msg)

            if getattr(self, "save_checkpoint_every_epoch", False) and last_batch_idx >= 0:
                avg_loss = epoch_loss / max(1, micro_step) if micro_step else None
                self.save_checkpoint(
                    epoch=epoch_idx + 1,
                    step=self.runtime.global_step,
                    train_loss=avg_loss,
                    val_loss=eval_metrics,
                    best_metric_key="val_loss",
                )
                self._log_saved_checkpoint("epoch", epoch_idx + 1, self.runtime.global_step)

        self._maybe_save_final_checkpoint(self.num_epochs)


def main(config_path: str | None = None):
    """Entrypoint for ``TrainDFlashRecipe``."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainDFlashRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
