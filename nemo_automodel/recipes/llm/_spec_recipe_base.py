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

"""Shared base for the speculative-decoding draft recipes (EAGLE, DFlash, DSpark).

Every draft recipe hand-rolls its own training loop around a frozen target and a
trainable draft, but they persist and restore that draft the same way, step the
same warmup + cosine LR schedule, and follow the same checkpoint cadence. This
base owns that shared lifecycle so a fix lands in one place instead of drifting
across four copies. Model construction, the training loop, and evaluation stay
in each recipe.
"""

from __future__ import annotations

import logging
import os
import pathlib
from types import SimpleNamespace
from typing import ClassVar

import torch
import torch.distributed as dist
from huggingface_hub import constants as hf_constants
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components.checkpoint.checkpointing import (
    CheckpointingConfig,
    load_torch_ckpt,
    save_config,
    save_losses,
)
from nemo_automodel.components.checkpoint.utils import find_latest_checkpoint, resolve_restore_from_to_checkpoint_dir
from nemo_automodel.components.loggers.wandb_utils import init_wandb_run, suppress_wandb_log_messages
from nemo_automodel.recipes.base_recipe import BaseRecipe, _is_checkpoint_model_config_compatible
from nemo_automodel.recipes.llm._spec_train_utils import (
    make_warmup_cosine_schedule,
    optim_steps_per_epoch,
    resolve_wandb_kwargs,
    resolve_warmup_steps,
)

logger = logging.getLogger(__name__)


def _step_cadence_hit(every: int | None, step: int) -> bool:
    """Return True when ``every`` (``ckpt_every_steps``) is a positive integer dividing ``step``."""
    return bool(every and every > 0 and step % every == 0)


class SpecDecodeRecipeBase(BaseRecipe):
    """Checkpoint, LR-schedule, and logging lifecycle shared by the draft recipes.

    Subclasses build ``self.draft_model`` and ``self.trainer_module`` (which wraps
    the draft as ``.draft_model``, optionally inside ``DistributedDataParallel``),
    then call :meth:`_setup_training_state` and :meth:`_build_checkpointer` from
    ``setup()``. They persist their recipe-specific metadata through
    :meth:`_save_extra_state` / :meth:`_load_extra_state`.
    """

    # Which ranks own the RNG checkpoint file. The EAGLE recipes key it on the
    # global rank, so they keep that layout to stay resume-compatible with their
    # existing checkpoints; the block-diffusion recipes key it on the dp rank so
    # context-parallel peers (which share a dp rank and RNG state) do not race.
    rng_checkpoint_on_dp_ranks: ClassVar[bool] = True
    # Floor on the ratio-derived LR warmup length (see ``resolve_warmup_steps``).
    min_warmup_steps: ClassVar[int] = 1

    def _build_draft_optimizer(self, opt_cfg) -> torch.optim.Optimizer:
        """Build the draft optimizer (AdamW over the trainable parameters)."""
        return torch.optim.AdamW(
            [p for p in self.trainer_module.parameters() if p.requires_grad],
            lr=self.peak_lr,
            betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )

    def _setup_training_state(self, recipe_cfg) -> None:
        """Build the optimizer and LR schedule, read the loop / checkpoint cadence knobs,
        create ``output_dir``, and reset the step counters.

        Runs once ``self.trainer_module`` and ``self.train_dataloader`` exist.

        The two checkpoint cadence knobs are independent: ``ckpt_every_steps`` saves
        every N optimizer steps (None or <=0 disables it) and
        ``save_checkpoint_every_epoch`` saves at each epoch boundary. The
        fully-trained model is always saved once the run completes; these only add
        intermediate checkpoints. Field names mirror ``StepScheduler``.

        Args:
            recipe_cfg: The ``recipe_args`` config node.
        """
        opt_cfg = self.cfg.optimizer
        self.peak_lr = float(opt_cfg.lr)
        self.optimizer = self._build_draft_optimizer(opt_cfg)
        self.grad_accumulation_steps = recipe_cfg.get("grad_accumulation_steps", 1)
        self.max_grad_norm = recipe_cfg.get("max_grad_norm", 1.0)
        self.num_epochs = recipe_cfg.num_epochs
        self.log_every_steps = recipe_cfg.get("log_every_steps", 10)
        self.ckpt_every_steps = recipe_cfg.get("ckpt_every_steps", None)
        self.save_checkpoint_every_epoch = recipe_cfg.get("save_checkpoint_every_epoch", False)
        self.output_dir = pathlib.Path(recipe_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.num_batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            self.num_batches_per_epoch = 0
        # Ceil division counts a trailing partial accumulation window as a real
        # optimizer step: the loop flushes that window at the end of each epoch, so
        # the schedule must cover it or the final epoch trains at ``min_lr_ratio``.
        self.total_optim_steps = max(
            1, self.num_epochs * optim_steps_per_epoch(self.num_batches_per_epoch, self.grad_accumulation_steps)
        )
        self.warmup_steps = resolve_warmup_steps(
            float(opt_cfg.get("warmup_ratio", 0.05)), self.total_optim_steps, self.min_warmup_steps
        )
        self.min_lr_ratio = float(opt_cfg.get("min_lr_ratio", 0.1))
        # Warmup + cosine: drafts trained from scratch diverge under a flat LR
        # once AdamW's second-moment estimates have settled.
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, make_warmup_cosine_schedule(self.warmup_steps, self.total_optim_steps, self.min_lr_ratio)
        )
        self.runtime = SimpleNamespace(global_step=0)
        self._resume_epoch = 0

    def _init_wandb_run(self, default_name: str) -> None:
        """Start the optional rank-0 W&B run from the top-level ``wandb:`` block.

        No run is started off rank 0, without a ``wandb:`` block, or when the block
        sets ``enable: false`` (see ``resolve_wandb_kwargs``).
        """
        self.wandb_run = None
        wandb_cfg = self.cfg.get("wandb", None)
        if not self.dist_env.is_main or wandb_cfg is None:
            return
        wandb_kwargs = resolve_wandb_kwargs(wandb_cfg.to_dict())
        if wandb_kwargs is None:
            return
        suppress_wandb_log_messages()
        self.wandb_run = init_wandb_run(wandb_kwargs, self.cfg.to_dict(), default_name=default_name)

    def _build_checkpointer(self, target_path: str) -> None:
        """Build the draft checkpointer using the same plumbing as the standard recipes."""
        ckpt_cfg = self.cfg.get("checkpoint", None)
        default_dir = str(self.output_dir / "checkpoints")
        # The draft is built directly and bypasses ``apply_model_infrastructure``,
        # which is where ``_pre_shard_hf_state_dict_keys`` would normally be
        # attached. Capture the keys here so the consolidated-safetensors export has
        # something to diff against instead of ``None``.
        draft_state_dict_keys = list(self.draft_model.state_dict().keys())
        # LoRA drafts save/load adapter-only checkpoints; the consolidated
        # full-draft export does not apply to them.
        is_peft = getattr(self, "peft_config", None) is not None
        ckpt_kwargs = dict(
            enabled=True,
            checkpoint_dir=default_dir,
            model_save_format="safetensors",
            model_repo_id=str(target_path),
            model_cache_dir=hf_constants.HF_HUB_CACHE,
            save_consolidated=not is_peft,
            is_peft=is_peft,
            model_state_dict_keys=draft_state_dict_keys,
        )
        if ckpt_cfg is not None:
            user_cfg = ckpt_cfg.to_dict() if hasattr(ckpt_cfg, "to_dict") else dict(ckpt_cfg)
            user_cfg.pop("restore_from", None)
            ckpt_kwargs.update(user_cfg)
        if ckpt_kwargs.get("model_state_dict_keys") is None:
            ckpt_kwargs["model_state_dict_keys"] = draft_state_dict_keys

        self.checkpoint_config = CheckpointingConfig(**ckpt_kwargs)
        # The draft is never tp- or cp-sharded (it is replicated, or FSDP2-sharded only
        # over dp), so key the checkpoint shard on the dp coordinate, which tp/cp peers
        # share, rather than the global rank. Without a mesh the global rank is used.
        dp_mesh = getattr(self, "dp_mesh", None)
        dp_rank = dp_mesh.get_local_rank() if dp_mesh is not None else (dist.get_rank() if dist.is_initialized() else 0)
        self.checkpointer = self.checkpoint_config.build(dp_rank=dp_rank, tp_rank=0, pp_rank=0, moe_mesh=None)
        self._log_checkpoint_retention_policy(self.checkpoint_config)

    def _module(self):
        """Return the trainer module, unwrapped from DDP."""
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
        is_final_checkpoint: bool = False,
    ) -> None:
        """Persist the draft model, optimizer, scheduler, RNG, and recipe meta.

        Overrides ``BaseRecipe.save_checkpoint`` because these recipes hold several
        ``nn.Module`` attributes (frozen target, target wrapper, trainer module
        wrapping the draft) and only the draft is persisted as the main model.

        ``is_final_checkpoint`` is computed by the caller (the hand-rolled loops have
        no ``step_scheduler`` for the checkpointer to infer it from);
        ``save_consolidated: final`` exports HF safetensors only when it is True.
        """
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None or not checkpointer.config.enabled:
            return
        checkpointer.async_wait()
        checkpointer.lifecycle.complete_pending()

        ckpt_root = self.checkpoint_config.checkpoint_dir
        path = os.path.join(str(ckpt_root), f"epoch_{epoch}_step_{step}")
        is_dist_initialized = dist.is_initialized()
        is_rank_0 = (not is_dist_initialized) or dist.get_rank() == 0
        best_metric_name = next(iter(val_loss.keys())) if val_loss and len(val_loss) == 1 else best_metric_key
        best_val_metric = val_loss.get(best_metric_name) if val_loss else None

        checkpointer.lifecycle.reserve(path)

        if is_rank_0:
            loss_dict: dict[str, float] = {}
            if train_loss is not None:
                loss_dict["train_loss"] = float(train_loss)
            if val_loss:
                for k, v in val_loss.items():
                    loss_dict[k] = float(v)
            if loss_dict:
                save_losses(loss_dict, path)
        if is_dist_initialized:
            dist.barrier()

        draft_model = self._module().draft_model
        checkpointer.save_model(
            draft_model,
            path,
            peft_config=getattr(self, "peft_config", None),
            tokenizer=self.tokenizer,
            is_final_checkpoint=is_final_checkpoint,
        )
        self._on_draft_saved(draft_model, path, is_final_checkpoint=is_final_checkpoint, is_rank_0=is_rank_0)
        checkpointer.save_optimizer(self.optimizer, draft_model, path, self.lr_scheduler)
        if self.rng_checkpoint_on_dp_ranks:
            # cp peers share a dp_rank and, being seeded per dp_rank, hold identical
            # RNG state, so every peer would torch.save the same rng_dp_rank_N.pt and
            # race on a shared FS; let only the first cp peer write it.
            cp_mesh = getattr(self, "cp_mesh", None)
            if cp_mesh is None or cp_mesh.get_local_rank() == 0:
                checkpointer.save_on_dp_ranks(self.rng, "rng", path)
        else:
            checkpointer.save_on_global_ranks(self.rng, "rng", path)

        # Rank-0 writes followed by collectives, so they go through the same guard:
        # a failure here must abort every rank rather than only this one.
        def write_recipe_metadata() -> None:
            self._save_extra_state(path, epoch=epoch)
            try:
                save_config(self.cfg.raw_config, path)
            except (AttributeError, OSError) as e:
                logger.warning("Failed to save config snapshot: %s", e)

        checkpointer.lifecycle.run_coordinator_step(
            write_recipe_metadata,
            description=f"write recipe metadata to {path}",
        )
        if is_dist_initialized:
            dist.barrier()

        best = float(best_val_metric) if best_val_metric is not None else None
        if getattr(checkpointer.config, "is_async", False):
            checkpointer.lifecycle.defer_publication(path, best_val_metric=best, metric_key=best_metric_name)
        else:
            checkpointer.lifecycle.publish(path, best_val_metric=best, metric_key=best_metric_name)

    def _on_draft_saved(self, draft_model, path: str, *, is_final_checkpoint: bool, is_rank_0: bool) -> None:
        """Hook run right after the draft weights are written (e.g. an extra export)."""

    def _save_extra_state(self, path: str, epoch: int) -> None:
        """Persist the recipe-specific metadata (``global_step``, ``epoch``, ...) under ``path``."""
        raise NotImplementedError

    def _load_extra_state(self, ckpt_dir: str) -> None:
        """Restore the recipe-specific metadata written by :meth:`_save_extra_state`."""
        raise NotImplementedError

    def _save_meta(self, path: str, filename: str, epoch: int, **extra) -> None:
        """Write ``global_step``, ``epoch``, and the recipe's ``extra`` fields to ``path/filename``."""
        torch.save(
            {"global_step": self.runtime.global_step, "epoch": int(epoch), **extra}, os.path.join(path, filename)
        )

    def _load_meta(self, ckpt_dir: str, filename: str, legacy_filename: str | None = None) -> dict | None:
        """Load ``ckpt_dir/filename`` (or its legacy name) and restore ``global_step`` and the resume epoch.

        Returns:
            The loaded meta dict, or None when the checkpoint carries no meta file.
        """
        meta_path = os.path.join(ckpt_dir, filename)
        if legacy_filename is not None and not os.path.exists(meta_path):
            meta_path = os.path.join(ckpt_dir, legacy_filename)
        if not os.path.exists(meta_path):
            return None
        meta = load_torch_ckpt(
            meta_path,
            map_location="cpu",
            weights_only=not self.checkpoint_config.allow_legacy_pickle_restore,
        )
        self.runtime.global_step = int(meta.get("global_step", 0))
        self._resume_epoch = int(meta.get("epoch", 0))
        return meta

    def load_checkpoint(self, restore_from: str | None = None) -> None:
        """Resolve and restore a checkpoint produced by :meth:`save_checkpoint`.

        Restores the draft model, optimizer, LR scheduler, RNG, and the recipe meta.
        Target weights are not restored: the target is frozen and is re-loaded from
        its source on each run.
        """
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None or not checkpointer.config.enabled:
            return
        is_rank_0 = (not dist.is_initialized()) or dist.get_rank() == 0
        ckpt_root = self.checkpoint_config.checkpoint_dir

        if restore_from:
            ckpt_dir = resolve_restore_from_to_checkpoint_dir(ckpt_root, restore_from)
            if ckpt_dir is None:
                if is_rank_0:
                    logger.warning("restore_from='LATEST' but no checkpoint found in %s", ckpt_root)
                return
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")
        else:
            auto = find_latest_checkpoint(ckpt_root)
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
        checkpointer.load_model(draft_model, os.path.join(ckpt_dir, "model"))
        checkpointer.load_optimizer(self.optimizer, draft_model, ckpt_dir, self.lr_scheduler)
        try:
            if self.rng_checkpoint_on_dp_ranks:
                checkpointer.load_on_dp_ranks(self.rng, "rng", ckpt_dir)
            else:
                checkpointer.load_on_global_ranks(self.rng, "rng", ckpt_dir)
        except FileNotFoundError:
            logger.warning("RNG state not found in %s; continuing without restoring RNG.", ckpt_dir)
        self._load_extra_state(ckpt_dir)

    def _log_saved_checkpoint(self, kind: str, epoch: int, step: int) -> None:
        """Log a saved checkpoint on rank 0 when checkpointing is enabled."""
        ckpt_cfg = getattr(self, "checkpoint_config", None)
        if self.dist_env.is_main and ckpt_cfg is not None and ckpt_cfg.enabled:
            logger.info("Saved %s checkpoint to %s/epoch_%d_step_%d", kind, ckpt_cfg.checkpoint_dir, epoch, step)

    def _maybe_save_step_checkpoint(self, epoch: int) -> bool:
        """Save a checkpoint mid-epoch when ``ckpt_every_steps`` is configured.

        Called after every optimizer step. Saves whenever ``ckpt_every_steps`` is a
        positive integer and the current ``global_step`` is a multiple of it, and
        returns True if a checkpoint was written. The directory is named
        ``epoch_{epoch}_step_{global_step}`` so it never collides with the
        end-of-epoch checkpoint (which uses ``epoch + 1``).
        """
        if not _step_cadence_hit(getattr(self, "ckpt_every_steps", None), self.runtime.global_step):
            return False
        total_optim_steps = getattr(self, "total_optim_steps", None)
        is_final_checkpoint = total_optim_steps is not None and self.runtime.global_step >= total_optim_steps
        self.save_checkpoint(
            epoch=epoch,
            step=self.runtime.global_step,
            train_loss=None,
            val_loss=None,
            best_metric_key="val_loss",
            is_final_checkpoint=is_final_checkpoint,
        )
        self._log_saved_checkpoint("step", epoch, self.runtime.global_step)
        return True

    def _maybe_save_final_checkpoint(self, completed_epochs: int) -> bool:
        """Save the fully-trained model at the end of a run, unless a cadence already saved the final step.

        Without this the end-of-run state is easy to lose: with no cadence nothing
        is saved, and with a pure step cadence the final step is skipped whenever
        the total step count is not a multiple of ``ckpt_every_steps``.
        """
        gs = self.runtime.global_step
        if gs <= 0:
            return False
        if _step_cadence_hit(getattr(self, "ckpt_every_steps", None), gs) or getattr(
            self, "save_checkpoint_every_epoch", False
        ):
            return False
        self.save_checkpoint(
            epoch=completed_epochs,
            step=gs,
            train_loss=None,
            val_loss=None,
            best_metric_key="val_loss",
            is_final_checkpoint=True,
        )
        self._log_saved_checkpoint("final", completed_epochs, gs)
        return True

    def _wandb_log(self, data: dict, step: int) -> None:
        """Log a metrics dict to the rank-0 W&B run when one is active."""
        run = getattr(self, "wandb_run", None)
        if run is not None:
            run.log(data, step=step)
