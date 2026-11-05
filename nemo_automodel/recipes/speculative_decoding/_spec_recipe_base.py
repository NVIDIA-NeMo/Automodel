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

Every draft recipe trains a draft against a frozen target. They persist and
restore that draft the same way, step the same warmup + cosine LR schedule,
follow the same checkpoint cadence, and run the same epoch / gradient-accumulation
loop. This base owns that shared lifecycle so a fix lands in one place instead of
drifting across copies. Model construction, the per-micro-batch forward, metric
reporting, and evaluation stay in each recipe, which plugs into the loop through
the hooks below ``run_train_validation_loop``.
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
from nemo_automodel.components.distributed.tp_replicas import synchronize_tp_replica_gradients
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.wandb_utils import init_wandb_run, suppress_wandb_log_messages
from nemo_automodel.recipes.base_recipe import BaseRecipe, _is_checkpoint_model_config_compatible
from nemo_automodel.recipes.speculative_decoding._spec_train_utils import (
    make_warmup_cosine_schedule,
    optim_steps_per_epoch,
    resolve_wandb_kwargs,
    resolve_warmup_steps,
    should_sync_grads,
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
                save_config(self.cfg.to_yaml_dict(use_orig_values=True), path)
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
        """Log a metrics dict to the rank-0 W&B and Trackio runs when active."""
        run = getattr(self, "wandb_run", None)
        if run is not None:
            run.log(data, step=step)
        trackio_logger = getattr(self, "trackio_logger", None)
        if trackio_logger is not None:
            trackio_logger.log_metrics(data, step=step)

    # ------------------------------------------------------------------
    # Training loop
    #
    # One epoch / gradient-accumulation / checkpoint loop serves every draft
    # recipe. A recipe supplies how one micro-batch runs (forward + backward) and
    # how its metrics are accumulated and reported; the hooks below default to the
    # plain DDP behavior, and recipes override only what differs.
    # ------------------------------------------------------------------

    def run_train_validation_loop(self) -> None:
        """Train for the remaining epochs, then save the final checkpoint.

        :meth:`_on_train_end` runs on every exit path (normal, nothing left to do,
        or an exception), so recipes release loggers and remote targets there.
        """
        self.trainer_module.train()
        start_epoch = max(0, int(getattr(self, "_resume_epoch", 0)))
        try:
            if start_epoch >= self.num_epochs:
                if self.dist_env.is_main:
                    logger.info("All %d epochs already completed; nothing to do.", self.num_epochs)
                return
            self._on_train_start(start_epoch)
            pbar = self._make_progress_bar(total=self.total_optim_steps, initial=self.runtime.global_step)
            try:
                for epoch_idx in range(start_epoch, self.num_epochs):
                    if not self._train_one_epoch(epoch_idx, pbar):
                        break
                self._maybe_save_final_checkpoint(self.num_epochs)
                self._finish_checkpointing()
            finally:
                if pbar is not None:
                    pbar.close()
        finally:
            self._on_train_end()

    def _train_one_epoch(self, epoch_idx: int, pbar=None) -> bool:
        """Run one epoch, its evaluation, and its epoch checkpoint.

        An epoch is one or more *segments*: a contiguous pass over the currently
        bound dataloader. :meth:`_at_window_boundary` may rebind the dataloader
        (EAGLE-3 on-policy regen), which ends the segment and starts a new one; the
        epoch ends when a dataloader is exhausted without a rebind. Rebinding only
        happens at a gradient-accumulation window boundary, so no partial window is
        ever carried across segments.

        Args:
            epoch_idx: Zero-based epoch index.
            pbar: Optional progress bar advanced once per optimizer step.

        Returns:
            False when ``stop_at_step_budget`` stopped the run inside this epoch.
        """
        window = self._new_train_window()
        # Summed on the loss's device; read once at the epoch checkpoint so the hot
        # loop never blocks on a host sync.
        epoch_loss: torch.Tensor | float = 0.0
        micro_steps = 0
        batches_seen = 0
        pending_micro_batches = 0
        start_step = self.runtime.global_step
        reached_budget = False

        segment_done = False
        while not segment_done and not reached_budget:
            sampler = getattr(self.train_dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch_idx)
            # Recomputed per segment: a rebound dataloader may have a different
            # length, and ``should_sync_grads`` keys the final partial window off it.
            try:
                num_batches = len(self.train_dataloader)
            except TypeError:
                num_batches = None
            for batch_idx, batch in enumerate(self._iter_train_batches()):
                batches_seen += 1
                sync_ctx = self._grad_sync_context(pending_micro_batches, batch_idx, num_batches)
                metrics = self._train_micro_batch(batch, sync_ctx)
                if metrics is None:
                    # Skipped on every rank (see TrainDFlashRecipe); not part of the window.
                    continue
                self._accumulate_train_metrics(window, metrics)
                epoch_loss = epoch_loss + metrics.loss.detach()
                micro_steps += 1
                pending_micro_batches += 1

                if pending_micro_batches == self.grad_accumulation_steps:
                    grad_norm = self._optimizer_step(epoch_idx, pbar, pending_micro_batches)
                    pending_micro_batches = 0
                    if self.runtime.global_step % self.log_every_steps == 0:
                        self._log_train_window(window, epoch_idx, grad_norm, pbar)
                        window = self._new_train_window()
                    # Window boundary: the only safe point to stop on the step budget
                    # or rebind the dataloader. Both branch on ``global_step``, which
                    # is identical on every rank.
                    if self.stop_at_step_budget and self.runtime.global_step >= self.total_optim_steps:
                        reached_budget = True
                        break
                    if self._at_window_boundary():
                        break
            else:
                segment_done = True

        # Flush the trailing partial accumulation window. When the epoch length is
        # not a multiple of ``grad_accumulation_steps``, up to
        # ``grad_accumulation_steps - 1`` micro-batches ran ``backward()`` without an
        # optimizer step; their gradients would otherwise be wiped by the next
        # epoch's ``zero_grad``. Every data-parallel rank reaches this point with the
        # same ``pending_micro_batches`` because the samplers pad ranks to equal
        # length; a variable-length sampler would desync the flush collectives.
        if pending_micro_batches > 0:
            grad_norm = self._optimizer_step(epoch_idx, pbar, pending_micro_batches)
            self._after_trailing_flush(window, epoch_idx, grad_norm)

        self._on_epoch_end(epoch_idx, batches_seen)
        eval_metrics = self._run_eval()
        val_metrics = self._report_epoch(epoch_idx, self.runtime.global_step - start_step, eval_metrics)
        if getattr(self, "save_checkpoint_every_epoch", False):
            self.save_checkpoint(
                epoch=epoch_idx + 1,
                step=self.runtime.global_step,
                train_loss=float(epoch_loss) / micro_steps if micro_steps else None,
                val_loss=val_metrics,
                best_metric_key="val_loss",
                is_final_checkpoint=epoch_idx + 1 >= self.num_epochs,
            )
            self._log_saved_checkpoint("epoch", epoch_idx + 1, self.runtime.global_step)
        return not reached_budget

    def _optimizer_step(self, epoch_idx: int, pbar, pending_micro_batches: int):
        """Apply one optimizer update from the accumulated gradients.

        Each micro-batch divided its loss by ``grad_accumulation_steps``. A trailing
        partial window has only ``pending_micro_batches`` contributors, so its
        gradients are rescaled to the scale of a full window before clipping.

        Returns:
            The pre-clip gradient norm reported by :meth:`_clip_grad_norm`.
        """
        self._sync_grads_before_clip()
        if pending_micro_batches < self.grad_accumulation_steps:
            scale = float(self.grad_accumulation_steps) / float(pending_micro_batches)
            for param in self.trainer_module.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)
        grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self._after_optimizer_step()
        self.runtime.global_step += 1
        if pbar is not None:
            pbar.update(1)
        self._maybe_save_step_checkpoint(epoch_idx)
        return grad_norm

    # Loop hooks. Defaults match a DDP (or single-process) draft.

    #: Stop once ``global_step`` reaches ``total_optim_steps`` instead of when the
    #: epoch range is exhausted (EAGLE-3: regen splits an epoch into segments).
    stop_at_step_budget: ClassVar[bool] = False
    #: FSDP2 drafts defer the gradient reduce-scatter to the last micro-batch of a
    #: window unless ``distributed.defer_fsdp_grad_sync`` turns it off (DSpark).
    defer_fsdp_grad_sync: bool = True

    def _on_train_start(self, start_epoch: int) -> None:
        """Called once before the first epoch that trains."""

    def _on_train_end(self) -> None:
        """Finish the W&B and Trackio runs; runs on every exit path.

        Best-effort, so a teardown failure never masks the exception that ended
        training.
        """
        for name in ("wandb_run", "trackio_logger"):
            run = getattr(self, name, None)
            if run is None:
                continue
            try:
                run.finish()
            except Exception:
                logger.exception("Failed to finish %s", name)

    def _finish_checkpointing(self) -> None:
        """Wait for and publish pending checkpoints after the last epoch."""
        self._finalize_and_close_checkpointer()

    def _iter_train_batches(self):
        """Return the iterable of training items for the current segment."""
        return self.train_dataloader

    def _grad_sync_context(self, pending_micro_batches: int, batch_idx: int, num_batches: int | None):
        """Return the context that gates this micro-batch's gradient reduction.

        Only the micro-batch an optimizer step follows needs to reduce gradients:
        DDP runs the others under ``no_sync()`` and FSDP2 defers their
        reduce-scatter (see ``get_sync_ctx``). A single-process draft never syncs.
        """
        is_optim_step = should_sync_grads(
            pending_micro_batches=pending_micro_batches,
            grad_accumulation_steps=self.grad_accumulation_steps,
            batch_idx=batch_idx,
            batches_per_epoch=num_batches,
        )
        return get_sync_ctx(self.trainer_module, is_optim_step, self.defer_fsdp_grad_sync)

    def _forward_micro_batch(self, batch):
        """Return the training metrics (with a differentiable ``loss``) of one micro-batch."""
        raise NotImplementedError

    def _train_micro_batch(self, batch, sync_ctx):
        """Run forward and backward for one micro-batch.

        Returns:
            The micro-batch metrics, or None when every rank skipped it.
        """
        with sync_ctx:
            metrics = self._forward_micro_batch(batch)
            (metrics.loss / float(self.grad_accumulation_steps)).backward()
        return metrics

    def _sync_grads_before_clip(self) -> None:
        """Reduce gradients that the data-parallel wrapper does not, before clipping."""
        synchronize_tp_replica_gradients([self.trainer_module], getattr(self, "device_mesh", None))

    def _clip_grad_norm(self):
        """Clip the draft gradients and return the pre-clip norm."""
        return torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)

    def _after_optimizer_step(self) -> None:
        """Called after each optimizer and LR-scheduler step."""

    def _at_window_boundary(self) -> bool:
        """Called after each full accumulation window.

        Returns:
            True when the training dataloader was rebound and a new segment must start.
        """
        return False

    def _new_train_window(self):
        """Return an empty accumulator for the logged training metrics."""
        raise NotImplementedError

    def _accumulate_train_metrics(self, window, metrics) -> None:
        """Add one micro-batch's metrics to ``window``."""
        window.add(metrics)

    def _log_train_window(self, window, epoch_idx: int, grad_norm, pbar) -> None:
        """Report ``window``; called on every rank at the log cadence, which then starts a new window."""
        raise NotImplementedError

    def _after_trailing_flush(self, window, epoch_idx: int, grad_norm) -> None:
        """Called on every rank after the epoch's trailing partial window steps."""

    def _on_epoch_end(self, epoch_idx: int, batches_seen: int) -> None:
        """Called after the epoch's last optimizer step, before evaluation."""

    def _report_epoch(self, epoch_idx: int, completed_steps: int, eval_metrics):
        """Log the end-of-epoch summary.

        Returns:
            The validation metrics handed to the epoch checkpoint.
        """
        return eval_metrics
