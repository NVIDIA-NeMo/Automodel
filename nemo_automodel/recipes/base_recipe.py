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

import getpass
import json
import logging
import os
import re
import socket
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizerBase

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.checkpoint.checkpointing import save_config
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler


def has_load_restore_state(object):
    """
    Checks whether object has load_state_dict and state_dict functions.

    TODO: also need to check function signatures.

    Args:
        object (any): the object to check.

    Returns:
        bool: returns True if has callable load_state_dict and state_dict
    """
    return all(callable(getattr(object, attr, None)) for attr in ("load_state_dict", "state_dict"))


def is_dataloader(object):
    """
    Checks whether object is a dataloader.

    Args:
        object (any): the object to check.

    Returns:
        bool: returns True if object is a dataloader.
    """
    return isinstance(object, StatefulDataLoader) and has_load_restore_state(object)


def is_tokenizer(object):
    """
    Checks whether object is a tokenizer or VLM processor.

    Args:
        object (any): the object to check.

    Returns:
        bool: returns True if object is a tokenizer or VLM processor.
    """
    return isinstance(object, (PreTrainedTokenizerBase, ProcessorMixin, NeMoAutoTokenizer))


def is_lr_scheduler(object):
    """
    Checks whether object is a learning rate scheduler.

    Args:
        object (any): the object to check.

    Returns:
        bool: returns True if object is an OptimizerParamScheduler.
    """
    return isinstance(object, OptimizerParamScheduler) or (
        isinstance(object, list)
        and all(isinstance(item, OptimizerParamScheduler) for item in object)
        and len(object) > 0
    )


def is_optimizer(object):
    """
    Checks whether object is an optimizer.
    """
    return isinstance(object, Optimizer) or (
        isinstance(object, list) and len(object) > 0 and all(isinstance(item, Optimizer) for item in object)
    )


def is_model(object):
    """
    Checks whether object is a model.
    """
    return isinstance(object, nn.Module) or (
        isinstance(object, list) and len(object) > 0 and all(isinstance(item, nn.Module) for item in object)
    )


def _list_existing_checkpoints(ckpt_root: Path) -> list[Path]:
    """Return existing checkpoint directories under ckpt_root (matching '*step_*')."""
    if not ckpt_root.exists():
        return []
    return list(ckpt_root.glob("*step_*"))


def _resolve_restore_from_to_ckpt_dir(checkpoint_dir: str, restore_from: str) -> str | None:
    """
    Resolve restore_from to a checkpoint directory.

    Returns:
        - str: resolved checkpoint directory
        - None: if restore_from='LATEST' but no checkpoint found (caller should start fresh)
    """
    # Handle "LATEST" keyword for convenience
    if restore_from.upper() == "LATEST":
        return _find_latest_checkpoint(checkpoint_dir)

    # If restore_from is just a directory name (no path separator), treat it as
    # relative to checkpoint_dir. Otherwise use as-is (absolute or relative path).
    if os.path.sep not in restore_from and not os.path.isabs(restore_from):
        return os.path.join(checkpoint_dir, restore_from)
    return restore_from


def _format_existing_ckpt_collision_error(ckpt_root: Path, existing_ckpts: list[Path]) -> str:
    """Format an error message for 'starting fresh with existing checkpoints'."""
    return "\n".join(
        [
            f"\n{'=' * 80}",
            "ERROR: Checkpoint directory contains existing checkpoints",
            f"{'=' * 80}",
            f"Directory: {ckpt_root}",
            f"Found {len(existing_ckpts)} existing checkpoint(s):",
            f"  {', '.join([p.name for p in existing_ckpts[:5]])}",
            f"  {'...' if len(existing_ckpts) > 5 else ''}",
            "",
            "Starting fresh training with existing checkpoints will cause checkpoint",
            "name collisions when saving (e.g., epoch_0_step_100 already exists).",
            "",
            "To FIX, choose ONE of:",
            "  1. Resume training:",
            "       Set checkpoint.restore_from: 'LATEST' (for latest checkpoint)",
            "       Or checkpoint.restore_from: 'epoch_0_step_100' (subdirectory name)",
            "       Or checkpoint.restore_from: './path/to/checkpoint' (full path)",
            "  2. Start fresh with unique directory:",
            "       Set checkpoint.checkpoint_dir: './checkpoints/my_exp_name'",
            "  3. Start fresh by removing existing checkpoints:",
            f"       Run: rm -rf {ckpt_root}/*step_*",
            f"{'=' * 80}",
        ]
    )


def _format_missing_checkpoint_dir_error(checkpoint_dir: str, restore_from: str, resolved_ckpt_dir: str) -> str:
    """Format a helpful error message for a missing checkpoint directory."""
    error_msg = [
        f"\n{'=' * 80}",
        "ERROR: Checkpoint directory does not exist",
        f"{'=' * 80}",
        f"Specified: checkpoint.restore_from: '{restore_from}'",
        f"Resolved to: {resolved_ckpt_dir}",
        "",
        "Please check:",
        "  1. The checkpoint directory exists",
        f"  2. The path is correct (restore_from: '{restore_from}')",
        f"  3. Available checkpoints in {checkpoint_dir}:",
    ]

    ckpt_root = Path(checkpoint_dir)
    available_ckpts = _list_existing_checkpoints(ckpt_root)
    if available_ckpts:
        error_msg += [f"       {', '.join([p.name for p in available_ckpts[:5]])}"]
        if len(available_ckpts) > 5:
            error_msg += [f"       ... and {len(available_ckpts) - 5} more"]
    else:
        error_msg += (
            ["       (no checkpoints found)"] if ckpt_root.exists() else ["       (checkpoint_dir does not exist)"]
        )

    error_msg += [f"{'=' * 80}"]
    return "\n".join(error_msg)


def _is_rank_0() -> bool:
    """True if distributed is not initialized or this process is rank 0.
    TODO(@akoumpa): deprecate in favor of deviemesh api
    """
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _dist_barrier() -> None:
    """Barrier if torch.distributed is initialized.
    TODO(@akoumpa): deprecate in favor of deviemesh api
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


class BaseRecipe:
    """
    BaseRecipe provides checkpoint load/save functionality for recipes.
    """

    def __setattr__(self, key, value):
        """
        Overriden __setattr__ to keep track of stateful classes.

        Args:
            key (str): attribute named.
            value (Any): Value assigned

        Raises:
            ValueError: if __state_tracked is attemped to be overwriten.

        """
        # assuming no one will do recipe.__dict__['__state_tracked'] = None
        if key == "__state_tracked":
            raise ValueError("cannot set __state_tracked")
        if "__state_tracked" not in self.__dict__:
            self.__dict__["__state_tracked"] = set()

        # Initialize best checkpoint tracking
        if "_best_val_loss" not in self.__dict__:
            self.__dict__["_best_val_loss"] = float("inf")

        # Track stateful objects unless they are validation/eval components.
        should_track = (
            is_model(value)
            or has_load_restore_state(value)
            or is_tokenizer(value)
            or is_lr_scheduler(value)
            or is_optimizer(value)
            or isinstance(value, ConfigNode)
            or is_dataloader(value)
        )

        if should_track and not any(substr in key.lower() for substr in ("val", "eval", "test", "loss")):
            assert key not in self.__dict__["__state_tracked"]
            self.__dict__["__state_tracked"].add(key)
        super().__setattr__(key, value)

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: dict[str, float] | None = None,
        best_metric_key: str = "default",
    ):
        """
        Save the current training state as a checkpoint.

        As long as the object has a 'load_state_dict' and 'state_dict' function, it will be saved.

        Args:
            epoch (int): The current epoch.
            step (int): The current step.
            train_loss (float): The current training loss.
            val_loss (dict[str, float]): The current validation losses.
            best_metric_key (str): The validation metric key used to select the best checkpoint.
        """
        if not self.checkpointer.config.enabled:
            return

        # Wait for any in-flight checkpoint (async case) to complete
        self.checkpointer.async_wait()

        # If a previous async checkpoint just finished, update the "latest" symlink now
        prev_pending = getattr(self, "_last_pending_checkpoint_dir", None)
        is_dist_initialized = torch.distributed.is_initialized()
        is_rank_0 = not is_dist_initialized or torch.distributed.get_rank() == 0
        if prev_pending is not None:
            if is_rank_0:
                self._update_latest_symlink(prev_pending)
            # clear and remember the last completed path
            setattr(self, "_last_pending_checkpoint_dir", None)
            if is_dist_initialized:
                torch.distributed.barrier()

        # If a previous async checkpoint just finished, also update the "best" symlink now (if pending)
        prev_best_pending = getattr(self, "_last_pending_best_checkpoint_info", None)
        if prev_best_pending is not None:
            if is_rank_0 and prev_best_pending.get("val") is not None:
                self._update_best_symlink(prev_best_pending["path"], float(prev_best_pending["val"]))
            setattr(self, "_last_pending_best_checkpoint_info", None)
            if is_dist_initialized:
                torch.distributed.barrier()

        path = self.checkpointer.config.checkpoint_dir
        path = os.path.join(path, f"epoch_{epoch}_step_{step}")

        best_val_metric = (
            val_loss[next(iter(val_loss.keys())) if len(val_loss) == 1 else best_metric_key] if val_loss else None
        )

        if is_rank_0:
            assert not os.path.exists(path), f"Checkpoint directory {path} already exists"
            os.makedirs(path, exist_ok=True)
            print(f"Saving checkpoint to {path}", flush=True)

            def to_item(x):
                if isinstance(x, torch.Tensor):
                    return x.item()
                return x

            # dump the train and val loss to a json file
            loss_dict = {"train_loss": train_loss}
            if val_loss:
                # the name of the key can be "default", so we rename it to "val_loss"
                key = next(iter(val_loss.keys()))
                loss_dict["val_loss"] = val_loss.pop(key) if len(val_loss) == 1 else loss_dict.update(val_loss)
            with open(os.path.join(path, "losses.json"), "w") as f:
                try:
                    json.dump({k: to_item(v) for k, v in loss_dict.items()}, f)
                except:
                    pass

        if is_dist_initialized:
            torch.distributed.barrier()

        model, optimizer, scheduler, tokenizer, config = None, None, None, None, None

        for key in sorted(self.__dict__["__state_tracked"]):
            if is_model(getattr(self, key)):
                if key == "teacher_model":
                    continue
                model = getattr(self, key)
            elif is_optimizer(getattr(self, key)):
                optimizer = getattr(self, key)
            elif isinstance(getattr(self, key), ConfigNode):
                config = getattr(self, key)
            elif is_lr_scheduler(getattr(self, key)):
                scheduler = getattr(self, key)
            elif is_tokenizer(getattr(self, key)):
                tokenizer = getattr(self, key)
            elif is_dataloader(getattr(self, key)) or isinstance(getattr(self, key), StatefulRNG):
                self.checkpointer.save_on_dp_ranks(getattr(self, key), key, path)
            else:
                if is_rank_0:
                    torch.save(
                        getattr(self, key).state_dict(),
                        os.path.join(path, f"{key}.pt"),
                    )

        self.checkpointer.save_model(model, path, peft_config=self.peft_config, tokenizer=tokenizer)
        self.checkpointer.save_optimizer(optimizer, model, path, scheduler)
        save_config(config.raw_config, path)
        if is_dist_initialized:
            torch.distributed.barrier()

        # Update latest symlink according to sync/async behavior
        if getattr(self.checkpointer.config, "is_async", False):
            # Async: defer symlink until the next call (after async_wait completes)
            setattr(self, "_last_pending_checkpoint_dir", path)
            # Defer best symlink update similarly, capturing the metric used for comparison
            if best_val_metric is not None:
                setattr(self, "_last_pending_best_checkpoint_info", {"path": path, "val": float(best_val_metric)})
        else:
            # Sync: update immediately
            if is_rank_0:
                self._update_latest_symlink(path)
                if best_val_metric is not None:
                    self._update_best_symlink(path, float(best_val_metric))
            if is_dist_initialized:
                torch.distributed.barrier()

    def _update_checkpoint_symlink(self, link_name: str, target_dir: str) -> None:
        """
        Create or update a symlink named `link_name` under the checkpoint root
        that points to `target_dir`.
        Assumes caller ensures rank 0 if needed.
        """
        ckpt_root = self.checkpointer.config.checkpoint_dir
        link_path = os.path.join(ckpt_root, link_name)
        if os.path.lexists(link_path):
            os.remove(link_path)

        ckpt_root_abs = os.path.abspath(ckpt_root)
        target_abs = os.path.abspath(target_dir)
        relative_target = os.path.relpath(target_abs, start=ckpt_root_abs)
        os.symlink(relative_target, link_path)

    def _update_latest_symlink(self, target_dir: str) -> None:
        """
        Create or update a symlink named "latest" under the checkpoint root
        that points to `target_dir`.
        Only called on rank 0.
        """
        self._update_checkpoint_symlink("LATEST", target_dir)

    def _update_best_symlink(self, target_dir: str, val_loss: float) -> None:
        """
        Create or update a symlink named "LOWEST_VAL" under the checkpoint root
        that points to the checkpoint with the lowest validation loss.
        Only called on rank 0.
        """
        # Update best checkpoint if this one is better
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._update_checkpoint_symlink("LOWEST_VAL", target_dir)
            logging.info(
                f"Updated LOWEST_VAL checkpoint symlink to {os.path.basename(target_dir)} (val_loss={val_loss:.4f})"
            )

    def _assert_safe_fresh_start(self, ckpt_root: Path, is_rank_0: bool) -> None:
        """
        Starting fresh training is only allowed if checkpoint_dir is empty (of '*step_*'),
        to prevent checkpoint name collisions during saving.
        """
        existing_ckpts = _list_existing_checkpoints(ckpt_root)
        if not existing_ckpts:
            return

        if is_rank_0:
            logging.error(_format_existing_ckpt_collision_error(ckpt_root, existing_ckpts))

        # Ensure all ranks fail together
        _dist_barrier()
        raise RuntimeError("Checkpoint directory contains existing checkpoints but restore_from not set")

    def _validate_checkpoint_dir_exists(self, ckpt_dir: str, restore_from: str, is_rank_0: bool) -> None:
        """Validate resolved checkpoint directory exists; raise FileNotFoundError with a helpful message."""
        if os.path.exists(ckpt_dir):
            return

        # Build helpful error message on rank 0
        if is_rank_0:
            error_msg = _format_missing_checkpoint_dir_error(
                checkpoint_dir=self.checkpointer.config.checkpoint_dir,
                restore_from=restore_from,
                resolved_ckpt_dir=ckpt_dir,
            )
        else:
            error_msg = f"Checkpoint directory does not exist: {ckpt_dir}"

        # Ensure all ranks fail together (before raising)
        _dist_barrier()
        raise FileNotFoundError(error_msg)

    def _load_checkpoint_tracked_state(self, ckpt_dir: str):
        """Load tracked state and return (model, optimizer, scheduler) for downstream loader calls."""
        model, optimizer, scheduler = None, None, None

        for key in sorted(self.__dict__["__state_tracked"]):
            obj = getattr(self, key)
            if is_model(obj):
                assert model is None, "Multiple models found in checkpoint"
                model = obj
            elif is_optimizer(obj):
                assert optimizer is None, "Multiple optimizers found in checkpoint"
                optimizer = obj
            elif is_lr_scheduler(obj):
                assert scheduler is None, "Multiple schedulers found in checkpoint"
                scheduler = obj
            elif is_dataloader(obj) or isinstance(obj, StatefulRNG):
                self.checkpointer.load_on_dp_ranks(obj, key, ckpt_dir)
            elif is_tokenizer(obj) or isinstance(obj, ConfigNode):
                # we don't need to load the tokenizer or config from the checkpoint
                # we only save the tokenizer for consolidated checkpoints for downstream use
                continue
            else:
                obj.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{key}.pt"), weights_only=False))

        return model, optimizer, scheduler

    def load_checkpoint(self, restore_from: str | None = None):
        """
        Loads checkpoint only when explicitly requested via restore_from.

        This method will:
        - Load checkpoint if restore_from is explicitly set to a path or "LATEST"
        - Start fresh training if restore_from is None
        - Fail early if checkpoint_dir contains checkpoints but restore_from is not set
          (prevents checkpoint name collisions during saving)

        Args:
            restore_from: Path to checkpoint directory to restore from. Options:
                         - None: Start fresh (fails if checkpoint_dir not empty)
                         - "LATEST": Auto-detect latest checkpoint in checkpoint_dir
                         - "epoch_0_step_100": Subdirectory name (relative to checkpoint_dir)
                         - "./path/to/checkpoint": Absolute or relative path
        """
        if not self.checkpointer.config.enabled:
            if _is_rank_0() and restore_from is not None:
                print("Enable checkpointing to resume from a checkpoint, skipping...", flush=True)
            return

        is_rank_0 = _is_rank_0()

        if not restore_from:
            # Starting fresh - ensure checkpoint dir is safe to avoid name collisions
            ckpt_root = Path(self.checkpointer.config.checkpoint_dir)
            self._assert_safe_fresh_start(ckpt_root, is_rank_0=is_rank_0)
            return

        ckpt_dir = _resolve_restore_from_to_ckpt_dir(self.checkpointer.config.checkpoint_dir, restore_from)
        if ckpt_dir is None:
            if is_rank_0:
                logging.warning(
                    "restore_from='LATEST' specified but no checkpoint found in "
                    f"{self.checkpointer.config.checkpoint_dir}. Starting fresh."
                )
            return

        self._validate_checkpoint_dir_exists(ckpt_dir, restore_from=restore_from, is_rank_0=is_rank_0)

        if is_rank_0:
            print(f"Loading checkpoint from {ckpt_dir}", flush=True)

        model, optimizer, scheduler = self._load_checkpoint_tracked_state(ckpt_dir)

        self.checkpointer.load_model(model, os.path.join(ckpt_dir, "model"))
        self.checkpointer.load_optimizer(optimizer, model, ckpt_dir, scheduler)

    def _log_experiment_details(self):
        """Log metadata and resolved config on main rank using YAML markers."""
        if not getattr(self, "dist_env", None) or not getattr(self.dist_env, "is_main", False):
            return
        details = {
            "Timestamp": datetime.now().isoformat(timespec="seconds"),
            "User": getpass.getuser(),
            "Host": socket.gethostname(),
            "World size": getattr(self.dist_env, "world_size", None),
            "Backend": getattr(getattr(self, "cfg", {}), "get", lambda *_: None)("dist_env.backend", "nccl"),
            "Recipe": self.__class__.__name__,
            "Model name": getattr(getattr(self, "cfg", None), "model", None)
            and getattr(self.cfg.model, "pretrained_model_name_or_path", None),
        }
        try:
            details_yaml = yaml.safe_dump(details, sort_keys=False, default_flow_style=False).strip()
            for line in ("Experiment_details:\n" + details_yaml).splitlines():
                logging.info(line)
        except Exception:
            logging.info(f"Experiment details: {details}")
        # Resolved config
        try:
            cfg_obj = getattr(self, "cfg", None)
            # Prefer YAML-ready dict that converts callables/classes to dotted paths and preserves typed scalars
            if hasattr(cfg_obj, "to_yaml_dict"):
                cfg_dict = cfg_obj.to_yaml_dict()
            elif hasattr(cfg_obj, "to_dict"):
                cfg_dict = cfg_obj.to_dict()
            else:
                cfg_dict = dict(cfg_obj) if cfg_obj is not None else {}

            # Print as clean YAML on stdout for easy copy/paste and readability
            cfg_yaml = yaml.safe_dump(cfg_dict, sort_keys=False, default_flow_style=False).strip()
            print(cfg_yaml, flush=True)
        except Exception:
            logging.info("Recipe config: <unavailable>")

    def _log_library_versions(self):
        """Log import paths and versions for nemo_automodel, transformers, and torch."""
        if not getattr(self, "dist_env", None) or not getattr(self.dist_env, "is_main", False):
            return
        try:
            import nemo_automodel as nemo_am

            nemo_path = Path(getattr(nemo_am, "__file__", "<unknown>")).resolve().as_posix()
        except Exception:
            nemo_path = "<unknown>"
        try:
            import transformers as hf_transformers

            tfm_path = Path(getattr(hf_transformers, "__file__", "<unknown>")).resolve().as_posix()
        except Exception:
            tfm_path = "<unknown>"
        libs = {
            "nemo_automodel": {"version": getattr(nemo_am, "__version__", None), "import_path": nemo_path},
            "transformers": {"version": getattr(hf_transformers, "__version__", None), "import_path": tfm_path},
            "torch": {"version": torch.__version__, "cuda": getattr(torch.version, "cuda", None)},
        }
        logging.info("Library versions:")
        for key, value in libs.items():
            if "cuda" in value:
                logging.info(f"- {key}: {value['version']} CUDA {value['cuda']}")
            else:
                logging.info(f"- {key}: {value['version']} ({value['import_path']})")

    def _log_model_and_optimizer_details(
        self,
        model: nn.Module | list[nn.Module] | None = None,
        optimizer: Optimizer | list[Optimizer] | None = None,
        lr_scheduler: OptimizerParamScheduler | list[OptimizerParamScheduler] | None = None,
    ):
        """Log model repr, parameter stats, param norm, optimizer and lr scheduler with YAML markers."""
        # Model repr
        if not isinstance(model, list):
            model = [model]

        for i, m in enumerate(model):
            if m is None:
                logging.info(f"Model Part {i}: <unavailable>")
                continue

            model_str = str(m)
            model_lines = model_str.splitlines()
            logging.info(f"Model Part {i}:")
            for line in model_lines[:40]:
                logging.info(line)
            if len(model_lines) > 40:
                logging.info("...")

        # Optimizer
        if optimizer:
            if not isinstance(optimizer, list):
                optimizer = [optimizer]
            for opt in optimizer:
                for line in ("Optimizer:\n" + str(opt)).splitlines():
                    logging.info(line)
        else:
            logging.info("Optimizer: <unavailable>")

        # LR scheduler
        if lr_scheduler:
            if not isinstance(lr_scheduler, list):
                lr_scheduler = [lr_scheduler]
            for sched in lr_scheduler:
                for line in ("LR scheduler:\n" + str(sched)).splitlines():
                    logging.info(line)
        else:
            logging.info("LR scheduler: <unavailable>")

    def _log_step_scheduler_details(self, step_scheduler: StepScheduler):
        """Log step scheduler details."""
        attrs = {
            "Gradient accumulation steps": step_scheduler.grad_acc_steps,
            "Checkpoint every steps": step_scheduler.ckpt_every_steps,
            "Current Epoch": step_scheduler.epoch,
            "Number of epochs": step_scheduler.num_epochs,
            "Validation every steps": step_scheduler.val_every_steps,
            "Max train steps": step_scheduler.max_steps,
        }
        logging.info("Step scheduler:")
        for k, v in attrs.items():
            logging.info(f"- {k}: {v}")

    def _get_dp_group(self, include_cp: bool = False):
        if not self.device_mesh:
            return None
        if include_cp and self.device_mesh["cp"].size() > 1:
            return self.device_mesh["dp_cp"].get_group()
        return self.device_mesh["dp"].get_group()

    def _get_dp_group_size(self, include_cp: bool = False):
        dp_group = self._get_dp_group(include_cp=include_cp)
        return 1 if dp_group is None else dp_group.size()

    def _get_cp_group_size(self):
        if not self.device_mesh or self.device_mesh["cp"].size() == 1:
            return 1
        return self.device_mesh["cp"].size()

    def _get_dp_rank(self, include_cp: bool = False):
        if not self.device_mesh:
            return 0
        if include_cp and self.device_mesh["cp"].size() > 1:
            return self.device_mesh.get_local_rank("dp_cp")
        return self.device_mesh.get_local_rank("dp")

    def _get_tp_rank(self):
        if not self.device_mesh or self.device_mesh["tp"].size() == 1:
            return 0
        return self.device_mesh.get_local_rank("tp")

    def _get_pp_rank(self):
        # PP is a special case because it'll only be present in the device mesh if pp is enabled
        if not self.device_mesh or "pp" not in self.device_mesh.mesh_dim_names or self.device_mesh["pp"].size() == 1:
            return 0
        return self.device_mesh.get_local_rank("pp")

    def _dp_allreduce(self, tensor, op=dist.ReduceOp.SUM, include_cp: bool = False):
        dp_group = self._get_dp_group(include_cp=include_cp)
        if dp_group is not None:
            tensor = tensor.cuda()
            dist.all_reduce(tensor, op=op, group=dp_group)
            tensor = tensor.cpu()
        return tensor


def _find_latest_checkpoint(checkpoint_dir):
    """
    Resolve the most recent checkpoint directory.

    Preference order:
      1) Valid LATEST symlink under checkpoint_dir
      2) Highest step directory under checkpoint_dir matching *step_*

    Returns:
        Path (or str) of the latest checkpoint directory, or None.
    """
    root = Path(checkpoint_dir)
    if not root.exists():
        return

    # Try LATEST symlink first
    latest_link = os.path.join(os.fspath(root), "LATEST")
    if os.path.islink(latest_link):
        try:
            resolved = os.readlink(latest_link)
            if not os.path.isabs(resolved):
                resolved = os.path.abspath(os.path.join(os.fspath(root), resolved))
            if os.path.isdir(resolved):
                return resolved
        except OSError:
            pass

    # Fallback to scanning
    checkpoint_files = list(root.glob("*step_*"))
    if not checkpoint_files:
        return

    def _step_num(path: Path):
        m = re.search(r"step_(\d+)$", path.stem)
        return int(m.group(1)) if m else -1

    latest = max(checkpoint_files, key=_step_num)
    if _step_num(latest) == -1:
        return

    return latest
