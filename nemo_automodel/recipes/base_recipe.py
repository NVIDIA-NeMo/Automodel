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
import logging
import os
import re
import socket
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizerBase

from nemo_automodel.components.checkpoint.checkpointing import (
    load_model,
    load_optimizer,
    save_config,
    save_model,
    save_optimizer,
)
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.step_scheduler import StepScheduler

try:
    import yaml as _yaml
except Exception:
    _yaml = None
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizerBase


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


def is_tokenizer(object):
    """
    Checks whether object is a tokenizer or VLM processor.

    Args:
        object (any): the object to check.

    Returns:
        bool: returns True if object is a tokenizer or VLM processor.
    """
    return isinstance(object, (PreTrainedTokenizerBase, ProcessorMixin))


def is_lr_scheduler(object):
    """
    Checks whether object is a learning rate scheduler.

    Args:
        object (any): the object to check.

    Returns:
        bool: returns True if object is an OptimizerParamScheduler.
    """
    return (
        isinstance(object, list)
        and all(isinstance(item, OptimizerParamScheduler) for item in object)
        and len(object) > 0
    )


def is_optimizer(object):
    """
    Checks whether object is an optimizer.
    """
    return isinstance(object, Optimizer) or (
        isinstance(object, list) and all(isinstance(item, Optimizer) for item in object) and len(object) > 0
    )


def is_model(object):
    """
    Checks whether object is a model.
    """
    return isinstance(object, nn.Module) or (
        isinstance(object, list) and all(isinstance(item, nn.Module) for item in object)
    )


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
        # Track stateful objects unless they are validation/eval components.
        should_track = (
            is_model(value)
            or has_load_restore_state(value)
            or is_tokenizer(value)
            or is_lr_scheduler(value)
            or is_optimizer(value)
            or isinstance(value, ConfigNode)
        )

        if should_track and not any(substr in key.lower() for substr in ("val", "eval", "test")):
            assert key not in self.__dict__["__state_tracked"]
            self.__dict__["__state_tracked"].add(key)
        super().__setattr__(key, value)

    def save_checkpoint(self, epoch: int, step: int):
        """
        Save the current training state as a checkpoint.

        As long as the object has a 'load_state_dict' and 'state_dict' function, it will be saved.

        Args:
            epoch (int): The current epoch.
            step (int): The current step.
        """
        if not self.checkpoint_config.enabled:
            return
        is_dist_initialized = torch.distributed.is_initialized()
        is_rank_0 = not is_dist_initialized or torch.distributed.get_rank() == 0

        if not is_dist_initialized:
            dp_group = None
        else:
            dp_group = self._get_dp_group()

        path = self.checkpoint_config.checkpoint_dir
        path = os.path.join(path, f"epoch_{epoch}_step_{step}")

        if is_rank_0:
            assert not os.path.exists(path), f"Checkpoint directory {path} already exists"
            os.makedirs(path, exist_ok=True)
            print(f"Saving checkpoint to {path}", flush=True)
        if is_dist_initialized:
            torch.distributed.barrier(dp_group)
        # TODO(@adil-a): Change this when we create a LR scheduler class
        model, optimizer, scheduler, tokenizer, config = None, None, None, None, None

        for key in self.__dict__["__state_tracked"]:
            if is_model(getattr(self, key)):
                model = getattr(self, key)
            elif is_optimizer(getattr(self, key)):
                optimizer = getattr(self, key)
            elif isinstance(getattr(self, key), ConfigNode):
                config = getattr(self, key)
            elif is_lr_scheduler(getattr(self, key)):
                scheduler = getattr(self, key)
            elif is_tokenizer(getattr(self, key)):
                tokenizer = getattr(self, key)
            else:
                if is_rank_0:
                    torch.save(
                        getattr(self, key).state_dict(),
                        os.path.join(path, f"{key}.pt"),
                    )

        save_model(model, path, self.checkpoint_config, peft_config=self.peft_config, tokenizer=tokenizer)
        save_optimizer(optimizer, model, path, scheduler)
        save_config(config.raw_config, path)
        if is_dist_initialized:
            torch.distributed.barrier(dp_group)

    def load_checkpoint(self, restore_from: str | None = None):
        """
        Loads the latest checkpoint.
        """
        if not self.checkpoint_config.enabled:
            if (
                not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            ) and restore_from is not None:
                print("Enable checkpointing to resume from a checkpoint, skipping...", flush=True)
            return

        if restore_from:
            ckpt_dir = restore_from
        else:
            # Determine the latest checkpoint directory (e.g. ".../step_42").
            ckpt_dir = _find_latest_checkpoint(self.checkpoint_config.checkpoint_dir)
            if ckpt_dir is None:
                return

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Loading checkpoint from {ckpt_dir}", flush=True)

        model, optimizer, scheduler = None, None, None
        for key in self.__dict__["__state_tracked"]:
            if is_model(getattr(self, key)):
                model = getattr(self, key)
            elif is_optimizer(getattr(self, key)):
                optimizer = getattr(self, key)
            elif is_lr_scheduler(getattr(self, key)):
                scheduler = getattr(self, key)
            elif is_tokenizer(getattr(self, key)) or isinstance(getattr(self, key), ConfigNode):
                # we don't need to load the tokenizer or config from the checkpoint
                # we only save the tokenizer for consolidated checkpoints for downstream use
                continue
            else:
                getattr(self, key).load_state_dict(torch.load(os.path.join(ckpt_dir, f"{key}.pt"), weights_only=False))

        load_model(model, ckpt_dir, self.checkpoint_config)
        load_optimizer(optimizer, model, ckpt_dir, scheduler)

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
            if _yaml is not None:
                details_yaml = _yaml.safe_dump(details, sort_keys=False, default_flow_style=False).strip()
            else:
                details_yaml = "\n".join(f"{k}: {v}" for k, v in details.items())
            list(map(logging.info, ("Experiment_details:\n" + details_yaml).splitlines()))
        except Exception:
            logging.info(f"Experiment details: {details}")
        # Resolved config
        try:
            cfg_obj = getattr(self, "cfg", None)
            cfg_dict = (
                cfg_obj.to_dict() if hasattr(cfg_obj, "to_dict") else (dict(cfg_obj) if cfg_obj is not None else {})
            )

            def rec_print(log_fn, cfg_dict: dict | None, indent: int = 2):
                if cfg_dict is None:
                    return
                for k, v in cfg_dict.items():
                    if isinstance(v, dict):
                        log_fn(f"{' ' * indent}{k}:")
                        rec_print(log_fn, v, indent + 2)
                    else:
                        log_fn(f"{' ' * indent}{k}: {v}")

            logging.info("Recipe config:")
            rec_print(logging.info, cfg_dict)
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
        optimizer: Optimizer | None = None,
        lr_scheduler: OptimizerParamScheduler | None = None,
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
            for line in ("Optimizer:\n" + str(optimizer[0])).splitlines():
                logging.info(line)
        else:
            logging.info("Optimizer: <unavailable>")

        # LR scheduler
        if lr_scheduler:
            for line in ("LR scheduler:\n" + str(lr_scheduler[0])).splitlines():
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

    def _get_dp_group(self):
        if not self.device_mesh:
            return None
        elif self.device_mesh["cp"].size() > 1:
            return self.device_mesh["dp_cp"].get_group()
        else:
            return self.device_mesh["dp"].get_group()

    def _get_dp_group_size(self):
        dp_group = self._get_dp_group()
        return 1 if dp_group is None else dp_group.size()

    def _dp_allreduce(self, tensor, op=dist.ReduceOp.SUM):
        dp_group = self._get_dp_group()
        if dp_group is not None:
            tensor = tensor.cuda()
            dist.all_reduce(tensor, op=op, group=dp_group)
            tensor = tensor.cpu()
        return tensor


def _find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the checkpoint directory and return it.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    # Accept checkpoints saved as either `step_<num>` or `epoch_<epoch>_step_<num>`
    # (or any other pattern that contains the substring `step_`).
    # This makes the checkpoint loading logic compatible with the naming scheme
    # used in `save_checkpoint`, which currently saves to `epoch_{epoch}_step_{step}`.
    checkpoint_files = list(checkpoint_dir.glob("*step_*"))
    if not checkpoint_files:
        return

    def _step_num(path: Path):
        """Return the numeric step from a path stem of the form step_<int>."""
        m = re.search(r"step_(\d+)$", path.stem)
        return int(m.group(1)) if m else -1

    latest = max(checkpoint_files, key=_step_num)

    # If no directory followed the expected "step_<int>" pattern, _step_num would be -1 for all of them.
    # Treat that as "no valid checkpoint".
    if _step_num(latest) == -1:
        return

    return latest
