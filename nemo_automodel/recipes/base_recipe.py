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

import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizerBase
from torch.utils.data import DataLoader

from nemo_automodel.components.checkpoint.checkpointing import (
    load_model,
    load_optimizer,
    save_model,
    save_optimizer,
)
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler


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
    return isinstance(object, OptimizerParamScheduler)


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
        # Explicitly treat DataLoader subclasses with state_dict support as stateful, regardless of attribute name.
        from torch.utils.data import DataLoader  # Local import to avoid circular deps.

        is_dataloader = isinstance(value, DataLoader) and has_load_restore_state(value)

        should_track = (
            isinstance(value, (nn.Module, Optimizer))
            or is_dataloader
            or has_load_restore_state(value)
            or is_tokenizer(value)
            or is_lr_scheduler(value)
        )

        # Keep historical behavior of skipping val/eval/test components **except** for dataloaders.
        if should_track and (is_dataloader or not any(substr in key.lower() for substr in ("val", "eval", "test"))):
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

        path = self.checkpoint_config.checkpoint_dir
        path = os.path.join(path, f"epoch_{epoch}_step_{step}")
        os.makedirs(path, exist_ok=True)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Saving checkpoint to {path}", flush=True)

        # TODO(@adil-a): Change this when we create a LR scheduler class
        model, optimizer, scheduler, tokenizer = None, None, None, None

        for key in self.__dict__["__state_tracked"]:
            obj = getattr(self, key)

            if isinstance(obj, nn.Module):
                model = obj
            elif isinstance(obj, Optimizer):
                optimizer = obj
            elif is_lr_scheduler(obj):
                scheduler = obj
            elif is_tokenizer(obj):
                tokenizer = obj

            # ---- General stateful components (incl. DataLoaders) ----
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if isinstance(obj, DataLoader) and has_load_restore_state(obj):
                # Each DP rank writes its own file – avoid collisions with rank suffix
                filename = f"{key}_rank{rank}.pt" if torch.distributed.is_initialized() else f"{key}.pt"
                torch.save(obj.state_dict(), os.path.join(path, filename))
                # Make sure all ranks finish writing before proceeding
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            else:
                # Historical behaviour: only rank0 writes
                if not torch.distributed.is_initialized() or rank == 0:
                    torch.save(obj.state_dict(), os.path.join(path, f"{key}.pt"))
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

        save_model(model, path, self.checkpoint_config, peft_config=self.peft_config, tokenizer=tokenizer)
        save_optimizer(optimizer, model, path, scheduler)

        # -------------------- Save config YAML --------------------
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            try:
                import yaml
                # Prefer attribute `cfg` if it exists (OmegaConf or dict)
                cfg_obj = getattr(self, "cfg", None)
                if cfg_obj is not None:
                    # Handle OmegaConf configs gracefully
                    try:
                        from omegaconf import OmegaConf

                        cfg_dict = OmegaConf.to_container(cfg_obj, resolve=True)
                    except ModuleNotFoundError:
                        cfg_dict = cfg_obj if isinstance(cfg_obj, dict) else None

                    if cfg_dict is not None:
                        with open(os.path.join(path, "config.yaml"), "w") as f:
                            yaml.safe_dump(cfg_dict, f, sort_keys=False)
            except Exception as e:  # pylint: disable=broad-except
                # Do not fail training if config serialization fails
                if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                    print(f"Warning: failed to save config.yaml – {e}", flush=True)

        # -------------------- Save generation config if present --------------------
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            try:
                import shutil

                repo_dir = None
                if (
                    hasattr(self, "checkpoint_config")
                    and self.checkpoint_config.model_cache_dir
                    and self.checkpoint_config.model_repo_id
                ):
                    repo_dir = os.path.join(
                        str(self.checkpoint_config.model_cache_dir), str(self.checkpoint_config.model_repo_id)
                    )

                # Look for generation_config.json or generate.json
                if repo_dir and os.path.exists(repo_dir):
                    target_file = None
                    for root, _dirs, files in os.walk(repo_dir):
                        for fname in files:
                            if fname in ("generation_config.json", "generate.json"):
                                target_file = os.path.join(root, fname)
                                break
                        if target_file:
                            break

                    if target_file and os.path.isfile(target_file):
                        shutil.copy2(target_file, os.path.join(path, os.path.basename(target_file)))
            except Exception as e:  # pylint: disable=broad-except
                if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                    print(f"Warning: failed to copy generation config – {e}", flush=True)

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
            obj = getattr(self, key)

            if isinstance(obj, nn.Module):
                model = obj
                continue
            if isinstance(obj, Optimizer):
                optimizer = obj
                continue
            if is_lr_scheduler(obj):
                scheduler = obj
                continue
            if is_tokenizer(obj):
                # tokenizer is saved only for consolidated checkpoints
                continue

            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

            # DataLoader: look for rank-specific file first, fall back to shared file (back-compat)
            preferred_fname = (
                f"{key}_rank{rank}.pt" if torch.distributed.is_initialized() and isinstance(obj, DataLoader) else f"{key}.pt"
            )
            path_preferred = os.path.join(ckpt_dir, preferred_fname)
            fallback_path = os.path.join(ckpt_dir, f"{key}.pt")

            load_path = path_preferred if os.path.exists(path_preferred) else fallback_path
            obj.load_state_dict(torch.load(load_path, weights_only=False))

        load_model(model, ckpt_dir, self.checkpoint_config)
        load_optimizer(optimizer, model, ckpt_dir, scheduler)


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
