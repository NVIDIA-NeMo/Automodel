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

"""Export an LLM or VLM DCP checkpoint to consolidated Hugging Face safetensors.

This tool rebuilds the original distributed model topology from ``config.yaml``
using the recipe recorded in that config (LLM or VLM), loads only the model
weights from a sharded ``torch_save`` checkpoint, and saves them back out as
Hugging Face-compatible safetensors. Because the model is rebuilt with the same
recipe that produced the checkpoint, a VLM checkpoint keeps its vision tower
instead of being silently exported as a text-only model.

Example:
    torchrun --nproc-per-node=8 scripts/export_llm_dcp_to_hf.py \
        --checkpoint-dir /path/to/checkpoints/epoch_0_step_17999 \
        --output-dir /path/to/hf_export
"""

from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path
from typing import Sequence

import torch
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.cli.utils import resolve_recipe_name
from nemo_automodel.components.checkpoint.checkpointing import save_config
from nemo_automodel.components.checkpoint.utils import is_rank_0
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.base_recipe import BaseRecipe

_CHECKPOINT_DIR_RE = re.compile(r"epoch_(\d+)_step_(\d+)$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for DCP to HF export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the saved training checkpoint directory, for example epoch_0_step_17999.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive the exported checkpoint root.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config.yaml path. Defaults to <checkpoint-dir>/config.yaml.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help=(
            "Optional override for model.pretrained_model_name_or_path. Use this when the recorded base "
            "model path in config.yaml is no longer valid."
        ),
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Explicit epoch number for the exported checkpoint directory name. Defaults to the source checkpoint.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Explicit step number for the exported checkpoint directory name. Defaults to the source checkpoint.",
    )
    return parser.parse_args(argv)


def infer_epoch_step(checkpoint_dir: str) -> tuple[int, int]:
    """Parse epoch and step numbers from a checkpoint directory name."""
    match = _CHECKPOINT_DIR_RE.search(Path(checkpoint_dir).name)
    if match is None:
        raise ValueError(
            "Could not infer epoch/step from checkpoint directory name. Pass both --epoch and --step explicitly."
        )
    return int(match.group(1)), int(match.group(2))


def resolve_epoch_step(checkpoint_dir: str, epoch: int | None, step: int | None) -> tuple[int, int]:
    """Resolve exported epoch and step from explicit flags or the source checkpoint name."""
    if epoch is not None and step is not None:
        return epoch, step
    inferred_epoch, inferred_step = infer_epoch_step(checkpoint_dir)
    return (
        inferred_epoch if epoch is None else epoch,
        inferred_step if step is None else step,
    )


def build_export_config(args: argparse.Namespace) -> ConfigNode:
    """Load config.yaml and apply export-specific overrides."""
    config_path = Path(args.config) if args.config is not None else Path(args.checkpoint_dir) / "config.yaml"
    cfg = parse_args_and_load_config(
        str(config_path),
        argv=[
            "--checkpoint.restore_from",
            "None",
            # Hidden directory that absorbs recipe-setup side effects (metric JSONL
            # logs, auto-resume scan) without touching the user-facing export root.
            "--checkpoint.checkpoint_dir",
            str(Path(args.output_dir) / ".export_workdir"),
            "--checkpoint.model_save_format",
            "safetensors",
            "--checkpoint.save_consolidated",
            "final",
            # An offline export must not log to remote experiment trackers.
            "--wandb",
            "None",
            "--mlflow",
            "None",
            "--comet",
            "None",
        ],
    )
    if cfg.get("peft", None) is not None:
        raise ValueError(
            "PEFT checkpoints are saved as a single consolidated adapter_model.safetensors already; "
            "this exporter only supports full-model torch_save checkpoints."
        )
    if args.model_name_or_path is not None:
        cfg.set_by_dotted("model.pretrained_model_name_or_path", args.model_name_or_path)
    return cfg


def build_recipe(cfg: ConfigNode) -> BaseRecipe:
    """Instantiate the recipe class recorded in ``cfg.recipe`` (LLM or VLM).

    The model topology must be rebuilt with the same recipe that wrote the
    checkpoint; otherwise an LLM recipe would construct a text-only model for a
    VLM checkpoint and drop its vision tower. ``allow_checkpoint_key_subset`` then
    raises on the unmatched keys instead of silently exporting a partial model.
    """
    raw = cfg.get("recipe", None)
    if raw is None:
        raise ValueError("config.yaml has no 'recipe' field, so the model architecture cannot be rebuilt for export.")
    module_name, _, class_name = resolve_recipe_name(raw).rpartition(".")
    recipe_cls = getattr(importlib.import_module(module_name), class_name)
    return recipe_cls(cfg)


def resolve_export_tokenizer(trainer: BaseRecipe):
    """Return the tokenizer (LLM) or processor (VLM) to persist with the export."""
    return getattr(trainer, "tokenizer", None) or getattr(trainer, "processor", None)


def resolve_model_for_export(
    trainer: BaseRecipe,
) -> torch.nn.Module | list[torch.nn.Module]:
    """Return the model object that should be passed to the checkpointer."""
    model_parts = trainer.model_parts
    if len(model_parts) > 1:
        return model_parts
    model = model_parts[0]
    if isinstance(model, DistributedDataParallel):
        model = model.module
    return model


def barrier() -> None:
    """Synchronize across ranks when torch.distributed is initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def close_trainer(trainer: BaseRecipe | None) -> None:
    """Best-effort cleanup for recipe-owned resources."""
    if trainer is None:
        return
    # The LLM recipe keeps validation loggers in a dict (one per val set); the VLM
    # recipe keeps a single logger. Handle both without assuming the shape.
    for attr in ("metric_logger_train", "metric_logger_valid"):
        logger_obj = getattr(trainer, attr, None)
        if logger_obj is None:
            continue
        loggers = logger_obj.values() if isinstance(logger_obj, dict) else [logger_obj]
        for logger in loggers:
            close = getattr(logger, "close", None)
            if close is not None:
                close()
    if hasattr(trainer, "checkpointer"):
        trainer.checkpointer.close()


def main(argv: Sequence[str] | None = None) -> None:
    """Restore a DCP checkpoint and re-save it as Hugging Face safetensors."""
    args = parse_args(argv)
    export_epoch, export_step = resolve_epoch_step(args.checkpoint_dir, args.epoch, args.step)
    export_root = Path(args.output_dir) / f"epoch_{export_epoch}_step_{export_step}"
    # Fail fast on every rank, before the expensive recipe setup and before
    # torch.distributed is initialized.
    if export_root.exists():
        raise FileExistsError(f"Export directory already exists: {export_root}")
    cfg = build_export_config(args)

    trainer: BaseRecipe | None = None
    try:
        trainer = build_recipe(cfg)
        trainer.setup()
        export_model = resolve_model_for_export(trainer)
        # Idempotent on every rank, so a filesystem failure raises everywhere
        # instead of stranding the other ranks in a collective.
        export_root.mkdir(parents=True, exist_ok=True)
        trainer.checkpointer.load_model(
            export_model,
            model_path=str(Path(args.checkpoint_dir) / "model"),
            allow_checkpoint_key_subset=True,
        )
        if is_rank_0():
            # to_yaml_dict() keeps _target_/*_fn entries as dotted-path strings so the
            # exported config.yaml stays loadable; to_dict() would serialize the
            # resolved Python objects as !!python/name tags that safe_load rejects.
            save_config(cfg.to_yaml_dict(), str(export_root))
        barrier()
        trainer.checkpointer.save_model(
            model=export_model,
            weights_path=str(export_root),
            tokenizer=resolve_export_tokenizer(trainer),
            is_final_checkpoint=True,
        )
    finally:
        close_trainer(trainer)

    if is_rank_0():
        output_dir = export_root / "model" / "consolidated"
        print(f"HF export is ready under: {output_dir}")


if __name__ == "__main__":
    main()
