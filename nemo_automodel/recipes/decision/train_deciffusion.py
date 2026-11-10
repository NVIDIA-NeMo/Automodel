# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Post-train Nemotron Deciffusion decision checkpoints on one-question decision rows (LoRA or full fine-tuning).

Each row is rendered with the serving prompt (one question, serving option order and codes), one mask token is
appended, the backbone reads the sequence bidirectionally, and the frozen diffusion head's logits at the mask,
restricted to the option codes, give the answer distribution. The loss is soft/hard cross-entropy, or gold
cross-entropy plus a weighted KL to soft labels (``train.kl_weight > 0``). A ``train.canonical_fraction`` share of the
training reads keep the serving option order; the rest are shuffled and re-coded by position.

* ``peft`` set: LoRA on the attention/MLP projections; embeddings and the diffusion head stay frozen.
* ``peft: null``: full fine-tuning of the decoder layers and final norm; embeddings and the diffusion head stay frozen.
* ``mode: eval`` scores the configured checkpoint (or ``init_from`` stage) on validation and test.

Validation selects the published stage (``LOWEST_VAL``): highest accuracy, then lowest cross-entropy; for LoRA the
starting checkpoint (step 0) is eligible. Report the test split separately; the validation split is not an untouched
test. Export a selected stage as a complete servable checkpoint with ``examples/decision/export_deciffusion.py``.

Run with::

    automodel examples/decision/deciffusion_support_routing_lora.yaml --nproc-per-node 1
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.utils import is_checkpoint_incomplete, read_checkpoint_pointer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.models.deciffusion.export import checkpoint_fingerprint
from nemo_automodel.components.models.deciffusion.objective import decision_loss
from nemo_automodel.components.models.deciffusion.rows import DecisionExample, encode_row, read_jsonl
from nemo_automodel.components.models.deciffusion.scorer import (
    full_finetune_trainable,
    gather_head_rows,
    option_logits,
    require_bidirectional_text_config,
)
from nemo_automodel.components.models.deciffusion.serving_prompt import readout_token_ids
from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig
from nemo_automodel.components.training.utils import clip_grad_norm
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.base_recipe import BaseRecipe

logger = logging.getLogger(__name__)

MODES = ("eval", "train")
STAGE_FORMAT = "deciffusion-decision-stage-v1"


def selection_value(accuracy: float, cross_entropy: float) -> float:
    """Lower-is-better scalar ordering (accuracy desc, cross-entropy asc) for up to 10^4 validation rows."""
    return -accuracy + 1e-6 * min(cross_entropy, 99.0)


class TrainDeciffusionDecisionRecipe(BaseRecipe):
    """LoRA or full post-training of Deciffusion decision checkpoints (see module docstring)."""

    def __init__(self, cfg: RecipeConfig | ConfigNode) -> None:
        """Store the recipe configuration."""
        self.cfg = cfg if isinstance(cfg, RecipeConfig) else RecipeConfig(cfg)

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        """Check the checkpoint contract, build the model, encode the splits and the optimizer."""
        self.dist_env = initialize_distributed(
            backend=self.cfg.get("dist_env.backend", "nccl"),
            timeout_minutes=self.cfg.get("dist_env.timeout_minutes", 30),
        )
        setup_logging()
        self.mode = self.cfg.get("mode", "train")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.mode == "train" and not self.cfg.checkpoint.enabled:
            raise ValueError("training requires checkpoint.enabled=true for validation-selected stages")
        self.seed = int(self.cfg.get("seed", 101))
        self.output_dir = Path(self.cfg.get("output_dir", f"runs/deciffusion_{self.mode}"))
        self.device = self.dist_env.device
        torch.manual_seed(self.seed)

        model_path = self.cfg.get("model.pretrained_model_name_or_path")
        revision = self.cfg.get("model.revision", None)
        if self.cfg.get("model.dlm_paradigm", "bidirectional") != "bidirectional":
            raise ValueError("model.dlm_paradigm must be 'bidirectional' for decision post-training")
        from transformers import AutoConfig, AutoTokenizer

        require_bidirectional_text_config(
            AutoConfig.from_pretrained(model_path, revision=revision, trust_remote_code=True)
        )

        (
            self.distributed_setup,
            self.mesh_context,
            self.distributed_config,
            self.device_mesh,
            self.moe_mesh,
            self.pp_enabled,
            self.pipeline_config,
            self.moe_parallel_config,
            self.activation_checkpointing,
        ) = self._distributed_setup_attributes(
            create_distributed_setup_from_config(self.cfg, world_size=self.dist_env.world_size)
        )
        if self.pp_enabled or self.mesh_context.tp_size > 1 or self.mesh_context.cp_size > 1:
            raise ValueError("Deciffusion decision post-training supports data parallelism only (tp=cp=pp=1)")

        self.peft_config = self.cfg.peft.instantiate() if self.cfg.get("peft", None) is not None else None
        self.model = self.cfg.model.instantiate(peft_config=self.peft_config, distributed_setup=self.distributed_setup)
        require_bidirectional_text_config(self.model.config)
        self.mask_token_id = int(self.model.config.mask_token_id)
        if self.peft_config is None:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(full_finetune_trainable(name))
        head = self.model.get_output_embeddings()
        if head is None or any(parameter.requires_grad for parameter in head.parameters()):
            raise ValueError("the diffusion head must exist and stay frozen; exclude it from peft target modules")
        if any(p.requires_grad for p in self.model.get_input_embeddings().parameters()):
            raise ValueError("the input embeddings must stay frozen")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True)
        self.max_tokens = int(self.cfg.get("data.max_tokens", 8192))
        self.soft_gold_mask = bool(self.cfg.get("train.soft_gold_mask", False))
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.examples: dict[str, list[DecisionExample]] = {}
        self.usable: list[int] = []
        for split in ("train", "validation", "test"):
            path = self.cfg.get(f"data.{split}", None)
            if not path:
                continue
            rows = read_jsonl(path)
            encoded = [self._encode(row) for row in rows]
            self.rows[split] = rows
            self.examples[split] = [example for example in encoded if example is not None]
            if split == "train":
                self.usable = [i for i, example in enumerate(encoded) if example is not None]
            skipped = len(rows) - len(self.examples[split])
            if skipped and self.dist_env.is_main:
                logger.warning(
                    "data.%s: skipped %d of %d rows (not one text question with a label among its options, "
                    "malformed, or over data.max_tokens=%d)",
                    split,
                    skipped,
                    len(rows),
                    self.max_tokens,
                )
        if self.mode == "train" and not self.usable:
            raise ValueError("data.train has no usable rows")
        token_ids = readout_token_ids(self.tokenizer)
        rows = gather_head_rows(self.model, token_ids).to(self.device)
        self.head_rows = {token: rows[i] for i, token in enumerate(token_ids)}

        checkpoint_config = copy.copy(self.cfg.checkpoint)
        checkpoint_config.checkpoint_dir = str(self.output_dir)
        self.checkpointer = checkpoint_config.build(
            dp_rank=self._get_dp_rank(),
            tp_rank=0,
            pp_rank=0,
            moe_mesh=self.moe_mesh,
            process_group=self._get_dp_group(),
        )
        self.identity = {
            "format": STAGE_FORMAT,
            "base": str(model_path),
            "revision": revision,
            "base_fingerprint": checkpoint_fingerprint(model_path) if Path(model_path).is_dir() else None,
            "method": "lora" if self.peft_config is not None else "full",
            "mask_token_id": self.mask_token_id,
            "readout_token_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
            "max_tokens": self.max_tokens,
        }
        init_from = self.cfg.get("init_from", None)
        if init_from:
            self._load_stage(Path(init_from))
        self.optimizer = None
        if self.mode == "train":
            self.optimizer = self.cfg.optimizer.build(
                self.model, device_mesh=self.device_mesh, is_peft=self.peft_config is not None
            )

    def _encode(self, row: dict[str, Any], shuffle_seed: int | None = None) -> DecisionExample | None:
        return encode_row(
            self.tokenizer,
            row,
            max_tokens=self.max_tokens,
            shuffle_seed=shuffle_seed,
            soft_gold_mask=self.soft_gold_mask,
        )

    # ------------------------------------------------------------------ scoring
    def _logits(self, example: DecisionExample) -> torch.Tensor:
        """Option logits of ``example``, fp32 tensor of shape ``[options]`` in prompt order."""
        return option_logits(
            self.model,
            example.prompt.input_ids,
            example.prompt.code_token_ids,
            mask_token_id=self.mask_token_id,
            rows=self.head_rows,
            device=self.device,
        )

    @torch.no_grad()
    def evaluate(self, split: str, *, predictions: bool = False) -> dict[str, Any]:
        """Accuracy and cross-entropy on ``split`` (identical on every rank), optionally with per-row probabilities."""
        self.model.eval()
        examples = self.examples[split]
        rank, world = self._get_dp_rank(), self._get_dp_group_size()
        shard = examples[rank::world]
        stats = torch.zeros(3, dtype=torch.float64, device=self.device)
        by_kind: dict[str, list[float]] = {}
        local_predictions = []
        for i in range(math.ceil(len(examples) / world)):  # same number of forwards on every rank
            example = shard[i] if i < len(shard) else examples[0]
            log_probs = self._logits(example).log_softmax(-1)
            if i >= len(shard):
                continue
            correct = float(int(log_probs.argmax()) == example.gold)
            stats += torch.tensor(
                [correct, -float(log_probs[example.gold]), 1.0], dtype=torch.float64, device=self.device
            )
            by_kind.setdefault(example.prompt.kind, [0.0, 0.0])
            by_kind[example.prompt.kind][0] += correct
            by_kind[example.prompt.kind][1] += 1
            if predictions:
                local_predictions.append(
                    {
                        "id": example.row_id,
                        "kind": example.prompt.kind,
                        "labels": list(example.prompt.labels),
                        "probabilities": log_probs.exp().tolist(),
                        "gold": example.prompt.labels[example.gold],
                    }
                )
        if dist.is_initialized() and world > 1:
            dist.all_reduce(stats, group=self._get_dp_group())
            gathered: list[Any] = [None] * world
            dist.all_gather_object(gathered, (by_kind, local_predictions), group=self._get_dp_group())
        else:
            gathered = [(by_kind, local_predictions)]
        kinds: dict[str, list[float]] = {}
        merged_predictions = []
        for part_kinds, part_predictions in gathered:
            for kind, (hits, count) in part_kinds.items():
                kinds.setdefault(kind, [0.0, 0.0])
                kinds[kind][0] += hits
                kinds[kind][1] += count
            merged_predictions += part_predictions
        self.model.train()
        count = max(1.0, float(stats[2]))
        summary: dict[str, Any] = {
            "split": split,
            "rows": int(stats[2]),
            "skipped_rows": len(self.rows[split]) - len(examples),
            "accuracy": float(stats[0]) / count,
            "cross_entropy": float(stats[1]) / count,
            "accuracy_by_type": {k: hits / n for k, (hits, n) in sorted(kinds.items())},
        }
        if predictions:
            summary["predictions"] = sorted(merged_predictions, key=lambda p: p["id"])
        return summary

    # ------------------------------------------------------------------ stage I/O
    def _save_stage(self, path: Path, validation: dict[str, Any] | None) -> None:
        """Publish the trainable weights (LoRA adapter or consolidated full weights) as an immutable stage."""
        self.checkpointer.lifecycle.reserve(str(path))
        self.checkpointer.save_model(self.model, str(path), peft_config=self.peft_config)
        value = selection_value(validation["accuracy"], validation["cross_entropy"]) if validation else None

        def write_metadata() -> None:
            (path / "stage.json").write_text(json.dumps({**self.identity, "validation": validation}, indent=2))
            if value is not None:
                (path / "losses.json").write_text(json.dumps({"selection": value}))

        self.checkpointer.lifecycle.run_coordinator_step(write_metadata, description="write Deciffusion stage metadata")
        self.checkpointer.async_wait()
        self.checkpointer.lifecycle.publish(str(path), best_val_metric=value, metric_key="selection")

    def _load_stage(self, path: Path) -> None:
        """Initialize weights from a published stage of the same base, method and readout contract."""
        if path.name in {"LOWEST_VAL", "LATEST"}:
            resolved = read_checkpoint_pointer(path.parent, path.name)
            if resolved is None:
                raise FileNotFoundError(f"stage pointer {path} has no published target")
            path = Path(resolved)
        if is_checkpoint_incomplete(path) or not (path / "stage.json").exists():
            raise ValueError(f"{path} is not a complete Deciffusion decision stage")
        saved = json.loads((path / "stage.json").read_text())
        mismatched = sorted(k for k in self.identity if saved.get(k) != self.identity[k])
        if mismatched:
            raise ValueError(f"stage {path} was trained with a different {mismatched}; refusing to load it")
        self.checkpointer.load_model(self.model, str(path / "model"))
        if self.dist_env.is_main:
            logger.info("Initialized weights from stage %s; optimizer and schedule are fresh", path)

    # ------------------------------------------------------------------ main loop
    def run_train_validation_loop(self) -> None:
        """Evaluate (``mode: eval``) or train, then report validation and test metrics of the selected weights."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == "train":
            selected = self._train()
            if selected is not None:
                self._load_stage(selected)
        for split in ("validation", "test"):
            if split in self.examples:
                summary = self.evaluate(split, predictions=True)
                if self.dist_env.is_main:
                    (self.output_dir / f"metrics_{split}.json").write_text(json.dumps(summary, indent=2))
                    logger.info(
                        "[%s] %s accuracy=%.4f ce=%.4f", self.mode, split, summary["accuracy"], summary["cross_entropy"]
                    )
        self.checkpointer.finalize()
        if dist.is_initialized():
            dist.barrier()

    def _log(self, name: str, record: dict[str, Any]) -> None:
        if self.dist_env.is_main:
            with (self.output_dir / name).open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            logger.info("%s %s", name.split(".")[0], json.dumps(record))

    def _train(self) -> Path | None:
        """Train; returns the validation-selected stage, or ``None`` when the start was not saved and stays best."""
        world = self._get_dp_group_size()
        rank = self._get_dp_rank()
        batch = int(self.cfg.get("train.global_batch_size", 16))
        if batch <= 0 or batch % world:
            raise ValueError(f"train.global_batch_size={batch} must be a positive multiple of the DP size {world}")
        if len(self.usable) < batch:
            raise ValueError(f"{len(self.usable)} usable training rows < train.global_batch_size {batch}")
        accum = batch // world
        steps_per_epoch = len(self.usable) // batch
        total = int(self.cfg.get("train.max_steps", 0) or 0) or max(
            1, int(steps_per_epoch * float(self.cfg.get("train.epochs", 1.0)))
        )
        warmup = max(1, int(float(self.cfg.get("train.warmup_fraction", 0.05)) * total))
        canonical_fraction = float(self.cfg.get("train.canonical_fraction", 0.5))
        kl_weight = float(self.cfg.get("train.kl_weight", 0.0))
        val_every = int(self.cfg.get("train.val_every_steps", 50))
        max_grad_norm = float(self.cfg.get("train.max_grad_norm", 1.0))
        is_lora = self.peft_config is not None

        step_schedule = StepSchedulerConfig(
            global_batch_size=batch, num_epochs=1, max_steps=total, preemption_signal=None
        ).build(dataloader=None, dp_group_size=world, local_batch_size=1, process_group=self._get_dp_group())
        lr = self.optimizer[0].param_groups[0]["lr"]
        scheduler_config = dataclasses.replace(
            self.cfg.lr_scheduler,
            lr_warmup_steps=warmup,
            lr_decay_steps=total,
            init_lr=0.0,
            max_lr=lr,
            min_lr=float(self.cfg.get("train.min_lr_ratio", 0.1)) * lr,
        )
        schedulers = scheduler_config.build(self.optimizer, step_schedule)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if self.dist_env.is_main:
            config = {
                "total_steps": total,
                "warmup_steps": warmup,
                "accumulation": accum,
                "usable_train_rows": len(self.usable),
                "train_rows": len(self.rows["train"]),
                "trainable_parameters": sum(p.numel() for p in trainable),
                **self.identity,
            }
            (self.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))

        def validate(step: int) -> dict[str, Any] | None:
            if "validation" not in self.examples:
                return None
            summary = self.evaluate("validation")
            record = {k: summary[k] for k in ("accuracy", "cross_entropy", "rows")}
            self._log("validation.jsonl", {"step": step, **record})
            return record

        start = validate(0)
        best = (start, 0)
        if is_lora:  # the start is eligible: LoRA B is zero, so the adapter equals the base
            self._save_stage(self.output_dir / "step_00000000", start)
        rows = self.rows["train"]
        order = list(self.usable)
        random.Random(self.seed).shuffle(order)
        step, epoch, cursor = 0, 0, 0
        self.model.train()
        while step < total:
            started = time.monotonic()
            if cursor + batch > len(order):
                epoch += 1
                cursor = 0
                order = list(self.usable)
                random.Random(self.seed + epoch).shuffle(order)
            window = order[cursor : cursor + batch]
            cursor += batch
            losses = []
            for j in range(accum):
                index = window[j * world + rank]
                seed = index * 1000003 + self.seed + epoch
                shuffle = None if random.Random(seed).random() < canonical_fraction else seed
                example = self._encode(rows[index], shuffle) or self._encode(rows[index])
                soft = torch.tensor(example.soft, device=self.device) if example.soft else None
                loss = decision_loss(self._logits(example), example.gold, soft, kl_weight=kl_weight)
                (loss / accum).backward()
                losses.append(loss.detach())
            grad_norm = clip_grad_norm(max_grad_norm, [self.model], device_mesh=self.device_mesh)
            for scheduler in schedulers:
                scheduler.step(1)  # lr for this update: warmup (step+1)/warmup, then cosine to min_lr
            for optimizer in self.optimizer:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            mean_loss = torch.stack(losses).mean()
            if dist.is_initialized() and world > 1:
                dist.all_reduce(mean_loss, group=self._get_dp_group())
                mean_loss /= world
            grad_norm = grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
            self._log(
                "training.jsonl",
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": round(float(mean_loss), 5),
                    "grad_norm": round(float(grad_norm), 4),
                    "lr": self.optimizer[0].param_groups[0]["lr"],
                    "seconds": round(time.monotonic() - started, 2),
                },
            )
            if step % val_every == 0 or step == total:
                record = validate(step)
                self._save_stage(self.output_dir / f"step_{step:08d}", record)
                if record is not None and (
                    best[0] is None
                    or selection_value(record["accuracy"], record["cross_entropy"])
                    < selection_value(best[0]["accuracy"], best[0]["cross_entropy"])
                ):
                    best = (record, step)
        if self.dist_env.is_main:
            selected = {"step": best[1], "validation": best[0]}
            (self.output_dir / "selected.json").write_text(json.dumps(selected, indent=2))
        if best[0] is None:
            return self.output_dir / f"step_{total:08d}"
        if best[1] == 0 and not is_lora:
            return None  # nothing beat the starting checkpoint; the base itself is selected
        return self.output_dir / f"step_{best[1]:08d}"


def main(config_path: str | None = None) -> None:
    """CLI entry point: ``python -m nemo_automodel.recipes.decision.train_deciffusion --config <yaml>``."""
    cfg = parse_args_and_load_config(config_path)
    recipe = TrainDeciffusionDecisionRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
