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

"""Post-train Nemotron 3.5 Decision checkpoints on SystemOne records (LoRA or backbone fine-tuning).

Every question of a record is its own read, rendered with the checkpoint's own decision contract
(``decision_template.jinja`` and ``decision.json``: the question's option order and the checkpoint's codes), which
ends in the mask token. The bidirectional Ministral 3 encoder reads it, and the frozen decision head's logits at the
mask, restricted to the option codes, give the answer distribution. Supervision, the objective and the metrics are the shared decision
components: ``question_targets`` aligns answers, rewards and soft labels with each prompt's option order,
``assemble_decision_batch`` and ``DecisionLoss`` (``loss.mode: sft``) compute the loss, and ``DecisionEvaluator``
scores validation and test. Training reads keep the checkpoint's option order unless ``train.canonical_fraction`` is
below 1, in which case the remaining share is shuffled and re-coded by position.

* ``peft`` set: LoRA on the attention/MLP projections; embeddings and the decision head stay frozen.
* ``peft: null``: backbone fine-tuning of the encoder blocks and final norm; embeddings and the decision head stay frozen.
* ``mode: eval`` scores the configured checkpoint (or ``init_from`` stage) on validation and test.

Validation selects the published stage (``LOWEST_VAL``): most correct validation decisions, then lowest NLL; for LoRA
the starting checkpoint (step 0) is a saved, selectable stage; backbone fine-tuning selects among trained stages. Report
the test split separately; the validation split is not an untouched test. Export a selected stage as a complete
servable checkpoint with ``examples/decision/export_nemotron_decision.py``.

Run with::

    automodel examples/decision/nemotron_decision_support_routing_lora.yaml --nproc-per-node 1
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
from nemo_automodel.components.datasets.decision.records import (
    QuestionTarget,
    load_records,
    pad_and_shard_records,
    question_targets,
    validate_record,
)
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.eval.decision.metrics import DecisionEvaluator, save_summary
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loss.decision import DecisionLossConfig
from nemo_automodel.components.loss.decision_batch import DecisionTargets, QuestionLogits, assemble_decision_batch
from nemo_automodel.components.models.nemotron_decision.export import checkpoint_fingerprint
from nemo_automodel.components.models.nemotron_decision.records import encode_question
from nemo_automodel.components.models.nemotron_decision.scorer import (
    full_finetune_trainable,
    gather_head_rows,
    option_logits,
    require_decision_config,
)
from nemo_automodel.components.models.nemotron_decision.serving_prompt import (
    DecisionContract,
    PreparedQuestion,
    readout_token_ids,
)
from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig
from nemo_automodel.components.training.utils import clip_grad_norm
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.base_recipe import BaseRecipe

logger = logging.getLogger(__name__)

MODES = ("eval", "train")
STAGE_FORMAT = "nemotron-decision-stage-v1"


def selection_value(correct: int, cross_entropy: float) -> float:
    """Lower-is-better scalar that orders stages by correct validation decisions, then by lower cross-entropy.

    The cross-entropy term lies in ``[0, 0.99]``, below the gap of one correct decision, so the order is exactly
    lexicographic for any validation size (every stage is scored on the same validation decisions).

    Args:
        correct: Number of correctly decided labelled validation questions.
        cross_entropy: Mean validation cross-entropy, the evaluator's ``nll`` (clipped to 99).
    """
    return -float(correct) + min(cross_entropy, 99.0) / 100.0


@dataclasses.dataclass(frozen=True)
class _Decision:
    """One supervised question of a record, encoded in the checkpoint's option order.

    Attributes:
        record: The validated SystemOne record that holds the question.
        prompt: Serving prompt of the question (option labels and codes in prompt order).
        target: Supervision aligned with ``prompt.labels``.
    """

    record: dict[str, Any]
    prompt: PreparedQuestion
    target: QuestionTarget


class TrainNemotronDecisionRecipe(BaseRecipe):
    """LoRA or full post-training of Nemotron 3.5 Decision checkpoints (see module docstring)."""

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
        self.loss_config: DecisionLossConfig = (
            self.cfg.loss.instantiate() if self.cfg.get("loss", None) is not None else DecisionLossConfig()
        )
        if self.loss_config.mode != "sft":
            raise ValueError("Nemotron Decision post-training supports loss.mode=sft only")
        self.loss_fn = self.loss_config.build()
        self.seed = int(self.cfg.get("seed", 101))
        self.output_dir = Path(self.cfg.get("output_dir", f"runs/nemotron_decision_{self.mode}"))
        self.device = self.dist_env.device
        torch.manual_seed(self.seed)

        model_path = self.cfg.get("model.pretrained_model_name_or_path")
        revision = self.cfg.get("model.revision", None)
        from transformers import AutoConfig, AutoTokenizer

        hf_config = AutoConfig.from_pretrained(model_path, revision=revision)
        require_decision_config(hf_config)
        self.contract = DecisionContract.from_pretrained(model_path, revision=revision)

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
            raise ValueError("Nemotron Decision post-training supports data parallelism only (tp=cp=pp=1)")

        self.peft_config = self.cfg.peft.instantiate() if self.cfg.get("peft", None) is not None else None
        self.model = self.cfg.model.instantiate(peft_config=self.peft_config, distributed_setup=self.distributed_setup)
        require_decision_config(self.model.config)
        self.mask_token_id = int(self.model.config.mask_token_id)
        if self.peft_config is None:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(full_finetune_trainable(name))
        head = self.model.get_output_embeddings()
        if head is None or any(parameter.requires_grad for parameter in head.parameters()):
            raise ValueError("the decision head must exist and stay frozen; exclude it from peft target modules")
        if any(p.requires_grad for p in self.model.get_input_embeddings().parameters()):
            raise ValueError("the input embeddings must stay frozen")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        self.max_tokens = int(self.cfg.get("data.max_tokens", 8192))
        read_limit = min(self.contract.max_input_tokens, int(hf_config.max_position_embeddings))
        if not 0 < self.max_tokens <= read_limit:
            raise ValueError(f"data.max_tokens must be in [1, {read_limit}] (the checkpoint's read limit)")
        self.slice_keys = list(self.cfg.get("data.slice_keys", []) or [])
        self.records: dict[str, list[dict[str, Any]]] = {}
        self.decisions: dict[str, list[_Decision]] = {}
        self.skipped: dict[str, int] = {}
        for split in ("train", "validation", "test"):
            path = self.cfg.get(f"data.{split}", None)
            if not path:
                continue
            records = load_records(path)
            if not records:
                raise ValueError(f"data.{split} must contain at least one record")
            for record in records:
                validate_record(record)
            self.records[split] = records
            self.decisions[split], self.skipped[split] = self._encode_split(records, train=split == "train")
            if not self.decisions[split]:
                raise ValueError(f"data.{split} has no supervised text decisions")
            if self.skipped[split] and self.dist_env.is_main:
                logger.warning(
                    "data.%s: skipped %d questions (media, outside the decision contract, or over data.max_tokens=%d)",
                    split,
                    self.skipped[split],
                    self.max_tokens,
                )
        if self.mode == "train" and "train" not in self.decisions:
            raise ValueError("data.train is required for mode=train")
        validation = self.decisions.get("validation", [])
        if self.mode == "train" and validation and not any(d.target.label >= 0 for d in validation):
            raise ValueError("data.validation needs gold answers: stages are selected by validation accuracy")
        token_ids = list(readout_token_ids(self.tokenizer, self.contract))
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
            "decision_contract_sha256": self.contract.sha256,
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

    def _encode_split(self, records: list[dict[str, Any]], *, train: bool) -> tuple[list[_Decision], int]:
        """Encode every supervised question of ``records`` in the checkpoint's option order.

        Training keeps the questions the objective weights (a gold answer, or soft labels when they replace the
        gold target); evaluation keeps the questions the evaluator scores (a gold answer or rewards).

        Args:
            records: Validated SystemOne records of one split.
            train: Whether the split trains the model.

        Returns:
            The decisions, and the number of questions this model cannot read (media, a question that does not fit
            the checkpoint's decision contract, or a prompt longer than ``data.max_tokens``).
        """
        decisions, skipped = [], 0
        for record in records:
            for question_id in record["questions"]:
                prompt = encode_question(
                    self.tokenizer,
                    self.contract,
                    record,
                    question_id,
                    mask_token_id=self.mask_token_id,
                    max_tokens=self.max_tokens,
                )
                if prompt is None:
                    skipped += 1
                    continue
                target = self._target(record, prompt)
                if train:
                    keep = float(self.loss_fn.normalization_weight(self._targets(record, target))) > 0
                else:
                    keep = target.label >= 0 or target.rewards is not None
                if keep:
                    decisions.append(_Decision(record, prompt, target))
        return decisions, skipped

    @staticmethod
    def _target(record: dict[str, Any], prompt: PreparedQuestion) -> QuestionTarget:
        """Supervision of ``prompt``'s question, aligned with the prompt's option order."""
        return question_targets(record, {prompt.question_id: prompt.labels})[0]

    def _targets(
        self, record: dict[str, Any], target: QuestionTarget, device: torch.device | None = None
    ) -> DecisionTargets:
        """One-question loss targets with the configured question weights, on ``device`` (CPU by default)."""
        return DecisionTargets.from_questions(
            record_ids=[str(record["id"])],
            questions=[[target]],
            question_weights=self.loss_config.question_weights,
            device=device,
        )

    # ------------------------------------------------------------------ scoring
    def _logits(self, prompt: PreparedQuestion) -> torch.Tensor:
        """Option logits of ``prompt``, fp32 tensor of shape ``[options]`` in prompt order."""
        return option_logits(
            self.model,
            prompt.input_ids,
            prompt.code_token_ids,
            mask_token_id=self.mask_token_id,
            rows=self.head_rows,
            device=self.device,
        )

    @torch.no_grad()
    def evaluate(self, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Score every decision of ``split`` with the shared ``DecisionEvaluator``.

        Every rank runs the same number of forwards; padded duplicates are scored once.

        Returns:
            The evaluator summary and its per-decision predictions, identical on every rank.
        """
        self.model.eval()
        decisions = self.decisions[split]
        world = self._get_dp_group_size()
        shard = pad_and_shard_records(range(len(decisions)), rank=self._get_dp_rank(), world_size=world)
        local = {index: self._logits(decisions[index].prompt).log_softmax(-1).tolist() for index in shard}
        gathered: list[dict[int, list[float]]] = [local]
        if dist.is_initialized() and world > 1:
            gathered = [{} for _ in range(world)]
            dist.all_gather_object(gathered, local, group=self._get_dp_group())
        log_probs: dict[int, list[float]] = {}
        for part in gathered:
            log_probs.update(part)
        evaluator = DecisionEvaluator(slice_keys=self.slice_keys)
        for index, decision in enumerate(decisions):
            target = decision.target
            evaluator.add(
                str(decision.record["id"]),
                target.question_id,
                target.option_ids,
                [math.exp(value) for value in log_probs[index]],
                target.label,
                target.rewards,
                decision.record.get("meta"),
            )
        self.model.train()
        return evaluator.summary(), evaluator.predictions()

    # ------------------------------------------------------------------ stage I/O
    def _save_stage(self, path: Path, validation: dict[str, Any] | None) -> None:
        """Publish the trainable weights (LoRA adapter or consolidated full weights) as an immutable stage."""
        self.checkpointer.lifecycle.reserve(str(path))
        self.checkpointer.save_model(self.model, str(path), peft_config=self.peft_config)
        value = selection_value(validation["correct"], validation["nll"]) if validation else None

        def write_metadata() -> None:
            (path / "stage.json").write_text(json.dumps({**self.identity, "validation": validation}, indent=2))
            if value is not None:
                (path / "losses.json").write_text(json.dumps({"selection": value}))

        self.checkpointer.lifecycle.run_coordinator_step(write_metadata, description="write decision stage metadata")
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
            raise ValueError(f"{path} is not a complete decision stage")
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
            self._load_stage(self._train())
        for split in ("validation", "test"):
            if split in self.decisions:
                self._report(split)
        self.checkpointer.finalize()
        if dist.is_initialized():
            dist.barrier()

    def _report(self, split: str) -> None:
        """Write ``metrics_{split}.json`` (evaluator summary) and ``predictions_{split}.jsonl`` on the main rank."""
        summary, predictions = self.evaluate(split)
        summary["run"] = {
            "mode": self.mode,
            "split": split,
            "init_from": str(self.cfg.get("init_from", None)),
            "skipped_questions": self.skipped[split],
        }
        if self.dist_env.is_main:
            save_summary(summary, self.output_dir / f"metrics_{split}.json")
            with (self.output_dir / f"predictions_{split}.jsonl").open("w") as handle:
                for row in predictions:
                    handle.write(json.dumps(row) + "\n")
            overall = summary["overall"]
            logger.info(
                "[%s] %s accuracy=%.4f nll=%.4f",
                self.mode,
                split,
                overall.get("accuracy", float("nan")),
                overall.get("nll", float("nan")),
            )

    def _log(self, name: str, record: dict[str, Any]) -> None:
        if self.dist_env.is_main:
            with (self.output_dir / name).open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            logger.info("%s %s", name.split(".")[0], json.dumps(record))

    def _validate(self, step: int) -> dict[str, Any] | None:
        """Selection record of the current weights on the validation split, or ``None`` without one."""
        if "validation" not in self.decisions:
            return None
        overall = self.evaluate("validation")[0]["overall"]
        labelled = sum(decision.target.label >= 0 for decision in self.decisions["validation"])
        record = {
            "accuracy": overall["accuracy"],
            "correct": round(overall["accuracy"] * labelled),
            "nll": overall["nll"],
            "decisions": labelled,
        }
        self._log("validation.jsonl", {"step": step, **record})
        return record

    def _train(self) -> Path:
        """Train; returns the validation-selected stage (the last stage when there is no validation split)."""
        world = self._get_dp_group_size()
        rank = self._get_dp_rank()
        batch = int(self.cfg.get("train.global_batch_size", 16))
        if batch <= 0 or batch % world:
            raise ValueError(f"train.global_batch_size={batch} must be a positive multiple of the DP size {world}")
        decisions = self.decisions["train"]
        if len(decisions) < batch:
            raise ValueError(f"{len(decisions)} training decisions < train.global_batch_size {batch}")
        accum = batch // world
        steps_per_epoch = len(decisions) // batch
        total = int(self.cfg.get("train.max_steps", 0) or 0) or max(
            1, int(steps_per_epoch * float(self.cfg.get("train.epochs", 1.0)))
        )
        warmup = max(1, int(float(self.cfg.get("train.warmup_fraction", 0.05)) * total))
        canonical_fraction = float(self.cfg.get("train.canonical_fraction", 1.0))
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
                "train_decisions": len(decisions),
                "train_records": len(self.records["train"]),
                "skipped_train_questions": self.skipped["train"],
                "trainable_parameters": sum(p.numel() for p in trainable),
                **self.identity,
            }
            (self.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))

        start = self._validate(0)
        # LoRA: the start is a saved, selectable stage (B is zero, so the adapter equals the base). Full fine-tuning
        # does not save the base weights, so selection is among trained stages, matching LOWEST_VAL.
        best: tuple[dict[str, Any] | None, int] = (start, 0) if is_lora else (None, 0)
        if is_lora:
            self._save_stage(self.output_dir / "step_00000000", start)
        order = list(range(len(decisions)))
        random.Random(self.seed).shuffle(order)
        step, epoch, cursor = 0, 0, 0
        self.model.train()
        while step < total:
            started = time.monotonic()
            if cursor + batch > len(order):
                epoch += 1
                cursor = 0
                order = list(range(len(decisions)))
                random.Random(self.seed + epoch).shuffle(order)
            window = order[cursor : cursor + batch]
            cursor += batch
            reads = []
            for j in range(accum):
                index = window[j * world + rank]
                decision = decisions[index]
                seed = index * 1000003 + self.seed + epoch
                shuffle = None if random.Random(seed).random() < canonical_fraction else seed
                prompt = (
                    encode_question(
                        self.tokenizer,
                        self.contract,
                        decision.record,
                        decision.prompt.question_id,
                        mask_token_id=self.mask_token_id,
                        max_tokens=self.max_tokens,
                        shuffle_seed=shuffle,
                    )
                    or decision.prompt
                )
                targets = self._targets(decision.record, self._target(decision.record, prompt), self.device)
                reads.append((decision, prompt, targets, self.loss_fn.normalization_weight(targets)))
            # Normalize over the eligible question weights of the whole window on every DP rank.
            window_weight = torch.stack([weight for *_, weight in reads]).sum()
            if dist.is_initialized() and world > 1:
                dist.all_reduce(window_weight, group=self._get_dp_group())
            window_loss = torch.zeros((), device=self.device)
            for decision, prompt, targets, weight in reads:
                scores = QuestionLogits(
                    str(decision.record["id"]), prompt.question_id, prompt.labels, self._logits(prompt)
                )
                assembled = assemble_decision_batch(logits=[[scores]], targets=targets)
                loss, _ = self.loss_fn(assembled.logits, assembled.targets)
                share = weight / window_weight
                # FSDP averages gradients over the DP ranks; scaling by the DP size keeps each read's window share.
                (loss * (world * share)).backward()
                window_loss += loss.detach() * share
            grad_norm = clip_grad_norm(max_grad_norm, [self.model], device_mesh=self.device_mesh)
            for scheduler in schedulers:
                scheduler.step(1)  # lr for this update: warmup (step+1)/warmup, then cosine to min_lr
            for optimizer in self.optimizer:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if dist.is_initialized() and world > 1:
                dist.all_reduce(window_loss, group=self._get_dp_group())
            grad_norm = grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
            self._log(
                "training.jsonl",
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": round(float(window_loss), 5),
                    "grad_norm": round(float(grad_norm), 4),
                    "lr": self.optimizer[0].param_groups[0]["lr"],
                    "seconds": round(time.monotonic() - started, 2),
                },
            )
            if step % val_every == 0 or step == total:
                record = self._validate(step)
                self._save_stage(self.output_dir / f"step_{step:08d}", record)
                if record is not None and (
                    best[0] is None
                    or selection_value(record["correct"], record["nll"])
                    < selection_value(best[0]["correct"], best[0]["nll"])
                ):
                    best = (record, step)
        if self.dist_env.is_main:
            selected = {"step": best[1], "validation": best[0]}
            (self.output_dir / "selected.json").write_text(json.dumps(selected, indent=2))
        if best[0] is None:
            return self.output_dir / f"step_{total:08d}"
        return self.output_dir / f"step_{best[1]:08d}"


def main(config_path: str | None = None) -> None:
    """CLI entry point: ``python -m nemo_automodel.recipes.decision.train_nemotron_decision --config <yaml>``."""
    cfg = parse_args_and_load_config(config_path)
    recipe = TrainNemotronDecisionRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
