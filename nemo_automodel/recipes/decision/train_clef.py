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

"""Post-train Cloudflare Clef (Qwen3.5 backbone + joint schema head) on SystemOne records.

Clef reads a state and a schema of typed questions and returns one logit per allowed option in a single forward pass:
the Qwen3.5 backbone produces final hidden states and a small transformer "joint schema head" routes evidence from the
state to every question and scores the options jointly.

This recipe follows the post-training recipe Cloudflare describe for Clef - LoRA on the backbone plus a fully trained
head - on top of NeMo AutoModel's FSDP2 + PEFT stack:

* The backbone is built with ``NeMoAutoModelForImageTextToText`` (FSDP2, LoRA, activation checkpointing).
* The joint schema head (~128M parameters) is replicated on every rank in fp32; its gradients are averaged across the
  data-parallel group after backward, like the projection head in ``distill_bi_encoder``.
* The head consumes the final hidden states returned by the backbone's public ``output_hidden_states`` interface.
* The head reads LM-head rows for option tokens.  The LM head is frozen and sharded, so the few rows it needs are
  gathered once into a ``LexicalTable``.
* ``mode: eval`` scores the model on validation/test (the *before* numbers); ``mode: train`` runs SFT
  (``loss.mode: sft``) or RL (``loss.mode: rl``).  For RL the reference log-probabilities of the starting policy are
  pre-computed in one inference pass, so no second model copy is needed.
* Each immutable stage uses the native PEFT checkpoint and DCP joint-head state. ``init_from`` initializes a new
  stage's weights (SFT -> RL); optimizers and schedules start afresh. Publisher export is a separate operation.

Run with::

    automodel examples/decision/clef_soc_triage_lora.yaml --nproc-per-node 8 --mode train --loss.mode sft
"""

from __future__ import annotations

import copy
import json
import logging
import math
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
from nemo_automodel.components.models.clef.joint_schema_head import (
    EncodedRecord,
    JointSchemaHead,
    build_lexical_table,
    clef_option_orders,
    encode_record,
    load_joint_head,
)
from nemo_automodel.components.models.clef.stage import ClefStageManifest
from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig
from nemo_automodel.components.training.utils import clip_grad_norm
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.base_recipe import BaseRecipe

logger = logging.getLogger(__name__)

MODES = ("eval", "train")


class TrainClefDecisionRecipe(BaseRecipe):
    """LoRA + joint-schema-head post-training for Clef decision models (see module docstring)."""

    def __init__(self, cfg: RecipeConfig | ConfigNode) -> None:
        """Store the recipe configuration (a ``ConfigNode`` parsed from YAML)."""
        self.cfg = cfg if isinstance(cfg, RecipeConfig) else RecipeConfig(cfg)

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        """Build the distributed backbone, the head, the encoded datasets and the optimizers."""
        self.dist_env = initialize_distributed(
            backend=self.cfg.get("dist_env.backend", "nccl"),
            timeout_minutes=self.cfg.get("dist_env.timeout_minutes", 30),
        )
        setup_logging()
        self.mode = self.cfg.get("mode", "train")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        self._require_stage_checkpoints()
        self.loss_config: DecisionLossConfig = (
            self.cfg.loss.instantiate() if self.cfg.get("loss", None) is not None else DecisionLossConfig()
        )
        self.tag = "eval" if self.mode == "eval" else self.loss_config.mode
        self.seed = int(self.cfg.get("seed", 1234))
        self.output_dir = Path(self.cfg.get("output_dir", f"runs/clef_{self.tag}"))
        self.device = self.dist_env.device
        torch.manual_seed(self.seed)

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
            raise ValueError("Clef decision post-training supports FSDP2 data parallelism only (tp=cp=pp=1)")

        self.peft_config = self.cfg.peft.instantiate() if self.cfg.get("peft", None) is not None else None
        self.model = self.cfg.model.instantiate(
            freeze_config=self.cfg.freeze_config,
            peft_config=self.peft_config,
            distributed_setup=self.distributed_setup,
        )

        from transformers import AutoTokenizer

        model_path = self.cfg.get("model.pretrained_model_name_or_path")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=self.cfg.get("model.revision", None))
        self.pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        self.head: JointSchemaHead = load_joint_head(
            self.cfg.get("head.path", None) or model_path,
            dtype=torch.float32,
            dropout=self.cfg.get("head.dropout", 0.0),
            revision=self.cfg.get("model.revision", None),
        ).to(self.device)
        # Preserve the config's normalized enums; dataclasses.replace reruns raw-input validation.
        checkpoint_config = copy.copy(self.cfg.checkpoint)
        checkpoint_config.checkpoint_dir = str(self.output_dir)
        self.checkpointer = checkpoint_config.build(
            dp_rank=self._get_dp_rank(),
            tp_rank=0,
            pp_rank=0,
            moe_mesh=self.moe_mesh,
            process_group=self._dp_group(),
        )
        init_from = self.cfg.get("init_from", None)
        if init_from:
            self._load_stage(Path(init_from))

        max_length = int(self.cfg.get("data.max_length", 4096))
        self.splits: dict[str, list[tuple[dict[str, Any], EncodedRecord]]] = {}
        for split in ("train", "validation", "test"):
            path = self.cfg.get(f"data.{split}", None)
            if not path:
                continue
            records = load_records(path, limit=self.cfg.get(f"data.limit_{split}", None))
            if not records:
                raise ValueError(f"data.{split} must contain at least one record")
            for record in records:
                validate_record(record)
            self.splits[split] = [(r, encode_record(self.tokenizer, r, max_length=max_length)) for r in records]
        if self.mode == "train" and "train" not in self.splits:
            raise ValueError("data.train is required for mode=train")
        self.slice_keys = list(self.cfg.get("data.slice_keys", []) or [])
        self.lexical = build_lexical_table(self.model, [e for split in self.splits.values() for _, e in split])

        self.optimizer = None
        if self.mode == "train":
            train_backbone = any(parameter.requires_grad for parameter in self.model.parameters())
            if train_backbone and self.peft_config is None:
                raise ValueError("CLEF backbone adaptation requires peft; otherwise freeze the backbone")
            if self.cfg.head_optimizer is None or (train_backbone and self.cfg.optimizer is None):
                raise ValueError("training requires head_optimizer and, for backbone adaptation, optimizer")
            self.optimizer = (
                self.cfg.optimizer.build(self.model, device_mesh=self.device_mesh, is_peft=True)
                if train_backbone
                else []
            )
            self.head_optimizer = self.cfg.head_optimizer.build(self.head)[0]

    def _dp_group(self) -> dist.ProcessGroup | None:
        return self._get_dp_group()

    # ------------------------------------------------------------------ stage I/O
    def _save_stage(self, path: Path, *, selection_score: float | None = None) -> None:
        """Publish immutable weights through native checkpoint/PEFT components.

        Args:
            path: Fresh checkpoint directory, shared by all data-parallel ranks.
            selection_score: Higher-is-better validation score for the native LOWEST_VAL pointer.
        """
        if not self.checkpointer.config.enabled:
            return
        has_adapter = any(parameter.requires_grad for parameter in self.model.parameters())
        self.checkpointer.lifecycle.reserve(str(path))
        if has_adapter:
            self.checkpointer.save_model(self.model, str(path), peft_config=self.peft_config)
        self.checkpointer.save_distributed_state(self.head, ClefStageManifest.HEAD_STATE_NAME, str(path))
        manifest = ClefStageManifest(has_adapter, self.head.config, self.tag)
        metric_key = f"negative_{self.cfg.get('train.eval_metric', 'accuracy')}"

        def write_metadata() -> None:
            manifest.write(path)
            if selection_score is not None:
                (path / "losses.json").write_text(json.dumps({metric_key: -selection_score}))

        self.checkpointer.lifecycle.run_coordinator_step(
            write_metadata,
            description="write CLEF stage initialization metadata",
        )
        self.checkpointer.async_wait()
        self.checkpointer.lifecycle.publish(
            str(path),
            best_val_metric=-selection_score if selection_score is not None else None,
            metric_key=metric_key,
        )

    def _load_stage(self, path: Path) -> None:
        """Initialize model/head weights, leaving optimizer and schedule state fresh.

        Args:
            path: Published native CLEF checkpoint or its LOWEST_VAL/LATEST pointer.
        """
        if path.name in {"LOWEST_VAL", "LATEST"}:
            resolved = read_checkpoint_pointer(path.parent, path.name)
            if resolved is None:
                raise FileNotFoundError(f"stage pointer {path} has no published target")
            path = resolved
        self.checkpointer.lifecycle.validate_checkpoint_dir_exists(str(path), str(path))
        if is_checkpoint_incomplete(path):
            raise ValueError(f"cannot initialize from incomplete checkpoint {path}")
        manifest = ClefStageManifest.load(path)
        if manifest.backbone_adapter:
            if self.peft_config is None:
                raise ValueError("initializing a backbone adapter requires the matching peft configuration")
            self.checkpointer.load_model(self.model, str(path / "model"))
        self.checkpointer.load_distributed_state(self.head, ClefStageManifest.HEAD_STATE_NAME, str(path))
        if self.dist_env.is_main:
            logger.info("Initialized stage weights from %s; optimizer/scheduler state is fresh", path)

    # ------------------------------------------------------------------ forward
    def _forward(self, batch: list[tuple[dict[str, Any], EncodedRecord]]) -> list[list[torch.Tensor]]:
        """Run backbone + head on a batch.

        Args:
            batch: Record identities and publisher-ordered encoded question/option spans.

        Returns:
            For each record, a list with one tensor of shape [options_i] (fp32 logits) per question, in Clef order.
        """
        return self.head.score_records(
            self.model,
            [encoded for _, encoded in batch],
            self.lexical,
            pad_token_id=self.pad_token_id,
            device=self.device,
        )

    def _shard(self, items: list, shuffle: bool, epoch: int = 0) -> list:
        """Equal-length per-rank shard (FSDP needs every rank to run the same number of forwards)."""
        return pad_and_shard_records(
            items,
            rank=self._get_dp_rank(),
            world_size=self._get_dp_group_size(),
            shuffle_seed=self.seed + epoch if shuffle else None,
        )

    @torch.no_grad()
    def _score(
        self, items: list[tuple[dict[str, Any], EncodedRecord]], batch_size: int
    ) -> dict[str, list[list[float]]]:
        """Per-record log-probabilities ``{record_id: [[logp per option] per question]}``, gathered on all ranks."""
        self.model.eval()
        self.head.eval()
        local: dict[str, list[list[float]]] = {}
        shard = self._shard(items, shuffle=False)
        for b in range(0, len(shard), batch_size):
            batch = shard[b : b + batch_size]
            for (record, _), logits in zip(batch, self._forward(batch), strict=True):
                local[str(record["id"])] = [q.float().log_softmax(-1).tolist() for q in logits]
        gathered: list[dict[str, list[list[float]]]] = [{} for _ in range(self._get_dp_group_size())]
        if dist.is_initialized() and self._get_dp_group_size() > 1:
            dist.all_gather_object(gathered, local, group=self._dp_group())
        else:
            gathered = [local]
        merged: dict[str, list[list[float]]] = {}
        for part in gathered:
            merged.update(part)
        return merged

    def evaluate(self, split: str) -> dict[str, Any]:
        """Score ``split`` and return the metrics summary (identical on every rank)."""
        items = self.splits[split]
        logprobs = self._score(items, int(self.cfg.get("eval.batch_size", 4)))
        evaluator = DecisionEvaluator(slice_keys=self.slice_keys)
        for record, encoded in items:
            targets = question_targets(record, clef_option_orders(encoded))
            probs = [[math.exp(x) for x in q] for q in logprobs[str(record["id"])]]
            evaluator.add_batch([record], [targets], [probs])
        return evaluator.summary()

    # ------------------------------------------------------------------ main loop
    def _require_stage_checkpoints(self) -> None:
        """Keep training, validation selection, and final reporting on the same saved stage."""
        if self.mode == "train" and not self.cfg.checkpoint.enabled:
            raise ValueError("CLEF training requires checkpoint.enabled=true for validation-selected stages")

    def run_train_validation_loop(self) -> None:
        """Evaluate (``mode: eval``) or train (``mode: train``), then report validation/test metrics."""
        self._require_stage_checkpoints()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == "train":
            self._train()
            best = read_checkpoint_pointer(self.output_dir, "LOWEST_VAL")
            if best is not None:
                self._load_stage(best)
        for split in ("validation", "test"):
            if split in self.splits:
                self._report(split)
        self.checkpointer.finalize()
        if dist.is_initialized():
            dist.barrier()

    def _report(self, split: str) -> None:
        summary = self.evaluate(split)
        summary["run"] = {"stage": self.tag, "split": split, "init_from": str(self.cfg.get("init_from", None))}
        if self.dist_env.is_main:
            save_summary(summary, self.output_dir / f"metrics_{split}.json")
            o = summary["overall"]
            logger.info(
                "[%s/%s] accuracy=%.4f exact_match=%.4f reward=%.4f brier=%.4f ece=%.4f",
                self.tag,
                split,
                o.get("accuracy", float("nan")),
                o.get("exact_match", float("nan")),
                o.get("reward", float("nan")),
                o.get("brier", float("nan")),
                o.get("ece", float("nan")),
            )

    def _val_score(self, metric: str) -> float | None:
        if metric not in {"accuracy", "exact_match", "reward", "expected_reward"}:
            raise ValueError("train.eval_metric must be accuracy, exact_match, reward, or expected_reward (maximized)")
        if "validation" not in self.splits:
            return None
        overall = self.evaluate("validation")["overall"]
        if metric not in overall:
            raise ValueError(f"validation metric {metric!r} is unavailable; available metrics: {sorted(overall)}")
        return overall[metric]

    def _train(self) -> None:
        loss_cfg = self.loss_config
        loss_fn = loss_cfg.build()
        epochs = int(self.cfg.get("train.epochs", 2))
        local_bs = int(self.cfg.get("train.local_batch_size", 2))
        global_bs = int(self.cfg.get("train.global_batch_size", 32))
        world = self._get_dp_group_size()
        if local_bs <= 0 or global_bs <= 0:
            raise ValueError("train.local_batch_size and train.global_batch_size must be positive")
        if global_bs % (local_bs * world):
            raise ValueError("train.global_batch_size must be divisible by local_batch_size * DP world size")
        accum = global_bs // (local_bs * world)
        metric = self.cfg.get("train.eval_metric", "accuracy")
        val_every = int(self.cfg.get("train.val_every_steps", 50))
        max_grad_norm = float(self.cfg.get("train.max_grad_norm", 1.0))
        train_items = self.splits["train"]

        reference = None
        if loss_cfg.mode == "rl" and loss_cfg.kl_coef > 0:
            start = time.perf_counter()
            reference = self._score(train_items, int(self.cfg.get("eval.batch_size", 4)))
            if self.dist_env.is_main:
                logger.info(
                    "Pre-computed reference log-probs for %d records (%.0fs)",
                    len(reference),
                    time.perf_counter() - start,
                )

        steps_per_epoch = len(self._shard(train_items, shuffle=False)) // (local_bs * accum)
        if steps_per_epoch == 0:
            raise ValueError(
                "data.train contains no complete optimizer batch; reduce train.global_batch_size or add records"
            )
        total = epochs * steps_per_epoch
        step_schedule = StepSchedulerConfig(
            global_batch_size=global_bs, num_epochs=epochs, max_steps=total, preemption_signal=None
        ).build(dataloader=None, dp_group_size=world, local_batch_size=local_bs, process_group=self._dp_group())
        schedulers = (
            self.cfg.lr_scheduler.build([*self.optimizer, self.head_optimizer], step_schedule)
            if self.cfg.lr_scheduler is not None
            else []
        )
        generator = torch.Generator(device=self.device).manual_seed(self.seed + self._get_dp_rank())

        best_score = self._val_score(metric)
        if self.dist_env.is_main:
            logger.info("[%s] step 0 val %s=%s (%d steps, accum %d)", self.tag, metric, best_score, total, accum)
        self._save_stage(self.output_dir / "step_00000000", selection_score=best_score)
        log = open(self.output_dir / "training_log.jsonl", "w") if self.dist_env.is_main else None
        step = 0
        for epoch in range(epochs):
            shard = self._shard(train_items, shuffle=True, epoch=epoch)
            micro = [shard[i : i + local_bs] for i in range(0, len(shard) - local_bs + 1, local_bs)]
            for s in range(0, len(micro) - accum + 1, accum):
                self.model.train(bool(self.optimizer))
                self.head.train()
                metrics_sum: dict[str, float] = {}
                window = micro[s : s + accum]
                window_targets = []
                local_weights = []
                for batch in window:
                    questions = [question_targets(record, clef_option_orders(encoded)) for record, encoded in batch]
                    ref_rows = None
                    if reference is not None:
                        ref_rows = [
                            [
                                QuestionLogits(
                                    str(record["id"]),
                                    target.question_id,
                                    target.option_ids,
                                    torch.tensor(row, device=self.device),
                                )
                                for target, row in zip(record_questions, reference[str(record["id"])], strict=True)
                            ]
                            for (record, _), record_questions in zip(batch, questions, strict=True)
                        ]
                    targets = DecisionTargets.from_questions(
                        record_ids=[str(record["id"]) for record, _ in batch],
                        questions=questions,
                        question_weights=loss_cfg.question_weights,
                        reference_logprobs=ref_rows,
                        device=self.device,
                    )
                    window_targets.append(targets)
                    local_weights.append(loss_fn.normalization_weight(targets))
                global_weight = torch.stack(local_weights).sum()
                if dist.is_initialized() and world > 1:
                    dist.all_reduce(global_weight, op=dist.ReduceOp.SUM, group=self._dp_group())
                if float(global_weight) <= 0:
                    raise ValueError("optimizer window has no supervised labels or explicit rewards on any DP rank")
                for batch, targets, local_weight in zip(window, window_targets, local_weights):
                    per_record = self._forward(batch)
                    identified = [
                        [
                            QuestionLogits(str(record["id"]), question.question_id, question.option_ids, values)
                            for question, values in zip(encoded.questions, record_logits, strict=True)
                        ]
                        for (record, encoded), record_logits in zip(batch, per_record, strict=True)
                    ]
                    assembled = assemble_decision_batch(logits=identified, targets=targets)
                    loss, metrics = loss_fn(assembled.logits, assembled.targets, generator=generator)
                    # FSDP and the replicated head each average over DP; compensate once,
                    # normalizing over all eligible questions in this complete optimizer window.
                    (loss * (world * local_weight / global_weight)).backward()
                    for k, v in {"loss": loss.detach(), **metrics}.items():
                        metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v) / accum
                self._sync_head_grads()
                # One coefficient preserves relative scaling and enforces a combined backbone/head budget.
                grad_norm = clip_grad_norm(max_grad_norm, [self.model, self.head], device_mesh=self.device_mesh)
                for o in [*self.optimizer, self.head_optimizer]:
                    o.step()
                    o.zero_grad(set_to_none=True)
                for sch in schedulers:
                    sch.step(1)
                step += 1
                if log is not None:
                    grad_norm = grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                    row = {"epoch": epoch, "step": step, "grad_norm": float(grad_norm), **metrics_sum}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if step % 10 == 0:
                        logger.info(
                            "[%s] step %d/%d %s", self.tag, step, total, {k: round(v, 4) for k, v in row.items()}
                        )
                if step % val_every == 0 or step == total:
                    score = self._val_score(metric)
                    if self.dist_env.is_main:
                        logger.info("[%s] step %d val %s=%s (best %s)", self.tag, step, metric, score, best_score)
                        if log is not None:
                            log.write(json.dumps({"step": step, f"val_{metric}": score}) + "\n")
                    if best_score is None or (score is not None and score > best_score):
                        best_score = score
                    self._save_stage(self.output_dir / f"step_{step:08d}", selection_score=score)
        if log is not None:
            log.close()

    def _sync_head_grads(self) -> None:
        """Average the replicated head's gradients over the data-parallel group."""
        world = self._get_dp_group_size()
        if world == 1 or not dist.is_initialized():
            return
        group = self._dp_group()
        for p in self.head.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)
            p.grad.div_(world)


def main(config_path: str | None = None) -> None:
    """CLI entry point: ``python -m nemo_automodel.recipes.decision.train_clef --config <yaml>``."""
    cfg = parse_args_and_load_config(config_path)
    recipe = TrainClefDecisionRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
