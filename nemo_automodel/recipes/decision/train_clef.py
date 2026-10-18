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
* Each stage writes ``adapter.safetensors`` (LoRA weights, by parameter name) and the head in the release layout
  (``joint_head.safetensors`` + ``joint_head_config.json``); ``init_from`` chains stages (SFT -> RL).

Run with::

    automodel examples/decision/clef_soc_triage_lora.yaml --nproc-per-node 8 --mode train --loss.mode sft
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.decision.losses import DecisionLossConfig, build_targets, pad_logits
from nemo_automodel.components.decision.metrics import DecisionEvaluator, save_summary
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.models.clef.joint_schema_head import (
    EncodedRecord,
    JointSchemaHead,
    LexicalTable,
    clef_option_orders,
    collate_records,
    encode_record,
    load_joint_head,
    option_token_ids,
    save_joint_head,
)
from nemo_automodel.components.training.utils import clip_grad_norm
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.shared.decision_records import (
    load_records,
    pad_and_shard_records,
    question_targets,
    validate_record,
)

logger = logging.getLogger(__name__)

MODES = ("eval", "train")
_WRAPPER_PREFIXES = ("_checkpoint_wrapped_module.", "_orig_mod.", "_fsdp_wrapped_module.")


def _canonical_name(name: str) -> str:
    """Parameter name without activation-checkpoint / compile wrapper prefixes."""
    for prefix in _WRAPPER_PREFIXES:
        name = name.replace(prefix, "")
    return name


def _cosine_with_warmup(total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))

    return fn


class TrainClefDecisionRecipe(BaseRecipe):
    """LoRA + joint-schema-head post-training for Clef decision models (see module docstring)."""

    def __init__(self, cfg: Any):
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
        self.lexical = self._build_lexical_table()

        self.optimizer = None
        if self.mode == "train":
            self.optimizer = (
                self.cfg.optimizer.build(self.model, device_mesh=self.device_mesh, is_peft=self.peft_config is not None)
                if any(parameter.requires_grad for parameter in self.model.parameters())
                else []
            )
            self.head_optimizer = torch.optim.AdamW(
                self.head.parameters(),
                lr=float(self.cfg.get("train.head_lr", 5e-5)),
                weight_decay=float(self.cfg.get("train.head_weight_decay", 0.0)),
            )

    def _dp_group(self):
        return self._get_dp_group()

    def _build_lexical_table(self) -> LexicalTable:
        """Gather the frozen LM-head rows of every option token in the data (once)."""
        ids = option_token_ids([e for split in self.splits.values() for _, e in split])
        weight = self.model.get_output_embeddings().weight
        with torch.no_grad():
            full = weight.full_tensor() if isinstance(weight, DTensor) else weight
            rows = full[torch.tensor(ids, device=full.device)].detach().clone()
            del full
        return LexicalTable(ids, rows).to(self.device)

    # ------------------------------------------------------------------ stage I/O
    def _trainable_named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [(_canonical_name(n), p) for n, p in self.model.named_parameters() if p.requires_grad]

    def _save_stage(self, path: Path) -> None:
        """Write LoRA weights (gathered to full tensors) and the head to ``path`` on rank 0."""
        from safetensors.torch import save_file

        tensors = {}
        for name, p in self._trainable_named_parameters():
            full = p.full_tensor() if isinstance(p, DTensor) else p
            if self.dist_env.is_main:
                tensors[name] = full.detach().float().cpu().contiguous()
        if self.dist_env.is_main:
            path.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(path / "adapter.safetensors"))
            peft = self.peft_config
            meta = {"dim": getattr(peft, "dim", None), "alpha": getattr(peft, "alpha", None), "stage": self.tag}
            (path / "adapter_config.json").write_text(json.dumps(meta, indent=2))
            save_joint_head(self.head, path, dtype=torch.float32)
        if dist.is_initialized():
            dist.barrier()

    def _load_stage(self, path: Path) -> None:
        """Load LoRA weights (re-sharded to each parameter's placement) and the head from a previous stage."""
        from safetensors.torch import load_file

        adapter = path / "adapter.safetensors"
        if adapter.exists():
            state = load_file(str(adapter))
            params = dict(self._trainable_named_parameters())
            unexpected = sorted(set(state) - set(params))
            if unexpected:
                raise KeyError(f"{adapter} has tensors outside this trainability policy, e.g. {unexpected[:3]}")
            missing = sorted(set(params) - set(state))
            if missing:
                raise KeyError(f"{adapter} is missing {len(missing)} trainable tensors, e.g. {missing[:3]}")
            with torch.no_grad():
                for name, p in params.items():
                    value = state[name].to(device=self.device, dtype=p.dtype)
                    if isinstance(p, DTensor):
                        value = distribute_tensor(value, p.device_mesh, p.placements)
                    p.copy_(value)
        head = load_joint_head(path, dtype=torch.float32, dropout=self.cfg.get("head.dropout", 0.0))
        self.head.load_state_dict(head.state_dict())
        if self.dist_env.is_main:
            logger.info("Loaded stage weights from %s", path)

    # ------------------------------------------------------------------ forward
    def _forward(self, batch: list[tuple[dict[str, Any], EncodedRecord]]) -> list[list[torch.Tensor]]:
        """Run backbone + head on a batch.

        Returns:
            For each record, a list with one tensor of shape [options_i] (fp32 logits) per question, in Clef order.
        """
        collated = collate_records([e for _, e in batch], self.pad_token_id, self.device)
        output = self.model(
            input_ids=collated["input_ids"],
            attention_mask=collated["attention_mask"],
            use_cache=False,
            logits_to_keep=1,
            output_hidden_states=True,
            **collated["media"],
        )
        if output.hidden_states is None:
            raise RuntimeError("the backbone must return final hidden states for CLEF training")
        hidden = output.hidden_states[-1]
        return self.head(hidden, collated["input_ids"], collated["attention_mask"], collated["records"], self.lexical)

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
            for (record, _), logits in zip(batch, self._forward(batch)):
                local[str(record["id"])] = [q.float().log_softmax(-1).tolist() for q in logits]
        gathered: list[dict] = [None] * self._get_dp_group_size()  # type: ignore[list-item]
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
    def run_train_validation_loop(self) -> None:
        """Evaluate (``mode: eval``) or train (``mode: train``), then report validation/test metrics."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == "train":
            self._train()
            best = self.output_dir / "best"
            if best.exists():
                self._load_stage(best)
        for split in ("validation", "test"):
            if split in self.splits:
                self._report(split)
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
        schedule = _cosine_with_warmup(total, float(self.cfg.get("train.warmup_ratio", 0.05)))
        schedulers = [torch.optim.lr_scheduler.LambdaLR(o, schedule) for o in self.optimizer]
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(self.head_optimizer, schedule))
        generator = torch.Generator(device=self.device).manual_seed(self.seed + self._get_dp_rank())

        best_score = self._val_score(metric)
        if self.dist_env.is_main:
            logger.info("[%s] step 0 val %s=%s (%d steps, accum %d)", self.tag, metric, best_score, total, accum)
        self._save_stage(self.output_dir / "best")
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
                    flat_targets = []
                    ref_rows = [] if reference is not None else None
                    for record, encoded in batch:
                        flat_targets.extend(question_targets(record, clef_option_orders(encoded)))
                        if ref_rows is not None:
                            ref_rows.extend(reference[str(record["id"])])
                    targets = build_targets(flat_targets, loss_cfg.question_weights, ref_rows).to(self.device)
                    window_targets.append(targets)
                    local_weights.append(loss_fn.normalization_weight(targets))
                global_weight = torch.stack(local_weights).sum()
                if dist.is_initialized() and world > 1:
                    dist.all_reduce(global_weight, op=dist.ReduceOp.SUM, group=self._dp_group())
                if float(global_weight) <= 0:
                    raise ValueError("optimizer window has no supervised labels or explicit rewards on any DP rank")
                for batch, targets, local_weight in zip(window, window_targets, local_weights):
                    per_record = self._forward(batch)
                    flat_logits = [q for rec in per_record for q in rec]
                    logits, _ = pad_logits(flat_logits)
                    loss, metrics = loss_fn(logits, targets, generator=generator)
                    # FSDP and the replicated head each average over DP; compensate once,
                    # normalizing over all eligible questions in this complete optimizer window.
                    (loss * (world * local_weight / global_weight)).backward()
                    for k, v in {"loss": loss.detach(), **metrics}.items():
                        metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v) / accum
                self._sync_head_grads()
                # Sharding-aware clipping for the FSDP2 (DTensor) LoRA parameters; the replicated head is clipped
                # separately because its gradients are already identical on every rank.
                grad_norm = clip_grad_norm(max_grad_norm, [self.model], device_mesh=self.device_mesh)
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_grad_norm)
                for o in [*self.optimizer, self.head_optimizer]:
                    o.step()
                    o.zero_grad(set_to_none=True)
                for sch in schedulers:
                    sch.step()
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
                        self._save_stage(self.output_dir / "best")
        self._save_stage(self.output_dir / "last")
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
