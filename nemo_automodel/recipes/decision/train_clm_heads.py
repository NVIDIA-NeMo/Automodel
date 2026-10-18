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

"""Post-train Contrastive Language Model (CLM) decision heads on SystemOne records.

CLM keeps its Qwen3-8B encoder frozen (the released heads, the ``clm-serve`` action cache and the vLLM pooling server
all depend on the exact encoder embedding space), so post-training touches only the two projection heads and the
logit scale.  The recipe therefore runs in two phases:

1. **Embed** every unique state / option text once with ``NeMoAutoModelBiEncoder`` (last-token pooling, L2
   normalisation and causal attention, matching the ``clm-serve`` input contract).  Texts are sharded
   across all ranks and cached on disk, so re-running SFT / RL / eval never re-encodes.
2. **Train or evaluate** the heads on rank 0 (they have ~20M parameters):

   - ``mode: eval`` scores ``heads.init_checkpoint`` on validation/test (the *before* numbers).
   - ``mode: train`` with ``loss.mode: sft`` fits a custom typed-decision SFT objective: label-smoothed CE plus Brier and RPS terms.
     This does not implement the publisher's soft preference-target fine-tuning objective.
   - ``mode: train`` with ``loss.mode: rl`` maximises the per-option reward tables (exact expected reward or GRPO)
     with a KL penalty to the heads the stage started from.

The trained heads are written in the ``contrastive-lm`` checkpoint format, so ``CLM_CKPT=<out>/heads.pt clm-serve``
serves them unchanged.

Run with::

    automodel examples/decision/clm_soc_triage.yaml --nproc-per-node 8 --mode train --loss.mode sft
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.decision.losses import DecisionLossConfig, build_targets, pad_logits
from nemo_automodel.components.decision.metrics import DecisionEvaluator, save_summary
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.models.clm.heads import (
    CLM_MAX_TOKENS,
    CLMEmbeddingCacheConfig,
    CLMHeadPair,
    EmbeddingTable,
    clm_batch,
    collect_clm_texts,
    load_clm_heads,
    save_clm_heads,
)
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.shared.decision_records import load_records, question_targets, validate_record

logger = logging.getLogger(__name__)

MODES = ("eval", "train")


def _cosine_with_warmup(total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


class TrainCLMHeadsRecipe(BaseRecipe):
    """Embed-once, train-heads recipe for CLM decision models (see module docstring)."""

    def __init__(self, cfg: Any):
        """Store the recipe configuration (a ``ConfigNode`` parsed from YAML)."""
        self.cfg = cfg

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        """Initialise distributed state, load data, and make sure every text has a cached embedding."""
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
        self.output_dir = Path(self.cfg.get("output_dir", f"runs/clm_{self.tag}"))
        self.device = self.dist_env.device

        self.splits: dict[str, list[dict[str, Any]]] = {}
        for split in ("train", "validation", "test"):
            path = self.cfg.get(f"data.{split}", None)
            if path:
                records = load_records(path, limit=self.cfg.get(f"data.limit_{split}", None))
                for record in records:
                    validate_record(record)
                self.splits[split] = records
        if self.mode != "eval" and "train" not in self.splits:
            raise ValueError("data.train is required for sft/rl")
        self.slice_keys = list(self.cfg.get("data.slice_keys", []) or [])

        texts = sorted({t for records in self.splits.values() for t in collect_clm_texts(records)})
        self.table = self._embed_all(texts)

    def _cache_path(self, texts: list[str]) -> Path:
        from transformers import __version__ as transformers_version

        encoder_name = self.cfg.get("encoder.pretrained_model_name_or_path")
        tokenizer_name = self.cfg.get("encode.tokenizer", encoder_name)
        revision = self.cfg.get("encoder.revision", "")
        tokenizer_revision = self.cfg.get(
            "encode.tokenizer_revision", revision if tokenizer_name == encoder_name else ""
        )
        config = CLMEmbeddingCacheConfig(
            encoder_name=encoder_name,
            encoder_revision=revision,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            max_length=int(self.cfg.get("encode.max_length", CLM_MAX_TOKENS)),
            pooling=self.cfg.get("encoder.pooling", "last"),
            l2_normalize=self.cfg.get("encoder.l2_normalize", True),
            is_causal=self.cfg.get("encoder.is_causal", True),
            dtype=str(self.cfg.get("encoder.dtype", self.cfg.get("encoder.torch_dtype", "bfloat16"))),
            attention_implementation=self.cfg.get("encoder.attn_implementation", "sdpa"),
            transformers_version=transformers_version,
        )
        return config.build(
            cache_dir=Path(self.cfg.get("encode.cache_dir", self.output_dir / "embedding_cache")), texts=texts
        )

    @torch.no_grad()
    def _embed_all(self, texts: list[str]) -> EmbeddingTable:
        """Embed ``texts`` with the frozen encoder, sharded over all ranks, and cache the result."""
        cache = self._cache_path(texts)
        if cache.exists():
            if self.dist_env.is_main:
                logger.info("Loading %d cached embeddings from %s", len(texts), cache)
            return EmbeddingTable.load(cache)

        from transformers import AutoTokenizer

        encoder_name = self.cfg.get("encoder.pretrained_model_name_or_path")
        tokenizer_name = self.cfg.get("encode.tokenizer", encoder_name)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=self.cfg.get(
                "encode.tokenizer_revision",
                self.cfg.get("encoder.revision") if tokenizer_name == encoder_name else None,
            ),
        )
        tokenizer.padding_side = "right"
        # vLLM's ``truncate_prompt_tokens`` keeps the *last* tokens; match it matching the input contract used by clm-serve.
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_length = int(self.cfg.get("encode.max_length", CLM_MAX_TOKENS))
        batch_size = int(self.cfg.get("encode.batch_size", 16))

        encoder = self.cfg.encoder.instantiate().to(self.device).eval()
        rank, world = self.dist_env.rank, self.dist_env.world_size
        # Sort by length so batches are dense, then deal texts round-robin to ranks.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        mine = order[rank::world]
        shard = torch.empty((len(mine), 0))
        rows = []
        start = time.perf_counter()
        for b in range(0, len(mine), batch_size):
            chunk = [texts[i] for i in mine[b : b + batch_size]]
            enc = tokenizer(chunk, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            rows.append(encoder.encode(enc).float().cpu())
            if self.dist_env.is_main and (b // batch_size) % 50 == 0:
                logger.info(
                    "embedded %d/%d texts on rank 0 (%.0fs)", b + len(chunk), len(mine), time.perf_counter() - start
                )
        if rows:
            shard = torch.cat(rows)
        del encoder
        torch.cuda.empty_cache()

        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"indices": mine, "embeddings": shard}, cache.with_suffix(f".rank{rank}"))
        if dist.is_initialized():
            dist.barrier()
        if self.dist_env.is_main:
            hidden = None
            embeddings = None
            for r in range(world):
                part = torch.load(cache.with_suffix(f".rank{r}"), weights_only=True)
                if part["embeddings"].numel():
                    hidden = part["embeddings"].shape[1]
                    if embeddings is None:
                        embeddings = torch.empty((len(texts), hidden))
                    embeddings[torch.tensor(part["indices"])] = part["embeddings"]
                cache.with_suffix(f".rank{r}").unlink()
            EmbeddingTable(texts, embeddings).save(cache)
            logger.info("Cached %d embeddings (dim %d) at %s", len(texts), hidden, cache)
        if dist.is_initialized():
            dist.barrier()
        return EmbeddingTable.load(cache)

    # ------------------------------------------------------------------ forward / eval
    def _forward(self, heads: CLMHeadPair, records: list[dict[str, Any]]):
        """Per-question logits and aligned targets for a batch of records.

        Returns:
            ``(logits, targets)``: ``logits`` is a flat list with one tensor of shape [options_i] per question of every
            record (record-major); ``targets[r]`` lists the :class:`QuestionTarget` of record ``r`` in the same order.
        """
        states, options, counts, orders = clm_batch(records, self.table, self.device)
        logits = heads.question_logits(states, options, counts)
        targets = [question_targets(r, o) for r, o in zip(records, orders)]
        return logits, targets

    @torch.no_grad()
    def evaluate(
        self, heads: CLMHeadPair, records: list[dict[str, Any]], batch_size: int = 256
    ) -> tuple[dict, list[dict]]:
        """Return ``(summary, per-question predictions)`` for ``records``."""
        heads.eval()
        evaluator = DecisionEvaluator(slice_keys=self.slice_keys)
        for b in range(0, len(records), batch_size):
            batch = records[b : b + batch_size]
            logits, targets = self._forward(heads, batch)
            probs, k = [], 0
            for ts in targets:
                probs.append([logits[k + j].float().softmax(-1).tolist() for j in range(len(ts))])
                k += len(ts)
            evaluator.add_batch(batch, targets, probs)
        return evaluator.summary(), evaluator.predictions()

    # ------------------------------------------------------------------ main loop
    def run_train_validation_loop(self) -> None:
        """Evaluate (``eval``) or train (``sft``/``rl``) the heads, then report before/after metrics."""
        if not self.dist_env.is_main:
            if dist.is_initialized():
                dist.barrier()
            return
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        heads = load_clm_heads(self.cfg.get("heads.init_checkpoint", None)).to(self.device)
        eval_splits = [s for s in ("validation", "test") if s in self.splits]

        if self.mode == "eval":
            for split in eval_splits:
                self._report(heads, split)
        else:
            best = self._train(heads)
            save_clm_heads(best, self.output_dir / "heads.pt", extra={"posttrain_mode": self.tag})
            logger.info("Saved post-trained heads to %s", self.output_dir / "heads.pt")
            for split in eval_splits:
                self._report(best, split)
        if dist.is_initialized():
            dist.barrier()

    def _report(self, heads: CLMHeadPair, split: str) -> dict:
        summary, predictions = self.evaluate(heads, self.splits[split])
        summary["run"] = {
            "stage": self.tag,
            "split": split,
            "init_checkpoint": str(self.cfg.get("heads.init_checkpoint", None) or "release"),
        }
        save_summary(summary, self.output_dir / f"metrics_{split}.json")
        with open(self.output_dir / f"predictions_{split}.jsonl", "w") as f:
            for row in predictions:
                f.write(json.dumps(row) + "\n")
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
        return summary

    def _train(self, heads: CLMHeadPair) -> CLMHeadPair:
        """Train ``heads`` in place; return a copy of the best heads by validation ``train.eval_metric``."""

        def tget(key: str, default: Any) -> Any:
            return self.cfg.get(f"train.{key}", default)

        loss_cfg = self.loss_config
        loss_fn = loss_cfg.build()
        epochs = int(tget("epochs", 20))
        batch_size = int(tget("batch_size", 64))
        patience = int(tget("patience", 5))
        metric = tget("eval_metric", "accuracy")
        train_records = list(self.splits["train"])
        val_records = self.splits.get("validation")

        reference = None
        if loss_cfg.mode == "rl":
            reference = copy.deepcopy(heads).eval()
            for p in reference.parameters():
                p.requires_grad_(False)
        params = [p for n, p in heads.named_parameters() if tget("train_logit_scale", True) or n != "logit_scale"]
        optimizer = torch.optim.AdamW(params, lr=float(tget("lr", 5e-4)), weight_decay=float(tget("weight_decay", 0.0)))
        steps_per_epoch = math.ceil(len(train_records) / batch_size)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _cosine_with_warmup(epochs * steps_per_epoch, float(tget("warmup_ratio", 0.1)))
        )
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        def val_score(h):
            if not val_records:
                return None
            summary, _ = self.evaluate(h, val_records)
            overall = summary["overall"]
            if metric not in overall:
                raise ValueError(f"validation metric {metric!r} is unavailable; available metrics: {sorted(overall)}")
            return overall[metric]

        best_score = val_score(heads)
        best = copy.deepcopy(heads)
        logger.info("[%s] epoch 0 val %s=%s", self.tag, metric, best_score)
        stale, step = 0, 0
        log_path = self.output_dir / "training_log.jsonl"
        with open(log_path, "w") as log:
            for epoch in range(1, epochs + 1):
                heads.train()
                random.shuffle(train_records)
                for b in range(0, len(train_records), batch_size):
                    batch = train_records[b : b + batch_size]
                    logits_list, targets = self._forward(heads, batch)
                    flat = [q for ts in targets for q in ts]
                    ref = None
                    if reference is not None:
                        with torch.no_grad():
                            ref_logits, _ = self._forward(reference, batch)
                            ref = [x.float().log_softmax(-1).tolist() for x in ref_logits]
                    t = build_targets(flat, loss_cfg.question_weights, ref).to(self.device)
                    logits, _ = pad_logits(logits_list)
                    if float(loss_fn.normalization_weight(t)) <= 0:
                        raise ValueError("training batch has no supervised labels or explicit rewards")
                    loss, metrics = loss_fn(logits, t, generator=generator)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, float(tget("grad_clip", 1.0)))
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    row = {"epoch": epoch, "step": step, "loss": loss.item(), "grad_norm": grad_norm.item()}
                    row.update({k: v.item() for k, v in metrics.items()})
                    row["lr"] = scheduler.get_last_lr()[0]
                    log.write(json.dumps(row) + "\n")
                score = val_score(heads)
                logger.info("[%s] epoch %d loss=%.4f val %s=%s", self.tag, epoch, row["loss"], metric, score)
                log.write(json.dumps({"epoch": epoch, f"val_{metric}": score}) + "\n")
                log.flush()
                if score is None or best_score is None or score > best_score:
                    best_score, best, stale = score, copy.deepcopy(heads), 0
                else:
                    stale += 1
                    if stale >= patience:
                        logger.info("[%s] early stopping after %d stale epochs", self.tag, stale)
                        break
        return best.eval()


def main(config_path: str | None = None) -> None:
    """CLI entry point: ``python -m nemo_automodel.recipes.decision.train_clm_heads --config <yaml>``."""
    cfg = parse_args_and_load_config(config_path)
    recipe = TrainCLMHeadsRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
