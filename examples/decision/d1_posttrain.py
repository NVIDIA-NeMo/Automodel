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

"""Native D1 frozen-backbone post-training on typed SOC decisions.

D1-3B trains additive released answer-token rows without untying embeddings.
Omni trains its released marker head. This recipe measures text adaptation;
the vision/audio modules are preserved and frozen, without media gain claims.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import logging
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import __version__ as transformers_version
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel.components.datasets.decision.records import (
    QuestionTarget,
    load_records,
    question_targets,
    validate_record,
)
from nemo_automodel.components.eval.decision.metrics import DecisionEvaluator, save_summary
from nemo_automodel.components.loss.decision import DecisionLossConfig
from nemo_automodel.components.loss.decision_batch import DecisionTargets, QuestionLogits, assemble_decision_batch
from nemo_automodel.components.models.d1.model import D1OmniModel
from nemo_automodel.components.models.d1.records import D1TextExample, encode_d1_record
from nemo_automodel.components.models.d1.releases import RELEASE_DTYPES, RELEASES
from nemo_automodel.components.models.d1.text_readout import D1TextReadout
from nemo_automodel.components.models.d1.vl_model import Lfm2VlForConditionalGeneration

LOGGER = logging.getLogger(__name__)
FRAMEWORK = "automodel"


@dataclass(frozen=True)
class NativeD1Config:
    """Pinned source and native single-device model construction."""

    model: str

    def build(self) -> D1OmniModel | Lfm2VlForConditionalGeneration:
        """Load the registered native model with the released checkpoint weights."""
        repository, revision, _ = RELEASES[self.model]
        return cast(
            D1OmniModel | Lfm2VlForConditionalGeneration,
            NeMoAutoModelForImageTextToText.from_pretrained(
                repository,
                revision=revision,
                code_revision=revision,
                trust_remote_code=True,
                dtype=RELEASE_DTYPES[self.model],
                attn_implementation="sdpa",
            )
            .cuda()
            .eval(),
        )


class D1PostTraining:
    """Own frozen native model features, released readout, and typed supervision."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.omni = args.model == "d1-omni"
        self.repository, self.revision, reference_class = RELEASES[args.model]
        published = get_class_from_dynamic_module(
            reference_class,
            self.repository,
            revision=self.revision,
            code_revision=self.revision,
        )
        self.published_class = published
        self.prompt = importlib.import_module(published.__module__.rsplit(".", 1)[0] + ".prompt")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.repository, revision=self.revision, code_revision=self.revision, trust_remote_code=True
        )
        self.model = NativeD1Config(args.model).build()
        self.model.requires_grad_(False)
        self.records = {
            split: load_records(args.data_dir / f"{split}.jsonl") for split in ("train", "validation", "test")
        }
        config = AutoConfig.from_pretrained(
            self.repository, revision=self.revision, code_revision=self.revision, trust_remote_code=True
        )
        self.examples: dict[str, list[D1TextExample]] = {}
        self.targets: dict[str, list[QuestionTarget]] = {}
        for split, records in self.records.items():
            self.examples[split] = []
            self.targets[split] = []
            for record in records:
                validate_record(record)
                encoded = encode_d1_record(
                    record,
                    tokenizer=self.tokenizer,
                    prompt=self.prompt,
                    omni=self.omni,
                    max_length=config.max_length if self.omni else config.text_config.max_position_embeddings,
                    temperatures=getattr(config, "temperatures", {}),
                )
                self.examples[split].extend(encoded)
                self.targets[split].extend(
                    question_targets(record, {example.question_id: example.option_ids for example in encoded})
                )
        self.meta_by_id = {
            str(record["id"]): record.get("meta", {}) for records in self.records.values() for record in records
        }
        self.readout: D1TextReadout = self.model.prepare_text_readout(
            [example for examples in self.examples.values() for example in examples]
        )
        self.head = self.readout.head
        self.head.requires_grad_(True)
        self.provenance = {
            "framework": FRAMEWORK,
            "model": self.repository,
            "revision": self.revision,
            "backbone_dtype": str(self.model.dtype),
            "readout_dtype": str(next(self.head.parameters()).dtype),
            "training_scope": "released marker head"
            if self.omni
            else "additive answer-token rows; tied backbone unchanged",
            "data_sha256": {
                split: hashlib.sha256((args.data_dir / f"{split}.jsonl").read_bytes()).hexdigest()
                for split in self.records
            },
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda": torch.version.cuda,
        }
        source_root = Path(inspect.getfile(type(self.model))).parent
        source_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_root.rglob("*.py"))
        }
        self.provenance["native_source_sha256"] = source_hashes
        self.provenance["recipe_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.cache = args.cache_dir / args.model
        self.cache.mkdir(parents=True, exist_ok=True)
        manifest = self.cache / "manifest.json"
        cache_key = {
            key: self.provenance[key]
            for key in (
                "framework",
                "model",
                "revision",
                "backbone_dtype",
                "torch",
                "transformers",
                "data_sha256",
                "native_source_sha256",
                "recipe_source_sha256",
            )
        }
        if manifest.exists() and json.loads(manifest.read_text()) != cache_key:
            raise ValueError("Frozen feature cache does not match the pinned model/framework/dataset")
        manifest.write_text(json.dumps(cache_key, indent=2) + "\n")

    def feature(self, split: str, index: int) -> torch.Tensor:
        """Cache one native frozen encoding; never cache readout predictions.

        Returns:
            Tensor of shape [hidden] for D1-3B or [sequence, hidden] for Omni,
            on the model device, detached from backbone training.
        """
        path = self.cache / f"{split}-{index}.pt"
        if path.exists():
            return torch.load(path, map_location="cuda", weights_only=True)
        ex = self.examples[split][index]
        ids = torch.tensor([ex.input_ids], device="cuda", dtype=torch.long)
        valid = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            feature = self.readout.extract_features(ids, valid)
        if not torch.isfinite(feature).all():
            raise ValueError("Nonfinite frozen native features")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(feature.detach().cpu(), stream)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return feature

    def logits(self, split: str, indices: list[int], head: torch.nn.Module | None = None) -> list[torch.Tensor]:
        """Return calibrated differentiable logits in each example's internal order.

        Returns:
            List of float32 Tensors of shape [options], one per requested
            question, retaining gradients to the selected readout head.
        """
        head = self.head if head is None else head
        examples = [self.examples[split][index] for index in indices]
        features = [self.feature(split, index) for index in indices]
        return self.readout.score(features, examples, head=head)

    def evaluate(self, split: str, destination: Path | None = None) -> dict:
        """Evaluate every decision; selection callers use validation only."""
        self.head.eval()
        evaluator = DecisionEvaluator()
        with torch.no_grad():
            for start in range(0, len(self.examples[split]), self.args.batch_size):
                indices = list(range(start, min(start + self.args.batch_size, len(self.examples[split]))))
                for index, logits in zip(indices, self.logits(split, indices)):
                    ex = self.examples[split][index]
                    target = self.targets[split][index]
                    evaluator.add(
                        ex.record_id,
                        target.question_id,
                        target.option_ids,
                        logits.softmax(-1).tolist(),
                        target.label,
                        target.rewards,
                        self.meta_by_id[ex.record_id],
                    )
        result = evaluator.summary()
        if destination is not None:
            save_summary(result, destination / f"metrics_{split}.json")
            (destination / f"predictions_{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in evaluator.predictions())
            )
        return result

    def train(self, destination: Path) -> None:
        """Train the head and select its checkpoint by held-out validation only."""
        reference = copy.deepcopy(self.head).eval().requires_grad_(False) if self.args.mode == "rl" else None
        criterion = DecisionLossConfig(mode=self.args.mode).build()
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=self.args.lr, weight_decay=0.0)
        selection = "reward" if self.args.mode == "rl" else "accuracy"
        best = self.evaluate("validation")["overall"][selection]
        torch.save(self.head.state_dict(), destination / "head.pt")
        step = 0
        with (destination / "training_log.jsonl").open("w") as log:
            for epoch in range(self.args.epochs):
                indices = list(range(len(self.examples["train"])))
                random.Random(self.args.seed + epoch).shuffle(indices)
                for start in range(0, len(indices), self.args.batch_size):
                    batch = indices[start : start + self.args.batch_size]
                    self.head.train()
                    optimizer.zero_grad(set_to_none=True)
                    examples = [self.examples["train"][index] for index in batch]
                    values = self.logits("train", batch)
                    ref = None
                    if reference is not None:
                        with torch.no_grad():
                            ref = [
                                [QuestionLogits(ex.record_id, ex.question_id, ex.option_ids, row.log_softmax(-1))]
                                for ex, row in zip(examples, self.logits("train", batch, reference), strict=True)
                            ]
                    targets = DecisionTargets.from_questions(
                        record_ids=[ex.record_id for ex in examples],
                        questions=[[self.targets["train"][index]] for index in batch],
                        reference_logprobs=ref,
                        device=values[0].device,
                    )
                    decision_batch = assemble_decision_batch(
                        logits=[
                            [QuestionLogits(ex.record_id, ex.question_id, ex.option_ids, row)]
                            for ex, row in zip(examples, values, strict=True)
                        ],
                        targets=targets,
                    )
                    loss, metrics = criterion(decision_batch.logits, decision_batch.targets)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite D1 training objective")
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
                    optimizer.step()
                    step += 1
                    row = {
                        "epoch": epoch,
                        "step": step,
                        "loss": float(loss.detach()),
                        "grad_norm": float(grad_norm),
                        **{key: float(value) for key, value in metrics.items()},
                    }
                    if not all(math.isfinite(value) for value in row.values()):
                        raise ValueError("Nonfinite D1 training metric")
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                validation = self.evaluate("validation")
                score = validation["overall"][selection]
                LOGGER.info("epoch=%s validation_%s=%s best=%s", epoch + 1, selection, score, best)
                if score > best:
                    best = score
                    torch.save(self.head.state_dict(), destination / "head.pt")
        self.head.load_state_dict(
            torch.load(destination / "head.pt", map_location="cuda", weights_only=True), strict=True
        )

    def check_published_baseline(self, destination: Path) -> None:
        """Compare released-size native decision probabilities with the pinned publisher.

        This probe uses validation examples, without training or selecting on test labels.
        The 0.02 absolute probability bound allows floating-point kernel differences;
        exact readout option identities and all five task kinds are still required.
        """
        reference = (
            self.published_class.from_pretrained(
                self.repository,
                revision=self.revision,
                dtype=self.model.dtype,
            )
            .cuda()
            .eval()
        )
        reference_dtype = reference.dtype
        checks = []
        try:
            if reference_dtype != self.model.dtype:
                raise ValueError("Native and publisher baseline models must use the same dtype")
            # Publisher lazy properties otherwise load a tokenizer from the latest revision.
            if self.omni:
                reference.tokenizer = self.tokenizer
            else:
                runner = importlib.import_module(self.published_class.__module__.rsplit(".", 1)[0] + ".runner")
                reference.engine = runner.SystemOne(model=reference, tokenizer=self.tokenizer)
            offset = 0
            for record in self.records["validation"][:3]:
                count = len(record["questions"])
                indices = list(range(offset, offset + count))
                with torch.no_grad():
                    values = self.logits("validation", indices)
                for index, logits in zip(indices, values):
                    ex = self.examples["validation"][index]
                    qid = ex.question_id
                    question = record["questions"][qid]
                    if not self.omni:
                        rendered = reference.engine.render(record["state"], self.prompt.as_question(question))
                        published_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
                        if tuple(published_ids) != ex.input_ids:
                            raise ValueError("Native and publisher single-question prompts differ")
                    answer = reference.system_one(record["state"], {qid: question})["answers"][qid]
                    if record["questions"][ex.question_id]["type"] == "noul":
                        probabilities = {"true": answer["noul"], "false": 1 - answer["noul"]}
                    else:
                        probabilities = answer["probabilities"]
                    expected = torch.tensor([probabilities[key] for key in ex.option_ids], device="cuda")
                    actual = logits.softmax(-1)
                    try:
                        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0)
                    except AssertionError:
                        if not self.omni:
                            ids = torch.tensor([ex.input_ids], device="cuda", dtype=torch.long)
                            with torch.no_grad():
                                native_full = self.model(ids, use_cache=False).logits[0, -1].float()
                                published_chain = reference(ids, use_cache=False).logits[0, -1].float()

                            def option_probabilities(vocabulary_logits: torch.Tensor) -> list[float]:
                                """Reduce vocabulary scores using the encoded publisher aliases.

                                Args:
                                    vocabulary_logits: Tensor of shape [vocab].

                                Returns:
                                    Probabilities in the example's target option order.
                                """
                                return (
                                    torch.stack([vocabulary_logits[list(group)].amax() for group in ex.alias_token_ids])
                                    .softmax(-1)
                                    .tolist()
                                )

                            logging.getLogger(__name__).error(
                                "Baseline diagnostic: %s",
                                json.dumps(
                                    {
                                        "record_id": ex.record_id,
                                        "question": qid,
                                        "native_adapter": actual.tolist(),
                                        "native_full_vocabulary": option_probabilities(native_full),
                                        "publisher_chain": option_probabilities(published_chain),
                                        "publisher_api": expected.tolist(),
                                    }
                                ),
                            )
                        raise
                    checks.append(
                        {
                            "record_id": ex.record_id,
                            "question": ex.question_id,
                            "option_ids": list(ex.option_ids),
                            "max_probability_error": float((actual - expected).abs().max()),
                        }
                    )
                offset += count
        finally:
            del reference
            torch.cuda.empty_cache()
        (destination / "published-baseline-parity.json").write_text(
            json.dumps(
                {
                    "revision": self.revision,
                    "tokenizer_revision": self.revision,
                    "request_shape": "one independent question per forward pass",
                    "dtype": str(self.model.dtype).removeprefix("torch."),
                    "publisher_dtype": str(reference_dtype).removeprefix("torch."),
                    "readout_dtype": str(next(self.head.parameters()).dtype).removeprefix("torch."),
                    "absolute_probability_tolerance": 0.02,
                    "validation_records": min(3, len(self.records["validation"])),
                    "checks": checks,
                },
                indent=2,
            )
            + "\n"
        )

    def run(self) -> None:
        """Save stage metrics and verify readout checkpoint reloading."""
        destination = self.args.output_dir
        destination.mkdir(parents=True, exist_ok=True)
        if self.args.input_dir is not None:
            prior = json.loads((self.args.input_dir / "provenance.json").read_text())
            for key in (
                "framework",
                "model",
                "revision",
                "backbone_dtype",
                "readout_dtype",
                "torch",
                "transformers",
                "data_sha256",
                "training_scope",
            ):
                if prior.get(key) != self.provenance[key]:
                    raise ValueError(f"Checkpoint provenance mismatch: {key}")
            self.head.load_state_dict(
                torch.load(self.args.input_dir / "head.pt", map_location="cuda", weights_only=True), strict=True
            )
        if self.args.mode != "base":
            self.train(destination)
        else:
            self.check_published_baseline(destination)
            torch.save(self.head.state_dict(), destination / "head.pt")
        before = self.logits("validation", [0])[0].detach()
        restored = copy.deepcopy(self.head)
        restored.load_state_dict(
            torch.load(destination / "head.pt", map_location="cuda", weights_only=True), strict=True
        )
        self.head.eval()
        restored.eval()
        after = self.logits("validation", [0], restored)[0].detach()
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        self.provenance["readout_reload_max_abs_error"] = float((after - before).abs().max())
        (destination / "provenance.json").write_text(json.dumps(self.provenance, indent=2) + "\n")
        for split in ("validation", "test"):
            summary = self.evaluate(split, destination)
            LOGGER.info("%s/%s %s", self.args.mode, split, summary["overall"])


def main() -> None:
    """Run a released baseline or a complete post-training stage."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=RELEASES, required=True)
    parser.add_argument("--mode", choices=("base", "sft", "rl"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")
    if args.mode == "rl" and args.input_dir is None:
        parser.error("RL requires the selected SFT checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("This recipe requires a real CUDA GPU")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    D1PostTraining(args).run()


if __name__ == "__main__":
    main()
