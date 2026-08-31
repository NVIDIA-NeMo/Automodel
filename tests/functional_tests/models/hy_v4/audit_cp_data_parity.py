# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prove that HY4 CP1 and CP2 consume identical packed training samples.

This is a data-only preflight for the 256-GPU HY4 topology.  It builds the
production ChatDataset and THD packer once, then replays the exact
``StatefulDistributedSampler`` layouts used by:

* CP1: DP32 x local batch 8
* CP2: DP16 x local batch 16

Both layouts have global batch 256 and gradient accumulation one.  Per-rank
sampler streams are reconstructed into the sampler's canonical global order,
and every packed record is hashed over input IDs, labels, positions, and both
valid sequence-length vectors.  Extra ``-1000`` values inserted only to make a
local collated metadata rectangle are intentionally not data and are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.llm.train_ft import _build_tokenizer

_SEMANTIC_FIELDS = ("input_ids", "labels", "position_ids", "seq_lens", "seq_lens_padded")


@dataclass(frozen=True)
class Topology:
    """Data-parallel portion of one CP topology."""

    name: str
    cp_size: int
    dp_size: int
    local_batch_size: int


def _update_tensor_hash(digest: Any, name: str, value: Any) -> torch.Tensor:
    """Add a named tensor's exact dtype, shape, and bytes to ``digest``."""
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return tensor


def _packed_record_summary(record: dict[str, Any]) -> tuple[str, int, int]:
    """Return semantic SHA256, supervised-token count, and non-tail-token count."""
    digest = hashlib.sha256()
    tensors: dict[str, torch.Tensor] = {}
    for field in _SEMANTIC_FIELDS:
        if field not in record:
            raise KeyError(f"Packed record is missing semantic field {field!r}.")
        tensors[field] = _update_tensor_hash(digest, field, record[field])

    labels = tensors["labels"].reshape(-1)
    label_tokens = int((labels != -100).sum())
    reversed_padding = torch.cumprod((labels.flip(0) == -100).to(torch.int64), dim=0)
    tail_padding = int(reversed_padding.sum())
    non_tail_tokens = int(labels.numel() - tail_padding)
    return digest.hexdigest(), label_tokens, non_tail_tokens


def _canonical_indices(dataset: Any, topology: Topology, *, steps: int, global_batch_size: int) -> list[int]:
    """Reassemble rank-strided sampler streams into canonical global order."""
    if topology.dp_size * topology.local_batch_size != global_batch_size:
        raise ValueError(
            f"{topology.name} is not GA=1: dp={topology.dp_size}, "
            f"local_batch={topology.local_batch_size}, global_batch={global_batch_size}."
        )

    required = steps * global_batch_size
    canonical: list[int | None] = [None] * required
    sampler_total_size = None
    for rank in range(topology.dp_size):
        sampler = StatefulDistributedSampler(
            dataset,
            seed=1234,
            drop_last=True,
            num_replicas=topology.dp_size,
            rank=rank,
            shuffle=True,
        )
        sampler_total_size = int(sampler.total_size)
        local_indices = iter(sampler)
        for local_offset in range(steps * topology.local_batch_size):
            try:
                dataset_index = int(next(local_indices))
            except StopIteration as error:
                raise ValueError(
                    f"{topology.name} sampler exhausted before {steps} steps; "
                    f"dataset={len(dataset)}, total_size={sampler_total_size}."
                ) from error
            step, row = divmod(local_offset, topology.local_batch_size)
            # DistributedSampler assigns canonical position p to rank
            # p % dp_size.  This reverses its rank-strided partition exactly.
            slot = step * global_batch_size + row * topology.dp_size + rank
            if canonical[slot] is not None:
                raise AssertionError(f"Duplicate canonical slot {slot} for {topology.name}.")
            canonical[slot] = dataset_index

    if any(index is None for index in canonical):
        raise AssertionError(f"Unfilled canonical sampler slots for {topology.name}.")
    return [int(index) for index in canonical]


def _read_training_counts(path: Path, steps: int) -> list[tuple[int, int]]:
    """Read ``(num_label_tokens, num_tokens_per_step)`` from a training JSONL."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_step = {int(record["step"]): record for record in records}
    missing = [step for step in range(steps) if step not in by_step]
    if missing:
        raise ValueError(f"Training log {path} is missing steps {missing[:8]}.")
    return [
        (int(by_step[step]["num_label_tokens"]), int(by_step[step]["num_tokens_per_step"])) for step in range(steps)
    ]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    """Build production packs and compare the two topology sampler streams."""
    cfg = load_yaml_config(args.config)
    cfg.set_by_dotted("model.pretrained_model_name_or_path", str(args.tokenizer))
    cfg.set_by_dotted("dataset.tokenizer.pretrained_model_name_or_path", str(args.tokenizer))
    cfg.set_by_dotted("dataset.split", args.dataset_split)
    cfg.set_by_dotted("dataset.seq_length", args.sequence_length)
    cfg.set_by_dotted("packed_sequence.packed_sequence_size", args.sequence_length)

    _, tokenizer = _build_tokenizer(cfg.model, cfg.dataset)
    loader_config = RecipeConfig(cfg).dataloader
    if loader_config is None or loader_config.packing is None:
        raise ValueError("HY4 CP data audit requires the production packed dataloader.")
    if loader_config.seed != 1234:
        raise ValueError(f"Expected production sampler seed 1234, got {loader_config.seed}.")

    dataset = loader_config._build_dataset(tokenizer=tokenizer, dataset_build_context=None)
    packed_dataset, _ = loader_config.packing.build(
        dataset,
        split=getattr(loader_config.dataset_config, "split", None),
        seed=loader_config.seed,
        supports_seq_lens=True,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        # HY4 owns contiguous packed CP.  The production recipe deliberately
        # fixes loader-side packing at CP1 so pack composition is topology-free.
        cp_size=1,
        attn_implementation="cudnn",
    )

    reference = Topology("cp1", cp_size=1, dp_size=args.reference_dp_size, local_batch_size=args.reference_lbs)
    candidate = Topology("cp2", cp_size=2, dp_size=args.candidate_dp_size, local_batch_size=args.candidate_lbs)
    reference_indices = _canonical_indices(
        packed_dataset,
        reference,
        steps=args.steps,
        global_batch_size=args.global_batch_size,
    )
    candidate_indices = _canonical_indices(
        packed_dataset,
        candidate,
        steps=args.steps,
        global_batch_size=args.global_batch_size,
    )
    if reference_indices != candidate_indices:
        mismatch = next(
            index for index, (left, right) in enumerate(zip(reference_indices, candidate_indices)) if left != right
        )
        raise AssertionError(
            f"CP sampler streams differ at canonical slot {mismatch}: "
            f"CP1={reference_indices[mismatch]}, CP2={candidate_indices[mismatch]}."
        )

    summary_cache: dict[int, tuple[str, int, int]] = {}
    step_reports: list[dict[str, Any]] = []
    global_digest = hashlib.sha256()
    for step in range(args.steps):
        begin = step * args.global_batch_size
        indices = reference_indices[begin : begin + args.global_batch_size]
        summaries = []
        for dataset_index in indices:
            if dataset_index not in summary_cache:
                summary_cache[dataset_index] = _packed_record_summary(packed_dataset[dataset_index])
            summaries.append(summary_cache[dataset_index])

        step_digest = hashlib.sha256()
        for sample_digest, _, _ in summaries:
            step_digest.update(bytes.fromhex(sample_digest))
        step_sha256 = step_digest.hexdigest()
        global_digest.update(bytes.fromhex(step_sha256))
        step_reports.append(
            {
                "step": step,
                "packed_samples": len(indices),
                "semantic_sha256": step_sha256,
                "num_label_tokens": sum(summary[1] for summary in summaries),
                "num_tokens_per_step": sum(summary[2] for summary in summaries),
            }
        )

    if args.reference_training_log is not None:
        observed_counts = _read_training_counts(args.reference_training_log, args.steps)
        rebuilt_counts = [(report["num_label_tokens"], report["num_tokens_per_step"]) for report in step_reports]
        if rebuilt_counts != observed_counts:
            mismatch = next(
                index
                for index, (rebuilt, observed) in enumerate(zip(rebuilt_counts, observed_counts))
                if rebuilt != observed
            )
            raise AssertionError(
                f"Rebuilt data does not match completed CP1 training at step {mismatch}: "
                f"rebuilt={rebuilt_counts[mismatch]}, observed={observed_counts[mismatch]}."
            )

    total_size_cp1 = len(packed_dataset) // reference.dp_size * reference.dp_size
    total_size_cp2 = len(packed_dataset) // candidate.dp_size * candidate.dp_size
    if total_size_cp1 != total_size_cp2:
        raise AssertionError(f"CP1/CP2 drop-last epoch sizes differ: {total_size_cp1} vs {total_size_cp2}.")

    return {
        "status": "PASS",
        "config": str(args.config),
        "tokenizer": str(args.tokenizer),
        "dataset_split": args.dataset_split,
        "packed_sequence_size": args.sequence_length,
        "packed_dataset_size": len(packed_dataset),
        "drop_last_epoch_size": total_size_cp1,
        "sampler_seed": loader_config.seed,
        "global_batch_size": args.global_batch_size,
        "steps": args.steps,
        "packed_samples_compared": len(reference_indices),
        "semantic_fields": list(_SEMANTIC_FIELDS),
        "reference": reference.__dict__,
        "candidate": candidate.__dict__,
        "gradient_accumulation_steps": {"cp1": 1, "cp2": 1},
        "canonical_sample_indices_equal": True,
        "semantic_sample_hashes_equal": True,
        "completed_cp1_training_counts_equal": args.reference_training_log is not None,
        "global_semantic_sha256": global_digest.hexdigest(),
        "per_step": step_reports,
    }


def main() -> None:
    """Parse the production data contract and emit a machine-readable proof."""
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "examples/llm_finetune/hy_v4/hy4_preview_tulu3_4k_cudnn.yaml",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset-split", default="train[:100000]")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--reference-dp-size", type=int, default=32)
    parser.add_argument("--reference-lbs", type=int, default=8)
    parser.add_argument("--candidate-dp-size", type=int, default=16)
    parser.add_argument("--candidate-lbs", type=int, default=16)
    parser.add_argument("--reference-training-log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit(args)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
