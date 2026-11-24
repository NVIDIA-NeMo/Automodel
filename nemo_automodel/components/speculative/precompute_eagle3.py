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

"""Precompute the EAGLE-3 offline target-output cache (SpecForge "offline" path).

The frozen target model's per-token supervision is the same on every epoch and
every run, yet the online recipe recomputes it each step. This script runs the
target once over a dataset and writes the supervision to disk; training then
reads it back via ``cached_target_path`` instead of loading or running the
target at all.

==============================  READ THIS  =================================
This is the SpecForge **offline** training path. It is **disk intensive** -- the
draft-vocab ``target_probs`` (``draft_vocab`` wide) and ``aux_hidden_states``
(``3 * target_hidden_size`` wide) together cost on the order of 100+ MB per
sample for an 8B target at seq-len 2048, i.e. multiple TB for a large corpus.
Modern practice trains **online**, where the target forward is cheap next to that
I/O; prefer the online recipe unless you re-train a draft repeatedly on a fixed,
bounded dataset. ``--target-probs-topk`` can shrink the ``target_probs`` field by
storing only its top-k mass, but that is a real disk/fidelity trade-off (a real
LLM distribution has a fat tail), so it is **off by default** -- see the flag's
help.
===========================================================================

Only EAGLE-3 is supported. EAGLE-1/2 supervise on the *full*-vocab target
distribution, which the recipe keeps on the online path only.

Typical usage:

    python -m nemo_automodel.components.speculative.precompute_eagle3 \\
        --target-model meta-llama/Llama-3.1-8B-Instruct \\
        --input-data Aeala/ShareGPT_Vicuna_unfiltered \\
        --output-dir /data/eagle3_cache/sharegpt_llama31 \\
        --seq-length 2048 --draft-vocab-size 8192 \\
        --batch-size 4 --shard-size 256 --dtype bf16

Then point the recipe at it: ``recipe_args.cached_target_path: /data/eagle3_cache/...``.
On one node, launch with ``torchrun --standalone --nproc-per-node=2 -m`` in
place of ``python -m`` and use a dense target that fits on each device.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from huggingface_hub.utils import WeakFileLock
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import HF_MODULES_CACHE

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.eagle3 import build_eagle3_dataloader, build_eagle3_token_mapping
from nemo_automodel.components.datasets.llm.eagle3_cache import (
    DTYPE_MAP,
    compress_target_probs,
    existing_shard_indices,
    is_compressed,
    manifest_path,
    read_manifest,
    validate_eagle3_shard,
    validate_target_embeddings,
    write_manifest,
    write_shard,
    write_target_embeddings,
)
from nemo_automodel.components.datasets.llm.offline_cache import (
    dataloader_for_sample_range,
    partition_cache_shards,
    sync_cache_phase,
    write_cache_shards,
)
from nemo_automodel.components.speculative.eagle import HFEagle3TargetModel
from nemo_automodel.components.speculative.eagle.core import _compute_target_distribution

logger = logging.getLogger(__name__)


def _compute_batch_cache(
    target_batch,
    selected_token_ids: torch.Tensor,
    selected_token_mask: torch.Tensor,
    cache_dtype: torch.dtype,
    target_probs_topk: int,
) -> dict[str, torch.Tensor]:
    """Turn one target-model batch into the per-sample tensors the trainer caches.

    Reuses ``_compute_target_distribution`` -- the exact function the online
    trainer calls -- so the cached supervision is numerically identical to the
    live path. The draft-vocab ``target_probs`` is stored in full, or -- when
    ``target_probs_topk`` compresses (a positive k below the draft vocab) -- as
    its top-k ``(values, indices)`` factorization. Float fields are downcast to
    ``cache_dtype``; everything is moved to CPU for writing.
    """
    target_probs, position_mask = _compute_target_distribution(
        target_logits=target_batch.logits,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        loss_mask=target_batch.loss_mask,
    )
    sample = {
        "input_ids": target_batch.input_ids.to(torch.long).cpu(),
        "attention_mask": target_batch.attention_mask.to(torch.long).cpu(),
        "loss_mask": target_batch.loss_mask.to(torch.long).cpu(),
        "aux_hidden_states": target_batch.aux_hidden_states.to(cache_dtype).cpu(),
        "position_mask": position_mask.to(torch.bool).cpu(),
    }
    if is_compressed(target_probs_topk, target_probs.shape[-1]):
        topk_values, topk_indices = compress_target_probs(target_probs, target_probs_topk)
        sample["target_probs_values"] = topk_values.to(cache_dtype).cpu()
        sample["target_probs_indices"] = topk_indices.cpu()
    else:
        sample["target_probs"] = target_probs.to(cache_dtype).cpu()
    return sample


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid CLI values before loading any model."""
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.shard_size < 1:
        raise ValueError(f"--shard-size must be >= 1, got {args.shard_size}")
    if args.shard_size % args.batch_size != 0:
        raise ValueError(
            f"--shard-size ({args.shard_size}) must be a multiple of --batch-size ({args.batch_size}) "
            "so shards align to batch boundaries."
        )
    if args.draft_vocab_size is not None and args.draft_vocab_size < 1:
        raise ValueError(f"--draft-vocab-size must be >= 1 or omitted, got {args.draft_vocab_size}")
    if args.target_probs_topk < 0:
        raise ValueError(f"--target-probs-topk must be >= 0 (0 = lossless), got {args.target_probs_topk}")
    if args.dtype not in DTYPE_MAP:
        raise ValueError(f"--dtype must be one of {sorted(DTYPE_MAP)}, got {args.dtype!r}")


def _ensure_resume_compatible(cache_dir: str, manifest: dict[str, Any], existing_shards: set[int]) -> None:
    """Check the manifest and shard numbering before any cache file is written."""
    if not os.path.exists(manifest_path(cache_dir)):
        if existing_shards:
            raise ValueError(
                f"--resume was requested for {cache_dir}, but its manifest is missing, so the existing "
                "shards cannot be verified. Use a fresh output directory."
            )
    else:
        recorded = read_manifest(cache_dir)
        recorded.pop("format_version", None)
        if recorded != manifest:
            mismatched = sorted(
                k
                for k in recorded.keys() | manifest.keys()
                if k not in recorded or k not in manifest or recorded[k] != manifest[k]
            )
            raise ValueError(
                f"The manifest at {cache_dir} does not match the current configuration "
                f"(mismatched fields: {mismatched}). Re-run with the original settings or use a fresh output directory."
            )

    shard_size = int(manifest["shard_size"])
    num_shards = (int(manifest["num_samples"]) + shard_size - 1) // shard_size
    unexpected = sorted(index for index in existing_shards if index >= num_shards)
    if unexpected:
        raise ValueError(f"{cache_dir} has shard indices outside the manifest's {num_shards} shards: {unexpected}.")


def _missing_shard_runs(
    num_shards: int, existing: set[int], *, world_size: int, rank: int
) -> Iterator[tuple[int, int]]:
    """Partition the ordered missing shards and coalesce each rank's consecutive indices."""
    missing = [index for index in range(num_shards) if index not in existing]
    offset, _, count = partition_cache_shards(len(missing), shard_size=1, world_size=world_size, rank=rank)
    if count == 0:
        return

    start = previous = missing[offset]
    for index in missing[offset + 1 : offset + count]:
        if index != previous + 1:
            yield start, previous + 1
            start = index
        previous = index
    yield start, previous + 1


def _precompute(
    args: argparse.Namespace,
    *,
    device: torch.device,
    target_wrapper: HFEagle3TargetModel,
    tokenizer: PreTrainedTokenizerBase,
    existing: set[int],
) -> int:
    cache_dtype = DTYPE_MAP[args.dtype]
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    if distributed:
        shard_indices = [sorted(existing)]
        dist.broadcast_object_list(shard_indices, src=0)
        existing = set(shard_indices[0])
    if rank == 0 and existing:
        logger.info("Resume: %d shard(s) already present, will be skipped.", len(existing))

    # Partition the ordered dataset without DistributedSampler's padded rows.
    with sync_cache_phase("EAGLE-3 cache input preparation"):
        dataloader = build_eagle3_dataloader(
            data_path=args.input_data,
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            split=args.split,
            distributed=False,
            shuffle_seed=args.shuffle_seed,
            mask_reasoning_content=args.mask_reasoning_content,
            mask_generation_prompt=args.mask_generation_prompt,
        )
        num_samples = len(dataloader.dataset)
        first_shard, first_sample, local_samples = partition_cache_shards(
            num_samples, args.shard_size, world_size, rank
        )
        local_shards = (local_samples + args.shard_size - 1) // args.shard_size
        local_loader = (
            dataloader_for_sample_range(dataloader, start=first_sample, end=first_sample + local_samples)
            if distributed
            else dataloader
        )
        target_vocab_size = int(target_wrapper.model.config.vocab_size)
        embeddings = target_wrapper.get_input_embeddings().weight
        aux_layer_ids = list(target_wrapper.aux_layer_ids)
        aux_hidden_dim = int(target_wrapper.model.config.hidden_size) * len(aux_layer_ids)
        special_token_ids = [
            getattr(tokenizer, name, None) for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
        ]
    if distributed:
        layout = (
            num_samples,
            target_vocab_size,
            tuple(embeddings.shape),
            aux_layer_ids,
            aux_hidden_dim,
            special_token_ids,
        )
        layouts = [None] * world_size
        dist.all_gather_object(layouts, layout)
        if any(other != layout for other in layouts):
            raise ValueError(f"EAGLE-3 target or dataset layout differs across ranks: {layouts}")

    # Count every owned row, including the shards that a resume will reuse.
    selected_token_ids, selected_token_mask = build_eagle3_token_mapping(
        local_loader,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=args.draft_vocab_size,
        special_token_ids=special_token_ids,
    )
    manifest = {
        "target_model": args.target_model,
        "target_vocab_size": target_vocab_size,
        "draft_vocab_size": int(selected_token_ids.numel()),
        "seq_length": args.seq_length,
        "dtype": args.dtype,
        "num_samples": num_samples,
        "shard_size": args.shard_size,
        "target_probs_topk": args.target_probs_topk,
        "aux_hidden_dim": aux_hidden_dim,
        "aux_layer_ids": aux_layer_ids,
        "selected_token_ids": selected_token_ids.tolist(),
        "input_data": str(args.input_data),
        "split": args.split,
        "shuffle_seed": int(args.shuffle_seed),
        "mask_reasoning_content": bool(args.mask_reasoning_content),
        "mask_generation_prompt": bool(args.mask_generation_prompt),
    }

    has_manifest = has_embeddings = False
    with sync_cache_phase("EAGLE-3 cache validation"):
        if rank == 0 and args.resume:
            has_manifest = os.path.exists(manifest_path(args.output_dir))
            _ensure_resume_compatible(args.output_dir, manifest, existing)
            if existing:
                has_embeddings = validate_target_embeddings(args.output_dir, shape=embeddings.shape)
        for index in range(first_shard, first_shard + local_shards):
            if index in existing:
                validate_eagle3_shard(args.output_dir, shard_index=index, manifest=manifest)
        selected_token_ids = selected_token_ids.to(device)
        selected_token_mask = selected_token_mask.to(device)

    with sync_cache_phase("EAGLE-3 cache metadata publication"):
        if rank == 0:
            if not has_manifest:
                write_manifest(args.output_dir, manifest)
            # With no committed shards, regenerate embeddings even if a prior
            # run was interrupted between publishing the manifest and embeddings.
            if not has_embeddings:
                write_target_embeddings(args.output_dir, embeddings)

    # Writers may finish at very different times after a sparse resume. Close
    # the group together so torchrun can supervise independent shard writes.
    if distributed:
        dist.destroy_process_group()

    def _compute_from_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Capture the on-disk tensors for this batch.

        Args:
            batch: Input IDs, attention and loss masks of shape [batch, sequence].

        Returns:
            IDs and masks of shape [batch, sequence], auxiliary states of shape
            [batch, sequence, aux_hidden], position mask of shape [batch, sequence, 1],
            and full or top-k probabilities of shape [batch, sequence, vocab or topk].
        """
        with torch.no_grad():
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            target_batch = target_wrapper.generate_batch(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], loss_mask=batch["loss_mask"]
            )
            return _compute_batch_cache(
                target_batch, selected_token_ids, selected_token_mask, cache_dtype, args.target_probs_topk
            )

    num_shards = (num_samples + args.shard_size - 1) // args.shard_size
    for start, end in _missing_shard_runs(num_shards, existing, world_size=world_size, rank=rank):
        start_sample = start * args.shard_size
        end_sample = min(end * args.shard_size, num_samples)
        logger.info(
            "Rank %d/%d: precomputing samples [%d, %d) from shard %d in %s",
            rank,
            world_size,
            start_sample,
            end_sample,
            start,
            args.output_dir,
        )
        loader = local_loader
        if start != first_shard or end != first_shard + local_shards:
            loader = dataloader_for_sample_range(dataloader, start=start_sample, end=end_sample)
        write_cache_shards(
            dataloader=loader,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            start_shard_index=start,
            compute_batch=_compute_from_batch,
            write_shard_fn=write_shard,
            logger=logger,
        )

    logger.info("Rank %d finished its cache shard writes in %s", rank, args.output_dir)
    return 0


def _run(args: argparse.Namespace) -> int:
    """Load a target replica and run the cache producer under Python or torchrun."""
    _validate_args(args)
    # Some schedulers set rank variables for ordinary Python commands.
    is_torchrun_worker = "LOCAL_RANK" in os.environ and "TORCHELASTIC_RUN_ID" in os.environ
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if is_torchrun_worker else 1
    rank = int(os.environ["RANK"]) if world_size > 1 else 0
    device = torch.device(args.device)
    if world_size > 1:
        if int(os.environ["LOCAL_WORLD_SIZE"]) != world_size:
            raise ValueError("Distributed EAGLE-3 precompute requires a single-node torchrun launch.")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
        elif device.type != "cpu":
            raise ValueError("Distributed EAGLE-3 precompute requires --device cuda or --device cpu.")

    existing = existing_shard_indices(args.output_dir) if rank == 0 else set()
    if existing and not args.resume:
        raise ValueError(f"{args.output_dir} already has shards; pass --resume to continue or use a fresh dir.")

    # Load an independent replica before initializing the process group, so
    # the target loader can read its checkpoint locally.
    # Transformers copies trusted code into a shared cache during the load.
    loader_lock = (
        WeakFileLock(Path(HF_MODULES_CACHE) / ".automodel-eagle3-precompute.lock")
        if world_size > 1 and args.trust_remote_code
        else nullcontext()
    )
    with loader_lock:
        tokenizer = NeMoAutoTokenizer.from_pretrained(args.target_model, trust_remote_code=args.trust_remote_code)
        factory = AutoModelForCausalLM if device.type == "cpu" else NeMoAutoModelForCausalLM
        compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        launcher_env = {}
        if world_size > 1 and device.type == "cuda":
            # NeMo consults the launcher topology even without a process
            # group. Use the original single-process weight-loading path for
            # every replica so resumes can freely change the world size.
            launcher_env = {
                name: os.environ.pop(name) for name in ("WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK")
            }
        try:
            model = factory.from_pretrained(
                args.target_model, trust_remote_code=args.trust_remote_code, dtype=compute_dtype
            ).to(device)
        finally:
            os.environ.update(launcher_env)
    model.requires_grad_(False)
    target_wrapper = HFEagle3TargetModel(model, aux_layer_ids=args.aux_layer_ids)

    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30))
    try:
        return _precompute(args, device=device, target_wrapper=target_wrapper, tokenizer=tokenizer, existing=existing)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute the EAGLE-3 offline target-output cache (SpecForge offline path; disk-heavy).",
    )
    parser.add_argument("--target-model", required=True, help="Target model path or HF repo id.")
    parser.add_argument("--input-data", required=True, help="HF dataset id or local chat dataset path.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the cache shards + manifest into.")
    parser.add_argument("--seq-length", type=int, default=2048, help="Sequence length.")
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=None,
        help="Draft vocabulary size (None = full target vocab; defeats most of the storage saving).",
    )
    parser.add_argument(
        "--target-probs-topk",
        type=int,
        default=0,
        help="Compress the dominant cache field by storing only the top-k of each token's "
        "draft-vocab target distribution (0 = off = store it in full, the default and exact). "
        "This is a real disk/fidelity trade-off, NOT free: a real LLM next-token distribution has "
        "a fat tail (top-64 of a Qwen3-8B distribution keeps only ~0.92 of the mass, KL ~1.3), so "
        "small k distorts the soft-CE target. Use a large k (or leave it off) for fidelity.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Target forward batch size.")
    parser.add_argument(
        "--shard-size", type=int, default=256, help="Samples per shard (must be a multiple of --batch-size)."
    )
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPE_MAP), help="Cache float dtype.")
    parser.add_argument("--device", default="cuda", help="Device to run the target on (cuda / cpu).")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--split", default=None, help="HF dataset split (supports slice syntax).")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Dataset ingestion shuffle seed.")
    parser.add_argument(
        "--aux-layer-ids",
        type=int,
        nargs="+",
        default=None,
        help="Target layers to capture (default: the EAGLE-3 low/mid/high recipe).",
    )
    parser.add_argument(
        "--mask-reasoning-content",
        action="store_true",
        help="Exclude rendered reasoning_content tokens from the cached loss mask. Recorded in the manifest; "
        "the trainer's mask_reasoning_content setting must match it.",
    )
    parser.add_argument(
        "--mask-generation-prompt",
        action="store_true",
        help="Exclude the template-supplied prefix of each assistant turn (role header and any empty reasoning "
        "block) from the cached loss mask. Recorded in the manifest; the trainer's mask_generation_prompt "
        "setting must match it.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Validate the cache and fill missing global shards.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Parses ``argv`` and returns the process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
