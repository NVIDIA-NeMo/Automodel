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

Typical usage (single device; large MoE targets that need sharding must use the
online path instead):

    python -m nemo_automodel.components.speculative.precompute_eagle3 \\
        --target-model meta-llama/Llama-3.1-8B-Instruct \\
        --input-data Aeala/ShareGPT_Vicuna_unfiltered \\
        --output-dir /data/eagle3_cache/sharegpt_llama31 \\
        --seq-length 2048 --draft-vocab-size 8192 \\
        --batch-size 4 --shard-size 256 --dtype bf16

Then point the recipe at it: ``recipe_args.cached_target_path: /data/eagle3_cache/...``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import torch

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
    write_manifest,
    write_shard,
    write_target_embeddings,
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
    """Refuse to ``--resume`` into a cache produced with a different configuration.

    Every manifest field shapes the shard contents or their addressing: the
    ``target_probs`` columns follow ``selected_token_ids`` (which moves with the
    dataset, shuffle seed, and draft vocab size), sample order follows the
    dataset, and tensor shapes follow ``seq_length`` / ``dtype``. Shards from a
    mismatched run are indistinguishable by shape alone, so without this check a
    resume after changing e.g. ``--input-data`` or ``--shuffle-seed`` would keep
    the old shards and bless them with the new manifest, silently corrupting the
    supervision that training reads back.
    """
    if not os.path.exists(manifest_path(cache_dir)):
        if existing_shards:
            raise ValueError(
                f"--resume was requested for {cache_dir}, but its manifest is missing, so the existing "
                "shards cannot be verified against the current configuration. Delete the directory and "
                "start fresh."
            )
        return
    recorded = read_manifest(cache_dir)
    recorded.pop("format_version", None)
    if recorded != manifest:
        mismatched = sorted(k for k in recorded.keys() | manifest.keys() if recorded.get(k) != manifest.get(k))
        raise ValueError(
            f"--resume was requested for {cache_dir}, but the recorded manifest does not match the current "
            f"run configuration (mismatched fields: {mismatched}). Existing shards would be silently "
            "inconsistent with newly written ones. Re-run with the original settings, or delete the "
            "directory and start fresh."
        )


def _run(args: argparse.Namespace) -> int:
    """Load the target, scan the dataset once, and write the sharded cache. Returns an exit code."""
    _validate_args(args)
    cache_dtype = DTYPE_MAP[args.dtype]

    # Probe for pre-existing shards regardless of ``--resume`` so the
    # clobber guard can actually fire; only treat them as skippable when
    # resuming.
    present = existing_shard_indices(args.output_dir)
    if present and not args.resume:
        raise ValueError(f"{args.output_dir} already has shards; pass --resume to continue or use a fresh dir.")
    existing = present if args.resume else set()
    if existing:
        logger.info("Resume: %d shard(s) already present, will be skipped.", len(existing))

    device = torch.device(args.device)
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = NeMoAutoTokenizer.from_pretrained(args.target_model, trust_remote_code=args.trust_remote_code)
    target_model = NeMoAutoModelForCausalLM.from_pretrained(
        args.target_model, trust_remote_code=args.trust_remote_code, torch_dtype=compute_dtype
    ).to(device)
    target_wrapper = HFEagle3TargetModel(target_model, aux_layer_ids=args.aux_layer_ids)

    # The draft seeds its embed_tokens from the target embeddings; the offline
    # training path never loads the target, so store them alongside the cache.
    write_target_embeddings(args.output_dir, target_model.get_input_embeddings().weight)

    # No shuffle: the cache order is fixed so sample i lives in shard i // shard_size.
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
    )
    num_samples = len(dataloader.dataset)
    target_vocab_size = int(target_model.config.vocab_size)

    special_token_ids = [
        getattr(tokenizer, name, None) for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    ]
    selected_token_ids, selected_token_mask = build_eagle3_token_mapping(
        dataloader,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=args.draft_vocab_size,
        special_token_ids=special_token_ids,
    )
    selected_token_ids = selected_token_ids.to(device)
    selected_token_mask = selected_token_mask.to(device)

    manifest = {
        "target_model": args.target_model,
        "target_vocab_size": target_vocab_size,
        "draft_vocab_size": int(selected_token_ids.numel()),
        "seq_length": args.seq_length,
        "dtype": args.dtype,
        "num_samples": num_samples,
        "shard_size": args.shard_size,
        "target_probs_topk": args.target_probs_topk,
        "aux_hidden_dim": int(target_model.config.hidden_size) * len(target_wrapper.aux_layer_ids),
        "aux_layer_ids": list(target_wrapper.aux_layer_ids),
        "selected_token_ids": selected_token_ids.cpu().tolist(),
    }
    if args.resume:
        _ensure_resume_compatible(args.output_dir, manifest, existing)
    # Written before the shard loop (every field is already known here) so an
    # interrupted run leaves a manifest behind for the resume check above. An
    # incomplete cache still cannot be trained on: CachedEagle3Dataset verifies
    # the shard count against ``num_samples`` before serving any sample.
    write_manifest(args.output_dir, manifest)

    logger.info(
        "Precomputing EAGLE-3 cache: %d samples, shard_size=%d, draft_vocab=%d, dtype=%s -> %s",
        num_samples,
        args.shard_size,
        selected_token_ids.numel(),
        args.dtype,
        args.output_dir,
    )

    shard_index = 0
    chunks: list[dict[str, torch.Tensor]] = []
    buffered = 0

    def _flush() -> None:
        # Slicing to ``shard_size`` is exact for a full shard and a harmless
        # no-op for the trailing partial shard (fewer rows than the slice bound).
        nonlocal shard_index, chunks, buffered
        if buffered == 0:
            return
        if shard_index not in existing:
            # Every chunk came from _compute_batch_cache with the same topk, so they
            # share one key set (full vs top-k) -- merge whatever that batch produced.
            merged = {k: torch.cat([c[k] for c in chunks], dim=0)[: args.shard_size] for k in chunks[0]}
            path = write_shard(args.output_dir, shard_index, merged)
            logger.info("Wrote %s (%d samples)", path, merged["input_ids"].shape[0])
        chunks = []
        buffered = 0
        shard_index += 1

    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch["input_ids"].shape[0]
            if shard_index in existing:
                # The whole shard is already cached -- advance past its batches
                # without paying the target forward.
                buffered += batch_size
                if buffered >= args.shard_size:
                    chunks = []
                    buffered -= args.shard_size
                    shard_index += 1
                continue
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target_batch = target_wrapper.generate_batch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                loss_mask=batch["loss_mask"],
            )
            chunks.append(
                _compute_batch_cache(
                    target_batch, selected_token_ids, selected_token_mask, cache_dtype, args.target_probs_topk
                )
            )
            buffered += batch_size
            if buffered >= args.shard_size:
                _flush()

    _flush()  # trailing partial shard

    logger.info("Done. Cache written to %s", args.output_dir)
    return 0


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
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip shard indices already present in --output-dir.")
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
