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

"""Precompute the DSpark offline target-supervision cache.

The online DSpark recipe runs a frozen target model every step to capture the
intermediate target hidden states consumed by the draft and the final hidden
state used by the TV / confidence losses. This script runs that target once and
writes those tensors to disk. Training can then set
``recipe_args.cached_target_path`` to stream the cache without loading or
running the target model.

Typical usage (single device):

    python -m nemo_automodel.components.speculative.precompute_dspark \\
        --target-model Qwen/Qwen3-0.6B \\
        --input-data /data/messages.jsonl \\
        --output-dir /data/dspark_cache/qwen3_06b \\
        --seq-length 2048 --batch-size 4 --shard-size 256 --dtype bf16
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import torch
from transformers import AutoConfig

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.dspark_cache import (
    DTYPE_MAP,
    build_cache_manifest,
    compute_batch_cache,
    existing_shard_indices,
    manifest_mismatch_fields,
    manifest_path,
    read_manifest,
    write_manifest,
    write_shard,
    write_target_weights,
)
from nemo_automodel.components.datasets.llm.eagle3 import build_eagle3_dataloader
from nemo_automodel.components.datasets.llm.offline_cache import (
    dataloader_from_sample,
    resume_start_sample,
    write_cache_shards,
)
from nemo_automodel.components.speculative.dspark.registry import build_target_layer_ids
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel
from nemo_automodel.components.speculative.dspark.target_utils import (
    DEEPSEEK_V4_MODEL_TYPE as _DEEPSEEK_V4_MODEL_TYPE,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    GEMMA4_MODEL_TYPES as _GEMMA4_MODEL_TYPES,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    GLM_5_2_MODEL_TYPE as _GLM_5_2_MODEL_TYPE,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    MINIMAX_M3_MODEL_TYPES as _MINIMAX_M3_MODEL_TYPES,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    apply_target_chat_template as _apply_target_chat_template,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    read_target_model_type as _read_target_model_type,
)

logger = logging.getLogger(__name__)

_UNSUPPORTED_OFFLINE_TARGET_MODEL_TYPES = (
    _DEEPSEEK_V4_MODEL_TYPE,
    _GLM_5_2_MODEL_TYPE,
    *_MINIMAX_M3_MODEL_TYPES,
)


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid CLI values before loading the target model."""
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.shard_size < 1:
        raise ValueError(f"--shard-size must be >= 1, got {args.shard_size}")
    if args.shard_size % args.batch_size != 0:
        raise ValueError(
            f"--shard-size ({args.shard_size}) must be a multiple of --batch-size ({args.batch_size}) "
            "so shards align to batch boundaries."
        )
    if args.draft_num_hidden_layers < 1:
        raise ValueError(f"--draft-num-hidden-layers must be >= 1, got {args.draft_num_hidden_layers}")
    if args.dtype not in DTYPE_MAP:
        raise ValueError(f"--dtype must be one of {sorted(DTYPE_MAP)}, got {args.dtype!r}")


def _ensure_resume_compatible(cache_dir: str, manifest: dict[str, Any], existing_shards: set[int]) -> None:
    """Refuse to resume into shards produced with a different configuration."""
    if existing_shards:
        resume_start_sample(existing_shards, int(manifest["shard_size"]))
    if not os.path.exists(manifest_path(cache_dir)):
        if existing_shards:
            raise ValueError(
                f"--resume was requested for {cache_dir}, but its manifest is missing, so existing shards "
                "cannot be verified. Delete the directory and start fresh."
            )
        return
    mismatched = manifest_mismatch_fields(read_manifest(cache_dir), manifest)
    if mismatched:
        raise ValueError(
            f"--resume was requested for {cache_dir}, but the recorded manifest does not match the current "
            f"run configuration (mismatched fields: {mismatched}). Re-run with the original settings, or "
            "delete the directory and start fresh."
        )


def _run(args: argparse.Namespace) -> int:
    """Load the target, scan the dataset once, and write the sharded cache."""
    _validate_args(args)
    cache_dtype = DTYPE_MAP[args.dtype]

    present = existing_shard_indices(args.output_dir)
    if present and not args.resume:
        raise ValueError(f"{args.output_dir} already has shards; pass --resume to continue or use a fresh dir.")
    existing = present if args.resume else set()
    if existing:
        logger.info("Resume: %d shard(s) already present, will be skipped.", len(existing))

    device = torch.device(args.device)
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_type = _read_target_model_type(args.target_model, args.trust_remote_code)
    if model_type in _UNSUPPORTED_OFFLINE_TARGET_MODEL_TYPES:
        raise ValueError(
            "precompute_dspark supports HF-loadable single-process targets only. "
            f"model_type={model_type!r}: DeepSeek V4 / GLM-5.2 targets use the distributed offline precompute "
            "(python -m nemo_automodel.recipes.llm.precompute_dspark_dist); MiniMax M3 is multimodal and the "
            "DSpark cache schema is text-only, so it needs the online training path."
        )

    target_config = AutoConfig.from_pretrained(args.target_model, trust_remote_code=args.trust_remote_code)
    target_text_config = target_config.text_config if model_type in _GEMMA4_MODEL_TYPES else target_config
    num_target_layers = int(target_text_config.num_hidden_layers)
    target_layer_ids = list(
        args.target_layer_ids or build_target_layer_ids(num_target_layers, args.draft_num_hidden_layers)
    )

    tokenizer = NeMoAutoTokenizer.from_pretrained(args.target_model, trust_remote_code=args.trust_remote_code)
    _apply_target_chat_template(tokenizer, args.chat_template)

    target_kwargs = {}
    if args.target_attn_implementation is not None:
        target_kwargs["attn_implementation"] = args.target_attn_implementation
    target_model = NeMoAutoModelForCausalLM.from_pretrained(
        args.target_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=compute_dtype,
        force_hf=args.target_force_hf,
        **target_kwargs,
    ).to(device)
    target_model.eval()
    target_model.requires_grad_(False)
    target_wrapper = HFDSparkTargetModel(target_model, target_layer_ids=target_layer_ids)

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
    )
    num_samples = len(dataloader.dataset)

    # The manifest guards resume for tensor-shaping target/cache settings. If dataset
    # or tokenization inputs change, use a fresh output directory instead of --resume.
    manifest = build_cache_manifest(
        target_model=args.target_model,
        target_model_type=model_type,
        target_text_config=target_text_config,
        seq_length=args.seq_length,
        dtype=args.dtype,
        num_samples=num_samples,
        shard_size=args.shard_size,
        target_layer_ids=list(target_wrapper.target_layer_ids),
    )
    if args.resume:
        _ensure_resume_compatible(args.output_dir, manifest, existing)
    write_target_weights(
        args.output_dir,
        target_model.get_input_embeddings(),
        target_model.get_output_embeddings(),
        dtype=cache_dtype,
    )
    write_manifest(args.output_dir, manifest)

    logger.info(
        "Precomputing DSpark cache: %d samples, shard_size=%d, layers=%s, dtype=%s -> %s",
        num_samples,
        args.shard_size,
        target_wrapper.target_layer_ids,
        args.dtype,
        args.output_dir,
    )

    start_sample = resume_start_sample(existing, args.shard_size) if existing else 0
    dataloader = dataloader_from_sample(dataloader, start_sample)

    def _compute_from_batch(batch):
        with torch.no_grad():
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target_batch = target_wrapper.generate_batch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                loss_mask=batch["loss_mask"],
            )
            return compute_batch_cache(target_batch, cache_dtype)

    write_cache_shards(
        dataloader=dataloader,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        start_shard_index=start_sample // args.shard_size,
        compute_batch=_compute_from_batch,
        write_shard_fn=write_shard,
        logger=logger,
    )

    logger.info("Done. Cache written to %s", args.output_dir)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute the DSpark offline target-supervision cache.")
    parser.add_argument("--target-model", required=True, help="Target model path or HF repo id.")
    parser.add_argument("--input-data", required=True, help="HF dataset id or local chat dataset path.")
    parser.add_argument("--output-dir", required=True, help="Directory to write cache shards + manifest into.")
    parser.add_argument("--seq-length", type=int, default=2048, help="Sequence length.")
    parser.add_argument("--batch-size", type=int, default=4, help="Target forward batch size.")
    parser.add_argument(
        "--shard-size", type=int, default=256, help="Samples per shard (must be a multiple of --batch-size)."
    )
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPE_MAP), help="Cache float dtype.")
    parser.add_argument("--device", default="cuda", help="Device to run the target on (cuda / cpu).")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--split", default=None, help="HF dataset split (supports slice syntax).")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Dataset ingestion shuffle seed.")
    parser.add_argument("--draft-num-hidden-layers", type=int, default=5, help="Draft layer count for default layers.")
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=None, help="Target layers to capture.")
    parser.add_argument(
        "--chat-template", default=None, help="Inline Jinja template or path to template/tokenizer config."
    )
    parser.add_argument("--mask-reasoning-content", action="store_true")
    parser.add_argument("--target-attn-implementation", default=None)
    parser.add_argument("--target-force-hf", action="store_true")
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
