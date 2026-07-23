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

"""Distributed (multi-node) precompute of the DSpark offline target-supervision cache.

The single-process ``precompute_dspark`` script loads the whole target on one box,
so very large targets (DeepSeek-V4-Flash, GLM-5.2) that do not fit on a single
8x80GB node cannot be precomputed with it. This entry point loads such a target
frozen through the same expert-parallel / FSDP distributed path the training recipe
uses (the routed experts shard across ranks), runs it once over the dataset, and
writes the same on-disk DSpark cache that ``train_dspark`` consumes through
``recipe_args.cached_target_path`` -- with no live target during draft training.

It is config-driven (it reuses a training-style YAML: the ``distributed`` block that
shapes the target's EP/FSDP mesh, plus the ``recipe_args`` target / data / draft
fields), and is launched with ``torchrun`` exactly like multi-node training::

    torchrun --nnodes=4 --node-rank=0 --nproc_per_node=8 \\
        --master-addr=<NODE0_IP> --master-port=29500 \\
        -m nemo_automodel.recipes.llm.precompute_dspark_dist \\
        -c examples/speculative/dspark/deepseek_v4_flash_dspark_precompute.yaml

Each rank forwards a contiguous, shard-aligned slice of the dataset and writes its
own global-indexed shards straight into the shared ``cache_output_dir`` (the fleet's
shared filesystem), so no post-hoc merge is needed. Small text targets (Qwen3,
Gemma4) are also accepted and simply run data-parallel-replicated for throughput.
MiniMax M3 (multimodal) is out of scope: the cache schema is text-only.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.dspark_cache import (
    DTYPE_MAP,
    build_cache_manifest,
    compute_batch_cache,
    existing_shard_indices,
    manifest_mismatch_fields,
    manifest_path,
    read_manifest,
    tokenizer_chat_template_sha256,
    write_manifest,
    write_shard,
    write_target_weights,
)
from nemo_automodel.components.datasets.llm.eagle3 import build_eagle3_dataloader
from nemo_automodel.components.datasets.llm.offline_cache import write_cache_shards_distributed
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.speculative.dspark.common import validate_target_layer_ids
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
from nemo_automodel.recipes.llm._dspark_target_build import (
    build_deepseek_v4_target,
    build_glm_5_2_target,
    gather_full_weight_module,
)

logger = logging.getLogger(__name__)


def _resolve_cache_settings(recipe_cfg) -> tuple[str, str, int]:
    """Extract and validate the ``(output_dir, dtype, shard_size)`` cache knobs."""
    output_dir = recipe_cfg.get("cache_output_dir", None)
    if not output_dir:
        raise ValueError("recipe_args.cache_output_dir is required for distributed DSpark precompute.")
    dtype = str(recipe_cfg.get("cache_dtype", "bf16"))
    if dtype not in DTYPE_MAP:
        raise ValueError(f"recipe_args.cache_dtype must be one of {sorted(DTYPE_MAP)}, got {dtype!r}.")
    shard_size = int(recipe_cfg.get("cache_shard_size", 256))
    batch_size = int(recipe_cfg.get("micro_batch_size", 1))
    if shard_size < 1:
        raise ValueError(f"recipe_args.cache_shard_size must be >= 1, got {shard_size}.")
    if shard_size % batch_size != 0:
        raise ValueError(
            f"recipe_args.cache_shard_size ({shard_size}) must be a multiple of micro_batch_size "
            f"({batch_size}) so shards align to batch boundaries."
        )
    return output_dir, dtype, shard_size


def _ensure_output_dir_compatible(output_dir: str, manifest: dict[str, Any]) -> None:
    """Refuse to write into a directory whose existing manifest describes a different cache.

    Distributed precompute writes are idempotent (atomic overwrite), so re-running the
    same config into the same directory safely recomputes. But mixing shards from a
    different target / dataset / shape silently corrupts the cache, so a mismatched
    existing manifest is a hard error (use a fresh ``cache_output_dir``). The manifest
    carries the run's input identity (dataset path/split, shuffle seed, masking,
    effective chat template), so a same-shape different-input rerun is also rejected
    here rather than interleaving old and new supervision. ``allow_incomplete`` lets
    a rerun continue into a directory whose previous run was interrupted.
    """
    if not os.path.exists(manifest_path(output_dir)):
        return
    mismatched = manifest_mismatch_fields(read_manifest(output_dir, allow_incomplete=True), manifest)
    if mismatched:
        raise ValueError(
            f"{output_dir} already holds a DSpark cache with a different configuration "
            f"(mismatched fields: {mismatched}). Use a fresh cache_output_dir or delete it."
        )


def _build_target(
    *,
    cfg,
    recipe_cfg,
    world_size: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    model_type: str,
    target_path: str,
    trust_remote_code: bool,
):
    """Build the frozen target for capture, dispatching on model type.

    DeepSeek V4 / GLM-5.2 load through the sharded EP/FSDP path; other single-process
    text targets (Qwen3, Gemma4) load replicated for data-parallel throughput.
    Returns ``(target_config, target_model)``.
    """
    if model_type in _MINIMAX_M3_MODEL_TYPES:
        raise ValueError(
            "MiniMax M3 is a multimodal target; the DSpark offline cache schema is text-only. "
            "Use the online MiniMax M3 training path instead."
        )
    if model_type == _DEEPSEEK_V4_MODEL_TYPE:
        target_config, target_model, _ = build_deepseek_v4_target(
            cfg=cfg,
            world_size=world_size,
            device=device,
            compute_dtype=compute_dtype,
            target_path=target_path,
            recipe_cfg=recipe_cfg,
            trust_remote_code=trust_remote_code,
        )
        return target_config, target_model
    if model_type == _GLM_5_2_MODEL_TYPE:
        target_config, target_model, _ = build_glm_5_2_target(
            cfg=cfg,
            world_size=world_size,
            device=device,
            compute_dtype=compute_dtype,
            target_path=target_path,
            recipe_cfg=recipe_cfg,
            trust_remote_code=trust_remote_code,
        )
        return target_config, target_model
    # Single-process text target (Qwen3, Gemma4): replicate on every rank; data
    # parallelism over the contiguous per-rank shard blocks provides the throughput.
    target_kwargs = {}
    target_attn_implementation = recipe_cfg.get("target_attn_implementation", None)
    if target_attn_implementation is not None:
        target_kwargs["attn_implementation"] = target_attn_implementation
    target_model = NeMoAutoModelForCausalLM.from_pretrained(
        target_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=compute_dtype,
        force_hf=bool(recipe_cfg.get("target_force_hf", False)),
        **target_kwargs,
    ).to(device)
    return target_model.config, target_model


def _make_sync_max_steps(world_size: int, device: torch.device):
    """Return an all-reduce-MAX reducer over the default group, or the identity."""
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return None

    def _sync(local_steps: int) -> int:
        tensor = torch.tensor([int(local_steps)], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return int(tensor.item())

    return _sync


def run(cfg) -> int:
    """Load the (possibly sharded) target and write the distributed DSpark cache."""
    dist_env = initialize_distributed(
        backend=cfg.get("dist_env", {}).get("backend", "nccl"),
        timeout_minutes=cfg.get("dist_env", {}).get("timeout_minutes", 30),
    )
    device = dist_env.device or torch.device("cpu")
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    recipe_cfg = cfg.recipe_args
    trust_remote_code = bool(recipe_cfg.get("trust_remote_code", False))
    target_path = recipe_cfg.target_model_name_or_path
    seq_length = int(recipe_cfg.seq_length)
    batch_size = int(recipe_cfg.get("micro_batch_size", 1))

    output_dir, cache_dtype_str, shard_size = _resolve_cache_settings(recipe_cfg)
    cache_dtype = DTYPE_MAP[cache_dtype_str]

    model_type = _read_target_model_type(target_path, trust_remote_code)
    target_config, target_model = _build_target(
        cfg=cfg,
        recipe_cfg=recipe_cfg,
        world_size=dist_env.world_size,
        device=device,
        compute_dtype=compute_dtype,
        model_type=model_type,
        target_path=target_path,
        trust_remote_code=trust_remote_code,
    )
    target_model.eval()
    target_model.requires_grad_(False)

    tokenizer = NeMoAutoTokenizer.from_pretrained(target_path, trust_remote_code=trust_remote_code)
    _apply_target_chat_template(tokenizer, recipe_cfg.get("chat_template", None))

    target_text_config = target_config.text_config if model_type in _GEMMA4_MODEL_TYPES else target_config
    num_target_layers = int(target_text_config.num_hidden_layers)
    draft_num_hidden_layers = int(recipe_cfg.get("draft_num_hidden_layers", 5))
    target_layer_ids = list(
        recipe_cfg.get("target_layer_ids", None) or build_target_layer_ids(num_target_layers, draft_num_hidden_layers)
    )
    target_layer_ids = validate_target_layer_ids(target_layer_ids, num_target_layers)
    target_wrapper = HFDSparkTargetModel(target_model, target_layer_ids=target_layer_ids)

    # Resolved once and passed to BOTH the dataloader and the manifest, so the
    # recorded input identity is exactly what the forward pass consumed.
    train_split = recipe_cfg.get("train_split", None)
    shuffle_seed = int(recipe_cfg.get("shuffle_seed", 42))
    mask_reasoning_content = bool(recipe_cfg.get("mask_reasoning_content", False))
    dataloader = build_eagle3_dataloader(
        data_path=recipe_cfg.train_data_path,
        tokenizer=tokenizer,
        seq_length=seq_length,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(recipe_cfg.get("num_workers", 0)),
        split=train_split,
        distributed=False,
        shuffle_seed=shuffle_seed,
        mask_reasoning_content=mask_reasoning_content,
    )
    num_samples = len(dataloader.dataset)

    manifest = build_cache_manifest(
        target_model=target_path,
        target_model_type=model_type,
        target_text_config=target_text_config,
        seq_length=seq_length,
        dtype=cache_dtype_str,
        num_samples=num_samples,
        shard_size=shard_size,
        target_layer_ids=list(target_wrapper.target_layer_ids),
        train_data_path=str(recipe_cfg.train_data_path),
        train_split=train_split,
        shuffle_seed=shuffle_seed,
        mask_reasoning_content=mask_reasoning_content,
        chat_template_sha256=tokenizer_chat_template_sha256(tokenizer),
    )

    # Gather the (possibly DTensor-sharded) target embed_tokens / lm_head to full
    # tensors: an all-gather that EVERY rank must enter in lockstep, before only rank
    # zero writes the manifest + weights.
    embed_full = gather_full_weight_module(target_model.get_input_embeddings())
    head_full = gather_full_weight_module(target_model.get_output_embeddings())
    if dist_env.is_main:
        os.makedirs(output_dir, exist_ok=True)
        _ensure_output_dir_compatible(output_dir, manifest)
        present = existing_shard_indices(output_dir)
        if present:
            logger.info("%s already has %d shard(s); they will be overwritten idempotently.", output_dir, len(present))
        write_target_weights(output_dir, embed_full, head_full, dtype=cache_dtype)
        # Staged publish: the manifest goes down as incomplete before the first
        # shard and is flipped to complete only after the final barrier, so an
        # interrupted run can never be consumed as a valid cache.
        write_manifest(output_dir, {**manifest, "complete": False})
    # The gathered full embed/lm_head tensors are only needed for the rank-zero write
    # above; on a V4/GLM-scale vocab they hold several GB per rank, so free them before
    # the long forward loop.
    del embed_full, head_full
    if dist_env.world_size > 1:
        dist.barrier()

    logger.info(
        "Precomputing DSpark cache (distributed): %d samples, shard_size=%d, layers=%s, dtype=%s -> %s",
        num_samples,
        shard_size,
        target_wrapper.target_layer_ids,
        cache_dtype_str,
        output_dir,
    )

    def _compute_from_batch(batch):
        with torch.no_grad():
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target_batch = target_wrapper.generate_batch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                loss_mask=batch["loss_mask"],
            )
            return compute_batch_cache(target_batch, cache_dtype)

    write_cache_shards_distributed(
        dataloader=dataloader,
        output_dir=output_dir,
        shard_size=shard_size,
        world_size=dist_env.world_size,
        rank=dist_env.rank,
        compute_batch=_compute_from_batch,
        write_shard_fn=write_shard,
        logger=logger,
        sync_max_steps=_make_sync_max_steps(dist_env.world_size, device),
    )

    if dist_env.world_size > 1:
        dist.barrier()
    if dist_env.is_main:
        write_manifest(output_dir, {**manifest, "complete": True})
        logger.info("Done. Distributed DSpark cache written to %s", output_dir)
    return 0


def main(config_path: str | None = None) -> int:
    """CLI entry point. Parses ``-c <config.yaml>`` and runs the precompute."""
    logging.basicConfig(
        level=getattr(logging, os.environ.get("DSPARK_PRECOMPUTE_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args_and_load_config(config_path)
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
