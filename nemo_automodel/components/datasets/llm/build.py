# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""LLM training and validation dataloader builders.

Used by ``recipes/llm/train_ft.py`` (and re-exported there for back-compat).
Also exports a few small recipe-private helpers that configure CP/THD paths
(``_uses_te_dot_product_attention``, ``_uses_thd_collater``,
``_get_num_thd_chunks``) — they're imported by recipes (kd.py, train_ft.py)
and by ``build_dataloader`` itself.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from transformers import AutoConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_automodel._transformers.auto_tokenizer import (
    _build_tokenizer,
    _get_model_name,
    compute_trust_remote_code_from_model,
)
from nemo_automodel.components.datasets.llm.megatron.sampler import create_megatron_sampler
from nemo_automodel.components.datasets.llm.megatron_dataset import MegatronPretraining
from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset
from nemo_automodel.components.distributed.utils import FirstRankPerNode
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.utils.model_utils import _supports_seq_lens

logger = logging.getLogger(__name__)


# ── THD / TE helper detectors (recipe-private) ────────────────────────


def _uses_te_dot_product_attention(model_or_cfg):
    """Check whether the model uses TE DotProductAttention.

    Accepts either an instantiated nn.Module (preferred — inspects actual modules)
    or a config object (fallback — checks backend.attn string).
    """
    if isinstance(model_or_cfg, torch.nn.Module):
        try:
            from transformer_engine.pytorch.attention import DotProductAttention
        except ImportError:
            return False
        return any(isinstance(m, DotProductAttention) for m in model_or_cfg.modules())
    # Config fallback for call sites before model is built
    return (
        hasattr(model_or_cfg, "backend") and hasattr(model_or_cfg.backend, "attn") and model_or_cfg.backend.attn == "te"
    )


def _uses_thd_collater(cfg_dataloader):
    """``True`` if the configured collate fn is the packed-sequence THD collater."""
    from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater

    return (
        True
        if hasattr(cfg_dataloader, "collate_fn") and cfg_dataloader.collate_fn == packed_sequence_thd_collater
        else False
    )


def _get_num_thd_chunks(pp_enabled, cfg):
    """Number of THD chunks the dataloader should produce per batch (PP-aware)."""
    if pp_enabled:
        return cfg.step_scheduler.local_batch_size // cfg.get("distributed.pipeline.pp_microbatch_size", 1)
    return 1


# ── Dataloader builders ────────────────────────────────────────────────


def build_dataloader(
    cfg_ds,
    cfg_dl,
    cfg_model,
    cfg_ps,
    seed,
    local_batch_size,
    global_batch_size,
    max_steps,
    val_check_interval,
    dp_rank,
    dp_world_size,
    pp_enabled,
    cp_size=1,
    model: Optional[nn.Module] = None,
) -> tuple[DataLoader, PreTrainedTokenizerBase]:
    """Build a DataLoader for the dataset.

    Args:
        cfg_ds: Dataset configuration.
        cfg_dl: DataLoader configuration.
        cfg_model: Model configuration.
        cfg_ps: Packed sequence configuration.
        seed: Random seed.
        local_batch_size: Local batch size.
        global_batch_size: Global batch size.
        max_steps: Maximum number of steps.
        val_check_interval: Validation check interval.
        dp_rank: Data parallel rank.
        dp_world_size: Data parallel world size.
        pp_enabled: Whether pipeline parallelism is enabled.
        cp_size: Context parallel size.
        model: Optional model instance. If provided and packed sequences are enabled,
            seq_lens will only be included if the model's forward() accepts it.
    Returns:
        The instantiated DataLoader and tokenizer.
    """
    with ScopedRNG(seed=seed, ranked=True):
        kwargs, tokenizer = _build_tokenizer(cfg_model, cfg_ds)
        # Megatron specific kwargs
        if cfg_ds._target_ == MegatronPretraining:
            kwargs["global_batch_size"] = global_batch_size
            kwargs["trainer_max_steps"] = max_steps if max_steps is not None else None
            kwargs["trainer_val_check_interval"] = val_check_interval
            ds = cfg_ds.instantiate(**kwargs)
            ds.build()
        else:
            with FirstRankPerNode():
                ds = cfg_ds.instantiate(**kwargs)

        # If using an IterableDataset, per-rank sharding for unique samples
        if isinstance(ds, IterableDataset):
            if callable(getattr(ds, "shard", None)):
                ds = ds.shard(dp_world_size, dp_rank)
                logging.info(f"Sharded IterableDataset via dataset.shard: world_size={dp_world_size}, rank={dp_rank}")
            elif hasattr(ds, "dataset"):
                # HuggingFace streaming datasets: split by file shards when possible.
                from datasets.distributed import split_dataset_by_node

                assert hasattr(ds, "dataset"), "dataset must have a dataset attribute"
                ds.dataset = split_dataset_by_node(ds.dataset, world_size=dp_world_size, rank=dp_rank)
                logging.info(f"Sharded dataset via split_dataset_by_node: world_size={dp_world_size}")
            else:
                logging.warning("IterableDataset does not support sharding; Data may be duplicated across ranks.")

        packed_sequence_size = getattr(cfg_ps, "packed_sequence_size", 0)
        packing_strategy = getattr(cfg_ps, "packing_strategy", "thd")

        # check if packed sequence is supported (only for thd strategy)
        supports_seq_lens = _supports_seq_lens(model)
        if packed_sequence_size > 0 and packing_strategy == "thd" and not supports_seq_lens:
            logging.warning("Packed sequence is not supported without seq_lens; disabling packed sequence")
            packed_sequence_size = 0

        # Apply packing if configured
        if packed_sequence_size > 0:
            logger.info(f"Packing dataset with size: {packed_sequence_size}, strategy: {packing_strategy}")
            if hasattr(ds, "shuffle"):
                ds = ds.shuffle(seed)

            if packing_strategy == "neat":
                from nemo_automodel.components.datasets.llm.neat_packing import neat_pack_dataset
                from nemo_automodel.components.datasets.utils import neat_packed_collater
                from nemo_automodel.components.models.common.packing import configure_packing, get_attn_implementation

                ds = neat_pack_dataset(
                    ds,
                    split=cfg_ds.split,
                    pack_size=packed_sequence_size,
                    max_packs=getattr(cfg_ps, "max_packs", None),
                    padding_idx=getattr(tokenizer, "pad_token_id", 0),
                    drop_long_samples=getattr(cfg_ps, "drop_long_samples", True),
                )
                _attn_impl = get_attn_implementation(cfg_model)
                configure_packing(attn_implementation=_attn_impl)
                # Set collater with attn_implementation so it produces the right mask format
                cfg_dl.collate_fn = lambda batch, _ai=_attn_impl: neat_packed_collater(batch, attn_implementation=_ai)
                logger.info(f"Configured neat packing for attn_implementation={_attn_impl}")
            else:
                # "thd" — existing packing logic
                ds = pack_dataset(
                    ds,
                    split=cfg_ds.split,
                    packed_sequence_size=packed_sequence_size,
                    max_packs=getattr(cfg_ps, "max_packs", None),
                    padding_idx=getattr(tokenizer, "pad_token_id", 0),
                    cp_size=cp_size,
                )

        if isinstance(ds, MegatronPretraining):
            ds = ds.get_dataset(split=cfg_ds.splits_to_build)
            dataloader_type = cfg_dl.get("dataloader_type", "single")
            if "dataloader_type" in cfg_dl:
                del cfg_dl.dataloader_type
            batch_sampler = create_megatron_sampler(
                dataset_len=len(ds),
                micro_batch_size=local_batch_size,
                global_batch_size=global_batch_size,
                dataloader_type=dataloader_type,
                rank=dp_rank,
                world_size=dp_world_size,
            )
            dl_kwargs = {"batch_sampler": batch_sampler}
        elif not isinstance(ds, IterableDataset):
            shuffle = cfg_dl.get("shuffle", True)
            if "shuffle" in cfg_dl:
                del cfg_dl.shuffle

            group_by_length = cfg_dl.get("group_by_length", False)
            if "group_by_length" in cfg_dl:
                del cfg_dl.group_by_length

            if group_by_length:
                from nemo_automodel.components.datasets.llm.length_grouped_sampler import (
                    LengthGroupedSampler as LLMLengthGroupedSampler,
                )

                sampler = LLMLengthGroupedSampler(
                    dataset=ds,
                    batch_size=local_batch_size,
                    seed=seed,
                    num_replicas=dp_world_size,
                    rank=dp_rank,
                )
            else:
                dist_sampler_kwargs = {
                    "num_replicas": dp_world_size,
                    "rank": dp_rank,
                    "shuffle": shuffle,
                }
                sampler = StatefulDistributedSampler(
                    ds,
                    seed=seed,
                    drop_last=True,
                    **dist_sampler_kwargs,
                )
            dl_kwargs = {"sampler": sampler, "batch_size": local_batch_size}
            if pp_enabled:
                dl_kwargs["drop_last"] = True
        else:
            logging.info("Using IterableDataset; skipping sampler.")
            # Optional shuffle for streaming IterableDataset (uses HF dataset shuffle if available)
            shuffle = cfg_dl.get("shuffle", False)
            shuffle_buffer_size = cfg_dl.get("shuffle_buffer_size", 10000)
            # Do not pass shuffle-related kwargs to the DataLoader when using IterableDataset
            # But leave them in dl config to be consistent
            if hasattr(cfg_dl, "shuffle"):
                del cfg_dl.shuffle
            if hasattr(cfg_dl, "shuffle_buffer_size"):
                del cfg_dl.shuffle_buffer_size

            if shuffle and hasattr(ds, "shuffle"):
                try:
                    ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
                    logging.info(f"Shuffling IterableDataset with buffer_size={shuffle_buffer_size}, seed={seed}")
                except Exception as e:
                    logging.warning(f"IterableDataset shuffle skipped due to error: {e}")
            dl_kwargs = {}

        # Handle collate_fn with optional mask precomputation for pipeline parallelism
        dl_kwargs = dl_kwargs | {"dataset": ds}

        # Handle collate_fn instantiation if it's a ConfigNode
        if hasattr(cfg_dl, "collate_fn"):
            if hasattr(cfg_dl.collate_fn, "_target_"):
                collate_cfg = cfg_dl.collate_fn
                dl_kwargs["collate_fn"] = lambda batch: collate_cfg.instantiate(batch=batch)
            else:
                dl_kwargs["collate_fn"] = cfg_dl.collate_fn
            assert callable(dl_kwargs["collate_fn"]), "collate_fn must be callable"

        # Chain with mask precomputation if PP is enabled
        if pp_enabled:
            from nemo_automodel.components.datasets.utils import add_causal_masks_to_batch

            try:
                hf_model_config = AutoConfig.from_pretrained(
                    _get_model_name(cfg_model), trust_remote_code=compute_trust_remote_code_from_model(cfg_model)
                )
            except Exception:
                logger.warning(
                    "Failed to load model config for causal mask precomputation. "
                    "Pipeline parallel mask precomputation will be skipped."
                )
            else:
                if "collate_fn" in dl_kwargs:
                    # Case 1: PP enabled + collate_fn exists -> chain them
                    # base_collate_fn -> add_causal_masks_to_batch
                    base_collate_fn = dl_kwargs["collate_fn"]

                    def chained_collate_fn(batch, base_fn=base_collate_fn, config=hf_model_config):
                        batch = base_fn(batch)  # Apply base collate (padding, batching, etc.)
                        batch = add_causal_masks_to_batch(batch, model_config=config)  # Add masks
                        return batch

                    dl_kwargs["collate_fn"] = chained_collate_fn
                else:
                    # Case 2: PP enabled + no collate_fn -> only add masks
                    dl_kwargs["collate_fn"] = lambda batch, config=hf_model_config: add_causal_masks_to_batch(
                        batch, model_config=config
                    )

        try:
            import torch.multiprocessing as mp

            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        return cfg_dl.instantiate(**dl_kwargs), tokenizer


def build_validation_dataloader(cfg, dp_world_size, dp_rank, pp_enabled, model: Optional[nn.Module] = None):
    """Build validation dataloaders from validation dataset config entries."""

    def _prepare_val_ds_name(val_ds_name):
        val_ds_name = val_ds_name.replace("validation_dataset", "")
        if len(val_ds_name) > 1 and val_ds_name[0] in ("_", "-", "."):
            val_ds_name = val_ds_name[1:]
        if val_ds_name == "":
            val_ds_name = "default"
        return val_ds_name

    # Build validation dataloader if the config provides it
    val_dataloaders = {}
    for val_ds_name in filter(lambda x: x.startswith("validation_dataset"), cfg.to_dict().keys()):
        val_ds_cfg = cfg.get(val_ds_name, None)
        val_ds_name = _prepare_val_ds_name(val_ds_name)
        val_dataloaders[val_ds_name] = build_dataloader(
            val_ds_cfg,
            cfg.validation_dataloader,
            cfg.model,
            cfg_ps=cfg.get("packed_sequence", None)
            if _uses_te_dot_product_attention(cfg.model) and _uses_thd_collater(cfg.dataloader)
            else None,
            seed=cfg.get("seed", 42),
            local_batch_size=cfg.get("step_scheduler.local_batch_size", 1),
            global_batch_size=cfg.get("step_scheduler.global_batch_size", 1),
            max_steps=cfg.get("step_scheduler.max_steps", None),
            val_check_interval=cfg.get("step_scheduler.val_every_steps", None),
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            pp_enabled=pp_enabled,
            cp_size=cfg.get("distributed.cp_size", 1),
            model=model,
        )[0]

    return val_dataloaders


__all__ = [
    "_get_num_thd_chunks",
    "_uses_te_dot_product_attention",
    "_uses_thd_collater",
    "build_dataloader",
    "build_validation_dataloader",
]
