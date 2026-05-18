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

"""VLM dataloader builder.

Mirrors the LLM counterpart in ``components/datasets/llm/build.py`` but for
image-text datasets: loads an :class:`AutoProcessor`, supports VLM neat
packing with mRoPE position IDs, and dispatches collate functions by
processor type.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin

from nemo_automodel.components.datasets.llm.formatting_utils import _resolve_chat_template
from nemo_automodel.components.datasets.vlm.collate_fns import COLLATE_FNS
from nemo_automodel.components.distributed.utils import FirstRankPerNode
from nemo_automodel.components.training.rng import ScopedRNG


def build_vlm_dataloader(
    cfg_ds,
    cfg_dl,
    pretrained_model_name_or_path,
    cfg_processor,
    device_mesh,
    seed,
    local_batch_size,
    cfg_model=None,
    cfg_ps=None,
    get_rope_index=None,
) -> tuple[DataLoader, ProcessorMixin]:
    """Build a DataLoader for the VLM dataset.

    Args:
        cfg_ds: Dataset configuration.
        cfg_dl: DataLoader configuration.
        pretrained_model_name_or_path: Pretrained model name or path for processor loading.
        cfg_processor: Processor configuration or None.
        device_mesh: Device mesh for distributed training.
        seed: Random seed.
        local_batch_size: Local batch size.
        cfg_model: Model configuration (used to detect attention backend).
        cfg_ps: Packed sequence configuration (top-level ``packed_sequence:`` section).
            When provided, takes precedence over ``dataset.packing``.
        get_rope_index: Optional ``model.get_rope_index`` callable. When provided,
            VLM neat packing computes mRoPE 3D position IDs per sample so packed
            mRoPE-aware models (Qwen2.5-VL, Qwen3-VL, ...) preserve multimodal
            position semantics across pack boundaries instead of falling back to
            plain 1D positions.

    Returns:
        The instantiated DataLoader and processor.
    """
    dist_sampler_kwargs = {
        "shuffle": cfg_dl.get("shuffle", True),
    }
    if device_mesh is not None:
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

        dp_mesh = get_flat_mesh(device_mesh, "dp")
        dist_sampler_kwargs |= {
            "num_replicas": dp_mesh.size(),
            "rank": dp_mesh.get_local_rank(),
        }

    with ScopedRNG(seed=seed, ranked=True):
        processor = None
        processor_kwargs = {}

        with FirstRankPerNode():
            if (
                cfg_processor is not None
                and hasattr(cfg_processor, "instantiate")
                and hasattr(cfg_processor, "_target_")
            ):
                processor = cfg_processor.instantiate()
            elif cfg_processor is not None:
                processor_kwargs = cfg_processor.to_dict()

            if processor is None:
                try:
                    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **processor_kwargs)
                except Exception as e:
                    # AutoProcessor.from_pretrained internally loads AutoConfig. Configs
                    # whose layer_types length differs from num_hidden_layers trip
                    # validate_layer_type. Relax the validator and retry once.
                    err = str(e)
                    if "num_hidden_layers" in err and ("layer_types" in err or "layer types" in err):
                        from nemo_automodel._transformers.v4_patches.layer_types import (
                            relax_layer_types_validator,
                        )

                        relax_layer_types_validator()
                        try:
                            processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **processor_kwargs)
                        except Exception as retry_exc:
                            processor = None
                            logging.warning(
                                f"AutoProcessor not available for {pretrained_model_name_or_path} ({retry_exc}). "
                            )
                    else:
                        processor = None
                        logging.warning(f"AutoProcessor not available for {pretrained_model_name_or_path} ({e}). ")

            chat_template_raw = cfg_ds.__dict__.pop("chat_template", None)
            if chat_template_raw is not None and processor is not None:
                processor.chat_template = _resolve_chat_template(chat_template_raw)
                processor.tokenizer.chat_template = processor.chat_template

            _path_or_ds = getattr(cfg_ds, "path_or_dataset", None) or cfg_ds.get("path_or_dataset", None)
            if _path_or_ds is not None:
                ds = cfg_ds.instantiate(path_or_dataset=_path_or_ds)
            else:
                ds = cfg_ds.instantiate()

        # Resolve packing config: top-level packed_sequence (LLM-style) takes
        # precedence over legacy dataset.packing (backward compat).
        if cfg_ps is not None:
            _ps_enabled = getattr(cfg_ps, "pack_size", 0) > 0
            packing_cfg = cfg_ps if _ps_enabled else None
            pretokenize = getattr(cfg_ps, "pretokenize", _ps_enabled)
            max_length = getattr(cfg_ps, "max_length", None)
        else:
            _legacy = cfg_ds.get("packing", None)
            _ps_enabled = _legacy is not None and _legacy.get("enabled", False)
            packing_cfg = _legacy if _ps_enabled else None
            max_length = cfg_ds.get("max_length", None)
            pretokenize = cfg_ds.get("pretokenize", max_length is not None)

        if pretokenize:
            from nemo_automodel.components.datasets.vlm.collate_fns import pad_collate_fn
            from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper

            ds_raw = ds
            truncate = cfg_ds.get("truncate", max_length is not None)

            post_tokenize_hook = cfg_ps.get("post_tokenize_hook_fn", None) if cfg_ps is not None else None

            ds = PreTokenizedDatasetWrapper(
                ds_raw,
                processor,
                max_length=max_length,
                truncate=truncate,
                post_tokenize_hook=post_tokenize_hook,
            )

            if packing_cfg:
                from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
                from nemo_automodel.components.datasets.vlm.neat_packing_vlm import neat_pack_dataset_vlm
                from nemo_automodel.components.models.common.packing import configure_packing, get_attn_implementation

                ds = neat_pack_dataset_vlm(
                    ds,
                    pack_size=packing_cfg.get("pack_size", max_length),
                    padding_idx=getattr(processor.tokenizer, "pad_token_id", 0) or 0,
                    drop_long_samples=packing_cfg.get("drop_long_samples", True),
                    max_packs=packing_cfg.get("max_packs", None),
                    ds_raw=ds_raw,
                    packing_ratio=packing_cfg.get("packing_ratio", 1.0),
                    processor=processor,
                    balance_media_tokens=packing_cfg.get("balance_media_tokens", True),
                    get_rope_index=get_rope_index,
                )
                _pad_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0
                _collate_max_length = packing_cfg.get("collate_max_length", None)
                _attn_impl = get_attn_implementation(cfg_model)

                configure_packing(attn_implementation=_attn_impl)
                logging.info(f"Configured VLM neat packing for attn_implementation={_attn_impl}")

                collate_fn = lambda examples, _pi=_pad_id, _ml=_collate_max_length, _ai=_attn_impl: (
                    neat_packed_vlm_collater(
                        examples,
                        padding_idx=_pi,
                        max_length=_ml,
                        attn_implementation=_ai,
                    )
                )
            else:
                collate_cfg = cfg_dl.get("collate_fn", None)
                if collate_cfg:
                    collate_fn = lambda examples: collate_cfg.instantiate(examples=examples, processor=processor)
                else:
                    collate_fn = lambda examples: pad_collate_fn(examples, processor)

            sampler = torch.utils.data.distributed.DistributedSampler(
                ds,
                **dist_sampler_kwargs,
            )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds,
                **dist_sampler_kwargs,
            )
            collate_cfg = cfg_dl.get("collate_fn", None)
            if collate_cfg:
                collate_fn = lambda examples: collate_cfg.instantiate(examples=examples, processor=processor)
            else:
                processor_type = type(processor).__name__
                if processor_type not in COLLATE_FNS:
                    logging.warning(f"You are using {processor_type} with default collate function.")
                    processor_type = "default"
                collate_fn = lambda examples: COLLATE_FNS[processor_type](examples, processor)

        if hasattr(ds, "robust_collate"):
            collate_fn = ds.robust_collate(collate_fn)

        return cfg_dl.instantiate(
            dataset=ds, sampler=sampler, collate_fn=collate_fn, batch_size=local_batch_size
        ), processor


def build_validation_dataloader(
    cfg,
    *,
    device_mesh,
    pretrained_model_name_or_path: str,
    get_rope_index=None,
) -> dict:
    """Build VLM validation dataloaders from ``validation_dataset*`` cfg entries.

    Mirrors the LLM-side helper: scans ``cfg.to_dict().keys()`` for any key
    starting with ``validation_dataset`` (e.g. ``validation_dataset``,
    ``validation_dataset_val``, ``validation_dataset-test``,
    ``validation_dataset.foo``) and builds one DataLoader per entry, keyed by
    the suffix (``default`` for the bare key)."""

    def _prepare_val_ds_name(val_ds_name):
        val_ds_name = val_ds_name.replace("validation_dataset", "")
        if len(val_ds_name) > 1 and val_ds_name[0] in ("_", "-", "."):
            val_ds_name = val_ds_name[1:]
        if val_ds_name == "":
            val_ds_name = "default"
        return val_ds_name

    val_dataloaders = {}
    for val_ds_name in filter(lambda x: x.startswith("validation_dataset"), cfg.to_dict().keys()):
        val_ds_cfg = cfg.get(val_ds_name, None)
        val_ds_name = _prepare_val_ds_name(val_ds_name)
        val_dataloaders[val_ds_name] = build_vlm_dataloader(
            val_ds_cfg,
            cfg.validation_dataloader,
            pretrained_model_name_or_path,
            cfg.get("processor", None),
            device_mesh=device_mesh,
            seed=cfg.get("seed", 42),
            local_batch_size=cfg.get("step_scheduler.local_batch_size", 1),
            get_rope_index=get_rope_index,
        )[0]

    return val_dataloaders


__all__ = ["build_validation_dataloader", "build_vlm_dataloader"]
