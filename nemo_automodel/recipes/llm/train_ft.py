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

from __future__ import annotations

import inspect
import logging
import pathlib
import time
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.nn as nn
import wandb
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader, IterableDataset
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import TRANSFORMERS_CACHE, ContextManagers
from transformers.utils.hub import TRANSFORMERS_CACHE
from wandb import Settings

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.megatron.sampler import create_megatron_sampler
from nemo_automodel.components.datasets.llm.megatron_dataset import MegatronPretraining
from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.ddp import DDPManager
from nemo_automodel.components.distributed.init_utils import (
    get_rank_safe,
    get_world_size_safe,
    initialize_distributed,
)
from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.utils import FirstRankPerNode, get_sync_ctx
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.loggers.mlflow_utils import build_mlflow
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.quantization.fp8 import apply_fp8_to_model, build_fp8_config
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.components.training.utils import (
    count_tail_padding,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.compile_utils import (
    build_compile_config,
    compile_model,
)
from nemo_automodel.components.utils.model_utils import (
    _supports_logits_to_keep,
    init_empty_weights,
    print_trainable_parameters,
    resolve_trust_remote_code,
)
from nemo_automodel.recipes.base_recipe import BaseRecipe

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from nemo_automodel.components.distributed.init_utils import DistInfo

logger = logging.getLogger(__name__)


# ---------------------------
#  Stateless helper functions
# ---------------------------
def _get_model_name(cfg_model):
    if cfg_model.get("pretrained_model_name_or_path", None) is not None:
        return cfg_model.pretrained_model_name_or_path
    elif cfg_model.get("config", None) is not None:
        return cfg_model.config.get("pretrained_model_name_or_path", None)
    else:
        return None


def _uses_te_dot_product_attention(cfg_model):
    return (
        True
        if hasattr(cfg_model, "backend") and hasattr(cfg_model.backend, "attn") and cfg_model.backend.attn == "te"
        else False
    )


def _uses_thd_collater(cfg_dataloader):
    from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater

    return (
        True
        if hasattr(cfg_dataloader, "collate_fn") and cfg_dataloader.collate_fn == packed_sequence_thd_collater
        else False
    )


def _get_num_thd_chunks(pp_enabled, cfg):
    if pp_enabled:
        return cfg.step_scheduler.local_batch_size // cfg.autopipeline.pp_microbatch_size
    return 1


def build_model_and_optimizer(
    device,
    cfg_model,
    cfg_opt,
    cfg_peft,
    model_wrapper,
    seed,
    checkpointer: Checkpointer,
    has_packed_sequence=False,
    tp_size=1,
    cp_size=1,
    cfg_fp8=None,
    cfg_compile=None,
    cfg_qat=None,
    cfg_quantization=None,
    autopipeline: AutoPipeline | None = None,
    loss_fn=None,
    parallelize_fn=None,
    load_base_model=True,
    unfreeze_modules: list[str] | None = None,
) -> tuple[nn.Module | AutoPipeline, list[str], list["Optimizer"], nn.Module, dict]:  # noqa: F821
    """
    Build and initialize a model and optimizer.

    Args:
        device: The target device.
        model_wrapper: Optional parallelism wrapper.
        cfg_model: Configuration for model instantiation.
        cfg_opt: Configuration for optimizer instantiation.
        use_hf_fa2: Whether to use HF's flash_attention_2. This takes precedence over Pytorch's sdpa_methods for attn.
        cfg_peft: Configuration for PEFT.
        model_wrapper: Optional parallelism wrapper.
        seed: Random seed.
        tp_size: Tensor parallel size.
        cp_size: Column parallel size.
        cfg_fp8: Configuration for FP8.
        cfg_compile: Configuration for torch.compile.
        unfreeze_modules: List of module names/substrings to unfreeze (e.g. ["classifier"]). Applied after PEFT freezing but before optimizer creation.

    Returns:
        The instantiated model on the specified device, the state dict keys before any parallelization, the optimizer, the loss function, and param_info dict.
    """
    is_meta_device = not isinstance(model_wrapper, (MegatronFSDPManager, DDPManager)) and not cfg_model.get(
        "force_hf", False
    )

    init_ctx = ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else nullcontext()
    with ScopedRNG(seed=seed, ranked=True):
        kwargs = {"tp_size": tp_size, "cp_size": cp_size, "has_packed_sequence": has_packed_sequence}

        if cfg_quantization is not None:
            logger.info("Model weight quantization enabled with BitsAndBytes")
            from nemo_automodel.components.quantization.qlora import create_bnb_config

            kwargs["quantization_config"] = create_bnb_config(cfg_quantization)

        # Instantiate the model in meta device to avoid OOM
        with init_ctx:
            model = cfg_model.instantiate(**kwargs)

            if checkpointer.config.dequantize_base_checkpoint is None:
                # try to infer whether the base weights are quantized
                try:
                    checkpointer.config.dequantize_base_checkpoint = hasattr(model.config, "quantization_config")
                except:
                    checkpointer.config.dequantize_base_checkpoint = False

            # Optionally apply PEFT (e.g., LoRA/DoRA, etc)
            if cfg_peft is not None:
                if tp_size > 1:
                    logger.info("Disabling Triton with TP ({})".format(tp_size))
                    cfg_peft.use_triton = False
                if autopipeline is not None:
                    logger.info("Enabling PEFT with Pipeline Parallelism")
                    logger.info("Disabling Triton with Pipeline Parallelism Enabled.")
                    cfg_peft.use_triton = False
                apply_lora_to_linear_modules(
                    model, cfg_peft, quantization_config=kwargs.get("quantization_config", None)
                )

        if cfg_fp8 is not None:
            fp8_config = build_fp8_config(cfg_fp8)
            model = apply_fp8_to_model(model, config=fp8_config)

        # Apply QAT if configured (torchao QAT)
        if cfg_qat is not None and cfg_qat.get("enabled", False):
            if cfg_peft is not None:
                raise ValueError("QAT with PEFT is not supported in 25.11")
            from nemo_automodel.components.quantization.qat import prepare_qat_model

            if any(map(lambda x: x.dtype != torch.bfloat16, model.parameters())):
                logger.warning("QAT is only supported for bfloat16 models. Support will be added in future release.")
                quit(code=0)
            quantizer = cfg_qat.quantizer.instantiate(precision=torch.bfloat16, scales_precision=torch.bfloat16)
            model, qat_mode = prepare_qat_model(model, quantizer)
            # Attach helpers for delayed fake-quant toggling if desired
            model._qat_mode = qat_mode  # type: ignore[attr-defined]

        # Explicitly unfreeze specified modules (e.g. task heads) that need full fine-tuning
        if unfreeze_modules:
            for name, param in model.named_parameters():
                if any(module_name in name for module_name in unfreeze_modules):
                    param.requires_grad_(True)
            logging.info(f"Unfroze parameters matching: {unfreeze_modules}")

    trainable_params, total_params = print_trainable_parameters(model)
    param_info = {
        "trainable_params": trainable_params,
        "total_params": total_params,
    }

    # hold a list copy of the model state dict keys before any parallelization
    state_dict_keys = list(model.state_dict().keys())

    if not _supports_logits_to_keep(model) and not isinstance(loss_fn, MaskedCrossEntropy):
        logger.warning("logits_to_keep not found in model.forward. Using MaskedCrossEntropy instead.")
        loss_fn = MaskedCrossEntropy()

    if autopipeline is not None:
        if get_world_size_safe() == 1:
            logger.info("World size is 1, skipping autopipeline.")
        else:
            autopipeline.build(model, loss_fn=loss_fn, parallelize_fn=parallelize_fn)
            for mp in autopipeline.parts:
                checkpointer.load_base_model(
                    mp,
                    device,
                    cfg_model.get("cache_dir", TRANSFORMERS_CACHE),
                    _get_model_name(cfg_model),
                    getattr(cfg_peft, "lora_A_init", None),
                    load_base_model=load_base_model,
                )

            # Create optimizer for all model parts
            trainable_params = []
            for i, model_part in enumerate(autopipeline.parts):
                trainable_params.append(
                    {
                        "params": list(filter(lambda x: x.requires_grad, model_part.parameters())),
                        "name": f"rank_{get_rank_safe()}_model_part_{i}",
                    }
                )
            model = autopipeline
    else:
        load_weights = False
        if parallelize_fn is not None and get_world_size_safe() > 1:
            parallelize_fn(
                model,
                world_mesh=model_wrapper.device_mesh,
                moe_mesh=getattr(model_wrapper, "moe_mesh", None),
                pp_enabled=False,
                dp_axis_names=(
                    ("dp_replicate", "dp_shard_cp")
                    if "dp_replicate" in model_wrapper.device_mesh.mesh_dim_names
                    and "dp_shard_cp" in model_wrapper.device_mesh.mesh_dim_names
                    else ("dp_shard_cp",)
                ),
                cp_axis_name="cp",
                tp_axis_name="tp",
                ep_axis_name="ep",
                ep_shard_axis_names=("ep_shard",),
            )
            load_weights = True
        elif callable(getattr(model_wrapper, "parallelize", None)):
            # FSDP2 and MegatronFSDP should already be on the correct device
            if isinstance(model_wrapper, MegatronFSDPManager):
                # MegatronFSDP instantiate optimizer inside parallelize_function
                trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
                assert len(trainable_params) > 0, "trainable_params cannot be empty"
                if tp_size > 1:
                    # TP does not support foreach
                    cfg_opt.foreach = False
                optimizer = cfg_opt.instantiate(params=trainable_params)

                model, optimizer = model_wrapper.parallelize(model, optimizer)

                return model, state_dict_keys, [optimizer], loss_fn, param_info

            else:
                load_weights = True
                model = model_wrapper.parallelize(model)

        # Load the weights into the model in parallel.
        if is_meta_device and load_weights:
            checkpointer.load_base_model(
                model,
                device,
                cfg_model.get("cache_dir", TRANSFORMERS_CACHE),
                _get_model_name(cfg_model),
                getattr(cfg_peft, "lora_A_init", None),
                load_base_model=load_base_model,
            )

        # ensure the model is on device
        model = model.to(device)

        # Apply torch.compile if configured
        if cfg_compile is not None:
            compile_config = build_compile_config(cfg_compile)
            model = compile_model(model, compile_config)

    if tp_size > 1:
        # TP does not support foreach
        cfg_opt.foreach = False

    if hasattr(model, "parts"):
        optimizer = []
        for part in model.parts:
            trainable_params = list(filter(lambda x: x.requires_grad, part.parameters()))
            assert len(trainable_params) > 0, "trainable_params cannot be empty"
            optimizer.append(cfg_opt.instantiate(params=trainable_params))
    else:
        trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
        assert len(trainable_params) > 0, "trainable_params cannot be empty"
        optimizer = [cfg_opt.instantiate(params=trainable_params)]

    return model, state_dict_keys, optimizer, loss_fn, param_info


def build_checkpoint_config(cfg_ckpt, cache_dir, model_repo_id, is_peft) -> CheckpointingConfig:
    """Build a checkpoint configuration.

    Args:
        cfg_ckpt: Configuration for checkpointing.
        cache_dir: Cache directory for the model.
        model_repo_id: Model repository ID.
        is_peft: Whether the model is PEFT.
        state_dict_keys: Copy of the model state dict keys before any parallelization.

    Returns:
        The instantiated checkpoint configuration.
    """

    ckpt_kwargs = dict(
        enabled=True,
        checkpoint_dir="checkpoints/",
        model_save_format="safetensors",
        model_repo_id=model_repo_id,
        model_cache_dir=cache_dir if cache_dir is not None else TRANSFORMERS_CACHE,
        save_consolidated=True,
        is_peft=is_peft,
    )
    if cfg_ckpt is not None:
        cfg_ckpt = cfg_ckpt.to_dict()
        cfg_ckpt.pop("restore_from", None)
        cfg_ckpt.pop("load_base_model", None)
        ckpt_kwargs |= cfg_ckpt
    if ckpt_kwargs.get("is_peft", False) and ckpt_kwargs.get("model_save_format") == "torch_save":
        raise ValueError(
            "PEFT checkpointing is not supported for torch_save format. Save using `safetensors` format instead."
        )
    checkpoint_config = CheckpointingConfig(**ckpt_kwargs)
    return checkpoint_config


def build_loss_fn(cfg_loss):
    """Build a loss function.

    Args:
        cfg_loss (ConfigNode): Loss function configuration.

    Returns:
        The instantiated loss function on the specified device.
    """
    return cfg_loss.instantiate()


def _build_tokenizer(cfg_model, cfg_ds):
    def compute_trust_remote_code():
        if hasattr(cfg_model, "trust_remote_code"):
            return getattr(cfg_model, "trust_remote_code")
        return resolve_trust_remote_code(_get_model_name(cfg_model))

    trust_remote_code = compute_trust_remote_code()
    # if tokenizer is not provided, use the model config to instantiate it
    if "tokenizer" not in cfg_ds and _get_model_name(cfg_model) is not None:
        logging.info("Using model config to instantiate tokenizer")
        tokenizer = NeMoAutoTokenizer.from_pretrained(_get_model_name(cfg_model), trust_remote_code=trust_remote_code)
    elif cfg_ds.get("tokenizer", None) is None:
        tokenizer = None
    elif "_target_" not in cfg_ds.tokenizer:
        tokenizer_dict = cfg_ds.tokenizer.to_dict()
        trust_remote_code = tokenizer_dict.pop("trust_remote_code", trust_remote_code)
        tokenizer = NeMoAutoTokenizer.from_pretrained(**tokenizer_dict, trust_remote_code=trust_remote_code)
    else:
        trust_remote_code = cfg_ds.tokenizer.to_dict().pop("trust_remote_code", trust_remote_code)
        tokenizer = cfg_ds.tokenizer.instantiate(trust_remote_code=trust_remote_code)

    # Finally, check if the dataset target accepts a tokenizer parameter
    kwargs = {}
    if tokenizer is not None and callable(cfg_ds._target_):
        try:
            sig = inspect.signature(cfg_ds._target_)
            if "tokenizer" in sig.parameters:
                kwargs["tokenizer"] = tokenizer
        except (ValueError, TypeError):
            # If we can't get the signature, skip adding tokenizer
            pass
    return kwargs, tokenizer


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
    supports_seq_lens=True,
    cp_size=1,
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
        supports_seq_lens: Whether the model supports seq_lens (Default: True).
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
            try:
                if ds.num_shards >= dp_world_size:
                    ds = ds.shard(dp_world_size, dp_rank)
                    logging.info(
                        f"Sharded IterableDataset via dataset.shard: world_size={dp_world_size}, rank={dp_rank}"
                    )
                else:
                    from datasets.distributed import split_dataset_by_node

                    ds.dataset = split_dataset_by_node(ds.dataset, world_size=dp_world_size, rank=dp_rank)
                    logging.info(f"Sharded dataset via split_dataset_by_node: world_size={dp_world_size}")
            except Exception as e:
                logging.warning(f"IterableDataset sharding skipped due to error: {e}")

        packed_sequence_size = getattr(cfg_ps, "packed_sequence_size", 0)
        # check if packed sequence is supported
        if packed_sequence_size > 0 and not supports_seq_lens:
            logging.warning("Packed sequence is not supported without seq_lens; disabling packed sequence")
            packed_sequence_size = 0

        # Apply packing if configured
        if packed_sequence_size > 0:
            logger.info(f"Packing dataset with size: {packed_sequence_size}")
            if hasattr(ds, "shuffle"):
                ds = ds.shuffle(seed)
            ds = pack_dataset(
                ds,
                split=cfg_ds.split,  # Assumes split is defined in dataset config
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

            hf_model_config = AutoConfig.from_pretrained(_get_model_name(cfg_model))

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


def build_distributed(cfg_dist: Dict[str, Any]) -> "DistInfo":  # noqa: F821
    """Build and initialize distributed training resources.

    Args:
        cfg_dist: Configuration for distributed training.

    Returns:
        Distributed training information from initialize_distributed.
    """
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)


def build_step_scheduler(cfg, dataloader, dp_group_size, local_batch_size):
    """Build the step scheduler.

    Args:
        cfg: configuration for the StepScheduler class.
        dataloader: the training dataloader, used for extracting the epoch_len (in batches).
        dp_group_size: the size of the data parallel group.
        micro_batch_size: the size of the micro batch.

    Returns:
        StepScheduler: the configured StepScheduler.
    """
    assert "_target_" not in cfg, "_target_ not permitted in step scheduler"
    default_kwargs = dict(
        num_epochs=10,
        global_batch_size=32,
        local_batch_size=local_batch_size,
        dp_size=dp_group_size,
        ckpt_every_steps=100,
        dataloader=dataloader,
    )
    if cfg is not None:
        default_kwargs |= cfg.to_dict()
    return StepScheduler(**default_kwargs)


def build_lr_scheduler(cfg, optimizer, step_scheduler) -> list[OptimizerParamScheduler] | None:  # noqa: F821
    """Build the learning rate scheduler.

    Args:
        cfg: Configuration for the OptimizerParamScheduler.
        optimizer: The optimizer to be scheduled.
        step_scheduler: The step scheduler to extract training parameters.

    Returns:
        OptimizerParamScheduler: The configured learning rate scheduler, or None if not configured.
    """
    if cfg is None:
        return None

    # Calculate total steps for the training run
    total_epochs = step_scheduler.num_epochs
    epoch_len = len(step_scheduler.dataloader)
    grad_acc_steps = step_scheduler.grad_acc_steps

    # Total optimizer steps (accounting for gradient accumulation)
    total_steps = (total_epochs * epoch_len) // grad_acc_steps
    if step_scheduler.max_steps is not None:
        total_steps = min(total_steps, step_scheduler.max_steps)

    # Set defaults for scheduler parameters
    optimizer_param_schedulers = []
    user_kwargs = cfg.to_dict()
    default_kwargs = dict(
        lr_warmup_steps=min(1000, total_steps // 10),  # 10% warmup or max 1000 steps
        lr_decay_steps=total_steps,
        lr_decay_style="cosine",
        wd_incr_steps=total_steps,
        wd_incr_style="constant",
    )

    if not isinstance(optimizer, list):
        optimizer = [optimizer]

    for opt in optimizer:
        base_lr = opt.param_groups[0]["lr"]
        default_kwargs.update(
            dict(
                optimizer=opt,
                init_lr=base_lr * 0.1,  # Start warmup at 10% of base LR
                max_lr=base_lr,
                min_lr=base_lr * 0.01,  # End at 1% of base LR
                start_wd=opt.param_groups[0].get("weight_decay", 0.0),
                end_wd=opt.param_groups[0].get("weight_decay", 0.0),
            )
        )
        default_kwargs.update(user_kwargs)
        optimizer_param_schedulers.append(OptimizerParamScheduler(**default_kwargs))

    logger.info(
        f"Building LR scheduler with total_steps={total_steps}, "
        f"warmup_steps={default_kwargs['lr_warmup_steps']}, "
        f"decay_style={default_kwargs['lr_decay_style']}"
    )

    return optimizer_param_schedulers


def build_wandb(cfg) -> wandb.Run:
    """Instantiates wandb and returns the instance. If no name is given, it will use the model name.

    Args:
        cfg: Configuration for wandb.

    Returns:
        The wandb instance.
    """
    assert cfg.get("wandb", None) is not None
    kwargs = cfg.wandb.to_dict()
    if kwargs.get("name", "") == "":
        kwargs["name"] = "_".join(_get_model_name(cfg.model).split("/")[-2:])
    run = wandb.init(
        **kwargs,
        config=cfg.to_dict(),
        settings=Settings(silent=True),
    )
    return run


def calculate_loss(loss_fn, **kwargs) -> torch.Tensor:
    """Calculate the loss.

    Args:
        loss_fn: Loss function.
        **kwargs: Keyword arguments for the loss function.

    Returns:
        The loss.
    """
    loss_fn_kwargs = {"num_label_tokens": kwargs.pop("num_label_tokens", None)}
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        model = kwargs.pop("model")
        labels = kwargs.pop("labels")

        # find the lm_head in the model
        lm_head = None
        if hasattr(model, "get_output_embeddings"):
            lm_head = model.get_output_embeddings().weight
        else:
            for n, p in model.named_parameters(remove_duplicate=False):
                if "lm_head" in n and n.endswith(".weight"):
                    lm_head = p
                    break
        if lm_head is None:
            raise ValueError("lm_head.weight not found in model")

        # unshard the possibly sharded lm_head
        lm_head = lm_head.full_tensor() if hasattr(lm_head, "full_tensor") else lm_head
        loss_fn_kwargs.update(
            {
                "hidden_states": kwargs.pop("hidden_states"),
                "labels": labels,
                "lm_weight": lm_head,
            }
        )
    else:
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": kwargs.pop("labels"),
            }
        )

    return loss_fn(**loss_fn_kwargs)


def build_validation_dataloader(cfg, dp_world_size, dp_rank, pp_enabled):
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
            cfg_ps=None,  # Use unpacked config for validation
            seed=cfg.get("seed", 42),
            local_batch_size=cfg.get("step_scheduler.local_batch_size", 1),
            global_batch_size=cfg.get("step_scheduler.global_batch_size", 1),
            max_steps=cfg.get("step_scheduler.max_steps", None),
            val_check_interval=cfg.get("step_scheduler.val_every_steps", None),
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            pp_enabled=False,
            supports_seq_lens=True,
            cp_size=cfg.get("distributed.cp_size", 1),
        )[0]

    return val_dataloaders


def parallelize_for_pp(
    model: nn.Module,
    *,
    world_mesh: DeviceMesh,
    moe_mesh: Optional[DeviceMesh] = None,
    pp_enabled: bool = False,
    dp_axis_names: Union[tuple[str, ...], str] = ("data_parallel",),
    cp_axis_name: Optional[str] = None,
    tp_axis_name: Optional[str] = None,
    ep_axis_name: Optional[str] = None,
    ep_shard_axis_names: Optional[tuple[str, ...]] = None,
    model_wrapper: Optional[Any] = None,
) -> nn.Module:
    if model_wrapper is not None:
        if callable(getattr(model_wrapper, "parallelize", None)):
            model = model_wrapper.parallelize(model)
    return model


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------


class TrainFinetuneRecipeForNextTokenPrediction(BaseRecipe):
    """Recipe for fine-tuning a model for next-token prediction.

    This class orchestrates training, from setup to main training loop.
    """

    def __init__(self, cfg):
        """Initialize the recipe with configuration.

        Args:
            cfg: Configuration dictionary/object for training.
        """
        self.cfg = cfg

    # ------------------ build phase ------------------
    def setup(self):
        """Builds all components needed for training/validation/logging/checkpointing/etc.

        This is the last place where self.cfg should be referenced.

        Raises:
            NotImplemented: Raises if it tries to restore a checkpoint; will be removed.
        """
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        # setups logging and adds the rankfilter to logging
        setup_logging()

        apply_cache_compatibility_patches()
        # Set up the stateful random number generator
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

        self.device_mesh = None
        self.moe_mesh = None
        self.model_wrapper = None
        if "distributed" in self.cfg:
            self.model_wrapper = self.cfg.distributed.instantiate(world_size=self.dist_env.world_size)
            self.device_mesh = getattr(self.model_wrapper, "device_mesh", None)
            self.moe_mesh = getattr(self.model_wrapper, "moe_mesh", None)

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info("🚀 View run at {}".format(run.url))

        self.mlflow_logger = None
        if self.dist_env.is_main and hasattr(self.cfg, "mlflow"):
            self.mlflow_logger = build_mlflow(self.cfg)
            self.mlflow_logger.log_params(self.cfg.to_dict())
            logging.info("MLflow experiment tracking enabled")

        # Log experiment details on main rank
        self._log_experiment_details()
        self._log_library_versions()

        self.pp_enabled: bool = (
            True if hasattr(self.model_wrapper, "pp_size") and self.model_wrapper.pp_size > 1 else False
        )
        autopipeline_cfg = self.cfg.get("autopipeline", None)
        if self.pp_enabled:
            pp_batch_size = self.cfg.step_scheduler.local_batch_size
            pp_microbatch_size = self.cfg.autopipeline.pp_microbatch_size
            assert pp_batch_size // self.cfg.autopipeline.pp_microbatch_size >= self.model_wrapper.pp_size, (
                f"pp_batch_size {pp_batch_size} // pp_microbatch_size {self.cfg.autopipeline.pp_microbatch_size} must be greater than or equal to pp_size {self.model_wrapper.pp_size}"
            )
            if (
                self.cfg.distributed.get("cp_size", 1) > 1
                and _uses_te_dot_product_attention(self.cfg.model)
                and _uses_thd_collater(self.cfg.dataloader)
            ):
                pp_microbatch_size = 1
                pp_batch_size = pp_batch_size // self.cfg.autopipeline.pp_microbatch_size
                logging.info(
                    f"Overriding pp_batch_size: {pp_batch_size}, pp_microbatch_size: {pp_microbatch_size} for THD"
                )

            assert autopipeline_cfg is not None, (
                "AutoPipeline configuration is required when pipeline parallelism is enabled"
            )
            assert not isinstance(self.model_wrapper, MegatronFSDPManager), (
                "MegatronFSDPManager is not supported when pipeline parallelism is enabled"
            )
            # Create AutoPipeline from config
            autopipeline = autopipeline_cfg.instantiate(
                world_mesh=self.device_mesh,
                moe_mesh=self.moe_mesh,
                pp_axis_name="pp",
                dp_axis_names=(
                    ("dp_replicate", "dp_shard_cp")
                    if "dp_replicate" in self.device_mesh.mesh_dim_names
                    and "dp_shard_cp" in self.device_mesh.mesh_dim_names
                    else ("dp_shard_cp",)
                ),
                cp_axis_name="cp" if "cp" in self.device_mesh.mesh_dim_names else None,
                tp_axis_name="tp" if "tp" in self.device_mesh.mesh_dim_names else None,
                ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
                ep_shard_axis_names=(
                    ("ep_shard",) if self.moe_mesh is not None and "ep_shard" in self.moe_mesh.mesh_dim_names else None
                ),
                pp_batch_size=pp_batch_size,
                pp_microbatch_size=pp_microbatch_size,
                patch_stage_backward_maybe_with_nosync=self.cfg.get("model.backend.enable_fsdp_optimizations", False),
                device=torch.cuda.current_device(),
            )
            assert isinstance(autopipeline, AutoPipeline), (
                f"autopipeline {autopipeline.__class__} is not an instance of AutoPipeline"
            )
        else:
            autopipeline = None

        # Build components
        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()
        self.loss_fn = build_loss_fn(self.cfg.loss_fn)
        parallelize_fn = getattr(self.cfg.get("parallelizer", None), "instantiate", None)
        if parallelize_fn is None and self.pp_enabled:
            parallelize_fn = partial(parallelize_for_pp, model_wrapper=self.model_wrapper)

        # Build checkpoint config
        checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            _get_model_name(self.cfg.model),
            True if self.cfg.get("peft", None) else False,
        )

        # Create Checkpointer instance
        self.checkpointer = Checkpointer(
            config=checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
        )

        model, model_state_dict_keys, self.optimizer, self.loss_fn, self.param_info = build_model_and_optimizer(
            self.dist_env.device,
            self.cfg.model,
            self.cfg.optimizer,
            self.peft_config,
            self.model_wrapper,
            has_packed_sequence=self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0,
            seed=self.cfg.get("seed", 42),
            tp_size=self.cfg.get("distributed.tp_size", 1),
            cp_size=self.cfg.get("distributed.cp_size", 1),
            cfg_fp8=self.cfg.get("fp8", None),
            cfg_compile=self.cfg.get("compile", None),
            cfg_quantization=self.cfg.get("quantization", None),
            cfg_qat=self.cfg.get("qat", None),
            autopipeline=autopipeline,
            loss_fn=self.loss_fn,
            parallelize_fn=parallelize_fn,
            load_base_model=self.cfg.get("checkpoint.load_base_model", True),
            checkpointer=self.checkpointer,
        )
        self.checkpointer.config.model_state_dict_keys = model_state_dict_keys

        if isinstance(model, AutoPipeline):
            self.model_parts = model.parts
            self.pp = model
        else:
            self.model_parts = [model]
            self.pp = None

        self.dataloader, self.tokenizer = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            self.cfg.model,
            self.cfg.get("packed_sequence", None),
            seed=self.cfg.get("seed", 42),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            global_batch_size=self.cfg.get("step_scheduler.global_batch_size", 1),
            max_steps=self.cfg.get("step_scheduler.max_steps", None),
            val_check_interval=self.cfg.get("step_scheduler.val_every_steps", None),
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
            pp_enabled=self.pp_enabled,
            supports_seq_lens=True,
            cp_size=self.cfg.get("distributed.cp_size", 1),
        )
        self.val_dataloaders = build_validation_dataloader(
            self.cfg,
            self._get_dp_group_size(),
            self._get_dp_rank(),
            self.pp_enabled,
        )
        self.best_metric_key = self.cfg.get("checkpoint.best_metric_key", "default")
        # Scheduler
        self.step_scheduler = build_step_scheduler(
            self.cfg.get("step_scheduler", None),
            self.dataloader,
            self._get_dp_group_size(),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )

        # Build learning rate scheduler
        self.lr_scheduler = build_lr_scheduler(self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler)

        # Log model, parameter counts, norms, optimizer and scheduler
        self._log_model_and_optimizer_details(self.model_parts, self.optimizer, self.lr_scheduler)

        # Handle delayed fake-quant toggling for QAT if configured
        self._qat_disable_fn, self._qat_enable_fn, self._qat_enable_after = self._setup_qat(self.cfg, self.model_parts)

        restore_from = self.cfg.get("checkpoint.restore_from", None)
        # Initialize JSONL loggers
        self.metric_logger_train = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = {
            name: build_metric_logger(
                pathlib.Path(self.checkpointer.config.checkpoint_dir)
                / (f"validation_{name}.jsonl" if name != "default" else "validation.jsonl")
            )
            for name in self.val_dataloaders.keys()
        }

        # Optionally resume
        self.load_checkpoint(restore_from)

        # Log step scheduler details
        self._log_step_scheduler_details(self.step_scheduler)

    def _setup_qat(self, cfg, model_parts: list[nn.Module]):
        if not cfg.get("qat.enabled", False):
            return None, None, None
        from nemo_automodel.components.quantization.qat import (
            get_disable_fake_quant_fn,
            get_enable_fake_quant_fn,
        )

        qat_cfg = cfg.qat
        _qat_enable_after = qat_cfg.get("fake_quant_after_n_steps", 0)
        # Collect mode from any model part that has it
        qat_mode = None
        if hasattr(model_parts[0], "_qat_mode"):
            qat_mode = getattr(model_parts[0], "_qat_mode")

        if qat_mode is None:
            return None, None, None

        _qat_disable_fn = get_disable_fake_quant_fn(qat_mode)
        _qat_enable_fn = get_enable_fake_quant_fn(qat_mode)
        if _qat_disable_fn is not None and _qat_enable_after is not None:
            try:
                # start with fake-quant disabled, will enable later
                for part in model_parts:
                    _qat_disable_fn(part)
                logger.info("QAT fake-quant disabled initially; will enable after %s steps", _qat_enable_after)
            except Exception as e:
                logger.warning("Failed to disable fake-quant at setup: %s", e)
        return _qat_disable_fn, _qat_enable_fn, _qat_enable_after

    def _enable_qat_if_delayed(self, step: int):
        if getattr(self, "_qat_enable_after", None) is None:
            return
        if step < self._qat_enable_after or self._qat_enable_fn is None:
            return
        try:
            for mp in self.model_parts:
                self._qat_enable_fn(mp)
            logger.info("Enabled QAT fake-quant after step %s", step)
            # Enable one
            self._qat_enable_after = None
        except Exception as e:
            logger.warning("Failed to enable fake-quant: %s", e)

    # ------------------ main loop ------------------
    def run_train_validation_loop(self):
        """Run the training loop over all epochs and batches.

        For each batch, perform a forward pass, compute loss, backpropagate,
        and update model parameters when necessary. Also prints loss every gradient step.
        """
        for mp in self.model_parts:
            mp.train()
        self.timestamp = time.perf_counter()

        for epoch in self.step_scheduler.epochs:
            self.step_scheduler.set_epoch(epoch)
            # The step scheduler yields a list of batches with the following properties:
            # 1. len(batches) == grad_acc_steps
            # 2. len(batches[0]) == batch_size
            for batches in self.step_scheduler:
                # If QAT delayed fake-quant is configured, enable after threshold
                self._enable_qat_if_delayed(self.step_scheduler.step)
                train_log_data = self._run_train_optim_step(batches, 1.0)
                # log
                self.log_train_metrics(train_log_data)

                # Run validation every val_every_steps
                val_losses = {}
                if self.step_scheduler.is_val_step:
                    if self.pp_enabled:
                        logger.warning("Validation is not supported for pipeline parallelism")
                    else:
                        for val_name, val_dataloader in self.val_dataloaders.items():
                            val_log_data = self._run_validation_epoch(val_dataloader)
                            val_losses[val_name] = val_log_data.metrics["val_loss"]
                            self.log_val_metrics(val_name, val_log_data, self.metric_logger_valid[val_name])
                    for mp in self.model_parts:
                        mp.train()

                # Save the checkpoint every ckpt_every_steps
                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(
                        epoch,
                        self.step_scheduler.step,
                        train_log_data.metrics["loss"],
                        val_losses,
                        best_metric_key=self.best_metric_key,
                    )
        # Close JSONL loggers after training loop completes
        self.metric_logger_train.close()
        for v in self.metric_logger_valid.values():
            v.close()

        self.checkpointer.close()

    # ------------------ helpers ------------------
    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        loss_buffer,
        num_label_tokens,
        num_batches,
        is_train: bool = True,
    ):
        # Move batch to device (handle both tensors and dicts of tensors like causal_mask_mapping)
        batch = {
            k: (
                {dk: dv.to(self.dist_env.device, non_blocking=True) if dv is not None else None for dk, dv in v.items()}
                if isinstance(v, dict)
                else (v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            )
            for k, v in batch.items()
        }
        train_ctx, batch = make_cp_batch_and_ctx(
            self.device_mesh,
            batch,
            use_te=_uses_te_dot_product_attention(self.cfg.model) and _uses_thd_collater(self.cfg.dataloader),
            padding_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
            num_chunks=_get_num_thd_chunks(self.pp_enabled, self.cfg),
        )
        labels = batch.pop("labels")

        if self.pp_enabled:
            if not is_train:
                logging.info("Skipping forward pass for validation because pipeline parallelism is enabled")
                return

            with train_ctx():
                losses = [] if self.pp.info.has_last_stage else None
                if self.pp.info.has_last_stage:
                    masked_labels = labels.clone()
                    targets = masked_labels
                else:
                    targets = None

                input_ids = batch.pop("input_ids")
                if self.pp.info.has_first_stage:
                    self.pp.info.schedule.step(input_ids, target=targets, losses=losses, **batch)
                else:
                    self.pp.info.schedule.step(target=targets, losses=losses, **batch)

            if self.pp.info.has_last_stage:
                local_loss = torch.sum(torch.stack(losses))
            else:
                local_loss = torch.tensor(0.0, device=self.dist_env.device)

            loss_buffer.append(local_loss.clone().detach())
        else:
            model = self.model_parts[0]
            sync_ctx = get_sync_ctx(model, idx == num_batches - 1) if is_train else nullcontext()
            with train_ctx(), sync_ctx:
                if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                    # use num_logits_to_keep to avoid full logits matrix in memory
                    out = model(logits_to_keep=1, **batch)
                    if "hidden_states" not in out:
                        raise ValueError(
                            "FusedLinearCrossEntropy requires the model to output hidden states. Set `model.output_hidden_states=True` in the config."
                        )
                else:
                    out = model(**batch)

                local_loss = calculate_loss(
                    self.loss_fn,
                    logits=getattr(out, "logits", out),
                    labels=labels,
                    model=model,
                    hidden_states=out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else None,
                    num_label_tokens=num_label_tokens,
                )
                loss_buffer.append(local_loss.clone().detach())
                if is_train:
                    (local_loss * self._get_dp_group_size(include_cp=True)).backward()

    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        """Execute a single training step.

        Args:
            batches: List of batches of training data.
            max_grad_norm: Gradient clipping norm. Optional, if None will not clip gradients.
        """

        num_label_tokens = torch.tensor(
            sum((batch["labels"] != -100).sum().item() for batch in batches), dtype=torch.long
        )
        num_label_tokens = self._dp_allreduce(num_label_tokens).item()
        loss_buffer = []

        # number of tokens in the batch, excluding any tail padding.
        num_tokens_in_batch = torch.tensor(
            sum(batch["labels"].numel() - count_tail_padding(batch["labels"]) for batch in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()

        num_batches = len(batches)
        prepare_for_grad_accumulation(self.model_parts, pp_enabled=self.pp_enabled)

        for i, batch in enumerate(batches):
            if i == num_batches - 1:
                prepare_for_final_backward(self.model_parts, pp_enabled=self.pp_enabled)

            self._forward_backward_step(
                i, batch, loss_buffer=loss_buffer, num_label_tokens=num_label_tokens, num_batches=num_batches
            )

        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm,
            self.model_parts,
            norm_type=2.0,
            pp_enabled=self.pp_enabled,
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
            pp_axis_name="pp" if self.pp_enabled else None,
            foreach=True,
            num_label_tokens=num_label_tokens,
            dp_group_size=self._get_dp_group_size(include_cp=True),
        )

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model_parts[0].finish_grad_sync()

        self.checkpointer.maybe_wait_for_staging()
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()

        if hasattr(self.model_parts[0], "update_moe_gate_bias"):
            for mp in self.model_parts:
                mp.update_moe_gate_bias()

        if self.lr_scheduler is not None:
            for scheduler in self.lr_scheduler:
                scheduler.step(1)

        # Precompute FP8 scales
        fp8_config = self.cfg.get("fp8", None)
        if (
            fp8_config is not None
            and fp8_config.get("enabled", False)
            and fp8_config.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and not self.pp_enabled
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model_parts[0])

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model_parts[0].install_optimized_model_weights()
        # self.model_parts[0].zero_grad_buffer()

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta
        reporting_loss = torch.sum(torch.stack(loss_buffer))
        reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
        if self.pp_enabled:
            reporting_loss = reporting_loss / num_label_tokens
            reporting_loss = reporting_loss.to(self.dist_env.device)
            # Send loss to first rank if pp group rank is 0
            src_rank = self.device_mesh.mesh.reshape(-1)[-1].item()
            if self.dist_env.rank == src_rank:
                torch.distributed.send(reporting_loss, dst=0)
            elif self.dist_env.is_main:
                torch.distributed.recv(reporting_loss, src=src_rank)

        reporting_loss = reporting_loss.cpu().item()
        # fix reporting_loss, tps across ranks

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": reporting_loss,
                "grad_norm": grad_norm,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / self._get_cp_group_size() / max(self._get_dp_group_size(), 1),
                "num_tokens_per_step": num_tokens_in_batch,
                "num_label_tokens": num_label_tokens,
            },
        )

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one pass over a single validation dataloader.

        Args:
            val_name: Name of the validation dataset.
            val_dataloader: DataLoader for the validation dataset.
        """
        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()

            total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.dist_env.device)
            total_num_label_tokens = 0

            for batch in val_dataloader:
                loss_buffer = []
                num_label_tokens = (batch["labels"] != -100).sum().item()
                self._forward_backward_step(
                    0,
                    batch,
                    loss_buffer=loss_buffer,
                    num_label_tokens=None,  # we will normalize outside.
                    num_batches=1,
                    is_train=False,
                )

                total_loss += torch.sum(torch.stack(loss_buffer)).item()
                total_num_label_tokens += num_label_tokens

        total_loss = self._dp_allreduce(total_loss, include_cp=True).item()
        total_num_label_tokens = self._dp_allreduce(torch.tensor(total_num_label_tokens, dtype=torch.long)).item()
        val_loss = total_loss / max(total_num_label_tokens, 1e-8)

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": val_loss,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "num_label_tokens": total_num_label_tokens,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )

    def log_val_metrics(self, val_name, log_data, metric_logger=None):
        """Log metrics to wandb, MLflow and other loggers
        Args:
            log_data: MetricsSample object, containing:
                step: int, the current step.
                epoch: int, the current epoch.
                metrics: Dict[str, float], containing:
                    "val_loss": Validation loss.
                    "lr": Learning rate.
                    "num_label_tokens": Number of label tokens.
                    "mem": Memory allocated.
        """

        # Pipeline parallelism does not support validation -> log_data is None
        if not self.dist_env.is_main or log_data is None:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict() | {"val_name": val_name}, step=log_data.step)

        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics(log_data.to_dict(), step=log_data.step)

        # JSONL validation log
        if not metric_logger is None:
            metric_logger.log(log_data)

        logging.info(
            '[val] name "{}" | step {} | epoch {} | loss {:.4f} | lr {:.2e} | num_label_tokens {}'.format(
                val_name,
                log_data.step,
                log_data.epoch,
                log_data.metrics["val_loss"],
                log_data.metrics["lr"],
                log_data.metrics["num_label_tokens"],
            )
        )

    def log_train_metrics(self, log_data):
        """Log metrics to wandb and other loggers.

        Args:
            log_data: MetricsSample object, containing:
                step: int, the current step.
                epoch: int, the current epoch.
                metrics: Dict[str, float], containing:
                    "loss": Training loss.
                    "grad_norm": Grad norm from the training step.
                    "lr": Learning rate.
                    "mem": Memory allocated.
                    "tps": Tokens per second.
                    "tps_per_gpu": Tokens per second per GPU.
                    "num_label_tokens": Number of label tokens.
        """
        if not self.dist_env.is_main:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=self.step_scheduler.step)

        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics(log_data.to_dict(), step=log_data.step)

        # JSONL training log
        self.metric_logger_train.log(log_data)
        logging.info(
            "step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | num_label_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["loss"],
                log_data.metrics["grad_norm"],
                log_data.metrics["lr"],
                log_data.metrics["mem"],
                log_data.metrics["tps"],
                log_data.metrics["tps_per_gpu"],
                log_data.metrics["num_label_tokens"],
            )
        )
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config_path=None):
    """Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the trainer, and initiates the training loop.
    """
    if config_path is None:
        config_path = pathlib.Path(__file__).parent.resolve() / "llama_3_2_1b_hellaswag.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
