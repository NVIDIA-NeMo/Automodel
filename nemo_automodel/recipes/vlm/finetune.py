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

import logging
import pathlib
import time
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.nn as nn
import wandb
from huggingface_hub import constants as hf_constants
from megatron_fsdp.fully_shard import fully_shard_optimizer
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from transformers import AutoProcessor

from transformers.processing_utils import ProcessorMixin
from wandb import Settings

from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.vlm.collate_fns import COLLATE_FNS
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.utils import FirstRankPerNode, get_sync_ctx
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.quantization.fp8 import build_fp8_config
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.components.training.utils import (
    count_tail_padding,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.compile_utils import build_compile_config
from nemo_automodel.components.utils.model_utils import (
    _supports_logits_to_keep,
    apply_parameter_freezing,
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


def _freeze_model(
    model: Union[nn.Module, AutoPipeline], cfg_freeze: Optional[Dict[str, Any]] = None, freeze_embeddings: bool = True
):
    """
    Freeze the model.

    Args:
        model: The model to freeze (can be nn.Module or AutoPipeline).
        cfg_freeze: The configuration for freezing the model.
        freeze_embeddings: Whether to freeze embeddings.

    Returns:
        nn.Module or AutoPipeline: The frozen model.
    """
    # Handle AutoPipeline by applying freezing to each part
    if isinstance(model, AutoPipeline):
        for part in model.parts:
            _freeze_model(part, cfg_freeze, freeze_embeddings)
        return model

    if cfg_freeze is not None:
        apply_parameter_freezing(model, cfg_freeze)
    elif freeze_embeddings:
        logging.info("Freezing embeddings")
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.weight.requires_grad = False
    return model


def build_model_and_optimizer(
    cfg_model,
    cfg_opt,
    cfg_freeze,
    cfg_peft,
    model_wrapper,
    seed,
    checkpointer: Checkpointer,
    tp_size=1,
    cp_size=1,
    freeze_embeddings=True,
    cfg_fp8=None,
    cfg_compile=None,
    loss_fn=None,
    parallelize_fn=None,
    autopipeline: AutoPipeline | None = None,
) -> tuple[nn.Module | AutoPipeline, list["Optimizer"], nn.Module]:  # noqa: F821
    """Build and initialize a model for VLM.

    Returns:
        The instantiated model, optimizer, and loss function.
    """
    with ScopedRNG(seed=seed, ranked=True):
        # Build infrastructure kwargs
        kwargs = {
            "tp_size": tp_size,
            "cp_size": cp_size,
            "parallelize_fn": parallelize_fn,
            "checkpointer": checkpointer,
            "peft_config": cfg_peft,
            "model_wrapper": model_wrapper,
            "loss_fn": loss_fn,
            "autopipeline": autopipeline,
        }
        if cfg_fp8 is not None:
            fp8_config = build_fp8_config(cfg_fp8)
            kwargs["fp8_config"] = fp8_config
        if cfg_compile is not None:
            kwargs["compile_config"] = build_compile_config(cfg_compile)

        # Check if using NeMoAutoModel
        is_nemo_auto_model = cfg_model.get("_target_", None) in (
            NeMoAutoModelForImageTextToText.from_config,
            NeMoAutoModelForImageTextToText.from_pretrained,
        )

        if is_nemo_auto_model:
            # NeMoAutoModel handles infrastructure internally
            model = cfg_model.instantiate(**kwargs)
        else:
            raise ValueError(
                f"VLM finetuning requires NeMoAutoModelForImageTextToText. "
                f"Got model target: {cfg_model.get('_target_', None)}"
            )

    # VLM-specific: Apply freezing after model is built
    # For PEFT, base model params are already frozen; this handles additional freezing (e.g., embeddings)
    model = _freeze_model(model, cfg_freeze, freeze_embeddings)

    # Check loss function compatibility
    if not _supports_logits_to_keep(model) and not isinstance(loss_fn, MaskedCrossEntropy):
        logger.warning("logits_to_keep not found in model.forward. Using MaskedCrossEntropy instead.")
        loss_fn = MaskedCrossEntropy()

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
        optimizer = cfg_opt.instantiate(params=trainable_params)
        if isinstance(model_wrapper, MegatronFSDPManager) and torch.distributed.get_world_size() > 1:
            fully_shard_optimizer(model, optimizer)
        optimizer = [optimizer]

    return model, optimizer, loss_fn


def build_checkpoint_config(cfg_ckpt, cache_dir, model_repo_id, is_peft) -> CheckpointingConfig:
    """Build a checkpoint configuration.

    Args:
        cfg_ckpt: Configuration for checkpointing.
        cache_dir: Cache directory for the model.
        model_repo_id: Model repository ID.
        is_peft: Whether the model is PEFT.

    Returns:
        The instantiated checkpoint configuration.
    """
    ckpt_kwargs = dict(
        enabled=True,
        checkpoint_dir="checkpoints/",
        model_save_format="safetensors",
        model_repo_id=model_repo_id,
        model_cache_dir=cache_dir if cache_dir is not None else hf_constants.HF_HUB_CACHE,
        save_consolidated=True,
        is_peft=is_peft,
    )
    if cfg_ckpt is not None:
        cfg_ckpt = cfg_ckpt.to_dict()
        cfg_ckpt.pop("restore_from", None)
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
        cfg_loss: Loss function configuration.

    Returns:
        The instantiated loss function.
    """
    return cfg_loss.instantiate()


def build_dataloader(
    cfg_ds, cfg_dl, pretrained_model_name_or_path, cfg_processor, device_mesh, seed, local_batch_size
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

    Returns:
        The instantiated DataLoader and processor.
    """
    dist_sampler_kwargs = {
        "shuffle": cfg_dl.get("shuffle", True),
    }
    if device_mesh is not None:
        dist_sampler_kwargs |= {
            "num_replicas": device_mesh["dp"].size(),
            "rank": device_mesh["dp"].get_local_rank(),
        }

    with ScopedRNG(seed=seed, ranked=True):
        processor = None
        processor_kwargs = {}
        if cfg_processor is not None and hasattr(cfg_processor, "instantiate"):
            processor = cfg_processor.instantiate()
        elif cfg_processor is not None:
            processor_kwargs = cfg_processor.to_dict()

        # If no processor was instantiated, try AutoProcessor
        if processor is None:
            try:
                processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **processor_kwargs)
            except Exception as e:
                # Some models do not provide an AutoProcessor
                processor = None
                logging.warning(f"AutoProcessor not available for {pretrained_model_name_or_path} ({e}). ")

        with FirstRankPerNode():
            ds = cfg_ds.instantiate(path_or_dataset=cfg_ds.path_or_dataset)

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
                processor_type = "default"
                logging.warning(f"You are using {processor_type} with default collate function.")
            collate_fn = lambda examples: COLLATE_FNS[processor_type](examples, processor)

        return cfg_dl.instantiate(
            dataset=ds, sampler=sampler, collate_fn=collate_fn, batch_size=local_batch_size
        ), processor

        # Ensure spawn start method to avoid fork-safety issues with CUDA/JIT
        try:
            import torch.multiprocessing as mp

            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        return cfg_dl.instantiate(
            dataset=ds, sampler=sampler, collate_fn=collate_fn, batch_size=local_batch_size
        ), processor


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


class FinetuneRecipeForVLM(BaseRecipe):
    """Recipe for fine-tuning a VLM model."""

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

        # Build components with VLM-specific functions
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

        if self.cfg.get("clip_grad_norm.max_norm", None) is not None:
            self.max_grad_norm = float(self.cfg.clip_grad_norm.max_norm)
        else:
            logging.info("No clip_grad_norm.max_norm specified in config, using default value of 1.0")
            self.max_grad_norm = 1.0

        # Create Checkpointer instance
        self.checkpointer = Checkpointer(
            config=checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
        )

        model, self.optimizer, self.loss_fn = build_model_and_optimizer(
            self.cfg.model,
            self.cfg.optimizer,
            self.cfg.get("freeze_config", None),
            self.peft_config,
            self.model_wrapper,
            seed=self.cfg.get("seed", 42),
            tp_size=self.cfg.get("distributed.tp_size", 1),
            cp_size=self.cfg.get("distributed.cp_size", 1),
            cfg_fp8=self.cfg.get("fp8", None),
            cfg_compile=self.cfg.get("compile", None),
            loss_fn=self.loss_fn,
            parallelize_fn=parallelize_fn,
            checkpointer=self.checkpointer,
            autopipeline=autopipeline,
        )

        if isinstance(model, AutoPipeline):
            self.model_parts = model.parts
            self.pp = model
        else:
            self.model_parts = [model]
            self.pp = None

        self.dataloader, self.processor = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            _get_model_name(self.cfg.model),
            self.cfg.get("processor", None),
            device_mesh=self.device_mesh,
            seed=self.cfg.get("seed", 42),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            self.val_dataloader, _ = build_dataloader(
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                _get_model_name(self.cfg.model),
                self.cfg.get("processor", None),
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
                local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
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

        restore_from = self.cfg.get("checkpoint.restore_from", None)

        # Initialize JSONL loggers
        self.metric_logger_train = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "validation.jsonl"
        )

        # Optionally resume
        self.load_checkpoint(restore_from)

        # Log step scheduler details
        self._log_step_scheduler_details(self.step_scheduler)

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
            for batch_idx, batches in enumerate(self.step_scheduler):
                log_data = self._run_train_optim_step(batches, self.max_grad_norm)
                # log
                self.log_train_metrics(log_data)

                val_loss = {}
                if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                    if self.pp_enabled:
                        logger.warning("Validation is not supported for pipeline parallelism")
                    else:
                        val_log_data = self._run_validation_epoch(self.val_dataloader)
                        val_loss["val_loss"] = val_log_data.metrics["val_loss"]
                        self.log_val_metrics(val_log_data)
                    for mp in self.model_parts:
                        mp.train()

                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(
                        epoch,
                        self.step_scheduler.step,
                        log_data.metrics["loss"],
                        val_loss,
                        best_metric_key=self.best_metric_key,
                    )

        # Close JSONL loggers after training loop completes
        self.metric_logger_train.close()
        self.metric_logger_valid.close()

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
        batch = {
            k: (
                {dk: dv.to(self.dist_env.device, non_blocking=True) if dv is not None else None for dk, dv in v.items()}
                if isinstance(v, dict)
                else (v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            )
            for k, v in batch.items()
        }

        train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch)
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

                # VLM: Custom chunking for pixel_values and image_grid_hws
                # These tensors have non-standard structure that can't be naively chunked by dim 0
                # TODO: @HuiyingLi properly handle pixel_values split
                pixel_values = batch.pop("pixel_values", None)
                image_grid_hws = batch.pop("image_grid_hws", None)

                if self.pp.info.has_first_stage and pixel_values is not None and image_grid_hws is not None:
                    stage0_model = self.model_parts[0]
                    n_microbatches = self.pp._info.schedule._n_microbatches
                    batch_size = input_ids.shape[0]
                    n_images = image_grid_hws.shape[0]

                    patch_counts = image_grid_hws[:, 0] * image_grid_hws[:, 1]
                    cumsum = torch.cumsum(patch_counts, dim=0)

                    pixel_values_chunks = []
                    image_grid_hws_chunks = []

                    if n_images == batch_size:
                        # 1 image per sample
                        images_per_mb = batch_size // n_microbatches
                        for mb_idx in range(n_microbatches):
                            img_start = mb_idx * images_per_mb
                            img_end = min(img_start + images_per_mb, n_images)

                            image_grid_hws_chunks.append(image_grid_hws[img_start:img_end])

                            patch_start = 0 if img_start == 0 else cumsum[img_start - 1].item()
                            patch_end = cumsum[img_end - 1].item() if img_end > 0 else 0
                            pixel_values_chunks.append(pixel_values[int(patch_start) : int(patch_end)])
                    else:
                        pixel_values_chunks.append(pixel_values)
                        image_grid_hws_chunks.append(image_grid_hws)
                        for _ in range(n_microbatches - 1):
                            pixel_values_chunks.append(pixel_values[:0])
                            image_grid_hws_chunks.append(image_grid_hws[:0])
                        logging.warning(
                            f"VLM chunking: n_images={n_images} != batch_size={batch_size}, giving all images to first microbatch"
                        )

                    # Store pre-chunked tensors on model for forward to use
                    stage0_model._vlm_pixel_values_chunks = pixel_values_chunks
                    stage0_model._vlm_image_grid_hws_chunks = image_grid_hws_chunks
                    stage0_model._vlm_chunk_idx = 0

                if self.pp.info.has_first_stage:
                    self.pp.info.schedule.step(input_ids, target=targets, losses=losses, **batch)
                else:
                    self.pp.info.schedule.step(target=targets, losses=losses, **batch)

                # Clear stored VLM chunks after PP step
                if self.pp.info.has_first_stage and pixel_values is not None:
                    stage0_model = self.model_parts[0]
                    stage0_model._vlm_pixel_values_chunks = None
                    stage0_model._vlm_image_grid_hws_chunks = None
                    stage0_model._vlm_chunk_idx = None

            if self.pp.info.has_last_stage:
                local_loss = torch.sum(torch.stack(losses))
            else:
                local_loss = torch.tensor(0.0, device=self.dist_env.device)

            loss_buffer.append(local_loss.clone().detach())
        else:
            model = self.model_parts[0]
            sync_ctx = (
                get_sync_ctx(
                    model,
                    idx == num_batches - 1,
                    defer_fsdp_grad_sync=getattr(self.model_wrapper, "defer_fsdp_grad_sync", True),
                )
                if is_train
                else nullcontext()
            )
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
            max_grad_norm=max_grad_norm,
            model_parts=self.model_parts,
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
        # self.model.finish_grad_sync()

        self.checkpointer.maybe_wait_for_staging()
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad(set_to_none=True)

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
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model_parts[0])

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model.install_optimized_model_weights()
        # self.model.zero_grad_buffer()

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta
        reporting_loss = torch.sum(torch.stack(loss_buffer))
        reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
        if self.pp_enabled:
            # PP uses sum reduction per microbatch (no internal normalization).
            # Divide by num_label_tokens to get the mean loss, same as non-PP.
            reporting_loss = reporting_loss / num_label_tokens
            reporting_loss = reporting_loss.float().to(self.dist_env.device)
            # Send loss to first rank from the last PP stage of rank0's mesh coords.
            # This avoids picking a global-rank sender from a different EP/PP group.
            if self.device_mesh is not None and "pp" in self.device_mesh.mesh_dim_names:
                dim_names = list(self.device_mesh.mesh_dim_names)
                mesh = self.device_mesh.mesh
                idx = []
                for name in dim_names:
                    if name == "pp":
                        idx.append(-1)
                    else:
                        idx.append(0)
                src_rank = mesh[tuple(idx)].item()
            else:
                src_rank = self.device_mesh.mesh.reshape(-1)[-1].item()
            if self.dist_env.rank == src_rank:
                torch.distributed.send(reporting_loss, dst=0)
            elif self.dist_env.is_main:
                torch.distributed.recv(reporting_loss, src=src_rank)

        reporting_loss = reporting_loss.item()
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
        """Run one pass over `self.val_dataloader`."""
        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()

            total_loss = 0.0
            total_tokens = 0
            total_num_label_tokens = 0
            for batch in val_dataloader:
                batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
                labels = batch.pop("labels")
                num_label_tokens = (labels != -100).sum().item()

                if (
                    self.device_mesh
                    and "position_ids" not in batch
                    and (self.device_mesh["cp"].size() > 1 or self.device_mesh["tp"].size() > 1)
                ):
                    batch["position_ids"] = (
                        torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).to(self.model_parts[0].device)
                    )

                train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels)
                with train_ctx():
                    if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                        out = self.model_parts[0](logits_to_keep=1, **batch)
                    else:
                        out = self.model_parts[0](**batch)
                    local_loss = calculate_loss(
                        self.loss_fn,
                        logits=getattr(out, "logits", out),
                        labels=labels,
                        model=self.model_parts[0],
                        hidden_states=out.hidden_states[-1]
                        if getattr(out, "hidden_states", None) is not None
                        else None,
                        num_label_tokens=num_label_tokens,
                    )
                    total_num_label_tokens += num_label_tokens

                total_loss += local_loss.item() * num_label_tokens
                total_tokens += num_label_tokens

        # Aggregate across ranks if distributed is initialized
        total_loss = self._dp_allreduce(torch.FloatTensor([total_loss]), include_cp=True).item()
        total_tokens = self._dp_allreduce(torch.LongTensor([total_tokens]), include_cp=True).item()
        total_num_label_tokens = self._dp_allreduce(torch.LongTensor([total_num_label_tokens])).item()

        val_loss = total_loss / max(total_tokens, 1e-8)

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

    def log_val_metrics(self, log_data):
        """Log metrics to wandb and other loggers
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

        if not self.dist_env.is_main or log_data is None:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=log_data.step)

        # JSONL validation log
        self.metric_logger_valid.log(log_data)

        logging.info(
            "[val] step {} | epoch {} | loss {:.4f} | lr {:.2e} | num_label_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["val_loss"],
                log_data.metrics["lr"],
                log_data.metrics["num_label_tokens"],
            )
        )

    def log_train_metrics(self, log_data) -> float:
        """Log metrics to wandb.

        Args:
            train_loss: Training loss.
            grad_norm: Grad norm from the training step.
            num_tokens_in_batch: Total number of loss tokens.
            tps: Tokens per second.
        """
        if not self.dist_env.is_main:
            return

        # Log to remote services (WandB) according to step_scheduler frequency
        if self.step_scheduler.is_remote_logging_step:
            if wandb.run is not None:
                wandb.log(log_data.to_dict(), step=self.step_scheduler.step)

        # JSONL training log (always log for detailed local records)
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
        config_path = pathlib.Path(__file__).parent.resolve() / "gemma3" / "gemma3_vl_4b_cord_v2.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
