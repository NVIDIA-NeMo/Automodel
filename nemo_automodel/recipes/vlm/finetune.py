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
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from transformers import AutoProcessor
from transformers.integrations.accelerate import init_empty_weights
from transformers.modeling_utils import no_init_weights
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TRANSFORMERS_CACHE, ContextManagers
from wandb import Settings

from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig, load_model_from_base_checkpoint
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.vlm.collate_fns import COLLATE_FNS
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.nvfsdp import NVFSDPManager
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.quantization.fp8 import apply_fp8_to_model, build_fp8_config
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.components.training.utils import count_tail_padding
from nemo_automodel.components.utils.compile_utils import (
    build_compile_config,
    compile_model,
)
from nemo_automodel.components.utils.dist_utils import get_sync_ctx
from nemo_automodel.components.utils.model_utils import apply_parameter_freezing, print_trainable_parameters
from nemo_automodel.recipes.base_recipe import BaseRecipe

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from nemo_automodel.components.distributed.init_utils import DistInfo

logger = logging.getLogger(__name__)

# ---------------------------
#  Stateless helper functions
# ---------------------------


def _freeze_model(model: nn.Module, cfg_freeze: Optional[Dict[str, Any]] = None, freeze_embeddings: bool = True):
    """
    Freeze the model.

    Args:
        model: The model to freeze.
        cfg_freeze: The configuration for freezing the model.
        freeze_embeddings: Whether to freeze embeddings.

    Returns:
        nn.Module: The frozen model.
    """
    if cfg_freeze is not None:
        apply_parameter_freezing(model, cfg_freeze)
    elif freeze_embeddings:
        logging.info("Freezing embeddings")
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.weight.requires_grad = False
    return model


def _build_optimizer(model: nn.Module, cfg_opt: Dict[str, Any], tp_size: int):
    """
    Build the optimizer.

    Args:
        model: The model to build the optimizer for.
        cfg_opt: The configuration for the optimizer.
        tp_size: The tensor parallel size.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
    assert len(trainable_params) > 0, "trainable_params cannot be empty"
    if tp_size > 1 and cfg_opt.get("foreach", False):
        cfg_opt.foreach = False
    return cfg_opt.instantiate(params=trainable_params)


def build_model_and_optimizer(
    device,
    cfg_model,
    cfg_opt,
    cfg_freeze,
    cfg_peft,
    model_wrapper,
    seed,
    tp_size=1,
    freeze_embeddings=True,
    cfg_fp8=None,
    cfg_compile=None,
) -> tuple[nn.Module, "Optimizer"]:  # noqa: F821
    """
    Build and initialize a model for VLM.

    Args:
        device: The target device.
        cfg_model: Configuration for model instantiation.
        cfg_opt: Configuration for optimizer instantiation.
        cfg_freeze: Configuration for freezing parameters.
        cfg_peft: Configuration for PEFT.
        model_wrapper: Optional parallelism wrapper.
        seed: Random seed.
        tp_size: Tensor parallel size.
        freeze_embeddings: Whether to freeze embeddings.
        cfg_fp8: Configuration for FP8.
        cfg_compile: Configuration for torch.compile.

    Returns:
        The instantiated model on the specified device and optimizer.
    """
    is_meta_device = False
    init_ctx = nullcontext()
    if hasattr(cfg_model, "is_meta_device"):
        is_meta_device = cfg_model.is_meta_device
        if is_meta_device and isinstance(model_wrapper, NVFSDPManager):
            raise ValueError("Meta device initialization is not supported with NVFSDPManager")
        init_ctx = ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else init_ctx
        del cfg_model.is_meta_device

    with StatefulRNG(seed=seed, ranked=True):
        kwargs = {}

        # Instantiate the model in meta device to avoid OOM
        with init_ctx:
            model = cfg_model.instantiate(**kwargs)
            model = _freeze_model(model, cfg_freeze, freeze_embeddings)
            # Optionally apply PEFT (e.g., LoRA/DoRA, etc)
            if cfg_peft is not None:
                apply_lora_to_linear_modules(model, cfg_peft)

            if cfg_fp8 is not None:
                fp8_config = build_fp8_config(cfg_fp8)
                model = apply_fp8_to_model(model, config=fp8_config)

        print_trainable_parameters(model)

        if callable(getattr(model_wrapper, "parallelize", None)):
            if isinstance(model_wrapper, NVFSDPManager):
                trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
                assert len(trainable_params) > 0, "trainable_params cannot be empty"
                if tp_size > 1:
                    cfg_opt.foreach = False
                optimizer = cfg_opt.instantiate(params=trainable_params)
                model, optimizer = model_wrapper.parallelize(model, optimizer)
                return model, optimizer
            else:
                model = model_wrapper.parallelize(model)

                # Load the weights into the model in parallel.
                if is_meta_device:
                    load_model_from_base_checkpoint(
                        model,
                        device,
                        cfg_peft is not None,
                        cfg_model.get("cache_dir", TRANSFORMERS_CACHE),
                        cfg_model.pretrained_model_name_or_path,
                        getattr(cfg_peft, "lora_A_init", None),
                    )
        else:
            model = model.to(device)

        optimizer = _build_optimizer(model, cfg_opt, tp_size)

        # Apply torch.compile if configured
        if cfg_compile is not None:
            compile_config = build_compile_config(cfg_compile)
            model = compile_model(model, compile_config)

        return model, optimizer


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
        enabled=False,
        checkpoint_dir="checkpoints/",
        model_save_format="safetensors",
        model_repo_id=model_repo_id,
        model_cache_dir=cache_dir if cache_dir is not None else TRANSFORMERS_CACHE,
        save_consolidated=False,
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


def build_dataloader(cfg_ds, cfg_dl, cfg_model, cfg_processor, device_mesh, seed) -> tuple[DataLoader, ProcessorMixin]:
    """Build a DataLoader for the VLM dataset.

    Args:
        cfg_ds: Dataset configuration.
        cfg_dl: DataLoader configuration.
        cfg_model: Model configuration.
        cfg_processor: Processor configuration or None.
        device_mesh: Device mesh for distributed training.
        seed: Random seed.

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

    with StatefulRNG(seed=seed, ranked=True):
        processor = None
        processor_kwargs = {}
        if cfg_processor is not None and hasattr(cfg_processor, "instantiate"):
            processor = cfg_processor.instantiate()
        elif cfg_processor is not None:
            processor_kwargs = cfg_processor.to_dict()

        # If no processor was instantiated, try AutoProcessor
        if processor is None:
            try:
                processor = AutoProcessor.from_pretrained(cfg_model.pretrained_model_name_or_path, **processor_kwargs)
            except Exception as e:
                # Some models do not provide an AutoProcessor
                processor = None
                logging.warning(f"AutoProcessor not available for {cfg_model.pretrained_model_name_or_path} ({e}). ")

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

        return cfg_dl.instantiate(dataset=ds, sampler=sampler, collate_fn=collate_fn), processor

        # Ensure spawn start method to avoid fork-safety issues with CUDA/JIT
        try:
            import torch.multiprocessing as mp

            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        return cfg_dl.instantiate(dataset=ds, sampler=sampler, collate_fn=collate_fn), processor


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


def build_step_scheduler(cfg, dataloader):
    """Build the step scheduler.

    Args:
        cfg: configuration for the StepScheduler class.
        dataloader: the training dataloader, used for extracting the epoch_len (in batches).

    Returns:
        StepScheduler: the configured StepScheduler.
    """
    assert "_target_" not in cfg, "_target_ not permitted in step scheduler"
    default_kwargs = dict(
        num_epochs=10,
        grad_acc_steps=10,
        ckpt_every_steps=100,
        dataloader=dataloader,
    )
    if cfg is not None:
        default_kwargs |= cfg.to_dict()
    return StepScheduler(**default_kwargs)


def build_lr_scheduler(cfg, optimizer, step_scheduler) -> OptimizerParamScheduler | None:  # noqa: F821
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

    # Extract learning rate from optimizer
    base_lr = optimizer.param_groups[0]["lr"]

    # Set defaults for scheduler parameters
    default_kwargs = dict(
        optimizer=optimizer,
        init_lr=base_lr * 0.1,  # Start warmup at 10% of base LR
        max_lr=base_lr,
        min_lr=base_lr * 0.01,  # End at 1% of base LR
        lr_warmup_steps=min(1000, total_steps // 10),  # 10% warmup or max 1000 steps
        lr_decay_steps=total_steps,
        lr_decay_style="cosine",
        start_wd=optimizer.param_groups[0].get("weight_decay", 0.0),
        end_wd=optimizer.param_groups[0].get("weight_decay", 0.0),
        wd_incr_steps=total_steps,
        wd_incr_style="constant",
    )

    # Override with user-provided config
    if cfg is not None:
        user_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        default_kwargs.update(user_cfg)

    logger.info(
        f"Building LR scheduler with total_steps={total_steps}, "
        f"warmup_steps={default_kwargs['lr_warmup_steps']}, "
        f"decay_style={default_kwargs['lr_decay_style']}"
    )

    return OptimizerParamScheduler(**default_kwargs)


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
        kwargs["name"] = "_".join(cfg.get("model.pretrained_model_name_or_path").split("/")[-2:])
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

        # Replace labels with -100 where mask is 0 (don't compute loss for these positions)
        # -100 is the default ignore index in PyTorch's cross entropy loss
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

        self.device_mesh = None
        self.model_wrapper = None
        if "distributed" in self.cfg:
            self.model_wrapper = self.cfg.distributed.instantiate(world_size=self.dist_env.world_size)
            self.device_mesh = getattr(self.model_wrapper, "device_mesh", None)

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info("🚀 View run at {}".format(run.url))

        # Log experiment details on main rank
        self._log_experiment_details()
        self._log_library_versions()

        # Build components with VLM-specific functions
        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()
        self.model, self.optimizer = build_model_and_optimizer(
            self.dist_env.device,
            self.cfg.model,
            self.cfg.optimizer,
            self.cfg.get("freeze_config", None),
            self.peft_config,
            self.model_wrapper,
            seed=self.cfg.get("seed", 42),
            tp_size=self.cfg.get("distributed.tp_size", 1),
            cfg_fp8=self.cfg.get("fp8", None),
            cfg_compile=self.cfg.get("compile", None),
        )
        self.loss_fn = build_loss_fn(self.cfg.loss_fn)
        self.dataloader, self.processor = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            self.cfg.model,
            self.cfg.get("processor", None),
            device_mesh=self.device_mesh,
            seed=self.cfg.get("seed", 42),
        )

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            self.val_dataloader, _ = build_dataloader(
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                self.cfg.model,
                self.cfg.get("processor", None),
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
            )

        # Initialize metrics required for calculating loss
        self.total_local_num_loss_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        self.forward_data_store = []

        # Scheduler
        self.step_scheduler = build_step_scheduler(self.cfg.get("step_scheduler", None), self.dataloader)

        # Build learning rate scheduler
        self.lr_scheduler = build_lr_scheduler(self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler)

        # Log model, parameter counts, norms, optimizer and scheduler
        self._log_model_and_optimizer_details(self.model, self.optimizer, self.lr_scheduler)

        # Build checkpointing config
        restore_from = self.cfg.get("checkpoint.restore_from", None)
        self.checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            self.cfg.model.pretrained_model_name_or_path,
            True if self.cfg.get("peft", None) else False,
        )

        # Set up the stateful random number generator
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

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
        self.model.train()
        self.timestamp = time.perf_counter()
        self.num_nonpad_tokens = 0
        for epoch in self.step_scheduler.epochs:
            self.step_scheduler.set_epoch(epoch)
            self.model.train()
            for batch_idx, batches in enumerate(self.step_scheduler):
                reporting_loss, grad_norm, tps, num_tokens_in_batch, num_label_tokens = self._run_train_optim_step(
                    batches, 1.0
                )
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(1)

                # log
                self.log_train_metrics(reporting_loss, grad_norm, num_tokens_in_batch, tps, num_label_tokens)

                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(epoch, self.step_scheduler.step)

                if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                    self._run_validation_epoch()
                    self.model.train()

    def _run_train_optim_step(self, batches, max_grad_norm=1.0):
        """Execute a single training step.

        Args:
            batches: List of batches of training data.
            max_grad_norm: Gradient clipping norm. Optional, if None will not clip gradients.
        """
        num_label_tokens = sum((batch["labels"] != -100).sum().item() for batch in batches)
        loss_buffer = []

        # number of tokens in the batch, excluding any tail padding.
        num_tokens_in_batch = sum(batch["labels"].numel() - count_tail_padding(batch["labels"]) for batch in batches)
        num_tokens_in_batch = self._dp_allreduce(torch.LongTensor([num_tokens_in_batch])).item()

        num_batches = len(batches)
        for i, batch in enumerate(batches):
            batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")

            if (
                "position_ids" not in batch
                and self.device_mesh is not None
                and (self.device_mesh["cp"].size() > 1 or self.device_mesh["tp"].size() > 1)
            ):
                batch["position_ids"] = torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).to(self.model.device)

            train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels)
            with train_ctx(), get_sync_ctx(self.model, i == num_batches - 1):
                if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                    # use num_logits_to_keep to avoid full logits matrix in memory
                    out = self.model(logits_to_keep=1, **batch)
                    if "hidden_states" not in out:
                        raise ValueError(
                            "FusedLinearCrossEntropy requires the model to output hidden states. Set `model.output_hidden_states=True` in the config."
                        )
                else:
                    out = self.model(**batch)
                local_loss = calculate_loss(
                    self.loss_fn,
                    logits=out.logits,
                    labels=labels,
                    model=self.model,
                    hidden_states=out.hidden_states[-1] if "hidden_states" in out else None,
                    num_label_tokens=num_label_tokens,
                )
                loss_buffer.append(local_loss.clone().detach())
                local_loss.backward()

        grad_norm = 0.0
        # Clip gradients **after** any rescaling.
        # TODO(@boxiangw): Fix TP gradient clipping
        if max_grad_norm is not None and (not self.device_mesh or self.device_mesh["tp"].size() == 1):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_grad_norm
            )
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()

        # Note(nvFSDP): Need to call these functions for nvFSDP if not using latest api
        # self.model.finish_grad_sync()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Precompute FP8 scales
        fp8_config = self.cfg.get("fp8", None)
        if (
            fp8_config is not None
            and fp8_config.get("enabled", False)
            and fp8_config.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(1)

        # Note(nvFSDP): Need to call these functions for nvFSDP if not using latest api
        # self.model.install_optimized_model_weights()
        # self.model.zero_grad_buffer()

        # TPS is calculated as follows (assuming grad-accumulation-steps=2):
        # fwd 0 | bwd 0 | fwd 1 | bwd 1 | opt 0 | fwd 2 | bwd 2 | ...
        # ^                                     ^
        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta
        reporting_loss = torch.sum(torch.stack(loss_buffer)).item()
        # fix reporting_loss, tps across ranks
        return reporting_loss, grad_norm, tps, num_tokens_in_batch, num_label_tokens

    @torch.no_grad()
    def _run_validation_epoch(self):
        """Run one pass over `self.val_dataloader`."""
        with StatefulRNG(seed=1, ranked=True):
            self.model.eval()

            total_loss = 0.0
            total_tokens = 0

            for batch in self.val_dataloader:
                batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
                labels = batch.pop("labels")
                num_label_tokens = (labels != -100).sum()

                if (
                    self.device_mesh
                    and "position_ids" not in batch
                    and (self.device_mesh["cp"].size() > 1 or self.device_mesh["tp"].size() > 1)
                ):
                    batch["position_ids"] = (
                        torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).to(self.model.device)
                    )

                train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels)
                with train_ctx():
                    if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                        out = self.model(logits_to_keep=1, **batch)
                    else:
                        out = self.model(**batch)
                    local_loss = calculate_loss(
                        self.loss_fn,
                        logits=out.logits,
                        labels=labels,
                        model=self.model,
                        hidden_states=out.hidden_states[-1] if "hidden_states" in out else None,
                        num_label_tokens=num_label_tokens,
                    )

                total_loss += local_loss.item()

        # Aggregate across ranks if distributed is initialized
        total_loss = self._dp_allreduce(torch.FloatTensor([total_loss])).item()
        total_tokens = self._dp_allreduce(torch.LongTensor([total_tokens])).item()

        val_loss = total_loss / max(total_tokens, 1e-8)
        if self.dist_env.is_main:
            if wandb.run is not None:
                wandb.log({"val_loss": val_loss, "step": self.step_scheduler.step, "epoch": self.step_scheduler.epoch})
        current_lr = self.optimizer.param_groups[0]["lr"]
        logging.info(
            "[val] step {} | epoch {} | loss {:.4f} | lr {:.2e}".format(
                self.step_scheduler.step, self.step_scheduler.epoch, val_loss, current_lr
            )
        )

    def log_train_metrics(self, train_loss, grad_norm, num_tokens_in_batch, tps, num_label_tokens) -> float:
        """Log metrics to wandb.

        Args:
            train_loss: Training loss.
            grad_norm: Grad norm from the training step.
            num_tokens_in_batch: Total number of loss tokens.
            tps: Tokens per second.
        """
        log_data = {
            "step": self.step_scheduler.step,
            "epoch": self.step_scheduler.epoch,
            "train_loss": train_loss,
            "grad_norm": grad_norm,
            "num_tokens_per_step": num_tokens_in_batch,
            "tps": tps,
        }
        current_lr = self.optimizer.param_groups[0]["lr"]
        log_data["learning_rate"] = current_lr

        if wandb.run is not None:
            wandb.log(log_data)

        logging.info(
            "step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | tps {:.2f} | num_label_tokens {}".format(
                self.step_scheduler.step,
                self.step_scheduler.epoch,
                train_loss,
                grad_norm,
                current_lr,
                torch.cuda.max_memory_allocated() / 1024**3,
                tps,
                num_label_tokens,
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
