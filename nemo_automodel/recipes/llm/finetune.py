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
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.distributed.device_mesh import _mesh_resources
from torch.utils.data import DataLoader
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from wandb import Settings

from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.packed_sequence import PackedSequence
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.nvfsdp import NVFSDPManager
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.components.training.utils import count_tail_padding
from nemo_automodel.components.utils.dist_utils import (
    clip_gradients,
    get_sync_ctx,
    reduce_loss,
    rescale_gradients,
)
from nemo_automodel.recipes.base_recipe import BaseRecipe

logger = logging.getLogger(__name__)

# ---------------------------
#  Stateless helper functions
# ---------------------------


def build_model_and_optimizer(
    device,
    cfg_model,
    cfg_opt,
    use_hf_fa2,
    cfg_peft,
    model_wrapper,
    seed,
    tp_size=1,
    freeze_embeddings=True,
    cfg_fp8=None,
) -> tuple[nn.Module, "Optimizer"]:  # noqa: F821
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
        freeze_embeddings: Whether to freeze embeddings.

    Returns:
        The instantiated model on the specified device and optimizer.
    """
    with StatefulRNG(seed=seed, ranked=True):
        kwargs = {}
        if use_hf_fa2:
            kwargs["attn_implementation"] = "flash_attention_2"
            logger.warning(
                "Packed sequence is supported only with Flash Attention. "
                "Setting model's attn_implementation to flash_attention_2"
            )
        # Add FP8 config if provided
        if cfg_fp8 is not None:
            kwargs["fp8_config"] = cfg_fp8.instantiate()

        model = cfg_model.instantiate(**kwargs)
        if freeze_embeddings:
            logging.info("Freezing embeddings")
            for m in model.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.requires_grad_(False)
        # Optionally apply PEFT (e.g., LoRA/DoRA, etc)
        if cfg_peft is not None:
            apply_lora_to_linear_modules(model, cfg_peft)

    if callable(getattr(model_wrapper, "parallelize", None)):
        # FSDP2 and nvFSDP should already be on the correct device
        if isinstance(model_wrapper, NVFSDPManager):
            # nvFSDP instantiate optimizer inside parallelize_function
            trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
            assert len(trainable_params) > 0, "trainable_params cannot be empty"
            if tp_size > 1:
                # TP does not support foreach
                cfg_opt.foreach = False
            optimizer = cfg_opt.instantiate(params=trainable_params)

            model, optimizer = model_wrapper.parallelize(model, optimizer)

            return model, optimizer

        else:
            model = model_wrapper.parallelize(model)
    else:
        model = model.to(device)

    trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
    assert len(trainable_params) > 0, "trainable_params cannot be empty"
    if tp_size > 1:
        # TP does not support foreach
        cfg_opt.foreach = False
    optimizer = cfg_opt.instantiate(params=trainable_params)

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
    from transformers.utils import TRANSFORMERS_CACHE

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
        The instantiated loss function on the specified device.
    """
    return cfg_loss.instantiate()


def build_dataloader(
    cfg_ds, cfg_dl, cfg_model, cfg_ps, device_mesh, seed
) -> tuple[DataLoader, PreTrainedTokenizerBase]:
    """Build a DataLoader for the dataset.

    Args:
        cfg_ds: Dataset configuration.
        cfg_dl: DataLoader configuration.
        cfg_model: Model configuration.
        cfg_ps: Packed sequence configuration.
        device_mesh: Device mesh.
        seed: Random seed.

    Returns:
        The instantiated DataLoader and tokenizer.
    """
    dist_sampler_kwargs = {
        "shuffle": cfg_dl.get("shuffle", True),
    }
    if device_mesh is not None:
        dist_sampler_kwargs |= {
            "num_replicas": device_mesh["dp"].size(),
            "rank": device_mesh["dp"].get_local_rank(),
        }
    if "tokenizer" not in cfg_ds:
        logging.info("Using model config to instantiate tokenizer")
        trust_remote_code = getattr(cfg_model, "trust_remote_code", False)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg_model.pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
    elif "_target_" not in cfg_ds.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(**cfg_ds.tokenizer.to_dict())
    else:
        tokenizer = cfg_ds.tokenizer.instantiate()

    with StatefulRNG(seed=seed, ranked=True):
        ds = cfg_ds.instantiate(tokenizer=tokenizer)
        # Apply packing if configured
        if getattr(cfg_ps, "packed_sequence_size", 0) > 0:
            logger.info(f"Packing dataset with size: {cfg_ps.packed_sequence_size}")
            ds = PackedSequence(
                ds,
                split=cfg_ds.split,  # Assumes split is defined in dataset config
                packed_sequence_size=cfg_ps.packed_sequence_size,
                split_across_pack=getattr(cfg_ps, "split_across_pack", False),
                max_packs=getattr(cfg_ps, "max_packs", None),
            ).pack()

        sampler = StatefulDistributedSampler(
            ds,
            seed=seed,
            drop_last=True,
            **dist_sampler_kwargs,
        )

        # Handle collate_fn instantiation if it's a ConfigNode
        dl_kwargs = {"dataset": ds, "sampler": sampler}
        if hasattr(cfg_dl, "collate_fn") and hasattr(cfg_dl.collate_fn, "_target_"):
            collate_cfg = cfg_dl.collate_fn
            dl_kwargs["collate_fn"] = lambda batch: collate_cfg.instantiate(batch=batch)

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
    loss_fn_kwargs = {}
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        model = kwargs.pop("model")

        # Replace labels with -100 where mask is 0 (don't compute loss for these positions)
        # -100 is the default ignore index in PyTorch's cross entropy loss
        labels = kwargs.pop("labels")
        if "mask" in kwargs:
            loss_mask = kwargs.pop("mask")
            labels.masked_fill_(loss_mask == 0, -100)

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
                "mask": kwargs.pop("mask"),
            }
        )

    return loss_fn(**loss_fn_kwargs)


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------


class FinetuneRecipeForNextTokenPrediction(BaseRecipe):
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

        self.device_mesh = None
        self.model_wrapper = None
        if "distributed" in self.cfg:
            self.model_wrapper = self.cfg.distributed.instantiate(world_size=self.dist_env.world_size)
            self.device_mesh = getattr(self.model_wrapper, "device_mesh", None)

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info("🚀 View run at {}".format(run.url))

        # Check if packed_sequence_size > 0 and use HF's flash_attention_2 for attn implementation.
        use_hf_fa2 = self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0

        # Build components
        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()
        self.model, self.optimizer = build_model_and_optimizer(
            self.dist_env.device,
            self.cfg.model,
            self.cfg.optimizer,
            use_hf_fa2,
            self.peft_config,
            self.model_wrapper,
            seed=self.cfg.get("seed", 42),
            tp_size=self.cfg.get("distributed.tp_size", 1),
            cfg_fp8=self.cfg.get("fp8", None),
        )
        self.loss_fn = build_loss_fn(self.cfg.loss_fn)
        self.dataloader, self.tokenizer = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            self.cfg.model,
            self.cfg.get("packed_sequence", None),
            device_mesh=self.device_mesh,
            seed=self.cfg.get("seed", 42),
        )

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            # For validation, do not use packed sequences for fair comparison with baseline

            self.val_dataloader, _ = build_dataloader(
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                self.cfg.model,
                cfg_ps=None,  # Use unpacked config for validation
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
            for batch_idx, batch in enumerate(self.step_scheduler):
                self._run_train_step(batch, self.step_scheduler.is_optim_step, 1.0)
                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(epoch, self.step_scheduler.step)

                if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                    self._run_validation_epoch()

    # ------------------ helpers ------------------
    def _run_train_step(self, batch, is_optim_step, clip_norm=1.0):
        """Execute a single training step.

        Args:
            batch: Batch of training data.
            is_optim_step: Flag indicating if a gradient step should be applied.
            clip_norm: Gradient clipping norm.
        """
        self.model.train()

        batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        loss_mask = batch.pop("loss_mask", None)
        if loss_mask is None:
            loss_mask = (labels.detach() != -100).to(torch.int)

        if (
            "position_ids" not in batch
            and self.device_mesh is not None
            and (self.device_mesh["cp"].size() > 1 or self.device_mesh["tp"].size() > 1)
        ):
            batch["position_ids"] = torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).to(self.model.device)

        train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels, loss_mask)
        with train_ctx():
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
                mask=loss_mask,
                model=self.model,
                hidden_states=out.hidden_states[-1] if "hidden_states" in out else None,
            )

        # local_num_loss_tokens are the number of tokens that are used for loss calculation
        # in pretraining, this excludes padding tokens. In SFT, this additionally
        # excludes the context tokens.
        local_num_loss_tokens = loss_mask.sum().detach().to(torch.int)
        # num_nonpad_tokens are the number of non-padding tokens
        self.num_nonpad_tokens += labels.numel() - count_tail_padding(labels)
        self.total_local_num_loss_tokens += local_num_loss_tokens
        self.forward_data_store.append(local_loss.detach())

        with get_sync_ctx(self.model, is_optim_step):
            local_loss.backward()

        grad_norm = None
        if is_optim_step:
            rescale_gradients(
                self.model,
                self.total_local_num_loss_tokens,
                self.device_mesh[
                    ("dp_cp" if "dp_cp" in _mesh_resources.root_to_flatten_mapping.get(self.device_mesh, {}) else "dp")
                ].get_group()
                if self.device_mesh is not None
                else None,
            )

            # Clip gradients **after** any rescaling.
            # TODO(@boxiangw): Fix TP gradient clipping
            if not self.device_mesh or self.device_mesh["tp"].size() == 1:
                grad_norm = clip_gradients(self.model, clip_norm)
            else:
                # TODO: TP WAR
                grad_norm = 0.0

            # Note(nvFSDP): Need to call these functions for nvFSDP if not using latest api
            # self.model.finish_grad_sync()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Precompute FP8 scales
            if (
                self.cfg.get("fp8", None) is not None
                and self.model.precompute_float8_dynamic_scale_for_fsdp
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
            tps = self.num_nonpad_tokens / time_delta
            self.num_nonpad_tokens = 0
            # log
            reporting_loss = self.log_train_metrics(grad_norm, tps)
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info(
                "step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem: {:.2f} GiB | tps {:.2f}".format(
                    self.step_scheduler.step,
                    self.step_scheduler.epoch,
                    reporting_loss,
                    grad_norm,
                    current_lr,
                    torch.cuda.max_memory_allocated() / 1024**3,
                    tps,
                )
            )
            torch.cuda.reset_peak_memory_stats()

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
                loss_mask = batch.pop("loss_mask", None)
                if loss_mask is None:
                    loss_mask = (labels.detach() != -100).to(torch.int)

                if (
                    self.device_mesh
                    and "position_ids" not in batch
                    and (self.device_mesh["cp"].size() > 1 or self.device_mesh["tp"].size() > 1)
                ):
                    batch["position_ids"] = (
                        torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).to(self.model.device)
                    )

                train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels, loss_mask)
                with train_ctx():
                    if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                        out = self.model(logits_to_keep=1, **batch)
                    else:
                        out = self.model(**batch)
                    local_loss = calculate_loss(
                        self.loss_fn,
                        logits=out.logits,
                        labels=labels,
                        mask=loss_mask,
                        model=self.model,
                        hidden_states=out.hidden_states[-1] if "hidden_states" in out else None,
                    )

                total_loss += local_loss.item()
                total_tokens += loss_mask.sum().item()

        # Aggregate across ranks if distributed is initialized
        if dist.is_initialized():
            tensor = torch.tensor([total_loss, total_tokens], device=self.dist_env.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = tensor.tolist()

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

    def log_train_metrics(self, grad_norm, tps) -> float:
        """Log metrics to wandb.

        Args:
            grad_norm: Grad norm from the training step.
            tps: Tokens per second.

        Returns:
            Reporting loss.
        """
        if not self.device_mesh:
            dp_group = None
        elif self.device_mesh["cp"].size() > 1:
            dp_group = self.device_mesh["dp_cp"].get_group()
        else:
            dp_group = self.device_mesh["dp"].get_group()

        total_loss, total_num_loss_tokens = reduce_loss(
            self.forward_data_store, self.total_local_num_loss_tokens, per_token_loss=True, dp_group=dp_group
        )
        reporting_loss = (total_loss / total_num_loss_tokens).item()
        grad_norm = grad_norm.item() if not isinstance(grad_norm, float) else grad_norm  # TP WAR
        self.total_local_num_loss_tokens.zero_()
        self.forward_data_store = []
        log_data = {
            "train_loss": reporting_loss,
            "loss_sum": total_loss,
            "step": self.step_scheduler.step,
            "epoch": self.step_scheduler.epoch,
            "grad_norm": grad_norm,
            "num_tokens_per_step": total_num_loss_tokens,
            "tps": tps,
        }
        if self.optimizer.param_groups:
            log_data["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        if wandb.run is not None:
            wandb.log(log_data)
        return reporting_loss


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
    trainer = FinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
