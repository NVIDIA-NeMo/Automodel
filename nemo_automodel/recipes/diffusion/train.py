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
import os
from math import ceil
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import wandb
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.recipes.llm.train_ft import build_distributed, build_wandb
from torch.distributed.fsdp import MixedPrecisionPolicy
from huggingface_hub.constants import HF_HUB_CACHE

from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
from nemo_automodel.components.flow_matching.pipeline import FlowMatchingPipeline, create_adapter


def build_model_and_optimizer(
    *,
    model_id: str,
    finetune_mode: bool,
    learning_rate: float,
    device: torch.device,
    dtype: torch.dtype,
    cpu_offload: bool = False,
    fsdp_cfg: Optional[Dict[str, Any]] = None,
    ddp_cfg: Optional[Dict[str, Any]] = None,
    attention_backend: Optional[str] = None,
    optimizer_cfg: Optional[Dict[str, Any]] = None,
    pipeline_spec: Optional[Dict[str, Any]] = None,
) -> tuple[NeMoAutoDiffusionPipeline, torch.optim.Optimizer, Any]:
    """Build the diffusion model, parallel scheme, and optimizer.

    Args:
        model_id: Pretrained model name or path.
        finetune_mode: Whether to load for finetuning (True) or pretraining (False).
        learning_rate: Learning rate for optimizer.
        device: Target device.
        dtype: Model dtype.
        cpu_offload: Whether to enable CPU offload (FSDP only).
        fsdp_cfg: FSDP configuration dict. Mutually exclusive with ddp_cfg.
        ddp_cfg: DDP configuration dict. Mutually exclusive with fsdp_cfg.
        attention_backend: Optional attention backend override.
        optimizer_cfg: Optional optimizer configuration.
        pipeline_spec: Pipeline specification for pretraining (from_config).
            Required when finetune_mode is False. Should contain:
            - transformer_cls: str (e.g., "WanTransformer3DModel", "FluxTransformer2DModel")
            - subfolder: str (e.g., "transformer")
            - Optional: pipeline_cls, load_full_pipeline, enable_gradient_checkpointing

    Returns:
        Tuple of (pipeline, optimizer, device_mesh or None).

    Raises:
        ValueError: If both fsdp_cfg and ddp_cfg are provided.
        ValueError: If finetune_mode is False and pipeline_spec is not provided.
    """
    # Validate mutually exclusive configs
    if fsdp_cfg is not None and ddp_cfg is not None:
        raise ValueError(
            "Cannot specify both 'fsdp' and 'ddp' configurations. "
            "Please provide only one distributed training strategy."
        )

    logging.info("[INFO] Building NeMoAutoDiffusionPipeline with transformer parallel scheme...")

    if not dist.is_initialized():
        logging.info("[WARN] torch.distributed not initialized; proceeding in single-process mode")

    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Build manager args based on which config is provided
    if ddp_cfg is not None:
        # DDP configuration
        logging.info("[INFO] Using DDP (DistributedDataParallel) for training")
        manager_args: Dict[str, Any] = {
            "_manager_type": "ddp",
            "backend": ddp_cfg.get("backend", "nccl"),
            "world_size": world_size,
            "activation_checkpointing": ddp_cfg.get("activation_checkpointing", False),
        }
    else:
        # FSDP configuration (default)
        fsdp_cfg = fsdp_cfg or {}
        logging.info("[INFO] Using FSDP2 (Fully Sharded Data Parallel) for training")

        dp_size = fsdp_cfg.get("dp_size", None)

        if dp_size is None:
            denom = max(1, fsdp_cfg.get("tp_size", 1) * fsdp_cfg.get("cp_size", 1) * fsdp_cfg.get("pp_size", 1))
            dp_size = max(1, world_size // denom)

        manager_args: Dict[str, Any] = {
            "_manager_type": "fsdp2",
            "dp_size": fsdp_cfg.get("dp_size", None),
            "dp_replicate_size": fsdp_cfg.get("dp_replicate_size", None),
            "tp_size": fsdp_cfg.get("tp_size", 1),
            "cp_size": fsdp_cfg.get("cp_size", 1),
            "pp_size": fsdp_cfg.get("pp_size", 1),
            "backend": "nccl",
            "world_size": world_size,
            "use_hf_tp_plan": fsdp_cfg.get("use_hf_tp_plan", False),
            "activation_checkpointing": fsdp_cfg.get("activation_checkpointing", True),
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=dtype,
                reduce_dtype=torch.float32,
                output_dtype=dtype,
            ),
        }

    parallel_scheme = {"transformer": manager_args}

    if finetune_mode:
        # Finetuning: load from pretrained weights
        logging.info("[INFO] Loading pretrained model for finetuning")
        pipe, created_managers = NeMoAutoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device=device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
            load_for_training=True,
            low_cpu_mem_usage=True,
        )
    else:
        # Pretraining: initialize with random weights using pipeline_spec
        if pipeline_spec is None:
            raise ValueError(
                "pipeline_spec is required for pretraining (finetune_mode=False). "
                "Please provide pipeline_spec in your YAML config with at least:\n"
                "  pipeline_spec:\n"
                "    transformer_cls: 'WanTransformer3DModel'  # or 'FluxTransformer2DModel', etc.\n"
                "    subfolder: 'transformer'"
            )
        logging.info("[INFO] Initializing model with random weights for pretraining")
        pipe, created_managers = NeMoAutoDiffusionPipeline.from_config(
            model_id,
            pipeline_spec=pipeline_spec,
            torch_dtype=dtype,
            device=device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
        )
    fsdp2_manager = created_managers["transformer"]
    transformer_module = pipe.transformer
    if attention_backend is not None:
        logging.info(f"[INFO] Setting attention backend to {attention_backend}")
        transformer_module.set_attention_backend(attention_backend)

    trainable_params = [p for p in transformer_module.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found in transformer module!")

    optimizer_cfg = optimizer_cfg or {}
    weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    betas = optimizer_cfg.get("betas", (0.9, 0.999))
    # TODO: Support other optimizers
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay, betas=betas)

    logging.info("[INFO] Optimizer config: lr=%s, weight_decay=%s, betas=%s", learning_rate, weight_decay, betas)

    trainable_count = sum(1 for p in transformer_module.parameters() if p.requires_grad)
    frozen_count = sum(1 for p in transformer_module.parameters() if not p.requires_grad)
    logging.info(f"[INFO] Trainable parameters: {trainable_count}, Frozen parameters: {frozen_count}")

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        logging.info(f"[INFO] GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

    logging.info("[INFO] NeMoAutoDiffusion setup complete (pipeline + optimizer)")

    return pipe, optimizer, getattr(fsdp2_manager, "device_mesh", None)


def build_lr_scheduler(
    cfg,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> Optional[OptimizerParamScheduler]:
    """Build the learning rate scheduler.

    Args:
        cfg: Configuration for the OptimizerParamScheduler from YAML. If None, no scheduler
            is created and constant LR is used. Supports:
            - lr_decay_style: constant, linear, cosine, inverse-square-root, WSD
            - lr_warmup_steps: Number of warmup steps (or fraction < 1 for percentage)
            - min_lr: Minimum LR after decay
            - init_lr: Initial LR for warmup (defaults to 10% of max_lr if warmup enabled)
            - wd_incr_style: constant, linear, cosine (for weight decay scheduling)
            - wsd_decay_steps: WSD-specific decay steps
            - lr_wsd_decay_style: WSD-specific decay style (cosine, linear, exponential, minus_sqrt)
        optimizer: The optimizer to be scheduled.
        total_steps: Total number of optimizer steps for the training run.

    Returns:
        OptimizerParamScheduler instance, or None if cfg is None.
    """
    if cfg is None:
        return None

    user_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

    base_lr = optimizer.param_groups[0]["lr"]
    base_wd = optimizer.param_groups[0].get("weight_decay", 0.0)

    # Compute defaults from runtime values
    default_cfg: Dict[str, Any] = {
        "optimizer": optimizer,
        "lr_warmup_steps": min(1000, total_steps // 10),
        "lr_decay_steps": total_steps,
        "lr_decay_style": "cosine",
        "init_lr": base_lr * 0.1,
        "max_lr": base_lr,
        "min_lr": base_lr * 0.01,
        "start_wd": base_wd,
        "end_wd": base_wd,
        "wd_incr_steps": total_steps,
        "wd_incr_style": "constant",
    }

    # Handle warmup as fraction before merging
    if "lr_warmup_steps" in user_cfg:
        warmup = user_cfg["lr_warmup_steps"]
        if isinstance(warmup, float) and 0 < warmup < 1:
            user_cfg["lr_warmup_steps"] = int(warmup * total_steps)

    # WSD defaults if user specifies WSD style
    if user_cfg.get("lr_decay_style") == "WSD":
        default_cfg["wsd_decay_steps"] = max(1, total_steps // 10)
        default_cfg["lr_wsd_decay_style"] = "cosine"

    # User config overrides defaults
    default_cfg.update(user_cfg)

    # If user disabled warmup, set init_lr = max_lr
    if default_cfg["lr_warmup_steps"] == 0:
        default_cfg["init_lr"] = default_cfg["max_lr"]

    # Ensure warmup < decay steps
    if default_cfg["lr_warmup_steps"] >= default_cfg["lr_decay_steps"]:
        default_cfg["lr_warmup_steps"] = max(0, default_cfg["lr_decay_steps"] - 1)

    logging.info(
        f"[INFO] LR Scheduler: style={default_cfg['lr_decay_style']}, "
        f"warmup={default_cfg['lr_warmup_steps']}, total={default_cfg['lr_decay_steps']}, "
        f"max_lr={default_cfg['max_lr']}, min_lr={default_cfg['min_lr']}"
    )

    return OptimizerParamScheduler(
        optimizer=default_cfg["optimizer"],
        init_lr=default_cfg["init_lr"],
        max_lr=default_cfg["max_lr"],
        min_lr=default_cfg["min_lr"],
        lr_warmup_steps=default_cfg["lr_warmup_steps"],
        lr_decay_steps=default_cfg["lr_decay_steps"],
        lr_decay_style=default_cfg["lr_decay_style"],
        start_wd=default_cfg["start_wd"],
        end_wd=default_cfg["end_wd"],
        wd_incr_steps=default_cfg["wd_incr_steps"],
        wd_incr_style=default_cfg["wd_incr_style"],
        wsd_decay_steps=default_cfg.get("wsd_decay_steps"),
        lr_wsd_decay_style=default_cfg.get("lr_wsd_decay_style"),
    )


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


class TrainDiffusionRecipe(BaseRecipe):
    """Training recipe for diffusion models."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        setup_logging()

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            if run is not None:
                logging.info("🚀 View run at {}".format(run.url))

        self.seed = self.cfg.get("seed", 42)
        self.rng = StatefulRNG(seed=self.seed, ranked=True)

        self.model_id = self.cfg.get("model.pretrained_model_name_or_path")
        self.attention_backend = self.cfg.get("model.attention_backend")
        self.learning_rate = self.cfg.get("optim.learning_rate", 5e-6)
        self.bf16 = torch.bfloat16

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
        self.local_world_size = max(self.local_world_size, 1)
        self.num_nodes = max(1, self.world_size // self.local_world_size)
        self.node_rank = dist.get_rank() // self.local_world_size if dist.is_initialized() else 0

        logging.info("[INFO] Diffusion Trainer with Flow Matching")
        logging.info(
            f"[INFO] Total GPUs: {self.world_size}, GPUs per node: {self.local_world_size}, Num nodes: {self.num_nodes}"
        )
        logging.info(f"[INFO] Node rank: {self.node_rank}, Local rank: {self.local_rank}")
        logging.info(f"[INFO] Learning rate: {self.learning_rate}")

        # Get distributed training configs (mutually exclusive)
        fsdp_cfg = self.cfg.get("fsdp", None)
        ddp_cfg = self.cfg.get("ddp", None)
        fm_cfg = self.cfg.get("flow_matching", {})

        # Validate mutually exclusive distributed configs
        if fsdp_cfg is not None and ddp_cfg is not None:
            raise ValueError(
                "Cannot specify both 'fsdp' and 'ddp' configurations in YAML. "
                "Please provide only one distributed training strategy."
            )

        self.cpu_offload = fsdp_cfg.get("cpu_offload", False) if fsdp_cfg else False

        # Flow matching configuration
        self.adapter_type = fm_cfg.get("adapter_type", "simple")
        self.timestep_sampling = fm_cfg.get("timestep_sampling", "logit_normal")
        self.logit_mean = fm_cfg.get("logit_mean", 0.0)
        self.logit_std = fm_cfg.get("logit_std", 1.0)
        self.flow_shift = fm_cfg.get("flow_shift", 3.0)
        self.mix_uniform_ratio = fm_cfg.get("mix_uniform_ratio", 0.1)
        self.sigma_min = fm_cfg.get("sigma_min", 0.0)
        self.sigma_max = fm_cfg.get("sigma_max", 1.0)
        self.num_train_timesteps = fm_cfg.get("num_train_timesteps", 1000)
        self.i2v_prob = fm_cfg.get("i2v_prob", 0.3)
        self.use_loss_weighting = fm_cfg.get("use_loss_weighting", True)
        self.log_interval = fm_cfg.get("log_interval", 100)
        self.summary_log_interval = fm_cfg.get("summary_log_interval", 10)

        # Adapter-specific configuration
        adapter_kwargs = fm_cfg.get("adapter_kwargs", {})
        self.adapter_kwargs = adapter_kwargs.to_dict()

        logging.info("[INFO] Flow Matching V2 Pipeline")
        logging.info(f"[INFO]   - Adapter type: {self.adapter_type}")
        logging.info(f"[INFO]   - Timestep sampling: {self.timestep_sampling}")
        logging.info(f"[INFO]   - Flow shift: {self.flow_shift}")
        logging.info(f"[INFO]   - Mix uniform ratio: {self.mix_uniform_ratio}")
        logging.info(f"[INFO]   - Use loss weighting: {self.use_loss_weighting}")

        # Get pipeline_spec for pretraining mode (required when mode != "finetune")
        pipeline_spec_cfg = self.cfg.get("model.pipeline_spec", None)
        pipeline_spec = pipeline_spec_cfg.to_dict() if pipeline_spec_cfg is not None else None

        (self.pipe, self.optimizer, self.device_mesh) = build_model_and_optimizer(
            model_id=self.model_id,
            finetune_mode=self.cfg.get("model.mode", "finetune").lower() == "finetune",
            learning_rate=self.learning_rate,
            device=self.device,
            dtype=self.bf16,
            cpu_offload=self.cpu_offload,
            fsdp_cfg=fsdp_cfg,
            ddp_cfg=ddp_cfg,
            optimizer_cfg=self.cfg.get("optim.optimizer", {}),
            attention_backend=self.attention_backend,
            pipeline_spec=pipeline_spec,
        )

        self.model = self.pipe.transformer
        self.peft_config = None

        checkpoint_cfg = self.cfg.get("checkpoint", None)

        self.num_epochs = self.cfg.step_scheduler.num_epochs
        self.log_every = self.cfg.get("step_scheduler.log_every", 5)

        # Strictly require checkpoint config from YAML (no fallback)
        if checkpoint_cfg is None:
            raise ValueError(
                "checkpoint config is required in YAML (enabled, checkpoint_dir, model_save_format, save_consolidated)"
            )

        # Build BaseRecipe-style checkpointing configuration (DCP/TORCH_SAVE) from YAML
        model_state_dict_keys = list(self.model.state_dict().keys())
        model_cache_dir = self.cfg.get("model.cache_dir", None)
        self.checkpoint_config = CheckpointingConfig(
            enabled=checkpoint_cfg.get("enabled"),
            checkpoint_dir=checkpoint_cfg.get("checkpoint_dir"),
            model_save_format=checkpoint_cfg.get("model_save_format"),
            model_cache_dir=model_cache_dir if model_cache_dir is not None else HF_HUB_CACHE,
            model_repo_id=self.model_id,
            save_consolidated=checkpoint_cfg.get("save_consolidated"),
            is_peft=False,
            model_state_dict_keys=model_state_dict_keys,
        )
        self.restore_from = checkpoint_cfg.get("restore_from", None)
        self.checkpointer = Checkpointer(
            config=self.checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=None,
        )

        dataloader_cfg = self.cfg.get("data.dataloader")
        if not hasattr(dataloader_cfg, "instantiate"):
            raise RuntimeError("data.dataloader must be a config node with instantiate()")

        self.dataloader, self.sampler = dataloader_cfg.instantiate(
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
            batch_size=self.cfg.step_scheduler.local_batch_size,
        )

        self.raw_steps_per_epoch = len(self.dataloader)
        if self.raw_steps_per_epoch == 0:
            raise RuntimeError("Training dataloader is empty; cannot proceed with training")

        # Derive DP size consistent with model parallel config
        if ddp_cfg is not None:
            # DDP uses pure data parallelism across all ranks
            self.dp_size = self.world_size
        else:
            # FSDP may have TP/CP/PP dimensions
            _fsdp_cfg = fsdp_cfg or {}
            tp_size = _fsdp_cfg.get("tp_size", 1)
            cp_size = _fsdp_cfg.get("cp_size", 1)
            pp_size = _fsdp_cfg.get("pp_size", 1)
            denom = max(1, tp_size * cp_size * pp_size)
            self.dp_size = _fsdp_cfg.get("dp_size", None)
            if self.dp_size is None:
                self.dp_size = max(1, self.world_size // denom)

        # Infer local micro-batch size from dataloader if available
        self.local_batch_size = self.cfg.step_scheduler.local_batch_size
        # Desired global effective batch size across all DP ranks and nodes
        self.global_batch_size = self.cfg.step_scheduler.global_batch_size
        # Steps per epoch after gradient accumulation
        grad_acc_steps = max(1, self.global_batch_size // max(1, self.local_batch_size * self.dp_size))
        self.steps_per_epoch = ceil(self.raw_steps_per_epoch / grad_acc_steps)

        # Calculate total optimizer steps for LR scheduler
        total_steps = self.num_epochs * self.steps_per_epoch

        # Build LR scheduler (returns None if lr_scheduler not in config)
        # Wrap in list for compatibility with checkpointing (OptimizerState expects list)
        lr_scheduler = build_lr_scheduler(
            self.cfg.get("lr_scheduler", None),
            self.optimizer,
            total_steps,
        )
        self.lr_scheduler = [lr_scheduler] if lr_scheduler is not None else None

        self.global_step = 0
        self.start_epoch = 0
        # Initialize StepScheduler for gradient accumulation and step/epoch bookkeeping
        self.step_scheduler = StepScheduler(
            global_batch_size=self.cfg.step_scheduler.global_batch_size,
            local_batch_size=self.cfg.step_scheduler.local_batch_size,
            dp_size=int(self.dp_size),
            ckpt_every_steps=self.cfg.step_scheduler.ckpt_every_steps,
            dataloader=self.dataloader,
            val_every_steps=None,
            start_step=int(self.global_step),
            start_epoch=int(self.start_epoch),
            num_epochs=int(self.num_epochs),
        )

        self.load_checkpoint(self.restore_from)

        # Init Flow Matching Pipeline V2 with model adapter
        model_adapter = create_adapter(self.adapter_type, **self.adapter_kwargs)
        self.flow_matching_pipeline = FlowMatchingPipeline(
            model_adapter=model_adapter,
            num_train_timesteps=self.num_train_timesteps,
            timestep_sampling=self.timestep_sampling,
            flow_shift=self.flow_shift,
            i2v_prob=self.i2v_prob,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mix_uniform_ratio=self.mix_uniform_ratio,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            use_loss_weighting=self.use_loss_weighting,
            log_interval=self.log_interval,
            summary_log_interval=self.summary_log_interval,
            device=self.device,
        )
        logging.info(f"[INFO] Flow Matching Pipeline V2 initialized with {self.adapter_type} adapter")

        if is_main_process():
            os.makedirs(self.checkpoint_config.checkpoint_dir, exist_ok=True)

        if dist.is_initialized():
            dist.barrier()

    def run_train_validation_loop(self):
        logging.info("[INFO] Starting T2V training with Flow Matching")
        logging.info(f"[INFO] Global Batch size: {self.global_batch_size}; Local Batch size: {self.local_batch_size}")
        logging.info(f"[INFO] Num nodes: {self.num_nodes}; DP size: {self.dp_size}")

        # Keep global_step synchronized with scheduler
        global_step = int(self.step_scheduler.step)

        for epoch in self.step_scheduler.epochs:
            if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(epoch)

            # Optionally wrap dataloader with tqdm for rank-0
            if is_main_process():
                from tqdm import tqdm

                self.step_scheduler.dataloader = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            else:
                self.step_scheduler.dataloader = self.dataloader

            epoch_loss = 0.0
            num_steps = 0

            for batch_group in self.step_scheduler:
                self.optimizer.zero_grad(set_to_none=True)

                micro_losses = []
                for micro_batch in batch_group:
                    try:
                        weighted_loss, average_weighted_loss, loss_mask, metrics = self.flow_matching_pipeline.step(
                            model=self.model,
                            batch=micro_batch,
                            device=self.device,
                            dtype=self.bf16,
                            global_step=global_step,
                        )
                    except Exception as exc:
                        logging.info(f"[ERROR] Training step failed at epoch {epoch}, step {num_steps}: {exc}")
                        video_shape = micro_batch.get("video_latents", torch.tensor([])).shape
                        text_shape = micro_batch.get("text_embeddings", torch.tensor([])).shape
                        logging.info(f"[DEBUG] Batch shapes - video: {video_shape}, text: {text_shape}")
                        raise

                    # Use average_weighted_loss for backprop (scalar for gradient accumulation)
                    (average_weighted_loss / len(batch_group)).backward()
                    micro_losses.append(float(average_weighted_loss.item()))

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler[0].step(1)

                group_loss_mean = float(sum(micro_losses) / len(micro_losses))
                epoch_loss += group_loss_mean
                num_steps += 1
                global_step = int(self.step_scheduler.step)

                if self.log_every and self.log_every > 0 and is_main_process() and (global_step % self.log_every == 0):
                    avg_loss = epoch_loss / num_steps
                    log_dict = {
                        "train_loss": group_loss_mean,
                        "train_avg_loss": avg_loss,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm,
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                    if wandb.run is not None:
                        wandb.log(log_dict, step=global_step)

                    # Update tqdm if present
                    if hasattr(self.step_scheduler.dataloader, "set_postfix"):
                        self.step_scheduler.dataloader.set_postfix(
                            {
                                "loss": f"{group_loss_mean:.4f}",
                                "avg": f"{(avg_loss):.4f}",
                                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                                "gn": f"{grad_norm:.2f}",
                            }
                        )

                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(epoch, global_step)

            avg_loss = epoch_loss / num_steps
            logging.info(f"[INFO] Epoch {epoch + 1} complete. avg_loss={avg_loss:.6f}")

            if is_main_process() and wandb.run is not None:
                wandb.log({"epoch/avg_loss": avg_loss, "epoch/num": epoch + 1}, step=global_step)

        if is_main_process():
            logging.info(f"[INFO] Saved final checkpoint at step {global_step}")
            if wandb.run is not None:
                wandb.finish()

        logging.info("[INFO] Training complete!")

    def _get_dp_rank(self, include_cp: bool = False) -> int:
        """Get data parallel rank, handling DDP mode where device_mesh is None."""
        # In DDP mode, device_mesh is None, so use torch.distributed directly
        device_mesh = getattr(self, "device_mesh", None)
        if device_mesh is None:
            return dist.get_rank() if dist.is_initialized() else 0
        # Otherwise, use the parent implementation
        return super()._get_dp_rank(include_cp=include_cp)

    def _get_dp_group_size(self, include_cp: bool = False) -> int:
        """Get data parallel world size, handling DDP mode where device_mesh is None."""
        # In DDP mode, device_mesh is None, so use torch.distributed directly
        device_mesh = getattr(self, "device_mesh", None)
        if device_mesh is None:
            return dist.get_world_size() if dist.is_initialized() else 1
        # Otherwise, use the parent implementation
        return super()._get_dp_group_size(include_cp=include_cp)
