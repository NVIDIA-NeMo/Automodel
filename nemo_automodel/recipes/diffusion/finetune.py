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

from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.recipes.llm.train_ft import build_distributed, build_wandb
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages

import os
import logging
import re
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import wandb
from math import ceil

from nemo_automodel.components._diffusers.flow_matching.training_step_t2v import (
    step_fsdp_transformer_t2v,
)

from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from torch.distributed.fsdp import MixedPrecisionPolicy
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig
from transformers.utils.hub import TRANSFORMERS_CACHE


def build_model_and_optimizer(
    *,
    model_id: str,
    learning_rate: float,
    device: torch.device,
    bf16_dtype: torch.dtype,
    cpu_offload: bool = True,
    tp_size: int = 1,
    cp_size: int = 1,
    pp_size: int = 1,
    dp_size: Optional[int] = None,
    use_hf_tp_plan: bool = False,
    optimizer_cfg: Optional[Dict[str, Any]] = None,
) -> tuple[NeMoAutoDiffusionPipeline, dict[str, Dict[str, Any]], torch.optim.Optimizer]:
    """Build the WAN 2.1 diffusion model, parallel scheme, and optimizer."""

    logging.info("[INFO] Building NeMoAutoDiffusionPipeline with transformer parallel scheme...")

    if not dist.is_initialized():
        logging.info("[WARN] torch.distributed not initialized; proceeding in single-process mode")

    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if dp_size is None:
        denom = max(1, tp_size * cp_size * pp_size)
        dp_size = max(1, world_size // denom)

    fsdp2_manager = FSDP2Manager(
        dp_size=dp_size,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        backend="nccl",
        world_size=world_size,
        use_hf_tp_plan=use_hf_tp_plan,
        activation_checkpointing=True,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=bf16_dtype,
            reduce_dtype=bf16_dtype,
            output_dtype=bf16_dtype,
        ),
    )

    parallel_scheme = {"transformer": fsdp2_manager}

    pipe = NeMoAutoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=bf16_dtype,
        device=device,
        parallel_scheme=parallel_scheme,
        load_for_training=True,
        components_to_load=["transformer"]
    )

    transformer_module = getattr(pipe, "transformer", None)
    if transformer_module is None:
        raise RuntimeError("transformer not found in pipeline after parallelization")

    model_map: dict[str, Dict[str, Any]] = {
        "transformer": {
            "fsdp_transformer": transformer_module
        }
    }

    trainable_params = [p for p in transformer_module.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found in transformer module!")

    optimizer_cfg = optimizer_cfg or {}
    weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    betas = optimizer_cfg.get("betas", (0.9, 0.999))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas
    )

    logging.info(
        "[INFO] Optimizer config: lr=%s, weight_decay=%s, betas=%s",
        learning_rate,
        weight_decay,
        betas
    )

    trainable_count = sum(1 for p in transformer_module.parameters() if p.requires_grad)
    frozen_count = sum(1 for p in transformer_module.parameters() if not p.requires_grad)
    logging.info(f"[INFO] Trainable parameters: {trainable_count}, Frozen parameters: {frozen_count}")

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        logging.info(f"[INFO] GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

    logging.info("[INFO] NeMoAutoDiffusion setup complete (pipeline + optimizer)")

    return pipe, model_map, optimizer, fsdp2_manager.device_mesh


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int,
    steps_per_epoch: int,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Build the cosine annealing learning rate scheduler."""

    total_steps = max(1, num_epochs * max(1, steps_per_epoch))
    logging.info(f"[INFO] Scheduler configured for {total_steps} total steps")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=eta_min,
    )

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

class TrainWan21DiffusionRecipe(BaseRecipe):
    """Config-driven wrapper around WAN 2.1 T2V training."""

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

        self.model_id = self.cfg.get(
            "model.pretrained_model_name_or_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        )
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
        self.node_rank = (
            dist.get_rank() // self.local_world_size if dist.is_initialized() else 0
        )

        logging.info("[INFO] WAN 2.1 T2V Trainer with Flow Matching")
        logging.info(
            f"[INFO] Total GPUs: {self.world_size}, GPUs per node: {self.local_world_size}, Num nodes: {self.num_nodes}"
        )
        logging.info(f"[INFO] Node rank: {self.node_rank}, Local rank: {self.local_rank}")
        logging.info(f"[INFO] Learning rate: {self.learning_rate}")

        fsdp_cfg = self.cfg.get("fsdp", {})
        fm_cfg = self.cfg.get("flow_matching", {})

        self.cpu_offload = fsdp_cfg.get("cpu_offload", True)
        self.use_sigma_noise = fm_cfg.get("use_sigma_noise", True)
        self.timestep_sampling = fm_cfg.get("timestep_sampling", "uniform")
        self.logit_mean = fm_cfg.get("logit_mean", 0.0)
        self.logit_std = fm_cfg.get("logit_std", 1.0)
        self.flow_shift = fm_cfg.get("flow_shift", 3.0)
        self.mix_uniform_ratio = fm_cfg.get("mix_uniform_ratio", 0.1)

        logging.info(
            f"[INFO] Flow matching: {'ENABLED' if self.use_sigma_noise else 'DISABLED'}"
        )
        if self.use_sigma_noise:
            logging.info(f"[INFO]   - Timestep sampling: {self.timestep_sampling}")
            logging.info(f"[INFO]   - Flow shift: {self.flow_shift}")
            logging.info(f"[INFO]   - Mix uniform ratio: {self.mix_uniform_ratio}")

        tp_size = fsdp_cfg.get("tp_size", 1)
        cp_size = fsdp_cfg.get("cp_size", 1)
        pp_size = fsdp_cfg.get("pp_size", 1)
        dp_size = fsdp_cfg.get("dp_size", None)
        use_hf_tp_plan = fsdp_cfg.get("use_hf_tp_plan", False)

        (
            self.pipe,
            self.model_map,
            self.optimizer,
            self.device_mesh
        ) = build_model_and_optimizer(
            model_id=self.model_id,
            learning_rate=self.learning_rate,
            device=self.device,
            bf16_dtype=self.bf16,
            cpu_offload=self.cpu_offload,
            tp_size=tp_size,
            cp_size=cp_size,
            pp_size=pp_size,
            dp_size=dp_size,
            use_hf_tp_plan=use_hf_tp_plan,
            optimizer_cfg=self.cfg.get("optim.optimizer", {}),
        )

        # Expose the trainable module as a single model for BaseRecipe tracking
        self.model = self.model_map["transformer"]["fsdp_transformer"]
        self.peft_config = None

        batch_cfg = self.cfg.get("batch", {})
        training_cfg = self.cfg.get("training", {})
        logging_cfg = self.cfg.get("logging", {})
        checkpoint_cfg = self.cfg.get("checkpoint", None)

        self.batch_size_per_node = batch_cfg.get("batch_size_per_node", 1)
        self.num_epochs = training_cfg.get("num_epochs", 1)
        self.save_every = logging_cfg.get("save_every", 500)
        self.log_every = logging_cfg.get("log_every", 5)

        # Strictly require checkpoint config from YAML (no fallback)
        if checkpoint_cfg is None:
            raise ValueError("checkpoint config is required in YAML (enabled, checkpoint_dir, model_save_format, save_consolidated)")
        if not checkpoint_cfg.get("enabled", False):
            raise ValueError("checkpoint.enabled must be true in YAML for diffusion training")

        # Build BaseRecipe-style checkpointing configuration (DCP/TORCH_SAVE) from YAML
        model_state_dict_keys = list(self.model.state_dict().keys())
        model_cache_dir = self.cfg.get("model.cache_dir", None)
        self.checkpoint_config = CheckpointingConfig(
            enabled=checkpoint_cfg.get("enabled"),
            checkpoint_dir=checkpoint_cfg.get("checkpoint_dir"),
            model_save_format=checkpoint_cfg.get("model_save_format"),
            model_cache_dir=model_cache_dir if model_cache_dir is not None else TRANSFORMERS_CACHE,
            model_repo_id=self.model_id,
            save_consolidated=checkpoint_cfg.get("save_consolidated"),
            is_peft=False,
            model_state_dict_keys=model_state_dict_keys,
        )
        self.restore_from = checkpoint_cfg.get("restore_from", None)

        dataloader_cfg = self.cfg.get("data.dataloader")
        if not hasattr(dataloader_cfg, "instantiate"):
            raise RuntimeError("data.dataloader must be a config node with instantiate()")

        dataloader_obj = dataloader_cfg.instantiate()
        if isinstance(dataloader_obj, tuple):
            self.dataloader, self.sampler = dataloader_obj
        else:
            self.dataloader = dataloader_obj
            self.sampler = getattr(dataloader_obj, "sampler", None)

        self.raw_steps_per_epoch = len(self.dataloader)
        if self.raw_steps_per_epoch == 0:
            raise RuntimeError("Training dataloader is empty; cannot proceed with training")

        # Derive DP size consistent with model parallel config
        denom = max(1, tp_size * cp_size * pp_size)
        self.dp_size = fsdp_cfg.get("dp_size", None)
        if self.dp_size is None:
            self.dp_size = max(1, self.world_size // denom)

        # Infer local micro-batch size from dataloader if available
        self.local_batch_size = getattr(self.dataloader, "batch_size", 1)
        # Desired global effective batch size across all DP ranks and nodes
        self.global_batch_size = max(1, int(self.batch_size_per_node) * int(self.num_nodes))

        # Steps per epoch after gradient accumulation
        # grad_acc_steps must be an integer; StepScheduler will assert divisibility
        grad_acc_steps = max(1, self.global_batch_size // max(1, self.local_batch_size * self.dp_size))
        self.steps_per_epoch = ceil(self.raw_steps_per_epoch / grad_acc_steps)

        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            num_epochs=self.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
        )

        self.global_step = 0
        self.start_epoch = 0
        # Initialize StepScheduler for gradient accumulation and step/epoch bookkeeping
        self.step_scheduler = StepScheduler(
            global_batch_size=int(self.global_batch_size),
            local_batch_size=int(self.local_batch_size),
            dp_size=int(self.dp_size),
            ckpt_every_steps=int(self.save_every) if self.save_every else 1,
            dataloader=self.dataloader,
            val_every_steps=None,
            start_step=int(self.global_step),
            start_epoch=int(self.start_epoch),
            num_epochs=int(self.num_epochs),
        )
        # Optional resume only through config-defined restore_from
        if self.restore_from:
            self.load_checkpoint(restore_from=self.restore_from)

        if is_main_process():
            os.makedirs(self.checkpoint_config.checkpoint_dir, exist_ok=True)

            if wandb.run is None:
                config = {
                    "model_id": self.model_id,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "batch_size_per_node": self.batch_size_per_node,
                    "total_batch_size": self.batch_size_per_node * self.num_nodes,
                    "num_nodes": self.num_nodes,
                    "gpus_per_node": self.local_world_size,
                    "total_gpus": self.world_size,
                    "approach": "wan_t2v_flow_matching",
                    "cpu_offload": self.cpu_offload,
                    "use_sigma_noise": self.use_sigma_noise,
                    "timestep_sampling": self.timestep_sampling,
                    "logit_mean": self.logit_mean,
                    "logit_std": self.logit_std,
                    "flow_shift": self.flow_shift,
                    "mix_uniform_ratio": self.mix_uniform_ratio,
                }
                wandb.init(
                    project="wan-t2v-flow-matching",
                    config=config,
                    resume=self.restore_from is not None,
                )

        if dist.is_initialized():
            dist.barrier()

    def run_train_validation_loop(self):
        logging.info("[INFO] Starting T2V training with Flow Matching")
        logging.info(f"[INFO] Batch size per node: {self.batch_size_per_node}")
        logging.info(f"[INFO] Total effective batch size: {self.batch_size_per_node * self.num_nodes}")

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
                        loss, _ = step_fsdp_transformer_t2v(
                            pipe=self.pipe,
                            model_map=self.model_map,
                            batch=micro_batch,
                            device=self.device,
                            bf16=self.bf16,
                            use_sigma_noise=self.use_sigma_noise,
                            timestep_sampling=self.timestep_sampling,
                            logit_mean=self.logit_mean,
                            logit_std=self.logit_std,
                            flow_shift=self.flow_shift,
                            mix_uniform_ratio=self.mix_uniform_ratio,
                            global_step=global_step,
                        )
                    except Exception as exc:
                        logging.info(
                            f"[ERROR] Training step failed at epoch {epoch}, step {num_steps}: {exc}"
                        )
                        video_shape = micro_batch.get("video_latents", torch.tensor([])).shape
                        text_shape = micro_batch.get("text_embeddings", torch.tensor([])).shape
                        logging.info(
                            f"[DEBUG] Batch shapes - video: {video_shape}, text: {text_shape}"
                        )
                        raise

                    (loss / max(1, len(batch_group))).backward()
                    micro_losses.append(float(loss.item()))

                trainable_params = [
                    p
                    for p in self.model_map["transformer"]["fsdp_transformer"].parameters()
                    if p.requires_grad and p.grad is not None
                ]

                grad_norm = 0.0
                if trainable_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

                self.optimizer.step()
                self.lr_scheduler.step()

                group_loss_mean = float(sum(micro_losses) / max(1, len(micro_losses)))
                epoch_loss += group_loss_mean
                num_steps += 1
                global_step = int(self.step_scheduler.step)

                if (
                    self.log_every
                    and self.log_every > 0
                    and is_main_process()
                    and (global_step % self.log_every == 0)
                ):
                    avg_loss = epoch_loss / max(num_steps, 1)
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

            avg_loss = epoch_loss / max(num_steps, 1)
            logging.info(f"[INFO] Epoch {epoch + 1} complete. avg_loss={avg_loss:.6f}")

            if is_main_process() and wandb.run is not None:
                wandb.log({"epoch/avg_loss": avg_loss, "epoch/num": epoch + 1}, step=global_step)

        logging.info("[INFO] Training complete, saving final checkpoint...")

        self.save_checkpoint(epoch=self.step_scheduler.epoch, step=global_step)

        if is_main_process():
            logging.info(f"[INFO] Saved final checkpoint at step {global_step}")
            if wandb.run is not None:
                wandb.finish()

        logging.info("[INFO] Training complete!")