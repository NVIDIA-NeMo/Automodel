# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Fine-tune a Praxis world-action policy on recorded episodes or DROID windows.

The model is :class:`~nemo_automodel.components.models.praxis.PraxisPolicy` under FSDP2, trained on
all weights or through LoRA adapters (the ``peft`` section). The frozen Wan2.2 VAE, the Qwen text
encoder and the DROID window reader come from the reference ``praxis`` package, which must be
importable; each rank encodes its own windows on the fly. Checkpoints are written in the Praxis
layout that the reference simulation and server load.

Run::

    automodel examples/wam/praxis/praxis_1_libero.yaml --nproc-per-node 8
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import _init_peft_adapters
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.models.praxis.model import PraxisPolicy
from nemo_automodel.components.models.praxis.objective import EncodedWindow, ObjectiveSettings, PraxisObjective
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes._typed_config import RecipeConfig, _model_name_from_cfg
from nemo_automodel.recipes.base_recipe import BaseRecipe

if TYPE_CHECKING:
    from praxis.text_encoder import TextEncoder
    from praxis.train.data import TrainingWindow
    from praxis.vae import Autoencoder

    from nemo_automodel.components.models.praxis.config import JsonValue

logger = logging.getLogger(__name__)


def lora_trains_backbone(policy: PraxisPolicy) -> bool:
    """Whether any trainable parameter sits in the video backbone, so that its autograd graph must be kept.

    Linear and grouped-expert adapters are the only trainable parameters after
    :func:`apply_lora_to_linear_modules`; the base weights are frozen.

    Args:
        policy: The model after :func:`apply_lora_to_linear_modules`.
    """
    return any(parameter.requires_grad for parameter in policy.backbone.parameters())


def draw_seed(seed: int, step: int, micro: int, rank: int, world: int) -> int:
    """Seed of the random draws (timesteps, noise, dropout) of one window: per (step, micro-step, rank)."""
    return (seed * 1_000_003 + step * 65_536 + micro * world + rank) & 0x7FFFFFFFFFFFFFFF


@dataclass
class Encoders:
    """The frozen encoders every window goes through, and the encoding of the blank caption."""

    vae: Autoencoder
    text_encoder: TextEncoder
    text_uncond: Tensor


class TrainWAMFinetuneRecipe(BaseRecipe):
    """Fine-tune Praxis-1 on training windows with FSDP2, all weights or LoRA adapters.

    YAML sections: ``model`` (``PraxisPolicy.from_pretrained``), optional ``peft``, ``encoders`` (``vae``
    and ``text_encoder`` of the reference package), ``objective`` (``ObjectiveSettings``), ``dataset``
    and ``dataloader``, ``optimizer``, ``lr_scheduler``, ``step_scheduler``, ``clip_grad_norm``,
    ``distributed`` (FSDP2 with ``mp_policy``) and ``checkpoint``.
    """

    def __init__(self, cfg):
        """Keep the configuration; everything is built in :meth:`setup`."""
        self.cfg = cfg if isinstance(cfg, RecipeConfig) else RecipeConfig(cfg)

    # ------------------ build phase ------------------

    def setup(self):
        """Build the sharded model, the encoders, the data, the optimizer and the checkpointer."""
        setup_logging()
        self.dist_env = initialize_distributed(
            backend=self.cfg.get("dist_env.backend", "nccl"),
            timeout_minutes=self.cfg.get("dist_env.timeout_minutes", 1),
        )
        seed = self.cfg.get("seed", 42)
        self.rng = StatefulRNG(seed=seed, ranked=True)
        (
            self.distributed_setup,
            self.mesh_context,
            self.distributed_config,
            self.device_mesh,
            self.moe_mesh,
            self.pp_enabled,
            _pipeline_config,
            _moe_parallel_config,
            _activation_checkpointing,
        ) = self._distributed_setup_attributes(
            create_distributed_setup_from_config(self.cfg, world_size=self.dist_env.world_size)
        )
        if self.pp_enabled or self.device_mesh["tp"].size() > 1 or self.device_mesh["cp"].size() > 1:
            raise ValueError("Praxis trains with data parallelism only; set tp_size, cp_size and pp_size to 1")
        device = self.dist_env.device
        dp_rank, dp_size = self._get_dp_rank(), self._get_dp_group_size()

        peft_node = self.cfg.get("peft", None)
        self.peft_config = peft_node.instantiate() if peft_node is not None else None
        policy = self.cfg.model.instantiate(device="meta")
        if not isinstance(policy, PraxisPolicy):
            raise TypeError(f"model must build a PraxisPolicy, got {type(policy).__name__}")
        if self.peft_config is not None:
            n_adapted = apply_lora_to_linear_modules(policy, self.peft_config)
            if n_adapted == 0:
                raise RuntimeError("no module matched peft.target_modules")
            logger.info(
                "LoRA on %d linear layers (dim %d, alpha %d)", n_adapted, self.peft_config.dim, self.peft_config.alpha
            )
        checkpoint_config = self.cfg.checkpoint
        checkpoint_config.is_peft = self.peft_config is not None
        self.checkpointer = checkpoint_config.build(
            dp_rank=dp_rank,
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
            process_group=self.mesh_context.process_group,
            pp_group=self._get_pp_group(),
        )
        # The model-owned parallelizer: FSDP2 over the data-parallel mesh, expert parallelism when
        # distributed.ep_size > 1, activation checkpointing of every block.
        parallelize_model(policy, self.mesh_context)
        policy.to_empty(device=device)
        self.checkpointer.load_base_model(
            policy, device, root_dir=str(policy.pretrained_path.parent), model_name=str(policy.pretrained_path)
        )
        if self.peft_config is not None:
            # Linear and grouped-expert adapters alike, on the materialized storage.
            _init_peft_adapters(policy, self.peft_config.lora_A_init)
        policy.place_rotary_tables(device)
        if self.peft_config is None:
            policy.requires_grad_(True)
        self.model = policy
        self.model_config: dict[str, JsonValue] = json.loads((policy.pretrained_path / "config.json").read_text())

        encoders = self.cfg.get("encoders")
        text_encoder = encoders.text_encoder.instantiate(device=device)
        blank_caption = str(self.model_config["text_encoder"]["blank_caption"])
        self.encoders = Encoders(
            vae=encoders.vae.instantiate(device=device),
            text_encoder=text_encoder,
            text_uncond=text_encoder.encode([blank_caption])[0].to(device),
        )
        # LoRA adapts the action expert only: the decoder tower and the backbone autograd graph are skipped.
        # Adapters on the backbone need its autograd graph; adapters on the action expert alone do not.
        settings = self.cfg.get("objective").instantiate(
            train_backbone=self.peft_config is None or lora_trains_backbone(policy)
        )
        if not isinstance(settings, ObjectiveSettings):
            raise TypeError(f"objective must build ObjectiveSettings, got {type(settings).__name__}")
        self.objective = PraxisObjective(policy, settings, self.encoders.text_uncond).train()
        self.untrack_state("objective")  # the policy inside it is checkpointed as self.model

        dataset = self.cfg.get("dataset").instantiate(
            model_config=self.model_config, seed=seed, rank=dp_rank, world_size=dp_size
        )
        self.dataloader = self.cfg.get("dataloader").instantiate(dataset=dataset)

        self.optimizer = self.cfg.optimizer.build(
            policy, device_mesh=self.device_mesh, is_peft=self.peft_config is not None
        )
        self.step_scheduler = self.cfg.step_scheduler.build(
            self.dataloader,
            dp_size,
            self.cfg.get("step_scheduler.local_batch_size", 1),
            process_group=self.mesh_context.process_group,
        )
        self.lr_scheduler = (
            self.cfg.lr_scheduler.build(self.optimizer, self.step_scheduler)
            if self.cfg.lr_scheduler is not None
            else None
        )
        self.max_grad_norm = self.cfg.get("clip_grad_norm.max_norm", None)
        self.wandb_run = None
        if self.dist_env.is_main and self.cfg.wandb is not None:
            self.wandb_run = self.cfg.wandb.build(
                run_config=self.cfg.to_dict(), model_name=_model_name_from_cfg(self.cfg.model)
            )
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logger.info("rank %d: %.2fM trainable local elements", self.dist_env.rank, trainable / 1e6)
        self.load_checkpoint(self.cfg.get("checkpoint.restore_from", None))

    # ------------------ train phase ------------------

    def _encode(self, window: TrainingWindow) -> EncodedWindow:
        """Encode one window with the frozen VAE and text encoder on this rank."""
        from praxis.train.objective import encode_window

        encoded = encode_window(
            window, self.encoders.vae, self.encoders.text_encoder, self.model_config, self.dist_env.device
        )
        return EncodedWindow(
            text=encoded.text,
            anchor=encoded.anchor,
            chunks=list(encoded.chunks),
            actions=encoded.actions,
            proprio=encoded.proprio,
        )

    def _optimizer_step(self, batches: list[TrainingWindow], step: int) -> dict[str, float]:
        """Accumulate the gradients of ``batches``, clip, step the optimizer and the schedulers."""
        device = self.dist_env.device
        rank, world = self.dist_env.rank, self.dist_env.world_size
        seed = self.cfg.get("seed", 42)
        # The shared router injects the balance gradient directly; scale it like the main loss.
        self._set_moe_aux_loss_backward_scale(num_batches=len(batches), num_label_tokens=0)
        sums = torch.zeros(4, device=device)
        data_time = compute_time = 0.0
        for micro, window in enumerate(batches):
            t_data = time.time()
            batch = self._encode(window)
            data_time += time.time() - t_data
            t_compute = time.time()
            # Gradient sync stays on for every micro-step: FSDP2 reduce-scatters each micro-step and
            # accumulates sharded gradients (turning it off keeps every block's full gradient alive).
            generator = torch.Generator().manual_seed(draw_seed(seed, step, micro, rank, world))
            result = self.objective(batch, generator)
            (result.loss / len(batches)).backward()
            sums += torch.stack([result.loss.detach(), result.loss_video, result.loss_action, result.action_dropped])
            compute_time += time.time() - t_compute
        t_optim = time.time()
        max_norm = float("inf") if self.max_grad_norm is None else float(self.max_grad_norm)
        # Expert-parallel gradients are sums over all ranks' tokens; the shared helper scales them to the
        # data-parallel mean of the other parameters before clipping (a no-op without expert parallelism).
        grad_norm = scale_grads_and_clip_grad_norm(
            max_norm,
            [self.model],
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
            dp_group_size=world,
        )
        for optimizer in self.optimizer:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.model.update_moe_gate_bias()
        lr = self.optimizer[0].param_groups[0]["lr"]
        for scheduler in self.lr_scheduler or []:
            scheduler.step(1)
        torch.cuda.synchronize(device)
        optimizer_time = time.time() - t_optim
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        sums /= world * len(batches)
        # A skipped term (the video loss of a LoRA run) is NaN; the JSON record carries the mean of the computed ones.
        loss, loss_video, loss_action, action_dropped = (float(x) for x in sums.tolist())
        return {
            "loss": loss,
            "loss_video": loss_video if math.isfinite(loss_video) else None,
            "loss_action": loss_action,
            "action_dropped": action_dropped,
            "grad_norm": float(grad_norm),
            "lr": float(lr),
            "data_time_s": data_time,
            "compute_time_s": compute_time,
            "optimizer_time_s": optimizer_time,
        }

    def run_train_validation_loop(self):
        """Train for the scheduled steps, logging every step and checkpointing on the scheduler's steps."""
        metrics_path = Path(self.checkpointer.config.checkpoint_dir) / "metrics.jsonl"
        if self.dist_env.is_main:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.train()
        t_start = time.time()
        t_fetch = time.time()
        for epoch in self.step_scheduler.epochs:
            self.step_scheduler.set_epoch(epoch)
            for batches in self.step_scheduler:
                step = self.step_scheduler.step
                t_step = time.time()
                record = self._optimizer_step(batches, step)
                record.update(
                    step=step,
                    epoch=epoch,
                    fetch_time_s=t_step - t_fetch,
                    step_time_s=time.time() - t_step,
                    elapsed_s=time.time() - t_start,
                    max_mem_gib=torch.cuda.max_memory_allocated(self.dist_env.device) / 1024**3,
                    reserved_gib=torch.cuda.memory_reserved(self.dist_env.device) / 1024**3,
                    alloc_retries=torch.cuda.memory_stats(self.dist_env.device).get("num_alloc_retries", 0),
                )
                if self.dist_env.is_main:
                    with metrics_path.open("a") as f:
                        f.write(json.dumps(record) + "\n")
                    logger.info(
                        "step %d loss %.4f video %.4f action %.4f gnorm %.3f lr %.2e %.1fs (fetch %.1fs encode %.1fs "
                        "fwd/bwd %.1fs optimizer %.1fs) mem %.1fGiB reserved %.1fGiB alloc retries %d",
                        step,
                        record["loss"],
                        float("nan") if record["loss_video"] is None else record["loss_video"],
                        record["loss_action"],
                        record["grad_norm"],
                        record["lr"],
                        record["step_time_s"],
                        record["fetch_time_s"],
                        record["data_time_s"],
                        record["compute_time_s"],
                        record["optimizer_time_s"],
                        record["max_mem_gib"],
                        record["reserved_gib"],
                        record["alloc_retries"],
                    )
                    if self.wandb_run is not None and self.step_scheduler.is_remote_logging_step:
                        self.wandb_run.log(record, step=step)
                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(epoch, step, record["loss"])
                t_fetch = time.time()
        self._finalize_and_close_checkpointer()
        if self.wandb_run is not None:
            self.wandb_run.finish()
