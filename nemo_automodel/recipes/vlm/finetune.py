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
from transformers import AutoProcessor
from wandb import Settings

from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.vlm.collate_fns import COLLATE_FNS
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.nvfsdp import NVFSDPManager
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.training.base_recipe import BaseRecipe
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.components.training.utils import count_tail_padding
from nemo_automodel.components.utils.dist_utils import (
    clip_gradients,
    get_sync_ctx,
    reduce_loss,
    rescale_gradients,
)
from nemo_automodel.components.utils.model_utils import apply_parameter_freezing, print_trainable_parameters

logger = logging.getLogger(__name__)

# ---------------------------
#  Stateless helper functions
# ---------------------------


def build_model_and_optimizer(
    device,
    cfg_model,
    cfg_opt,
    cfg_freeze,
    cfg_peft,
    model_wrapper,
    seed,
    tp_size=1,
) -> tuple[nn.Module, "Optimizer"]:  # noqa: F821
    """Build and initialize a model for VLM."""
    with StatefulRNG(seed=seed, ranked=True):
        model = cfg_model.instantiate()

        if cfg_freeze is not None:
            apply_parameter_freezing(model, cfg_freeze)
        else:
            for m in model.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.requires_grad = False

        # Optionally apply PEFT (e.g., LoRA/DoRA, etc)
        if cfg_peft is not None:
            apply_lora_to_linear_modules(model, cfg_peft)

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
        else:
            model = model.to(device)

        trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
        assert len(trainable_params) > 0, "trainable_params cannot be empty"
        if tp_size > 1:
            cfg_opt.foreach = False
        optimizer = cfg_opt.instantiate(params=trainable_params)

        return model, optimizer


def build_checkpoint_config(cfg_ckpt, cache_dir, model_repo_id, is_peft):
    """Build a checkpoint configuration."""
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
    return CheckpointingConfig(**ckpt_kwargs)


def build_loss_fn(device, cfg_loss):
    """Build a loss function.

    Args:
        device: The target device.
        cfg_loss: Loss function configuration or a callable loss function.

    Returns:
        The instantiated loss function on the specified device.
    """
    if callable(cfg_loss):
        return cfg_loss
    else:
        return cfg_loss.instantiate().to(device)


def build_dataloader(cfg_ds, cfg_dl, cfg_model, cfg_processor, device_mesh, seed) -> DataLoader:
    """Build a VLM dataloader."""
    dist_sampler_kwargs = {
        "shuffle": cfg_dl.get("shuffle", True),
    }
    if device_mesh is not None:
        dist_sampler_kwargs |= {
            "num_replicas": device_mesh["data_parallel"].size(),
            "rank": device_mesh["data_parallel"].get_local_rank(),
        }

    with StatefulRNG(seed=seed, ranked=True):
        if cfg_processor is not None:
            if hasattr(cfg_processor, "instantiate"):
                processor = cfg_processor.instantiate()
            else:
                processor_kwargs = cfg_processor.to_dict()
                processor = AutoProcessor.from_pretrained(cfg_model.pretrained_model_name_or_path, **processor_kwargs)
        else:
            processor = AutoProcessor.from_pretrained(cfg_model.pretrained_model_name_or_path)

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

        return cfg_dl.instantiate(dataset=ds, sampler=sampler, collate_fn=collate_fn)


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


def build_wandb(cfg):
    """Instantiates wandb and returns the instance.

    If no name is given, it will use the model name.
    """
    assert cfg.get("wandb", None) is not None
    kwargs = cfg.wandb.to_dict()
    if kwargs.get("name", "") == "":
        kwargs["name"] = "_".join(cfg.get("model.pretrained_model_name_or_path").split("/")[-2:])
    run = wandb.init(
        **kwargs,
        config=cfg,
        settings=Settings(silent=True),
    )
    return run


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

    def setup(self):
        """Override setup to use VLM-specific builders."""
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

        # Build components with VLM-specific functions
        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()
        self.model, self.optimizer = build_model_and_optimizer(
            self.dist_env.device,
            self.cfg.model,
            self.cfg.optimizer,
            self.cfg.get("freeze_config", None),  # VLM-specific
            self.peft_config,
            self.model_wrapper,
            seed=self.cfg.get("seed", 42),
            tp_size=self.cfg.get("distributed.tp_size", 1),
        )
        self.loss_fn = build_loss_fn(self.dist_env.device, self.cfg.loss_fn)
        self.dataloader = build_dataloader(  # VLM-specific
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
            self.val_dataloader = build_dataloader(  # VLM-specific
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                self.cfg.model,
                self.cfg.get("processor", None),
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
            )

        # Initialize metrics required for calculating loss
        self.total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        self.forward_data_store = []

        # Scheduler
        self.step_scheduler = build_step_scheduler(self.cfg.get("step_scheduler", None), self.dataloader)

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
        self.num_tokens = 0
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
            clip_norm: Gradient clipping norm value.

        Returns:
            Grad norm from the training step.
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
            and (self.device_mesh["context_parallel"].size() > 1 or self.device_mesh["tensor_parallel"].size() > 1)
        ):
            batch["position_ids"] = torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).to(self.model.device)

        train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels, loss_mask)
        with train_ctx():
            out = self.model(**batch)
            local_loss = self.loss_fn(
                out.logits.view(-1, out.logits.size(-1)), labels.view(-1), mask=loss_mask, reduction="sum"
            )

        local_num_tokens = loss_mask.sum().detach().to(torch.int)
        self.num_tokens += labels.numel() - count_tail_padding(labels)
        self.total_num_tokens += local_num_tokens
        self.forward_data_store.append(local_loss.detach())

        with get_sync_ctx(self.model, is_optim_step):
            local_loss.backward()

        grad_norm = None
        if is_optim_step:
            rescale_gradients(
                self.model,
                self.total_num_tokens,
                self.device_mesh[
                    (
                        "dp_cp"
                        if "dp_cp" in _mesh_resources.root_to_flatten_mapping.get(self.device_mesh, {})
                        else "data_parallel"
                    )
                ].get_group()
                if self.device_mesh is not None
                else None,
            )

            # Clip gradients **after** any rescaling.
            # TODO(@boxiangw): Fix TP gradient clipping
            if not self.device_mesh or self.device_mesh["tensor_parallel"].size() == 1:
                grad_norm = clip_gradients(self.model, clip_norm)
            else:
                # TODO: TP WAR
                grad_norm = 0.0

            # Note(nvFSDP): Need to call these functions for nvFSDP if not using latest api
            # self.model.finish_grad_sync()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Note(nvFSDP): Need to call these functions for nvFSDP if not using latest api
            # self.model.install_optimized_model_weights()
            # self.model.zero_grad_buffer()

            # TPS is calculated as follows (assuming grad-accumulation-steps=2):
            # fwd 0 | bwd 0 | fwd 1 | bwd 1 | opt 0 | fwd 2 | bwd 2 | ...
            # ^                                     ^
            t = time.perf_counter()
            time_delta = t - self.timestamp
            self.timestamp = t
            tps = self.num_tokens / time_delta
            self.num_tokens = 0
            # log
            reporting_loss = self.log_train_metrics(grad_norm, tps)
            logging.info(
                "step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | mem: {:.2f} GiB | tps {:.2f}".format(
                    self.step_scheduler.step,
                    self.step_scheduler.epoch,
                    reporting_loss,
                    grad_norm,
                    torch.cuda.max_memory_allocated() / 1024**3,
                    tps,
                )
            )
            torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def _run_validation_epoch(self) -> float:
        """Run one pass over `self.val_dataloader` and return average loss per token."""
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
                    and (
                        self.device_mesh["context_parallel"].size() > 1
                        or self.device_mesh["tensor_parallel"].size() > 1
                    )
                ):
                    batch["position_ids"] = (
                        torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).to(self.model.device)
                    )

                train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels, loss_mask)
                with train_ctx():
                    out = self.model(**batch)
                    local_loss = self.loss_fn(
                        out.logits.view(-1, out.logits.size(-1)), labels.view(-1), mask=loss_mask, reduction="sum"
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
        logging.info(
            "[val] step {} | epoch {} | loss {:.4f}".format(
                self.step_scheduler.step, self.step_scheduler.epoch, val_loss
            )
        )

    def log_train_metrics(self, grad_norm, tps):
        """Log metrics to wandb.

        Args:
            grad_norm: Grad norm from the training step.
            tps: Tokens per second throughput metric.

        Returns:
            Reporting loss.
        """
        if not self.device_mesh:
            dp_group = None
        elif self.device_mesh["context_parallel"].size() > 1:
            dp_group = self.device_mesh["dp_cp"].get_group()
        else:
            dp_group = self.device_mesh["data_parallel"].get_group()

        total_loss, total_num_tokens = reduce_loss(
            self.forward_data_store, self.total_num_tokens, per_token_loss=True, dp_group=dp_group
        )
        reporting_loss = (total_loss / total_num_tokens).item()
        grad_norm = grad_norm.item() if not isinstance(grad_norm, float) else grad_norm  # TP WAR
        self.total_num_tokens.zero_()
        self.forward_data_store = []
        log_data = {
            "train_loss": reporting_loss,
            "loss_sum": total_loss,
            "step": self.step_scheduler.step,
            "epoch": self.step_scheduler.epoch,
            "grad_norm": grad_norm,
            "num_tokens_per_step": total_num_tokens,
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


def main():
    """Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the trainer, and initiates the training loop.
    """
    script_path = pathlib.Path(__file__).parent.resolve()
    cfg = parse_args_and_load_config(script_path / "gemma_3_vl_4b_cord_v2.yaml")
    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
