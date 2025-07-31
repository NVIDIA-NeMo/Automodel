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

"""Pretraining recipe for next-token prediction using NanoGPT-style datasets.

This module reuses the bulk of the fine-tuning recipe implementation
(`finetune.py`) but removes PEFT-specific paths and ensures the embedding
layers remain **trainable**.  It is designed to work out-of-the-box with the
``BinTokenDataset`` that streams token sequences from the binary shards
produced by the *modded-NanoGPT* preprocessing script.
"""
from __future__ import annotations

import logging

import torch.nn as nn

import pathlib
import time
from typing import Any, Dict

import torch
import torch.distributed as dist
import wandb
from torch.distributed.device_mesh import _mesh_resources
from torch.utils.data import DataLoader
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


def build_model_and_optimizer(
    device,
    cfg_model,
    cfg_opt,
    use_hf_fa2,
    cfg_peft,
    model_wrapper,
    seed,
    tp_size: int = 1,
    freeze_embeddings: bool = True,
):
    """Instantiate model and optimizer (copied)."""
    with StatefulRNG(seed=seed, ranked=True):
        kwargs = {}
        if use_hf_fa2:
            kwargs["attn_implementation"] = "flash_attention_2"
            logging.warning(
                "Packed sequence is supported only with Flash Attention. "
                "Setting model's attn_implementation to flash_attention_2"
            )
        model = cfg_model.instantiate(**kwargs)
        if freeze_embeddings:
            logging.info("Freezing embeddings")
            for m in model.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.requires_grad_(False)
        if cfg_peft is not None:
            apply_lora_to_linear_modules(model, cfg_peft)

    if callable(getattr(model_wrapper, "parallelize", None)):
        if isinstance(model_wrapper, NVFSDPManager):
            trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
            assert trainable_params, "trainable_params cannot be empty"
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
    assert trainable_params, "trainable_params cannot be empty"
    if tp_size > 1:
        cfg_opt.foreach = False
    optimizer = cfg_opt.instantiate(params=trainable_params)
    return model, optimizer


def build_checkpoint_config(cfg_ckpt, cache_dir, model_repo_id, is_peft):
    """Build checkpoint configuration (copied)."""
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
    return CheckpointingConfig(**ckpt_kwargs)


def build_loss_fn(cfg_loss):
    return cfg_loss.instantiate()


def build_dataloader(
    cfg_ds,
    cfg_dl,
    cfg_model,
    cfg_ps,
    device_mesh,
    seed,
):
    dist_sampler_kwargs = {"shuffle": cfg_dl.get("shuffle", True)}
    if device_mesh is not None:
        dist_sampler_kwargs |= {
            "num_replicas": device_mesh["data_parallel"].size(),
            "rank": device_mesh["data_parallel"].get_local_rank(),
        }


    with StatefulRNG(seed=seed, ranked=True):
        ds = cfg_ds.instantiate()
        if getattr(cfg_ps, "packed_sequence_size", 0) > 0:
            logging.info(f"Packing dataset with size: {cfg_ps.packed_sequence_size}")
            ds = PackedSequence(
                ds,
                split=cfg_ds.split,
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
        return cfg_dl.instantiate(dataset=ds, sampler=sampler)


def build_distributed(cfg_dist: Dict[str, Any]):
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)


def build_step_scheduler(cfg, dataloader):
    default_kwargs = dict(num_epochs=10, grad_acc_steps=10, ckpt_every_steps=100, dataloader=dataloader)
    if cfg is not None:
        default_kwargs |= cfg.to_dict()
    return StepScheduler(**default_kwargs)


def build_lr_scheduler(cfg, optimizer, step_scheduler):
    if cfg is None:
        return None
    total_epochs = step_scheduler.num_epochs
    epoch_len = len(step_scheduler.dataloader)
    grad_acc_steps = step_scheduler.grad_acc_steps
    total_steps = (total_epochs * epoch_len) // grad_acc_steps
    base_lr = optimizer.param_groups[0]["lr"]
    default_kwargs = dict(
        optimizer=optimizer,
        init_lr=base_lr * 0.1,
        max_lr=base_lr,
        min_lr=base_lr * 0.01,
        lr_warmup_steps=min(1000, total_steps // 10),
        lr_decay_steps=total_steps,
        lr_decay_style="cosine",
        start_wd=optimizer.param_groups[0].get("weight_decay", 0.0),
        end_wd=optimizer.param_groups[0].get("weight_decay", 0.0),
        wd_incr_steps=total_steps,
        wd_incr_style="constant",
    )
    user_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    default_kwargs.update(user_cfg)
    logging.info(
        "Building LR scheduler with total_steps=%s, warmup_steps=%s, decay_style=%s",
        total_steps,
        default_kwargs["lr_warmup_steps"],
        default_kwargs["lr_decay_style"],
    )
    return OptimizerParamScheduler(**default_kwargs)


def build_wandb(cfg):
    kwargs = cfg.wandb.to_dict()
    if kwargs.get("name", "") == "":
        kwargs["name"] = "_".join(cfg.get("model.pretrained_model_name_or_path").split("/")[-2:])
    return wandb.init(**kwargs, config=cfg, settings=Settings(silent=True))


def calculate_loss(loss_fn, **kwargs):
    loss_fn_kwargs = {}
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        model = kwargs.pop("model")
        labels = kwargs.pop("labels")
        if "mask" in kwargs:
            loss_mask = kwargs.pop("mask")
            labels.masked_fill_(loss_mask == 0, -100)
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


class PretrainRecipeForNextTokenPrediction(BaseRecipe):
    """
    Pretrain recipe for next token prediction.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
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
            logging.info("🚀 View run at %s", run.url)

        use_hf_fa2 = self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0

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
        )

        self.loss_fn = build_loss_fn(self.cfg.loss_fn)
        self.dataloader = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            self.cfg.model,
            self.cfg.get("packed_sequence", None),
            device_mesh=self.device_mesh,
            seed=self.cfg.get("seed", 42),
        )

        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            self.val_dataloader = build_dataloader(
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                self.cfg.model,
                cfg_ps=None,
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
            )

        self.total_local_num_loss_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        self.forward_data_store = []

        self.step_scheduler = build_step_scheduler(self.cfg.get("step_scheduler", None), self.dataloader)
        self.lr_scheduler = build_lr_scheduler(self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler)

        restore_from = self.cfg.get("checkpoint.restore_from", None)
        self.checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            self.cfg.model.pretrained_model_name_or_path,
            True if self.cfg.get("peft", None) else False,
        )

        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)
        self.load_checkpoint(restore_from)

    def run_train_validation_loop(self):
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

    def _run_train_step(self, batch, is_optim_step, clip_norm=1.0):
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
        local_num_loss_tokens = loss_mask.sum().detach().to(torch.int)
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
                    (
                        "dp_cp"
                        if "dp_cp" in _mesh_resources.root_to_flatten_mapping.get(self.device_mesh, {})
                        else "data_parallel"
                    )
                ].get_group()
                if self.device_mesh is not None
                else None,
            )
            if not self.device_mesh or self.device_mesh["tensor_parallel"].size() == 1:
                grad_norm = clip_gradients(self.model, clip_norm)
            else:
                grad_norm = 0.0
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(1)
            t = time.perf_counter()
            time_delta = t - self.timestamp
            self.timestamp = t
            tps = self.num_nonpad_tokens / time_delta
            self.num_nonpad_tokens = 0
            reporting_loss = self.log_train_metrics(grad_norm, tps)
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info(
                "step %s | epoch %s | loss %.4f | grad_norm %.4f | lr %.2e | mem: %.2f GiB | tps %.2f",
                self.step_scheduler.step,
                self.step_scheduler.epoch,
                reporting_loss,
                grad_norm,
                current_lr,
                torch.cuda.max_memory_allocated() / 1024**3,
                tps,
            )
            torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def _run_validation_epoch(self):
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
                    batch["position_ids"] = torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).to(self.model.device)
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
            "[val] step %s | epoch %s | loss %.4f | lr %.2e",
            self.step_scheduler.step,
            self.step_scheduler.epoch,
            val_loss,
            current_lr,
        )

    def log_train_metrics(self, grad_norm, tps):
        if not self.device_mesh:
            dp_group = None
        elif self.device_mesh["context_parallel"].size() > 1:
            dp_group = self.device_mesh["dp_cp"].get_group()
        else:
            dp_group = self.device_mesh["data_parallel"].get_group()
        total_loss, total_num_loss_tokens = reduce_loss(
            self.forward_data_store, self.total_local_num_loss_tokens, per_token_loss=True, dp_group=dp_group
        )
        reporting_loss = (total_loss / total_num_loss_tokens).item()
        grad_norm = grad_norm.item() if not isinstance(grad_norm, float) else grad_norm
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
    trainer = PretrainRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
