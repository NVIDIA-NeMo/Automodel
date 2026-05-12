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

"""Knowledge Distillation recipe for Vision-Language Models with NeMo AutoModel.

This recipe fine-tunes a *student* VLM using the logits of a frozen *teacher* VLM. It
extends ``FinetuneRecipeForVLM`` adding:

1. teacher_model — an additional VLM loaded in ``eval`` mode
2. kd_loss_fn    — KL-divergence between temperature-scaled distributions
3. kd_ratio      — linear mix between CE loss and KD loss

The training loop preserves all VLM-specific input handling (pixel_values, image_grid_thw,
etc.) and passes multimodal inputs to both teacher and student models.

The loss becomes:
    loss = (1-kd_ratio) * ce_loss + kd_ratio * kd_loss

Pipeline parallelism is not supported in this recipe.

The file exposes ``KnowledgeDistillationRecipeForVLM`` and a ``main`` entry-point
so it can be launched exactly the same way as other recipes:

    python -m torch.distributed.run --nproc-per-node=8 \\
        nemo_automodel/recipes/vlm/kd.py \\
        -c examples/vlm_kd/qwen3_5/qwen3_5_vl_4b_kd.yaml
"""

from __future__ import annotations

import logging
import pathlib
import time
from contextlib import nullcontext
from typing import Optional

import torch
import wandb
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.training.utils import count_tail_padding
from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS, filter_forward_kwargs
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM, build_model, calculate_loss

logger = logging.getLogger(__name__)


def _build_kd_loss_fn(cfg_kd):
    if cfg_kd is None:
        logger.info("No KD loss function provided, using KLDivLoss")
        return torch.nn.KLDivLoss(reduction="batchmean")
    return cfg_kd.instantiate()


def _build_teacher_model(
    cfg_teacher,
    cfg_freeze,
    seed: int,
    device_mesh=None,
    moe_mesh=None,
    distributed_config=None,
    device=None,
) -> torch.nn.Module:
    """Build and initialize the teacher VLM for knowledge distillation.

    Uses the same ``build_model`` as the student but without PEFT, FP8, or QAT
    since the teacher should be frozen in full precision.

    Args:
        cfg_teacher: Configuration for teacher model instantiation.
        cfg_freeze: Freeze configuration for the teacher model.
        seed: Random seed for reproducibility.
        device_mesh: Device mesh for distributed training.
        moe_mesh: MOE mesh for expert parallelism.
        distributed_config: Strategy-specific distributed config.
        device: Device to place the teacher model on.

    Returns:
        The frozen teacher model ready for inference.
    """
    assert cfg_teacher is not None, "`teacher_model` section missing from YAML config"
    logger.info("Instantiating teacher VLM model")

    teacher_model = build_model(
        cfg_teacher,
        cfg_freeze=cfg_freeze,
        cfg_peft=None,
        seed=seed,
        cfg_fp8=None,
        cfg_compile=None,
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        distributed_config=distributed_config,
        pipeline_config=None,
        cfg_moe=None,
        activation_checkpointing=False,
    )

    if device is not None:
        teacher_model = teacher_model.to(device)

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    return teacher_model


def _verify_tokenizer_compatibility(student_cfg, teacher_cfg, trust_remote_code=True):
    if student_cfg is None or teacher_cfg is None:
        raise ValueError("Student and teacher model configs are required")
    student_name = student_cfg.get("pretrained_model_name_or_path", None)
    teacher_name = teacher_cfg.get("pretrained_model_name_or_path", None)
    if student_name is None or teacher_name is None:
        raise ValueError("Both student and teacher must specify pretrained_model_name_or_path")
    student_tokenizer = NeMoAutoTokenizer.from_pretrained(student_name, trust_remote_code=trust_remote_code)
    teacher_tokenizer = NeMoAutoTokenizer.from_pretrained(teacher_name, trust_remote_code=trust_remote_code)
    if student_tokenizer.vocab_size != teacher_tokenizer.vocab_size:
        raise ValueError(
            "Student and teacher tokenizers have different vocab sizes; support will be added in the future"
        )
    if student_tokenizer.pad_token != teacher_tokenizer.pad_token:
        raise ValueError("Student and teacher tokenizers have different pad tokens")
    del student_tokenizer, teacher_tokenizer


class KnowledgeDistillationRecipeForVLM(FinetuneRecipeForVLM):
    """Fine-tune a student VLM via knowledge distillation from a teacher VLM."""

    def setup(self):
        """Build student & teacher, dataloaders, optimizers, etc."""
        _verify_tokenizer_compatibility(self.cfg.get("model", None), self.cfg.get("teacher_model", None))

        super().setup()

        if self.pp_enabled:
            raise NotImplementedError("Pipeline parallelism is not supported for VLM knowledge distillation yet.")

        self.teacher_model = _build_teacher_model(
            cfg_teacher=self.cfg.get("teacher_model", None),
            cfg_freeze=self.cfg.get("teacher_freeze_config", None),
            seed=self.cfg.get("seed", 42),
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            distributed_config=self.distributed_config,
            device=self.dist_env.device,
        )

        logger.info("Teacher Model: " + str(self.teacher_model))

        self.kd_loss_fn = _build_kd_loss_fn(self.cfg.get("kd_loss_fn", None))
        self.kd_ratio: float = float(self.cfg.get("kd_ratio", 0.5))
        logger.info("KD Loss config: " + str(self.cfg.get("kd_loss_fn", None)))
        temperature = getattr(self.kd_loss_fn, "temperature", "N/A")
        logger.info(f"Knowledge-distillation enabled: ratio={self.kd_ratio}, T={temperature}")

        self._kd_loss_buffer: list[torch.Tensor] = []
        self._ce_loss_buffer: list[torch.Tensor] = []

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
        """Override the forward backward step to include knowledge distillation loss."""
        batch = {
            k: (
                {dk: dv.to(self.dist_env.device, non_blocking=True) if dv is not None else None for dk, dv in v.items()}
                if isinstance(v, dict)
                else (v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            )
            for k, v in batch.items()
        }

        _model = self.model_parts[0]
        _cp_active = (
            self.device_mesh is not None
            and "cp" in getattr(self.device_mesh, "mesh_dim_names", ())
            and self.device_mesh["cp"].size() > 1
            and not self.pp_enabled
        )
        if _cp_active and hasattr(_model, "prepare_model_inputs_for_cp"):
            mm_kwargs = {k: batch[k] for k in VLM_INPUT_KEYS if batch.get(k) is not None}
            with torch.no_grad():
                prepared = _model(_pre_embed_only=True, **mm_kwargs)
            for k in VLM_INPUT_KEYS:
                batch.pop(k, None)
            batch.update(prepared)

        train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch)
        labels = batch.pop("labels")

        model = self.model_parts[0]
        sync_ctx = (
            get_sync_ctx(
                model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )
        with sync_ctx, train_ctx():
            # Teacher forward (no grad) — free intermediates immediately.
            with torch.no_grad():
                teacher_batch = filter_forward_kwargs(self.teacher_model, batch)
                teacher_out = self.teacher_model(**teacher_batch)
                teacher_logits = getattr(teacher_out, "logits", teacher_out).detach()
                del teacher_out, teacher_batch

            # Student forward.
            student_batch = filter_forward_kwargs(model, batch)
            student_keep_last = isinstance(self.loss_fn, FusedLinearCrossEntropy)
            if student_keep_last:
                student_out = model(logits_to_keep=1, **student_batch)
            else:
                student_out = model(**student_batch)
            del student_batch

            student_logits = getattr(student_out, "logits", student_out)
            hidden_states = (
                student_out.hidden_states[-1] if getattr(student_out, "hidden_states", None) is not None else None
            )
            del student_out

            # CE loss (skip when kd_ratio >= 1.0).
            if self.kd_ratio >= 1.0:
                ce_loss = student_logits.new_tensor(0.0, dtype=student_logits.dtype)
            else:
                ce_loss = calculate_loss(
                    self.loss_fn,
                    logits=student_logits,
                    labels=labels,
                    model=model,
                    hidden_states=hidden_states,
                    num_label_tokens=num_label_tokens,
                )
            del hidden_states

            kd_loss = self.kd_loss_fn(
                student_logits,
                teacher_logits,
                labels,
                num_batch_labels=num_label_tokens,
            )
            del teacher_logits

            local_loss = (1.0 - self.kd_ratio) * ce_loss + self.kd_ratio * kd_loss
            loss_buffer.append(local_loss.detach().clone())
            self._ce_loss_buffer.append(ce_loss.detach().clone())
            self._kd_loss_buffer.append(kd_loss.detach().clone())
            if is_train:
                (local_loss * self._get_dp_group_size(include_cp=True)).backward()

    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        """Execute a single training step with KD loss tracking."""
        num_label_tokens = torch.tensor(
            sum((batch["labels"] != -100).sum().item() for batch in batches), dtype=torch.long
        )
        num_label_tokens = self._dp_allreduce(num_label_tokens).item()

        loss_buffer: list[torch.Tensor] = []

        num_tokens_in_batch = torch.tensor(
            sum(batch["labels"].numel() - count_tail_padding(batch["labels"]) for batch in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()

        num_batches = len(batches)
        for i, batch in enumerate(batches):
            self._forward_backward_step(
                i, batch, loss_buffer=loss_buffer, num_label_tokens=num_label_tokens, num_batches=num_batches
            )

        grad_norm = 0
        if max_grad_norm is not None:
            if not self.device_mesh or self.device_mesh["tp"].size() == 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model_parts[0].parameters() if p.requires_grad], max_grad_norm
                )
                if hasattr(grad_norm, "full_tensor"):
                    grad_norm = grad_norm.full_tensor()

            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()

        self.checkpointer.maybe_wait_for_staging()
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()

        if self.lr_scheduler is not None:
            for scheduler in self.lr_scheduler:
                scheduler.step(1)

        fp8_config = self.cfg.get("fp8", None)
        if (
            fp8_config is not None
            and fp8_config.get("enabled", False)
            and fp8_config.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model_parts[0])

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta
        reporting_loss = torch.sum(torch.stack(loss_buffer))
        reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
        reporting_loss = reporting_loss.cpu().item()

        ce_loss = self._dp_allreduce(torch.stack(self._ce_loss_buffer).sum(), include_cp=True).item()
        kd_loss = self._dp_allreduce(torch.stack(self._kd_loss_buffer).sum(), include_cp=True).item()
        self._ce_loss_buffer.clear()
        self._kd_loss_buffer.clear()

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": reporting_loss,
                "ce_loss": ce_loss,
                "kd_loss": kd_loss,
                "grad_norm": grad_norm,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / max(self._get_dp_group_size(), 1),
                "num_tokens_per_step": num_tokens_in_batch,
                "num_label_tokens": num_label_tokens,
                "kd_ratio": self.kd_ratio,
                "temperature": getattr(self.kd_loss_fn, "temperature", float("nan")),
            },
        )

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one validation pass with KD loss computation."""
        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()

            total_loss = 0.0
            total_ce_loss = 0.0
            total_kd_loss = 0.0
            total_num_label_tokens = 0
            loss_buffer: list[torch.Tensor] = []

            for batch in val_dataloader:
                num_label_tokens = (batch["labels"] != -100).sum().item()
                self._forward_backward_step(
                    0,
                    batch,
                    loss_buffer=loss_buffer,
                    num_label_tokens=num_label_tokens,
                    num_batches=1,
                    is_train=False,
                )
                # _forward_backward_step produces per-token-averaged losses.
                # Multiply back by num_label_tokens to get the sum for weighted averaging.
                total_loss += loss_buffer[-1].item() * num_label_tokens
                total_ce_loss += self._ce_loss_buffer[-1].item() * num_label_tokens
                total_kd_loss += self._kd_loss_buffer[-1].item() * num_label_tokens
                total_num_label_tokens += num_label_tokens

            self._ce_loss_buffer.clear()
            self._kd_loss_buffer.clear()

        total_loss = self._dp_allreduce(
            torch.tensor(total_loss, dtype=torch.float32, device=self.dist_env.device), include_cp=True
        ).item()
        total_ce_loss = self._dp_allreduce(
            torch.tensor(total_ce_loss, dtype=torch.float32, device=self.dist_env.device), include_cp=True
        ).item()
        total_kd_loss = self._dp_allreduce(
            torch.tensor(total_kd_loss, dtype=torch.float32, device=self.dist_env.device), include_cp=True
        ).item()
        total_num_label_tokens = self._dp_allreduce(torch.tensor(total_num_label_tokens, dtype=torch.long)).item()

        val_loss = total_loss / max(total_num_label_tokens, 1e-8)
        val_ce_loss = total_ce_loss / max(total_num_label_tokens, 1e-8)
        val_kd_loss = total_kd_loss / max(total_num_label_tokens, 1e-8)
        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": val_loss,
                "ce_loss": val_ce_loss,
                "kd_loss": val_kd_loss,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "num_label_tokens": total_num_label_tokens,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )

    def log_val_metrics(self, log_data):
        if not self.dist_env.is_main or log_data is None:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=log_data.step)

        self.metric_logger_valid.log(log_data)

        if self.kd_ratio >= 1.0:
            logging.info(
                "[val] step {} | epoch {} | loss {:.4f} | kd_loss {:.4f} | lr {:.2e} | num_label_tokens {}".format(
                    log_data.step,
                    log_data.epoch,
                    log_data.metrics["val_loss"],
                    log_data.metrics["kd_loss"],
                    log_data.metrics["lr"],
                    log_data.metrics["num_label_tokens"],
                )
            )
        else:
            logging.info(
                "[val] step {} | epoch {} | loss {:.4f} | ce_loss {:.4f} | kd_loss {:.4f} | lr {:.2e} | num_label_tokens {}".format(
                    log_data.step,
                    log_data.epoch,
                    log_data.metrics["val_loss"],
                    log_data.metrics["ce_loss"],
                    log_data.metrics["kd_loss"],
                    log_data.metrics["lr"],
                    log_data.metrics["num_label_tokens"],
                )
            )

    def log_train_metrics(self, log_data) -> float:
        if not self.dist_env.is_main:
            return

        if self.step_scheduler.is_remote_logging_step:
            if wandb.run is not None:
                wandb.log(log_data.to_dict(), step=log_data.step)

        self.metric_logger_train.log(log_data)

        if self.kd_ratio >= 1.0:
            logging.info(
                "step {} | epoch {} | "
                "loss {:.4f} | kd_loss {:.4f} | "
                "lr {:.2e} | mem {:.2f} GiB | tps {:.2f} | kd_ratio {:.2f} | temperature {:.2f}".format(
                    log_data.step,
                    log_data.epoch,
                    log_data.metrics["loss"],
                    log_data.metrics["kd_loss"],
                    log_data.metrics["lr"],
                    log_data.metrics["mem"],
                    log_data.metrics["tps"],
                    log_data.metrics["kd_ratio"],
                    log_data.metrics["temperature"],
                )
            )
        else:
            logging.info(
                "step {} | epoch {} | "
                "loss {:.4f} | ce_loss {:.4f} | kd_loss {:.4f} | "
                "lr {:.2e} | mem {:.2f} GiB | tps {:.2f} | kd_ratio {:.2f} | temperature {:.2f}".format(
                    log_data.step,
                    log_data.epoch,
                    log_data.metrics["loss"],
                    log_data.metrics["ce_loss"],
                    log_data.metrics["kd_loss"],
                    log_data.metrics["lr"],
                    log_data.metrics["mem"],
                    log_data.metrics["tps"],
                    log_data.metrics["kd_ratio"],
                    log_data.metrics["temperature"],
                )
            )
        torch.cuda.reset_peak_memory_stats()


def main(config_path=None):
    """Run the VLM KD recipe from CLI or directly."""
    if config_path is None:
        config_path = (
            pathlib.Path(__file__).parent.resolve().parent.parent
            / "examples"
            / "vlm_kd"
            / "qwen3_5"
            / "qwen3_5_vl_4b_kd.yaml"
        )
    cfg = parse_args_and_load_config(config_path)
    trainer = KnowledgeDistillationRecipeForVLM(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":  # pragma: no cover
    main()
