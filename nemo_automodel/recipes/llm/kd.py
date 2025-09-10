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

"""Knowledge Distillation recipe for next-token prediction with NeMo-AutoModel.

This recipe fine-tunes a *student* model using the logits of a frozen *teacher* model. It
extends ``FinetuneRecipeForNextTokenPrediction`` adding:

1. teacher_model – an additional HF/NeMo model loaded in ``eval`` mode
2. kd_loss_fn     – KL-divergence between temperature-scaled distributions
3. kd_ratio       – linear mix between CE loss and KD loss

The training loop is copied from the parent class but the loss becomes:
    loss = (1-kd_ratio) * ce_loss + kd_ratio * kd_loss

The file exposes ``KnowledgeDistillationRecipeForNextTokenPrediction`` and a
``main`` entry-point so it can be launched exactly the same way as other
recipes:

    python -m torch.distributed.run --nproc-per-node=8 \
        nemo_automodel/recipes/llm/knowledge_distillation.py \
        -c examples/llm/llama_3_2_1b_kd.yaml
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import wandb
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from transformers import AutoTokenizer

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.training.utils import count_tail_padding
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    calculate_loss,
)

logger = logging.getLogger(__name__)


def _build_kd_loss_fn(cfg_kd):
    if cfg_kd is None:
        logger.info("No KD loss function provided, using KLDivLoss")
        return torch.nn.KLDivLoss(reduction="batchmean")
    return cfg_kd.instantiate()


def _build_teacher_model(cfg_teacher, seed, has_packed_sequence, device, model_wrapper, device_mesh):
    assert cfg_teacher is not None, "`teacher_model` section missing from YAML config"
    logger.info("Instantiating teacher model")

    # Build teacher model using the same approach as student model
    with ScopedRNG(seed=seed, ranked=True):
        kwargs: Dict[str, Any] = {}
        if has_packed_sequence > 0:
            kwargs["attn_implementation"] = "flash_attention_2"

        teacher_model = cfg_teacher.instantiate(**kwargs)

        # For teacher model, we'll apply FSDP2 sharding if the same model_wrapper is available
        # but we need to be careful about device placement and parallelization
        if model_wrapper is not None and hasattr(model_wrapper, "parallelize"):
            logger.info("Applying FSDP2 sharding to teacher model")
            # Create a new model wrapper instance for the teacher to avoid conflicts
            # with the student model's parallelization
            try:
                teacher_model = model_wrapper.parallelize(teacher_model)
            except Exception as e:
                logger.warning(f"Failed to parallelize teacher model with FSDP2: {e}")
                logger.info("Falling back to simple device placement for teacher model")
                teacher_model = teacher_model.to(device)
        else:
            teacher_model = teacher_model.to(device)

        # Set teacher to eval mode and freeze parameters
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        return teacher_model


def _verify_tokenizer_compatibility(student_cfg, teacher_cfg, trust_remote_code=True):
    if student_cfg is None or teacher_cfg is None:
        raise ValueError("Student and teacher model configs are required")
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_cfg.pretrained_model_name_or_path, trust_remote_code=trust_remote_code
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_cfg.pretrained_model_name_or_path, trust_remote_code=trust_remote_code
    )
    if student_tokenizer.vocab_size != teacher_tokenizer.vocab_size:
        raise ValueError(
            "Student and teacher tokenizers have different vocab sizes; Support will be added in the future"
        )
    if student_tokenizer.pad_token != teacher_tokenizer.pad_token:
        raise ValueError("Student and teacher tokenizers have different pad tokens")
    del student_tokenizer, teacher_tokenizer


class KnowledgeDistillationRecipeForNextTokenPrediction(TrainFinetuneRecipeForNextTokenPrediction):
    """Fine-tune a student model via knowledge distillation."""

    def setup(self):  # noqa: C901 – same complexity as parent
        """Build student & teacher, dataloaders, optimizers, etc."""
        # Right now, we only support tokenizer compatibility for the same tokenizer.
        # We will add support for different tokenizers in the future.
        _verify_tokenizer_compatibility(self.cfg.get("model", None), self.cfg.get("teacher_model", None))

        # Let the parent class build *everything* for the student first
        super().setup()
        if self.pp_enabled:
            raise ValueError("Pipeline parallelism support will be added in the future for knowledge distillation")

        # teacher specific
        self.teacher_model = _build_teacher_model(
            self.cfg.get("teacher_model", None),
            self.cfg.get("seed", 42),
            self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0,
            self.dist_env.device,
            self.model_wrapper,
            self.device_mesh,
        )
        logger.info("Teacher Model: " + str(self.teacher_model))
        # KD
        self.kd_loss_fn = _build_kd_loss_fn(self.cfg.get("kd_loss_fn", None))
        self.kd_ratio: float = float(self.cfg.get("kd_ratio", 0.5))
        logger.info("KD Loss config: " + str(self.cfg.get("kd_loss_fn", None)))
        logger.info(f"Knowledge-distillation enabled: ratio={self.kd_ratio}, T={self.kd_loss_fn.temperature}")

        # Buffers for logging
        self._kd_loss_buffer = []
        self._ce_loss_buffer = []

    #  Override the forward backward step to inject KD loss
    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        num_label_tokens,
        num_batches,
        is_train: bool = True,
    ):
        """Override the forward backward step to include knowledge distillation loss."""
        batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels)

        model = self.model_parts[0]
        sync_ctx = get_sync_ctx(model, idx == num_batches - 1) if is_train else nullcontext()
        with train_ctx(), sync_ctx:
            # Student forward
            student_keep_last = isinstance(self.loss_fn, FusedLinearCrossEntropy)
            if student_keep_last:
                # Student forward keeping only last token logits to match loss_fn
                student_out = model(logits_to_keep=1, **batch)
            else:
                student_out = model(**batch)

            student_logits = getattr(student_out, "logits", student_out)  # shape (B, S, V)
            # Cross-entropy loss against true labels (same as parent)
            ce_loss = calculate_loss(
                self.loss_fn,
                logits=student_logits,
                labels=labels,
                model=model,
                hidden_states=student_out.hidden_states[-1] if "hidden_states" in student_out else None,
                num_label_tokens=num_label_tokens,
            )
            # No grad for teacher forward
            with torch.no_grad():
                teacher_logits = self.teacher_model(**batch)
                teacher_logits = getattr(teacher_logits, "logits", teacher_logits).detach()

            # Reminder: kd_loss is normalized by num_label_tokens,
            # which typically is larger than the number of labels in this batch,
            # because it contains the total number of labels for all batches contained
            # in one optimization step (grad_acc_steps = gbs / mbs).
            kd_loss = self.kd_loss_fn(
                student_logits,
                teacher_logits,
                labels,
                num_batch_labels=num_label_tokens,
            )
            local_loss = (1.0 - self.kd_ratio) * ce_loss + self.kd_ratio * kd_loss
            if is_train:
                (local_loss * self._get_dp_group_size()).backward()
            # return the losses for logging
            return local_loss.clone().detach(), kd_loss.clone().detach(), ce_loss.clone().detach()

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
        for i, batch in enumerate(batches):
            local_loss, kd_loss, ce_loss = self._forward_backward_step(
                i, batch, num_label_tokens=num_label_tokens, num_batches=num_batches
            )
            loss_buffer.append(local_loss)
            self._ce_loss_buffer.append(ce_loss)
            self._kd_loss_buffer.append(kd_loss)

        grad_norm = 0
        # Clip gradients **after** any rescaling.
        # TODO(@boxiangw): Fix TP gradient clipping
        if max_grad_norm is not None:
            if not self.device_mesh or self.device_mesh["tp"].size() == 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model_parts[0].parameters() if p.requires_grad], max_grad_norm
                )
                if hasattr(grad_norm, "full_tensor"):
                    grad_norm = grad_norm.full_tensor()  # collect the summed grad norm across ranks

            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()

        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()

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
        reporting_loss = self._dp_allreduce(reporting_loss)
        reporting_loss = reporting_loss.cpu().item()
        # fix reporting_loss, tps across ranks
        return reporting_loss, grad_norm, tps, num_tokens_in_batch, num_label_tokens

    def log_train_metrics(
        self, train_loss, grad_norm, num_tokens_in_batch, tps, num_label_tokens, kd_loss=None
    ) -> float:
        """Override to log KD-specific metrics."""
        # Calculate average CE and KD losses from the buffers
        if len(getattr(self, "_ce_loss_buffer", [])) > 0:
            ce_loss = self._dp_allreduce(torch.stack(self._ce_loss_buffer).sum()).item()
            kd_loss = self._dp_allreduce(torch.stack(self._kd_loss_buffer).sum()).item()
            # Clear buffers for next step
            self._ce_loss_buffer.clear()
            self._kd_loss_buffer.clear()
        else:
            ce_loss = 0.0
            kd_loss = 0.0

        # assumes all model parts' optimizers have the same learning rate
        current_lr = self.optimizer[0].param_groups[0]["lr"]
        # log_data
        if wandb.run is not None:
            log_data = {
                "step": self.step_scheduler.step,
                "epoch": self.step_scheduler.epoch,
                "train_loss": train_loss,
                "ce_loss": ce_loss,
                "kd_loss": kd_loss,
                "grad_norm": grad_norm,
                "num_tokens_per_step": num_tokens_in_batch,
                "tps": tps,
                "tps_per_gpu": tps / self._get_dp_group_size(),
                "learning_rate": current_lr,
            }
            wandb.log(log_data, step=self.step_scheduler.step)

        logging.info(
            "step {} | epoch {} | "
            "loss {:.4f} | ce_loss {:.4f} | kd_loss {:.4f} | "
            "lr {:.2e} | tps {:.2f} | kd_ratio {:.2f} | temperature {:.2f}".format(
                self.step_scheduler.step,
                self.step_scheduler.epoch,
                train_loss,
                ce_loss,
                kd_loss,
                current_lr,
                tps,
                self.kd_ratio,
                self.kd_loss_fn.temperature,
            )
        )
        torch.cuda.reset_peak_memory_stats()


# Entry point
def main(config_path="examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml"):
    """Run the KD recipe from CLI or directly."""
    cfg = parse_args_and_load_config(config_path)
    trainer = KnowledgeDistillationRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":  # pragma: no cover
    main()
