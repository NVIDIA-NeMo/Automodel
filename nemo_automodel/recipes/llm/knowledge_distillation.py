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
import pathlib
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import _mesh_resources

from nemo_automodel.components.training.utils import (
    clip_gradients,
    count_tail_padding,
    get_sync_ctx,
    reduce_loss,
    rescale_gradients,
)
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.training.rng import StatefulRNG

from nemo_automodel.recipes.llm.finetune import (
    FinetuneRecipeForNextTokenPrediction,
    build_dataloader,
    build_distributed,
    build_loss_fn,
    build_lr_scheduler,
    build_step_scheduler,
    build_wandb,
    calculate_loss,
)
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

logger = logging.getLogger(__name__)


class KnowledgeDistillationRecipeForNextTokenPrediction(FinetuneRecipeForNextTokenPrediction):
    """Fine-tune a student model via knowledge distillation."""

    def setup(self):  # noqa: C901 – same complexity as parent
        """Build student & teacher, dataloaders, optimizers, etc."""

        # Let the parent class build *everything* for the student first
        super().setup()

        # ---------------- teacher specific ----------------
        cfg_teacher = self.cfg.teacher_model
        assert cfg_teacher is not None, "`teacher_model` section missing from YAML config"

        # We only need a frozen teacher – no optimizer / grad / parallelism
        logger.info("Instantiating teacher model …")
        with StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True):
            kwargs: Dict[str, Any] = {}
            if self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0:
                kwargs["attn_implementation"] = "flash_attention_2"
            self.teacher_model = cfg_teacher.instantiate(**kwargs)
        self.teacher_model.to(self.dist_env.device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

        # ---------------- KD hyper-params ----------------
        self.kd_ratio: float = float(self.cfg.get("kd_ratio", 0.5))
        self.temperature: float = float(self.cfg.get("temperature", 1.0))
        if kd_cfg := self.cfg.get("kd_loss_fn", None):
            self.kd_loss_fn = build_loss_fn(kd_cfg)
        else:
            # Default to KLDiv if nothing provided.
            self.kd_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

        logger.info(
            f"Knowledge-distillation enabled – ratio={self.kd_ratio}, T={self.temperature}"
        )

    # ---------------------------------------------------------------------
    #  Override the train step to inject KD loss
    # ---------------------------------------------------------------------
    def _run_train_step(self, batch, is_optim_step, clip_norm: float = 1.0):  # noqa: C901 – mirrors parent
        self.model.train()

        # Move tensors to device
        batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        loss_mask = batch.pop("loss_mask", None)
        if loss_mask is None:
            loss_mask = (labels.detach() != -100).to(torch.int)

        if (
            "position_ids" not in batch
            and self.device_mesh is not None
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
            # --- Student forward ---
            if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                student_out = self.model(logits_to_keep=1, **batch)
            else:
                student_out = self.model(**batch)

            student_logits = student_out.logits  # shape (B, S, V)

            # --- Teacher forward (no grad) ---
            with torch.no_grad():
                teacher_logits = self.teacher_model(**batch).logits.detach()

            # Cross-entropy loss against true labels (same as parent)
            ce_loss = calculate_loss(
                self.loss_fn,
                logits=student_logits,
                labels=labels,
                mask=loss_mask,
                model=self.model,
                hidden_states=student_out.hidden_states[-1] if "hidden_states" in student_out else None,
            )

            # KL-divergence between softened distributions
            s_logits = F.log_softmax(student_logits / self.temperature, dim=-1)
            t_logits = F.softmax(teacher_logits / self.temperature, dim=-1)
            kd_loss = self.kd_loss_fn(s_logits, t_logits) * (self.temperature ** 2)

            local_loss = (1.0 - self.kd_ratio) * ce_loss + self.kd_ratio * kd_loss

        # --- accounting / backward identical to parent implementation ---
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
                        "dp_cp" if "dp_cp" in _mesh_resources.root_to_flatten_mapping.get(self.device_mesh, {}) else "data_parallel"
                    )
                ].size()
                if self.device_mesh
                else 1,
            )
            grad_norm = clip_gradients(self.model.trainable_parameters, clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Logging ------------------------------
            tps = self.log_train_metrics(
                grad_norm,
                self.num_nonpad_tokens / max((time.perf_counter() - self.timestamp), 1e-8),
            )
            if self.dist_env.is_main:
                logger.info(
                    f"[train] step {self.step_scheduler.step} | epoch {self.step_scheduler.epoch} | "
                    f"loss {local_loss.item():.4f} | ce {ce_loss.item():.4f} | kd {kd_loss.item():.4f} | "
                    f"lr {self.optimizer.param_groups[0]['lr']:.2e} | tps {tps:.2f}"
                )

            # Reset counters for next optim step
            self.num_nonpad_tokens = 0
            self.timestamp = time.perf_counter()

    # ------------------------------------------------------------------
    # Validation uses the parent implementation (only CE loss) – acceptable
    # for quick reference. Could be overridden to include kd_loss if needed.
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path=None):  # noqa: D401 – simple wrapper
    """Run the KD recipe from CLI or directly."""
    if config_path is None:
        config_path = pathlib.Path(__file__).parent / "llama_3_2_1b_kd.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = KnowledgeDistillationRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":  # pragma: no cover
    main() 