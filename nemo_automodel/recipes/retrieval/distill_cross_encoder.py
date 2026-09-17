# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import dataclasses
import logging
import math
from contextlib import nullcontext

import torch
import torch.distributed
import torch.nn.functional as F

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loss.cross_encoder_distill import GroupedScoreKLLoss
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.recipes.retrieval.train_bi_encoder import _get_autocast_ctx, _unwrap_model_for_attrs
from nemo_automodel.recipes.retrieval.train_cross_encoder import TrainCrossEncoderRecipe, accuracy

logger = logging.getLogger(__name__)


class DistillCrossEncoderRecipe(TrainCrossEncoderRecipe):
    """Cross-encoder training with a frozen teacher distilled through grouped-score KL.

    Extends :class:`TrainCrossEncoderRecipe` without modifying it: the supervised grouped
    cross-entropy is kept as-is and blended with a KL term on the teacher's score
    distribution. Validation, checkpointing, gradient clipping, and LR scheduling are all
    inherited.
    """

    def setup(self):
        """Build the student through the parent recipe, then the frozen teacher and KD loss."""
        super().setup()

        self.loss_weights = {
            "ce": float(self.cfg.get("ce_loss_weight", 0.5)),
            "kd": float(self.cfg.get("kd_loss_weight", 0.5)),
        }
        # A negative weight would pass the >0 teacher-construction check below while failing the
        # ==0 skip in the training step, reaching a None teacher. Two zeros are accepted by both
        # and train on an identically zero loss. NaN fails every comparison, so it has to be
        # rejected explicitly or it slips through as a NaN loss on the first step.
        for name, value in self.loss_weights.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name}_loss_weight must be >= 0 and finite, got {value}")
        if max(self.loss_weights.values()) == 0.0:
            raise ValueError("ce_loss_weight and kd_loss_weight are both 0.0, which makes the loss constant")

        # The top-level VL config, not text_config: only the former is what the scoring head
        # divides by, and the two are set independently.
        student = _unwrap_model_for_attrs(self.model_parts[0])
        self.student_internal_temperature = float(student.config.temperature)
        # VL configs nest the vocabulary under text_config; text-only ones expose it directly.
        self.student_vocab_size = int(getattr(student.config, "text_config", student.config).vocab_size)

        # The teacher is only ever used to produce KD targets, so a zero KD weight makes it
        # dead weight: skip construction entirely rather than paying for an unused model.
        self.teacher_model = None
        self.teacher_internal_temperature = self.student_internal_temperature
        if self.loss_weights["kd"] > 0.0:
            self._build_teacher()

        kd_temperature = float(self.cfg.get("kd_temperature", 0.1))
        self.kd_loss = self.cfg.get("kd_loss", None)
        if self.kd_loss is not None:
            self.kd_loss = self.kd_loss.instantiate()
        else:
            self.kd_loss = GroupedScoreKLLoss(
                temperature=kd_temperature,
                reference_temperature=self.student_internal_temperature,
            )

        logger.info(
            "Cross-encoder distillation: ce_loss_weight=%.4g kd_loss_weight=%.4g kd_temperature=%.4g "
            "student_internal_temperature=%.4g teacher_internal_temperature=%.4g compensation=%.4g",
            self.loss_weights["ce"],
            self.loss_weights["kd"],
            kd_temperature,
            self.student_internal_temperature,
            self.teacher_internal_temperature,
            (kd_temperature / self.student_internal_temperature) ** 2,
        )

    def _build_teacher(self) -> None:
        """Instantiate the frozen teacher on the student's mesh with activation checkpointing off.

        The attribute name ``teacher_model`` is required: ``BaseRecipe`` skips that exact key
        when saving and restoring tracked model state, so any other name would make the
        teacher shadow the student in checkpoints.
        """
        # reshard_after_forward is left at the config's value so the YAML controls both models.
        # Note that a teacher never runs backward, so nothing releases its gathered parameters:
        # false keeps the whole teacher unsharded for the run, trading memory for one all-gather
        # per forward. Activation checkpointing is read off the strategy config by the FSDP2
        # manager and off the setup elsewhere, so it has to be cleared in both places; with no
        # backward there are no activations to trade compute for. dataclasses.replace keeps the
        # student's mesh_context, so both models shard over the same device mesh.
        strategy = self.distributed_setup.strategy_config
        if strategy is not None:
            strategy = dataclasses.replace(strategy, activation_checkpointing=False)
        teacher_setup = dataclasses.replace(
            self.distributed_setup,
            strategy_config=strategy,
            activation_checkpointing=False,
        )

        # super().setup() ends in load_checkpoint(), so the restored RNG state is live here;
        # building the teacher outside a scoped RNG would shift the student's dropout stream.
        with ScopedRNG(seed=self.cfg.get("seed", 42), ranked=True):
            self.teacher_model = self.cfg.teacher_model.instantiate(
                distributed_setup=teacher_setup,
                peft_config=None,
            )
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

        teacher = _unwrap_model_for_attrs(self.teacher_model)
        # One tokenization feeds both models, so a vocabulary mismatch silently scores different text.
        teacher_vocab_size = int(getattr(teacher.config, "text_config", teacher.config).vocab_size)
        if teacher_vocab_size != self.student_vocab_size:
            raise ValueError(
                f"Teacher vocab_size {teacher_vocab_size} does not match student {self.student_vocab_size}; "
                "both models must consume the same tokenized batch."
            )

        self.teacher_internal_temperature = float(teacher.config.temperature)
        if self.teacher_internal_temperature != self.student_internal_temperature:
            logger.warning(
                "Student and teacher config.temperature differ (%.4g vs %.4g). Each is applied to its "
                "own logits, so raw scores stay correct, but the two models sharpen differently.",
                self.student_internal_temperature,
                self.teacher_internal_temperature,
            )

    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        loss_buffer,
        num_batches,
        is_train: bool = True,
        modality_loss_buffers: dict[str, list[torch.Tensor]] | None = None,
    ):
        """Forward and backward pass blending grouped cross-entropy with teacher KL.

        Args:
            idx: Index of this micro-batch within the gradient accumulation window.
            batch: Collated cross-encoder batch of flattened query-passage pairs.
            loss_buffer: Accumulator the blended scalar loss is appended to.
            num_batches: Number of micro-batches in the accumulation window.
            is_train: Whether to run the backward pass and gradient sync.
            modality_loss_buffers: Accepted for compatibility with
                ``TrainBiEncoderRecipe._run_train_optim_step``, which always forwards it.
                Unused here, as in the parent recipe.
        """
        batch = {
            k: v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        labels = batch.pop("labels")

        model = self.model_parts[0]
        train_ctx = _get_autocast_ctx(self.distributed_config)
        sync_ctx = (
            get_sync_ctx(
                model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )
        with train_ctx, sync_ctx:
            # Both losses need per-query groups; softmax over the flattened [B*P, 1] logits
            # would be identically 1.0 and silently reduce every loss to zero. Upcast once here:
            # the backbone emits bf16 logits already divided by config.temperature, so at
            # temperature 0.02 they sit near magnitude 260 where the bf16 grid spacing is 2, and
            # both softmaxes below would otherwise inherit that coarseness.
            student_logits = model(**batch, return_dict=True).logits.view(-1, self.train_n_passages).float()

            # Apply the recipe-level temperature exactly as TrainCrossEncoderRecipe does, so
            # ce_loss_weight 1.0 reproduces plain SFT for any config rather than only for configs
            # that leave it at 1.0. KD deliberately does not see this factor: it works on raw scores.
            loss_ce = F.cross_entropy(student_logits / self.temperature, labels)

            # Skip the teacher when KD weight is 0 so we don't pay for an unused forward.
            if self.loss_weights["kd"] == 0.0:
                loss_kd = student_logits.new_zeros(())
            else:
                # force_dummy_vision keeps every rank issuing the vision tower's all-gather on
                # text-only batches. The student gets that for free from run_dummy_vision because
                # it is in train mode; the teacher is in eval mode, and _build_teacher sets
                # reshard_after_forward=True, so it re-gathers every forward and would otherwise
                # deadlock as soon as one rank has images and another does not.
                with torch.no_grad():
                    teacher_logits = (
                        self.teacher_model(**batch, return_dict=True, force_dummy_vision=True)
                        .logits.view(-1, self.train_n_passages)
                        .float()
                    )
                # Recover raw scores by undoing each model's own internal division.
                loss_kd = self.kd_loss(
                    student_logits * self.student_internal_temperature,
                    teacher_logits * self.teacher_internal_temperature,
                )

            # TODO: add a score-level MSE term (Jina's L_score) as a third weighted component.
            # KL matches ranking within a group, not the teacher's actual score values.
            # CE only pushes the labeled positive to the top. Neither copies the teacher's scale.
            loss = self.loss_weights["ce"] * loss_ce + self.loss_weights["kd"] * loss_kd

            loss_buffer.append(loss.clone().detach())
            self._acc_buffer.append(accuracy(student_logits, labels))
            self._loss_component_buffer.append(
                {"loss_ce": loss_ce.detach().float(), "loss_kd": loss_kd.detach().float()}
            )

            if is_train:
                # Scale loss by number of gradient accumulation steps to get correct average gradients
                # FSDP/DDP will handle averaging across DP ranks automatically
                scaled_loss = loss / num_batches
                scaled_loss.backward()

    def _run_train_optim_step(self, batches, max_grad_norm=None):
        """Run the inherited optimizer step, then attach the per-component loss metrics."""
        self._loss_component_buffer = []
        result = super()._run_train_optim_step(batches, max_grad_norm)

        dp_group_size = self._get_dp_group_size(include_cp=True)
        for key in ("loss_ce", "loss_kd"):
            value = torch.stack([item[key] for item in self._loss_component_buffer]).mean()
            if torch.distributed.is_initialized():
                value = self._dp_allreduce(value, include_cp=True) / dp_group_size
            result.metrics[key] = value.item()
        return result


def main(default_config_path="examples/retrieval/distillation/ministral3_vl_ranker_distill.yaml"):
    """Entry point: load config, build the cross-encoder distillation recipe, and train."""
    cfg = parse_args_and_load_config(default_config_path)
    recipe = DistillCrossEncoderRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
