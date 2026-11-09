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

"""DSpine draft-model training recipe (Qwen3-style targets).

DSpine (Tencent, https://github.com/Tencent/DSpine) keeps DFlash's one-pass
parallel block draft, makes the in-block attention causal, and after every draft
layer injects each position's predecessor code into it, so the left-to-right
dependency is carried through the depth of the network while all positions are
still drafted in parallel. Decoding rescores each position's top-k candidates by
redoing the last injection with the exact previously chosen token. See
``nemo_automodel.components.speculative.dflash.draft_qwen3_dspine`` and
``nemo_automodel.components.speculative.dflash.dspine_core``.

This recipe reuses every piece of the DFlash recipe -- online target hidden-state
capture, anchor sampling, the block attention mask (in its causal form),
gradient accumulation, and checkpointing -- and only swaps in the DSpine draft
class, stamps the ``dflash_config`` fields SGLang's DSPINE loader reads, also
captures the target logits (the objective distills toward them), builds the
draft's fixed codebook from the target head, and swaps in the DSpine trainer.

The saved draft (``architectures: ["DSpineDraftModel"]``, ``selected_prefix_*``
config keys and ``selected_prefix_bridge.*`` tensors) loads directly into
SGLang with ``--speculative-algorithm DSPINE``.

IMPORTANT: as with DFlash, regenerate the training responses with the target
model first -- training teacher-forces ground-truth tokens while inference is
autoregressive, and the distribution mismatch hurts acceptance length otherwise.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from transformers import PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.speculative.dflash.dspine_core import (
    DSpineSchedule,
    DSpineStepMetrics,
    DSpineTrainerModule,
)
from nemo_automodel.components.speculative.dflash.registry import DFlashDraftSpec
from nemo_automodel.components.speculative.dflash.target import DFlashTargetBatch, HFDFlashTargetModel
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe

logger = logging.getLogger(__name__)

# Names of the DSpine validation statistics, in the fixed order every rank
# all-reduces them.
_EVAL_METRICS = (
    "val_backbone_ce",
    "val_backbone_l1",
    "val_refine_ce",
    "val_refine_l1",
    "val_alignment",
    "val_candidate_coverage",
    "val_base_accuracy",
    "val_base_accept_len",
)


class TrainDSpineRecipe(TrainDFlashRecipe):
    """Recipe for DSpine draft-model training: causal blocks + adjacent causal injection."""

    def _draft_cls(self, draft_spec: DFlashDraftSpec) -> type[torch.nn.Module]:
        """Build the DSpine draft, rejecting a target family that has none."""
        if draft_spec.dspine_cls is None:
            raise ValueError(
                f"No DSpine draft is registered for the {draft_spec.draft_cls.__name__} family; "
                "DSpine supports the Qwen3 and Qwen3.5 target families."
            )
        return draft_spec.dspine_cls

    def _build_qwen3_draft_config(
        self,
        recipe_cfg,
        *,
        target_text_config: PretrainedConfig,
        draft_cls: type[torch.nn.Module],
        draft_num_hidden_layers: int,
        num_target_layers: int,
        target_layer_ids: list[int],
        attention_backend: str,
    ) -> Qwen3Config:
        """Derive the DFlash draft config, then mark it causal: DSpine drafts with causal blocks.

        The DFlash builder stamps ``is_causal: false`` (DFlash's blocks are
        bidirectional); the DSpine draft and SGLang's DSPINE loader need it true,
        as the released DSpine drafters ship it.
        """
        draft_config = super()._build_qwen3_draft_config(
            recipe_cfg,
            target_text_config=target_text_config,
            draft_cls=draft_cls,
            draft_num_hidden_layers=draft_num_hidden_layers,
            num_target_layers=num_target_layers,
            target_layer_ids=target_layer_ids,
            attention_backend=attention_backend,
        )
        draft_config.is_causal = True
        return draft_config

    def _build_dflash_config(self, recipe_cfg, target_layer_ids: list[int]) -> dict:
        """Extend the DFlash draft config with the fields SGLang's DSPINE loader reads.

        Key names and defaults follow the released ``tencent/Qwen3-8B-DSpine-b16``
        drafter, so a checkpoint trained here is served without conversion.
        """
        cfg = super()._build_dflash_config(recipe_cfg, target_layer_ids)
        num_layers = int(recipe_cfg.get("draft_num_hidden_layers", 5))
        message_size = int(recipe_cfg.get("message_size", 512))
        if message_size % 128:
            logger.warning(
                "message_size=%d is not a multiple of 128; SGLang's DSPINE kernels require one, so this draft "
                "will train but not serve there.",
                message_size,
            )
        teacher_layers = int(recipe_cfg.get("teacher_forcing_layers", 3))
        if not 0 <= teacher_layers <= num_layers:
            raise ValueError(
                f"teacher_forcing_layers must be in [0, draft_num_hidden_layers={num_layers}], got {teacher_layers}."
            )
        cfg.update(
            {
                "draft_block_causal": True,
                "selected_prefix_code_bridge": {
                    "code_size": int(recipe_cfg.get("code_size", 1024)),
                    "rank": message_size,
                    "eps": 1e-6,
                },
                "selected_prefix_feature_layers": list(range(1, num_layers + 1)),
                "selected_prefix_feature_writes": True,
                "selected_prefix_feature_write_mode": "independent",
                "selected_prefix_terminal": False,
                "selected_prefix_decode_mode": "serial",
                "selected_prefix_feature_gt_training": True,
                "selected_prefix_feature_teacher_layers": list(range(1, teacher_layers + 1)),
                "selected_prefix_feature_teacher_schedule": {
                    "kind": "block_bernoulli",
                    "peak_probability": float(recipe_cfg.get("teacher_forcing_probability", 0.5)),
                    "hold_fraction": float(recipe_cfg.get("teacher_forcing_hold_fraction", 1 / 6)),
                    "zero_fraction": float(recipe_cfg.get("teacher_forcing_zero_fraction", 1 / 3)),
                },
                "selected_prefix_last_write_refine": {
                    "kind": "exact_sender",
                    "version": 1,
                    "head_topk": int(recipe_cfg.get("refine_top_k", 16)),
                    "weight": 1.0,
                },
            }
        )
        return cfg

    def _build_target_wrapper(self, target_layer_ids: list[int]) -> HFDFlashTargetModel:
        """Capture the target's full-vocab logits too -- DSpine distills toward them."""
        return HFDFlashTargetModel(
            self.target_model,
            target_layer_ids=target_layer_ids,
            capture_logits=True,
            cp_mesh=getattr(self, "cp_mesh", None),
        )

    def _build_trainer_module(self, attention_backend: str, recipe_cfg) -> DSpineTrainerModule:
        """Initialize the draft's codebook from the target head and build the DSpine trainer.

        Runs on the freshly built draft, before the checkpoint restore in
        ``setup`` -- which overwrites the codebook and readouts with the trained
        values on resume -- and before the DDP wrap.
        """
        if (recipe_cfg.get("loss_type", None) or "dflash") != "dflash":
            raise ValueError(
                "loss_type is only supported by the DFlash recipe; the DSpine trainer has its own objective "
                "and would silently ignore it."
            )
        lm_head = self.target_model.get_output_embeddings()
        self.draft_model.initialize_transfer_space(lm_head.weight)
        if dist.is_initialized() and dist.get_world_size() > 1:
            self.draft_model.broadcast_transfer_space(src=0)
        return DSpineTrainerModule(
            draft_model=self.draft_model,
            target_lm_head=lm_head,
            target_embed_tokens=self.target_model.get_input_embeddings(),
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            attention_backend=attention_backend,
            num_anchors=int(recipe_cfg.get("num_anchors", 512)),
            # Paper and SpecForge default; set null explicitly to disable the decay.
            loss_decay_gamma=recipe_cfg.get("loss_decay_gamma", 7.0),
            sliding_window=self.draft_sliding_window,
            distribution_ce_weight=float(recipe_cfg.get("distribution_ce_weight", 0.1)),
            candidate_mass_threshold=float(recipe_cfg.get("candidate_mass_threshold", 1e-4)),
            objective_chunk_blocks=int(recipe_cfg.get("objective_chunk_blocks", 128)),
        )

    def setup(self):
        """Build everything via the DFlash recipe, then the DSpine step schedule."""
        super().setup()
        recipe_cfg = self.cfg.recipe_args
        schedule_cfg = self.draft_model.config.dflash_config["selected_prefix_feature_teacher_schedule"]
        warmup = recipe_cfg.get("alignment_warmup_steps", None)
        self.dspine_schedule = DSpineSchedule(
            alignment_weight=float(recipe_cfg.get("alignment_weight", 0.5)),
            # Unset follows the LR warmup, as in SpecForge.
            alignment_warmup_steps=self.warmup_steps if warmup is None else int(warmup),
            teacher_forcing_probability=float(schedule_cfg["peak_probability"]),
            teacher_forcing_hold_fraction=float(schedule_cfg["hold_fraction"]),
            teacher_forcing_zero_fraction=float(schedule_cfg["zero_fraction"]),
            total_steps=self.total_optim_steps,
        )
        self._last_dspine_metrics = None

    def _run_trainer_step(self, target_batch: DFlashTargetBatch) -> DSpineStepMetrics:
        """Forward through the DSpine wrapper with the target logits and the step's schedule.

        Args:
            target_batch: The captured target batch. ``input_ids`` / ``loss_mask``
                are tensors of shape [batch, sequence]; ``hidden_states`` is a
                Tensor of shape [batch, sequence, layers * hidden]; ``logits`` is a
                Tensor of shape [batch, sequence, vocab] (captured because this
                recipe's wrapper sets ``capture_logits``); the packing fields are
                as documented on ``DFlashTargetBatch``.

        Returns:
            DSpineStepMetrics for the micro-batch.
        """
        step = self.runtime.global_step
        metrics = self.trainer_module(
            input_ids=target_batch.input_ids,
            hidden_states=target_batch.hidden_states,
            loss_mask=target_batch.loss_mask,
            target_logits=target_batch.logits,
            position_ids=target_batch.position_ids,
            seq_lens=target_batch.seq_lens,
            doc_remaining=target_batch.doc_remaining,
            alignment_beta=self.dspine_schedule.alignment_beta(step),
            teacher_forcing=self.dspine_schedule.teacher_forcing(step),
        )
        self._last_dspine_metrics = metrics
        return metrics

    def _log_extra_train_metrics(self, epoch_idx: int) -> None:
        """Log the DSpine diagnostics for the most recent step (rank-0 local)."""
        m = getattr(self, "_last_dspine_metrics", None)
        if m is None:
            return
        weight = float(m.weight_sum.clamp_min(1e-30))
        valid = float(m.valid_tokens.clamp_min(1))
        logger.info(
            "  dspine: backbone_ce=%.4f backbone_l1=%.4f refine_ce=%.4f refine_l1=%.4f alignment=%.4f "
            "candidate_coverage=%.4f base_acc=%.4f accept_len=%.3f base_accept_len=%.3f",
            float(m.backbone_ce_sum) / weight,
            float(m.backbone_l1_sum) / weight,
            float(m.refine_ce),
            float(m.refine_l1),
            float(m.alignment_sum) / weight,
            float(m.candidate_hits) / valid,
            float(m.base_correct_tokens) / valid,
            float(m.accept_len),
            float(m.base_accept_len),
        )

    def _extra_train_metric_sums(self, metrics: DSpineStepMetrics) -> dict[str, tuple[float, float]]:
        """Return the DSpine diagnostics as window sums.

        Loss-like terms are decay-weight means (the refinement terms over the
        positions their gates keep), coverage and accuracy are token means, and
        acceptance lengths are block means, matching how ``train/accuracy`` and
        ``train/accept_len`` are averaged.

        Args:
            metrics: DSpineStepMetrics whose fields are scalar tensors of shape [],
                with layouts documented by that dataclass.

        Returns:
            Metric names mapped to (numerator, denominator) float pairs.
        """
        values = super()._extra_train_metric_sums(metrics)
        weight = float(metrics.weight_sum)
        valid_tokens = float(metrics.valid_tokens)
        values.update(
            {
                "train/backbone_ce": (float(metrics.backbone_ce_sum), weight),
                "train/backbone_l1": (float(metrics.backbone_l1_sum), weight),
                "train/refine_ce": (
                    float(metrics.refine_ce) * float(metrics.refine_ce_weight),
                    float(metrics.refine_ce_weight),
                ),
                "train/refine_l1": (
                    float(metrics.refine_l1) * float(metrics.refine_l1_weight),
                    float(metrics.refine_l1_weight),
                ),
                "train/alignment": (float(metrics.alignment_sum), weight),
                "train/candidate_coverage": (float(metrics.candidate_hits), valid_tokens),
                "train/base_accuracy": (float(metrics.base_correct_tokens), valid_tokens),
                "train/base_accept_len": (float(metrics.base_accept_len_sum), float(metrics.valid_blocks)),
            }
        )
        return values

    def _extra_eval_metric_sums(self, metrics: DSpineStepMetrics) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return additive DSpine validation statistics.

        The shared validation loop SUM-reduces each pair before division.

        Args:
            metrics: DSpineStepMetrics whose fields are scalar tensors of shape []
                on the trainer device, with layouts documented by that dataclass.

        Returns:
            Metric names mapped to detached (numerator, denominator) pairs of
            scalar tensors of shape [] on the trainer device.
        """
        weight = metrics.weight_sum.detach()
        refine_ce_weight = metrics.refine_ce_weight.detach()
        refine_l1_weight = metrics.refine_l1_weight.detach()
        valid_tokens = metrics.valid_tokens.detach()
        return {
            "val_backbone_ce": (metrics.backbone_ce_sum.detach(), weight),
            "val_backbone_l1": (metrics.backbone_l1_sum.detach(), weight),
            "val_refine_ce": (metrics.refine_ce.detach() * refine_ce_weight, refine_ce_weight),
            "val_refine_l1": (metrics.refine_l1.detach() * refine_l1_weight, refine_l1_weight),
            "val_alignment": (metrics.alignment_sum.detach(), weight),
            "val_candidate_coverage": (metrics.candidate_hits.detach(), valid_tokens),
            "val_base_accuracy": (metrics.base_correct_tokens.detach(), valid_tokens),
            "val_base_accept_len": (metrics.base_accept_len_sum.detach(), metrics.valid_blocks.detach()),
        }

    def _empty_extra_eval_metric_sums(self) -> dict[str, list[torch.Tensor]]:
        """Create rank-symmetric DSpine validation accumulators."""
        return {
            name: [torch.zeros((), device=self.device), torch.zeros((), device=self.device)] for name in _EVAL_METRICS
        }


def main(config_path: str | None = None):
    """Entrypoint for ``TrainDSpineRecipe``."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainDSpineRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
