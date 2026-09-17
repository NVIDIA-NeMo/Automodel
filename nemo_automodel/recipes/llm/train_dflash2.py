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

"""DFlash 2 draft-model training recipe (Qwen3-style targets).

DFlash 2 (https://inco.ai/blog/dflash2/) keeps DFlash's one-pass parallel block
draft and adds two cheap modules: a two-tap dynamic convolution around every
draft sublayer, which carries the short-range within-block work and removes most
of DFlash's suffix decay, and a pairwise path selector that walks one coherent
path through each position's top-k candidates instead of keeping every top-1 pick
independently. See ``nemo_automodel.components.speculative.dflash.dflash2_core``.

This recipe reuses every piece of the DFlash recipe -- online target hidden-state
capture, anchor sampling, the block attention mask, gradient accumulation, and
checkpointing -- and only swaps in the DFlash 2 draft class, stamps the extra
``dflash_config`` fields a serving engine needs to rebuild it, and swaps in the
DFlash 2 trainer wrapper.

IMPORTANT: as with DFlash, regenerate the training responses with the target
model first -- training teacher-forces ground-truth tokens while inference is
autoregressive, and the distribution mismatch hurts acceptance length otherwise.
"""

from __future__ import annotations

import logging

import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.speculative.dflash.dflash2_core import DFlash2TrainerModule
from nemo_automodel.components.speculative.dflash.registry import DFlashDraftSpec
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe

logger = logging.getLogger(__name__)


class TrainDFlash2Recipe(TrainDFlashRecipe):
    """Recipe for DFlash 2 draft-model training: in-block convolutions + path selector."""

    def _draft_cls(self, draft_spec: DFlashDraftSpec) -> type[torch.nn.Module]:
        """Build the DFlash 2 draft instead of the plain DFlash one."""
        return draft_spec.draft2_cls

    def _build_dflash_config(self, recipe_cfg, target_layer_ids: list[int]) -> dict:
        """Extend the DFlash draft config with the convolution and selector shapes."""
        cfg = super()._build_dflash_config(recipe_cfg, target_layer_ids)
        cfg.update(
            {
                "conv_kernel_size": int(recipe_cfg.get("conv_kernel_size", 2)),
                "conv_group_size": int(recipe_cfg.get("conv_group_size", 16)),
                "selector_rank": int(recipe_cfg.get("selector_rank", 256)),
                "selector_top_k": int(recipe_cfg.get("selector_top_k", 16)),
            }
        )
        return cfg

    def _build_trainer_module(self, attention_backend: str, recipe_cfg):
        """Build the DFlash 2 trainer wrapper (block CE + candidate-selection CE)."""
        if (recipe_cfg.get("loss_type", None) or "dflash") != "dflash":
            raise ValueError(
                "loss_type is only supported by the DFlash recipe; the DFlash 2 trainer teacher-forces the "
                "selector's predecessor from the fixed-anchor block layout and would silently ignore it."
            )
        return DFlash2TrainerModule(
            draft_model=self.draft_model,
            target_lm_head=self.target_model.get_output_embeddings(),
            target_embed_tokens=self.target_model.get_input_embeddings(),
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            attention_backend=attention_backend,
            num_anchors=int(recipe_cfg.get("num_anchors", 512)),
            # Paper default (Appendix A.1) for the shipped block_size=16 configs;
            # matches DFlashDecayLoss's own default. Set null explicitly in YAML
            # to disable the position decay (uniform weighting).
            loss_decay_gamma=recipe_cfg.get("loss_decay_gamma", 7.0),
            selector_loss_weight=float(recipe_cfg.get("selector_loss_weight", 1.0)),
            sliding_window=self.draft_sliding_window,
        )

    def setup(self):
        """Build everything via the DFlash recipe, then reset the per-step metric cache."""
        super().setup()
        self._last_dflash2_metrics = None

    def _run_trainer_step(self, target_batch):
        """Forward through the DFlash 2 wrapper and cache the step's diagnostics."""
        metrics = super()._run_trainer_step(target_batch)
        self._last_dflash2_metrics = metrics
        return metrics

    def _log_extra_train_metrics(self, epoch_idx: int) -> None:
        """Log the DFlash 2 diagnostics for the most recent step (rank-0 local)."""
        m = getattr(self, "_last_dflash2_metrics", None)
        if m is None:
            return
        logger.info(
            "  dflash2: base_loss=%.4f selector_loss=%.4f base_acc=%.4f "
            "accept_len=%.3f base_accept_len=%.3f candidate_recall=%.4f",
            float(m.base_loss),
            float(m.selector_loss),
            float(m.base_accuracy),
            float(m.accept_len),
            float(m.base_accept_len),
            float(m.candidate_recall),
        )

    def _extra_train_metric_sums(self, metrics) -> dict[str, tuple[float, float]]:
        """Return the selector and backbone diagnostics as window sums.

        The two loss terms are per-micro-batch means, so their denominator is the
        micro-batch count; the accuracy-like curves are token- or block-weighted,
        matching how ``train/loss`` and ``train/accuracy`` are averaged.
        """
        values = super()._extra_train_metric_sums(metrics)
        valid_tokens = float(metrics.valid_tokens)
        values.update(
            {
                "train/base_loss": (float(metrics.base_loss), 1.0),
                "train/selector_loss": (float(metrics.selector_loss), 1.0),
                "train/base_accuracy": (float(metrics.base_correct_tokens), valid_tokens),
                "train/base_accept_len": (float(metrics.base_accept_len_sum), float(metrics.valid_blocks)),
                "train/candidate_recall": (float(metrics.candidate_recall) * valid_tokens, valid_tokens),
            }
        )
        return values

    def _extra_eval_metric_sums(self, metrics) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return additive DFlash 2 validation statistics.

        Every returned value is a pair of scalar tensors on the trainer device.
        The shared validation loop SUM-reduces each pair before division.
        """
        loss_weight = metrics.loss_weight.detach()
        valid_tokens = metrics.valid_tokens.detach()
        valid_blocks = metrics.valid_blocks.detach()
        return {
            "val_base_loss": (metrics.base_loss.detach() * loss_weight, loss_weight),
            "val_selector_loss": (metrics.selector_loss.detach() * loss_weight, loss_weight),
            "val_base_accuracy": (metrics.base_correct_tokens.detach(), valid_tokens),
            "val_base_accept_len": (metrics.base_accept_len_sum.detach(), valid_blocks),
            "val_candidate_recall": (metrics.candidate_recall.detach() * valid_tokens, valid_tokens),
        }

    def _empty_extra_eval_metric_sums(self) -> dict[str, list[torch.Tensor]]:
        """Create rank-symmetric DFlash 2 validation accumulators."""
        return {
            name: [torch.zeros((), device=self.device), torch.zeros((), device=self.device)]
            for name in (
                "val_base_loss",
                "val_selector_loss",
                "val_base_accuracy",
                "val_base_accept_len",
                "val_candidate_recall",
            )
        }


def main(config_path: str | None = None):
    """Entrypoint for ``TrainDFlash2Recipe``."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainDFlash2Recipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
