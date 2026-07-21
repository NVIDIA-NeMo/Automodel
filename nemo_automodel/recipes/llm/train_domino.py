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

"""Domino draft-model training recipe (Qwen3-style targets).

Domino (sgl-project/SpecForge#571) extends the DFlash parallel draft backbone
with a lightweight causal correction head (a GRU state plus a low-rank logit
correction; see ``nemo_automodel.components.speculative.dflash.domino_core``).
This recipe reuses every piece of the DFlash recipe -- online target hidden-state
capture, anchor sampling, the block attention mask, gradient accumulation, and
checkpointing -- and only swaps in the Domino trainer wrapper, enables the Domino
head on the draft via ``dflash_config``, and drives the base-anchor ``lambda_base``
curriculum.
"""

from __future__ import annotations

import logging

import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.speculative.dflash.domino_core import DominoTrainerModule, get_lambda_base
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe

logger = logging.getLogger(__name__)


class TrainDominoRecipe(TrainDFlashRecipe):
    """Recipe for Domino draft-model training: DFlash backbone + causal correction head."""

    def _build_dflash_config(self, recipe_cfg, target_layer_ids: list[int]) -> dict:
        """Extend the DFlash draft config with the Domino head fields."""
        cfg = super()._build_dflash_config(recipe_cfg, target_layer_ids)
        cfg.update(
            {
                "projector_type": "domino",
                "emb_dim": int(recipe_cfg.get("emb_dim", 256)),
                "gru_hidden_dim": int(recipe_cfg.get("gru_hidden_dim", 1024)),
                "pure_draft_prefix_len": int(recipe_cfg.get("pure_draft_prefix_len", 1)),
                "shift_label": bool(recipe_cfg.get("shift_label", True)),
            }
        )
        return cfg

    def _build_trainer_module(self, attention_backend: str, recipe_cfg):
        """Build the Domino trainer wrapper on the (Domino-head-enabled) DFlash draft."""
        if (recipe_cfg.get("loss_type", None) or "dflash") != "dflash":
            raise ValueError(
                "loss_type is only supported by the DFlash recipe; the Domino trainer has its own "
                "dual-logit objective and would silently ignore it."
            )
        return DominoTrainerModule(
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
            shift_label=self.draft_model.shift_label,
        )

    def setup(self):
        """Build everything via the DFlash recipe, then read the lambda_base schedule."""
        super().setup()
        recipe_cfg = self.cfg.recipe_args
        self.lambda_base_start = float(recipe_cfg.get("lambda_base_start", 1.0))
        self.lambda_base_decay_ratio = float(recipe_cfg.get("lambda_base_decay_ratio", 0.5))
        self._last_domino_metrics = None

    def _run_trainer_step(self, target_batch):
        """Forward through the Domino wrapper, injecting the current curriculum weight."""
        lambda_base = get_lambda_base(
            global_step=self.runtime.global_step,
            total_steps=self.total_optim_steps,
            lambda_start=self.lambda_base_start,
            decay_ratio=self.lambda_base_decay_ratio,
        )
        metrics = self.trainer_module(
            input_ids=target_batch.input_ids,
            hidden_states=target_batch.hidden_states,
            loss_mask=target_batch.loss_mask,
            lambda_base=lambda_base,
            position_ids=target_batch.position_ids,
            seq_lens=target_batch.seq_lens,
            doc_remaining=target_batch.doc_remaining,
        )
        self._last_domino_metrics = metrics
        return metrics

    def _log_extra_train_metrics(self, epoch_idx: int) -> None:
        """Log the Domino-specific diagnostics for the most recent step (rank-0 local)."""
        m = getattr(self, "_last_domino_metrics", None)
        if m is None:
            return
        logger.info(
            "  domino: final_loss=%.4f base_loss=%.4f base_acc=%.4f "
            "accept_len=%.3f base_accept_len=%.3f lambda_base=%.3f",
            float(m.final_loss),
            float(m.base_loss),
            float(m.base_accuracy),
            float(m.accept_len),
            float(m.base_accept_len),
            float(m.lambda_base),
        )

    def _extra_train_wandb_metrics(self, metrics) -> dict[str, float]:
        """Return Domino head and curriculum diagnostics for W&B."""
        values = super()._extra_train_wandb_metrics(metrics)
        values.update(
            {
                "train/final_loss": float(metrics.final_loss),
                "train/base_loss": float(metrics.base_loss),
                "train/base_accuracy": float(metrics.base_accuracy),
                "train/base_accept_len": float(metrics.base_accept_len),
                "train/lambda_base": float(metrics.lambda_base),
            }
        )
        return values

    def _extra_eval_metric_sums(self, metrics) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return additive Domino base-head validation statistics.

        Every returned value is a pair of scalar tensors on the trainer device.
        The shared validation loop SUM-reduces each pair before division.
        """
        loss_weight = metrics.loss_weight.detach()
        valid_tokens = metrics.valid_tokens.detach()
        valid_blocks = metrics.valid_blocks.detach()
        return {
            "val_final_loss": (metrics.final_loss.detach() * loss_weight, loss_weight),
            "val_base_loss": (metrics.base_loss.detach() * loss_weight, loss_weight),
            "val_base_accuracy": (metrics.base_correct_tokens.detach(), valid_tokens),
            "val_base_accept_len": (metrics.base_accept_len_sum.detach(), valid_blocks),
        }

    def _empty_extra_eval_metric_sums(self) -> dict[str, list[torch.Tensor]]:
        """Create rank-symmetric Domino validation accumulators."""
        return {
            name: [torch.zeros((), device=self.device), torch.zeros((), device=self.device)]
            for name in ("val_final_loss", "val_base_loss", "val_base_accuracy", "val_base_accept_len")
        }


def main(config_path: str | None = None):
    """Entrypoint for ``TrainDominoRecipe``."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainDominoRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
