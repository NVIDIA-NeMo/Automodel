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
        return DominoTrainerModule(
            draft_model=self.draft_model,
            target_lm_head=self.target_model.get_output_embeddings(),
            target_embed_tokens=self.target_model.get_input_embeddings(),
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            attention_backend=attention_backend,
            num_anchors=int(recipe_cfg.get("num_anchors", 512)),
            loss_decay_gamma=recipe_cfg.get("loss_decay_gamma", None),
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


def main(config_path: str | None = None):
    """Entrypoint for ``TrainDominoRecipe``."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainDominoRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
