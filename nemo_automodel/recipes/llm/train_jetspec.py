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

"""JetSpec draft-model training recipe (Qwen3-style targets).

JetSpec (arXiv:2606.18394) reuses the DFlash parallel draft backbone but trains it
as a *causal* parallel tree drafter: in-block attention is causal (so each branch
is conditioned on its own prefix) and the draft is distilled against the target's
per-position soft distribution with a temperature-scaled forward-KL loss. See
``nemo_automodel.components.speculative.dflash.jetspec_core``.

This recipe reuses every piece of the DFlash recipe -- online target hidden-state
capture, anchor sampling, the block attention mask machinery, gradient
accumulation, and checkpointing -- and only (a) enables target-logit capture so
the teacher distribution is available, and (b) swaps in the JetSpec trainer
wrapper (causal mask + forward-KL).

IMPORTANT: as with DFlash, regenerate the training responses with the target
model first -- training teacher-forces ground-truth tokens while inference is
autoregressive, and the distribution mismatch hurts acceptance length otherwise.
"""

from __future__ import annotations

import logging

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.speculative.dflash.jetspec_core import JetSpecTrainerModule
from nemo_automodel.components.speculative.dflash.target import HFDFlashTargetModel
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe

logger = logging.getLogger(__name__)


class TrainJetSpecRecipe(TrainDFlashRecipe):
    """Recipe for JetSpec draft-model training: DFlash backbone + causal mask + forward-KL."""

    def _build_target_wrapper(self, target_layer_ids: list[int]) -> HFDFlashTargetModel:
        """Capture the target's full-vocab logits too -- JetSpec distills against them."""
        return HFDFlashTargetModel(self.target_model, target_layer_ids=target_layer_ids, capture_logits=True)

    def _build_trainer_module(self, attention_backend: str, recipe_cfg):
        """Build the JetSpec trainer wrapper (causal parallel drafting + forward-KL)."""
        return JetSpecTrainerModule(
            draft_model=self.draft_model,
            target_lm_head=self.target_model.get_output_embeddings(),
            target_embed_tokens=self.target_model.get_input_embeddings(),
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            attention_backend=attention_backend,
            num_anchors=int(recipe_cfg.get("num_anchors", 512)),
            kd_temperature=float(recipe_cfg.get("kd_temperature", 1.0)),
            kd_chunk_size=int(recipe_cfg.get("kd_chunk_size", 0)),
        )

    def _run_trainer_step(self, target_batch):
        """Forward through the JetSpec wrapper, passing the captured teacher logits."""
        return self.trainer_module(
            input_ids=target_batch.input_ids,
            hidden_states=target_batch.hidden_states,
            loss_mask=target_batch.loss_mask,
            target_logits=target_batch.logits,
        )


def main(config_path: str | None = None):
    """Entrypoint for ``TrainJetSpecRecipe``."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainJetSpecRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
