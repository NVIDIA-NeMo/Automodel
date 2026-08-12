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

"""Run normal CI finetuning while recording a non-blocking reproducibility reference."""

from __future__ import annotations

import os
from pathlib import Path

import torch.distributed as dist

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from tests.functional_tests.checkpoint_robustness.resume_trajectory import (
    _persist_training_reproducibility,
    _TrainingReproducibilityRecorder,
)


def _recipe_class(domain: str):
    """Return the existing recipe class for one supported CI domain."""
    if domain == "llm":
        from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

        return TrainFinetuneRecipeForNextTokenPrediction
    if domain == "vlm":
        from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

        return FinetuneRecipeForVLM
    if domain == "retrieval":
        from nemo_automodel.recipes.retrieval import TrainBiEncoderRecipe

        return TrainBiEncoderRecipe
    raise ValueError(f"Unsupported training reproducibility domain: {domain!r}")


def main() -> None:
    """Run the configured recipe and persist its per-rank training trajectory."""
    domain = os.environ["AUTOMODEL_REPRODUCIBILITY_DOMAIN"]
    artifact_dir = Path(os.environ["AUTOMODEL_REPRODUCIBILITY_DIR"])
    cfg = parse_args_and_load_config()
    trainer = _recipe_class(domain)(cfg)
    trainer.setup()
    recorder = _TrainingReproducibilityRecorder(trainer)
    recorder.attach()
    trainer.run_train_validation_loop()
    _persist_training_reproducibility(recorder, artifact_dir, lifecycle="normal")
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
