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

"""VLM entry point for text-only checkpoint robustness and source-load parity."""

from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import run_checkpoint_robustness


def test_checkpoint_robustness_vlm() -> None:
    """Run checkpoint robustness with the VLM finetune recipe and text-only logits."""
    from transformers import AutoModelForImageTextToText

    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    run_checkpoint_robustness(
        recipe_cls=FinetuneRecipeForVLM,
        hf_model_cls=AutoModelForImageTextToText,
    )


if __name__ == "__main__":
    test_checkpoint_robustness_vlm()
