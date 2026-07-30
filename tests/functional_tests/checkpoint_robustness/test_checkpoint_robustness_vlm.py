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

import os

from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import (
    _DEFAULT_INPUT_IDS,
    _DEFAULT_PROMPT,
    _report_phase,
    _start_preflight_watchdog,
    _stop_preflight_watchdog,
    run_checkpoint_robustness,
)


def _get_vlm_input_ids(processor_name: str | None) -> list[int]:
    """Encode the parity prompt through the same processor family used by VLM data loading."""
    if processor_name is None:
        return _DEFAULT_INPUT_IDS

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        processor_name,
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.encode(_DEFAULT_PROMPT, add_special_tokens=False)


def test_checkpoint_robustness_vlm() -> None:
    """Run checkpoint robustness with the VLM finetune recipe and text-only logits."""
    _start_preflight_watchdog()
    try:
        _report_phase("Preflight: importing AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText

        _report_phase("Preflight: AutoModelForImageTextToText import complete")
        _report_phase("Preflight: importing FinetuneRecipeForVLM")
        from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

        _report_phase("Preflight: FinetuneRecipeForVLM import complete")
        run_checkpoint_robustness(
            recipe_cls=FinetuneRecipeForVLM,
            hf_model_cls=AutoModelForImageTextToText,
            input_ids_loader=_get_vlm_input_ids,
        )
    finally:
        _stop_preflight_watchdog()


if __name__ == "__main__":
    test_checkpoint_robustness_vlm()
