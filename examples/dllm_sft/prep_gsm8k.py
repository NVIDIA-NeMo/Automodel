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

"""Prepare GSM8K as an OpenAI-messages JSONL for the DiffusionGemma SFT/LoRA recipes.

The ``ChatDataset`` used by ``diffusion_gemma_{sft,lora}.yaml`` consumes the
OpenAI chat-messages schema (a ``messages`` list per row); the raw
``openai/gsm8k`` dataset is ``{question, answer}``, so we convert it once here.

Usage (from the repo root):

    python examples/dllm_sft/prep_gsm8k.py            # writes ./gsm8k_chat_train.jsonl

Then run the recipe (the YAML's ``path_or_dataset_id`` points at the output):

    torchrun --standalone --nproc-per-node=8 \
        examples/dllm_sft/finetune.py -c examples/dllm_sft/diffusion_gemma_sft.yaml
"""

import argparse
import json

from datasets import load_dataset


def main() -> None:
    """Convert the requested GSM8K split to an OpenAI-messages JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="gsm8k_chat_train.jsonl", help="output JSONL path")
    parser.add_argument("--split", default="train", help="GSM8K split to convert")
    args = parser.parse_args()

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    with open(args.output, "w") as f:
        for example in dataset:
            messages = [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ]
            f.write(json.dumps({"messages": messages}) + "\n")
    print(f"Wrote {len(dataset)} rows -> {args.output}")


if __name__ == "__main__":
    main()
