# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""
Generation script for loading saved VLM checkpoints and performing inference.

This script demonstrates multiple ways to load a checkpoint from SFT training
and use it for image-text generation tasks.

Usage:
    # Method 1: Load from HuggingFace-compatible consolidated checkpoint
    python generate.py --checkpoint_path /path/to/checkpoint/epoch_X_step_Y/model/consolidated --prompt <prompt> --image_url <image_url or local path>

    # Method 2: Load from distributed checkpoint
    python generate.py --checkpoint_path /path/to/checkpoint/epoch_X_step_Y --base_model google/gemma-3-4b-it --prompt <prompt> --image_url <image_url or local path>
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.distributed
from transformers import AutoProcessor, AutoConfig
from PIL import Image
import requests

from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
from nemo_automodel.checkpoint.checkpointing import load_model, CheckpointingConfig

import logging

from nemo_automodel.loggers.log_utils import setup_logging


def load_model_from_checkpoint(
    checkpoint_path: str,
    base_model: Optional[str] = None,
    is_peft: bool = False,
    model_save_format: str = "torch_save",
    use_liger_kernel: bool = False,
) -> NeMoAutoModelForImageTextToText:
    checkpoint_path = Path(checkpoint_path)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if (checkpoint_path / "config.json").exists():
        model = NeMoAutoModelForImageTextToText.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            use_liger_kernel=use_liger_kernel,
        )
    else:
        if base_model is None:
            raise ValueError("base_model required for distributed checkpoint")
        config = AutoConfig.from_pretrained(base_model)
        with torch.device(device_map):
            model = NeMoAutoModelForImageTextToText.from_config(
                config, torch_dtype=torch.bfloat16, use_liger_kernel=False
            )

        checkpoint_config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=checkpoint_path.parent,
            model_save_format=model_save_format,
            model_cache_dir="",
            model_repo_id="",
            save_consolidated=False,
            is_peft=is_peft,
        )
        load_model(model, str(checkpoint_path), checkpoint_config)
    logging.info(f"✅ Model loaded successfully from {checkpoint_path}")
    return model


def generate_response(
    model: NeMoAutoModelForImageTextToText,
    processor: AutoProcessor,
    image_url: str,
    prompt: str,
    max_new_tokens: int = 32,
) -> str:

    if image_url.startswith("http"):
        image = Image.open(requests.get(image_url, stream=True).raw)
    else:
        image = Image.open(image_url)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    prompt_length = len(
        processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
    )
    return generated_text[prompt_length:].strip()


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--is_peft", action="store_true")
    parser.add_argument(
        "--model_save_format",
        type=str,
        default="torch_save",
        choices=["torch_save", "safetensors"],
    )
    parser.add_argument("--image_url", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)

    args = parser.parse_args()

    logging.info(
        f"Loading model type base_model:{args.base_model} from checkpoint_path:{args.checkpoint_path}"
    )

    model = load_model_from_checkpoint(
        args.checkpoint_path,
        args.base_model,
        args.is_peft,
        args.model_save_format,
        use_liger_kernel=False,
    )
    processor_path = args.base_model if args.base_model else args.checkpoint_path
    processor = AutoProcessor.from_pretrained(processor_path)
    response = generate_response(
        model, processor, args.image_url, args.prompt, args.max_new_tokens
    )
    logging.info(f"Response: {response}")


if __name__ == "__main__":
    main()
