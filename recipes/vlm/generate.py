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
    python generate.py --checkpoint-path /path/to/checkpoint/epoch_X_step_Y/model/consolidated \
        --prompt <prompt> --image <image_url or local path>

    # Method 2: Load from distributed checkpoint
    python generate.py --checkpoint-path /path/to/checkpoint/epoch_X_step_Y \
        --base-model google/gemma-3-4b-it --prompt <prompt> --image <image_url or local path>
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import requests
import torch
import torch.distributed
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
from nemo_automodel.checkpoint.checkpointing import CheckpointingConfig, load_model
from nemo_automodel.loggers.log_utils import setup_logging

# TODO: Parse config from YAML and run generate with FSDP2/distributed in general


def load_model_from_checkpoint(
    checkpoint_path: str,
    base_model: Optional[str] = None,
    is_peft: bool = False,
    model_save_format: str = "torch_save",
    use_liger_kernel: bool = False,
) -> NeMoAutoModelForImageTextToText:
    """Load a VLM model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory
        base_model: Base model name for distributed checkpoints
        is_peft: Whether the checkpoint is a PEFT checkpoint
        model_save_format: Format of the saved model ("torch_save" or "safetensors")
        use_liger_kernel: Whether to use Liger kernel optimizations

    Returns:
        Loaded NeMoAutoModelForImageTextToText model
    """
    # initialize distributed

    from nemo_automodel.distributed.init_utils import initialize_distributed

    initialize_distributed(backend="nccl", timeout_minutes=10)

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
    """Generate a text response from an image and text prompt.

    Args:
        model: The loaded VLM model
        processor: The model's processor for tokenization
        image_url: URL or local path to the image
        prompt: Text prompt for the model
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        Generated text response
    """

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
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    prompt_length = len(processor.decode(inputs["input_ids"][0], skip_special_tokens=True))
    return generated_text[prompt_length:].strip()


def main():
    """Main function to run VLM generation from command line arguments."""
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--is-peft", action="store_true")
    parser.add_argument(
        "--model-save-format",
        type=str,
        default="torch_save",
        choices=["torch_save", "safetensors"],
    )
    parser.add_argument(
        "--image",
        "--image-path",
        "--image-url",
        dest="image_url",
        type=str,
        default=None,
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--output-format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output format: 'text' for plain text or 'json' for JSON format",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional file path to write the output to",
    )

    args = parser.parse_args()

    logging.info(f"Loading model type base_model:{args.base_model} from checkpoint_path:{args.checkpoint_path}")

    model = load_model_from_checkpoint(
        args.checkpoint_path,
        args.base_model,
        args.is_peft,
        args.model_save_format,
        use_liger_kernel=False,
    )
    processor_path = args.base_model if args.base_model else args.checkpoint_path
    processor = AutoProcessor.from_pretrained(processor_path)
    response = generate_response(model, processor, args.image_url, args.prompt, args.max_new_tokens)

    # Format and output response
    if args.output_format == "json":
        output = {
            "prompt": args.prompt,
            "image_url": args.image_url,
            "response": response,
        }
        output_text = json.dumps(output, indent=2)
    else:
        output_text = response

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output_text)
        logging.info(f"Output written to {args.output_file}")
    else:
        logging.info(output_text)


if __name__ == "__main__":
    main()
