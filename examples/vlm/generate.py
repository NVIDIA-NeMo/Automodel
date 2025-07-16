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
import glob
import json
import logging
import os
from typing import Optional

import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from nemo_automodel._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
from nemo_automodel.checkpoint.checkpointing import CheckpointingConfig, load_model
from nemo_automodel.loggers.log_utils import setup_logging

# TODO: Parse config from YAML and run generate with FSDP2/distributed in general


def get_checkpoint_type(checkpoint_path: str) -> str:
    """Get the type of the checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        'torch_save' if the checkpoint is a DCP checkpoint or 'safetensors' if the checkpoint is a safetensors checkpoint
    """
    safetensors = glob.glob(os.path.join(checkpoint_path, "model", "*.safetensors"))
    if len(safetensors) > 0:
        return "safetensors"
    else:
        return "torch_save"


def is_peft_checkpoint(checkpoint_path: str) -> bool:
    """Check if the checkpoint is a PEFT checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        True if the checkpoint is a PEFT checkpoint, False otherwise
    """
    return os.path.exists(os.path.join(checkpoint_path, "model", "adapter_model.safetensors"))


def is_consolidated_safetensors_checkpoint(checkpoint_path: str) -> bool:
    """Check if the checkpoint is a consolidated safetensors checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        True if the checkpoint is a consolidated safetensors checkpoint, False otherwise
    """
    return os.path.exists(os.path.join(checkpoint_path, "model", "model.safetensors.index.json"))


def apply_peft_to_model(model: NeMoAutoModelForImageTextToText, checkpoint_path: str):
    """Apply PEFT to the model.

    Args:
        model: The model to apply PEFT to
        checkpoint_path: Path to the checkpoint directory
    """
    peft_dict = {}
    peft_config_path = os.path.join(checkpoint_path, "model", "adapter_config.json")
    automodel_peft_config_path = os.path.join(checkpoint_path, "model", "automodel_peft_config.json")
    with open(peft_config_path, "r") as f:
        restored_peft_config = json.load(f)
        peft_dict["dim"] = restored_peft_config["r"]
        peft_dict["alpha"] = restored_peft_config["lora_alpha"]
    with open(automodel_peft_config_path, "r") as f:
        automodel_peft_dict = json.load(f)
        peft_dict |= automodel_peft_dict
    peft_config = PeftConfig.from_dict(peft_dict)
    apply_lora_to_linear_modules(model, peft_config)


def load_model_from_checkpoint(
    checkpoint_path: str,
    base_model_path: Optional[str] = None,
    use_liger_kernel: bool = False,
) -> NeMoAutoModelForImageTextToText:
    """Load a VLM model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory
        base_model_path: Path to the base model checkpoint. This can either be something like 'google/gemma-3-4b-it' or a local path to the base model. This is not required if restoring from a consolidated HF checkpoint.
        use_liger_kernel: Whether to use Liger kernel optimizations

    Returns:
        Loaded NeMoAutoModelForImageTextToText model
    """
    # initialize distributed
    from nemo_automodel.distributed.init_utils import initialize_distributed

    initialize_distributed(backend="nccl", timeout_minutes=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(checkpoint_path, "model")
    if is_consolidated_safetensors_checkpoint(checkpoint_path):
        model = NeMoAutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_liger_kernel=use_liger_kernel,
        ).to(device)
        return model

    if base_model_path is None:
        raise ValueError("base_model_path is required if not restoring from a consolidated HF checkpoint.")

    model = NeMoAutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        use_liger_kernel=use_liger_kernel,
    ).to(device)

    if is_peft_checkpoint(checkpoint_path):
        apply_peft_to_model(model, checkpoint_path)

    checkpoint_config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=checkpoint_path,
        model_save_format=get_checkpoint_type(checkpoint_path),
        model_cache_dir="",
        model_repo_id=base_model_path,
        save_consolidated=False,
        is_peft=is_peft_checkpoint(checkpoint_path),
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
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--base-model-path",
        type=str,
        required=False,
        help="The path to the base model checkpoint. This can either be something like 'google/gemma-3-4b-it' or a local path to the base model. This is not required if restoring from a consolidated HF checkpoint.",
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

    logging.info(f"Loading model type base_model:{args.base_model_path} from checkpoint_path:{args.checkpoint_path}")

    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        base_model_path=args.base_model_path,
        use_liger_kernel=False,
    )
    processor_path = args.base_model_path if args.base_model_path else args.checkpoint_path
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
