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
Script to convert biencoder distributed checkpoint to HuggingFace format.

This script loads a checkpoint saved by the biencoder training recipe and converts
it to HuggingFace format that can be loaded with AutoModel.from_pretrained().

Usage:
    # Convert checkpoint to HuggingFace format
    python examples/biencoder/convert_to_hf.py --ckpt_path path/to/checkpoint --output_dir path/to/output

Example:
    python examples/biencoder/convert_to_hf.py \
        --ckpt_path output/llama3_2_1b_biencoder/checkpoints/epoch_0_step_485 \
        --output_dir output/hf_model
"""

import argparse
import os

import torch.distributed.checkpoint as dist_cp
import yaml

from nemo_automodel.components.config.loader import ConfigNode


def load_config(checkpoint_path: str) -> ConfigNode:
    """Load configuration from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        ConfigNode with the training configuration
    """
    config_file = os.path.join(checkpoint_path, "config.yaml")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    return ConfigNode(config_dict)


def load_model_from_checkpoint(checkpoint_path: str):
    """Load biencoder model from distributed checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Tuple of (model, config)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load config
    cfg = load_config(checkpoint_path)
    print("Configuration loaded successfully")

    # Initialize model from config
    print("Initializing model...")
    model = cfg.model.instantiate()
    print(f"Model initialized: {model.__class__.__name__}")

    # Load model weights from distributed checkpoint
    model_path = os.path.join(checkpoint_path, "model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    print(f"Loading model weights from: {model_path}")

    # Create state dict for loading
    state_dict = model.state_dict()

    # Load using distributed checkpoint
    try:
        dist_cp.load(
            state_dict=state_dict,
            checkpoint_id=model_path,
            storage_reader=dist_cp.FileSystemReader(model_path),
            no_dist=True,  # Load without distributed setup
        )
        model.load_state_dict(state_dict, strict=True)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        raise

    model.eval()
    print("Model loaded and set to eval mode")

    return model, cfg


def load_tokenizer(cfg: ConfigNode):
    """Load tokenizer from config.

    Args:
        cfg: Configuration containing tokenizer settings

    Returns:
        Tokenizer instance
    """
    tokenizer = cfg.tokenizer.instantiate()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    return tokenizer


def convert_to_hf_format(model, tokenizer, cfg, output_dir: str):
    """Convert distributed checkpoint to HuggingFace format.

    Args:
        model: Loaded biencoder model
        tokenizer: Tokenizer
        cfg: Configuration
        output_dir: Output directory for HuggingFace checkpoint
    """
    print(f"Converting checkpoint to HuggingFace format at: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    hf_model = model.lm_q

    original_model_name = cfg.model.pretrained_model_name_or_path
    hf_model.config._name_or_path = original_model_name
    print(f"Setting _name_or_path to: {original_model_name}")

    hf_model.__class__.register_for_auto_class("AutoModel")
    hf_model.config.__class__.register_for_auto_class("AutoConfig")

    # Save model using save_pretrained
    print("Saving model with save_pretrained()...")
    hf_model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )
    print("Model saved to model.safetensors")

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    print("Tokenizer saved")

    print(f"\n{'=' * 80}")
    print("✓ Conversion complete!")
    print(f"{'=' * 80}")
    print(f"\nCheckpoint saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert biencoder distributed checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert checkpoint to HuggingFace format
  python examples/biencoder/convert_to_hf.py \
      --ckpt_path output/llama3_2_1b_biencoder/checkpoints/epoch_0_step_485 \
      --output_dir output/hf_model

  # Use default paths
  python examples/biencoder/convert_to_hf.py
        """,
    )
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to distributed checkpoint directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/hf_model",
        help="Output directory for HuggingFace checkpoint (default: %(default)s)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("BIENCODER CHECKPOINT CONVERTER")
    print("=" * 80)
    print(f"\nInput checkpoint: {args.ckpt_path}")
    print(f"Output directory: {args.output_dir}")

    # Load model and config
    print("\nLoading checkpoint...")
    model, cfg = load_model_from_checkpoint(args.ckpt_path)

    # Load tokenizer
    tokenizer = load_tokenizer(cfg)
    print("Tokenizer loaded successfully")

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {num_params:,} parameters")

    # Convert to HuggingFace format
    convert_to_hf_format(model, tokenizer, cfg, args.output_dir)
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
