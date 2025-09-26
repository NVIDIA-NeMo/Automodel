#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Example script showing how to use timers and MFU calculation."""

import argparse

import torch
import yaml

from nemo_automodel.recipes.llm.train_ft import (
    main,
)


def create_example_config(output_path: str, model_name: str = "gpt2"):
    """Create an example configuration with timers enabled."""
    config = {
        # Timer configuration
        "timers": {
            "enabled": True,
            "log_level": 2,
            "log_option": "minmax",
            "log_interval": 10,
            "calculate_mfu": True,
            "theoretical_peak_tflops": 1979.0,  # A100 80GB
            "warmup_steps": 10,
        },
        # Model configuration
        "model": {
            "_target_": "transformers.AutoModelForCausalLM.from_pretrained",
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": False,
        },
        # Dataset configuration
        "dataset": {
            "_target_": "nemo_automodel.components.datasets.llm.mock.MockDataset",
            "seq_len": 2048,
            "size": 10000,
        },
        # DataLoader configuration
        "dataloader": {
            "_target_": "torch.utils.data.DataLoader",
            "num_workers": 4,
            "pin_memory": True,
            "shuffle": True,
        },
        # Loss function
        "loss_fn": {
            "_target_": "torch.nn.CrossEntropyLoss",
        },
        # Optimizer configuration
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "foreach": True,
        },
        # Step scheduler configuration
        "step_scheduler": {
            "num_epochs": 1,
            "global_batch_size": 64,
            "local_batch_size": 8,
            "ckpt_every_steps": 100,
            "max_steps": 50,  # Run 50 steps for demo
        },
        # Distributed configuration
        "distributed": {
            "_target_": "nemo_automodel.components.distributed.nvfsdp.NVFSDPManager",
            "dp_size": torch.cuda.device_count(),
            "tp_size": 1,
        },
        # Learning rate scheduler
        "lr_scheduler": {
            "lr_warmup_steps": 10,
            "lr_decay_style": "cosine",
        },
        # Checkpointing
        "checkpoint": {
            "enabled": False,  # Disable for demo
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created example config at: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training with timers and MFU calculation")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file. If not provided, creates an example config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model to use (default: gpt2)",
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create an example config file and exit",
    )
    args = parser.parse_args()

    if args.create_config or args.config is None:
        config_path = "timer_example_config.yaml"
        create_example_config(config_path, args.model)
        if args.create_config:
            exit(0)
    else:
        config_path = args.config

    # Run training with timers
    print(f"Running training with config: {config_path}")
    print("=" * 80)
    print("Timer features enabled:")
    print("- Detailed timing breakdown:")
    print("  * Setup phase")
    print("  * Iteration (total)")
    print("  * Forward-backward passes")
    print("  * Gradient scaling (for pipeline parallelism)")
    print("  * Gradient norm calculation and clipping")
    print("  * Optimizer step")
    print("- MFU (Model FLOPs Utilization) calculation")
    print("=" * 80)

    # Run the training
    main(config_path)

    print("\nTraining completed! Check the output above for:")
    print("- Per-step timing information")
    print("- MFU percentage")
    print("- Final training summary with average iteration times")
