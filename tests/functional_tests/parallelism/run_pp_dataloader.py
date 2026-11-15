# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise production PP setup, spawned data loading, and optimization offline."""

import argparse
import json
import math
import multiprocessing
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import LlamaConfig


def main() -> None:
    """Prepare a tiny local config or train it on two pipeline ranks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    model_dir = args.work_dir / "model"
    if args.prepare_only:
        LlamaConfig(
            architectures=["LlamaForCausalLM"],
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            use_cache=False,
            tie_word_embeddings=False,
        ).save_pretrained(model_dir)
        return

    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.components.datasets.loader import close_dataloader
    from nemo_automodel.components.datasets.utils import collate_with_causal_masks
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    cfg = load_yaml_config(Path(__file__).with_name("pp_dataloader_tiny.yaml"))
    cfg.model.config.pretrained_model_name_or_path = str(model_dir)
    cfg.checkpoint.checkpoint_dir = str(args.work_dir / "output")
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    loader = None
    try:
        recipe.setup()
        loader = recipe.dataloader
        assert dist.get_world_size() == 2
        assert recipe.pp_enabled and recipe.mesh_context.pp_size == 2
        assert recipe.dataloader.num_workers == 2
        assert multiprocessing.get_start_method() == "spawn"
        collate_fn = recipe.dataloader.collate_fn

        parameter_count = torch.tensor(
            sum(parameter.numel() for part in recipe.model_parts for parameter in part.parameters()),
            device=recipe.dist_env.device,
            dtype=torch.int64,
        )
        dist.all_reduce(parameter_count)
        assert 0 < parameter_count.item() <= 1_000_000
        recipe.run_train_validation_loop()
        # Check after iteration so the original nested closure fails in worker
        # startup, while silently skipping mask precomputation cannot pass either.
        assert isinstance(collate_fn, partial)
        assert collate_fn.func is collate_with_causal_masks
        assert recipe.step_scheduler.step == 3
        dist.barrier()
        if dist.get_rank() == 0:
            records = [json.loads(line) for line in (args.work_dir / "output/training.jsonl").read_text().splitlines()]
            assert len(records) == 3
            for record in records:
                assert math.isfinite(record["loss"])
                assert math.isfinite(record["grad_norm"]) and record["grad_norm"] > 0
            print(f"PP DataLoader smoke passed: {parameter_count.item()} parameters, 3 optimizer steps")
    finally:
        # Also close workers when an assertion fails after setup.
        if loader is not None:
            close_dataloader(loader)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
