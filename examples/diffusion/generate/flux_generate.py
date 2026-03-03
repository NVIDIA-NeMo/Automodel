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
FLUX Inference Script with Multi-Resolution Dataloader (Embedding Injection)

This script loads a FLUX transformer and runs inference by extracting
pre-computed text embeddings directly from the multiresolution dataloader.
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import FluxPipeline

# Import the provided dataloader builder
from nemo_automodel.components.datasets.diffusion import build_flux_multiresolution_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX Inference with pre-computed embeddings")

    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Base FLUX model ID from HuggingFace",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory containing model/ subfolder or consolidated weights",
    )
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="Use original FLUX model without loading custom checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset cache directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Maximum number of images to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_outputs",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for the dataloader",
    )

    return parser.parse_args()


def load_sharded_checkpoint(transformer, checkpoint_dir, device="cuda"):
    import torch.distributed as dist
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint import load as dist_load
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp.api import ShardedStateDictConfig

    sharded_dir = os.path.join(checkpoint_dir, "model")
    if not os.path.isdir(sharded_dir):
        raise FileNotFoundError(f"Model directory not found: {sharded_dir}")

    # Initialize a single-process group if needed
    init_dist = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        init_dist = True

    try:
        transformer = transformer.to(device=device, dtype=torch.bfloat16)
        fsdp_transformer = FSDP(transformer, use_orig_params=True, device_id=torch.device(device))

        FSDP.set_state_dict_type(
            fsdp_transformer,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        )

        model_state = fsdp_transformer.state_dict()
        dist_load(state_dict=model_state, storage_reader=FileSystemReader(sharded_dir))
        fsdp_transformer.load_state_dict(model_state)
        transformer = fsdp_transformer.module
        print("[INFO] Successfully loaded sharded FSDP checkpoint")
    finally:
        if init_dist:
            dist.destroy_process_group()
    return transformer


def load_consolidated_checkpoint(transformer, checkpoint_path):
    print(f"[INFO] Loading consolidated checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    transformer.load_state_dict(state_dict, strict=True)
    print("[INFO] Loaded consolidated checkpoint")
    return transformer


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Initialize Dataloader ---
    print("=" * 80)
    print(f"Initializing Multiresolution Dataloader: {args.data_path}")

    dataloader, _ = build_flux_multiresolution_dataloader(
        cache_dir=args.data_path, batch_size=1, num_workers=args.num_workers, dynamic_batch_size=True, shuffle=False
    )
    print(f"[INFO] Dataloader ready. Batches: {len(dataloader)}")

    # --- 2. Initialize Model ---
    use_original = args.use_original or args.checkpoint is None

    print(f"\n[Pipeline] Loading FLUX pipeline from: {args.model_id}")
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)

    if not use_original:
        checkpoint_dir = Path(args.checkpoint)
        model_name = checkpoint_dir.name
        sharded_dir = checkpoint_dir / "model"
        consolidated_path = checkpoint_dir / "consolidated_model.bin"
        ema_path = checkpoint_dir / "ema_shadow.pt"

        if ema_path.exists():
            print("[INFO] Loading EMA checkpoint...")
            pipe.transformer.load_state_dict(torch.load(ema_path, map_location="cpu"))
        elif consolidated_path.exists():
            pipe.transformer = load_consolidated_checkpoint(pipe.transformer, str(consolidated_path))
        elif sharded_dir.exists():
            pipe.transformer = load_sharded_checkpoint(pipe.transformer, str(checkpoint_dir), device=device)
    else:
        model_name = "original"

    pipe.enable_model_cpu_offload()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. Inference Loop (Injecting Embeddings) ---
    print(f"\n[Inference] Generating {args.max_samples} samples using pre-computed embeddings...")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    count = 0
    for batch_idx, batch in enumerate(dataloader):
        if count >= args.max_samples:
            break

        try:
            # Extract metadata for logging/filenames
            current_prompt = batch["metadata"]["prompts"][0]
            source_path = batch["metadata"]["image_paths"][0]

            # Extract and move embeddings to device/dtype
            # batch['text_embeddings'] corresponds to T5 output
            # batch['pooled_prompt_embeds'] corresponds to CLIP pooled output
            prompt_embeds = batch["text_embeddings"].to(device=device, dtype=torch_dtype)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device=device, dtype=torch_dtype)

        except (KeyError, IndexError) as e:
            print(f"[WARN] Batch {batch_idx} missing required data. Skipping. Error: {e}")
            continue

        print(f"\n--- Sample {count + 1}/{args.max_samples} ---")
        print(f"  Source: {os.path.basename(source_path)}")
        print(f"  Prompt: {current_prompt[:80]}...")

        with torch.no_grad():
            # Pass embeddings directly to bypass internal encoders
            output = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
            )

        # Save output
        image = output.images[0]
        safe_prompt = (
            "".join(c if c.isalnum() or c in " _-" else "" for c in current_prompt)[:50].strip().replace(" ", "_")
        )
        output_filename = f"flux_{model_name}_sample{count:03d}_{safe_prompt}.png"
        image.save(output_dir / output_filename)
        print(f"  Saved to: {output_filename}")

        count += 1

    print("\n" + "=" * 80 + "\nInference complete!\n" + "=" * 80)


if __name__ == "__main__":
    main()
