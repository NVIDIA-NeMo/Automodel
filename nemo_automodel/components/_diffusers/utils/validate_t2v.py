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

import argparse
import os
import pickle
from pathlib import Path

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V Validation")

    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (optional)")

    # Data - load from .meta files
    p.add_argument("--meta_folder", type=str, required=True, help="Folder containing .meta files with prompts")

    # Generation settings
    p.add_argument("--num_samples", type=int, default=None, help="Number of samples (default: all)")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    # Video settings
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)

    # Output
    p.add_argument("--output_dir", type=str, default="./validation_outputs")

    return p.parse_args()


def load_prompts_from_meta_files(meta_folder: str):
    """
    Load prompts from .meta files.
    Each .meta file contains a 'metadata' dict with 'vila_caption'.

    Returns list of dicts: [{"prompt": "...", "name": "...", "meta_file": "..."}, ...]
    """
    meta_folder = Path(meta_folder)
    meta_files = sorted(list(meta_folder.glob("*.meta")))

    if not meta_files:
        raise FileNotFoundError(f"No .meta files found in {meta_folder}")

    print(f"[INFO] Found {len(meta_files)} .meta files")

    prompts = []

    for meta_file in meta_files:
        try:
            with open(meta_file, "rb") as f:
                data = pickle.load(f)

            # Extract prompt from metadata
            metadata = data.get("metadata", {})
            prompt = metadata.get("vila_caption", "")

            if not prompt:
                print(f"[WARNING] No vila_caption in {meta_file.name}, skipping...")
                continue

            # Get filename without extension
            name = meta_file.stem

            prompts.append({"prompt": prompt, "name": name, "meta_file": str(meta_file)})

        except Exception as e:
            print(f"[WARNING] Failed to load {meta_file.name}: {e}")
            continue

    if not prompts:
        raise ValueError(f"No valid prompts found in {meta_folder}")

    return prompts


def main():
    args = parse_args()

    print("=" * 80)
    print("WAN 2.1 Text-to-Video Validation")
    print("=" * 80)

    # Load prompts from .meta files
    print(f"\n[1] Loading prompts from .meta files in: {args.meta_folder}")
    prompts = load_prompts_from_meta_files(args.meta_folder)

    if args.num_samples:
        prompts = prompts[: args.num_samples]

    print(f"[INFO] Loaded {len(prompts)} prompts")

    # Show first few prompts
    print("\n[INFO] Sample prompts:")
    for i, item in enumerate(prompts[:3]):
        print(f"  {i + 1}. {item['name']}: {item['prompt'][:60]}...")

    # Load pipeline
    print(f"\n[2] Loading pipeline: {args.model_id}")
    pipe = WanPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # Enable VAE optimizations (critical for memory)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    print("[INFO] Enabled VAE slicing and tiling")

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\n[3] Loading checkpoint: {args.checkpoint}")

        # Try EMA checkpoint first (best quality)
        ema_path = os.path.join(args.checkpoint, "ema_shadow.pt")
        consolidated_path = os.path.join(args.checkpoint, "consolidated_model.bin")
        sharded_dir = os.path.join(args.checkpoint, "model")

        if os.path.exists(ema_path):
            print("[INFO] Loading EMA checkpoint (best quality)...")
            ema_state = torch.load(ema_path, map_location="cuda")
            pipe.transformer.load_state_dict(ema_state, strict=True)
            print("[INFO] ✅ Loaded from EMA checkpoint")
        elif os.path.exists(consolidated_path):
            print("[INFO] ############Loading consolidated checkpoint...")
            state_dict = torch.load(consolidated_path, map_location="cuda")
            pipe.transformer.load_state_dict(state_dict, strict=True)
            print("[INFO] ✅ ############Loaded from consolidated checkpoint")
        elif os.path.isdir(sharded_dir) and any(name.endswith(".distcp") for name in os.listdir(sharded_dir)):
            print(f"[INFO] Detected sharded FSDP checkpoint at: {sharded_dir}")
            print("[INFO] Loading sharded checkpoint via PyTorch Distributed Checkpoint (single process)...")

            import torch.distributed as dist
            from torch.distributed.checkpoint import FileSystemReader
            from torch.distributed.checkpoint import load as dist_load
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType
            from torch.distributed.fsdp.api import ShardedStateDictConfig

            # Initialize a single-process group if not already initialized
            init_dist = False
            if not dist.is_initialized():
                dist.init_process_group(backend="gloo", rank=0, world_size=1)
                init_dist = True

            # Wrap current transformer with FSDP to load sharded weights
            base_transformer = pipe.transformer

            # Ensure uniform dtype before FSDP wraps/flattening
            base_transformer.to(dtype=torch.bfloat16)
            fsdp_transformer = FSDP(base_transformer, use_orig_params=True)

            # Configure to expect sharded state dict
            FSDP.set_state_dict_type(
                fsdp_transformer,
                StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            )

            # Load shards into the FSDP-wrapped model
            model_state = fsdp_transformer.state_dict()
            dist_load(state_dict=model_state, storage_reader=FileSystemReader(sharded_dir))
            fsdp_transformer.load_state_dict(model_state)

            # Unwrap back to the original module for inference
            pipe.transformer = fsdp_transformer.module

            # Move to CUDA bf16 for inference
            pipe.transformer.to("cuda", dtype=torch.bfloat16)

            if init_dist:
                dist.destroy_process_group()

            print("[INFO] ✅ Loaded from sharded FSDP checkpoint")
        else:
            print("[WARNING] No consolidated or EMA checkpoint found")
            print("[INFO] Using base model")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate videos
    print("\n[4] Generating videos...")
    print(f"[INFO] Settings: {args.width}x{args.height}, {args.num_frames} frames, {args.num_inference_steps} steps")
    print(f"[INFO] Guidance scale: {args.guidance_scale}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        name = item["name"]

        print(f"\n[{i + 1}/{len(prompts)}] Generating: {name}")
        print(f"  Prompt: {prompt[:80]}...")

        try:
            # Generate from scratch (no latents needed!)
            generator = torch.Generator(device="cuda").manual_seed(args.seed + i)

            output = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).frames[0]

            # Save video
            output_path = os.path.join(args.output_dir, f"{name}.mp4")
            export_to_video(output, output_path, fps=args.fps)

            print(f"  ✅ Saved to {output_path}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("✅ Validation complete!")
    print(f"📁 Videos saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
