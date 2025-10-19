#!/usr/bin/env python3
# validate_dmd_t2v.py - Validation script for DMD-trained models with custom scheduler

import argparse
import json
import os
import pickle

# Import custom scheduler
import sys
from pathlib import Path

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

sys.path.insert(0, os.path.dirname(__file__))
from rectified_flow_scheduler import RectifiedFlowScheduler


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V DMD Validation")

    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to DMD checkpoint folder")

    # Data - load from .meta files
    p.add_argument("--meta_folder", type=str, required=True, help="Folder containing .meta files with prompts")

    # DMD-specific: discrete timestep schedule
    p.add_argument(
        "--denoising_steps",
        type=str,
        default="999,749,499,249,0",
        help="Discrete denoising steps (must match training)",
    )
    p.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of inference steps (auto-computed from denoising_steps)",
    )

    # Scheduler settings
    p.add_argument(
        "--timestep_shift",
        type=float,
        default=3.0,
        help="Timestep shift parameter (must match training)",
    )

    # Generation settings
    p.add_argument("--num_samples", type=int, default=None, help="Number of samples (default: all)")
    p.add_argument("--guidance_scale", type=float, default=1.0, help="CFG scale (DMD typically uses 1.0, no CFG)")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    # Video settings
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)

    # Output
    p.add_argument("--output_dir", type=str, default="./validation_dmd_outputs")

    return p.parse_args()


def load_prompts_from_meta_files(meta_folder: str):
    """Load prompts from .meta files."""
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

            metadata = data.get("metadata", {})
            prompt = metadata.get("vila_caption", "")

            if not prompt:
                print(f"[WARNING] No vila_caption in {meta_file.name}, skipping...")
                continue

            name = meta_file.stem

            prompts.append({"prompt": prompt, "name": name, "meta_file": str(meta_file)})

        except Exception as e:
            print(f"[WARNING] Failed to load {meta_file.name}: {e}")
            continue

    if not prompts:
        raise ValueError(f"No valid prompts found in {meta_folder}")

    return prompts


def setup_custom_scheduler(pipe, denoising_steps, timestep_shift=3.0):
    """
    Setup custom Rectified Flow scheduler for DMD inference.

    CRITICAL: DMD models are trained with custom Rectified Flow scheduler,
    so inference must use the SAME scheduler with SAME settings.
    """
    print("[SCHEDULER] Setting up custom Rectified Flow scheduler")
    print(f"[SCHEDULER] Denoising steps: {denoising_steps}")
    print(f"[SCHEDULER] Timestep shift: {timestep_shift}")

    # Create custom scheduler
    custom_scheduler = RectifiedFlowScheduler(
        num_train_timesteps=1000,
        shift=timestep_shift,
        use_discrete_timesteps=True,
    )

    # Set discrete timesteps for inference
    # Remove the zero timestep (we don't denoise at t=0)
    inference_steps = [t for t in denoising_steps if t > 0]
    custom_scheduler.timesteps = torch.tensor(inference_steps, dtype=torch.long)
    custom_scheduler.num_inference_steps = len(inference_steps)

    # Replace pipeline scheduler
    pipe.scheduler = custom_scheduler

    print(f"[SCHEDULER] ✓ Configured for {custom_scheduler.num_inference_steps} inference steps")
    print(f"[SCHEDULER] Timesteps: {custom_scheduler.timesteps.tolist()}")

    return pipe


def load_dmd_checkpoint(pipe, checkpoint_path):
    """
    Load DMD generator checkpoint.

    DMD saves checkpoints in a specific structure:
    - generator_consolidated.bin (for inference)
    - generator_model/ (sharded FSDP checkpoint)
    """
    print(f"[CHECKPOINT] Loading DMD checkpoint from: {checkpoint_path}")

    # Try consolidated checkpoint first (recommended for inference)
    generator_consolidated_path = os.path.join(checkpoint_path, "generator_consolidated.bin")

    if os.path.exists(generator_consolidated_path):
        print("[CHECKPOINT] Loading consolidated generator...")
        state_dict = torch.load(generator_consolidated_path, map_location="cuda")
        pipe.transformer.load_state_dict(state_dict, strict=True)
        print("[CHECKPOINT] ✓ Loaded consolidated generator")
        return True

    # Try standard checkpoint as fallback
    consolidated_path = os.path.join(checkpoint_path, "consolidated_model.bin")
    if os.path.exists(consolidated_path):
        print("[CHECKPOINT] Loading standard consolidated model...")
        state_dict = torch.load(consolidated_path, map_location="cuda")
        pipe.transformer.load_state_dict(state_dict, strict=True)
        print("[CHECKPOINT] ✓ Loaded standard checkpoint")
        return True

    # Try loading from sharded checkpoint (slower, but works)
    generator_model_path = os.path.join(checkpoint_path, "generator_model")
    if os.path.exists(generator_model_path):
        print("[WARNING] Only sharded checkpoint found")
        print("[INFO] Attempting to load from sharded checkpoint...")

        # Try to load sharded checkpoint using torch.distributed
        try:
            from torch.distributed.checkpoint import FileSystemReader
            from torch.distributed.checkpoint import load as dist_load

            # Load sharded state dict
            state_dict = {"model": pipe.transformer.state_dict()}
            dist_load(
                state_dict=state_dict,
                storage_reader=FileSystemReader(generator_model_path),
            )
            pipe.transformer.load_state_dict(state_dict["model"], strict=True)
            print("[CHECKPOINT] ✓ Loaded sharded checkpoint")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load sharded checkpoint: {e}")
            raise FileNotFoundError(
                "Sharded checkpoint found but loading failed. "
                "Please run consolidation during training or use consolidated checkpoint."
            )

    raise FileNotFoundError(
        f"No valid checkpoint found at {checkpoint_path}\n"
        f"Expected files:\n"
        f"  - generator_consolidated.bin (recommended)\n"
        f"  - consolidated_model.bin (fallback)\n"
        f"  - generator_model/ (requires distributed loading)"
    )


def main():
    args = parse_args()

    print("=" * 80)
    print("WAN 2.1 Text-to-Video DMD Validation")
    print("=" * 80)

    # Parse denoising steps
    denoising_steps = [int(x.strip()) for x in args.denoising_steps.split(",")]
    print(f"[INFO] Denoising steps: {denoising_steps}")

    # Compute num_inference_steps if not provided
    if args.num_inference_steps is None:
        args.num_inference_steps = len([s for s in denoising_steps if s > 0])

    print(f"[INFO] Number of inference steps: {args.num_inference_steps}")

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

    # Load DMD checkpoint
    print("\n[3] Loading DMD checkpoint")
    load_dmd_checkpoint(pipe, args.checkpoint)

    # Setup custom Rectified Flow scheduler
    print("\n[4] Setting up custom Rectified Flow scheduler")
    pipe = setup_custom_scheduler(pipe, denoising_steps, args.timestep_shift)

    # IMPORTANT: Disable CFG for DMD models
    if args.guidance_scale != 1.0:
        print("\n[WARNING] DMD models typically work best with guidance_scale=1.0")
        print(f"[WARNING] You specified guidance_scale={args.guidance_scale}")
        print("[WARNING] This may produce blurry results!")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save validation config
    config = {
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "denoising_steps": denoising_steps,
        "num_inference_steps": args.num_inference_steps,
        "timestep_shift": args.timestep_shift,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "seed": args.seed,
    }

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Generate videos
    print("\n[5] Generating videos...")
    print(f"[INFO] Settings: {args.width}x{args.height}, {args.num_frames} frames")
    print(f"[INFO] Inference steps: {args.num_inference_steps} (discrete)")
    print(f"[INFO] Guidance scale: {args.guidance_scale}")
    print(f"[INFO] Denoising schedule: {denoising_steps}")
    print(f"[INFO] Timestep shift: {args.timestep_shift}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        name = item["name"]

        print(f"\n[{i + 1}/{len(prompts)}] Generating: {name}")
        print(f"  Prompt: {prompt[:80]}...")

        try:
            # Generate video
            generator = torch.Generator(device="cuda").manual_seed(args.seed + i)

            # IMPORTANT: Custom scheduler handles discrete timesteps
            # The pipeline will automatically use pipe.scheduler
            output = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt if args.guidance_scale > 1.0 else None,
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

            print(f"  ✓ Saved to {output_path}")

            # Log inference info
            print(f"  Steps used: {args.num_inference_steps}")
            print(f"  Timesteps: {pipe.scheduler.timesteps.tolist()}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("✓ Validation complete!")
    print(f"📁 Videos saved to: {args.output_dir}")
    print(f"⚡ Inference speed: {args.num_inference_steps} steps (vs 40-50 for base model)")
    print(f"🚀 Speedup: ~{40 // args.num_inference_steps}x faster!")
    print("=" * 80)


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# 1. Basic validation with DMD checkpoint (4-step model):
# python validate_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --checkpoint ./wan_dmd_outputs/checkpoint-600 \
#     --denoising_steps 999,749,499,249,0 \
#     --timestep_shift 3.0

# 2. Limited samples for quick test:
# python validate_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --checkpoint ./wan_dmd_outputs/checkpoint-600 \
#     --denoising_steps 999,749,499,249,0 \
#     --num_samples 5

# 3. With 3-step model (ultra fast):
# python validate_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --checkpoint ./wan_dmd_outputs/checkpoint-600 \
#     --denoising_steps 999,499,0 \
#     --num_samples 5

# 4. Custom resolution:
# python validate_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --checkpoint ./wan_dmd_outputs/checkpoint-600 \
#     --denoising_steps 999,749,499,249,0 \
#     --height 720 \
#     --width 1280 \
#     --num_frames 121

# 5. Match training settings exactly:
# python validate_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --checkpoint ./wan_dmd_outputs/checkpoint-600 \
#     --denoising_steps 999,749,499,249,0 \
#     --timestep_shift 3.0 \
#     --guidance_scale 1.0

# ============================================================================
# IMPORTANT NOTES
# ============================================================================
#
# 1. ALWAYS use the EXACT same denoising_steps as training!
#    Training: [999, 749, 499, 249, 0]
#    Inference: Must use [999, 749, 499, 249, 0]
#
# 2. ALWAYS use the SAME timestep_shift as training!
#    Default: 3.0 (WAN 2.1 standard)
#
# 3. DMD models work best with guidance_scale=1.0 (no CFG)
#    This is because they're trained without CFG
#
# 4. The speedup comes from fewer steps:
#    Base model: 40-50 steps
#    DMD 4-step: 4 steps → 10x faster!
#    DMD 3-step: 3 steps → 13x faster!
#
# 5. Quality should be comparable to base model despite fewer steps
#    If quality is poor, check:
#    - Are you using the correct denoising_steps?
#    - Is timestep_shift correct (must match training)?
#    - Is guidance_scale set to 1.0?
#    - Did training complete successfully?
#
# 6. Custom scheduler is REQUIRED:
#    - Training uses RectifiedFlowScheduler
#    - Inference must use the SAME scheduler
#    - Diffusers default scheduler will NOT work correctly
