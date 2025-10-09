#!/usr/bin/env python3
# validate_t2v.py - Validation script for WAN 2.1 T2V fine-tuned model

import argparse
import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from dataloader import MetaFilesDataset
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from torch.distributed.checkpoint import load as dist_load
from torch.distributed.checkpoint import FileSystemReader


def setup_distributed():
    """Initialize torch.distributed and return (rank, local_rank, world_size)."""
    rank = 0
    local_rank = 0
    world_size = 1

    if dist.is_available() and (os.environ.get("RANK") is not None or os.environ.get("LOCAL_RANK") is not None):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=30),
            )

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def barrier_if_distributed():
    """Call barrier only when distributed is initialized."""
    if dist.is_available() and dist.is_initialized():
        if dist.get_world_size() > 1:
            try:
                dist.barrier(device_ids=[torch.cuda.current_device()])
            except TypeError:
                dist.barrier()


def is_main_process(rank):
    """Check if this is the main process."""
    return rank == 0


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V Validation")

    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID")

    # Checkpoint configuration
    p.add_argument("--checkpoint", type=str, default=None, help="Path to FSDP checkpoint (optional)")

    # Data configuration
    p.add_argument("--meta_folder", type=str, required=True, help="Path to folder containing .meta files")
    p.add_argument("--num_samples", type=int, default=None, help="Number of samples to validate (default: all)")

    # Generation configuration
    p.add_argument("--num_inference_steps", type=int, default=40, help="Number of inference steps")
    p.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    p.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for generation")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--fps", type=int, default=16, help="FPS for output video")
    p.add_argument("--height", type=int, default=480, help="Video height")
    p.add_argument("--width", type=int, default=832, help="Video width")
    p.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate")

    # Output configuration
    p.add_argument(
        "--output_dir", type=str, default="./t2v_validation_outputs", help="Output directory for generated videos"
    )

    return p.parse_args()


class WanT2VValidator:
    """Simple validator for WAN 2.1 T2V model with optional checkpoint."""

    def __init__(
        self,
        model_id: str,
        checkpoint: Optional[str] = None,
        local_rank: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.dtype = torch.bfloat16
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        if is_main_process(rank):
            print("[INFO] WAN 2.1 T2V Validator")
            print(f"[INFO] World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
            print(f"[INFO] Model: {model_id}")
            print(f"[INFO] Checkpoint: {checkpoint if checkpoint else 'None (base model)'}")

        self.pipe = None

    def setup_pipeline(self):
        """Load and setup the WAN T2V pipeline."""
        if is_main_process(self.rank):
            print(f"[INFO] Rank {self.rank}: Loading WAN 2.1 T2V pipeline...")

        self.pipe = WanPipeline.from_pretrained(self.model_id, torch_dtype=self.dtype)
        self.pipe.to(self.device)

        if is_main_process(self.rank):
            print(f"[INFO] Rank {self.rank}: Pipeline loaded")

    def load_checkpoint_if_needed(self):
        """Load checkpoint weights if provided."""
        if self.checkpoint is not None:
            if is_main_process(self.rank):
                print(f"[INFO] Rank {self.rank}: Loading checkpoint from {self.checkpoint}")

            model_path = os.path.join(self.checkpoint, "transformer_model")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Checkpoint not found at {model_path}")

            # Load transformer state dict
            model_state_dict = {"model": self.pipe.transformer.state_dict()}

            dist_load(
                state_dict=model_state_dict,
                storage_reader=FileSystemReader(model_path),
            )

            self.pipe.transformer.load_state_dict(model_state_dict["model"])

            if is_main_process(self.rank):
                print(f"[INFO] Rank {self.rank}: Checkpoint loaded successfully")
        else:
            if is_main_process(self.rank):
                print(f"[INFO] Rank {self.rank}: No checkpoint, using base model")

    @torch.no_grad()
    def validate(
        self,
        meta_folder: str,
        output_dir: str,
        num_samples: Optional[int] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.5,
        negative_prompt: str = "",
        seed: int = 42,
        fps: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 16,
    ):
        """Run validation on dataset."""
        if is_main_process(self.rank):
            print(f"[INFO] Starting T2V validation with {self.world_size} GPUs...")

        # Setup pipeline
        self.setup_pipeline()
        self.load_checkpoint_if_needed()

        # Synchronize after loading
        if self.world_size > 1:
            barrier_if_distributed()

        # Create output directory (only rank 0)
        if is_main_process(self.rank):
            os.makedirs(output_dir, exist_ok=True)

        if self.world_size > 1:
            barrier_if_distributed()

        # Load dataset
        dataset = MetaFilesDataset(meta_folder=meta_folder, device="cpu")

        # Determine samples to process
        total_samples = len(dataset)
        if num_samples is not None:
            total_samples = min(num_samples, total_samples)

        if is_main_process(self.rank):
            print(f"[INFO] Processing {total_samples} samples across {self.world_size} GPUs")
            print(f"[INFO] Each GPU will process ~{total_samples // self.world_size} samples")

        # Set random seed (different per rank for variety)
        torch.manual_seed(seed + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + self.rank)

        # Track progress per rank
        processed_count = 0

        # Process samples assigned to this rank
        for idx in range(total_samples):
            # Round-robin assignment
            if idx % self.world_size != self.rank:
                continue

            try:
                # Load data
                data = dataset[idx]

                # Extract information
                metadata = data["metadata"]
                file_info = data["file_info"]

                prompt = metadata.get("vila_caption", "")
                meta_filename = file_info["meta_filename"]

                print(f"[Rank {self.rank}] Processing {idx + 1}/{total_samples}: {meta_filename}")
                print(f"[Rank {self.rank}] Prompt: {prompt[:100]}...")
                print(f"[Rank {self.rank}] Generating {num_frames} frames at {width}x{height}")

                # Generate video
                generator = torch.Generator(device=self.device).manual_seed(seed + idx)

                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).frames[0]

                # Save video
                output_filename = meta_filename.replace(".meta", ".mp4")
                output_path = os.path.join(output_dir, output_filename)

                export_to_video(output, output_path, fps=fps)

                processed_count += 1
                print(f"[Rank {self.rank}] Saved to {output_path} ({processed_count} completed)")

            except Exception as e:
                print(f"[Rank {self.rank}] ERROR: Failed to process sample {idx}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Wait for all ranks to complete
        if self.world_size > 1:
            barrier_if_distributed()

        if is_main_process(self.rank):
            print("\n[INFO] Validation complete!")
            print(f"[INFO] Videos saved to {output_dir}")


def main():
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    args = parse_args()

    validator = WanT2VValidator(
        model_id=args.model_id,
        checkpoint=args.checkpoint,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
    )

    validator.validate(
        meta_folder=args.meta_folder,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        fps=args.fps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )

    # Cleanup
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


# Example usage:
#
# Single GPU (base model):
# python validate_t2v.py --meta_folder /path/to/meta --output_dir ./t2v_base_outputs
#
# 8 GPUs (base model):
# torchrun --nproc-per-node=8 validate_t2v.py \
#     --meta_folder /path/to/meta \
#     --output_dir ./t2v_base_outputs \
#     --num_samples 80
#
# 8 GPUs with checkpoint:
# torchrun --nproc-per-node=8 validate_t2v.py \
#     --meta_folder /path/to/meta \
#     --checkpoint ./wan_t2v_outputs/checkpoint-5000 \
#     --output_dir ./t2v_finetuned_outputs \
#     --num_samples 80
#
# Custom generation parameters:
# torchrun --nproc-per-node=8 validate_t2v.py \
#     --meta_folder /path/to/meta \
#     --checkpoint ./wan_t2v_outputs/checkpoint-5000 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --height 720 \
#     --width 1280 \
#     --num_frames 24 \
#     --fps 24 \
#     --output_dir ./t2v_finetuned_outputs \
#     --num_samples 80