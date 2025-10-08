#!/usr/bin/env python3
# validate_i2v.py - Validation script for WAN I2V LoRA model

import argparse
from datetime import timedelta
import os
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image
import numpy as np

from dataloader import MetaFilesDataset


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
    """Call barrier only when a default group exists and we have >1 ranks."""
    if dist.is_available() and dist.is_initialized():
        if dist.get_world_size() > 1:
            # For NCCL, specifying device_ids avoids certain hangs on multi-GPU nodes
            try:
                dist.barrier(device_ids=[torch.cuda.current_device()])
            except TypeError:
                # older PyTorch versions don't accept device_ids; fall back
                dist.barrier()


def is_main_process(rank):
    """Check if this is the main process."""
    return rank == 0


def parse_args():
    p = argparse.ArgumentParser("WAN 2.2 I2V Validation")
    
    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                   help="HuggingFace model ID")
    
    # LoRA configuration
    p.add_argument("--lora_checkpoint", type=str, default=None,
                   help="Path to LoRA checkpoint for transformer (high noise)")
    p.add_argument("--lora_checkpoint_2", type=str, default=None,
                   help="Path to LoRA checkpoint for transformer_2 (low noise)")
    p.add_argument("--lora_rank", type=int, default=16,
                   help="LoRA rank (must match training)")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha (must match training)")
    
    # Data configuration
    p.add_argument("--meta_folder", type=str, required=True,
                   help="Path to folder containing .meta files")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Number of samples to validate (default: all)")
    
    # Generation configuration
    p.add_argument("--num_inference_steps", type=int, default=40,
                   help="Number of inference steps")
    p.add_argument("--guidance_scale", type=float, default=3.5,
                   help="Guidance scale")
    p.add_argument("--negative_prompt", type=str, default="",
                   help="Negative prompt for generation")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--fps", type=int, default=16,
                   help="FPS for output video")
    
    # Output configuration
    p.add_argument("--output_dir", type=str, default="./validation_outputs",
                   help="Output directory for generated videos")
    
    return p.parse_args()


class WanI2VValidator:
    """Simple validator for WAN I2V model with optional LoRA."""
    
    def __init__(
        self,
        model_id: str,
        lora_checkpoint: Optional[str] = None,
        lora_checkpoint_2: Optional[str] = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        local_rank: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model_id = model_id
        self.lora_checkpoint = lora_checkpoint  # For transformer (high noise)
        self.lora_checkpoint_2 = lora_checkpoint_2  # For transformer_2 (low noise)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.dtype = torch.bfloat16
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        
        if is_main_process(rank):
            print(f"[INFO] WAN I2V Validator")
            print(f"[INFO] World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
            print(f"[INFO] Model: {model_id}")
            print(f"[INFO] Transformer LoRA: {lora_checkpoint if lora_checkpoint else 'None (base)'}")
            print(f"[INFO] Transformer_2 LoRA: {lora_checkpoint_2 if lora_checkpoint_2 else 'None (base)'}")
        
        self.pipe = None
    
    def setup_pipeline(self):
        """Load and setup the WAN pipeline."""
        if is_main_process(self.rank):
            print(f"[INFO] Rank {self.rank}: Loading WAN pipeline...")
        
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=self.dtype
        )
        self.pipe.to(self.device)
        
        if is_main_process(self.rank):
            print(f"[INFO] Rank {self.rank}: Pipeline loaded")
    
    def _install_lora_to_transformer(self, transformer, transformer_name: str):
        """Install LoRA modules to a transformer."""
        from lora_utils import wan_install_and_materialize_lora
        
        num_lora_modules = wan_install_and_materialize_lora(
            transformer,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=0.0  # No dropout for inference
        )
        
        if is_main_process(self.rank):
            print(f"[INFO] Rank {self.rank}: Installed {num_lora_modules} LoRA modules to {transformer_name}")
        return num_lora_modules
    
    def _load_lora_weights(self, transformer, lora_checkpoint: str, transformer_name: str):
        """Load LoRA weights from checkpoint."""
        from lora_utils import LoRALinear
        
        # Check for LoRA weights file
        lora_path = os.path.join(lora_checkpoint, f"{transformer_name}_lora_weights.pt")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA weights not found at {lora_path}")
        
        # Load LoRA state dict
        lora_state_dict = torch.load(lora_path, map_location="cpu")
        lora_state_dict = {k.replace('._checkpoint_wrapped_module._fsdp_wrapped_module.', '.'): lora_state_dict[k] for k in lora_state_dict.keys()}
        
        # Load LoRA parameters into modules
        loaded_count = 0
        for module_name, module in transformer.named_modules():
            if isinstance(module, LoRALinear):
                a_key = f"{module_name}.lora_A"
                b_key = f"{module_name}.lora_B"
                
                if a_key in lora_state_dict and b_key in lora_state_dict:
                    module.A.data.copy_(lora_state_dict[a_key].to(self.device))
                    module.B.data.copy_(lora_state_dict[b_key].to(self.device))
                    loaded_count += 1
        
        if is_main_process(self.rank):
            print(f"[INFO] Rank {self.rank}: Loaded {loaded_count} LoRA modules for {transformer_name}")
        
        if loaded_count == 0:
            raise RuntimeError(f"No LoRA weights loaded for {transformer_name}")
    
    def load_lora_if_needed(self):
        """Load LoRA weights if checkpoints provided."""
        # Load LoRA for transformer (high noise)
        if self.lora_checkpoint is not None:
            if is_main_process(self.rank):
                print(f"[INFO] Rank {self.rank}: Loading LoRA for transformer from {self.lora_checkpoint}")
            
            # Install LoRA modules
            self._install_lora_to_transformer(self.pipe.transformer, "transformer")
            
            # Load weights
            self._load_lora_weights(
                self.pipe.transformer, 
                self.lora_checkpoint, 
                "transformer"
            )
        else:
            if is_main_process(self.rank):
                print(f"[INFO] Rank {self.rank}: No LoRA checkpoint for transformer, using base model")
        
        # Load LoRA for transformer_2 (low noise)
        if self.lora_checkpoint_2 is not None:
            if not hasattr(self.pipe, "transformer_2") or self.pipe.transformer_2 is None:
                if is_main_process(self.rank):
                    print(f"[WARNING] Rank {self.rank}: transformer_2 not found in pipeline, skipping")
            else:
                if is_main_process(self.rank):
                    print(f"[INFO] Rank {self.rank}: Loading LoRA for transformer_2 from {self.lora_checkpoint_2}")
                
                # Install LoRA modules
                self._install_lora_to_transformer(self.pipe.transformer_2, "transformer_2")
                
                # Load weights
                self._load_lora_weights(
                    self.pipe.transformer_2,
                    self.lora_checkpoint_2,
                    "transformer_2"
                )
        else:
            if is_main_process(self.rank):
                print(f"[INFO] Rank {self.rank}: No LoRA checkpoint for transformer_2, using base model")
    
    def resize_image(self, image: Image.Image, max_area: int = 480 * 832) -> tuple:
        """Resize image following WAN pipeline's requirements."""
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        
        image = image.resize((width, height))
        return image, height, width
    
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
    ):
        """Run validation on dataset."""
        if is_main_process(self.rank):
            print(f"[INFO] Starting validation with {self.world_size} GPUs...")
        
        # Setup pipeline
        self.setup_pipeline()
        self.load_lora_if_needed()
        
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
            # Round-robin assignment: rank 0 gets 0,8,16... rank 1 gets 1,9,17...
            if idx % self.world_size != self.rank:
                continue
            
            try:
                # Load data
                data = dataset[idx]
                
                # Extract information
                first_frame = data['first_frame']  # RGB image
                metadata = data['metadata']
                file_info = data['file_info']
                
                prompt = metadata.get('vila_caption', '')
                num_frames = file_info.get('num_frames', 16)
                meta_filename = file_info['meta_filename']
                
                print(f"[Rank {self.rank}] Processing {idx+1}/{total_samples}: {meta_filename}")
                print(f"[Rank {self.rank}] Prompt: {prompt[:100]}...")
                print(f"[Rank {self.rank}] Generating {num_frames} frames")
                
                # Convert first frame to PIL Image
                # Handle both numpy array and torch tensor
                if isinstance(first_frame, np.ndarray):
                    # Already numpy array
                    if first_frame.ndim == 3:
                        # If [H, W, C], use directly
                        if first_frame.shape[2] == 3:
                            first_frame_np = first_frame
                        # If [C, H, W], transpose
                        elif first_frame.shape[0] == 3:
                            first_frame_np = first_frame.transpose(1, 2, 0)
                        else:
                            raise ValueError(f"Unexpected first_frame shape: {first_frame.shape}")
                    else:
                        raise ValueError(f"Unexpected first_frame dimensions: {first_frame.shape}")
                else:
                    # Torch tensor - convert to numpy
                    if first_frame.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                        # If in [-1, 1] range, denormalize
                        if first_frame.min() < 0:
                            first_frame = (first_frame + 1.0) / 2.0
                        
                        # Convert to [0, 255] uint8
                        first_frame_np = (first_frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    else:
                        first_frame_np = first_frame.permute(1, 2, 0).cpu().numpy()
                
                # Ensure uint8 format
                if first_frame_np.dtype != np.uint8:
                    if first_frame_np.max() <= 1.0:
                        first_frame_np = (first_frame_np * 255).astype(np.uint8)
                    else:
                        first_frame_np = first_frame_np.astype(np.uint8)
                
                image = Image.fromarray(first_frame_np)
                
                # Resize image according to WAN requirements
                image, height, width = self.resize_image(image)
                
                print(f"[Rank {self.rank}] Image size: {width}x{height}")
                
                # Generate video
                generator = torch.Generator(device=self.device).manual_seed(seed + idx)
                
                output = self.pipe(
                    image=image,
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
                output_filename = meta_filename.replace('.meta', '.mp4')
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
    local_rank, rank, world_size = setup_distributed()
    
    args = parse_args()
    
    validator = WanI2VValidator(
        model_id=args.model_id,
        lora_checkpoint=args.lora_checkpoint,
        lora_checkpoint_2=args.lora_checkpoint_2,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
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
    )
    
    # Cleanup
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


# Example usage:
#
# Single GPU (base model):
# python validate_i2v.py --meta_folder /path/to/meta --output_dir ./base_outputs
#
# 8 GPUs (base model):
# torchrun --nproc-per-node=8 validate_i2v.py \
#     --meta_folder /path/to/meta \
#     --output_dir ./base_outputs \
#     --num_samples 80
#
# 8 GPUs with transformer LoRA (high noise):
# torchrun --nproc-per-node=8 validate_i2v.py \
#     --meta_folder /path/to/meta \
#     --lora_checkpoint ./outputs/checkpoint-5000 \
#     --output_dir ./lora_outputs \
#     --num_samples 80
#
# 8 GPUs with BOTH LoRAs (dual transformer):
# torchrun --nproc-per-node=8 validate_i2v.py \
#     --meta_folder /path/to/meta \
#     --lora_checkpoint ./outputs/checkpoint-5000 \
#     --lora_checkpoint_2 ./outputs_t2/checkpoint-5000 \
#     --output_dir ./lora_outputs \
#     --num_samples 80
#
# Custom generation parameters (8 GPUs):
# torchrun --nproc-per-node=8 validate_i2v.py \
#     --meta_folder /path/to/meta \
#     --lora_checkpoint ./outputs/checkpoint-5000 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 24 \
#     --output_dir ./lora_outputs \
#     --num_samples 80     --meta_folder /path/to/meta \
#     --lora_checkpoint ./outputs/checkpoint-5000 \
#     --num_samples 10 \
#     --output_dir ./lora_outputs
#
# Custom generation parameters:
# python validate_i2v.py \
#     --meta_folder /path/to/meta \
#     --lora_checkpoint ./outputs/checkpoint-5000 \
#     --num_inference_steps 40 \
#     --guidance_scale 3.5 \
#     --fps 16 \
#     --output_dir ./lora_outputs