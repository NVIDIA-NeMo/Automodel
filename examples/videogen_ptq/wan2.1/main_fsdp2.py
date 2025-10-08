#!/usr/bin/env python3
# main_fsdp2.py - Entry point for WAN I2V LoRA training with FSDP2

import argparse
from trainer_fsdp2 import WanI2VLoRATrainerFSDP2
from dist_utils import print0


def parse_args():
    p = argparse.ArgumentParser("WAN 2.2 I2V LoRA with FSDP2")
    
    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                   help="HuggingFace model ID")
    
    # LoRA configuration
    p.add_argument("--lora_rank", type=int, default=16,
                   help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha (scaling factor)")
    
    # Training configuration
    p.add_argument("--meta_folder", type=str, required=True,
                   help="Path to folder containing .meta files")
    p.add_argument("--num_epochs", type=int, default=10,
                   help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size per GPU")
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="Learning rate")
    
    # Dual transformer configuration
    p.add_argument("--boundary_ratio", type=float, default=0.5,
                   help="Timestep boundary ratio for dual transformer (0.0 to disable)")
    p.add_argument("--train_transformer_2", action="store_true",
                   help="Train transformer_2 instead of transformer")
    
    # Memory optimization
    p.add_argument("--cpu_offload", action="store_true", default=True,
                   help="Enable CPU offloading for parameters (saves GPU memory)")
    p.add_argument("--no_cpu_offload", action="store_false", dest="cpu_offload",
                   help="Disable CPU offloading (faster but uses more GPU memory)")
    
    # Checkpointing
    p.add_argument("--save_every", type=int, default=500,
                   help="Save checkpoint every N steps")
    p.add_argument("--log_every", type=int, default=50,
                   help="Log metrics every N steps")
    p.add_argument("--output_dir", type=str, default="./wan_i2v_fsdp2_outputs",
                   help="Output directory for checkpoints")
    p.add_argument("--resume_checkpoint", type=str, default=None,
                   help="Path to checkpoint to resume from")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print0("[INFO] Starting WAN I2V LoRA training with FSDP2")
    print0(f"[INFO] Model: {args.model_id}")
    print0(f"[INFO] LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print0(f"[INFO] Training: {'transformer_2' if args.train_transformer_2 else 'transformer'}")
    print0(f"[INFO] Batch size: {args.batch_size} per GPU")
    print0(f"[INFO] CPU offload: {'ENABLED' if args.cpu_offload else 'DISABLED'}")
    
    # Create trainer
    trainer = WanI2VLoRATrainerFSDP2(
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        boundary_ratio=args.boundary_ratio,
        train_transformer_2=args.train_transformer_2,
        cpu_offload=args.cpu_offload,
    )
    
    # Start training
    trainer.train(
        meta_folder=args.meta_folder,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_every=args.save_every,
        log_every=args.log_every,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
    )
    
    print0("[INFO] Training complete!")


if __name__ == "__main__":
    main()


# Example usage:
# 
# Single GPU:
# python main_fsdp2.py --meta_folder /path/to/meta --batch_size 1
#
# Multi-GPU (FSDP2 automatically shards across all GPUs):
# torchrun --nproc-per-node=4 main_fsdp2.py --meta_folder /path/to/meta --batch_size 1
#
# 8 GPU training:
# torchrun --nproc-per-node=8 main_fsdp2.py --meta_folder /path/to/meta --batch_size 1
#
# Train transformer_2 (low timesteps):
# torchrun --nproc-per-node=4 main_fsdp2.py --meta_folder /path/to/meta --train_transformer_2 --batch_size 1
#
# Enable CPU offload for maximum memory savings (slower):
# torchrun --nproc-per-node=4 main_fsdp2.py --meta_folder /path/to/meta --cpu_offload --batch_size 1
#
# Resume from checkpoint:
# torchrun --nproc-per-node=4 main_fsdp2.py --meta_folder /path/to/meta --resume_checkpoint ./outputs/checkpoint-5000 --batch_size 1