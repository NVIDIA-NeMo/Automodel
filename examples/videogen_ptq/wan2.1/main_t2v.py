#!/usr/bin/env python3
# main_t2v.py - Entry point for WAN 2.1 T2V FULL fine-tuning with FSDP + Data Parallel

import argparse

from dist_utils import print0
from trainer_t2v import WanT2VTrainerFSDP


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V FULL fine-tuning with FSDP (intra-node) + DP (inter-node)")

    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID")

    # Training configuration
    p.add_argument("--meta_folder", type=str, required=True, help="Path to folder containing .meta files")
    p.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument(
        "--batch_size_per_node",
        type=int,
        default=1,
        help="Batch size per NODE (not per GPU). Each node processes 1 batch via FSDP.",
    )
    p.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate (lower for full fine-tuning)")

    # Memory optimization
    p.add_argument(
        "--cpu_offload",
        action="store_true",
        default=True,
        help="Enable CPU offloading for parameters (saves GPU memory)",
    )
    p.add_argument(
        "--no_cpu_offload",
        action="store_false",
        dest="cpu_offload",
        help="Disable CPU offloading (faster but uses more GPU memory)",
    )

    # Checkpointing
    p.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    p.add_argument("--log_every", type=int, default=50, help="Log metrics every N steps")
    p.add_argument("--output_dir", type=str, default="./wan_t2v_outputs", help="Output directory for checkpoints")
    p.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")

    return p.parse_args()


def main():
    args = parse_args()

    print0("[INFO] Starting WAN 2.1 T2V FULL fine-tuning with FSDP + Data Parallel")
    print0(f"[INFO] Model: {args.model_id}")
    print0(f"[INFO] Batch size per node: {args.batch_size_per_node}")
    print0(f"[INFO] Learning rate: {args.learning_rate}")
    print0(f"[INFO] CPU offload: {'ENABLED' if args.cpu_offload else 'DISABLED'}")

    # Create trainer
    trainer = WanT2VTrainerFSDP(
        model_id=args.model_id,
        learning_rate=args.learning_rate,
        cpu_offload=args.cpu_offload,
    )

    # Start training
    trainer.train(
        meta_folder=args.meta_folder,
        num_epochs=args.num_epochs,
        batch_size_per_node=args.batch_size_per_node,
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
# Single node with 8 GPUs (FSDP across 8 GPUs, 1 batch):
# torchrun --nproc-per-node=8 main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --learning_rate 1e-5
#
# 10 nodes, 8 GPUs each (80 GPUs total):
# - Each node: FSDP shards model across 8 GPUs
# - Across nodes: Data parallel (each node gets different batch)
# - Total effective batch size: 10 batches (1 per node)
#
# On each node, run:
# torchrun \
#     --nnodes=10 \
#     --nproc-per-node=8 \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=<master_addr>:<master_port> \
#     --rdzv-id=<job_id> \
#     main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --learning_rate 1e-5 \
#     --output_dir ./wan_t2v_outputs
#
# With CPU offload disabled (faster, more memory):
# torchrun --nnodes=10 --nproc-per-node=8 ... main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --no_cpu_offload
#
# Resume from checkpoint:
# torchrun --nnodes=10 --nproc-per-node=8 ... main_t2v.py \
#     --meta_folder /path/to/meta \
#     --resume_checkpoint ./wan_t2v_outputs/checkpoint-5000 \
#     --batch_size_per_node 1