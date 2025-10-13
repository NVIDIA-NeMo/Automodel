#!/usr/bin/env python3
# main_t2v.py - Entry point for WAN 2.1 T2V Flow Matching Training

import argparse

from dist_utils import print0
from trainer_t2v import WanT2VTrainerFSDP


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V Flow Matching Training")

    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HuggingFace model ID")

    # Training configuration
    p.add_argument("--meta_folder", type=str, required=True, help="Path to folder containing .meta files")
    p.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--batch_size_per_node", type=int, default=1, help="Batch size per NODE (not per GPU)")
    p.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")

    # Memory optimization
    p.add_argument("--cpu_offload", action="store_true", default=True, help="Enable CPU offloading for parameters")
    p.add_argument("--no_cpu_offload", action="store_false", dest="cpu_offload", help="Disable CPU offloading")

    # Flow matching arguments
    p.add_argument("--use_sigma_noise", action="store_true", default=True, help="Use flow matching noise scheduling")
    p.add_argument("--no_sigma_noise", action="store_false", dest="use_sigma_noise", help="Disable flow matching")
    p.add_argument(
        "--timestep_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "logit_normal", "mode"],
        help="Timestep sampling strategy",
    )
    p.add_argument("--logit_mean", type=float, default=0.0, help="Mean for logit-normal distribution")
    p.add_argument("--logit_std", type=float, default=1.0, help="Std for logit-normal distribution")
    p.add_argument("--flow_shift", type=float, default=3.0, help="Flow matching shift parameter")
    p.add_argument(
        "--mix_uniform_ratio", type=float, default=0.1, help="Ratio of uniform sampling mixed with density sampling"
    )

    # Checkpointing
    p.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    p.add_argument("--log_every", type=int, default=5, help="Log metrics every N steps")
    p.add_argument("--output_dir", type=str, default="./wan_t2v_flow_outputs", help="Output directory for checkpoints")
    p.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")

    return p.parse_args()


def main():
    args = parse_args()

    print0("[INFO] Starting WAN 2.1 T2V Flow Matching Training")
    print0(f"[INFO] Model: {args.model_id}")
    print0(f"[INFO] Batch size per node: {args.batch_size_per_node}")
    print0(f"[INFO] Learning rate: {args.learning_rate}")
    print0(f"[INFO] CPU offload: {'ENABLED' if args.cpu_offload else 'DISABLED'}")
    print0(f"[INFO] Flow matching: {'ENABLED' if args.use_sigma_noise else 'DISABLED'}")

    # Create trainer
    trainer = WanT2VTrainerFSDP(
        model_id=args.model_id,
        learning_rate=args.learning_rate,
        cpu_offload=args.cpu_offload,
        # Flow matching config
        use_sigma_noise=args.use_sigma_noise,
        timestep_sampling=args.timestep_sampling,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        flow_shift=args.flow_shift,
        mix_uniform_ratio=args.mix_uniform_ratio,
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


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# 1. Basic training with flow matching (default uniform sampling):
# torchrun --nproc-per-node=8 main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --learning_rate 5e-6 \
#     --flow_shift 3.0

# 2. With debug logging:
# DEBUG_TRAINING=1 torchrun --nproc-per-node=8 main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --learning_rate 5e-6 \
#     --flow_shift 3.0 \
#     --timestep_sampling uniform

# 3. Multi-node training (10 nodes × 8 GPUs):
# torchrun \
#     --nnodes=10 \
#     --nproc-per-node=8 \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=<master_addr>:<master_port> \
#     --rdzv-id=<job_id> \
#     main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --learning_rate 5e-6 \
#     --flow_shift 3.0 \
#     --output_dir ./wan_t2v_flow_outputs

# 4. With logit-normal sampling:
# torchrun --nproc-per-node=8 main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --timestep_sampling logit_normal \
#     --logit_mean 0.0 \
#     --logit_std 1.0

# 5. Disable flow matching (simple uniform):
# torchrun --nproc-per-node=8 main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --no_sigma_noise

# 6. Resume from checkpoint:
# torchrun --nproc-per-node=8 main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --resume_checkpoint ./wan_t2v_flow_outputs/checkpoint-5000

# 7. Custom flow shift:
# torchrun --nproc-per-node=8 main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --flow_shift 5.0  # Higher shift = more weight on noisy timesteps

# 8. CPU offload disabled (faster, more memory):
# torchrun --nproc-per-node=8 main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --no_cpu_offload
