#!/usr/bin/env python3
# main_dmd_t2v.py - Entry point for Pure DMD Training

import argparse
from dist_utils import print0
from trainer_dmd_t2v import WanDMDTrainerT2V


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V Pure DMD Training")
    
    # Model configuration
    p.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="HuggingFace model ID for base/generator model",
    )
    p.add_argument(
        "--teacher_model_path",
        type=str,
        default=None,
        help="Optional path to teacher model checkpoint",
    )
    
    # Training configuration
    p.add_argument(
        "--meta_folder",
        type=str,
        required=True,
        help="Path to folder containing .meta files",
    )
    p.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    p.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=1,
        help="Batch size per GPU",
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for student",
    )
    p.add_argument(
        "--critic_learning_rate",
        type=float,
        default=None,
        help="Learning rate for critic (defaults to same as student)",
    )
    
    # Memory optimization
    p.add_argument(
        "--cpu_offload",
        action="store_true",
        default=True,
        help="Enable CPU offloading",
    )
    p.add_argument(
        "--no_cpu_offload",
        action="store_false",
        dest="cpu_offload",
        help="Disable CPU offloading",
    )
    
    # DMD parameters
    p.add_argument(
        "--num_train_timestep",
        type=int,
        default=1000,
        help="Total training timesteps",
    )
    p.add_argument(
        "--min_step",
        type=int,
        default=20,
        help="Minimum timestep for sampling",
    )
    p.add_argument(
        "--max_step",
        type=int,
        default=980,
        help="Maximum timestep for sampling",
    )
    p.add_argument(
        "--real_guidance_scale",
        type=float,
        default=5.0,
        help="CFG scale for teacher model",
    )
    p.add_argument(
        "--fake_guidance_scale",
        type=float,
        default=0.0,
        help="CFG scale for critic model",
    )
    p.add_argument(
        "--timestep_shift",
        type=float,
        default=3.0,
        help="Rectified Flow shift parameter",
    )
    p.add_argument(
        "--loss_weight_type",
        type=str,
        default="constant",
        choices=["constant", "sigma"],
        help="Type of loss weighting",
    )
    p.add_argument(
        "--loss_weight_scale",
        type=float,
        default=1.0,
        help="Loss weight scaling factor",
    )
    
    # Alternating optimization
    p.add_argument(
        "--critic_steps",
        type=int,
        default=2,
        help="Number of critic updates per iteration (start with 2:1 ratio)",
    )
    p.add_argument(
        "--student_steps",
        type=int,
        default=1,
        help="Number of student updates per iteration",
    )
    
    # Checkpointing
    p.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./wan_dmd_outputs",
        help="Output directory for checkpoints",
    )
    p.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print0("=" * 80)
    print0("WAN 2.1 T2V Pure DMD Training")
    print0("=" * 80)
    print0(f"[INFO] Base model: {args.model_id}")
    if args.teacher_model_path:
        print0(f"[INFO] Teacher model: {args.teacher_model_path}")
    print0(f"[INFO] Batch size per GPU: {args.batch_size_per_gpu}")
    print0(f"[INFO] Student LR: {args.learning_rate}")
    print0(f"[INFO] Critic LR: {args.critic_learning_rate or args.learning_rate}")
    print0(f"[INFO] CPU offload: {'ENABLED' if args.cpu_offload else 'DISABLED'}")
    print0("[INFO] DMD config:")
    print0(f"  - Real guidance scale: {args.real_guidance_scale}")
    print0(f"  - Fake guidance scale: {args.fake_guidance_scale}")
    print0(f"  - Timestep range: [{args.min_step}, {args.max_step}]")
    print0(f"  - Timestep shift: {args.timestep_shift}")
    print0(f"  - Loss weight type: {args.loss_weight_type}")
    print0(f"  - Alternating: {args.critic_steps} critic / {args.student_steps} student")
    print0("=" * 80)
    
    # Create trainer
    trainer = WanDMDTrainerT2V(
        model_id=args.model_id,
        teacher_model_path=args.teacher_model_path,
        learning_rate=args.learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        cpu_offload=args.cpu_offload,
        # DMD parameters
        num_train_timestep=args.num_train_timestep,
        min_step=args.min_step,
        max_step=args.max_step,
        real_guidance_scale=args.real_guidance_scale,
        fake_guidance_scale=args.fake_guidance_scale,
        timestep_shift=args.timestep_shift,
        loss_weight_type=args.loss_weight_type,
        loss_weight_scale=args.loss_weight_scale,
        # Alternating optimization
        critic_steps=args.critic_steps,
        student_steps=args.student_steps,
    )
    
    # Start training
    trainer.train(
        meta_folder=args.meta_folder,
        num_epochs=args.num_epochs,
        batch_size_per_gpu=args.batch_size_per_gpu,
        save_every=args.save_every,
        log_every=args.log_every,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
    )
    
    print0("=" * 80)
    print0("[INFO] Pure DMD Training complete!")
    print0("=" * 80)


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# 1. Basic pure DMD training (single node, 8 GPUs):
# torchrun --nproc-per-node=8 main_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_gpu 1 \
#     --learning_rate 1e-5 \
#     --num_epochs 3

# 2. With sigma-weighted loss:
# torchrun --nproc-per-node=8 main_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --loss_weight_type sigma \
#     --loss_weight_scale 2.0

# 3. Multi-node training (8 nodes × 8 GPUs):
# torchrun \
#     --nnodes=8 \
#     --nproc-per-node=8 \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
#     main_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_gpu 1 \
#     --output_dir ./wan_dmd_outputs

# 4. With debug logging:
# DEBUG_TRAINING=1 torchrun --nproc-per-node=8 main_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_gpu 1

# 5. Custom critic/student ratio (3:1 for initial stabilization):
# torchrun --nproc-per-node=8 main_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --critic_steps 3 \
#     --student_steps 1

# 6. Resume from checkpoint:
# torchrun --nproc-per-node=8 main_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --resume_checkpoint ./wan_dmd_outputs/checkpoint-500

# ============================================================================
# KEY DIFFERENCES FROM SELF-FORCING VERSION
# ============================================================================
#
# 1. No backward simulation:
#    - OLD: Multi-step denoising through discrete timesteps
#    - NEW: Single random timestep per batch
#
# 2. Simpler pipeline:
#    - OLD: SelfForcingTrainingPipeline with exit logic
#    - NEW: SimpleDMDPipeline with single forward pass
#
# 3. Memory efficiency:
#    - OLD: Caches intermediate timesteps
#    - NEW: Only computes at one timestep
#
# 4. Cleaner gradients:
#    - OLD: Complex gradient masking at exit step
#    - NEW: Simple detach for critic/teacher in student update
#
# 5. Standard DMD:
#    - This implements the original DMD algorithm from the paper
#    - More stable, easier to debug
#    - Better matches typical distillation workflows
#
# ============================================================================