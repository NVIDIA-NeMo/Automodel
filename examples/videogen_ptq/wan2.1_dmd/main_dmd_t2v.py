#!/usr/bin/env python3
# main_dmd_t2v.py - Entry point for WAN 2.1 T2V DMD Training with Self-Forcing

import argparse

from dist_utils import print0
from trainer_dmd_t2v import WanDMDTrainerT2V


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V DMD Training with Self-Forcing")

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
        help="Optional path to teacher model checkpoint (if different from base)",
    )

    # Training configuration
    p.add_argument("--meta_folder", type=str, required=True, help="Path to folder containing .meta files")
    p.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (DMD with self-forcing trains fast, 1 epoch often sufficient)",
    )
    p.add_argument("--batch_size_per_gpu", type=int, default=1, help="Batch size per GPU")
    p.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for generator")
    p.add_argument(
        "--critic_learning_rate",
        type=float,
        default=None,
        help="Learning rate for critic (defaults to same as generator)",
    )

    # Memory optimization
    p.add_argument("--cpu_offload", action="store_true", default=True, help="Enable CPU offloading for parameters")
    p.add_argument("--no_cpu_offload", action="store_false", dest="cpu_offload", help="Disable CPU offloading")

    # DMD parameters
    p.add_argument("--num_train_timestep", type=int, default=1000, help="Total training timesteps")
    p.add_argument("--min_step", type=int, default=20, help="Minimum timestep for noise")
    p.add_argument("--max_step", type=int, default=999, help="Maximum timestep for noise (999 for WAN 2.1)")
    p.add_argument("--real_guidance_scale", type=float, default=5.0, help="CFG scale for teacher model")
    p.add_argument("--fake_guidance_scale", type=float, default=0.0, help="CFG scale for critic model")
    p.add_argument("--timestep_shift", type=float, default=3.0, help="Flow matching shift parameter")
    p.add_argument("--ts_schedule", action="store_true", default=True, help="Use dynamic timestep scheduling")
    p.add_argument(
        "--no_ts_schedule", action="store_false", dest="ts_schedule", help="Disable dynamic timestep scheduling"
    )
    p.add_argument("--ts_schedule_max", action="store_true", default=False, help="Use max timestep scheduling")
    p.add_argument("--min_score_timestep", type=int, default=0, help="Minimum timestep for critic")
    p.add_argument(
        "--denoising_loss_type",
        type=str,
        default="flow",
        choices=["flow", "epsilon", "x0"],
        help="Type of denoising loss for critic",
    )

    # Self-forcing backward simulation
    p.add_argument(
        "--denoising_step_list",
        type=str,
        default="999,749,499,249,0",
        help="Comma-separated list of discrete timesteps for self-forcing backward simulation",
    )

    # Alternating optimization
    p.add_argument("--generator_steps", type=int, default=1, help="Number of generator updates per iteration")
    p.add_argument("--critic_steps", type=int, default=1, help="Number of critic updates per iteration")

    # Checkpointing
    p.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    p.add_argument("--log_every", type=int, default=10, help="Log metrics every N steps")
    p.add_argument("--output_dir", type=str, default="./wan_dmd_t2v_outputs", help="Output directory for checkpoints")
    p.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")

    return p.parse_args()


def main():
    args = parse_args()

    print0("=" * 80)
    print0("WAN 2.1 T2V DMD Training with Self-Forcing")
    print0("=" * 80)
    print0(f"[INFO] Base model: {args.model_id}")
    if args.teacher_model_path:
        print0(f"[INFO] Teacher model: {args.teacher_model_path}")
    print0(f"[INFO] Batch size per GPU: {args.batch_size_per_gpu}")
    print0(f"[INFO] Generator LR: {args.learning_rate}")
    print0(f"[INFO] Critic LR: {args.critic_learning_rate or args.learning_rate}")
    print0(f"[INFO] CPU offload: {'ENABLED' if args.cpu_offload else 'DISABLED'}")
    print0("[INFO] DMD config:")
    print0(f"  - Real guidance scale: {args.real_guidance_scale}")
    print0(f"  - Fake guidance scale: {args.fake_guidance_scale}")
    print0(f"  - Timestep shift: {args.timestep_shift}")
    print0(f"  - Denoising loss type: {args.denoising_loss_type}")
    print0(f"  - Alternating: {args.generator_steps} gen / {args.critic_steps} critic")
    print0("[INFO] Self-Forcing: Pipeline internal to DMD model")
    print0("=" * 80)

    # Parse denoising step list
    denoising_step_list = [int(x.strip()) for x in args.denoising_step_list.split(",")]
    print0(f"[INFO] Self-forcing denoising steps: {denoising_step_list}")

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
        ts_schedule=args.ts_schedule,
        ts_schedule_max=args.ts_schedule_max,
        min_score_timestep=args.min_score_timestep,
        denoising_loss_type=args.denoising_loss_type,
        # Self-forcing backward simulation
        denoising_step_list=denoising_step_list,
        # Alternating optimization
        generator_steps=args.generator_steps,
        critic_steps=args.critic_steps,
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
    print0("[INFO] DMD Training with Self-Forcing complete!")
    print0("=" * 80)


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# 1. Basic DMD training with self-forcing (single node, 8 GPUs):
# torchrun --nproc-per-node=8 main_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_gpu 1 \
#     --learning_rate 1e-5 \
#     --num_epochs 1

# 2. Multi-node training (8 nodes × 8 GPUs):
# torchrun \
#     --nnodes=8 \
#     --nproc-per-node=8 \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
#     main_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_gpu 1 \
#     --learning_rate 1e-5 \
#     --output_dir ./wan_dmd_outputs

# 3. With debug logging:
# DEBUG_TRAINING=1 torchrun --nproc-per-node=8 main_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_gpu 1 \
#     --learning_rate 1e-5

# 4. Custom self-forcing denoising steps (4-step model):
# torchrun --nproc-per-node=8 main_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --denoising_step_list 1000,750,500,250,0 \
#     --batch_size_per_gpu 1

# 5. With separate teacher model:
# torchrun --nproc-per-node=8 main_dmd_t2v.py \
#     --meta_folder /path/to/meta \
#     --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
#     --teacher_model_path /path/to/teacher/checkpoint \
#     --batch_size_per_gpu 1

# ============================================================================
# KEY DIFFERENCES FROM ORIGINAL IMPLEMENTATION
# ============================================================================
#
# 1. Self-Forcing Pipeline:
#    - OLD: BackwardSimulationPipeline created separately in trainer
#    - NEW: SelfForcingTrainingPipeline created internally by DMD model
#
# 2. Random Timestep Selection:
#    - OLD: All timesteps computed, memory intensive
#    - NEW: ONE random timestep selected per batch (synchronized across ranks)
#
# 3. Gradient Flow:
#    - OLD: Unclear gradient management
#    - NEW: Gradients ONLY at randomly selected timestep
#
# 4. Memory Efficiency:
#    - OLD: Stores full trajectory
#    - NEW: Only stores final prediction (last_step_only=True)
#
# 5. Training Speed:
#    - OLD: ~2 hours on 64 H100 GPUs for unclear number of steps
#    - NEW: ~2 hours on 64 H100 GPUs for 600 steps (matches Self-Forcing paper)
#
# ============================================================================
