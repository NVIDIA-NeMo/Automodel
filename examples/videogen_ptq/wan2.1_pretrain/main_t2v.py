#!/usr/bin/env python3
# main_t2v.py - Entry point for WAN 2.1 T2V Flow Matching Training

import argparse

from dist_utils import print0
from trainer_t2v import WanT2VTrainerFSDP


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V Flow Matching Training")

    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 
                   help="HuggingFace model ID")
    
    # Training mode
    p.add_argument("--pretrain", action="store_true", default=False,
                   help="Enable pretraining mode (initialize from config instead of pretrained weights)")

    # Training configuration
    p.add_argument("--meta_folder", type=str, required=True, 
                   help="Path to folder containing .meta files")
    p.add_argument("--num_epochs", type=int, default=10, 
                   help="Number of training epochs")
    p.add_argument("--batch_size_per_node", type=int, default=1,
                   help="Batch size per NODE (not per GPU)")
    p.add_argument("--learning_rate", type=float, default=1e-5, 
                   help="Learning rate")

    # Memory optimization
    p.add_argument("--cpu_offload", action="store_true", default=True,
                   help="Enable CPU offloading for parameters")
    p.add_argument("--no_cpu_offload", action="store_false", dest="cpu_offload",
                   help="Disable CPU offloading")

    # Flow matching arguments
    p.add_argument("--use_sigma_noise", action="store_true", default=True,
                   help="Use flow matching noise scheduling")
    p.add_argument("--no_sigma_noise", action="store_false", dest="use_sigma_noise",
                   help="Disable flow matching")
    p.add_argument("--timestep_sampling", type=str, default="uniform",
                   choices=["uniform", "logit_normal", "mode"],
                   help="Timestep sampling strategy")
    p.add_argument("--logit_mean", type=float, default=0.0,
                   help="Mean for logit-normal distribution")
    p.add_argument("--logit_std", type=float, default=1.0,
                   help="Std for logit-normal distribution")
    p.add_argument("--flow_shift", type=float, default=3.0,
                   help="Flow matching shift parameter (for simple mode)")
    p.add_argument("--mix_uniform_ratio", type=float, default=0.1,
                   help="Ratio of uniform sampling mixed with density sampling")

    # Advanced pretrain options (Karras scheduling)
    p.add_argument("--use_karras_schedule", action="store_true", default=False,
                   help="Use Karras-style sigma scheduling (recommended for pretraining)")
    p.add_argument("--sigma_min", type=float, default=1e-3,
                   help="Minimum sigma for Karras schedule")
    p.add_argument("--sigma_max", type=float, default=1.0,
                   help="Maximum sigma for Karras schedule")
    p.add_argument("--rho", type=float, default=7.0,
                   help="Rho parameter for Karras schedule")
    
    # Loss weighting
    p.add_argument("--loss_weight_k_start", type=float, default=None,
                   help="Starting k value for loss weighting (default: 6.0 for pretrain, flow_shift for finetune)")
    p.add_argument("--loss_weight_k_end", type=float, default=2.0,
                   help="Ending k value for loss weighting (annealed)")
    p.add_argument("--anneal_loss_weight", action="store_true", default=False,
                   help="Anneal loss weight from k_start to k_end over training")
    
    # Warmup
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Number of warmup steps to clamp minimum sigma (0 = no warmup)")
    p.add_argument("--warmup_sigma_min_clamp", type=float, default=0.05,
                   help="Minimum sigma clamp during warmup")

    # Checkpointing
    p.add_argument("--save_every", type=int, default=500, 
                   help="Save checkpoint every N steps")
    p.add_argument("--log_every", type=int, default=5, 
                   help="Log metrics every N steps")
    p.add_argument("--output_dir", type=str, default="./wan_t2v_flow_outputs", 
                   help="Output directory for checkpoints")
    p.add_argument("--resume_checkpoint", type=str, default=None, 
                   help="Path to checkpoint to resume from")

    return p.parse_args()


def main():
    args = parse_args()

    # ========================================================================
    # SMART DEFAULTS: Auto-configure based on pretrain vs finetune mode
    # ========================================================================
    if args.pretrain:
        # PRETRAINING: Use recommended settings automatically
        mode = "PRETRAINING"
        
        # Override defaults for pretraining if not explicitly set
        if args.use_karras_schedule is False:
            args.use_karras_schedule = True
            print0("[AUTO] Enabled Karras schedule for pretraining")
        
        if args.anneal_loss_weight is False:
            args.anneal_loss_weight = True
            print0("[AUTO] Enabled loss weight annealing for pretraining")
        
        if args.loss_weight_k_start is None:
            args.loss_weight_k_start = 6.0
            print0(f"[AUTO] Set loss_weight_k_start={args.loss_weight_k_start} for pretraining")
        
        if args.warmup_steps == 0:
            args.warmup_steps = 10000
            print0(f"[AUTO] Set warmup_steps={args.warmup_steps} for pretraining")
        
        if args.timestep_sampling == "uniform":
            args.timestep_sampling = "logit_normal"
            print0("[AUTO] Using logit_normal sampling for pretraining")
        
        if args.mix_uniform_ratio == 0.1:
            args.mix_uniform_ratio = 0.2
            print0(f"[AUTO] Set mix_uniform_ratio={args.mix_uniform_ratio} for pretraining")
        
        if args.learning_rate == 1e-5:
            # Suggest higher learning rate for pretraining
            print0("[SUGGESTION] Consider using --learning_rate 1e-4 for pretraining")
    else:
        # FINETUNING: Use conservative settings
        mode = "FINETUNING"
        
        # Set loss_weight_k to flow_shift for backward compatibility
        if args.loss_weight_k_start is None:
            args.loss_weight_k_start = args.flow_shift
            print0(f"[AUTO] Set loss_weight_k={args.flow_shift} (matches flow_shift for finetuning)")
        
        # Warn if advanced pretrain features are enabled during finetuning
        if args.use_karras_schedule:
            print0("[WARNING] Karras schedule is enabled for finetuning (usually unnecessary)")
        if args.anneal_loss_weight:
            print0("[WARNING] Loss weight annealing enabled for finetuning (usually unnecessary)")
        if args.warmup_steps > 0:
            print0("[WARNING] Warmup enabled for finetuning (usually unnecessary)")

    print0(f"[INFO] Starting WAN 2.1 T2V Flow Matching {mode}")
    print0(f"[INFO] Model: {args.model_id}")
    print0(f"[INFO] Batch size per node: {args.batch_size_per_node}")
    print0(f"[INFO] Learning rate: {args.learning_rate}")
    print0(f"[INFO] CPU offload: {'ENABLED' if args.cpu_offload else 'DISABLED'}")
    print0(f"[INFO] Flow matching: {'ENABLED' if args.use_sigma_noise else 'DISABLED'}")
    
    if args.use_karras_schedule:
        print0(f"[INFO] Using Karras sigma schedule (sigma_min={args.sigma_min}, sigma_max={args.sigma_max}, rho={args.rho})")
    
    if args.anneal_loss_weight:
        print0(f"[INFO] Annealing loss weight: k={args.loss_weight_k_start} -> {args.loss_weight_k_end}")
    
    if args.warmup_steps > 0:
        print0(f"[INFO] Warmup: {args.warmup_steps} steps with sigma >= {args.warmup_sigma_min_clamp}")

    # Create trainer
    trainer = WanT2VTrainerFSDP(
        model_id=args.model_id,
        learning_rate=args.learning_rate,
        cpu_offload=args.cpu_offload,
        pretrain=args.pretrain,
        # Flow matching config
        use_sigma_noise=args.use_sigma_noise,
        timestep_sampling=args.timestep_sampling,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        flow_shift=args.flow_shift,
        mix_uniform_ratio=args.mix_uniform_ratio,
        # Advanced pretrain options
        use_karras_schedule=args.use_karras_schedule,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        loss_weight_k_start=args.loss_weight_k_start,
        loss_weight_k_end=args.loss_weight_k_end,
        anneal_loss_weight=args.anneal_loss_weight,
        warmup_steps=args.warmup_steps,
        warmup_sigma_min_clamp=args.warmup_sigma_min_clamp,
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

# 1. Basic FINETUNING (default - simple settings):
# torchrun --nproc-per-node=8 main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --learning_rate 5e-6

# 2. PRETRAINING (AUTO: Karras + annealing + warmup enabled automatically):
# torchrun --nproc-per-node=8 main_t2v.py \
#     --pretrain \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --learning_rate 1e-4

# 3. PRETRAINING with manual control (override auto settings):
# torchrun --nproc-per-node=8 main_t2v.py \
#     --pretrain \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --learning_rate 1e-4 \
#     --rho 5.0 \
#     --loss_weight_k_start 8.0 \
#     --warmup_steps 20000

# 4. FINETUNING with advanced features (if you really want):
# torchrun --nproc-per-node=8 main_t2v.py \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --learning_rate 5e-6 \
#     --use_karras_schedule \
#     --anneal_loss_weight

# 5. Multi-node PRETRAINING (10 nodes × 8 GPUs):
# torchrun \
#     --nnodes=10 \
#     --nproc-per-node=8 \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=<master_addr>:<master_port> \
#     --rdzv-id=<job_id> \
#     main_t2v.py \
#     --pretrain \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --learning_rate 1e-4 \
#     --use_karras_schedule \
#     --anneal_loss_weight \
#     --output_dir ./wan_t2v_pretrain_outputs

# 6. Resume PRETRAINING from checkpoint:
# torchrun --nproc-per-node=8 main_t2v.py \
#     --pretrain \
#     --meta_folder /path/to/meta \
#     --batch_size_per_node 1 \
#     --use_karras_schedule \
#     --anneal_loss_weight \
#     --resume_checkpoint ./wan_t2v_pretrain_outputs/checkpoint-5000