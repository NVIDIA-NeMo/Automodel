# trainer_dmd_t2v.py - WAN 2.1 T2V DMD Trainer with Alternating Optimization

import os
from typing import Optional

import torch
import torch.distributed as dist
import wandb
from dataloader import create_dataloader as create_base_dataloader
from diffusers import WanPipeline
from dist_utils import is_main_process, print0, setup_distributed
from dmd_t2v import DMDT2V
from fsdp_dmd_utils_t2v import (
    get_fsdp_dmd_trainable_parameters,
    load_fsdp_dmd_checkpoint,
    save_fsdp_dmd_checkpoint,
    setup_fsdp_for_dmd_t2v,
    test_fsdp_dmd_forward_pass,
    verify_fsdp_dmd_setup,
)
from pipeline_t2v import BackwardSimulationPipeline
from training_step_dmd_t2v import step_dmd_alternating


class WanDMDTrainerT2V:
    """
    WAN 2.1 T2V DMD Trainer with full parameter fine-tuning.

    This trainer implements Distribution Matching Distillation (DMD) with:
    - 3 full models: generator (student), teacher (real_score), critic (fake_score)
    - Backward simulation for data-free distillation
    - Alternating optimization of generator and critic
    - Memory-efficient FSDP with CPU offloading
    """

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        teacher_model_path: Optional[str] = None,
        learning_rate: float = 1e-5,
        critic_learning_rate: Optional[float] = None,
        cpu_offload: bool = True,
        # DMD parameters
        num_train_timestep: int = 1000,
        min_step: int = 20,
        max_step: int = 980,
        real_guidance_scale: float = 5.0,
        fake_guidance_scale: float = 0.0,
        timestep_shift: float = 3.0,
        ts_schedule: bool = True,
        ts_schedule_max: bool = False,
        min_score_timestep: int = 0,
        denoising_loss_type: str = "flow",
        # Backward simulation
        denoising_step_list: list = None,
        # Alternating optimization
        generator_steps: int = 1,
        critic_steps: int = 1,
    ):
        """
        Initialize DMD trainer.

        Args:
            model_id: HuggingFace model ID for base model
            teacher_model_path: Optional path to teacher checkpoint
            learning_rate: Learning rate for generator
            critic_learning_rate: Learning rate for critic (defaults to learning_rate)
            cpu_offload: Enable CPU offloading for parameters
            num_train_timestep: Total training timesteps (1000)
            min_step: Minimum timestep for noise (20)
            max_step: Maximum timestep for noise (980)
            real_guidance_scale: CFG scale for teacher (5.0)
            fake_guidance_scale: CFG scale for critic (0.0)
            timestep_shift: Flow matching shift parameter (3.0)
            ts_schedule: Use dynamic timestep scheduling
            ts_schedule_max: Use max timestep scheduling
            min_score_timestep: Minimum timestep for critic
            denoising_loss_type: "flow" or "epsilon" or "x0"
            denoising_step_list: List of discrete timesteps for backward simulation
            generator_steps: Number of generator updates per iteration
            critic_steps: Number of critic updates per iteration
        """
        self.model_id = model_id
        self.teacher_model_path = teacher_model_path
        self.learning_rate = learning_rate
        self.critic_learning_rate = critic_learning_rate or learning_rate
        self.cpu_offload = cpu_offload
        self.bf16 = torch.bfloat16

        # DMD config
        self.num_train_timestep = num_train_timestep
        self.min_step = min_step
        self.max_step = max_step
        self.real_guidance_scale = real_guidance_scale
        self.fake_guidance_scale = fake_guidance_scale
        self.timestep_shift = timestep_shift
        self.ts_schedule = ts_schedule
        self.ts_schedule_max = ts_schedule_max
        self.min_score_timestep = min_score_timestep
        self.denoising_loss_type = denoising_loss_type

        # Backward simulation config
        if denoising_step_list is None:
            self.denoising_step_list = [1000, 750, 500, 250, 0]
        else:
            self.denoising_step_list = denoising_step_list

        # Alternating optimization
        self.generator_steps = generator_steps
        self.critic_steps = critic_steps

        # Setup distributed training
        self.local_rank = setup_distributed()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device("cuda", self.local_rank)

        print0("[INFO] WAN 2.1 T2V DMD Trainer")
        print0(f"[INFO] Total GPUs: {self.world_size}, Local rank: {self.local_rank}")
        print0(f"[INFO] Generator LR: {self.learning_rate}")
        print0(f"[INFO] Critic LR: {self.critic_learning_rate}")
        print0(f"[INFO] CPU offload: {'ENABLED' if cpu_offload else 'DISABLED'}")
        print0("[INFO] DMD config:")
        print0(f"  - Real guidance scale: {real_guidance_scale}")
        print0(f"  - Timestep shift: {timestep_shift}")
        print0(f"  - Denoising steps: {self.denoising_step_list}")
        print0(f"  - Alternating: {generator_steps} gen / {critic_steps} critic")

        # Initialize components
        self.pipe = None
        self.model_map = {}
        self.dmd_model = None
        self.pipeline = None
        self.generator_optimizer = None
        self.critic_optimizer = None
        self.lr_scheduler = None

    def setup_pipeline(self):
        """Load WAN 2.1 T2V pipeline (transformer only)."""
        print0("[INFO] Loading WAN 2.1 T2V pipeline...")

        # Load pipeline without VAE or text encoder
        self.pipe = WanPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            vae=None,
            text_encoder=None,
        )

        # Remove VAE and text encoder if loaded
        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            del self.pipe.vae
            self.pipe.vae = None

        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            del self.pipe.text_encoder
            self.pipe.text_encoder = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print0("[INFO] Pipeline loaded (transformer + scheduler only)")

    def setup_fsdp(self):
        """Setup FSDP for 3 models (generator, teacher, critic)."""
        print0("[INFO] Setting up FSDP for DMD...")

        self.model_map = setup_fsdp_for_dmd_t2v(
            self.pipe,
            device=self.device,
            bf16=self.bf16,
            local_rank=self.local_rank,
            teacher_model_path=self.teacher_model_path,
            cpu_offload=self.cpu_offload,
        )

        # Verify setup
        verify_fsdp_dmd_setup(self.model_map)

        print0("[INFO] FSDP setup complete")

    def setup_dmd_model(self):
        """Initialize DMD model."""
        print0("[INFO] Setting up DMD model...")

        self.dmd_model = DMDT2V(
            model_map=self.model_map,
            scheduler=self.pipe.scheduler,
            device=self.device,
            bf16=self.bf16,
            num_train_timestep=self.num_train_timestep,
            min_step=self.min_step,
            max_step=self.max_step,
            real_guidance_scale=self.real_guidance_scale,
            fake_guidance_scale=self.fake_guidance_scale,
            timestep_shift=self.timestep_shift,
            ts_schedule=self.ts_schedule,
            ts_schedule_max=self.ts_schedule_max,
            min_score_timestep=self.min_score_timestep,
            denoising_loss_type=self.denoising_loss_type,
            denoising_step_list=self.denoising_step_list,  # CRITICAL: Pass discrete timesteps!
        )

        print0("[INFO] DMD model initialized")

    def setup_backward_simulation(self):
        """Initialize backward simulation pipeline."""
        print0("[INFO] Setting up backward simulation pipeline...")

        self.pipeline = BackwardSimulationPipeline(
            scheduler=self.pipe.scheduler,
            generator_model=self.model_map["generator"]["fsdp_transformer"],
            denoising_step_list=self.denoising_step_list,
            device=self.device,
            bf16=self.bf16,
            last_step_only=True,
        )

        print0("[INFO] Backward simulation pipeline initialized")

    def setup_optim(self):
        """Setup optimizers for generator and critic."""
        print0("[INFO] Setting up optimizers...")

        # Get trainable parameters
        generator_params, critic_params = get_fsdp_dmd_trainable_parameters(self.model_map, optimize_both=True)

        # Generator optimizer
        self.generator_optimizer = torch.optim.AdamW(
            generator_params, lr=self.learning_rate, weight_decay=0.01, betas=(0.9, 0.999)
        )

        # Critic optimizer
        self.critic_optimizer = torch.optim.AdamW(
            critic_params, lr=self.critic_learning_rate, weight_decay=0.01, betas=(0.9, 0.999)
        )

        # Temporary scheduler (will be reconfigured after dataloader)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.generator_optimizer, T_max=1000, eta_min=1e-6
        )

        print0(f"[INFO] Generator optimizer: {len(generator_params)} param tensors")
        print0(f"[INFO] Critic optimizer: {len(critic_params)} param tensors")

    def validate_setup(self):
        """Validate FSDP setup with test forward passes."""
        print0("[INFO] Validating FSDP setup...")

        test_fsdp_dmd_forward_pass(self.model_map, self.device, self.bf16)

        # Check memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print0(f"[INFO] GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

    def train(
        self,
        meta_folder: str,
        num_epochs: int = 10,
        batch_size_per_gpu: int = 1,
        save_every: int = 500,
        log_every: int = 10,
        output_dir: str = "./wan_dmd_t2v_outputs",
        resume_checkpoint: Optional[str] = None,
    ):
        """
        Train the DMD model with alternating optimization.

        Args:
            meta_folder: Path to preprocessed .meta files
            num_epochs: Number of training epochs
            batch_size_per_gpu: Batch size per GPU
            save_every: Save checkpoint every N steps
            log_every: Log metrics every N steps
            output_dir: Output directory for checkpoints
            resume_checkpoint: Optional checkpoint to resume from
        """
        print0("[INFO] Starting DMD training")
        print0(f"[INFO] Batch size per GPU: {batch_size_per_gpu}")
        print0(f"[INFO] Total effective batch size: {batch_size_per_gpu * self.world_size}")

        # Setup all components
        self.setup_pipeline()
        self.setup_fsdp()
        self.setup_dmd_model()
        self.setup_backward_simulation()
        self.setup_optim()
        self.validate_setup()

        # Create dataloader
        dataloader = create_base_dataloader(
            meta_folder=meta_folder,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=2,
            device="cpu",  # Load to CPU first, then move to GPU
        )

        steps_per_epoch = len(dataloader)
        total_steps = num_epochs * steps_per_epoch

        # Reconfigure scheduler with actual total steps
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.generator_optimizer, T_max=total_steps, eta_min=1e-6
        )
        print0(f"[INFO] Scheduler configured for {total_steps} total steps")

        global_step = 0
        start_epoch = 0

        # Resume from checkpoint
        if resume_checkpoint:
            global_step = load_fsdp_dmd_checkpoint(
                self.model_map, self.generator_optimizer, self.critic_optimizer, self.lr_scheduler, resume_checkpoint
            )
            start_epoch = global_step // steps_per_epoch

        # Create output directory
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

            config = {
                "model_id": self.model_id,
                "generator_lr": self.learning_rate,
                "critic_lr": self.critic_learning_rate,
                "num_epochs": num_epochs,
                "batch_size_per_gpu": batch_size_per_gpu,
                "total_batch_size": batch_size_per_gpu * self.world_size,
                "total_gpus": self.world_size,
                "approach": "wan_dmd_t2v_full",
                "cpu_offload": self.cpu_offload,
                # DMD config
                "real_guidance_scale": self.real_guidance_scale,
                "fake_guidance_scale": self.fake_guidance_scale,
                "timestep_shift": self.timestep_shift,
                "denoising_step_list": self.denoising_step_list,
                "denoising_loss_type": self.denoising_loss_type,
                "generator_steps": self.generator_steps,
                "critic_steps": self.critic_steps,
            }

            wandb.init(
                project="wan-dmd-t2v",
                config=config,
                resume=resume_checkpoint is not None,
            )

        if dist.is_initialized():
            dist.barrier()

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            iterable = dataloader
            if is_main_process():
                from tqdm import tqdm

                iterable = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            epoch_gen_loss = 0.0
            epoch_critic_loss = 0.0
            num_steps = 0

            for step, batch in enumerate(iterable):
                # Alternating optimization logic
                # Update generator for generator_steps
                for _ in range(self.generator_steps):
                    self.generator_optimizer.zero_grad(set_to_none=True)

                    try:
                        gen_loss, _, metrics = step_dmd_alternating(
                            dmd_model=self.dmd_model,
                            pipeline=self.pipeline,
                            batch=batch,
                            device=self.device,
                            bf16=self.bf16,
                            global_step=global_step,
                            update_generator=True,
                            update_critic=False,
                        )

                        if gen_loss is not None:
                            gen_loss.backward()

                            # Gradient clipping
                            gen_params = [
                                p
                                for p in self.model_map["generator"]["fsdp_transformer"].parameters()
                                if p.requires_grad and p.grad is not None
                            ]
                            gen_grad_norm = torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)

                            self.generator_optimizer.step()
                            self.lr_scheduler.step()

                            epoch_gen_loss += gen_loss.item()

                    except Exception as e:
                        print0(f"[ERROR] Generator step failed at epoch {epoch}, step {step}: {e}")
                        raise

                # Update critic for critic_steps
                for _ in range(self.critic_steps):
                    self.critic_optimizer.zero_grad(set_to_none=True)

                    try:
                        _, critic_loss, crit_metrics = step_dmd_alternating(
                            dmd_model=self.dmd_model,
                            pipeline=self.pipeline,
                            batch=batch,
                            device=self.device,
                            bf16=self.bf16,
                            global_step=global_step,
                            update_generator=False,
                            update_critic=True,
                        )

                        if critic_loss is not None:
                            critic_loss.backward()

                            # Gradient clipping
                            critic_params = [
                                p
                                for p in self.model_map["fake_score"]["fsdp_transformer"].parameters()
                                if p.requires_grad and p.grad is not None
                            ]
                            critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, max_norm=1.0)

                            self.critic_optimizer.step()

                            epoch_critic_loss += critic_loss.item()

                    except Exception as e:
                        print0(f"[ERROR] Critic step failed at epoch {epoch}, step {step}: {e}")
                        raise

                num_steps += 1
                global_step += 1

                # Logging
                if is_main_process() and (global_step % log_every == 0):
                    log_dict = {
                        "generator_loss": gen_loss.item() if gen_loss is not None else 0.0,
                        "critic_loss": critic_loss.item() if critic_loss is not None else 0.0,
                        "avg_gen_loss": epoch_gen_loss / (num_steps * self.generator_steps),
                        "avg_critic_loss": epoch_critic_loss / (num_steps * self.critic_steps),
                        "generator_lr": self.generator_optimizer.param_groups[0]["lr"],
                        "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                        "gen_grad_norm": float(gen_grad_norm) if "gen_grad_norm" in locals() else 0.0,
                        "critic_grad_norm": float(critic_grad_norm) if "critic_grad_norm" in locals() else 0.0,
                        "epoch": epoch,
                        "global_step": global_step,
                    }

                    wandb.log(log_dict, step=global_step)

                    if hasattr(iterable, "set_postfix"):
                        iterable.set_postfix(
                            {
                                "gen_loss": f"{gen_loss.item() if gen_loss else 0:.4f}",
                                "crit_loss": f"{critic_loss.item() if critic_loss else 0:.4f}",
                                "gen_lr": f"{log_dict['generator_lr']:.2e}",
                            }
                        )

                # Checkpointing
                if save_every and (global_step % save_every == 0):
                    save_fsdp_dmd_checkpoint(
                        self.model_map,
                        self.generator_optimizer,
                        self.critic_optimizer,
                        self.lr_scheduler,
                        output_dir,
                        global_step,
                        consolidate=True,  # Always save consolidated model
                    )

            # Epoch summary
            avg_gen_loss = epoch_gen_loss / max(num_steps * self.generator_steps, 1)
            avg_critic_loss = epoch_critic_loss / max(num_steps * self.critic_steps, 1)
            print0(f"[INFO] Epoch {epoch + 1} complete")
            print0(f"  Generator loss: {avg_gen_loss:.6f}")
            print0(f"  Critic loss: {avg_critic_loss:.6f}")

            if is_main_process():
                wandb.log(
                    {
                        "epoch/avg_gen_loss": avg_gen_loss,
                        "epoch/avg_critic_loss": avg_critic_loss,
                        "epoch/num": epoch + 1,
                    },
                    step=global_step,
                )

        # Final checkpoint
        if is_main_process():
            print0("[INFO] Training complete, saving final checkpoint...")

            save_fsdp_dmd_checkpoint(
                self.model_map,
                self.generator_optimizer,
                self.critic_optimizer,
                self.lr_scheduler,
                output_dir,
                global_step,
                consolidate=True,  # Always consolidate final checkpoint
            )

            print0(f"[INFO] Saved final checkpoint at step {global_step}")

            wandb.finish()

        print0("[INFO] DMD training complete!")
