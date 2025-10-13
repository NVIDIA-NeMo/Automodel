# trainer_t2v.py - WAN 2.1 T2V Trainer with Flow Matching

import os
from typing import Optional

import torch
import torch.distributed as dist
import wandb
from data_utils import create_dataloader
from diffusers import WanPipeline
from dist_utils import is_main_process, print0, setup_distributed
from fsdp2_utils_t2v import (
    get_fsdp_all_parameters,
    load_fsdp_checkpoint,
    save_fsdp_checkpoint,
    setup_fsdp_for_t2v_pipe,
    test_fsdp_forward_pass,
    verify_fsdp_setup,
)
from training_step_t2v import step_fsdp_transformer_t2v


class WanT2VTrainerFSDP:
    """
    WAN 2.1 T2V FULL fine-tuning trainer with flow matching.
    
    Features:
    - FSDP for intra-node parallelism
    - Data parallelism across nodes
    - Flow matching training (independent of inference scheduler)
    - Density-based timestep sampling
    - Loss weighting
    """

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        learning_rate: float = 1e-5,
        cpu_offload: bool = True,
        # Flow matching training parameters
        use_sigma_noise: bool = True,
        timestep_sampling: str = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        flow_shift: float = 3.0,
        mix_uniform_ratio: float = 0.1,
    ):
        self.model_id = model_id
        self.learning_rate = learning_rate
        self.cpu_offload = cpu_offload
        self.bf16 = torch.bfloat16

        # Flow matching config
        self.use_sigma_noise = use_sigma_noise
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.flow_shift = flow_shift
        self.mix_uniform_ratio = mix_uniform_ratio

        self.local_rank = setup_distributed()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device("cuda", self.local_rank)
        
        # Calculate node information
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
        self.num_nodes = self.world_size // self.local_world_size if self.local_world_size > 0 else 1
        self.node_rank = dist.get_rank() // self.local_world_size if dist.is_initialized() else 0

        print0("[INFO] WAN 2.1 T2V Trainer with Flow Matching")
        print0(f"[INFO] Total GPUs: {self.world_size}, GPUs per node: {self.local_world_size}, Num nodes: {self.num_nodes}")
        print0(f"[INFO] Node rank: {self.node_rank}, Local rank: {self.local_rank}")
        print0(f"[INFO] Learning rate: {learning_rate}")
        print0(f"[INFO] CPU offload: {'ENABLED' if cpu_offload else 'DISABLED'}")
        print0(f"[INFO] Flow matching: {'ENABLED' if use_sigma_noise else 'DISABLED'}")
        if use_sigma_noise:
            print0(f"[INFO]   - Timestep sampling: {timestep_sampling}")
            print0(f"[INFO]   - Flow shift: {flow_shift}")
            print0(f"[INFO]   - Mix uniform ratio: {mix_uniform_ratio}")

        self.pipe = None
        self.model_map = {}
        self.optimizer = None
        self.lr_scheduler = None

    def setup_pipeline(self):
        print0("[INFO] Loading WAN 2.1 T2V pipeline (transformer only)...")

        # Load pipeline without VAE or text encoder
        self.pipe = WanPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float32,
            vae=None,
            text_encoder=None,
        )

        # Explicitly delete VAE and text encoder if they were loaded
        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            print0("[INFO] Removing VAE from pipeline...")
            del self.pipe.vae
            self.pipe.vae = None

        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            print0("[INFO] Removing text encoder from pipeline...")
            del self.pipe.text_encoder
            self.pipe.text_encoder = None

        # Clear any cached memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print0("[INFO] Pipeline loaded (transformer + scheduler only)")

    def setup_fsdp(self):
        print0("[INFO] Setting up FSDP for full fine-tuning...")

        self.model_map = setup_fsdp_for_t2v_pipe(
            self.pipe,
            device=self.device,
            bf16=self.bf16,
            local_rank=self.local_rank,
            cpu_offload=self.cpu_offload,
        )

        # Verify setup
        verify_fsdp_setup(self.model_map)

        print0("[INFO] FSDP setup complete")

    def setup_optim(self):
        print0("[INFO] Setting up optimizer...")

        # Get ALL trainable parameters
        all_params = get_fsdp_all_parameters(self.model_map)

        if not all_params:
            raise RuntimeError("No trainable parameters found!")

        print0(f"[INFO] Optimizing {len(all_params)} parameters")

        self.optimizer = torch.optim.AdamW(
            all_params, 
            lr=self.learning_rate, 
            weight_decay=0.01, 
            betas=(0.9, 0.999)
        )

        # Temporary scheduler - will be reconfigured after dataloader is created
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000, 
            eta_min=1e-6
        )

    def validate_setup(self):
        """Validate FSDP setup with a test forward pass."""
        print0("[INFO] Validating FSDP setup...")

        test_fsdp_forward_pass(self.model_map, self.device, self.bf16)

        # Check memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print0(f"[INFO] GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

    def train(
        self,
        meta_folder: str,
        num_epochs: int = 10,
        batch_size_per_node: int = 1,
        save_every: int = 1000,
        log_every: int = 100,
        output_dir: str = "./wan_t2v_outputs",
        resume_checkpoint: Optional[str] = None,
    ):
        """
        Train the model with flow matching.
        
        Args:
            batch_size_per_node: Batch size for each node (NOT per GPU)
        """
        print0("[INFO] Starting T2V training with Flow Matching")
        print0(f"[INFO] Batch size per node: {batch_size_per_node}")
        print0(f"[INFO] Total effective batch size: {batch_size_per_node * self.num_nodes}")

        self.setup_pipeline()
        self.setup_fsdp()
        self.setup_optim()
        self.validate_setup()

        # Create dataloader
        dataloader, sampler = create_dataloader(meta_folder, batch_size_per_node, self.num_nodes)

        steps_per_epoch = len(dataloader)
        total_steps = num_epochs * steps_per_epoch

        # Reconfigure scheduler with actual total steps
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps, 
            eta_min=1e-6
        )
        print0(f"[INFO] Scheduler configured for {total_steps} total steps")

        global_step = 0
        start_epoch = 0

        if resume_checkpoint:
            global_step = load_fsdp_checkpoint(
                self.model_map, self.optimizer, self.lr_scheduler, resume_checkpoint
            )
            start_epoch = global_step // steps_per_epoch

        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

            config = {
                "model_id": self.model_id,
                "learning_rate": self.learning_rate,
                "num_epochs": num_epochs,
                "batch_size_per_node": batch_size_per_node,
                "total_batch_size": batch_size_per_node * self.num_nodes,
                "num_nodes": self.num_nodes,
                "gpus_per_node": self.local_world_size,
                "total_gpus": self.world_size,
                "approach": "wan_t2v_flow_matching",
                "cpu_offload": self.cpu_offload,
                # Flow matching config
                "use_sigma_noise": self.use_sigma_noise,
                "timestep_sampling": self.timestep_sampling,
                "logit_mean": self.logit_mean,
                "logit_std": self.logit_std,
                "flow_shift": self.flow_shift,
                "mix_uniform_ratio": self.mix_uniform_ratio,
            }

            wandb.init(
                project="wan-t2v-flow-matching",
                config=config,
                resume=resume_checkpoint is not None,
            )

        if dist.is_initialized():
            dist.barrier()

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            sampler.set_epoch(epoch)

            iterable = dataloader
            if is_main_process():
                from tqdm import tqdm
                iterable = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            epoch_loss = 0.0
            num_steps = 0

            for step, batch in enumerate(iterable):
                self.optimizer.zero_grad(set_to_none=True)

                try:
                    loss, metrics = step_fsdp_transformer_t2v(
                        pipe=self.pipe,
                        model_map=self.model_map,
                        batch=batch,
                        device=self.device,
                        bf16=self.bf16,
                        # Flow matching parameters
                        use_sigma_noise=self.use_sigma_noise,
                        timestep_sampling=self.timestep_sampling,
                        logit_mean=self.logit_mean,
                        logit_std=self.logit_std,
                        flow_shift=self.flow_shift,
                        mix_uniform_ratio=self.mix_uniform_ratio,
                        global_step=global_step,
                    )

                except Exception as e:
                    print0(f"[ERROR] Training step failed at epoch {epoch}, step {step}: {e}")
                    video_shape = batch.get("video_latents", torch.tensor([])).shape
                    text_shape = batch.get("text_embeddings", torch.tensor([])).shape
                    print0(f"[DEBUG] Batch shapes - video: {video_shape}, text: {text_shape}")
                    raise

                loss.backward()

                # Gradient clipping
                trainable_params = [
                    p for p in self.model_map["transformer"]["fsdp_transformer"].parameters() 
                    if p.requires_grad and p.grad is not None
                ]

                grad_norm = 0.0
                if trainable_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

                self.optimizer.step()
                self.lr_scheduler.step()

                epoch_loss += loss.item()
                num_steps += 1
                global_step += 1

                # Logging
                if is_main_process() and (global_step % log_every == 0):
                    log_dict = {
                        "train_loss": loss.item(),
                        "train_avg_loss": epoch_loss / num_steps,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm,
                        "epoch": epoch,
                        "global_step": global_step,
                    }

                    wandb.log(log_dict, step=global_step)

                    if hasattr(iterable, "set_postfix"):
                        iterable.set_postfix(
                            {
                                "loss": f"{loss.item():.4f}",
                                "avg": f"{(epoch_loss / num_steps):.4f}",
                                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                                "gn": f"{grad_norm:.2f}",
                            }
                        )

                # Checkpointing
                if save_every and (global_step % save_every == 0):
                    # FIXED: Better consolidation strategy
                    # - Consolidate every 1000 steps (or use save_every if it's >= 1000)
                    # - This ensures regular inference-ready checkpoints
                    save_fsdp_checkpoint(
                        self.model_map,
                        self.optimizer,
                        self.lr_scheduler,
                        output_dir,
                        global_step,
                        consolidate=True,  # Always save consolidated model
                    )                    

            # Epoch summary
            avg_loss = epoch_loss / max(num_steps, 1)
            print0(f"[INFO] Epoch {epoch + 1} complete. avg_loss={avg_loss:.6f}")

            if is_main_process():
                wandb.log({"epoch/avg_loss": avg_loss, "epoch/num": epoch + 1}, step=global_step)

        # Final checkpoint
        if is_main_process():
            print0("[INFO] Training complete, saving final checkpoint...")

            save_fsdp_checkpoint(
                self.model_map,
                self.optimizer,
                self.lr_scheduler,
                output_dir,
                global_step,
                consolidate=True,  # Always consolidate final checkpoint
            )

            print0(f"[INFO] Saved final checkpoint at step {global_step}")

            wandb.finish()

        print0("[INFO] Training complete!")