# trainer_fsdp2.py - WAN I2V LoRA Trainer with FSDP2
import os
import torch
import wandb
import torch.distributed as dist
from typing import Optional, Dict
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

from dist_utils import setup_distributed, is_main_process, print0, cast_model_to_dtype
from fsdp2_utils import (
    setup_fsdp2_for_pipe,
    verify_fsdp2_setup,
    test_fsdp2_forward_pass,
    get_fsdp2_lora_parameters,
    save_fsdp2_checkpoint,
    load_fsdp2_checkpoint,
)
from data_utils import create_dataloader
from training_step_fsdp2 import step_fsdp2_transformer
from checkpoint_io import save_manual_lora_checkpoint, load_manual_lora_checkpoint


class WanI2VLoRATrainerFSDP2:
    """
    WAN I2V LoRA trainer using FSDP2 for efficient memory usage.
    
    Key features:
    - FSDP2 shards frozen base parameters across GPUs
    - LoRA parameters remain unsharded for efficient training
    - Avoids dimension mismatch issues with conditional embeddings
    - Simpler than TP+CP but still memory efficient
    """
    
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 1e-4,
        boundary_ratio: Optional[float] = 0.5,
        train_transformer_2: bool = False,
        cpu_offload: bool = True,  # Enable CPU offload by default for max memory savings
    ):
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.boundary_ratio = boundary_ratio
        self.train_transformer_2 = train_transformer_2
        self.cpu_offload = cpu_offload
        self.bf16 = torch.bfloat16

        self.local_rank = setup_distributed()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device("cuda", self.local_rank)

        print0(f"[INFO] WAN I2V LoRA trainer with FSDP2")
        print0(f"[INFO] World size: {self.world_size}, Local rank: {self.local_rank}")
        print0(f"[INFO] LoRA rank={lora_rank} alpha={lora_alpha}")
        print0(f"[INFO] Training transformer: {'transformer_2' if train_transformer_2 else 'transformer'}")
        print0(f"[INFO] CPU offload: {'ENABLED' if cpu_offload else 'DISABLED'}")

        self.pipe = None
        self.model_map: Dict[str, Dict] = {}
        self.transformer_names = []
        self.optimizer = None
        self.lr_scheduler = None

    def setup_pipeline(self):
        print0("[INFO] Loading WAN pipeline...")
        
        vae = AutoencoderKLWan.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=torch.float32
        )
        
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id, vae=vae, torch_dtype=torch.float32,
            boundary_ratio=self.boundary_ratio
        )
        
        # Setup non-transformer components (not sharded)
        cast_model_to_dtype(self.pipe.vae, self.bf16)
        self.pipe.vae.to(self.device, dtype=self.bf16).requires_grad_(False)

        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            cast_model_to_dtype(self.pipe.text_encoder, self.bf16)
            self.pipe.text_encoder.to(self.device, dtype=self.bf16).requires_grad_(False)

        if hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None:
            cast_model_to_dtype(self.pipe.image_encoder, self.bf16)
            self.pipe.image_encoder.to(self.device, dtype=self.bf16).requires_grad_(False)

        print0("[INFO] Pipeline components loaded")

    def setup_fsdp2(self):
        print0("[INFO] Setting up FSDP2...")
        
        self.model_map, self.transformer_names = setup_fsdp2_for_pipe(
            self.pipe,
            device=self.device,
            bf16=self.bf16,
            local_rank=self.local_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            train_transformer_2=self.train_transformer_2,
            cpu_offload=self.cpu_offload,
        )
        
        # Verify setup
        verify_fsdp2_setup(self.model_map, self.transformer_names)
        
        # Note: Gradient checkpointing is handled by FSDP2's activation checkpointing
        # Model's built-in gradient_checkpointing is disabled in setup_fsdp2_for_pipe
        
        print0("[INFO] FSDP2 setup complete")

    def setup_optim(self):
        print0("[INFO] Setting up optimizer...")
        
        # Get LoRA parameters from FSDP2 wrapped models
        all_params = get_fsdp2_lora_parameters(self.model_map, self.transformer_names)
        
        if not all_params:
            raise RuntimeError("No LoRA parameters found!")
        
        print0(f"[INFO] Optimizing {len(all_params)} LoRA parameters")
        
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Temporary scheduler - will be reconfigured after dataloader is created
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )

    def validate_setup(self):
        """Validate FSDP2 setup with a test forward pass."""
        print0("[INFO] Validating FSDP2 setup...")
        
        test_fsdp2_forward_pass(
            self.model_map, 
            self.transformer_names, 
            self.device, 
            self.bf16
        )
        
        # Check memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print0(f"[INFO] GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

    def train(
        self,
        meta_folder: str,
        num_epochs: int = 10,
        batch_size: int = 1,
        save_every: int = 1000,
        log_every: int = 100,
        output_dir: str = "./wan_i2v_fsdp2_outputs",
        resume_checkpoint: Optional[str] = None,
    ):
        print0(f"[INFO] Starting FSDP2 training")
        
        self.setup_pipeline()
        self.setup_fsdp2()
        self.setup_optim()
        self.validate_setup()

        # Create dataloader
        dataloader, sampler = create_dataloader(meta_folder, batch_size, self.world_size)
        
        steps_per_epoch = len(dataloader)
        total_steps = num_epochs * steps_per_epoch
        
        # Reconfigure scheduler with actual total steps
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-6
        )
        print0(f"[INFO] Scheduler configured for {total_steps} total steps")

        global_step = 0
        start_epoch = 0

        if resume_checkpoint:
            global_step = load_manual_lora_checkpoint(  # Changed from load_fsdp2_checkpoint
                self.pipe, self.model_map, self.transformer_names,
                self.optimizer, self.lr_scheduler, resume_checkpoint
            )
            start_epoch = global_step // steps_per_epoch

        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)
            
            config = {
                "model_id": self.model_id,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "learning_rate": self.learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "total_batch_size": batch_size * self.world_size,
                "boundary_ratio": self.boundary_ratio,
                "approach": "wan_fsdp2_lora",
                "train_transformer_2": self.train_transformer_2,
                "world_size": self.world_size,
            }
            
            wandb.init(
                project="wan-i2v-fsdp2-lora",
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
                iterable = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            epoch_loss = 0.0
            num_steps = 0

            for step, batch in enumerate(iterable):
                self.optimizer.zero_grad(set_to_none=True)
                
                try:
                    loss = step_fsdp2_transformer(
                        pipe=self.pipe,
                        model_map=self.model_map,
                        transformer_names=self.transformer_names,
                        batch=batch,
                        device=self.device,
                        bf16=self.bf16,
                        boundary_ratio=self.boundary_ratio if self.boundary_ratio is not None else 0.0,
                        train_transformer_2=self.train_transformer_2,
                    )
                
                except Exception as e:
                    print0(f"[ERROR] Training step failed at epoch {epoch}, step {step}: {e}")
                    video_shape = batch.get("video_latents", torch.tensor([])).shape
                    text_shape = batch.get("text_embeddings", torch.tensor([])).shape
                    print0(f"[DEBUG] Batch shapes - video: {video_shape}, text: {text_shape}")
                    raise
                
                loss.backward()

                # Gradient clipping
                active_name = "transformer_2" if self.train_transformer_2 else "transformer"
                trainable_params = [p for p in self.model_map[active_name]["lora_params"] if p.grad is not None]

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
                        iterable.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "avg": f"{(epoch_loss/num_steps):.4f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                            "gn": f"{grad_norm:.2f}",
                        })

                if save_every and (global_step % save_every == 0):
                    save_manual_lora_checkpoint(  # Changed from save_fsdp2_checkpoint
                        self.pipe, self.model_map, self.transformer_names,
                        self.optimizer, self.lr_scheduler, output_dir, global_step
                    )

            # Epoch summary
            avg_loss = epoch_loss / max(num_steps, 1)
            print0(f"[INFO] Epoch {epoch+1} complete. avg_loss={avg_loss:.6f}")
            
            if is_main_process():
                wandb.log({"epoch/avg_loss": avg_loss, "epoch/num": epoch + 1}, step=global_step)

        # Final checkpoint
        if is_main_process():
            print0("[INFO] Training complete, saving final checkpoint...")
            
            save_manual_lora_checkpoint(
                self.pipe, self.model_map, self.transformer_names,
                self.optimizer, self.lr_scheduler, output_dir, global_step
            )
            
            print0(f"[INFO] Saved final checkpoint at step {global_step}")
            
            wandb.finish()
        
        print0("[INFO] Training complete!")