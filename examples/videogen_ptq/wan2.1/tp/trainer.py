import os
from typing import Dict, Optional

import torch
import torch.distributed as dist
import wandb
from checkpoint_io import load_lora_checkpoint, save_lora_checkpoint
from data_utils import create_dataloader
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from dist_utils import cast_model_to_dtype, is_main_process, print0, setup_distributed
from tp_hybrid import manual_allreduce_lora_gradients, setup_hybrid_for_pipe
from training_step import step_dual_transformer


class WanI2VLoRATrainer:
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 1e-4,
        boundary_ratio: Optional[float] = 0.5,
        use_peft_lora: bool = False,
        debug: bool = False,
        train_transformer_2: bool = False,  # Added flag
    ):
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.boundary_ratio = boundary_ratio
        self.use_peft_lora = use_peft_lora
        self.bf16 = torch.bfloat16
        self.debug = debug
        self.train_transformer_2 = train_transformer_2  # Store the flag

        self.local_rank = setup_distributed()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device("cuda", self.local_rank)

        print0("[INFO] WAN 2.2 I2V LoRA trainer init")
        print0(f"[INFO] world_size={self.world_size} local_rank={self.local_rank}")
        print0(f"[INFO] LoRA rank={lora_rank} alpha={lora_alpha}")
        print0(f"[INFO] Training transformer_2: {train_transformer_2}")
        print0("[INFO] Using Tensor Parallelism (TP) for transformers")

        self.pipe = None
        self.model_map: Dict[str, Dict] = {}
        self.transformer_names = []
        self.optimizer = None
        self.lr_scheduler = None

    def setup_pipeline(self):
        print0("[INFO] Loading pipeline ...")
        print0(f"[DEBUG] Model ID being used: {self.model_id}")

        # Debug VAE first
        vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)
        print0(
            f"[DEBUG] VAE latent_channels/z_dim: {getattr(vae.config, 'latent_channels', getattr(vae.config, 'z_dim', 'NOT_FOUND'))}"
        )

        # Load pipeline
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id, vae=vae, torch_dtype=torch.float32, boundary_ratio=self.boundary_ratio
        )

        # Debug transformer architecture
        print0("[DEBUG] === TRANSFORMER ARCHITECTURE ===")
        print0(f"[DEBUG] Transformer type: {type(self.pipe.transformer)}")

        # Check patch embedding layer specifically
        patch_emb = self.pipe.transformer.patch_embedding
        print0(f"[DEBUG] Patch embedding weight shape: {patch_emb.weight.shape}")
        print0(f"[DEBUG] Expected input channels: {patch_emb.weight.shape[1]}")

        # Check if there are multiple transformers
        if hasattr(self.pipe, "transformer_2"):
            print0(
                f"[DEBUG] Transformer_2 expected input channels: {self.pipe.transformer_2.patch_embedding.weight.shape[1]}"
            )

        # Check pipeline configuration for I2V specifics
        print0(f"[DEBUG] Pipeline has image_encoder: {hasattr(self.pipe, 'image_encoder')}")
        print0(f"[DEBUG] Pipeline boundary_ratio: {getattr(self.pipe, 'boundary_ratio', 'NOT_SET')}")

        # Setup other pipeline components
        cast_model_to_dtype(self.pipe.vae, self.bf16)
        self.pipe.vae.to(self.device, dtype=self.bf16).requires_grad_(False)

        if getattr(self.pipe, "text_encoder", None) is not None:
            cast_model_to_dtype(self.pipe.text_encoder, self.bf16)
            self.pipe.text_encoder.to(self.device, dtype=self.bf16).requires_grad_(False)

        if getattr(self.pipe, "image_encoder", None) is not None:
            cast_model_to_dtype(self.pipe.image_encoder, self.bf16)
            self.pipe.image_encoder.to(self.device, dtype=self.bf16).requires_grad_(False)

        print0("[INFO] Pipeline ready")

    def setup_hybrid(self):
        print0("[INFO] Setting up Tensor Parallelism + LoRA...")
        self.model_map, self.transformer_names = setup_hybrid_for_pipe(
            self.pipe,
            device=self.device,
            bf16=self.bf16,
            local_rank=self.local_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_peft=self.use_peft_lora,
            train_transformer_2=self.train_transformer_2,
        )

        # No need for CPU offloading anymore - unused transformer was never loaded!
        print0("[INFO] Memory optimization: Only active transformer was loaded to GPU")

        # Skip TP forward pass test to save memory and time
        print0("[INFO] Skipping TP forward pass test for memory savings")

        print0("[INFO] TP + LoRA setup complete")

    def setup_optim(self):
        # Only collect LoRA parameters from the active transformer
        active_transformer = "transformer_2" if self.train_transformer_2 else "transformer"

        if active_transformer not in self.model_map:
            raise RuntimeError(f"Active transformer {active_transformer} not found in model_map")

        all_params = self.model_map[active_transformer]["lora_params"]

        if not all_params:
            raise RuntimeError(f"No trainable LoRA parameters found in {active_transformer}!")

        print0(f"[INFO] Setting up optimizer for {len(all_params)} LoRA parameters from {active_transformer}")

        # Use regular AdamW optimizer (more stable than 8-bit version)
        self.optimizer = torch.optim.AdamW(all_params, lr=self.learning_rate, weight_decay=0.01, betas=(0.9, 0.999))
        print0("[INFO] Using regular AdamW optimizer")

        # Temporary scheduler; will reconfigure after dataloader length is known
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)

    def train(
        self,
        meta_folder: str,
        num_epochs: int = 10,
        batch_size: int = 1,
        save_every: int = 1000,
        log_every: int = 100,
        validate_every: int = 0,
        output_dir: str = "./wan_i2v_outputs",
        resume_checkpoint: Optional[str] = None,
    ):
        self.setup_pipeline()
        self.setup_hybrid()
        self.setup_optim()

        dataloader, sampler = create_dataloader(meta_folder, batch_size, self.world_size)
        steps_per_epoch = len(dataloader)
        total_steps = num_epochs * steps_per_epoch
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)
        print0(f"[INFO] Scheduler T_max={total_steps}")

        global_step = 0
        start_epoch = 0

        if resume_checkpoint:
            global_step = load_lora_checkpoint(
                self.pipe, self.model_map, self.transformer_names, self.optimizer, self.lr_scheduler, resume_checkpoint
            )
            start_epoch = global_step // steps_per_epoch

        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)
            wandb.init(
                project="wan-i2v-diffusers-lora",
                config=dict(
                    model_id=self.model_id,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lr=self.learning_rate,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    total_batch_size=batch_size * self.world_size,
                    boundary_ratio=self.boundary_ratio,
                    approach="wan_tp_lora_hybrid",
                    parallelism="tensor_parallel",
                    train_transformer_2=self.train_transformer_2,  # Track which transformer
                ),
                resume=resume_checkpoint is not None,
            )
        if dist.is_initialized():
            dist.barrier()

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

                loss = step_dual_transformer(
                    pipe=self.pipe,
                    model_map=self.model_map,
                    transformer_names=self.transformer_names,
                    batch=batch,
                    device=self.device,
                    bf16=self.bf16,
                    boundary_ratio=self.boundary_ratio if self.boundary_ratio is not None else 0.0,
                    train_transformer_2=self.train_transformer_2,  # Pass the flag
                )
                loss.backward()

                # Manual gradient synchronization for LoRA parameters (TP doesn't auto-sync LoRA)
                manual_allreduce_lora_gradients(self.model_map, self.transformer_names)

                # Gradient clipping - only for active transformer
                active_transformer = "transformer_2" if self.train_transformer_2 else "transformer"
                all_params = [p for p in self.model_map[active_transformer]["lora_params"] if p.grad is not None]

                grad_norm = 0.0
                if all_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

                self.optimizer.step()
                self.lr_scheduler.step()

                epoch_loss += loss.item()
                num_steps += 1
                global_step += 1

                if is_main_process() and (global_step % log_every == 0):
                    wandb.log(
                        dict(
                            train_loss=loss.item(),
                            train_avg_loss=epoch_loss / num_steps,
                            lr=self.optimizer.param_groups[0]["lr"],
                            grad_norm=grad_norm,
                            epoch=epoch,
                            global_step=global_step,
                        ),
                        step=global_step,
                    )
                    if hasattr(iterable, "set_postfix"):
                        iterable.set_postfix(
                            loss=f"{loss.item():.4f}",
                            avg=f"{(epoch_loss / num_steps):.4f}",
                            lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                            gn=f"{grad_norm:.2f}",
                        )

                if save_every and (global_step % save_every == 0):
                    save_lora_checkpoint(
                        self.pipe,
                        self.model_map,
                        self.transformer_names,
                        self.optimizer,
                        self.lr_scheduler,
                        output_dir,
                        global_step,
                    )

            print0(f"[INFO] Epoch {epoch + 1} done. avg_loss={epoch_loss / max(num_steps, 1):.6f}")
            if is_main_process():
                wandb.log({"epoch/avg_loss": epoch_loss / max(num_steps, 1), "epoch/num": epoch + 1}, step=global_step)

        if is_main_process():
            save_lora_checkpoint(
                self.pipe,
                self.model_map,
                self.transformer_names,
                self.optimizer,
                self.lr_scheduler,
                output_dir,
                global_step,
            )
            final_dir = os.path.join(output_dir, "final")
            os.makedirs(final_dir, exist_ok=True)
            old = {}
            for n in self.transformer_names:
                old[n] = getattr(self.pipe, n)
                setattr(self.pipe, n, self.model_map[n]["base_transformer"])
            try:
                self.pipe.save_lora_weights(final_dir)
                print0(f"[INFO] Saved final LoRA weights to {final_dir}")
            finally:
                for n in self.transformer_names:
                    setattr(self.pipe, n, old[n])
            wandb.finish()
        print0("[INFO] Training complete")
