# wan22_i2v_lora_training.py
# Distributed LoRA training for Wan 2.2 Image-to-Video
# Supports both transformer and transformer_2 parallelization

import logging
import os
import warnings
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from dataloader import MetaFilesDataset, collate_fn
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# --------------------------- rank-0 printing/logging ---------------------------


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def configure_logging_quiet_others():
    level = logging.INFO if is_main_process() else logging.ERROR
    logging.basicConfig(level=level)
    if dist.is_initialized() and dist.get_rank() != 0:
        warnings.filterwarnings("ignore")


# ------------------------------- setup helpers --------------------------------


def setup_dist():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    configure_logging_quiet_others()

    # Force BF16 path
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return local_rank


def cast_floating_params_and_buffers_(module: nn.Module, dtype: torch.dtype):
    """Cast ALL floating params & buffers to dtype."""
    for p in module.parameters(recurse=True):
        if p.dtype.is_floating_point:
            p.data = p.data.to(dtype)
    for name, b in module.named_buffers(recurse=True):
        if b.dtype.is_floating_point:
            b.data = b.to(dtype)


def collect_lora_target_modules(transformer: nn.Module) -> List[str]:
    """Collect target module names for LoRA."""
    target_modules = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            # Target specific layers based on WAN architecture
            if any(tag in name for tag in ["to_q", "to_k", "to_v", "to_out", "proj"]):
                target_modules.append(name)
            # Also target FFN layers that aren't already tensor-parallelized
            elif "ffn" in name and "net" not in name:  # Avoid TP'd ffn.net layers
                target_modules.append(name)
    return target_modules


# ---------------------- Tensor Parallel Application Functions -----------------------


def setup_tp_device_mesh(tp_size: int):
    """Setup 1D device mesh for tensor parallelism within each TP group"""
    return init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))


def apply_tp_to_ffn(ffn_module, tp_mesh):
    """Apply TP to FFN using PyTorch's built-in classes"""
    return parallelize_module(
        ffn_module,
        tp_mesh,
        parallelize_plan={
            "net.0.proj": ColwiseParallel(),  # GELU projection: 5120 -> 13824
            "net.2": RowwiseParallel(),  # Final projection: 13824 -> 5120
        },
    )


def apply_tp_to_transformer_block(block, tp_mesh):
    """Apply tensor parallelism to a WanTransformerBlock - conservative approach"""

    # SKIP attention parallelization due to norm_q/norm_k compatibility issues
    # The attention layers have RMSNorm applied after QKV projections, which creates
    # dimension mismatches when the projections are sharded but norms expect full tensors

    # Only parallelize FFN - this is safe and provides significant memory savings
    # FFN layers: (5120 -> 13824 -> 5120) are actually larger than attention layers
    if hasattr(block, "ffn"):
        block.ffn = apply_tp_to_ffn(block.ffn, tp_mesh)

    return block


def apply_tp_to_condition_embedder(embedder, tp_mesh):
    """Apply TP to condition embedder - keep all embeddings full-sized for compatibility"""

    # DO NOT parallelize any part of condition embedder
    # Reasons:
    # 1. Time embeddings (temb) must remain full-sized to match scale_shift_table
    # 2. Text embeddings flow to transformer blocks - keeping full-sized avoids potential issues
    # 3. Condition embedder is tiny compared to 40 transformer blocks - minimal TP benefit
    # 4. Safer and simpler to keep all condition embeddings as "interface" layers

    print0("[TP] Skipping condition embedder parallelization for compatibility")
    return embedder


def apply_tp_to_transformer(transformer, tp_mesh):
    """Apply tensor parallelism to entire WanTransformer3DModel"""
    print0(f"[TP] Applying PyTorch TP to transformer with {len(transformer.blocks)} blocks...")

    # Apply TP to condition embedder (text embedder only)
    transformer.condition_embedder = apply_tp_to_condition_embedder(transformer.condition_embedder, tp_mesh)

    # Apply TP to all transformer blocks
    for i, block in enumerate(transformer.blocks):
        transformer.blocks[i] = apply_tp_to_transformer_block(block, tp_mesh)
        if i % 10 == 0:
            print0(f"[TP] Parallelized block {i}/{len(transformer.blocks)}")

    # Apply TP to output projection
    transformer.proj_out = parallelize_module(
        transformer.proj_out,
        tp_mesh,
        parallelize_plan={"": RowwiseParallel()},  # 5120 -> 64
    )

    print0("[TP] PyTorch TP application complete!")
    return transformer


def prepare_image_latents_for_i2v(
    image_latents: torch.Tensor,
    num_frames: int,
    scheduler: FlowMatchEulerDiscreteScheduler,
    strength: float = 0.8,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Prepare image latents for I2V training following WAN pipeline logic.

    Args:
        image_latents: Encoded image latents (B, C, H, W)
        num_frames: Number of video frames
        scheduler: Diffusion scheduler
        strength: How much noise to add (0.0 = no noise, 1.0 = full noise)
        generator: Random generator for reproducibility

    Returns:
        Noisy latents ready for training (B, C, F, H, W)
    """
    batch_size, channels, height, width = image_latents.shape

    # 1. Broadcast image latents to all frames
    image_latents_broadcast = image_latents.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)

    # 2. Generate noise for all frames
    noise = torch.randn_like(image_latents_broadcast, generator=generator)

    # 3. Sample timestep based on strength
    timesteps = scheduler.timesteps
    timestep_idx = int(strength * (len(timesteps) - 1))
    timestep = timesteps[timestep_idx]

    # 4. Add noise using scheduler
    noisy_latents = scheduler.add_noise(image_latents_broadcast, noise, timestep)

    # 5. Apply temporal mask to preserve first frame (optional)
    frame_mask = torch.ones(num_frames, device=image_latents.device, dtype=image_latents.dtype)
    frame_mask[0] = 0.0  # Keep first frame pure (no noise)
    frame_mask = frame_mask.view(1, 1, num_frames, 1, 1)

    final_latents = image_latents_broadcast * (1 - frame_mask) + noisy_latents * frame_mask

    return final_latents, noise, timestep


class WanI2VLoRATrainer:
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 1e-4,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        tp_size: int = 2,
    ):
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.tp_size = tp_size
        self.bf16 = torch.bfloat16

        # Initialize distributed training
        self.local_rank = setup_dist()
        self.world_size = dist.get_world_size()
        self.device = torch.device("cuda", self.local_rank)

        # TP+DP configuration
        self.dp_size = self.world_size // self.tp_size
        self.tp_rank = dist.get_rank() % self.tp_size
        self.dp_rank = dist.get_rank() // self.tp_size

        print0(f"[INFO] WAN 2.2 I2V LoRA training: World={self.world_size}, TP={self.tp_size}, DP={self.dp_size}")
        print0(f"[INFO] Rank {dist.get_rank()}: TP rank {self.tp_rank}, DP rank {self.dp_rank}")

        # Create TP device mesh
        tp_group_ranks = [self.dp_rank * self.tp_size + i for i in range(self.tp_size)]
        self.tp_group = dist.new_group(tp_group_ranks)
        self.tp_mesh = setup_tp_device_mesh(self.tp_size)

        print0(f"[INFO] TP mesh created for ranks {tp_group_ranks}")

    def setup_pipeline(self):
        """Setup the WAN 2.2 pipeline with distributed parallelization."""
        print0("[INFO] Loading WAN 2.2 pipeline...")

        # Load VAE and pipeline on CPU first
        vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanImageToVideoPipeline.from_pretrained(self.model_id, vae=vae, torch_dtype=torch.float32)

        # Cast and move VAE and encoders to GPU
        cast_floating_params_and_buffers_(self.pipe.vae, torch.float32)  # Keep VAE in FP32
        self.pipe.vae.to(device=self.device)

        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            cast_floating_params_and_buffers_(self.pipe.text_encoder, self.bf16)
            self.pipe.text_encoder.to(device=self.device, dtype=self.bf16)

        if hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None:
            cast_floating_params_and_buffers_(self.pipe.image_encoder, self.bf16)
            self.pipe.image_encoder.to(device=self.device, dtype=self.bf16)

        # Freeze all components except transformers (which we'll add LoRA to)
        self.pipe.vae.requires_grad_(False)
        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            self.pipe.text_encoder.requires_grad_(False)
        if hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None:
            self.pipe.image_encoder.requires_grad_(False)

        print0("[INFO] Pipeline loaded and components frozen")

    def setup_lora_and_parallelization(self):
        """Setup LoRA adapters with PyTorch TP+DP parallelization."""
        print0("[INFO] Setting up LoRA adapters with PyTorch TP+DP...")

        # Collect transformer names
        transformer_names = [n for n in ("transformer", "transformer_2") if hasattr(self.pipe, n)]
        if not transformer_names:
            raise RuntimeError("No transformers found in pipeline")

        print0(f"[INFO] Found transformers: {transformer_names}")

        self.lora_models = []
        self.distributed_transformers = []

        # Apply TP+LoRA to each transformer
        for name in transformer_names:
            transformer = getattr(self.pipe, name)

            # Cast to BF16 and freeze base model
            cast_floating_params_and_buffers_(transformer, self.bf16)
            transformer.to(device=self.device, dtype=self.bf16)
            transformer.requires_grad_(False)  # Freeze base model

            if is_main_process():
                assert_uniform_dtype(transformer, self.bf16, name)

            # Step 1: Apply PyTorch tensor parallelism to base transformer
            print0(f"[INFO] Applying PyTorch TP to {name}...")
            transformer_tp = apply_tp_to_transformer(transformer, self.tp_mesh)

            # Step 2: Add LoRA to the TP'd transformer (LoRA will be replicated)
            print0(f"[INFO] Adding replicated LoRA to {name}...")

            # Collect LoRA target modules (avoid TP'd FFN layers)
            target_modules = collect_lora_target_modules(transformer_tp)
            print0(f"[INFO] {name} LoRA targets: {len(target_modules)} modules")

            # Setup LoRA config
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.0,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )

            # Apply LoRA to the TP'd transformer
            transformer_with_lora = get_peft_model(transformer_tp, lora_config)
            transformer_with_lora.print_trainable_parameters()

            # Step 3: Ensure LoRA parameters are replicated across all ranks
            print0(f"[INFO] Replicating LoRA parameters for {name}...")
            lora_params = {}

            # Collect all LoRA parameters
            for param_name, param in transformer_with_lora.named_parameters():
                if "lora_" in param_name and param.requires_grad:
                    lora_params[param_name] = param

            print0(f"[INFO] Found {len(lora_params)} LoRA parameters to replicate")

            # Broadcast LoRA parameters from rank 0 to all other ranks
            if dist.is_initialized():
                for param_name, param in lora_params.items():
                    # Ensure all ranks have identical LoRA parameters
                    dist.broadcast(param.data, src=0)

                print0(f"[INFO] LoRA parameters broadcast complete for {name}")

            # Set to training mode
            transformer_with_lora.train()

            # Replace in pipeline
            setattr(self.pipe, name, transformer_with_lora)
            self.lora_models.append(transformer_with_lora)
            self.distributed_transformers.append(transformer_tp)

        print0("[INFO] LoRA setup and PyTorch TP+DP parallelization complete")
        print0("[INFO] Base transformers: PyTorch TP SHARDED, LoRA parameters: REPLICATED")

    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler with custom LoRA gradient sync."""
        # Collect all LoRA parameters from both transformers
        self.lora_params = []
        for lora_model in self.lora_models:
            for param_name, param in lora_model.named_parameters():
                if "lora_" in param_name and param.requires_grad:
                    self.lora_params.append(param)

        print0(f"[INFO] Training {len(self.lora_params)} LoRA parameters")

        self.optimizer = torch.optim.AdamW(
            self.lora_params, lr=self.learning_rate, weight_decay=0.01, betas=(0.9, 0.999)
        )

        # Simple cosine scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)

    def sync_lora_gradients(self):
        """Manually synchronize LoRA gradients across all ranks."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return

        for param in self.lora_params:
            if param.grad is not None:
                # Average gradients across all ranks
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dist.get_world_size()

    def create_distributed_dataloader(self, meta_folder: str, batch_size: int = 1):
        """Create distributed dataloader for TP+DP training."""
        dataset = MetaFilesDataset(
            meta_folder=meta_folder,
            device="cpu",  # Load on CPU, will move to GPU in training loop
        )

        # For TP+DP, use DistributedSampler with DP groups
        # Each DP group sees different data, TP groups within DP see same data
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.dp_size,  # Number of DP groups
            rank=self.dp_rank,  # Which DP group this is
            shuffle=True,
        )

        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=2, pin_memory=True
        )

        return dataloader, sampler

    def training_step(self, batch: Dict) -> torch.Tensor:
        """Single training step."""
        # Move batch to device
        text_embeddings = batch["text_embeddings"].to(self.device, dtype=self.bf16)
        video_latents = batch["video_latents"].to(self.device, dtype=torch.float32)

        batch_size, channels, frames, height, width = video_latents.shape

        # For I2V training, we need:
        # 1. Input image latents (first frame as conditioning)
        # 2. Target video latents (what model should generate)
        image_latents = video_latents[:, :, 0]  # First frame as input (B, C, H, W)
        target_video_latents = video_latents  # Full video as target (B, C, F, H, W)

        # Sample random timesteps for each sample in batch
        timesteps = torch.randint(
            0, len(self.pipe.scheduler.timesteps), (batch_size,), device=self.device, dtype=torch.long
        )

        # Prepare initial latents by broadcasting image to all frames
        with torch.no_grad():
            # Broadcast image latents to all frames
            image_latents_broadcast = image_latents.unsqueeze(2).repeat(1, 1, frames, 1, 1)

            # Generate noise for the video latents
            noise = torch.randn_like(target_video_latents)

            # Add noise to target video latents (standard diffusion forward process)
            noisy_latents = self.pipe.scheduler.add_noise(target_video_latents, noise, timesteps)

            # Apply temporal mask to preserve first frame during training
            frame_mask = torch.ones(frames, device=self.device, dtype=torch.float32)
            frame_mask[0] = 0.0  # Keep first frame clean
            frame_mask = frame_mask.view(1, 1, frames, 1, 1)

            # Mix clean first frame with noisy subsequent frames
            final_latents = image_latents_broadcast * (1 - frame_mask) + noisy_latents * frame_mask

        # Convert to BF16 for training
        final_latents = final_latents.to(self.bf16)
        noise = noise.to(self.bf16)

        # Create timestep tensor for batch
        # timesteps = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)

        # Forward pass through transformers
        # Note: WAN 2.2 might use different transformers for different timestep ranges
        with torch.autocast(device_type="cuda", dtype=self.bf16):
            # Determine which transformer to use based on timestep
            # This logic depends on the boundary_ratio configuration
            if hasattr(self.pipe, "transformer_2") and hasattr(self.pipe.config, "boundary_ratio"):
                boundary_timestep = int(self.pipe.config.boundary_ratio * len(self.pipe.scheduler.timesteps))
                # Use transformer for high timesteps, transformer_2 for low timesteps
                use_transformer_2 = timesteps < boundary_timestep

                # Handle mixed batch - use transformer for simplicity, or split batch
                # For now, use transformer (you may want to batch by timestep range)
                transformer = self.pipe.transformer
            else:
                transformer = self.pipe.transformer

            # Predict noise
            noise_pred = transformer(
                sample=final_latents, timestep=timesteps, encoder_hidden_states=text_embeddings, return_dict=False
            )[0]

            # Compute loss (MSE between predicted and actual noise)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        return loss

    def train(
        self,
        meta_folder: str,
        num_epochs: int = 10,
        batch_size: int = 1,
        save_every: int = 1000,
        log_every: int = 100,
        output_dir: str = "./wan_i2v_lora_checkpoints",
    ):
        """Main training loop."""

        print0(f"[INFO] Starting training for {num_epochs} epochs")

        # Setup components
        self.setup_pipeline()
        self.setup_lora_and_parallelization()
        self.setup_optimizer_and_scheduler()

        # Create dataloader
        dataloader, sampler = self.create_distributed_dataloader(meta_folder, batch_size)

        # Create output directory
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

            # Initialize wandb logging
            wandb.init(
                project="wan-i2v-lora",
                config={
                    "model_id": self.model_id,
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "learning_rate": self.learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "num_frames": self.num_frames,
                    "height": self.height,
                    "width": self.width,
                    "tp_size": self.tp_size,
                    "dp_size": self.dp_size,
                    "parallelization": f"TP={self.tp_size}_DP={self.dp_size}",
                },
            )

        global_step = 0

        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)

            if is_main_process():
                pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            else:
                pbar = dataloader

            total_loss = 0.0

            for step, batch in enumerate(pbar):
                self.optimizer.zero_grad()

                try:
                    loss = self.training_step(batch)
                    loss.backward()

                    # Manually sync LoRA gradients across ranks
                    self.sync_lora_gradients()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.lora_params, max_norm=1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()

                    total_loss += loss.item()
                    global_step += 1

                    # Logging
                    if global_step % log_every == 0 and is_main_process():
                        avg_loss = total_loss / log_every if step > 0 else loss.item()
                        current_lr = self.optimizer.param_groups[0]["lr"]

                        print0(f"Step {global_step}: Loss={avg_loss:.6f}, LR={current_lr:.2e}")

                        wandb.log(
                            {"loss": avg_loss, "learning_rate": current_lr, "epoch": epoch, "global_step": global_step}
                        )

                        total_loss = 0.0

                    # Save checkpoint
                    if global_step % save_every == 0 and is_main_process():
                        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)

                        # Save LoRA weights for each transformer
                        for i, lora_model in enumerate(self.lora_models):
                            lora_model.save_pretrained(os.path.join(checkpoint_dir, f"transformer_{i}_lora"))

                        print0(f"[INFO] Saved checkpoint at step {global_step}")

                except Exception as e:
                    print0(f"[ERROR] Training step failed: {e}")
                    continue

            print0(f"[INFO] Completed epoch {epoch + 1}")

        # Final save
        if is_main_process():
            final_dir = os.path.join(output_dir, "final")
            os.makedirs(final_dir, exist_ok=True)

            for i, lora_model in enumerate(self.lora_models):
                lora_model.save_pretrained(os.path.join(final_dir, f"transformer_{i}_lora"))

            print0("[INFO] Training completed and final model saved")
            wandb.finish()


def main():
    """Main training function."""
    trainer = WanI2VLoRATrainer(
        model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        lora_rank=16,
        lora_alpha=32,
        learning_rate=1e-4,
        num_frames=81,
        height=480,
        width=832,
        tp_size=int(os.environ.get("TP_SIZE", "2")),  # Get TP size from environment
    )

    trainer.train(
        meta_folder="/code/hdvilla_sample/processed_meta",
        num_epochs=5,
        batch_size=1,  # Keep small for memory
        save_every=500,
        log_every=50,
        output_dir="./wan_i2v_lora_outputs",
    )


if __name__ == "__main__":
    main()

# Run with:
# export TP_SIZE=8
# torchrun --nproc-per-node=8 wan22_i2v_lora_training.py
