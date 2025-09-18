# wan22_i2v_lora_training.py
# Distributed LoRA training for Wan 2.2 Image-to-Video
# Uses PyTorch FSDP with proper I2V conditioning

import inspect
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from dataloader import MetaFilesDataset, collate_fn
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# --------------------------- Utility Functions ---------------------------


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    return not dist.is_initialized() or dist.get_rank() == 0


def print0(*args, **kwargs):
    """Print only from rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def configure_logging():
    """Configure logging to be quiet on non-main processes."""
    level = logging.INFO if is_main_process() else logging.ERROR
    logging.basicConfig(level=level)
    if dist.is_initialized() and dist.get_rank() != 0:
        warnings.filterwarnings("ignore")


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    configure_logging()

    # Force BF16 precision settings
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    return local_rank


def cast_model_to_dtype(module: nn.Module, dtype: torch.dtype):
    """Cast all floating point parameters and buffers to specified dtype."""
    for param in module.parameters(recurse=True):
        if param.dtype.is_floating_point:
            param.data = param.data.to(dtype)

    for buffer in module.buffers(recurse=True):
        if buffer.dtype.is_floating_point:
            buffer.data = buffer.data.to(dtype)


def validate_model_dtype(module: nn.Module, expected_dtype: torch.dtype, model_name: str):
    """Validate that all floating point parameters have the expected dtype."""
    param_dtypes = {p.dtype for p in module.parameters() if p.dtype.is_floating_point}
    buffer_dtypes = {b.dtype for b in module.buffers() if b.dtype.is_floating_point}
    all_dtypes = param_dtypes | buffer_dtypes

    if all_dtypes != {expected_dtype}:
        raise ValueError(f"{model_name} has mixed dtypes {all_dtypes}, expected {expected_dtype}")


def collect_lora_target_modules(transformer: nn.Module) -> List[str]:
    """Collect target module names for LoRA adaptation."""
    target_modules = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            # Target attention and MLP projections
            if any(keyword in name for keyword in ["to_q", "to_k", "to_v", "to_out", "fc1", "fc2", "proj"]):
                target_modules.append(name)
    return target_modules


def prepare_image_latents_for_i2v(
    image_latents: torch.Tensor,
    num_frames: int,
    scheduler: FlowMatchEulerDiscreteScheduler,
    strength: float = 0.8,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare image latents for I2V training following WAN pipeline logic.

    Args:
        image_latents: Encoded image latents (B, C, H, W)
        num_frames: Number of video frames
        scheduler: Diffusion scheduler
        strength: How much noise to add (0.0 = no noise, 1.0 = full noise)
        generator: Random generator for reproducibility

    Returns:
        Tuple of (final_latents, noise, timestep)
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

    # 5. Apply temporal mask to preserve first frame
    frame_mask = torch.ones(num_frames, device=image_latents.device, dtype=image_latents.dtype)
    frame_mask[0] = 0.0  # Keep first frame clean
    frame_mask = frame_mask.view(1, 1, num_frames, 1, 1)

    # 6. Combine clean first frame with noisy subsequent frames
    final_latents = image_latents_broadcast * (1 - frame_mask) + noisy_latents * frame_mask

    return final_latents, noise, timestep


class WanI2VLoRATrainer:
    """WAN 2.2 Image-to-Video LoRA trainer using PyTorch FSDP."""

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 1e-4,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
    ):
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.bf16 = torch.bfloat16

        # Initialize distributed training
        self.local_rank = setup_distributed()
        self.world_size = dist.get_world_size()
        self.device = torch.device("cuda", self.local_rank)

        print0("[INFO] WAN 2.2 I2V LoRA training initialized")
        print0(f"[INFO] World size: {self.world_size} GPUs")
        print0(f"[INFO] Local rank: {self.local_rank}")

        # Initialize components
        self.pipe = None
        self.lora_models = []
        self.optimizer = None
        self.lr_scheduler = None

    def setup_pipeline(self):
        """Load and setup the WAN 2.2 pipeline components."""
        print0("[INFO] Loading WAN 2.2 I2V pipeline...")

        # Load pipeline components
        vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)

        self.pipe = WanImageToVideoPipeline.from_pretrained(self.model_id, vae=vae, torch_dtype=torch.float32)

        # Cast and move frozen components to device
        cast_model_to_dtype(self.pipe.vae, self.bf16)
        self.pipe.vae.to(device=self.device, dtype=self.bf16)
        self.pipe.vae.requires_grad_(False)

        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            cast_model_to_dtype(self.pipe.text_encoder, self.bf16)
            self.pipe.text_encoder.to(device=self.device, dtype=self.bf16)
            self.pipe.text_encoder.requires_grad_(False)

        if hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None:
            cast_model_to_dtype(self.pipe.image_encoder, self.bf16)
            self.pipe.image_encoder.to(device=self.device, dtype=self.bf16)
            self.pipe.image_encoder.requires_grad_(False)

        print0("[INFO] Pipeline loaded, frozen components configured")

    def debug_transformer_signature(self, transformer: nn.Module, name: str):
        """Debug transformer forward signature for compatibility."""
        try:
            sig = inspect.signature(transformer.forward)
            params = list(sig.parameters.keys())
            print0(f"[DEBUG] {name} forward signature: {params}")

            expected_params = ["sample", "timestep", "encoder_hidden_states"]
            has_standard_sig = all(param in params for param in expected_params)
            print0(f"[DEBUG] {name} has standard diffusion signature: {has_standard_sig}")

            return has_standard_sig, params
        except Exception as e:
            print0(f"[WARNING] Could not inspect {name} signature: {e}")
            return False, []

    def setup_lora_and_fsdp(self):
        """Setup LoRA adapters outside FSDP and base transformers with FSDP."""
        print0("[INFO] Setting up LoRA (outside FSDP) + base transformers (with FSDP)...")

        # Find available transformers
        transformer_names = []
        for name in ["transformer", "transformer_2"]:
            if hasattr(self.pipe, name) and getattr(self.pipe, name) is not None:
                transformer_names.append(name)

        if not transformer_names:
            raise RuntimeError("No transformers found in pipeline")

        print0(f"[INFO] Found transformers: {transformer_names}")

        # FSDP configuration for base transformers only
        mixed_precision_policy = MixedPrecision(
            param_dtype=self.bf16,
            reduce_dtype=self.bf16,
            buffer_dtype=self.bf16,
        )

        self.lora_models = []
        self.base_transformers = []

        for name in transformer_names:
            transformer = getattr(self.pipe, name)

            # Cast to BF16 and freeze base model
            cast_model_to_dtype(transformer, self.bf16)
            transformer.requires_grad_(False)

            # Validate dtype consistency
            if is_main_process():
                validate_model_dtype(transformer, self.bf16, name)

            print0(f"[INFO] Debugging {name} signature...")
            self.debug_transformer_signature(transformer, name)

            # Move base transformer to device
            transformer = transformer.to(self.device)

            # Wrap ONLY the base transformer with FSDP
            print0(f"[INFO] Wrapping base {name} with FSDP...")
            fsdp_base_transformer = FSDP(
                transformer,
                sharding_strategy=ShardingStrategy.NO_SHARD,
                mixed_precision=mixed_precision_policy,
                auto_wrap_policy=None,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=self.local_rank,
                sync_module_states=True,
                param_init_fn=None,
                use_orig_params=True,
            )

            # Now add LoRA adapters OUTSIDE of FSDP
            print0(f"[INFO] Adding LoRA to {name} (outside FSDP)...")
            target_modules = collect_lora_target_modules(fsdp_base_transformer)
            print0(f"[INFO] {name} LoRA targeting {len(target_modules)} modules")

            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.0,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )

            # Apply LoRA to the FSDP-wrapped base transformer
            transformer_with_lora = get_peft_model(fsdp_base_transformer, lora_config)
            if is_main_process():
                transformer_with_lora.print_trainable_parameters()

            # Cast LoRA parameters to BF16 (they're added as float32)
            for name_param, param in transformer_with_lora.named_parameters():
                if "lora_" in name_param and param.requires_grad:
                    param.data = param.data.to(self.bf16)

            # Validate setup
            print0(f"[INFO] Validating {name} setup...")
            lora_param_count = sum(
                1
                for param_name, param in transformer_with_lora.named_parameters()
                if "lora_" in param_name and param.requires_grad
            )
            print0(f"[INFO] {name} has {lora_param_count} trainable LoRA parameters (outside FSDP)")

            transformer_with_lora.train()

            # Replace in pipeline and track
            setattr(self.pipe, name, transformer_with_lora)
            self.lora_models.append(transformer_with_lora)
            self.base_transformers.append(fsdp_base_transformer)

        print0("[INFO] Setup complete: Base transformers in FSDP, LoRA adapters outside FSDP")

    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler for LoRA parameters outside FSDP."""
        # Collect trainable LoRA parameters (they're outside FSDP)
        lora_params = []
        for model in self.lora_models:
            for param_name, param in model.named_parameters():
                if "lora_" in param_name and param.requires_grad:
                    lora_params.append(param)

        print0(f"[INFO] Optimizing {len(lora_params)} LoRA parameters (outside FSDP)")

        # Since LoRA params are outside FSDP, we need to handle gradient sync manually
        self.lora_params = lora_params

        self.optimizer = torch.optim.AdamW(lora_params, lr=self.learning_rate, weight_decay=0.01, betas=(0.9, 0.999))

        # Cosine annealing scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)

    def create_dataloader(self, meta_folder: str, batch_size: int = 1):
        """Create distributed dataloader."""
        dataset = MetaFilesDataset(
            meta_folder=meta_folder,
            device="cpu",  # Load on CPU, move to GPU in training loop
        )

        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=dist.get_rank(), shuffle=True)

        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=2, pin_memory=True
        )

        return dataloader, sampler

    def training_step(self, batch: Dict) -> torch.Tensor:
        """Execute single training step with proper I2V conditioning."""
        # Extract data from batch (only these 4 fields are available)
        text_embeddings = batch["text_embeddings"]  # (batch_size, seq_len, embed_dim)
        video_latents = batch["video_latents"]  # (batch_size, channels, frames, h, w)
        metadata = batch["metadata"]  # List of metadata dicts
        file_info = batch["file_info"]  # List of file info dicts

    def training_step(self, batch: Dict) -> torch.Tensor:
        """Execute single training step with proper I2V conditioning."""
        # Extract data from batch (only these 4 fields are available)
        text_embeddings = batch["text_embeddings"]  # (batch_size, seq_len, embed_dim)
        video_latents = batch["video_latents"]  # (batch_size, channels, frames, h, w)
        metadata = batch["metadata"]  # List of metadata dicts
        file_info = batch["file_info"]  # List of file info dicts

    def training_step(self, batch: Dict) -> torch.Tensor:
        """Execute single training step with proper I2V conditioning."""
        # Extract data from batch (only these 4 fields are available)
        text_embeddings = batch["text_embeddings"]  # (batch_size, seq_len, embed_dim)
        video_latents = batch["video_latents"]  # (batch_size, channels, frames, h, w)
        metadata = batch["metadata"]  # List of metadata dicts
        file_info = batch["file_info"]  # List of file info dicts

        # Move tensors to device with proper dtypes
        text_embeddings = text_embeddings.to(self.device, dtype=self.bf16)
        video_latents = video_latents.to(self.device, dtype=torch.float32)

        # Debug: Check actual tensor shapes
        if is_main_process():
            print0(f"[DEBUG] text_embeddings shape: {text_embeddings.shape}")
            print0(f"[DEBUG] video_latents shape: {video_latents.shape}")

        # Handle the actual dataloader format
        # Your dataloader provides samples with shape (1, C, F, H, W)
        # When batched, this becomes (batch_size, 1, C, F, H, W) - 6D tensor
        if len(video_latents.shape) == 6:
            # Remove the extra dimension: (batch_size, 1, C, F, H, W) -> (batch_size, C, F, H, W)
            video_latents = video_latents.squeeze(1)
            batch_size, channels, frames, height, width = video_latents.shape
        elif len(video_latents.shape) == 5:
            batch_size, channels, frames, height, width = video_latents.shape
        else:
            raise ValueError(f"Unexpected video_latents shape: {video_latents.shape}")

        # Similarly handle text_embeddings if needed
        if len(text_embeddings.shape) == 4:
            # (batch_size, 1, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
            text_embeddings = text_embeddings.squeeze(1)

        if is_main_process():
            print0(f"[DEBUG] After reshape - text_embeddings: {text_embeddings.shape}")
            print0(f"[DEBUG] After reshape - video_latents: {video_latents.shape}")
            print0(f"[DEBUG] Parsed dimensions: B={batch_size}, C={channels}, F={frames}, H={height}, W={width}")

        # Extract first frame as the base for I2V generation
        # In I2V training, we use the first frame as the starting point
        image_latents = video_latents[:, :, 0]  # (B, C, H, W)

        # Sample random timesteps for diffusion training
        timesteps = torch.randint(
            0, len(self.pipe.scheduler.timesteps), (batch_size,), device=self.device, dtype=torch.long
        )

        # Prepare latents for I2V training
        with torch.no_grad():
            # 1. Broadcast first frame to all temporal positions
            # This simulates the I2V pipeline where we start from a single image
            image_latents_broadcast = image_latents.unsqueeze(2).repeat(1, 1, frames, 1, 1)

            # 2. Generate noise for the full video sequence
            noise = torch.randn_like(video_latents)

            # 3. Add noise to the TARGET video (what model should learn to denoise to)
            noisy_latents = self.pipe.scheduler.add_noise(video_latents, noise, timesteps)

            # 4. Create temporal mask for I2V conditioning
            # In I2V: first frame stays clean (image input), subsequent frames are noisy
            frame_mask = torch.ones(frames, device=self.device, dtype=torch.float32)
            frame_mask[0] = 0.0  # First frame = clean image input
            frame_mask = frame_mask.view(1, 1, frames, 1, 1)

            # 5. INPUT to model: clean first frame + noisy subsequent frames
            # This teaches the model: "given clean frame 0, denoise frames 1-N"
            input_latents = (
                image_latents_broadcast * (1 - frame_mask)  # Clean first frame
                + noisy_latents * frame_mask  # Noisy subsequent frames
            )

        # Convert to BF16 for model forward pass
        input_latents = input_latents.to(self.bf16)
        noise = noise.to(self.bf16)

        # Forward pass through transformer(s)
        with torch.autocast(device_type="cuda", dtype=self.bf16):
            # Select appropriate transformer based on pipeline configuration
            transformer = self.pipe.transformer

            # Predict noise - use exact signature from debug
            try:
                noise_pred = transformer(
                    sample=input_latents, timestep=timesteps, encoder_hidden_states=text_embeddings, return_dict=False
                )[0]
            except TypeError as e:
                # Fallback: try without return_dict if signature doesn't support it
                print0(f"[WARNING] Trying fallback forward call due to: {e}")
                noise_pred = transformer(input_latents, timesteps, text_embeddings)
                if isinstance(noise_pred, tuple):
                    noise_pred = noise_pred[0]

            # Compute MSE loss between predicted and actual noise
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        return loss

    def save_checkpoint(self, output_dir: str, step: int):
        """Save LoRA checkpoint."""
        if not is_main_process():
            return

        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save LoRA adapters
        for i, lora_model in enumerate(self.lora_models):
            lora_save_path = os.path.join(checkpoint_dir, f"transformer_{i}_lora")

            # Extract PEFT model from FSDP wrapper
            peft_model = lora_model.module if hasattr(lora_model, "module") else lora_model
            peft_model.save_pretrained(lora_save_path)

        # Save training state
        torch.save(
            {
                "step": step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, "training_state.pt"),
        )

        print0(f"[INFO] Checkpoint saved at step {step}")

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

        # Setup all components
        try:
            self.setup_pipeline()
            self.setup_lora_and_fsdp()
            self.setup_optimizer_and_scheduler()
        except Exception as e:
            print0(f"[ERROR] Setup failed: {e}")
            raise

        # Create dataloader
        dataloader, sampler = self.create_dataloader(meta_folder, batch_size)

        # Initialize output directory and logging
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)
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
                },
            )

        global_step = 0

        # Training loop
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)

            if is_main_process():
                pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            else:
                pbar = dataloader

            epoch_loss = 0.0

            for step, batch in enumerate(pbar):
                self.optimizer.zero_grad()

                try:
                    # Training step
                    loss = self.training_step(batch)
                    loss.backward()

                    # Manual gradient synchronization for LoRA parameters (outside FSDP)
                    if dist.is_initialized():
                        for param in self.lora_params:
                            if param.grad is not None:
                                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                                param.grad /= self.world_size

                    # Gradient clipping on LoRA parameters
                    torch.nn.utils.clip_grad_norm_(self.lora_params, max_norm=1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()

                    epoch_loss += loss.item()
                    global_step += 1

                    # Logging
                    if global_step % log_every == 0 and is_main_process():
                        avg_loss = epoch_loss / (step + 1)
                        current_lr = self.optimizer.param_groups[0]["lr"]

                        print0(f"Step {global_step}: Loss={avg_loss:.6f}, LR={current_lr:.2e}")

                        wandb.log(
                            {"loss": avg_loss, "learning_rate": current_lr, "epoch": epoch, "global_step": global_step}
                        )

                    # Save checkpoint
                    if global_step % save_every == 0:
                        self.save_checkpoint(output_dir, global_step)

                except Exception as e:
                    print0(f"[ERROR] Training step failed: {e}")
                    print0("[ERROR] Stopping training due to critical error")
                    import traceback

                    traceback.print_exc()
                    if is_main_process():
                        wandb.finish()
                    return  # Stop training completely

            print0(f"[INFO] Epoch {epoch + 1} completed, avg loss: {epoch_loss / (step + 1):.6f}")

        # Final save
        if is_main_process():
            self.save_checkpoint(output_dir, global_step)
            wandb.finish()
            print0("[INFO] Training completed successfully")


def main():
    """Main function."""
    trainer = WanI2VLoRATrainer(
        model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        lora_rank=16,
        lora_alpha=32,
        learning_rate=1e-4,
        num_frames=81,
        height=480,
        width=832,
    )

    trainer.train(
        meta_folder="/linnanw/hdvilla_sample/processed_meta",
        num_epochs=5,
        batch_size=1,
        save_every=500,
        log_every=50,
        output_dir="./wan_i2v_lora_outputs",
    )


if __name__ == "__main__":
    main()

# Run with: torchrun --nproc-per-node=8 wan22_i2v_lora_training.py
