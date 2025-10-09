# wan22_i2v_lora_training_diffusers_native.py
# WAN 2.2 I2V LoRA training using Diffusers-native LoRA (no PEFT)
# Hybrid approach: FSDP on base transformers + manual gradient sync on LoRA params

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
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
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
    return not dist.is_initialized() or dist.get_rank() == 0


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def configure_logging():
    level = logging.INFO if is_main_process() else logging.ERROR
    logging.basicConfig(level=level)
    if dist.is_initialized() and dist.get_rank() != 0:
        warnings.filterwarnings("ignore")


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    configure_logging()

    # Force BF16 settings (WAN models are bf16-friendly)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Seed per-rank
    torch.manual_seed(42 + dist.get_rank())
    torch.cuda.manual_seed_all(42 + dist.get_rank())
    return local_rank


def cast_model_to_dtype(module: nn.Module, dtype: torch.dtype):
    for p in module.parameters(recurse=True):
        if p.dtype.is_floating_point:
            p.data = p.data.to(dtype)
    for b in module.buffers(recurse=True):
        if b.dtype.is_floating_point:
            b.data = b.data.to(dtype)


# --------------------------- LoRA Helpers (WAN) ---------------------------


def _install_wan_lora_processors(transformer: nn.Module):
    """
    Replace every attention processor with LoRAAttnProcessor2_0 (zero-arg for WAN).
    """
    if not hasattr(transformer, "attn_processors") or len(transformer.attn_processors) == 0:
        raise RuntimeError("Transformer has no attn_processors; cannot install LoRA.")

    attn_procs = {}
    for name in transformer.attn_processors.keys():
        attn_procs[name] = LoRAAttnProcessor2_0()
    transformer.set_attn_processor(attn_procs)
    print0(f"[INFO] Installed {len(attn_procs)} WAN LoRA processors")


def _collect_wan_lora_params(transformer: nn.Module) -> List[nn.Parameter]:
    """
    Collect nn.Parameters from WAN's LoRA processors by introspection.
    Expected attributes on each processor: to_q_lora, to_k_lora, to_v_lora, to_out_lora
    Each of those (when present) has submodules lora_up, lora_down containing Parameters.
    """
    if not hasattr(transformer, "attn_processors"):
        return []

    lora_params: List[nn.Parameter] = []

    def _maybe_collect(module):
        if module is None:
            return
        # Try both common patterns
        submods = []
        if hasattr(module, "lora_up"):
            submods.append(module.lora_up)
        if hasattr(module, "lora_down"):
            submods.append(module.lora_down)
        # A few builds use weight_A/weight_B naming; catch-all:
        for attr in ("A", "B", "up", "down"):
            sub = getattr(module, attr, None)
            if isinstance(sub, nn.Module):
                submods.append(sub)

        for sm in submods:
            for _, p in sm.named_parameters(recurse=True):
                p.requires_grad = True
                lora_params.append(p)

    for proc in transformer.attn_processors.values():
        # Standard names in diffusers for LoRAAttnProcessor2_0
        for attr in ("to_q_lora", "to_k_lora", "to_v_lora", "to_out_lora"):
            _maybe_collect(getattr(proc, attr, None))

    # Deduplicate while preserving object identity
    seen = set()
    unique_params = []
    for p in lora_params:
        if id(p) not in seen:
            unique_params.append(p)
            seen.add(id(p))
    return unique_params


def _move_params(params: List[nn.Parameter], device: torch.device, dtype: torch.dtype):
    for p in params:
        p.data = p.data.to(device=device, dtype=dtype)


# --------------------------- Trainer ---------------------------


class WanI2VDiffusersLoRATrainer:
    """WAN 2.2 I2V trainer using Diffusers-native LoRA with FSDP hybrid approach."""

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        lora_rank: int = 16,  # not used by WAN’s zero-arg proc – kept for CLI parity
        lora_alpha: int = 32,  # not used by WAN’s zero-arg proc – kept for CLI parity
        learning_rate: float = 1e-4,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        boundary_ratio: Optional[float] = 0.5,
        debug: bool = False,
    ):
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.boundary_ratio = boundary_ratio
        self.bf16 = torch.bfloat16
        self.debug = debug

        self.local_rank = setup_distributed()
        self.world_size = dist.get_world_size()
        self.device = torch.device("cuda", self.local_rank)

        print0("[INFO] WAN 2.2 I2V Diffusers-native LoRA training initialized")
        print0(f"[INFO] world_size={self.world_size}  local_rank={self.local_rank}")
        print0(f"[INFO] boundary_ratio={boundary_ratio}")

        self.pipe = None
        self.model_map: Dict[str, Dict] = {}
        self.transformer_names: List[str] = []
        self.optimizer = None
        self.lr_scheduler = None

    # -------- Pipeline --------

    def setup_pipeline(self):
        print0("[INFO] Loading WAN 2.2 I2V pipeline...")
        vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id, vae=vae, torch_dtype=torch.float32, boundary_ratio=self.boundary_ratio
        )

        # Freeze & cast encoders/vae
        cast_model_to_dtype(self.pipe.vae, self.bf16)
        self.pipe.vae.to(device=self.device, dtype=self.bf16).requires_grad_(False)

        if getattr(self.pipe, "text_encoder", None) is not None:
            cast_model_to_dtype(self.pipe.text_encoder, self.bf16)
            self.pipe.text_encoder.to(device=self.device, dtype=self.bf16).requires_grad_(False)

        if getattr(self.pipe, "image_encoder", None) is not None:
            cast_model_to_dtype(self.pipe.image_encoder, self.bf16)
            self.pipe.image_encoder.to(device=self.device, dtype=self.bf16).requires_grad_(False)

        print0("[INFO] Pipeline loaded.")

    def _num_train_timesteps(self) -> int:
        sch = self.pipe.scheduler
        if hasattr(sch, "num_train_timesteps") and isinstance(sch.num_train_timesteps, int):
            return sch.num_train_timesteps
        if hasattr(sch, "config") and hasattr(sch.config, "num_train_timesteps"):
            return int(sch.config.num_train_timesteps)
        return 1000

    def get_boundary_timestep(self, num_train_timesteps: Optional[int] = None) -> int:
        if self.boundary_ratio is None:
            return 0
        if num_train_timesteps is None:
            num_train_timesteps = self._num_train_timesteps()
        return max(0, int(self.boundary_ratio * num_train_timesteps))

    # -------- DDP helpers for LoRA params --------

    def _broadcast_lora_params(self):
        if self.world_size == 1:
            return
        print0("[INFO] Broadcasting LoRA parameters...")
        dist.barrier()
        for name in self.transformer_names:
            for p in self.model_map[name]["lora_params"]:
                dist.broadcast(p.data, src=0)
        dist.barrier()

    def _allreduce_lora_grads(self):
        if self.world_size == 1:
            return
        for name in self.transformer_names:
            for p in self.model_map[name]["lora_params"]:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(self.world_size)

    # -------- Hybrid parallelism --------

    def setup_hybrid_parallelism(self):
        print0("[INFO] Setting up FSDP + LoRA hybrid...")
        self.transformer_names = []
        for name in ["transformer", "transformer_2"]:
            if getattr(self.pipe, name, None) is not None:
                self.transformer_names.append(name)
        if not self.transformer_names:
            raise RuntimeError("No transformers found in pipeline")
        print0(f"[INFO] Found transformers: {self.transformer_names}")

        fsdp_mixed_precision = MixedPrecision(param_dtype=self.bf16, reduce_dtype=self.bf16, buffer_dtype=self.bf16)

        self.model_map = {}
        for name in self.transformer_names:
            base_transformer = getattr(self.pipe, name)

            # Cast & move the base transformer (non-LoRA) first
            cast_model_to_dtype(base_transformer, self.bf16)
            base_transformer.to(self.device)

            # Install WAN LoRA processors (zero-arg)
            _install_wan_lora_processors(base_transformer)

            # Collect LoRA params by introspection and move to device/dtype
            lora_params = _collect_wan_lora_params(base_transformer)
            if len(lora_params) == 0:
                raise RuntimeError(f"No LoRA parameters found for {name}")
            _move_params(lora_params, self.device, self.bf16)
            print0(f"[INFO] {name}: {len(lora_params)} LoRA trainable tensors")

            # FSDP wrap ONLY the base transformer
            fsdp_wrapped = FSDP(
                base_transformer,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=fsdp_mixed_precision,
                auto_wrap_policy=None,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=self.local_rank,
                sync_module_states=True,
                use_orig_params=False,
                ignored_modules=[],  # processors aren't modules; nothing to ignore
            )
            fsdp_wrapped.train()

            self.model_map[name] = {
                "fsdp": fsdp_wrapped,
                "lora_params": lora_params,
                "base_transformer": base_transformer,  # keep ref for save/load swap
            }
            setattr(self.pipe, name, fsdp_wrapped)

        print0("[INFO] Hybrid setup complete.")

    # -------- Optimizer / Scheduler --------

    def setup_optimizer_and_scheduler(self):
        all_lora_params = []
        for name in self.transformer_names:
            all_lora_params.extend(self.model_map[name]["lora_params"])

        if not all_lora_params:
            raise RuntimeError("No trainable LoRA parameters found!")

        self.optimizer = torch.optim.AdamW(
            all_lora_params, lr=self.learning_rate, weight_decay=0.01, betas=(0.9, 0.999)
        )
        # Temp scheduler; reset after dataloader is built
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)

    # -------- Data --------

    def create_dataloader(self, meta_folder: str, batch_size: int = 1):
        dataset = MetaFilesDataset(meta_folder=meta_folder, device="cpu")
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=dist.get_rank(), shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=2, pin_memory=True
        )
        return dataloader, sampler

    # -------- I2V conditioning --------

    def prepare_i2v_conditioning(
        self, video_latents: torch.Tensor, timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(video_latents)
        noisy_video_latents = self.pipe.scheduler.add_noise(video_latents, noise, timesteps)

        condition_mask = torch.zeros_like(video_latents, dtype=self.bf16)
        condition_mask[:, :, 0] = 1.0  # keep first frame clean

        conditioned_latents = condition_mask * video_latents + (1 - condition_mask) * noisy_video_latents
        return conditioned_latents, noise, condition_mask

    # -------- Train step --------

    def training_step(self, batch: Dict) -> torch.Tensor:
        text_embeddings = batch["text_embeddings"].to(self.device, dtype=self.bf16)
        video_latents = batch["video_latents"].to(self.device, dtype=torch.float32)

        if video_latents.ndim == 6:
            video_latents = video_latents.squeeze(1)
        if text_embeddings.ndim == 4:
            text_embeddings = text_embeddings.squeeze(1)

        num_train_timesteps = self._num_train_timesteps()
        timesteps = torch.randint(0, num_train_timesteps, (video_latents.shape[0],), device=self.device)

        with torch.no_grad():
            conditioned_latents, noise, condition_mask = self.prepare_i2v_conditioning(video_latents, timesteps)

        conditioned_latents = conditioned_latents.to(self.bf16)
        noise = noise.to(self.bf16)

        boundary_timestep = self.get_boundary_timestep(num_train_timesteps)
        use_t1 = timesteps >= boundary_timestep
        use_t2 = ~use_t1

        total_loss = None
        num_samples = 0

        with torch.autocast(device_type="cuda", dtype=self.bf16):
            if use_t1.any():
                idx = use_t1.nonzero(as_tuple=True)[0]
                model = self.model_map["transformer"]["fsdp"]
                out = model(
                    hidden_states=conditioned_latents[idx],
                    timestep=timesteps[idx],
                    encoder_hidden_states=text_embeddings[idx],
                    return_dict=False,
                )
                noise_pred = out[0] if isinstance(out, tuple) else out
                non_cond = 1 - condition_mask[idx]
                loss = torch.nn.functional.mse_loss(noise_pred * non_cond, noise[idx] * non_cond)
                total_loss = loss * len(idx) if total_loss is None else total_loss + loss * len(idx)
                num_samples += len(idx)

            if use_t2.any() and "transformer_2" in self.model_map:
                idx = use_t2.nonzero(as_tuple=True)[0]
                model = self.model_map["transformer_2"]["fsdp"]
                out = model(
                    hidden_states=conditioned_latents[idx],
                    timestep=timesteps[idx],
                    encoder_hidden_states=text_embeddings[idx],
                    return_dict=False,
                )
                noise_pred = out[0] if isinstance(out, tuple) else out
                non_cond = 1 - condition_mask[idx]
                loss = torch.nn.functional.mse_loss(noise_pred * non_cond, noise[idx] * non_cond)
                total_loss = loss * len(idx) if total_loss is None else total_loss + loss * len(idx)
                num_samples += len(idx)

        if total_loss is None:
            raise ValueError("No samples processed in training step")
        final_loss = total_loss / num_samples

        if torch.isnan(final_loss) or torch.isinf(final_loss):
            raise ValueError(f"Invalid loss: {final_loss.item()}")

        return final_loss

    # -------- Save / Load (swap out FSDP for lora save/load) --------

    def _swap_fsdp_for_io(self, use_base: bool):
        """
        If use_base=True, put base transformers on the pipeline for IO.
        Else, restore FSDP-wrapped ones.
        """
        for name in self.transformer_names:
            if use_base:
                setattr(self.pipe, name, self.model_map[name]["base_transformer"])
            else:
                setattr(self.pipe, name, self.model_map[name]["fsdp"])

    def save_checkpoint(self, output_dir: str, step: int):
        if not is_main_process():
            return
        ckpt = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt, exist_ok=True)
        try:
            self._swap_fsdp_for_io(use_base=True)
            self.pipe.save_lora_weights(ckpt)
            print0(f"[INFO] Saved LoRA weights to {ckpt}")
        except Exception as e:
            print0(f"[ERROR] Failed to save LoRA weights: {e}")
        finally:
            self._swap_fsdp_for_io(use_base=False)

        torch.save(
            {
                "step": step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
                "boundary_ratio": self.boundary_ratio,
                "transformer_names": self.transformer_names,
                "model_id": self.model_id,
            },
            os.path.join(ckpt, "training_state.pt"),
        )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if not os.path.exists(state_path):
            print0(f"[WARNING] No training_state.pt at {checkpoint_path}")
            return 0

        state = torch.load(state_path, map_location="cpu")
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(state["scheduler_state_dict"])

        try:
            self._swap_fsdp_for_io(use_base=True)
            self.pipe.load_lora_weights(checkpoint_path)
            print0(f"[INFO] Loaded LoRA weights from {checkpoint_path}")
        except Exception as e:
            print0(f"[ERROR] Failed to load LoRA weights: {e}")
        finally:
            self._swap_fsdp_for_io(use_base=False)

        if self.world_size > 1:
            self._broadcast_lora_params()

        return int(state.get("step", 0))

    # -------- Validation --------

    @torch.no_grad()
    def validate(self, val_dataloader) -> float:
        for name in self.transformer_names:
            self.model_map[name]["fsdp"].eval()

        total_loss = 0.0
        n = 0
        for batch in tqdm(val_dataloader, desc="Validation", disable=not is_main_process()):
            loss = self.training_step(batch)
            total_loss += loss.item()
            n += 1

        for name in self.transformer_names:
            self.model_map[name]["fsdp"].train()

        return total_loss / max(n, 1)

    # -------- Train loop --------

    def train(
        self,
        meta_folder: str,
        num_epochs: int = 10,
        batch_size: int = 1,
        save_every: int = 1000,
        log_every: int = 100,
        validate_every: int = 500,
        output_dir: str = "./wan_i2v_outputs",
        val_meta_folder: Optional[str] = None,
        resume_checkpoint: Optional[str] = None,
    ):
        print0(f"[INFO] Starting LoRA training for {num_epochs} epochs")

        # Setup
        self.setup_pipeline()
        self.setup_hybrid_parallelism()
        self._broadcast_lora_params()
        self.setup_optimizer_and_scheduler()

        # Data
        dataloader, sampler = self.create_dataloader(meta_folder, batch_size)
        val_dataloader = None
        if val_meta_folder:
            val_dataloader, _ = self.create_dataloader(val_meta_folder, batch_size)

        steps_per_epoch = len(dataloader)
        total_steps = num_epochs * steps_per_epoch
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)
        print0(f"[INFO] Scheduler set for {total_steps} steps")

        # Resume
        global_step = 0
        start_epoch = 0
        if resume_checkpoint:
            global_step = self.load_checkpoint(resume_checkpoint)
            start_epoch = global_step // max(steps_per_epoch, 1)

        # Logging
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)
            wandb.init(
                project="wan-i2v-diffusers-lora",
                config={
                    "model_id": self.model_id,
                    "learning_rate": self.learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "total_batch_size": batch_size * self.world_size,
                    "boundary_ratio": self.boundary_ratio,
                    "approach": "diffusers_native_wan_lora_fsdp_hybrid",
                },
                resume=resume_checkpoint is not None,
            )
        if dist.is_initialized():
            dist.barrier()

        # Train
        for epoch in range(start_epoch, num_epochs):
            sampler.set_epoch(epoch)
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=not is_main_process())

            epoch_loss = 0.0
            num_steps = 0

            for _, batch in enumerate(pbar):
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss = self.training_step(batch)
                    loss.backward()

                    # sync LoRA grads
                    self._allreduce_lora_grads()

                    # clip
                    grads = [
                        p
                        for name in self.transformer_names
                        for p in self.model_map[name]["lora_params"]
                        if p.grad is not None
                    ]
                    grad_norm = 0.0
                    if grads:
                        grad_norm = torch.nn.utils.clip_grad_norm_(grads, max_norm=1.0)
                        grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()

                    epoch_loss += loss.item()
                    num_steps += 1
                    global_step += 1

                    if is_main_process() and global_step % log_every == 0:
                        avg_loss = epoch_loss / num_steps
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/avg_loss": avg_loss,
                                "train/lr": self.optimizer.param_groups[0]["lr"],
                                "train/grad_norm": grad_norm,
                                "train/epoch": epoch,
                                "train/step": global_step,
                            },
                            step=global_step,
                        )
                        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", gn=f"{grad_norm:.2f}")

                    if validate_every > 0 and (global_step % validate_every == 0) and val_dataloader:
                        val_loss = self.validate(val_dataloader)
                        if is_main_process():
                            print0(f"[INFO] step {global_step}: val_loss={val_loss:.6f}")
                            wandb.log({"val/loss": val_loss}, step=global_step)

                    if global_step % save_every == 0 and is_main_process():
                        self.save_checkpoint(output_dir, global_step)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print0("[WARN] OOM; clearing cache & continuing.")
                        self.optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        continue
                    raise

            if is_main_process():
                wandb.log({"epoch/avg_loss": epoch_loss / max(num_steps, 1), "epoch/num": epoch + 1}, step=global_step)

        # Final save
        if is_main_process():
            self.save_checkpoint(output_dir, global_step)
            final_dir = os.path.join(output_dir, "final")
            os.makedirs(final_dir, exist_ok=True)
            try:
                self._swap_fsdp_for_io(use_base=True)
                self.pipe.save_lora_weights(final_dir)
                print0(f"[INFO] Saved final LoRA weights to {final_dir}")
            finally:
                self._swap_fsdp_for_io(use_base=False)
            wandb.finish()

        print0("[INFO] Training complete.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="WAN 2.2 I2V Diffusers-native LoRA Training")

    # Model
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--lora_rank", type=int, default=16)  # kept for CLI parity; WAN ignores
    parser.add_argument("--lora_alpha", type=int, default=32)  # kept for CLI parity; WAN ignores
    parser.add_argument("--boundary_ratio", type=float, default=0.5)

    # Train
    parser.add_argument("--meta_folder", type=str, required=True)
    parser.add_argument("--val_meta_folder", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # Video (kept for completeness; your dataloader should honor these)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)

    # IO
    parser.add_argument("--output_dir", type=str, default="./wan_i2v_outputs")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--validate_every", type=int, default=500)
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    trainer = WanI2VDiffusersLoRATrainer(
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        boundary_ratio=args.boundary_ratio,
        debug=args.debug,
    )

    trainer.train(
        meta_folder=args.meta_folder,
        val_meta_folder=args.val_meta_folder,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_every=args.save_every,
        log_every=args.log_every,
        validate_every=args.validate_every,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
    )


if __name__ == "__main__":
    main()
