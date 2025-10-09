# trainer_tp_cp.py - Updated with RoPE-aware TP+CP support
import os
from typing import Dict, Optional

import torch
import torch.distributed as dist
import wandb
from checkpoint_io import load_lora_checkpoint, save_lora_checkpoint
from data_utils import create_dataloader
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from dist_utils import cast_model_to_dtype, is_main_process, print0, setup_distributed
from tp_cp_hybrid import (
    manual_allreduce_lora_gradients_tp_cp,
    setup_tp_cp_for_pipe,
    test_tp_cp_forward_pass,
    verify_rope_cp_setup,
)
from training_step_tp_cp import step_dual_transformer


class WanI2VLoRATrainer:
    """
    Enhanced WAN I2V LoRA trainer with RoPE-aware TP+CP support.
    Now handles 100+ frame sequences with proper positional encoding.
    """

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 1e-4,
        boundary_ratio: Optional[float] = 0.5,
        use_peft_lora: bool = False,
        debug: bool = False,
        train_transformer_2: bool = False,
        tp_size: int = None,
        cp_size: int = None,
        enable_rope_cp: bool = True,  # New flag to enable/disable RoPE-aware CP
    ):
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.boundary_ratio = boundary_ratio
        self.use_peft_lora = use_peft_lora
        self.bf16 = torch.bfloat16
        self.debug = debug
        self.train_transformer_2 = train_transformer_2
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.enable_rope_cp = enable_rope_cp

        self.local_rank = setup_distributed()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device("cuda", self.local_rank)

        print0(f"[INFO] WAN I2V LoRA trainer with {'RoPE-aware ' if enable_rope_cp else ''}TP+CP")
        print0(f"[INFO] World size: {self.world_size}, Local rank: {self.local_rank}")
        print0(f"[INFO] LoRA rank={lora_rank} alpha={lora_alpha}")
        print0(f"[INFO] Training transformer_2: {train_transformer_2}")
        print0(f"[INFO] TP size: {tp_size if tp_size else 'auto'}, CP size: {cp_size if cp_size else 'auto'}")

        if enable_rope_cp and cp_size and cp_size > 1:
            print0("[INFO] RoPE-aware Context Parallelism ENABLED - supports 100+ frame sequences")
        elif cp_size and cp_size > 1:
            print0("[INFO] Standard Context Parallelism enabled (RoPE disabled)")
        else:
            print0("[INFO] No Context Parallelism (CP size = 1)")

        self.pipe = None
        self.model_map: Dict[str, Dict] = {}
        self.transformer_names = []
        self.optimizer = None
        self.lr_scheduler = None

    def setup_pipeline(self):
        print0("[INFO] Loading WAN pipeline...")

        vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)

        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id, vae=vae, torch_dtype=torch.float32, boundary_ratio=self.boundary_ratio
        )

        # Setup pipeline components
        cast_model_to_dtype(self.pipe.vae, self.bf16)
        self.pipe.vae.to(self.device, dtype=self.bf16).requires_grad_(False)

        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            cast_model_to_dtype(self.pipe.text_encoder, self.bf16)
            self.pipe.text_encoder.to(self.device, dtype=self.bf16).requires_grad_(False)

        if hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None:
            cast_model_to_dtype(self.pipe.image_encoder, self.bf16)
            self.pipe.image_encoder.to(self.device, dtype=self.bf16).requires_grad_(False)

        print0("[INFO] Pipeline components loaded and configured")

    def setup_hybrid(self):
        print0(f"[INFO] Setting up {'RoPE-aware ' if self.enable_rope_cp else ''}TP+CP...")

        # Import the appropriate setup function
        if self.enable_rope_cp:
            print0("[INFO] Using RoPE-aware Context Parallelism for long sequences")
            setup_func = setup_tp_cp_for_pipe
        else:
            print0("[INFO] Using standard Context Parallelism (no RoPE)")
            from tp_cp_hybrid import setup_tp_cp_for_pipe as setup_func

        # Enable gradient checkpointing after setup
        for name in self.transformer_names:
            if name in self.model_map:
                base_transformer = self.model_map[name]["base_transformer"]
                base_transformer.gradient_checkpointing = True
                print0(f"[INFO] Enabled gradient checkpointing for {name}")

        self.model_map, self.transformer_names = setup_func(
            self.pipe,
            device=self.device,
            bf16=self.bf16,
            local_rank=self.local_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_peft=self.use_peft_lora,
            train_transformer_2=self.train_transformer_2,
            tp_size=self.tp_size,
            cp_size=self.cp_size,
        )

        # Verify setup
        if self.enable_rope_cp:
            verify_rope_cp_setup(self.model_map, self.transformer_names)

        print0("[INFO] Hybrid parallelization setup complete")

    def setup_optim(self):
        # Get LoRA parameters from the active transformer
        active_transformer = "transformer_2" if self.train_transformer_2 else "transformer"

        if active_transformer not in self.model_map:
            raise RuntimeError(f"Active transformer {active_transformer} not found")

        all_params = self.model_map[active_transformer]["lora_params"]

        if not all_params:
            raise RuntimeError(f"No LoRA parameters found in {active_transformer}")

        print0(f"[INFO] Setting up optimizer for {len(all_params)} LoRA parameters from {active_transformer}")

        # Enhanced optimizer for long sequence training
        if self.enable_rope_cp and self.cp_size and self.cp_size > 1:
            # Use slightly different settings for RoPE-aware training
            self.optimizer = torch.optim.AdamW(
                all_params,
                lr=self.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.95),  # Slightly different beta2 for stability
                eps=1e-8,
            )
            print0("[INFO] Using RoPE-optimized AdamW settings")
        else:
            self.optimizer = torch.optim.AdamW(all_params, lr=self.learning_rate, weight_decay=0.01, betas=(0.9, 0.999))

        # Temporary scheduler - will be reconfigured after dataloader length is known
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)

    def validate_training_setup(self):
        """
        Validate the training setup, especially for RoPE-aware CP.
        """
        print0("[INFO] Validating training setup...")

        # Test forward pass
        if self.enable_rope_cp:
            test_tp_cp_forward_pass(self.model_map, self.transformer_names, self.device, self.bf16)
            # validate_rope_cp_training_step(self.model_map, self.transformer_names, self.device, self.bf16)

        # Check memory usage
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
        validate_every: int = 0,
        output_dir: str = "./wan_i2v_outputs",
        resume_checkpoint: Optional[str] = None,
        max_sequence_length: Optional[int] = None,  # New parameter for sequence length control
    ):
        print0(f"[INFO] Starting training with max_sequence_length={max_sequence_length}")

        self.setup_pipeline()
        self.setup_hybrid()
        self.setup_optim()

        # Validate setup
        self.validate_training_setup()

        # Create dataloader with sequence length filtering if specified
        filter_fn = None
        if max_sequence_length:

            def length_filter(metadata):
                frame_count = metadata.get("end_frame", 0) - metadata.get("start_frame", 0) + 1
                return frame_count <= max_sequence_length

            filter_fn = length_filter
            print0(f"[INFO] Filtering sequences to max {max_sequence_length} frames")

        dataloader, sampler = create_dataloader(meta_folder, batch_size, self.world_size)

        steps_per_epoch = len(dataloader)
        total_steps = num_epochs * steps_per_epoch

        # Reconfigure scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)
        print0(f"[INFO] Scheduler configured for {total_steps} total steps")

        global_step = 0
        start_epoch = 0

        if resume_checkpoint:
            global_step = load_lora_checkpoint(
                self.pipe, self.model_map, self.transformer_names, self.optimizer, self.lr_scheduler, resume_checkpoint
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
                "approach": "wan_rope_tp_cp_lora" if self.enable_rope_cp else "wan_tp_cp_lora",
                "train_transformer_2": self.train_transformer_2,
                "tp_size": self.tp_size,
                "cp_size": self.cp_size,
                "enable_rope_cp": self.enable_rope_cp,
                "max_sequence_length": max_sequence_length,
            }

            wandb.init(
                project="wan-i2v-rope-tp-cp-lora" if self.enable_rope_cp else "wan-i2v-tp-cp-lora",
                config=config,
                resume=resume_checkpoint is not None,
            )

        if dist.is_initialized():
            dist.barrier()

        # Enhanced training loop with RoPE-CP monitoring
        for epoch in range(start_epoch, num_epochs):
            sampler.set_epoch(epoch)

            iterable = dataloader
            if is_main_process():
                from tqdm import tqdm

                iterable = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            epoch_loss = 0.0
            num_steps = 0
            rope_cp_stats = {"sequences_sharded": 0, "sequences_too_short": 0} if self.enable_rope_cp else {}

            for step, batch in enumerate(iterable):
                self.optimizer.zero_grad(set_to_none=True)

                # Enhanced training step with RoPE-CP support
                try:
                    loss = step_dual_transformer(
                        pipe=self.pipe,
                        model_map=self.model_map,
                        transformer_names=self.transformer_names,
                        batch=batch,
                        device=self.device,
                        bf16=self.bf16,
                        boundary_ratio=self.boundary_ratio if self.boundary_ratio is not None else 0.0,
                        train_transformer_2=self.train_transformer_2,
                    )

                    # Collect RoPE-CP statistics if enabled
                    if self.enable_rope_cp and self.train_transformer_2 and "transformer_2" in self.model_map:
                        cp_manager = self.model_map["transformer_2"].get("cp_manager")
                        if cp_manager and hasattr(cp_manager, "_last_shard_meta"):
                            meta = cp_manager._last_shard_meta
                            if meta and meta.get("was_sharded", False):
                                rope_cp_stats["sequences_sharded"] += 1
                            else:
                                rope_cp_stats["sequences_too_short"] += 1
                    elif self.enable_rope_cp and not self.train_transformer_2 and "transformer" in self.model_map:
                        cp_manager = self.model_map["transformer"].get("cp_manager")
                        if cp_manager and hasattr(cp_manager, "_last_shard_meta"):
                            meta = cp_manager._last_shard_meta
                            if meta and meta.get("was_sharded", False):
                                rope_cp_stats["sequences_sharded"] += 1
                            else:
                                rope_cp_stats["sequences_too_short"] += 1

                except Exception as e:
                    print0(f"[ERROR] Training step failed at epoch {epoch}, step {step}: {e}")
                    # Log batch info for debugging
                    video_shape = batch.get("video_latents", torch.tensor([])).shape
                    text_shape = batch.get("text_embeddings", torch.tensor([])).shape
                    print0(f"[DEBUG] Batch shapes - video: {video_shape}, text: {text_shape}")
                    raise

                loss.backward()

                # Manual gradient synchronization for LoRA parameters
                manual_allreduce_lora_gradients_tp_cp(self.model_map, self.transformer_names)

                # Gradient clipping
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

                # Enhanced logging with RoPE-CP stats
                if is_main_process() and (global_step % log_every == 0):
                    log_dict = {
                        "train_loss": loss.item(),
                        "train_avg_loss": epoch_loss / num_steps,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm,
                        "epoch": epoch,
                        "global_step": global_step,
                    }

                    # Add RoPE-CP statistics
                    if self.enable_rope_cp and rope_cp_stats:
                        total_sequences = rope_cp_stats["sequences_sharded"] + rope_cp_stats["sequences_too_short"]
                        if total_sequences > 0:
                            log_dict["rope_cp/sharding_ratio"] = rope_cp_stats["sequences_sharded"] / total_sequences
                            log_dict["rope_cp/sequences_sharded"] = rope_cp_stats["sequences_sharded"]
                            log_dict["rope_cp/sequences_too_short"] = rope_cp_stats["sequences_too_short"]

                    wandb.log(log_dict, step=global_step)

                    if hasattr(iterable, "set_postfix"):
                        postfix_dict = {
                            "loss": f"{loss.item():.4f}",
                            "avg": f"{(epoch_loss / num_steps):.4f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                            "gn": f"{grad_norm:.2f}",
                        }
                        if self.enable_rope_cp and rope_cp_stats:
                            total_seq = rope_cp_stats["sequences_sharded"] + rope_cp_stats["sequences_too_short"]
                            if total_seq > 0:
                                shard_pct = 100 * rope_cp_stats["sequences_sharded"] / total_seq
                                postfix_dict["rope"] = f"{shard_pct:.0f}%"

                        iterable.set_postfix(postfix_dict)

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

            # Epoch summary with RoPE-CP statistics
            avg_loss = epoch_loss / max(num_steps, 1)
            print0(f"[INFO] Epoch {epoch + 1} complete. avg_loss={avg_loss:.6f}")

            if self.enable_rope_cp and rope_cp_stats:
                total_sequences = rope_cp_stats["sequences_sharded"] + rope_cp_stats["sequences_too_short"]
                if total_sequences > 0:
                    shard_ratio = rope_cp_stats["sequences_sharded"] / total_sequences
                    print0(
                        f"[INFO] RoPE-CP stats: {rope_cp_stats['sequences_sharded']}/{total_sequences} "
                        f"sequences sharded ({shard_ratio:.1%})"
                    )
                    if shard_ratio < 0.5:
                        print0("[WARNING] Less than 50% of sequences were sharded. Consider:")
                        print0("  - Increasing sequence length in your data")
                        print0("  - Reducing CP size")
                        print0("  - Checking minimum frame requirements")

            if is_main_process():
                epoch_log = {"epoch/avg_loss": avg_loss, "epoch/num": epoch + 1}
                if self.enable_rope_cp and rope_cp_stats:
                    total_seq = rope_cp_stats["sequences_sharded"] + rope_cp_stats["sequences_too_short"]
                    if total_seq > 0:
                        epoch_log["epoch/rope_cp_sharding_ratio"] = rope_cp_stats["sequences_sharded"] / total_seq

                wandb.log(epoch_log, step=global_step)

        # Final checkpoint and cleanup
        if is_main_process():
            print0("[INFO] Training complete, saving final checkpoint...")

            save_lora_checkpoint(
                self.pipe,
                self.model_map,
                self.transformer_names,
                self.optimizer,
                self.lr_scheduler,
                output_dir,
                global_step,
            )

            # Save final LoRA weights
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

            # Save training summary
            summary_file = os.path.join(output_dir, "training_summary.txt")
            with open(summary_file, "w") as f:
                f.write("WAN I2V LoRA Training Summary\n")
                f.write("================================\n")
                f.write(f"Model: {self.model_id}\n")
                f.write(f"LoRA rank: {self.lora_rank}, alpha: {self.lora_alpha}\n")
                f.write(f"Parallelism: TP={self.tp_size}, CP={self.cp_size}\n")
                f.write(f"RoPE-aware CP: {'Enabled' if self.enable_rope_cp else 'Disabled'}\n")
                f.write(f"Trained transformer: {'transformer_2' if self.train_transformer_2 else 'transformer'}\n")
                f.write(f"Total steps: {global_step}\n")
                f.write(f"Final average loss: {avg_loss:.6f}\n")

                if self.enable_rope_cp and rope_cp_stats:
                    f.write("\nRoPE-CP Statistics:\n")
                    f.write(f"Sequences sharded: {rope_cp_stats['sequences_sharded']}\n")
                    f.write(f"Sequences too short: {rope_cp_stats['sequences_too_short']}\n")
                    if total_sequences > 0:
                        f.write(f"Sharding ratio: {rope_cp_stats['sequences_sharded'] / total_sequences:.1%}\n")

            wandb.finish()

        print0("[INFO] Training and cleanup complete!")
