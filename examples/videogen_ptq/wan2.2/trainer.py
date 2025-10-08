import os, torch, wandb, torch.distributed as dist
from typing import Optional, Dict
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

from .dist_utils import setup_distributed, is_main_process, print0, cast_model_to_dtype
from .fsdp_hybrid import setup_hybrid_for_pipe
from .lora_utils import broadcast_params, allreduce_grads
from .data_utils import create_dataloader
from .training_step import step_dual_transformer
from .checkpoint_io import save_lora_checkpoint, load_lora_checkpoint

class WanI2VLoRATrainer:
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 1e-4,
        boundary_ratio: Optional[float] = 0.5,
        debug: bool = False,
    ):
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.boundary_ratio = boundary_ratio
        self.bf16 = torch.bfloat16
        self.debug = debug

        self.local_rank = setup_distributed()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device("cuda", self.local_rank)

        print0("[INFO] WAN 2.2 I2V LoRA trainer init")
        print0(f"[INFO] world_size={self.world_size} local_rank={self.local_rank}")
        print0(f"[INFO] LoRA rank={lora_rank} alpha={lora_alpha} (WAN-native)")

        self.pipe = None
        self.model_map: Dict[str, Dict] = {}
        self.transformer_names = []
        self.optimizer = None
        self.lr_scheduler = None

    def setup_pipeline(self):
        print0("[INFO] Loading pipeline ...")
        vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanImageToVideoPipeline.from_pretrained(self.model_id, vae=vae, torch_dtype=torch.float32, boundary_ratio=self.boundary_ratio)

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
        self.model_map, self.transformer_names = setup_hybrid_for_pipe(
            self.pipe, device=self.device, bf16=self.bf16,
            local_rank=self.local_rank, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha
        )
        # Broadcast LoRA params so all ranks start the same
        all_params = []
        for n in self.transformer_names:
            all_params += self.model_map[n]["lora_params"]
        broadcast_params(all_params, world_size=self.world_size, src=0)

    def setup_optim(self):
        all_params = []
        for n in self.transformer_names:
            all_params += self.model_map[n]["lora_params"]
        if not all_params:
            raise RuntimeError("No trainable LoRA parameters found!")
        self.optimizer = torch.optim.AdamW(all_params, lr=self.learning_rate, weight_decay=0.01, betas=(0.9, 0.999))
        # temp scheduler; will reconfigure after dataloader length is known
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
            global_step = load_lora_checkpoint(self.pipe, self.model_map, self.transformer_names, self.optimizer, self.lr_scheduler, resume_checkpoint)
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
                    approach="wan_fsdp_lora_hybrid",
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
                iterable = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

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
                )
                loss.backward()

                # manual grad sync for LoRA
                all_params = []
                for n in self.transformer_names:
                    all_params += [p for p in self.model_map[n]["lora_params"] if p.grad is not None]
                allreduce_grads(all_params, self.world_size)

                grad_norm = 0.0
                if all_params:
                    g = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    grad_norm = float(g) if torch.is_tensor(g) else g

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
                            avg=f"{(epoch_loss/num_steps):.4f}",
                            lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                            gn=f"{grad_norm:.2f}",
                        )

                if save_every and (global_step % save_every == 0):
                    save_lora_checkpoint(self.pipe, self.model_map, self.transformer_names, self.optimizer, self.lr_scheduler, output_dir, global_step)

            print0(f"[INFO] Epoch {epoch+1} done. avg_loss={epoch_loss/max(num_steps,1):.6f}")
            if is_main_process():
                wandb.log({"epoch/avg_loss": epoch_loss / max(num_steps, 1), "epoch/num": epoch + 1}, step=global_step)

        if is_main_process():
            save_lora_checkpoint(self.pipe, self.model_map, self.transformer_names, self.optimizer, self.lr_scheduler, output_dir, global_step)
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
