from __future__ import annotations

from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.recipes.llm.train_ft import build_distributed, build_wandb
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages

# trainer_t2v.py - WAN 2.1 T2V Trainer with Flow Matching (localized)
import os
import logging
from typing import Optional

import torch
import torch.distributed as dist
import wandb
from nemo_automodel.recipes.diffusion.wan21.data_utils import create_dataloader
from diffusers import WanPipeline
from nemo_automodel.recipes.diffusion.wan21.dist_utils import is_main_process, setup_distributed
from nemo_automodel.recipes.diffusion.wan21.fsdp2_utils_t2v import (
    get_fsdp_all_parameters,
    setup_fsdp_for_t2v_pipe,
    test_fsdp_forward_pass,
    verify_fsdp_setup,
)
from nemo_automodel.recipes.diffusion.wan21.training_step_t2v import step_fsdp_transformer_t2v
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.api import ShardedOptimStateDictConfig, ShardedStateDictConfig
import torch.distributed.checkpoint as dcp
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState


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

        logging.info("[INFO] WAN 2.1 T2V Trainer with Flow Matching")
        logging.info(f"[INFO] Total GPUs: {self.world_size}, GPUs per node: {self.local_world_size}, Num nodes: {self.num_nodes}")
        logging.info(f"[INFO] Node rank: {self.node_rank}, Local rank: {self.local_rank}")
        logging.info(f"[INFO] Learning rate: {learning_rate}")
        logging.info(f"[INFO] CPU offload: {'ENABLED' if cpu_offload else 'DISABLED'}")
        logging.info(f"[INFO] Flow matching: {'ENABLED' if use_sigma_noise else 'DISABLED'}")
        if use_sigma_noise:
            logging.info(f"[INFO]   - Timestep sampling: {timestep_sampling}")
            logging.info(f"[INFO]   - Flow shift: {flow_shift}")
            logging.info(f"[INFO]   - Mix uniform ratio: {mix_uniform_ratio}")

        self.pipe = None
        self.model_map = {}
        self.optimizer = None
        self.lr_scheduler = None

    def setup_pipeline(self):
        logging.info("[INFO] Loading WAN 2.1 T2V pipeline (transformer only)...")

        # Load pipeline without VAE or text encoder
        self.pipe = WanPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float32,
            vae=None,
            text_encoder=None,
        )

        # Explicitly delete VAE and text encoder if they were loaded
        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            logging.info("[INFO] Removing VAE from pipeline...")
            del self.pipe.vae
            self.pipe.vae = None

        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            logging.info("[INFO] Removing text encoder from pipeline...")
            del self.pipe.text_encoder
            self.pipe.text_encoder = None

        # Clear any cached memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info("[INFO] Pipeline loaded (transformer + scheduler only)")

    def nemo_auto_diffusion(
        self,
        tp_size: int = 1,
        cp_size: int = 1,
        pp_size: int = 1,
        dp_size: Optional[int] = None,
        use_hf_tp_plan: bool = False,
    ):
        """Load pipeline via NeMoAutoDiffusionPipeline with a transformer-only parallel scheme,
        then set up optimizer and scheduler similar to setup_optim.

        This uses FSDP2Manager to parallelize the transformer component only. It also
        prepares a lightweight model_map for downstream utilities that expect it.
        """
        logging.info("[INFO] Building NeMoAutoDiffusionPipeline with transformer parallel scheme...")

        if not dist.is_initialized():
            logging.info("[WARN] torch.distributed not initialized; proceeding in single-process mode")

        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Infer DP size if not provided
        if dp_size is None:
            denom = max(1, tp_size * cp_size * pp_size)
            dp_size = max(1, world_size // denom)

        # Manager for per-component parallelization
        fsdp2_manager = FSDP2Manager(
            dp_size=dp_size,
            tp_size=tp_size,
            cp_size=cp_size,
            pp_size=pp_size,
            backend="nccl",
            world_size=world_size,
            use_hf_tp_plan=use_hf_tp_plan,
            activation_checkpointing=True,
            # offload_policy=CPUOffloadPolicy(True),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=self.bf16,
                reduce_dtype=self.bf16,
                output_dtype=self.bf16,
            )
        )

        parallel_scheme = {"transformer": fsdp2_manager}

        # Construct the pipeline and parallelize only the transformer
        self.pipe = NeMoAutoDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.bf16,
            device=self.device,
            parallel_scheme=parallel_scheme,
            load_for_training=True
        )

        # Prepare a minimal model_map interface
        transformer_module = getattr(self.pipe, "transformer", None)
        if transformer_module is None:
            raise RuntimeError("transformer not found in pipeline after parallelization")

        self.model_map = {
            "transformer": {
                "fsdp_transformer": transformer_module,
                "base_transformer": transformer_module,
            }
        }

        # Set up optimizer and a temporary scheduler (reconfigured later after dataloader)
        logging.info("[INFO] Setting up optimizer for NeMoAutoDiffusionPipeline (transformer)")
        trainable_params = [p for p in transformer_module.parameters() if p.requires_grad]
        verify_fsdp_setup(self.model_map)
        if not trainable_params:
            raise RuntimeError("No trainable parameters found in transformer module!")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        logging.info("[INFO] NeMoAutoDiffusion setup complete (pipeline + optimizer")

    def setup_fsdp(self):
        logging.info("[INFO] Setting up FSDP for full fine-tuning...")

        self.model_map = setup_fsdp_for_t2v_pipe(
            self.pipe,
            device=self.device,
            bf16=self.bf16,
            local_rank=self.local_rank,
            cpu_offload=self.cpu_offload,
        )

        # Verify setup
        verify_fsdp_setup(self.model_map)

        logging.info("[INFO] FSDP setup complete")

    def setup_optim(self):
        logging.info("[INFO] Setting up optimizer...")

        # Get ALL trainable parameters
        all_params = get_fsdp_all_parameters(self.model_map)

        if not all_params:
            raise RuntimeError("No trainable parameters found!")

        logging.info(f"[INFO] Optimizing {len(all_params)} parameters")

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
        logging.info("[INFO] Validating FSDP setup...")

        test_fsdp_forward_pass(self.model_map, self.device, self.bf16)

        # Check memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            logging.info(f"[INFO] GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

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
        logging.info("[INFO] Starting T2V training with Flow Matching")
        logging.info(f"[INFO] Batch size per node: {batch_size_per_node}")
        logging.info(f"[INFO] Total effective batch size: {batch_size_per_node * self.num_nodes}")

        # self.setup_pipeline()
        # self.setup_fsdp()
        # self.setup_optim()
        self.nemo_auto_diffusion()
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
        logging.info(f"[INFO] Scheduler configured for {total_steps} total steps")

        global_step = 0
        start_epoch = 0

        if resume_checkpoint:
            global_step = load_fsdp2_checkpoint(
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
                    logging.info(f"[ERROR] Training step failed at epoch {epoch}, step {step}: {e}")
                    video_shape = batch.get("video_latents", torch.tensor([])).shape
                    text_shape = batch.get("text_embeddings", torch.tensor([])).shape
                    logging.info(f"[DEBUG] Batch shapes - video: {video_shape}, text: {text_shape}")
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
                    logging.info(f"[INFO] Logging: {log_dict}")
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
                    save_fsdp2_checkpoint(
                        self.model_map,
                        self.optimizer,
                        self.lr_scheduler,
                        output_dir,
                        global_step,
                        save_consolidated=False,
                    )                  

        		# Epoch summary
            avg_loss = epoch_loss / max(num_steps, 1)
            logging.info(f"[INFO] Epoch {epoch + 1} complete. avg_loss={avg_loss:.6f}")

            if is_main_process():
                wandb.log({"epoch/avg_loss": avg_loss, "epoch/num": epoch + 1}, step=global_step)
            

        # Final checkpoint: all ranks must participate in DCP save (it uses barriers)
        logging.info("[INFO] Training complete, saving final checkpoint...")

        save_fsdp2_checkpoint(
            self.model_map,
            self.optimizer,
            self.lr_scheduler,
            output_dir,
            global_step,
            save_consolidated=False,
        )

        if is_main_process():
            logging.info(f"[INFO] Saved final checkpoint at step {global_step}")
            wandb.finish()

        logging.info("[INFO] Training complete!")


def save_fsdp2_checkpoint(
    model_map: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: str,
    step: int,
    *,
    save_consolidated: bool = False,
) -> None:
    """Save an FSDP2 checkpoint using PyTorch Distributed Checkpointing (DCP).

    This saves sharded model weights and optimizer state under
    f"{output_dir}/checkpoint-{step}". Optionally writes a consolidated
    full-state model for inference.
    """
    from torch.distributed.checkpoint import FileSystemWriter
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    fsdp_model = model_map["transformer"]["fsdp_transformer"]

    # Sharded model and optimizer via DCP
    model_state = {"model": ModelState(fsdp_model)}
    optim_state = {"optim": OptimizerState(fsdp_model, optimizer, scheduler)}

    model_path = os.path.join(ckpt_dir, "transformer_model")
    optim_path = os.path.join(ckpt_dir, "optimizer")

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(optim_path, exist_ok=True)

    dcp.save(model_state, storage_writer=FileSystemWriter(model_path))
    if dist.is_initialized():
        dist.barrier()
    dcp.save(optim_state, storage_writer=FileSystemWriter(optim_path))

    if dist.is_initialized():
        dist.barrier()

    # Optional consolidated full-state dict for inference on rank 0
    if save_consolidated:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        consolidated_saved = False
        try:
            # Only use FSDP FULL_STATE_DICT path if the module is actually FSDP
            if isinstance(fsdp_model, FSDP):
                with FSDP.state_dict_type(
                    fsdp_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                ):
                    full_state = fsdp_model.state_dict()

                if (not dist.is_initialized()) or dist.get_rank() == 0:
                    consolidated_path = os.path.join(ckpt_dir, "consolidated_model.bin")
                    torch.save(full_state, consolidated_path)
                    consolidated_saved = True
        except Exception as e:
            logging.info(f"[WARN] Skipping consolidated save due to error: {e}")

        if not consolidated_saved:
            logging.info("[INFO] Consolidated save skipped (module not FSDP or unsupported)")

    # Save training state on rank 0
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        training_state = {
            "step": step,
            "scheduler": scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))


def load_fsdp2_checkpoint(
    model_map: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str,
) -> int:
    """Load an FSDP2 checkpoint saved by save_fsdp2_checkpoint.

    Returns the restored global step, or 0 if not found.
    """
    from torch.distributed.checkpoint import FileSystemReader

    if not os.path.exists(ckpt_path):
        logging.info(f"[WARNING] Checkpoint {ckpt_path} not found")
        return 0

    fsdp_model = model_map["transformer"]["fsdp_transformer"]

    # Load model
    model_dir = os.path.join(ckpt_path, "transformer_model")
    if os.path.exists(model_dir):
        model_state = {"model": ModelState(fsdp_model)}
        dcp.load(model_state, storage_reader=FileSystemReader(model_dir))

    if dist.is_initialized():
        dist.barrier()

    # Load optimizer + scheduler
    optim_dir = os.path.join(ckpt_path, "optimizer")
    if os.path.exists(optim_dir):
        optim_state = {"optim": OptimizerState(fsdp_model, optimizer, scheduler)}
        dcp.load(optim_state, storage_reader=FileSystemReader(optim_dir))

    # Load training state and return step
    train_state_path = os.path.join(ckpt_path, "training_state.pt")
    if os.path.exists(train_state_path):
        state = torch.load(train_state_path, map_location="cpu")
        step = int(state.get("step", 0))
        if "scheduler" in state:
            try:
                scheduler.load_state_dict(state["scheduler"]) 
            except Exception:
                pass
        return step

    return 0


class TrainWan21DiffusionRecipe(BaseRecipe):
    """Config-driven wrapper around WAN 2.1 T2V training."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            build_wandb(self.cfg)
        setup_logging()

        fm = self.cfg.get("flow_matching", {})
        fsdp = self.cfg.get("fsdp", {})

        self.trainer = WanT2VTrainerFSDP(
            model_id=self.cfg.get(
                "model.pretrained_model_name_or_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            ),
            learning_rate=self.cfg.get("optim.learning_rate", 5e-6),
            cpu_offload=fsdp.get("cpu_offload", True),
            use_sigma_noise=fm.get("use_sigma_noise", True),
            timestep_sampling=fm.get("timestep_sampling", "uniform"),
            logit_mean=fm.get("logit_mean", 0.0),
            logit_std=fm.get("logit_std", 1.0),
            flow_shift=fm.get("flow_shift", 3.0),
            mix_uniform_ratio=fm.get("mix_uniform_ratio", 0.1),
        )

    def run_train_validation_loop(self):
        log = self.cfg.get("logging", {})
        ckpt = self.cfg.get("checkpoint", {})
        batch = self.cfg.get("batch", {})
        train = self.cfg.get("training", {})

        self.trainer.train(
            meta_folder=self.cfg.get("data.meta_folder"),
            num_epochs=train.get("num_epochs", 1),
            batch_size_per_node=batch.get("batch_size_per_node", 1),
            save_every=log.get("save_every", 500),
            log_every=log.get("log_every", 5),
            output_dir=ckpt.get("output_dir", "./wan_t2v_flow_outputs"),
            resume_checkpoint=ckpt.get("resume", None),
        )


