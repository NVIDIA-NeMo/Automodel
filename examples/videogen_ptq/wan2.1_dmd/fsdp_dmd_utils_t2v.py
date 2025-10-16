# fsdp_dmd_utils_t2v.py - FSDP setup for DMD with 3 full models

import os
from typing import Dict

import torch
import torch.distributed as dist
from dist_utils import cast_model_to_dtype, is_main_process, print0
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedOptimStateDictConfig, ShardedStateDictConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def create_fsdp_mixed_precision_policy(bf16):
    """Create mixed precision policy for FSDP."""
    return MixedPrecision(
        param_dtype=bf16,
        reduce_dtype=bf16,
        buffer_dtype=bf16,
    )


def detect_transformer_block_types(transformer):
    """Detect transformer block types for activation checkpointing."""
    block_types = set()

    if hasattr(transformer, "blocks") and transformer.blocks is not None:
        if len(transformer.blocks) > 0:
            block_type = type(transformer.blocks[0])
            block_types.add(block_type)
            print0(f"[FSDP] Detected transformer block type: {block_type.__name__}")

    return block_types


def create_fsdp_wrap_policy(transformer):
    """Create FSDP wrapping policy for WAN 2.1 transformer."""
    wrap_module_types = set()

    if hasattr(transformer, "blocks") and transformer.blocks is not None:
        if len(transformer.blocks) > 0:
            block_type = type(transformer.blocks[0])
            wrap_module_types.add(block_type)
            print0(f"[FSDP] Will wrap transformer blocks of type: {block_type.__name__}")

    if not wrap_module_types:
        print0("[FSDP] WARNING: No transformer blocks found")
        return None

    return ModuleWrapPolicy(wrap_module_types)


def apply_fsdp_activation_checkpointing(fsdp_model, transformer_block_types):
    """Apply activation checkpointing to transformer blocks."""
    if not transformer_block_types:
        print0("[FSDP] No block types for activation checkpointing")
        return

    def check_fn(submodule):
        return type(submodule) in transformer_block_types

    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=lambda module: checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=check_fn,
    )

    print0("[FSDP] Applied activation checkpointing")


def setup_fsdp_for_dmd_t2v(
    pipe,
    device,
    bf16,
    local_rank: int,
    teacher_model_path: str = None,
    cpu_offload: bool = True,
):
    """
    Setup FSDP for DMD with 3 full models.

    Memory strategy:
    1. Generator: GPU, trainable, CPU offload
    2. Teacher (real_score): GPU/CPU, frozen, aggressive CPU offload
    3. Critic (fake_score): GPU, trainable, CPU offload

    Args:
        pipe: WAN pipeline
        device: Training device
        bf16: BFloat16 dtype
        local_rank: Local rank
        teacher_model_path: Path to teacher checkpoint (if different from base)
        cpu_offload: Enable CPU offloading for parameters

    Returns:
        model_map: Dictionary with generator, real_score, fake_score
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print0("[FSDP DMD] Setting up 3 models for DMD training")
    print0(f"  World size: {world_size}, Rank: {rank}")
    print0(f"  CPU offload: {'ENABLED' if cpu_offload else 'DISABLED'}")

    # Get base transformer from pipeline
    if not hasattr(pipe, "transformer") or pipe.transformer is None:
        raise RuntimeError("transformer not found in pipeline")

    base_transformer = pipe.transformer

    # Determine CPU offload config
    cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None

    # ============================================================================
    # CRITICAL: Create all 3 transformers BEFORE wrapping any in FSDP
    # This is because once wrapped, state_dict() returns FSDP shards, not regular tensors
    # ============================================================================

    print0("[FSDP DMD] Creating all transformer instances before FSDP wrapping...")

    # Get the original state dict BEFORE any FSDP wrapping
    original_state_dict = base_transformer.state_dict()

    # ============================================================================
    # 1. GENERATOR (Student) - Will use base_transformer directly
    # ============================================================================
    print0("[FSDP DMD] Preparing Generator (student)...")
    generator_transformer = base_transformer

    # ============================================================================
    # 2. TEACHER (Real Score) - Create copy BEFORE FSDP wrapping
    # ============================================================================
    print0("[FSDP DMD] Preparing Teacher (real_score)...")

    if teacher_model_path and os.path.exists(teacher_model_path):
        print0(f"  Loading teacher from: {teacher_model_path}")
        from diffusers import WanPipeline

        # Load full pipeline
        teacher_pipe = WanPipeline.from_pretrained(
            teacher_model_path,
            torch_dtype=torch.float32,
        )

        # Extract transformer
        teacher_transformer = teacher_pipe.transformer

        # Clean up VAE
        if hasattr(teacher_pipe, "vae") and teacher_pipe.vae is not None:
            del teacher_pipe.vae
            teacher_pipe.vae = None

        # Clean up text encoder
        if hasattr(teacher_pipe, "text_encoder") and teacher_pipe.text_encoder is not None:
            del teacher_pipe.text_encoder
            teacher_pipe.text_encoder = None

        # Delete pipeline
        del teacher_pipe

        # Force cleanup
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print0("  ✓ Teacher loaded from checkpoint")
    else:
        # Use base model as teacher (create independent copy)
        print0("  Creating teacher from base model...")

        # Create new instance with same config
        teacher_transformer = type(base_transformer)(**base_transformer.config)

        # Load the original state dict (not FSDP-wrapped)
        teacher_transformer.load_state_dict(original_state_dict)

        print0("  ✓ Teacher created from base transformer")

    # ============================================================================
    # 3. CRITIC (Fake Score) - Create copy BEFORE FSDP wrapping
    # ============================================================================
    print0("[FSDP DMD] Preparing Critic (fake_score)...")

    # Load full pipeline for critic
    from diffusers import WanPipeline

    critic_pipe = WanPipeline.from_pretrained(
        pipe.config._name_or_path,
        torch_dtype=torch.float32,
    )

    # Extract transformer
    critic_transformer = critic_pipe.transformer

    # Clean up VAE
    if hasattr(critic_pipe, "vae") and critic_pipe.vae is not None:
        del critic_pipe.vae
        critic_pipe.vae = None

    # Clean up text encoder
    if hasattr(critic_pipe, "text_encoder") and critic_pipe.text_encoder is not None:
        del critic_pipe.text_encoder
        critic_pipe.text_encoder = None

    # Delete pipeline
    del critic_pipe

    # Force cleanup
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print0("  ✓ Critic loaded and cleaned up")

    # Delete original_state_dict to free memory
    del original_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print0("[FSDP DMD] ✓ All transformer instances created")

    # ============================================================================
    # Now wrap each transformer in FSDP
    # ============================================================================

    # Detect block types for wrapping policy (same for all models)
    block_types = detect_transformer_block_types(generator_transformer)
    auto_wrap_policy = create_fsdp_wrap_policy(generator_transformer)
    mixed_precision_policy = create_fsdp_mixed_precision_policy(bf16)

    # ============================================================================
    # Wrap 1: GENERATOR
    # ============================================================================
    print0("[FSDP DMD] Wrapping Generator in FSDP...")

    cast_model_to_dtype(generator_transformer, bf16)
    generator_transformer.to(device)

    # All parameters trainable
    for param in generator_transformer.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in generator_transformer.parameters() if p.requires_grad)
    print0(f"  Generator: {trainable_params:,} trainable parameters")

    # Disable built-in gradient checkpointing
    if hasattr(generator_transformer, "gradient_checkpointing"):
        generator_transformer.gradient_checkpointing = False

    fsdp_generator = FSDP(
        generator_transformer,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=cpu_offload_config,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    # Configure sharded state dict for generator
    FSDP.set_state_dict_type(
        fsdp_generator,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    )

    # Apply activation checkpointing
    if block_types:
        apply_fsdp_activation_checkpointing(fsdp_generator, block_types)

    print0("[FSDP DMD] ✓ Generator wrapped in FSDP")

    # ============================================================================
    # Wrap 2: TEACHER
    # ============================================================================
    print0("[FSDP DMD] Wrapping Teacher in FSDP...")

    cast_model_to_dtype(teacher_transformer, bf16)
    teacher_transformer.to(device)

    # All parameters frozen
    for param in teacher_transformer.parameters():
        param.requires_grad = False

    teacher_transformer.eval()

    frozen_params = sum(p.numel() for p in teacher_transformer.parameters())
    print0(f"  Teacher: {frozen_params:,} frozen parameters")

    # Aggressive CPU offload for teacher
    teacher_offload_config = CPUOffload(offload_params=True)

    fsdp_teacher = FSDP(
        teacher_transformer,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=teacher_offload_config,  # Always offload teacher
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    print0("[FSDP DMD] ✓ Teacher wrapped in FSDP")

    # ============================================================================
    # Wrap 3: CRITIC
    # ============================================================================
    print0("[FSDP DMD] Wrapping Critic in FSDP...")

    cast_model_to_dtype(critic_transformer, bf16)
    critic_transformer.to(device)

    # All parameters trainable
    for param in critic_transformer.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in critic_transformer.parameters() if p.requires_grad)
    print0(f"  Critic: {trainable_params:,} trainable parameters")

    # Disable built-in gradient checkpointing
    if hasattr(critic_transformer, "gradient_checkpointing"):
        critic_transformer.gradient_checkpointing = False

    fsdp_critic = FSDP(
        critic_transformer,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=cpu_offload_config,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    # Configure sharded state dict for critic
    FSDP.set_state_dict_type(
        fsdp_critic,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    )

    # Apply activation checkpointing
    if block_types:
        apply_fsdp_activation_checkpointing(fsdp_critic, block_types)

    print0("[FSDP DMD] ✓ Critic wrapped in FSDP")

    # Final memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ============================================================================
    # Create model map
    # ============================================================================
    model_map = {
        "generator": {
            "fsdp_transformer": fsdp_generator,
            "base_transformer": generator_transformer,
        },
        "real_score": {
            "fsdp_transformer": fsdp_teacher,
            "base_transformer": teacher_transformer,
        },
        "fake_score": {
            "fsdp_transformer": fsdp_critic,
            "base_transformer": critic_transformer,
        },
    }

    print0("[FSDP DMD] All models setup complete")

    return model_map


def verify_fsdp_dmd_setup(model_map: Dict):
    """Verify FSDP DMD setup."""
    print0("[FSDP DMD] Verifying setup...")

    for model_name in ["generator", "real_score", "fake_score"]:
        fsdp_model = model_map[model_name]["fsdp_transformer"]

        print0(f"[FSDP DMD] {model_name}:")
        print0(f"  - FSDP wrapped: {isinstance(fsdp_model, FSDP)}")

        trainable_count = sum(1 for p in fsdp_model.parameters() if p.requires_grad)
        frozen_count = sum(1 for p in fsdp_model.parameters() if not p.requires_grad)

        print0(f"  - Trainable parameters: {trainable_count}")
        print0(f"  - Frozen parameters: {frozen_count}")

        if model_name == "real_score" and trainable_count > 0:
            print0("[WARNING] Teacher should be frozen!")

        if model_name in ["generator", "fake_score"] and trainable_count == 0:
            raise RuntimeError(f"{model_name} has no trainable parameters!")

    print0("[FSDP DMD] ✓ Verification complete")


def test_fsdp_dmd_forward_pass(model_map: Dict, device, bf16):
    """Test forward pass for all three models."""
    print0("[FSDP DMD] Testing forward passes...")

    # T2V uses 16-channel input: [B, C, F, H, W]
    batch_size = 1
    num_channels = 16
    num_frames = 8
    height = 32
    width = 32

    dummy_input = torch.randn(batch_size, num_channels, num_frames, height, width, device=device, dtype=bf16)

    # CRITICAL FIX: timestep should be [B], NOT [B, F]
    # The transformer broadcasts timestep internally across frames
    dummy_timestep = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)

    dummy_encoder_hidden = torch.randn(batch_size, 77, 4096, device=device, dtype=bf16)

    print0("[FSDP DMD] Test tensor shapes:")
    print0(f"  - Input: {dummy_input.shape}")
    print0(f"  - Timestep: {dummy_timestep.shape}")
    print0(f"  - Encoder hidden: {dummy_encoder_hidden.shape}")

    for model_name in ["generator", "real_score", "fake_score"]:
        print0(f"[FSDP DMD] Testing {model_name}...")

        model = model_map[model_name]["fsdp_transformer"]

        try:
            with torch.no_grad():
                output = model(
                    hidden_states=dummy_input,
                    timestep=dummy_timestep.to(bf16),  # [B] shape
                    encoder_hidden_states=dummy_encoder_hidden,
                    return_dict=False,
                )
                if isinstance(output, tuple):
                    output = output[0]

                print0(f"  ✓ {model_name} forward pass successful! Output shape: {output.shape}")
        except Exception as e:
            print0(f"  ✗ {model_name} forward pass failed: {e}")
            import traceback

            traceback.print_exc()
            raise  # Re-raise to stop execution

    print0("[FSDP DMD] ✓ Forward pass tests complete")


def get_fsdp_dmd_trainable_parameters(model_map: Dict, optimize_both: bool = True):
    """
    Get trainable parameters for DMD training.

    Args:
        model_map: Model map with generator, real_score, fake_score
        optimize_both: If True, return separate param groups for generator and critic
                      If False, only return generator params

    Returns:
        If optimize_both: (generator_params, critic_params)
        Else: generator_params
    """
    generator_params = [p for p in model_map["generator"]["fsdp_transformer"].parameters() if p.requires_grad]

    print0(f"[FSDP DMD] Generator: {len(generator_params)} trainable parameter tensors")

    if len(generator_params) == 0:
        raise RuntimeError("No trainable parameters in generator!")

    if optimize_both:
        critic_params = [p for p in model_map["fake_score"]["fsdp_transformer"].parameters() if p.requires_grad]

        print0(f"[FSDP DMD] Critic: {len(critic_params)} trainable parameter tensors")

        if len(critic_params) == 0:
            raise RuntimeError("No trainable parameters in critic!")

        return generator_params, critic_params
    else:
        return generator_params


def save_fsdp_dmd_checkpoint(
    model_map: Dict,
    generator_optimizer,
    critic_optimizer,
    scheduler,
    output_dir: str,
    step: int,
    consolidate: bool = None,
):
    """Save FSDP DMD checkpoint with 3 models."""
    from torch.distributed.checkpoint import FileSystemWriter
    from torch.distributed.checkpoint import save as dist_save
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    if consolidate is None:
        consolidate = step % 1000 == 0

    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")

    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    print0(f"[FSDP DMD] Saving checkpoint to {ckpt_dir}")
    if consolidate:
        print0("[FSDP DMD] Will save consolidated models (inference-ready)")

    # Save generator
    print0("[FSDP DMD] Saving generator...")
    fsdp_generator = model_map["generator"]["fsdp_transformer"]

    generator_state_dict = {"model": fsdp_generator.state_dict()}
    generator_path = os.path.join(ckpt_dir, "generator_model")

    if is_main_process():
        os.makedirs(generator_path, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    dist_save(
        state_dict=generator_state_dict,
        storage_writer=FileSystemWriter(generator_path),
    )

    # Save generator optimizer
    generator_optim_state = FSDP.optim_state_dict(model=fsdp_generator, optim=generator_optimizer)
    generator_optim_path = os.path.join(ckpt_dir, "generator_optimizer")

    if is_main_process():
        os.makedirs(generator_optim_path, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    dist_save(
        state_dict={"optimizer": generator_optim_state},
        storage_writer=FileSystemWriter(generator_optim_path),
    )

    print0("[FSDP DMD] ✓ Generator saved")

    # Save critic
    print0("[FSDP DMD] Saving critic...")
    fsdp_critic = model_map["fake_score"]["fsdp_transformer"]

    critic_state_dict = {"model": fsdp_critic.state_dict()}
    critic_path = os.path.join(ckpt_dir, "critic_model")

    if is_main_process():
        os.makedirs(critic_path, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    dist_save(
        state_dict=critic_state_dict,
        storage_writer=FileSystemWriter(critic_path),
    )

    # Save critic optimizer
    critic_optim_state = FSDP.optim_state_dict(model=fsdp_critic, optim=critic_optimizer)
    critic_optim_path = os.path.join(ckpt_dir, "critic_optimizer")

    if is_main_process():
        os.makedirs(critic_optim_path, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    dist_save(
        state_dict={"optimizer": critic_optim_state},
        storage_writer=FileSystemWriter(critic_optim_path),
    )

    print0("[FSDP DMD] ✓ Critic saved")

    # Save consolidated models if requested
    if consolidate:
        print0("[FSDP DMD] Consolidating models...")

        import time

        start_time = time.time()

        # Consolidate generator
        with FSDP.state_dict_type(
            fsdp_generator,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            generator_consolidated = fsdp_generator.state_dict()

        if is_main_process() and len(generator_consolidated) > 0:
            generator_consolidated_path = os.path.join(ckpt_dir, "generator_consolidated.bin")
            torch.save(generator_consolidated, generator_consolidated_path)
            file_size_gb = os.path.getsize(generator_consolidated_path) / 1024**3
            print0(f"[FSDP DMD] ✓ Generator consolidated ({file_size_gb:.2f} GB)")

        # Consolidate critic
        with FSDP.state_dict_type(
            fsdp_critic,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            critic_consolidated = fsdp_critic.state_dict()

        if is_main_process() and len(critic_consolidated) > 0:
            critic_consolidated_path = os.path.join(ckpt_dir, "critic_consolidated.bin")
            torch.save(critic_consolidated, critic_consolidated_path)
            file_size_gb = os.path.getsize(critic_consolidated_path) / 1024**3
            print0(f"[FSDP DMD] ✓ Critic consolidated ({file_size_gb:.2f} GB)")

        consolidation_time = time.time() - start_time
        print0(f"[FSDP DMD] Consolidation took {consolidation_time:.1f}s")

    # Save training state
    if is_main_process():
        training_state = {
            "step": step,
            "scheduler": scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))
        print0("[FSDP DMD] ✓ Training state saved")

    if dist.is_initialized():
        dist.barrier()

    print0(f"[FSDP DMD] ✓ Checkpoint saved at step {step}")


def load_fsdp_dmd_checkpoint(model_map: Dict, generator_optimizer, critic_optimizer, scheduler, ckpt_path: str) -> int:
    """Load FSDP DMD checkpoint."""
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint import load as dist_load

    if not os.path.exists(ckpt_path):
        print0(f"[WARNING] Checkpoint {ckpt_path} not found")
        return 0

    print0(f"[FSDP DMD] Loading checkpoint from {ckpt_path}")

    # Load generator
    generator_path = os.path.join(ckpt_path, "generator_model")
    if os.path.exists(generator_path):
        fsdp_generator = model_map["generator"]["fsdp_transformer"]
        generator_state_dict = {"model": fsdp_generator.state_dict()}

        dist_load(
            state_dict=generator_state_dict,
            storage_reader=FileSystemReader(generator_path),
        )

        fsdp_generator.load_state_dict(generator_state_dict["model"])
        print0("[FSDP DMD] ✓ Generator loaded")

        # Load generator optimizer
        generator_optim_path = os.path.join(ckpt_path, "generator_optimizer")
        if os.path.exists(generator_optim_path):
            generator_optim_state = {
                "optimizer": FSDP.optim_state_dict(
                    model=fsdp_generator,
                    optim=generator_optimizer,
                )
            }

            dist_load(
                state_dict=generator_optim_state,
                storage_reader=FileSystemReader(generator_optim_path),
            )

            FSDP.optim_state_dict_to_load(
                model=fsdp_generator,
                optim=generator_optimizer,
                optim_state_dict=generator_optim_state["optimizer"],
            )
            print0("[FSDP DMD] ✓ Generator optimizer loaded")

    # Load critic
    critic_path = os.path.join(ckpt_path, "critic_model")
    if os.path.exists(critic_path):
        fsdp_critic = model_map["fake_score"]["fsdp_transformer"]
        critic_state_dict = {"model": fsdp_critic.state_dict()}

        dist_load(
            state_dict=critic_state_dict,
            storage_reader=FileSystemReader(critic_path),
        )

        fsdp_critic.load_state_dict(critic_state_dict["model"])
        print0("[FSDP DMD] ✓ Critic loaded")

        # Load critic optimizer
        critic_optim_path = os.path.join(ckpt_path, "critic_optimizer")
        if os.path.exists(critic_optim_path):
            critic_optim_state = {
                "optimizer": FSDP.optim_state_dict(
                    model=fsdp_critic,
                    optim=critic_optimizer,
                )
            }

            dist_load(
                state_dict=critic_optim_state,
                storage_reader=FileSystemReader(critic_optim_path),
            )

            FSDP.optim_state_dict_to_load(
                model=fsdp_critic,
                optim=critic_optimizer,
                optim_state_dict=critic_optim_state["optimizer"],
            )
            print0("[FSDP DMD] ✓ Critic optimizer loaded")

    # Load training state
    training_state_path = os.path.join(ckpt_path, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        scheduler.load_state_dict(training_state["scheduler"])
        step = int(training_state.get("step", 0))
        print0(f"[FSDP DMD] ✓ Training state loaded from step {step}")
    else:
        step = 0

    if dist.is_initialized():
        dist.barrier()

    return step
