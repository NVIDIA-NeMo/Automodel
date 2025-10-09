# fsdp2_utils_t2v.py - FSDP setup for WAN 2.1 T2V FULL fine-tuning
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

    # Store references to modules we should NOT wrap
    ignore_modules = set()

    if hasattr(transformer, "condition_embedder"):
        ignore_modules.add(transformer.condition_embedder)
        print0("[FSDP] Will NOT wrap condition_embedder (prevents dimension mismatch)")

    if hasattr(transformer, "patch_embedding"):
        ignore_modules.add(transformer.patch_embedding)
        print0("[FSDP] Will NOT wrap patch_embedding (prevents dimension mismatch)")

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


def setup_fsdp_for_t2v_pipe(
    pipe,
    device,
    bf16,
    local_rank: int,
    cpu_offload: bool = False,
):
    """Setup FSDP for WAN 2.1 T2V transformer - FULL fine-tuning."""

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print0(f"[FSDP] Setting up T2V FULL fine-tuning with world_size={world_size}, rank={rank}")
    print0(f"[FSDP] CPU offloading: {'ENABLED' if cpu_offload else 'DISABLED'}")

    cpu_offload_config = None
    if cpu_offload:
        cpu_offload_config = CPUOffload(offload_params=True)
        print0("[FSDP] CPU offload will move parameters to CPU when not in use")

    # WAN 2.1 T2V has single transformer
    if not hasattr(pipe, "transformer") or pipe.transformer is None:
        raise RuntimeError("transformer not found in pipeline")

    print0("[INFO] Setting up single transformer for T2V")

    base_transformer = pipe.transformer

    cast_model_to_dtype(base_transformer, bf16)
    base_transformer.to(device)

    # FULL FINE-TUNING: All parameters trainable
    for param in base_transformer.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in base_transformer.parameters() if p.requires_grad)
    print0(f"[FSDP] Full fine-tuning: {trainable_params:,} trainable parameters")

    # Disable built-in gradient checkpointing
    if hasattr(base_transformer, "gradient_checkpointing"):
        base_transformer.gradient_checkpointing = False
        print0("[FSDP] Disabled built-in gradient_checkpointing")

    auto_wrap_policy = create_fsdp_wrap_policy(base_transformer)
    mixed_precision_policy = create_fsdp_mixed_precision_policy(bf16)

    print0("[FSDP] Wrapping transformer with FSDP...")

    fsdp_transformer = FSDP(
        base_transformer,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=cpu_offload_config,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    print0("[FSDP] Successfully wrapped transformer")
    if cpu_offload:
        print0("[FSDP] CPU offload enabled - parameters will be offloaded to CPU")

    # Configure sharded state dict
    FSDP.set_state_dict_type(
        fsdp_transformer,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    )

    print0("[FSDP] Configured optimizer state sharding")

    # Apply activation checkpointing
    block_types = detect_transformer_block_types(base_transformer)
    if block_types:
        print0("[FSDP] Applying activation checkpointing...")
        apply_fsdp_activation_checkpointing(fsdp_transformer, block_types)

    # Verify all parameters are trainable
    trainable_count = sum(1 for p in fsdp_transformer.parameters() if p.requires_grad)
    frozen_count = sum(1 for p in fsdp_transformer.parameters() if not p.requires_grad)
    print0(f"[FSDP] Trainable parameters: {trainable_count}, Frozen: {frozen_count}")

    model_map = {
        "transformer": {
            "fsdp_transformer": fsdp_transformer,
            "base_transformer": base_transformer,
        }
    }

    pipe.transformer = fsdp_transformer

    print0("[FSDP] Transformer setup complete")

    return model_map


def verify_fsdp_setup(model_map: Dict):
    """Verify FSDP setup."""
    print0("[FSDP] Verifying setup...")

    fsdp_model = model_map["transformer"]["fsdp_transformer"]

    print0(f"[FSDP] transformer:")
    print0(f"  - FSDP wrapped: {isinstance(fsdp_model, FSDP)}")

    trainable_count = sum(1 for p in fsdp_model.parameters() if p.requires_grad)
    frozen_count = sum(1 for p in fsdp_model.parameters() if not p.requires_grad)

    print0(f"  - Trainable parameters: {trainable_count}")
    print0(f"  - Frozen parameters: {frozen_count}")

    if trainable_count == 0:
        raise RuntimeError("No trainable parameters found!")


def test_fsdp_forward_pass(model_map: Dict, device, bf16):
    """Test forward pass."""
    print0("[FSDP] Testing forward pass...")

    model = model_map["transformer"]["fsdp_transformer"]
    model.train()

    # T2V uses 16-channel input (no conditioning concatenation)
    dummy_input = torch.randn(1, 16, 8, 32, 32, device=device, dtype=bf16)
    dummy_timestep = torch.randint(0, 1000, (1,), device=device)
    dummy_encoder_hidden = torch.randn(1, 77, 4096, device=device, dtype=bf16)

    print0("[FSDP] Testing transformer")

    try:
        with torch.no_grad():
            output = model(
                hidden_states=dummy_input,
                timestep=dummy_timestep,
                encoder_hidden_states=dummy_encoder_hidden,
                return_dict=False,
            )
            if isinstance(output, tuple):
                output = output[0]

            print0(f"[FSDP] Transformer forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print0(f"[WARNING] Transformer test failed: {e}")
        import traceback
        traceback.print_exc()


def get_fsdp_all_parameters(model_map: Dict):
    """Get ALL trainable parameters for full fine-tuning."""
    fsdp_model = model_map["transformer"]["fsdp_transformer"]
    all_params = [p for p in fsdp_model.parameters() if p.requires_grad]
    print0(f"[FSDP] Collected {len(all_params)} trainable parameters")
    
    if len(all_params) == 0:
        raise RuntimeError("No trainable parameters found!")
    
    return all_params


def save_fsdp_checkpoint(model_map: Dict, optimizer, scheduler, output_dir: str, step: int):
    """Save FSDP checkpoint with sharded optimizer states - FULL MODEL."""
    from torch.distributed.checkpoint import save as dist_save
    from torch.distributed.checkpoint import FileSystemWriter

    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    print0(f"[FSDP] Saving sharded checkpoint to {ckpt_dir}")

    # Save model state dict (sharded)
    fsdp_model = model_map["transformer"]["fsdp_transformer"]
    model_state_dict = {"model": fsdp_model.state_dict()}
    model_path = os.path.join(ckpt_dir, "transformer_model")
    os.makedirs(model_path, exist_ok=True)

    dist_save(
        state_dict=model_state_dict,
        storage_writer=FileSystemWriter(model_path),
    )

    print0("[FSDP] Saved transformer model state")

    # Save optimizer state dict (sharded)
    optimizer_state = FSDP.optim_state_dict(
        model=fsdp_model,
        optim=optimizer,
    )

    optim_path = os.path.join(ckpt_dir, "optimizer")
    os.makedirs(optim_path, exist_ok=True)

    dist_save(
        state_dict={"optimizer": optimizer_state},
        storage_writer=FileSystemWriter(optim_path),
    )

    print0("[FSDP] Saved sharded optimizer state")

    # Save scheduler and step (rank 0 only)
    if is_main_process():
        training_state = {
            "step": step,
            "scheduler": scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))
        print0("[FSDP] Saved training state")

    if dist.is_initialized():
        dist.barrier()

    print0(f"[FSDP] Checkpoint saved at step {step}")


def load_fsdp_checkpoint(model_map: Dict, optimizer, scheduler, ckpt_path: str) -> int:
    """Load FSDP checkpoint with sharded optimizer states - FULL MODEL."""
    from torch.distributed.checkpoint import load as dist_load
    from torch.distributed.checkpoint import FileSystemReader

    if not os.path.exists(ckpt_path):
        print0(f"[WARNING] Checkpoint {ckpt_path} not found")
        return 0

    print0(f"[FSDP] Loading sharded checkpoint from {ckpt_path}")

    # Load model state dict
    model_path = os.path.join(ckpt_path, "transformer_model")
    if not os.path.exists(model_path):
        print0("[WARNING] Model checkpoint not found")
        return 0

    fsdp_model = model_map["transformer"]["fsdp_transformer"]
    model_state_dict = {"model": fsdp_model.state_dict()}

    dist_load(
        state_dict=model_state_dict,
        storage_reader=FileSystemReader(model_path),
    )

    fsdp_model.load_state_dict(model_state_dict["model"])
    print0("[FSDP] Loaded transformer model state")

    # Load optimizer state dict
    optim_path = os.path.join(ckpt_path, "optimizer")
    if os.path.exists(optim_path):
        optimizer_state = {
            "optimizer": FSDP.optim_state_dict(
                model=fsdp_model,
                optim=optimizer,
            )
        }

        dist_load(
            state_dict=optimizer_state,
            storage_reader=FileSystemReader(optim_path),
        )

        FSDP.optim_state_dict_to_load(
            model=fsdp_model,
            optim=optimizer,
            optim_state_dict=optimizer_state["optimizer"],
        )

        print0("[FSDP] Loaded sharded optimizer state")

    # Load scheduler and step
    training_state_path = os.path.join(ckpt_path, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        scheduler.load_state_dict(training_state["scheduler"])
        step = int(training_state.get("step", 0))
        print0(f"[FSDP] Loaded training state from step {step}")
    else:
        step = 0

    if dist.is_initialized():
        dist.barrier()

    return step