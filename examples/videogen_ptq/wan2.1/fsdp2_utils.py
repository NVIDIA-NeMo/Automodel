# fsdp2_utils.py - FSDP2 setup for WAN LoRA training with optimizer state sharding
import os
import torch
import torch.distributed as dist
from typing import Dict, List, Set
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType, CPUOffload
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.api import ShardedOptimStateDictConfig, ShardedStateDictConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from dist_utils import print0, cast_model_to_dtype, is_main_process
from lora_utils import (
    wan_install_and_materialize_lora, 
    collect_wan_lora_parameters,
    LoRALinear
)


def create_fsdp2_mixed_precision_policy(bf16):
    """Create mixed precision policy for FSDP2."""
    return MixedPrecision(
        param_dtype=bf16,
        reduce_dtype=bf16,
        buffer_dtype=bf16,
    )


def detect_transformer_block_types(transformer):
    """Detect transformer block types for activation checkpointing."""
    block_types = set()
    
    if hasattr(transformer, 'blocks') and transformer.blocks is not None:
        if len(transformer.blocks) > 0:
            block_type = type(transformer.blocks[0])
            block_types.add(block_type)
            print0(f"[FSDP2] Detected transformer block type: {block_type.__name__}")
    
    return block_types


def create_fsdp2_wrap_policy(transformer):
    """
    Create FSDP2 wrapping policy.
    
    CRITICAL rules based on TP implementation:
    - DON'T wrap condition_embedder (causes dimension mismatch)
    - DON'T wrap patch_embedding (causes dimension mismatch)
    - DON'T wrap LoRA modules (keep unsharded)
    - DO wrap transformer blocks (these get sharded)
    """
    wrap_module_types = set()
    
    if hasattr(transformer, 'blocks') and transformer.blocks is not None:
        if len(transformer.blocks) > 0:
            block_type = type(transformer.blocks[0])
            wrap_module_types.add(block_type)
            print0(f"[FSDP2] Will wrap transformer blocks of type: {block_type.__name__}")
    
    # Store references to modules we should NOT wrap
    ignore_modules = set()
    
    if hasattr(transformer, 'condition_embedder'):
        ignore_modules.add(transformer.condition_embedder)
        print0("[FSDP2] Will NOT wrap condition_embedder (prevents dimension mismatch)")
    
    if hasattr(transformer, 'patch_embedding'):
        ignore_modules.add(transformer.patch_embedding)
        print0("[FSDP2] Will NOT wrap patch_embedding (prevents dimension mismatch)")
    
    if not wrap_module_types:
        print0("[FSDP2] WARNING: No transformer blocks found")
        return None
    
    # Use ModuleWrapPolicy with the set of module types
    # This automatically wraps instances of these types
    return ModuleWrapPolicy(wrap_module_types)


def apply_fsdp2_activation_checkpointing(fsdp_model, transformer_block_types):
    """Apply activation checkpointing to transformer blocks."""
    if not transformer_block_types:
        print0("[FSDP2] No block types for activation checkpointing")
        return
    
    def check_fn(submodule):
        if type(submodule) in transformer_block_types:
            return True
        if isinstance(submodule, LoRALinear):
            return False
        return False
    
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=lambda module: checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=check_fn,
    )
    
    print0(f"[FSDP2] Applied activation checkpointing")


def setup_fsdp2_for_pipe(
    pipe,
    device,
    bf16,
    local_rank: int,
    lora_rank: int,
    lora_alpha: int,
    train_transformer_2: bool = False,
    cpu_offload: bool = False,
):
    """Setup FSDP2 for WAN transformer with LoRA."""
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    print0(f"[FSDP2] Setting up with world_size={world_size}, rank={rank}")
    print0(f"[FSDP2] CPU offloading: {'ENABLED' if cpu_offload else 'DISABLED'}")
    
    cpu_offload_config = None
    if cpu_offload:
        cpu_offload_config = CPUOffload(offload_params=True)
        print0("[FSDP2] CPU offload will move parameters to CPU when not in use")
    
    # Determine which transformer to train
    transformer_names: List[str] = []
    
    if train_transformer_2:
        if hasattr(pipe, "transformer_2") and getattr(pipe, "transformer_2") is not None:
            transformer_names.append("transformer_2")
            print0("[INFO] Setting up transformer_2 only")
        else:
            raise RuntimeError("transformer_2 not found")
    else:
        if hasattr(pipe, "transformer") and getattr(pipe, "transformer") is not None:
            transformer_names.append("transformer")
            print0("[INFO] Setting up transformer only")
        else:
            raise RuntimeError("transformer not found")
    
    print0(f"[FSDP2] Will setup: {transformer_names}")
    
    model_map: Dict[str, Dict] = {}
    
    for name in transformer_names:
        base_transformer = getattr(pipe, name)
        
        cast_model_to_dtype(base_transformer, bf16)
        base_transformer.to(device)
        
        print0(f"[FSDP2] Installing LoRA on {name} before FSDP wrapping...")
        
        num_lora_modules = wan_install_and_materialize_lora(
            base_transformer, 
            rank=lora_rank, 
            alpha=lora_alpha,
            dropout=0.05
        )
        
        lora_params = collect_wan_lora_parameters(base_transformer)
        
        if len(lora_params) == 0:
            raise RuntimeError(f"No LoRA parameters found for {name}")
        
        print0(f"[FSDP2] Installed LoRA: {num_lora_modules} modules, {len(lora_params)} parameters")
        
        # Set requires_grad BEFORE FSDP wrapping
        for param_name, param in base_transformer.named_parameters():
            if any(lora_p is param for lora_p in lora_params):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        print0(f"[FSDP2] Set requires_grad for {name}")
        
        # CRITICAL: Disable model's built-in gradient checkpointing
        # We'll use FSDP2's activation checkpointing instead
        if hasattr(base_transformer, 'gradient_checkpointing'):
            base_transformer.gradient_checkpointing = False
            print0(f"[FSDP2] Disabled built-in gradient_checkpointing for {name}")
        
        if hasattr(base_transformer, 'enable_gradient_checkpointing'):
            # Don't call this - it would conflict with FSDP2's checkpointing
            print0(f"[FSDP2] Skipping enable_gradient_checkpointing for {name}")
        
        auto_wrap_policy = create_fsdp2_wrap_policy(base_transformer)
        mixed_precision_policy = create_fsdp2_mixed_precision_policy(bf16)
        
        print0(f"[FSDP2] Wrapping {name} with FSDP2...")
        
        fsdp_transformer = FSDP(
            base_transformer,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=cpu_offload_config,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
        
        print0(f"[FSDP2] Successfully wrapped {name}")
        if cpu_offload:
            print0(f"[FSDP2] CPU offload enabled for {name}")
        
        # Configure sharded state dict
        FSDP.set_state_dict_type(
            fsdp_transformer,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
        )
        
        print0(f"[FSDP2] Configured optimizer state sharding for {name}")
        
        block_types = detect_transformer_block_types(base_transformer)
        
        if block_types:
            print0(f"[FSDP2] Applying activation checkpointing to {name}...")
            apply_fsdp2_activation_checkpointing(fsdp_transformer, block_types)
        
        final_lora_params = []
        for module in fsdp_transformer.modules():
            if isinstance(module, LoRALinear):
                final_lora_params.extend([module.A, module.B])
        
        print0(f"[FSDP2] Verified {len(final_lora_params)} LoRA parameters after wrapping")
        
        if len(final_lora_params) == 0:
            raise RuntimeError(f"LoRA parameters lost after FSDP wrapping for {name}")
        
        model_map[name] = {
            "fsdp_transformer": fsdp_transformer,
            "base_transformer": base_transformer,
            "lora_params": final_lora_params,
            "num_lora_modules": num_lora_modules,
        }
        
        setattr(pipe, name, fsdp_transformer)
        
        print0(f"[FSDP2] {name} setup complete")
    
    print0("[FSDP2] All transformers setup complete")
    
    total_lora_params = sum(len(model_map[name]["lora_params"]) for name in transformer_names)
    print0(f"[FSDP2] Total LoRA parameters: {total_lora_params}")
    
    if total_lora_params == 0:
        raise RuntimeError("No LoRA parameters found after FSDP setup!")
    
    return model_map, transformer_names


def verify_fsdp2_setup(model_map: Dict, transformer_names: List[str]):
    """Verify FSDP2 setup."""
    print0("[FSDP2] Verifying setup...")
    
    for name in transformer_names:
        if name not in model_map:
            continue
        
        fsdp_model = model_map[name]["fsdp_transformer"]
        lora_params = model_map[name]["lora_params"]
        
        print0(f"[FSDP2] {name}:")
        print0(f"  - FSDP wrapped: {isinstance(fsdp_model, FSDP)}")
        print0(f"  - LoRA parameters: {len(lora_params)}")
        print0(f"  - LoRA modules: {model_map[name]['num_lora_modules']}")
        
        trainable_count = sum(1 for p in fsdp_model.parameters() if p.requires_grad)
        frozen_count = sum(1 for p in fsdp_model.parameters() if not p.requires_grad)
        
        print0(f"  - Trainable parameters: {trainable_count}")
        print0(f"  - Frozen parameters: {frozen_count}")


def test_fsdp2_forward_pass(model_map: Dict, transformer_names: List[str], device, bf16):
    """Test forward pass."""
    print0("[FSDP2] Testing forward pass...")
    
    for name in transformer_names:
        if name not in model_map:
            continue
        
        model = model_map[name]["fsdp_transformer"]
        model.train()
        
        dummy_input = torch.randn(1, 36, 8, 32, 32, device=device, dtype=bf16)
        dummy_timestep = torch.randint(0, 1000, (1,), device=device)
        dummy_encoder_hidden = torch.randn(1, 77, 4096, device=device, dtype=bf16)
        
        print0(f"[FSDP2] Testing {name}")
        
        try:
            with torch.no_grad():
                output = model(
                    hidden_states=dummy_input,
                    timestep=dummy_timestep,
                    encoder_hidden_states=dummy_encoder_hidden,
                    return_dict=False
                )
                if isinstance(output, tuple):
                    output = output[0]
                
                print0(f"[FSDP2] {name} forward pass successful!")
        except Exception as e:
            print0(f"[WARNING] {name} test failed: {e}")


def get_fsdp2_lora_parameters(model_map: Dict, transformer_names: List[str]) -> List[torch.nn.Parameter]:
    """Get all LoRA parameters."""
    all_params = []
    
    for name in transformer_names:
        if name in model_map:
            all_params.extend(model_map[name]["lora_params"])
    
    print0(f"[FSDP2] Collected {len(all_params)} LoRA parameters")
    
    return all_params


def save_fsdp2_checkpoint(model_map: Dict, transformer_names: List[str], optimizer, scheduler, 
                          output_dir: str, step: int):
    """Save FSDP2 checkpoint with sharded optimizer states."""
    from torch.distributed.checkpoint import save_state_dict
    
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print0(f"[FSDP2] Saving sharded checkpoint to {ckpt_dir}")
    
    # Save model state dict (sharded)
    for name in transformer_names:
        if name not in model_map:
            continue
        
        fsdp_model = model_map[name]["fsdp_transformer"]
        model_state_dict = {"model": fsdp_model.state_dict()}
        model_path = os.path.join(ckpt_dir, f"{name}_model")
        os.makedirs(model_path, exist_ok=True)
        
        save_state_dict(
            state_dict=model_state_dict,
            storage_writer=torch.distributed.checkpoint.FileSystemWriter(model_path),
        )
        
        print0(f"[FSDP2] Saved {name} model state")
    
    # Save optimizer state dict (sharded)
    optimizer_state = FSDP.optim_state_dict(
        model=model_map[transformer_names[0]]["fsdp_transformer"],
        optim=optimizer,
    )
    
    optim_path = os.path.join(ckpt_dir, "optimizer")
    os.makedirs(optim_path, exist_ok=True)
    
    save_state_dict(
        state_dict={"optimizer": optimizer_state},
        storage_writer=torch.distributed.checkpoint.FileSystemWriter(optim_path),
    )
    
    print0(f"[FSDP2] Saved sharded optimizer state")
    
    # Save scheduler and step (rank 0 only)
    if is_main_process():
        training_state = {
            "step": step,
            "scheduler": scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))
        print0(f"[FSDP2] Saved training state")
    
    if dist.is_initialized():
        dist.barrier()
    
    print0(f"[FSDP2] Checkpoint saved at step {step}")


def load_fsdp2_checkpoint(model_map: Dict, transformer_names: List[str], optimizer, scheduler,
                          ckpt_path: str) -> int:
    """Load FSDP2 checkpoint with sharded optimizer states."""
    from torch.distributed.checkpoint import load_state_dict
    
    if not os.path.exists(ckpt_path):
        print0(f"[WARNING] Checkpoint {ckpt_path} not found")
        return 0
    
    print0(f"[FSDP2] Loading sharded checkpoint from {ckpt_path}")
    
    # Load model state dict
    for name in transformer_names:
        if name not in model_map:
            continue
        
        model_path = os.path.join(ckpt_path, f"{name}_model")
        if not os.path.exists(model_path):
            print0(f"[WARNING] Model checkpoint for {name} not found")
            continue
        
        fsdp_model = model_map[name]["fsdp_transformer"]
        model_state_dict = {"model": fsdp_model.state_dict()}
        
        load_state_dict(
            state_dict=model_state_dict,
            storage_reader=torch.distributed.checkpoint.FileSystemReader(model_path),
        )
        
        fsdp_model.load_state_dict(model_state_dict["model"])
        print0(f"[FSDP2] Loaded {name} model state")
    
    # Load optimizer state dict
    optim_path = os.path.join(ckpt_path, "optimizer")
    if os.path.exists(optim_path):
        optimizer_state = {"optimizer": FSDP.optim_state_dict(
            model=model_map[transformer_names[0]]["fsdp_transformer"],
            optim=optimizer,
        )}
        
        load_state_dict(
            state_dict=optimizer_state,
            storage_reader=torch.distributed.checkpoint.FileSystemReader(optim_path),
        )
        
        FSDP.optim_state_dict_to_load(
            model=model_map[transformer_names[0]]["fsdp_transformer"],
            optim=optimizer,
            optim_state_dict=optimizer_state["optimizer"],
        )
        
        print0(f"[FSDP2] Loaded sharded optimizer state")
    
    # Load scheduler and step
    training_state_path = os.path.join(ckpt_path, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        scheduler.load_state_dict(training_state["scheduler"])
        step = int(training_state.get("step", 0))
        print0(f"[FSDP2] Loaded training state from step {step}")
    else:
        step = 0
    
    if dist.is_initialized():
        dist.barrier()
    
    return step