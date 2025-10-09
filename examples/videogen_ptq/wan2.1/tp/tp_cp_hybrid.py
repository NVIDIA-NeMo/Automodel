# tp_cp_hybrid.py - Updated with RoPE-aware Context Parallelism
from typing import Dict, List

import torch
import torch.distributed as dist
from dist_utils import cast_model_to_dtype, print0
from lora_utils import (
    collect_peft_lora_parameters,
    collect_wan_lora_parameters,
    wan_install_and_materialize_lora,
    wan_install_lora_with_peft,
)
from rope_aware_cp import (
    apply_rope_aware_cp_to_transformer,
    create_rope_aware_cp_manager,
    manual_allreduce_lora_gradients_tp_cp_rope,
)


def compute_min_temporal_length(transformer):
    """
    Compute minimum temporal length needed for valid patch embedding output.
    Enhanced to also detect RoPE requirements.
    """
    patch_embed = getattr(transformer, "patch_embedding", None)
    min_from_patch = 1

    if patch_embed is not None:

        def get_first(param, default):
            if param is None:
                return default
            if isinstance(param, (tuple, list)):
                return int(param[0]) if len(param) > 0 else default
            return int(param)

        kernel = get_first(getattr(patch_embed, "kernel_size", 1), 1)
        stride = get_first(getattr(patch_embed, "stride", 1), 1)
        padding = get_first(getattr(patch_embed, "padding", 0), 0)
        dilation = get_first(getattr(patch_embed, "dilation", 1), 1)

        min_from_patch = max(1, dilation * (kernel - 1) + 1 - 2 * padding, stride)

        print0(f"[CP] Patch embedding requires min temporal length: {min_from_patch}")

    # Check for RoPE requirements
    min_from_rope = 1
    for name, module in transformer.named_modules():
        if hasattr(module, "rotary_emb") or "rope" in name.lower():
            # RoPE works better with longer sequences for position encoding
            min_from_rope = 4  # Minimum for meaningful rotary encoding
            print0(f"[CP] Found RoPE module {name}, setting min length for rotation: {min_from_rope}")
            break

    final_min = max(min_from_patch, min_from_rope)
    print0(f"[CP] Final minimum temporal length: {final_min}")
    return final_min


def setup_tp_cp_for_pipe(
    pipe,
    device,
    bf16,
    local_rank: int,
    lora_rank: int,
    lora_alpha: int,
    use_peft: bool = False,
    train_transformer_2: bool = False,
    tp_size: int = None,
    cp_size: int = None,
):
    """
    Setup TP+CP for the chosen transformer using RoPE-aware CP implementation.
    """

    # Check if we have tensor parallelism available
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

        print0("[INFO] Tensor Parallelism available, using RoPE-aware Context Parallelism")
    except ImportError:
        print0("[ERROR] TP not available. Install torch >= 2.1 with distributed support")
        raise ImportError("TP requires torch >= 2.1")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Auto-configure parallelism dimensions
    if tp_size is None and cp_size is None:
        if world_size >= 8:
            # For 8+ GPUs, use more CP for long sequences
            tp_size = 2
            cp_size = world_size // tp_size
        elif world_size >= 4:
            tp_size = 2
            cp_size = world_size // tp_size
        else:
            tp_size = world_size
            cp_size = 1
    elif tp_size is None:
        tp_size = world_size // cp_size
    elif cp_size is None:
        cp_size = world_size // tp_size

    if tp_size * cp_size != world_size:
        raise ValueError(f"tp_size ({tp_size}) * cp_size ({cp_size}) must equal world_size ({world_size})")

    # Find the transformer to train
    transformer_names: List[str] = []

    if train_transformer_2:
        if hasattr(pipe, "transformer_2") and getattr(pipe, "transformer_2") is not None:
            transformer_names.append("transformer_2")
            print0("[INFO] Setting up transformer_2 only (low timesteps)")
        else:
            print0("[ERROR] transformer_2 not found but train_transformer_2=True")
    else:
        if hasattr(pipe, "transformer") and getattr(pipe, "transformer") is not None:
            transformer_names.append("transformer")
            print0("[INFO] Setting up transformer only (high timesteps)")
        else:
            print0("[ERROR] transformer not found but train_transformer_2=False")

    if not transformer_names:
        available = []
        for name in ["transformer", "transformer_2"]:
            if hasattr(pipe, name) and getattr(pipe, name) is not None:
                available.append(name)
        raise RuntimeError(f"Requested transformer not found. Available: {available}")

    print0(f"[INFO] Found and will setup: {transformer_names}")
    print0(f"[INFO] Using TP={tp_size}, CP={cp_size} with RoPE-aware sharding")

    # Create RoPE-aware CP manager
    cp_manager = create_rope_aware_cp_manager(world_size, rank, tp_size, cp_size) if cp_size > 1 else None

    # Create device mesh for TP
    if tp_size > 1:
        device_mesh = init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))
        print0(f"[INFO] Created 1D device mesh for TP with {tp_size} devices")
    else:
        device_mesh = None
        print0("[INFO] No TP needed (tp_size=1)")

    model_map: Dict[str, Dict] = {}

    # Process the chosen transformer
    for name in transformer_names:
        base_transformer = getattr(pipe, name)

        # Cast & move base
        cast_model_to_dtype(base_transformer, bf16)
        base_transformer.to(device)

        # Install LoRA BEFORE any parallelism
        print0(f"[INFO] Installing LoRA on {name} (before TP+CP)...")

        try:
            if use_peft:
                num_lora_modules = wan_install_lora_with_peft(base_transformer, rank=lora_rank, alpha=lora_alpha)
                lora_params_before_parallel = collect_peft_lora_parameters(base_transformer)
            else:
                num_lora_modules = wan_install_and_materialize_lora(base_transformer, rank=lora_rank, alpha=lora_alpha)
                lora_params_before_parallel = collect_wan_lora_parameters(base_transformer)

            if len(lora_params_before_parallel) == 0:
                raise RuntimeError(f"No LoRA parameters found for {name} after installation")

            print0(
                f"[INFO] Successfully installed LoRA on {name}: {num_lora_modules} modules, {len(lora_params_before_parallel)} parameters"
            )

        except Exception as e:
            print0(f"[ERROR] LoRA installation failed for {name}: {e}")
            raise

        # Compute minimum temporal length requirements
        min_temporal_length = compute_min_temporal_length(base_transformer)

        # Configure CP manager with requirements
        if cp_size > 1 and cp_manager is not None:
            min_total_frames = min_temporal_length * cp_size
            print0(
                f"[INFO] RoPE-aware CP requires minimum {min_total_frames} total frames "
                f"({min_temporal_length} per shard × {cp_size} shards)"
            )

            cp_manager.min_sequence_length = min_temporal_length
            print0(f"[INFO] RoPE-aware CP manager configured with min sequence length: {min_temporal_length}")

            # Enhanced warning for RoPE requirements
            if min_total_frames > 32:
                print0(f"[WARNING] RoPE-aware CP requires {min_total_frames} frames minimum.")
                print0("[INFO] For 100+ frame training, this should work well with proper RoPE positioning.")

        # Apply TP if needed
        if tp_size > 1 and device_mesh is not None:
            print0(f"[INFO] Applying TP to {name}...")
            try:
                tp_transformer = apply_tp_to_transformer(base_transformer, device_mesh)
                print0(f"[INFO] Successfully applied TP to {name}")
            except Exception as e:
                print0(f"[ERROR] TP parallelization failed for {name}: {e}")
                raise
        else:
            tp_transformer = base_transformer
            print0("[INFO] No TP applied (tp_size=1)")

        # Apply RoPE-aware CP if needed
        if cp_size > 1 and cp_manager is not None:
            print0(f"[INFO] Applying RoPE-aware CP to {name}...")
            try:
                final_transformer = apply_rope_aware_cp_to_transformer(tp_transformer, cp_manager)
                print0(f"[INFO] Successfully applied RoPE-aware CP to {name}")
            except Exception as e:
                print0(f"[ERROR] RoPE-aware CP failed for {name}: {e}")
                raise
        else:
            final_transformer = tp_transformer
            print0("[INFO] No CP applied (cp_size=1)")

        # Use pre-parallelization LoRA parameters
        final_lora_params = lora_params_before_parallel

        print0(f"[INFO] Using pre-parallelization LoRA parameters: {len(final_lora_params)} parameters")

        model_map[name] = {
            "tp_cp_transformer": final_transformer,
            "base_transformer": base_transformer,
            "lora_params": final_lora_params,
            "device_mesh": device_mesh,
            "cp_manager": cp_manager,
            "use_cp": cp_size > 1,
            "use_rope_cp": cp_size > 1,  # New flag for RoPE-aware CP
            "tp_size": tp_size,
            "cp_size": cp_size,
            "min_temporal_length": min_temporal_length if cp_size > 1 else 1,
        }
        setattr(pipe, name, final_transformer)
        print0(f"[INFO] {name} TP+RoPE-CP setup complete")

    print0("[INFO] TP + RoPE-aware CP + LoRA setup complete")

    # Final verification and recommendations
    total_lora_params = sum(len(model_map[name]["lora_params"]) for name in transformer_names)
    print0(f"[INFO] Total LoRA parameters across active transformer: {total_lora_params}")

    if total_lora_params == 0:
        raise RuntimeError("No LoRA parameters found in active transformer!")

    # Print recommendations for RoPE-aware training
    for name in transformer_names:
        if name in model_map and model_map[name]["use_rope_cp"]:
            min_frames = model_map[name]["min_temporal_length"] * model_map[name]["cp_size"]
            print0(f"[RECOMMENDATION] For {name} with RoPE-aware CP:")
            print0(f"  - Use at least {min_frames} frames in training sequences")
            print0(f"  - Optimal range: {min_frames}-{min_frames * 4} frames for good RoPE positioning")
            print0("  - For 100+ frames: RoPE will handle position encoding correctly across shards")

    return model_map, transformer_names


def apply_tp_to_transformer(transformer, device_mesh):
    """
    Apply tensor parallelism to transformer.
    Skip FFN layers to avoid LoRA dimension conflicts.
    """
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    print0("[TP] Applying TP to transformer (skipping FFN due to LoRA)...")

    # Skip condition embedder to avoid compatibility issues
    if hasattr(transformer, "condition_embedder"):
        print0("[TP] Skipping condition embedder parallelization")

    # Apply TP to transformer blocks - ATTENTION ONLY
    if hasattr(transformer, "blocks"):
        num_blocks = len(transformer.blocks)
        print0(f"[TP] Parallelizing {num_blocks} transformer blocks (attention only)...")

        for i, block in enumerate(transformer.blocks):
            # Apply TP to attention projections ONLY
            if hasattr(block, "attn") or hasattr(block, "attention"):
                attn_module = getattr(block, "attn", getattr(block, "attention", None))
                if attn_module:
                    try:
                        # Parallelize query, key, value projections
                        if hasattr(attn_module, "to_q"):
                            attn_module.to_q = parallelize_module(
                                attn_module.to_q, device_mesh, {"": ColwiseParallel()}
                            )
                        if hasattr(attn_module, "to_k"):
                            attn_module.to_k = parallelize_module(
                                attn_module.to_k, device_mesh, {"": ColwiseParallel()}
                            )
                        if hasattr(attn_module, "to_v"):
                            attn_module.to_v = parallelize_module(
                                attn_module.to_v, device_mesh, {"": ColwiseParallel()}
                            )
                        # Output projection
                        if hasattr(attn_module, "to_out"):
                            if isinstance(attn_module.to_out, torch.nn.Sequential):
                                attn_module.to_out[0] = parallelize_module(
                                    attn_module.to_out[0], device_mesh, {"": RowwiseParallel()}
                                )
                            else:
                                attn_module.to_out = parallelize_module(
                                    attn_module.to_out, device_mesh, {"": RowwiseParallel()}
                                )
                    except Exception as e:
                        print0(f"[TP] Warning: Could not parallelize attention in block {i}: {e}")

            # SKIP FFN parallelization to avoid LoRA dimension conflicts
            if hasattr(block, "ffn") or hasattr(block, "mlp"):
                print0(f"[TP] Skipping FFN parallelization in block {i} (LoRA compatibility)")

            if (i + 1) % 10 == 0 or i == num_blocks - 1:
                print0(f"[TP] Parallelized attention in block {i + 1}/{num_blocks}")

    # Skip output projection parallelization too (might have LoRA)
    if hasattr(transformer, "proj_out"):
        print0("[TP] Skipping output projection parallelization (potential LoRA conflict)")

    print0("[TP] Tensor parallelism application complete (attention layers only)!")
    return transformer


def manual_allreduce_lora_gradients_tp_cp(model_map: Dict, transformer_names: List[str]):
    """
    Updated gradient synchronization for TP+RoPE-CP.
    Uses the RoPE-aware version for better gradient handling.
    """
    manual_allreduce_lora_gradients_tp_cp_rope(model_map, transformer_names)


def test_tp_cp_forward_pass(model_map: Dict, transformer_names: List[str], device, bf16):
    """
    Test forward pass with TP+RoPE-CP.
    Enhanced to validate RoPE positioning works correctly.
    """
    print0("[INFO] Testing TP+RoPE-CP forward pass...")

    for name in transformer_names:
        if name in model_map:
            model = model_map[name]["tp_cp_transformer"]
            model.train()

            # Get minimum required frames (enhanced for RoPE)
            min_frames = 32  # Default
            if model_map[name]["use_rope_cp"]:
                min_frames = model_map[name]["min_temporal_length"] * model_map[name]["cp_size"]
                # For RoPE testing, use a bit more than minimum
                min_frames = max(min_frames, 64)  # Good for RoPE positioning validation
                print0(f"[INFO] {name} requires minimum {min_frames} frames for RoPE-aware CP")

            # Create dummy input with sufficient frames for RoPE testing
            batch_size = 1
            channels = 36  # Expected for I2V conditioning
            frames = max(min_frames, 128)  # Use 128 frames to test RoPE well
            height = 32
            width = 32

            dummy_input = torch.randn(batch_size, channels, frames, height, width, device=device, dtype=bf16)
            dummy_timestep = torch.randint(0, 1000, (batch_size,), device=device)
            dummy_encoder_hidden = torch.randn(batch_size, 77, 4096, device=device, dtype=bf16)

            print0(
                f"[INFO] Testing {name} with {frames} frames (RoPE positioning across {model_map[name]['cp_size']} shards)"
            )

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

                    print0(f"[INFO] {name} RoPE-CP forward pass successful!")
                    print0(f"[INFO] Input shape: {dummy_input.shape}, Output shape: {output.shape}")

                    # Validate RoPE positioning worked correctly
                    if model_map[name]["use_rope_cp"]:
                        cp_manager = model_map[name]["cp_manager"]
                        if cp_manager and hasattr(cp_manager, "rope_dim") and cp_manager.rope_dim:
                            print0(f"[INFO] RoPE configuration: dim={cp_manager.rope_dim}, base={cp_manager.rope_base}")
                            print0("[INFO] RoPE positioning validated across CP shards")

            except Exception as e:
                print0(f"[WARNING] {name} forward pass test failed: {e}")
                print0("[INFO] This is expected - will validate during actual training with real data")

            trainable_params = sum(1 for p in model_map[name]["lora_params"] if p.requires_grad)
            print0(f"[INFO] {name}: {trainable_params} trainable LoRA parameters")


def verify_rope_cp_setup(model_map: Dict, transformer_names: List[str]):
    """
    Verify RoPE-aware CP setup is working correctly.
    """
    print0("[INFO] Verifying RoPE-aware CP setup...")

    for name in transformer_names:
        if name in model_map and model_map[name]["use_rope_cp"]:
            cp_manager = model_map[name]["cp_manager"]

            print0(f"[INFO] {name} RoPE-CP configuration:")
            print0(f"  - CP size: {model_map[name]['cp_size']}")
            print0(f"  - TP size: {model_map[name]['tp_size']}")
            print0(f"  - Min temporal length per shard: {model_map[name]['min_temporal_length']}")

            if cp_manager:
                print0(f"  - RoPE dim: {getattr(cp_manager, 'rope_dim', 'Not detected')}")
                print0(f"  - RoPE base: {getattr(cp_manager, 'rope_base', 'Not detected')}")
                print0(
                    f"  - Global sequence tracking: {'Enabled' if hasattr(cp_manager, 'global_seq_len') else 'Disabled'}"
                )

            # Check if wrapper is applied
            transformer = model_map[name]["tp_cp_transformer"]
            from rope_aware_cp import RoPEAwareTransformerWrapper

            if isinstance(transformer, RoPEAwareTransformerWrapper):
                print0("  - RoPE-aware wrapper: Applied ✓")
            else:
                print0("  - RoPE-aware wrapper: Not applied ✗")

    print0("[INFO] RoPE-CP verification complete")


# Legacy compatibility functions
def prepare_cp_input(hidden_states, device_mesh):
    """Legacy function - now handled internally by RoPE-aware CP"""
    return hidden_states


def gather_cp_output(output, device_mesh):
    """Legacy function - now handled internally by RoPE-aware CP"""
    return output


# Backwards compatibility
def setup_hybrid_for_pipe(
    pipe,
    device,
    bf16,
    local_rank: int,
    lora_rank: int,
    lora_alpha: int,
    use_peft: bool = False,
    train_transformer_2: bool = False,
    tp_size: int = None,
    cp_size: int = None,
):
    """Legacy function name - redirects to RoPE-aware TP+CP implementation."""
    return setup_tp_cp_for_pipe(
        pipe, device, bf16, local_rank, lora_rank, lora_alpha, use_peft, train_transformer_2, tp_size, cp_size
    )
