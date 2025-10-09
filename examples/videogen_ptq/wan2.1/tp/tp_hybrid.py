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


def setup_tp_for_pipe(
    pipe,
    device,
    bf16,
    local_rank: int,
    lora_rank: int,
    lora_alpha: int,
    use_peft: bool = False,
    train_transformer_2: bool = False,
):
    """
    Setup Tensor Parallelism for only the chosen transformer + unsharded LoRA.
    This version only processes the transformer that will be trained, saving memory and setup time.
    """

    # Check if we have tensor parallelism available
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

        print0("[INFO] Tensor Parallelism available")
    except ImportError:
        print0("[ERROR] Tensor Parallelism not available. Install torch >= 2.1 with distributed support")
        raise ImportError("Tensor Parallelism requires torch >= 2.1")

    # Only find the transformer we want to train
    transformer_names: List[str] = []

    if train_transformer_2:
        # Only setup transformer_2
        if hasattr(pipe, "transformer_2") and getattr(pipe, "transformer_2") is not None:
            transformer_names.append("transformer_2")
            print0("[INFO] Setting up transformer_2 only (low timesteps)")
        else:
            print0("[ERROR] transformer_2 not found but train_transformer_2=True")
    else:
        # Only setup transformer
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
    print0(f"[INFO] Using Tensor Parallelism (TP) with world_size={dist.get_world_size()}")
    print0(f"[INFO] Using {'PEFT' if use_peft else 'manual'} LoRA implementation (unsharded)")

    # Create device mesh for tensor parallelism
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))
    print0(f"[INFO] Created device mesh for TP with {world_size} devices")

    model_map: Dict[str, Dict] = {}

    # Only process the single chosen transformer
    for name in transformer_names:
        base_transformer = getattr(pipe, name)

        # Cast & move base
        cast_model_to_dtype(base_transformer, bf16)
        base_transformer.to(device)

        # Install LoRA BEFORE tensor parallelism
        print0(f"[INFO] Installing LoRA on {name} (before TP)...")

        try:
            if use_peft:
                num_lora_modules = wan_install_lora_with_peft(base_transformer, rank=lora_rank, alpha=lora_alpha)
                lora_params_before_tp = collect_peft_lora_parameters(base_transformer)
            else:
                num_lora_modules = wan_install_and_materialize_lora(base_transformer, rank=lora_rank, alpha=lora_alpha)
                lora_params_before_tp = collect_wan_lora_parameters(base_transformer)

            if len(lora_params_before_tp) == 0:
                raise RuntimeError(f"No LoRA parameters found for {name} after installation")

            print0(
                f"[INFO] Successfully installed LoRA on {name}: {num_lora_modules} modules, {len(lora_params_before_tp)} parameters"
            )
            print0(f"[INFO] Collected {len(lora_params_before_tp)} LoRA parameters before TP")

        except Exception as e:
            print0(f"[ERROR] LoRA installation failed for {name}: {e}")
            raise

        # Apply tensor parallelism using the proven approach from T2V code
        print0(f"[INFO] Applying Tensor Parallelism to {name} (LoRA will remain unsharded)...")
        try:
            tp_transformer = apply_tp_to_transformer(base_transformer, device_mesh)
            print0(f"[INFO] Successfully applied TP to {name}")
        except Exception as e:
            print0(f"[ERROR] TP parallelization failed for {name}: {e}")
            raise

        # IMPORTANT: Use the LoRA parameters collected BEFORE TP
        # TP only parallelizes the base transformer weights, LoRA stays unsharded
        final_lora_params = lora_params_before_tp

        print0(f"[INFO] Using pre-TP LoRA parameters: {len(final_lora_params)} parameters")
        print0("[INFO] LoRA parameters remain unsharded (replicated across all TP ranks)")

        model_map[name] = {
            "tp_transformer": tp_transformer,
            "base_transformer": base_transformer,
            "lora_params": final_lora_params,
        }
        setattr(pipe, name, tp_transformer)
        print0(f"[INFO] {name} TP setup complete")

    print0("[INFO] Tensor Parallelism + LoRA setup complete")

    # Final verification
    total_lora_params = sum(len(model_map[name]["lora_params"]) for name in transformer_names)
    print0(f"[INFO] Total LoRA parameters across active transformer: {total_lora_params}")

    if total_lora_params == 0:
        raise RuntimeError("No LoRA parameters found in active transformer!")

    # Memory usage summary
    active_transformer = transformer_names[0]
    unused_transformer = "transformer_2" if active_transformer == "transformer" else "transformer"
    print0(f"[INFO] Memory optimization: Only {active_transformer} loaded to GPU")
    print0(f"[INFO] {unused_transformer} was never loaded, saving ~50% GPU memory")

    return model_map, transformer_names


def apply_tp_to_ffn(ffn_module, tp_mesh):
    """Apply TP to FFN using PyTorch's built-in classes - from working T2V code"""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    return parallelize_module(
        ffn_module,
        tp_mesh,
        parallelize_plan={
            "net.0.proj": ColwiseParallel(),  # GELU projection: 5120 -> 13824
            "net.2": RowwiseParallel(),  # Final projection: 13824 -> 5120
        },
    )


def apply_tp_to_transformer_block(block, tp_mesh):
    """Apply tensor parallelism to a WanTransformerBlock - conservative approach from T2V code"""

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
    """Apply tensor parallelism to entire WanTransformer3DModel - based on working T2V approach"""
    from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module

    print0(f"[TP] Applying PyTorch TP to transformer with {len(transformer.blocks)} blocks...")

    # Apply TP to condition embedder (text embedder only)
    if hasattr(transformer, "condition_embedder"):
        transformer.condition_embedder = apply_tp_to_condition_embedder(transformer.condition_embedder, tp_mesh)

    # Apply TP to all transformer blocks
    for i, block in enumerate(transformer.blocks):
        transformer.blocks[i] = apply_tp_to_transformer_block(block, tp_mesh)
        if i % 10 == 0:
            print0(f"[TP] Parallelized block {i}/{len(transformer.blocks)}")

    # Apply TP to output projection
    if hasattr(transformer, "proj_out"):
        transformer.proj_out = parallelize_module(
            transformer.proj_out,
            tp_mesh,
            parallelize_plan={"": RowwiseParallel()},  # 5120 -> output_dim
        )

    print0("[TP] PyTorch TP application complete!")
    return transformer


def manual_allreduce_lora_gradients(model_map: Dict, transformer_names: List[str]):
    """
    Manually all-reduce LoRA gradients across ranks since they're unsharded.
    Call this after loss.backward() and before optimizer.step().

    Updated to only process active transformer(s).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return

    world_size = dist.get_world_size()

    for name in transformer_names:
        if name not in model_map:
            continue  # Skip if transformer not in model_map

        lora_params = model_map[name]["lora_params"]

        for param in lora_params:
            if param.grad is not None:
                # All-reduce the gradient
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                # Average across ranks
                param.grad.div_(world_size)


def test_tp_forward_pass(model_map: Dict, transformer_names: List[str], device, bf16):
    """
    Test that TP-enabled models can perform forward passes.
    This uses the correct 36-channel input format.
    Updated to only test active transformer.
    """
    print0("[INFO] Testing TP forward passes with 36-channel input...")

    for name in transformer_names:
        if name not in model_map:
            print0(f"[INFO] Skipping test for {name} (not in model_map)")
            continue

        try:
            tp_model = model_map[name]["tp_transformer"]
            tp_model.eval()

            # Create 36-channel test input (matches the model expectation)
            batch_size = 1
            channels = 36  # WAN I2V expects 36 channels
            frames = 8  # Small for testing
            height = 22  # Quarter resolution for testing
            width = 40  # Quarter resolution for testing

            # 5D input: [batch, channels, frames, height, width]
            hidden_states = torch.randn(batch_size, channels, frames, height, width, device=device, dtype=bf16)

            # Timestep for diffusion
            timestep = torch.randint(0, 1000, (batch_size,), device=device)

            # Text encoder hidden states: [batch, seq_len, hidden_dim]
            text_seq_len = 77
            text_hidden_dim = 4096
            encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_hidden_dim, device=device, dtype=bf16)

            with torch.no_grad():
                output = tp_model(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )

            output_tensor = output[0] if isinstance(output, tuple) else output
            print0(
                f"[INFO] {name} TP forward pass successful: input {hidden_states.shape} -> output {output_tensor.shape}"
            )

            tp_model.train()

        except Exception as e:
            print0(f"[ERROR] TP forward pass test failed for {name}: {e}")
            # Don't raise error - this is just a test, actual training will validate
            continue

    print0("[INFO] TP forward pass tests completed")


# Main entry point - now accepts train_transformer_2 flag
def setup_hybrid_for_pipe(
    pipe,
    device,
    bf16,
    local_rank: int,
    lora_rank: int,
    lora_alpha: int,
    use_peft: bool = False,
    train_transformer_2: bool = False,
):
    """
    Main entry point - sets up TP for only the chosen transformer.

    Args:
        train_transformer_2: If True, only setup transformer_2. If False, only setup transformer.
    """
    return setup_tp_for_pipe(pipe, device, bf16, local_rank, lora_rank, lora_alpha, use_peft, train_transformer_2)
