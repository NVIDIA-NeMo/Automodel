# wan22_t2v_tp_dp_pytorch.py
# TP+DP for Wan 2.2 using PyTorch's built-in tensor parallel classes
# More reliable than custom implementation

import logging
import os
import sys
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

# --------------------------- rank-0 printing/logging ---------------------------


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def configure_logging_quiet_others():
    level = logging.INFO if is_main_process() else logging.ERROR
    logging.basicConfig(level=level)
    if dist.is_initialized() and dist.get_rank() != 0:
        warnings.filterwarnings("ignore")
        sys.stdout = open(os.devnull, "w")


# ------------------------------- setup helpers --------------------------------


def setup_dist():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    configure_logging_quiet_others()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return local_rank


def cast_floating_params_and_buffers_(module: nn.Module, dtype: torch.dtype):
    """Cast ALL floating params & buffers to dtype."""
    for p in module.parameters(recurse=True):
        if p.dtype.is_floating_point:
            p.data = p.data.to(dtype)
    for name, b in module.named_buffers(recurse=True):
        if b.dtype.is_floating_point:
            b.data = b.to(dtype)


def assert_uniform_dtype(module: nn.Module, want: torch.dtype, name: str):
    dtypes = {p.dtype for p in module.parameters() if p.dtype.is_floating_point}
    assert dtypes == {want}, f"{name}: param dtypes not uniform {want}: {dtypes}"


# ---------------------- Tensor Parallel Application Functions -----------------------


def setup_tp_device_mesh(tp_size: int):
    """Setup 1D device mesh for tensor parallelism within each TP group"""
    return init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))


def apply_tp_to_ffn(ffn_module, tp_mesh):
    """Apply TP to FFN using PyTorch's built-in classes"""
    return parallelize_module(
        ffn_module,
        tp_mesh,
        parallelize_plan={
            "net.0.proj": ColwiseParallel(),  # GELU projection: 5120 -> 13824
            "net.2": RowwiseParallel(),  # Final projection: 13824 -> 5120
        },
    )


def apply_tp_to_transformer_block(block, tp_mesh):
    """Apply tensor parallelism to a WanTransformerBlock - conservative approach"""

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
    """Apply tensor parallelism to entire WanTransformer3DModel"""
    print0(f"[TP] Applying PyTorch TP to transformer with {len(transformer.blocks)} blocks...")

    # Apply TP to condition embedder (text embedder only)
    transformer.condition_embedder = apply_tp_to_condition_embedder(transformer.condition_embedder, tp_mesh)

    # Apply TP to all transformer blocks
    for i, block in enumerate(transformer.blocks):
        transformer.blocks[i] = apply_tp_to_transformer_block(block, tp_mesh)
        if i % 10 == 0:
            print0(f"[TP] Parallelized block {i}/{len(transformer.blocks)}")

    # Apply TP to output projection
    transformer.proj_out = parallelize_module(
        transformer.proj_out,
        tp_mesh,
        parallelize_plan={"": RowwiseParallel()},  # 5120 -> 64
    )

    print0("[TP] PyTorch TP application complete!")
    return transformer


# ----------------------------------- main -------------------------------------


def main():
    local_rank = setup_dist()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    bf16 = torch.bfloat16

    # Configuration for TP+DP
    tp_size = int(os.environ.get("TP_SIZE", "2"))
    dp_size = world_size // tp_size

    print0(f"[Config] World size: {world_size}, TP size: {tp_size}, DP size: {dp_size}")
    assert world_size % tp_size == 0, f"World size {world_size} must be divisible by TP size {tp_size}"

    # Determine TP and DP ranks
    tp_rank = dist.get_rank() % tp_size
    dp_rank = dist.get_rank() // tp_size

    print0(f"[Ranks] Global rank {dist.get_rank()}: TP rank {tp_rank}, DP rank {dp_rank}")

    # Create TP device mesh for this TP group
    # Each TP group gets its own mesh: [0,1], [2,3], [4,5], [6,7] for tp_size=2
    tp_group_ranks = [dp_rank * tp_size + i for i in range(tp_size)]

    # We need to set up the device mesh within each TP group
    # For this, we create a temporary sub-communicator
    # tp_group = dist.new_group(tp_group_ranks)

    # Initialize device mesh for this TP group
    tp_mesh = setup_tp_device_mesh(tp_size)

    print0(f"[Mesh] Created TP mesh for ranks {tp_group_ranks}: {tp_mesh}")

    # -------- Load pipeline --------
    print0("[Loading] Loading VAE and pipeline...")
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers", vae=vae, torch_dtype=torch.float32)

    # Setup VAE and encoders
    cast_floating_params_and_buffers_(pipe.vae, bf16)
    pipe.vae.to(device=device, dtype=bf16)

    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        cast_floating_params_and_buffers_(pipe.text_encoder, bf16)
        pipe.text_encoder.to(device=device, dtype=bf16)

    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        cast_floating_params_and_buffers_(pipe.image_encoder, bf16)
        pipe.image_encoder.to(device=device, dtype=bf16)

    # -------- Apply TP to transformers --------
    transformer_names = [n for n in ("transformer", "transformer_2") if hasattr(pipe, n)]
    if not transformer_names:
        raise RuntimeError("Wan2.2 transformers not found.")

    print0(f"[TP] Applying PyTorch TP to: {transformer_names}")

    for name in transformer_names:
        transformer = getattr(pipe, name)

        # Cast and move to device
        cast_floating_params_and_buffers_(transformer, bf16)
        transformer.to(device=device, dtype=bf16)

        if is_main_process():
            assert_uniform_dtype(transformer, bf16, name)

        # Apply PyTorch tensor parallelism
        transformer_tp = apply_tp_to_transformer(transformer, tp_mesh)
        transformer_tp.eval()

        setattr(pipe, name, transformer_tp)
        print0(f"[TP] Completed PyTorch TP for {name}")

    dist.barrier()
    print0("[TP] All transformers successfully parallelized with PyTorch TP!")

    # -------- Inference --------
    print0("[Inference] Starting distributed inference...")

    # Different seeds per DP rank for variety
    torch.manual_seed(42 + dp_rank)

    height, width = 480, 848
    prompt = (
        "The video begins with a close-up of a white bowl filled with shredded coleslaw, "
        "which has a mix of purple cabbage and white cabbage, and is garnished with a sprinkle "
        "of seasoning. The coleslaw is placed on a wooden cutting board. As the video progresses, "
        "the camera pans to the right, revealing a burger with a sesame seed bun, a beef patty, "
        "melted yellow cheese, slices of red tomato, and crispy bacon."
    )

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=bf16):
        out = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=111,
            guidance_scale=4.0,
            guidance_scale_2=3.0,
            num_inference_steps=20,
        ).frames[0]

    # Save output from main process
    if is_main_process():
        export_to_video(out, "t2v_pytorch_tp_dp_rank0.mp4", fps=24)
        print0("[Inference] Saved t2v_pytorch_tp_dp_rank0.mp4")

    dist.barrier()
    print0(f"[Complete] PyTorch TP+DP inference completed! TP={tp_size}, DP={dp_size}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# Usage:
# export TP_SIZE=2
# torchrun --nproc-per-node=8 wan22_t2v_tp_dp_pytorch.py
