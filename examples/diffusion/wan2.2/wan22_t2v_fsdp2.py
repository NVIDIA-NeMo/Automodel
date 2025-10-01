# wan22_t2v_fsdp2.py
# FSDP2 + optional TP for Wan 2.2 video generation using Automodel's FSDP2 utilities

import logging
import os
import sys
import warnings
from typing import Dict

import torch
from torch.distributed.tensor.placement_types import Shard
import torch.distributed as dist
import torch.nn as nn
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.parallelizer import fsdp2_strategy_parallelize


logger = logging.getLogger(__name__)


TP_COMPATIBLE_TYPES = (nn.Linear, nn.Embedding)


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


def setup_dist() -> int:
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
    for _, b in module.named_buffers(recurse=True):
        if b.dtype.is_floating_point:
            b.data = b.to(dtype)


def assert_uniform_dtype(module: nn.Module, want: torch.dtype, name: str):
    dtypes = {p.dtype for p in module.parameters() if p.dtype.is_floating_point}
    assert dtypes == {want}, f"{name}: param dtypes not uniform {want}: {dtypes}"

class DebugRowwiseParallel(RowwiseParallel):
    def _apply(self, module, device_mesh):
        print(f"RowwiseParallel._apply called with module type: {type(module)}")
        print(f"Module: {module}")
        print(f"Is Linear: {isinstance(module, torch.nn.Linear)}")
        print(f"Is Embedding: {isinstance(module, torch.nn.Embedding)}")
        return super()._apply(module, device_mesh)



# ---------------------- Tensor Parallel Plan Helpers ---------------------------


def build_wan_tp_plan(transformer: nn.Module) -> Dict[str, object]:
    """Build TP plan for Wan transformer: shard FFN with Colwise/Rowwise pattern.

    Applies ColwiseParallel to FFN expand projection and RowwiseParallel to 
    FFN down projection and proj_out to properly handle tensor parallel sharding.
    """
    plan = {}
    
    # Explicitly add ColwiseParallel and RowwiseParallel for each block
    # Must list all blocks individually - wildcards don't work with nested paths
    for i in range(40):  # 40 blocks total (0-39)
        plan[f"blocks.{i}.ffn.net.0.proj"] = ColwiseParallel()
        plan[f"blocks.{i}.ffn.net.2"] = DebugRowwiseParallel()
    
    # Add the final projection layer
    plan["proj_out"] = DebugRowwiseParallel()
    
    return plan


def apply_fsdp2_to_transformer(
    transformer: nn.Module,
    manager: FSDP2Manager,
    tp_plan: Dict[str, object],
) -> nn.Module:
    """Apply TP (optional) + FSDP2 sharding via Automodel utilities."""

    tp_mesh: DeviceMesh = manager.device_mesh["tp"]
    if tp_mesh.size() <= 1:
        print0("[FSDP2] TP mesh size <= 1, skipping TP parallelization")
        transformer.eval()
        return transformer

    print0(f"!!!! [FSDP2] Applying TP to transformer")

    # Original TP plan
    parallelize_module(transformer, tp_mesh, tp_plan)

    # # Verbose approach
    # for i, block in enumerate(transformer.blocks):
    #     # FFN expand projection: blocks.{i}.ffn.net.0.proj
    #     gelu_module = block.ffn.net[0]
    #     if hasattr(gelu_module, 'proj') and isinstance(gelu_module.proj, nn.Linear):
    #         print0(f"  Applying ColwiseParallel to blocks.{i}.ffn.net.0.proj")
    #         gelu_module.proj = parallelize_module(
    #             gelu_module.proj, 
    #             tp_mesh, 
    #             {"": ColwiseParallel()}
    #         )
        
    #     # FFN down projection: blocks.{i}.ffn.net.2
    #     down_proj = block.ffn.net[2]
    #     if isinstance(down_proj, nn.Linear):
    #         print0(f"  Applying RowwiseParallel to blocks.{i}.ffn.net.2")
    #         block.ffn.net[2] = parallelize_module(
    #             down_proj,
    #             tp_mesh,
    #             {"": RowwiseParallel()}
    #         )
    
    # # Final projection layer
    # if hasattr(transformer, 'proj_out') and isinstance(transformer.proj_out, nn.Linear):
    #     print0("  Applying RowwiseParallel to proj_out")
    #     transformer.proj_out = parallelize_module(
    #         transformer.proj_out,
    #         tp_mesh,
    #         {"": RowwiseParallel()}
    #     )

    transformer.eval()
    print0("[FSDP2] Applied TP parallelization to transformer")
    return transformer


# ----------------------------------- main -------------------------------------


def main():
    local_rank = setup_dist()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    bf16 = torch.bfloat16

    # Configuration for TP+FSDP2/DP
    tp_size = int(os.environ.get("TP_SIZE", "8"))
    cp_size = int(os.environ.get("CP_SIZE", "1"))
    pp_size = int(os.environ.get("PP_SIZE", "1"))

    assert tp_size >= 1, "TP_SIZE must be >= 1"
    assert cp_size >= 1, "CP_SIZE must be >= 1"
    assert pp_size >= 1, "PP_SIZE must be >= 1"

    total_parallel_ranks = tp_size * cp_size * pp_size
    assert (
        world_size % total_parallel_ranks == 0
    ), f"World size {world_size} must be divisible by TP*CP*PP ({total_parallel_ranks})"

    dp_size = world_size // total_parallel_ranks

    print0(
        f"[Config] World size: {world_size}, TP size: {tp_size}, CP size: {cp_size}, PP size: {pp_size}, DP size: {dp_size}"
    )

    tp_rank = dist.get_rank() % tp_size
    dp_rank = dist.get_rank() // tp_size
    print0(f"[Ranks] Global rank {dist.get_rank()}: TP rank {tp_rank}, DP rank {dp_rank}")

    # -------- Load pipeline --------
    print0("[Loading] Loading VAE and pipeline...")
    # Add FSDP logic here in this method
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32
    )
    #Subclass this and add all parallelization logic here
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers", vae=vae, torch_dtype=torch.float32
    )
    print0("[Loaded] VAE and pipeline...")
    print0("[Loaded] Casting and moving to device...")
    # Setup VAE and encoders
    cast_floating_params_and_buffers_(pipe.vae, bf16)
    pipe.vae.to(device=device, dtype=bf16)
    print0("Finished casting and moving to device for VAE...")
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        cast_floating_params_and_buffers_(pipe.text_encoder, bf16)
        pipe.text_encoder.to(device=device, dtype=bf16)
    print0("Finished casting and moving to device for text encoder...")
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        cast_floating_params_and_buffers_(pipe.image_encoder, bf16)
        pipe.image_encoder.to(device=device, dtype=bf16)
    print0("Finished casting and moving to device for image encoder...")
    # -------- Setup Automodel FSDP2 manager --------
    print0("[Setup] Setting up FSDP2 manager...")

    fsdp2_manager = FSDP2Manager(
        dp_size=dp_size,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        backend="nccl",
        world_size=world_size,
    )
    print0("[Setup] Finished setting up FSDP2 manager...")
    # -------- Apply TP + FSDP2 to transformers --------
    print0("[Setup] Applying TP + FSDP2 to transformers...")
    transformer_names = [n for n in ("transformer", "transformer_2") if hasattr(pipe, n)]
    if not transformer_names:
        raise RuntimeError("Wan2.2 transformers not found.")

    print0(f"[FSDP2] Applying Automodel FSDP2 to: {transformer_names}")

    for name in transformer_names:
        transformer = getattr(pipe, name)

        # Cast and move to device prior to sharding
        cast_floating_params_and_buffers_(transformer, bf16)
        transformer.to(device=device, dtype=bf16)

        if is_main_process():
            assert_uniform_dtype(transformer, bf16, name)

        tp_plan = build_wan_tp_plan(transformer) if tp_size > 1 else {}
        transformer_tp = apply_fsdp2_to_transformer(transformer, fsdp2_manager, tp_plan)

        setattr(pipe, name, transformer_tp)
        print0(f"[FSDP2] Completed Automodel FSDP2 parallelization for {name}")
    print0("[Setup] Finished applying TP + FSDP2 to transformers...")
    dist.barrier()
    print0("[FSDP2] All transformers successfully parallelized with Automodel FSDP2!")

    # -------- Inference --------
    print0("[Inference] Starting distributed inference...")
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

    if is_main_process():
        export_to_video(out, "t2v_fsdp2_rank0.mp4", fps=24)
        print0("[Inference] Saved t2v_fsdp2_rank0.mp4")

    dist.barrier()
    print0(
        f"[Complete] Automodel FSDP2 inference completed! TP={tp_size}, CP={cp_size}, PP={pp_size}, DP={dp_size}"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

