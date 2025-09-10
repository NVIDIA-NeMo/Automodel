# wan22_t2v_fsdp2_dp_bf16_dual_rank0.py
# DP-only over 8 GPUs, BF16 everywhere, wrap BOTH Wan 2.2 transformers.
# Prints/logs only on rank 0.

import logging
import os
import sys
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager

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
        # Silence stdout on non-zero ranks (keep stderr for errors)
        sys.stdout = open(os.devnull, "w")


# ------------------------------- setup helpers --------------------------------


def setup_dist():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    configure_logging_quiet_others()

    # Force BF16 path; avoid TF32 upcasts/silent fp32 math
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return local_rank


def cast_floating_params_and_buffers_(module: nn.Module, dtype: torch.dtype):
    """Cast ALL floating params & buffers (incl. biases, norms, scale tables) to dtype."""
    for p in module.parameters(recurse=True):
        if p.dtype.is_floating_point:
            p.data = p.data.to(dtype)
    for name, b in module.named_buffers(recurse=True):
        if b.dtype.is_floating_point:
            setattr(module, name, b.to(dtype))


def assert_uniform_dtype(module: nn.Module, want: torch.dtype, name: str):
    dtypes = {p.dtype for p in module.parameters() if p.is_floating_point()}
    assert dtypes == {want}, f"{name}: param dtypes not uniform {want}: {dtypes}"


# ----------------------------------- main -------------------------------------


def main():
    local_rank = setup_dist()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    bf16 = torch.bfloat16

    # -------- Load pipeline on CPU --------
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers", vae=vae, torch_dtype=torch.float32)

    # Disable slow slice modes
    # try: pipe.disable_attention_slicing()
    # except Exception: pass
    # try: pipe.vae.disable_slicing()
    # except Exception: pass

    # Put encoders/vae on our local GPU in BF16 (flip VAE back to fp32 if you see quality issues)
    cast_floating_params_and_buffers_(pipe.vae, bf16)
    pipe.vae.to(device=device, dtype=bf16)
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        cast_floating_params_and_buffers_(pipe.text_encoder, bf16)
        pipe.text_encoder.to(device=device, dtype=bf16)
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        cast_floating_params_and_buffers_(pipe.image_encoder, bf16)
        pipe.image_encoder.to(device=device, dtype=bf16)

    # -------- Collect BOTH Wan 2.2 transformers and cast to BF16 on CPU --------
    transformer_names = [n for n in ("transformer", "transformer_2") if hasattr(pipe, n)]
    if not transformer_names:
        raise RuntimeError("Wan2.2 transformers not found on pipeline (expected transformer/transformer_2).")
    print0(f"[info] Wrapping transformers: {transformer_names}")

    for name in transformer_names:
        mod = getattr(pipe, name)
        cast_floating_params_and_buffers_(mod, bf16)
        if is_main_process():
            assert_uniform_dtype(mod, bf16, name)

    # -------- FSDP2 DP-only (BF16) and wrap BOTH transformers --------
    mp = MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=bf16, output_dtype=bf16, cast_forward_inputs=True)
    mgr = FSDP2Manager(
        dp_size=world_size,  # shard across all ranks
        dp_replicate_size=1,  # pure sharding, no replicas
        tp_size=1,
        cp_size=1,  # DP-only
        sequence_parallel=False,
        mp_policy=mp,
        backend="nccl",
        world_size=world_size,
    )

    for name in transformer_names:
        wrapped = mgr.parallelize(getattr(pipe, name))
        wrapped.eval()
        setattr(pipe, name, wrapped)

    # -------- Inference (identical on all ranks) --------
    torch.manual_seed(0)

    # 480p (16:9, codec/vae-friendly)
    height, width = 480, 848

    prompt = (
        "The video begins with a close-up of a white bowl filled with shredded coleslaw, "
        "which has a mix of purple cabbage and white cabbage, and is garnished with a sprinkle "
        "of seasoning. The coleslaw is placed on a wooden cutting board. As the video progresses, "
        "the camera pans to the right, revealing a burger with a sesame seed bun, a beef patty, "
        "melted yellow cheese, slices of red tomato, and crispy bacon. The burger is accompanied "
        "by a side salad that includes green lettuce and slices of cucumber. The background shows "
        "a red and white striped tablecloth, and there are condiments like ketchup and mustard "
        "visible in the periphery"
    )

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=bf16):
        out = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=111,
            guidance_scale=4.0,
            guidance_scale_2=3.0,
            num_inference_steps=40,
        ).frames[0]

    if is_main_process():
        export_to_video(out, "t2v_out.mp4", fps=24)
        print0("[info] Saved t2v_out.mp4")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()


# run with torchrun --nproc-per-node=8 gen_wan_multiGPU.py
