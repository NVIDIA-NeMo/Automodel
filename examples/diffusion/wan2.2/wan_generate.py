# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

from nemo_automodel._diffusers import NeMoAutoDiffusionPipeline
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 T2V FSDP2 generation")
    default_prompt = (
        "The video begins with a close-up of a white bowl filled with shredded coleslaw, "
        "which has a mix of purple cabbage and white cabbage, and is garnished with a sprinkle "
        "of seasoning. The coleslaw is placed on a wooden cutting board. As the video progresses, "
        "the camera pans to the right, revealing a burger with a sesame seed bun, a beef patty, "
        "melted yellow cheese, slices of red tomato, and crispy bacon."
    )

    parser.add_argument("--prompt", type=str, default=default_prompt, help="Text prompt for generation")
    parser.add_argument("--height", type=int, default=480, help="Output video height")
    parser.add_argument("--width", type=int, default=848, help="Output video width")
    parser.add_argument("--num-frames", type=int, default=111, help="Number of frames to generate")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG scale for main guidance")
    parser.add_argument("--guidance-scale-2", type=float, default=3.0, help="CFG scale for secondary guidance")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Number of diffusion steps")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for output video")
    parser.add_argument("--output", type=str, default="t2v_fsdp2_rank0.mp4", help="Output video filename")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (dp rank added)")
    # Parallelism sizes
    parser.add_argument(
        "--tp-size",
        type=int,
        default=8,
        help="Tensor-parallel group size",
    )
    parser.add_argument(
        "--cp-size",
        type=int,
        default=1,
        help="Context-parallel group size",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="Pipeline-parallel group size",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data-parallel group size",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    initialize_distributed(backend="nccl", timeout_minutes=10)
    setup_logging()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    bf16 = torch.bfloat16

    # Configuration for TP+CP+PP+DP
    tp_size = args.tp_size
    cp_size = args.cp_size
    pp_size = args.pp_size
    dp_size = args.dp_size
    dp_rank = local_rank // (tp_size * cp_size * pp_size)

    # -------- Load pipeline --------
    logging.info("[Loading] Loading VAE and pipeline...")
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16
    )
    # Build per-component managers mapping
    fsdp2_manager = FSDP2Manager(
        dp_size=dp_size,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        backend="nccl",
        world_size=world_size,
        use_hf_tp_plan=False,
    )

    # Wan pipelines typically have components like: 'vae', 'text_encoder', 'image_encoder', 'transformer', 'transformer_2'
    # Parallelize only the heavy transformer components
    parallel_scheme = {}
    for name in ("transformer", "transformer_2"):
        parallel_scheme[name] = fsdp2_manager

    # Build pipeline with Automodel's parallelizing pipeline
    pipe = NeMoAutoDiffusionPipeline.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        vae=vae,
        torch_dtype=bf16,
        device=device,
        parallel_scheme=parallel_scheme,
    )
    logging.info("[Setup] Pipeline loaded and parallelized via NeMoAutoDiffusionPipeline")
    dist.barrier()

    # -------- Inference --------
    logging.info("[Inference] Starting distributed inference...")
    torch.manual_seed(args.seed + dp_rank)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=bf16):
        out = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            num_inference_steps=args.num_inference_steps,
        ).frames[0]

    if dist.get_rank() == 0:
        export_to_video(out, args.output, fps=args.fps)
        logging.info(f"[Inference] Saved {args.output}")

    dist.barrier()
    logging.info(
        f"[Complete] Automodel FSDP2 inference completed! TP={tp_size}, CP={cp_size}, PP={pp_size}, DP={dp_size}"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
