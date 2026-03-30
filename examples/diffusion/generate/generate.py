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

"""
Unified Diffusion Generation Script

Single entry point for generating images and videos from all supported diffusion
models (FLUX, Wan 2.1/2.2, HunyuanVideo). Supports single-GPU and distributed
inference with optional checkpoint loading.

Usage:
    # Single-GPU
    python examples/diffusion/generate/generate.py \
        -c examples/diffusion/generate/configs/generate_wan.yaml

    # Multi-GPU distributed
    torchrun --nproc-per-node=8 \
        examples/diffusion/generate/generate.py \
        -c examples/diffusion/generate/configs/generate_wan_distributed.yaml

    # With checkpoint and custom prompts
    python examples/diffusion/generate/generate.py \
        -c examples/diffusion/generate/configs/generate_wan.yaml \
        --model.checkpoint ./checkpoints/step_1000 \
        --inference.prompts '["A dog running on a beach"]'
"""

import inspect
import logging
from pathlib import Path

import torch
import torch.distributed as dist

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.shared.transformers_patches import patch_t5_layer_norm

logger = logging.getLogger(__name__)

# Pipeline class name -> output type mapping
_PIPELINE_OUTPUT_TYPES = {
    "FluxPipeline": "image",
    "WanPipeline": "video",
    "HunyuanVideoPipeline": "video",
    "HunyuanVideo15Pipeline": "video",
}


def maybe_init_distributed(cfg):
    """Initialize distributed environment if configured.

    Args:
        cfg: Config node with optional `distributed` section.

    Returns:
        DistInfo if distributed is configured, None otherwise.
    """
    dist_cfg = getattr(cfg, "distributed", None)
    if dist_cfg is None:
        return None

    from nemo_automodel.components.distributed.init_utils import initialize_distributed

    backend = getattr(dist_cfg, "backend", "nccl")
    timeout = getattr(dist_cfg, "timeout_minutes", 10)
    dist_info = initialize_distributed(backend=backend, timeout_minutes=timeout)
    logger.info("Distributed initialized: rank=%d, world_size=%d", dist_info.rank, dist_info.world_size)
    return dist_info


def load_pipeline(cfg, dist_info):
    """Load the diffusion pipeline, auto-detecting model type.

    Uses DiffusionPipeline for single-GPU or NeMoAutoDiffusionPipeline for
    distributed inference with parallelization.

    Args:
        cfg: Config node with `model.pretrained_model_name_or_path`.
        dist_info: DistInfo from maybe_init_distributed, or None.

    Returns:
        A diffusers pipeline instance.
    """
    from diffusers import DiffusionPipeline

    model_id = cfg.model.pretrained_model_name_or_path
    dtype_str = getattr(cfg.inference, "dtype", "bfloat16")
    torch_dtype = _resolve_dtype(dtype_str)

    # Apex's FusedRMSNorm doesn't support bf16. Patch T5LayerNorm before loading
    # any pipeline that may use a T5 text encoder (FLUX, HunyuanVideo, etc.).
    if torch_dtype == torch.bfloat16:
        patch_t5_layer_norm()

    if dist_info is not None and hasattr(cfg.distributed, "parallel_scheme"):
        # Distributed path: use NeMoAutoDiffusionPipeline with parallelization
        from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

        parallel_scheme = _build_parallel_scheme(cfg.distributed.parallel_scheme, dist_info)
        pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            parallel_scheme=parallel_scheme,
        )
        logger.info("Loaded distributed pipeline: %s", type(pipe).__name__)
    else:
        # Single-GPU path: standard diffusers auto-detection
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        # Skip .to("cuda") when CPU offload is configured — enable_model_cpu_offload()
        # expects the pipeline on CPU so it can set up per-module device hooks.
        vae_cfg = getattr(cfg, "vae", None)
        if not (vae_cfg is not None and getattr(vae_cfg, "enable_cpu_offload", False)):
            pipe.to("cuda")
        logger.info("Loaded pipeline: %s", type(pipe).__name__)

    _fix_text_encoder_weight_tying(pipe)
    return pipe


def _fix_text_encoder_weight_tying(pipe):
    """Fix UMT5 text encoder weight tying for transformers>=5.0.0.

    The Wan 2.1 checkpoint stores the token embedding as "shared.weight",
    which transformers<5 automatically tied to "encoder.embed_tokens.weight".
    In v5+, this tying no longer happens during from_pretrained(), leaving
    embed_tokens zero-initialized and producing all-zero text embeddings.
    """
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        return

    if (
        hasattr(text_encoder, "shared")
        and hasattr(text_encoder, "encoder")
        and hasattr(text_encoder.encoder, "embed_tokens")
        and text_encoder.encoder.embed_tokens.weight.data_ptr() != text_encoder.shared.weight.data_ptr()
    ):
        text_encoder.encoder.embed_tokens.weight = text_encoder.shared.weight
        logger.info("Fixed UMT5 text encoder weight tying (shared.weight -> embed_tokens.weight)")


def _build_parallel_scheme(scheme_cfg, dist_info):
    """Build parallel_scheme dict from config for NeMoAutoDiffusionPipeline.

    Args:
        scheme_cfg: Config node mapping component names to their parallelism settings.
        dist_info: DistInfo with distributed environment details.

    Returns:
        Dict mapping component names to manager kwargs dicts.
    """
    parallel_scheme = {}
    for comp_name in dir(scheme_cfg):
        if comp_name.startswith("_"):
            continue
        comp_cfg = getattr(scheme_cfg, comp_name)
        if comp_cfg is None:
            continue
        manager_args = {
            "backend": "nccl",
            "world_size": dist_info.world_size,
            "use_hf_tp_plan": False,
        }
        # Copy parallelism sizes from config
        for key in ("tp_size", "cp_size", "pp_size", "dp_size", "dp_replicate_size"):
            val = getattr(comp_cfg, key, None)
            if val is not None:
                manager_args[key] = val
        parallel_scheme[comp_name] = manager_args
    return parallel_scheme


def load_checkpoint_into_pipeline(pipe, cfg):
    """Load a training checkpoint into the pipeline's transformer.

    Expects a consolidated HF safetensors checkpoint produced by training
    with model_save_format: safetensors, save_consolidated: true, and
    diffusers_compatible: true. The checkpoint directory should contain
    model/consolidated/ with diffusion_pytorch_model.safetensors.index.json
    and the corresponding safetensors files.

    Uses the standard diffusers from_pretrained() API for loading.

    Args:
        pipe: The diffusion pipeline with a `.transformer` attribute.
        cfg: Config node with `model.checkpoint` path.
    """
    checkpoint = getattr(cfg.model, "checkpoint", None)
    if not checkpoint:
        return

    dtype_str = getattr(cfg.inference, "dtype", "bfloat16")
    torch_dtype = _resolve_dtype(dtype_str)

    checkpoint_dir = Path(checkpoint)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    consolidated_dir = checkpoint_dir / "model" / "consolidated"
    if not consolidated_dir.is_dir():
        raise FileNotFoundError(
            f"No consolidated checkpoint directory found at {consolidated_dir}. "
            "Ensure training used save_consolidated: true."
        )

    logger.info("Loading consolidated safetensors checkpoint from %s", consolidated_dir)
    pipe.transformer = pipe.transformer.__class__.from_pretrained(
        str(consolidated_dir), torch_dtype=torch_dtype
    ).to("cuda")
    logger.info("Loaded consolidated safetensors checkpoint")


def apply_optimizations(pipe, cfg):
    """Apply VAE and memory optimizations to the pipeline.

    Args:
        pipe: The diffusion pipeline.
        cfg: Config node with optional `vae` section.
    """
    vae_cfg = getattr(cfg, "vae", None)
    if vae_cfg is None:
        return

    if hasattr(pipe, "vae"):
        if getattr(vae_cfg, "enable_slicing", False):
            pipe.vae.enable_slicing()
            logger.info("Enabled VAE slicing")
        if getattr(vae_cfg, "enable_tiling", False):
            pipe.vae.enable_tiling()
            logger.info("Enabled VAE tiling")

    if getattr(vae_cfg, "enable_cpu_offload", False):
        pipe.enable_model_cpu_offload()
        logger.info("Enabled model CPU offload")


def detect_output_type(pipe):
    """Detect whether the pipeline produces images or videos.

    Uses a class name lookup table, with a fallback that checks if the
    pipeline's __call__ method accepts a `num_frames` parameter.

    Args:
        pipe: The diffusion pipeline instance.

    Returns:
        "image" or "video"
    """
    class_name = type(pipe).__name__
    output_type = _PIPELINE_OUTPUT_TYPES.get(class_name)
    if output_type is not None:
        return output_type

    # Fallback: check if pipeline accepts num_frames
    try:
        sig = inspect.signature(pipe.__call__)
        if "num_frames" in sig.parameters:
            return "video"
    except (ValueError, TypeError):
        pass

    return "image"


def run_inference(pipe, cfg, is_rank0):
    """Run inference on all configured prompts and save outputs.

    Args:
        pipe: The diffusion pipeline.
        cfg: Config node with `inference` and `output` sections.
        is_rank0: Whether this is the main process (for saving outputs).
    """
    from diffusers.utils import export_to_video

    output_type = detect_output_type(pipe)
    prompts = cfg.inference.prompts
    max_samples = getattr(cfg.inference, "max_samples", len(prompts))
    prompts = prompts[:max_samples]

    output_dir = Path(getattr(cfg.output, "output_dir", "./inference_outputs"))
    fps = getattr(cfg.output, "fps", 16)

    if is_rank0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build common pipeline kwargs
    pipe_kwargs = {}
    for key in ("num_inference_steps", "guidance_scale", "height", "width"):
        val = getattr(cfg.inference, key, None)
        if val is not None:
            pipe_kwargs[key] = val

    # Merge model-specific pipeline_kwargs (convert ConfigNode to plain dict)
    extra_kwargs = getattr(cfg.inference, "pipeline_kwargs", None)
    if extra_kwargs is not None:
        pipe_kwargs.update(extra_kwargs.to_dict())

    seed = getattr(cfg, "seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Generating %d samples (%s mode)", len(prompts), output_type)
    logger.info("Pipeline kwargs: %s", pipe_kwargs)

    for i, prompt_text in enumerate(prompts):
        logger.info("[%d/%d] Prompt: %s", i + 1, len(prompts), prompt_text[:80])

        generator = torch.Generator(device="cuda").manual_seed(seed + i)

        with torch.no_grad():
            output = pipe(prompt=prompt_text, generator=generator, **pipe_kwargs)

        if not is_rank0:
            continue

        # Save output
        safe_name = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt_text)[:50].strip().replace(" ", "_")

        if output_type == "video":
            frames = output.frames[0]
            output_path = output_dir / f"sample_{i:03d}_{safe_name}.mp4"
            export_to_video(frames, str(output_path), fps=fps)
        else:
            image = output.images[0]
            output_path = output_dir / f"sample_{i:03d}_{safe_name}.png"
            image.save(str(output_path))

        logger.info("Saved: %s", output_path)


def _resolve_dtype(dtype_str):
    """Convert a dtype string to a torch.dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def main():
    cfg = parse_args_and_load_config()
    setup_logging()

    # 1. Initialize distributed (if configured)
    dist_info = maybe_init_distributed(cfg)
    is_rank0 = dist_info is None or dist_info.is_main

    # 2. Load pipeline
    pipe = load_pipeline(cfg, dist_info)

    # 3. Load checkpoint (if configured)
    load_checkpoint_into_pipeline(pipe, cfg)

    # 4. Apply VAE / memory optimizations
    apply_optimizations(pipe, cfg)

    # 5. Synchronize before inference
    if dist_info is not None:
        dist.barrier()

    # 6. Run inference
    run_inference(pipe, cfg, is_rank0)

    # 7. Cleanup
    if dist_info is not None:
        dist.barrier()
        dist.destroy_process_group()
        logger.info("Distributed inference complete")


if __name__ == "__main__":
    main()
