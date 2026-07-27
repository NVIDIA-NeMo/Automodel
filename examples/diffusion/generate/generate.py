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
models (FLUX, Qwen-Image, Qwen-Image-Edit, Wan 2.1/2.2, HunyuanVideo, LTX-2).
Supports single-GPU and distributed inference with optional checkpoint loading.

Pipelines that return an audio track alongside video (e.g. LTX-2) have it muxed
into the output mp4; video-only pipelines are unaffected.

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

import gc
import inspect
import logging
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.shared.transformers_patches import patch_t5_layer_norm

logger = logging.getLogger(__name__)

# Pipeline class name -> output type mapping
_PIPELINE_OUTPUT_TYPES = {
    "FluxPipeline": "image",
    "Flux2Pipeline": "image",
    "QwenImagePipeline": "image",
    "WanPipeline": "video",
    "HunyuanVideoPipeline": "video",
    "HunyuanVideo15Pipeline": "video",
    "LTX2Pipeline": "video",
}

# AAC encodes fixed-size frames; PyAV requires input frames of this length.
_AAC_FRAME_SAMPLES = 1024


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

    Uses NeMoAutoDiffusionPipeline for both single-GPU and distributed
    inference. When no distributed config is present, parallelization is
    skipped automatically.

    Args:
        cfg: Config node with `model.pretrained_model_name_or_path`.
        dist_info: DistInfo from maybe_init_distributed, or None.

    Returns:
        A diffusers pipeline instance.
    """
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    model_id = cfg.model.pretrained_model_name_or_path
    dtype_str = getattr(cfg.inference, "dtype", "bfloat16")
    torch_dtype = _resolve_dtype(dtype_str)

    # Apex's FusedRMSNorm doesn't support bf16. Patch T5LayerNorm before loading
    # any pipeline that may use a T5 text encoder (FLUX, HunyuanVideo, etc.).
    if torch_dtype == torch.bfloat16:
        patch_t5_layer_norm()

    # Build parallel_scheme from distributed config (None for single-GPU).
    parallel_scheme = None
    if dist_info is not None and hasattr(cfg.distributed, "parallel_scheme"):
        parallel_scheme = _build_parallel_scheme(cfg.distributed.parallel_scheme, dist_info)

    # CPU offload requires modules to stay on CPU so enable_model_cpu_offload()
    # can install per-module device hooks (called later in apply_optimizations).
    vae_cfg = getattr(cfg, "vae", None)
    cpu_offload = vae_cfg is not None and getattr(vae_cfg, "enable_cpu_offload", False)

    pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        parallel_scheme=parallel_scheme,
        move_to_device=not cpu_offload,
    )

    _fix_text_encoder_weight_tying(pipe)
    logger.info("Loaded pipeline: %s (distributed=%s)", type(pipe).__name__, parallel_scheme is not None)
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
    """Load training checkpoint(s) into the pipeline's transformer(s).

    Supports both single-transformer pipelines (Wan2.1, FLUX, HunyuanVideo) and
    two-transformer pipelines (Wan2.2-T2V-A14B with ``transformer`` for the
    high-noise stage and ``transformer_2`` for the low-noise stage).

    Single-transformer path: set ``model.checkpoint`` to load into ``pipe.transformer``.

    Two-transformer path (Wan2.2): set ``model.checkpoint_high_noise`` and/or
    ``model.checkpoint_low_noise``. Each is independently optional — a missing
    one leaves that stage's transformer at its hub-pretrained weights, which is
    useful for sanity-checking a partial finetune.

    Expects consolidated HF safetensors checkpoints produced by training with
    ``model_save_format: safetensors`` and ``save_consolidated: true``.

    Args:
        pipe: The diffusion pipeline. May expose ``transformer_2`` for Wan2.2.
        cfg: Config node with one of ``model.checkpoint``,
            ``model.checkpoint_high_noise``, or ``model.checkpoint_low_noise``.

    Raises:
        ValueError: If both single-stage and two-stage checkpoint fields are set.
    """
    checkpoint = getattr(cfg.model, "checkpoint", None)
    checkpoint_high = getattr(cfg.model, "checkpoint_high_noise", None)
    checkpoint_low = getattr(cfg.model, "checkpoint_low_noise", None)

    if checkpoint and (checkpoint_high or checkpoint_low):
        raise ValueError(
            "model.checkpoint is mutually exclusive with "
            "model.checkpoint_high_noise / model.checkpoint_low_noise. "
            "Use the latter pair for two-transformer pipelines (Wan2.2) and "
            "model.checkpoint for single-transformer pipelines."
        )

    dtype_str = getattr(cfg.inference, "dtype", "bfloat16")
    torch_dtype = _resolve_dtype(dtype_str)

    if checkpoint:
        _load_checkpoint_into_attr(pipe, "transformer", checkpoint, torch_dtype)
        return

    if checkpoint_high:
        _load_checkpoint_into_attr(pipe, "transformer", checkpoint_high, torch_dtype)
    if checkpoint_low:
        if getattr(pipe, "transformer_2", None) is None:
            raise ValueError(
                "model.checkpoint_low_noise is set but the loaded pipeline has no "
                "transformer_2 attribute. This option only applies to two-stage "
                "models like Wan2.2-T2V-A14B."
            )
        _load_checkpoint_into_attr(pipe, "transformer_2", checkpoint_low, torch_dtype)


def _load_checkpoint_into_attr(pipe, attr_name, checkpoint, torch_dtype):
    """Load a single consolidated/sharded checkpoint into ``pipe.<attr_name>``."""
    checkpoint_dir = Path(checkpoint)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    ema_path = checkpoint_dir / "ema_shadow.pt"
    consolidated_path = checkpoint_dir / "consolidated_model.bin"
    consolidated_st_dir = checkpoint_dir / "model" / "consolidated"
    sharded_dir = checkpoint_dir / "model"

    target = getattr(pipe, attr_name)
    if target is None:
        raise AttributeError(f"Pipeline has no attribute {attr_name!r} to load checkpoint into")

    # Set by the branches that build a replacement module rather than loading
    # weights into ``target`` in place; consumed by the swap below.
    new_module = None

    # Load checkpoints to CPU first: load_state_dict copies into the target's
    # existing (already-on-device) parameters, so a GPU map_location would hold a
    # second full copy of the weights on-device and roughly double peak GPU memory.
    # This matters for Wan2.2 two-stage inference, where this runs once per stage
    # and the first transformer is still resident when the second loads.
    if ema_path.exists():
        logger.info("Loading EMA checkpoint from %s into %s", ema_path, attr_name)
        ema_state = torch.load(ema_path, map_location="cpu", weights_only=True)
        target.load_state_dict(ema_state, strict=True)
    elif consolidated_path.exists():
        logger.info("Loading consolidated checkpoint from %s into %s", consolidated_path, attr_name)
        state_dict = torch.load(consolidated_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        target.load_state_dict(state_dict, strict=True)
    elif consolidated_st_dir.is_dir() and any(
        name.endswith(".safetensors") for name in os.listdir(consolidated_st_dir)
    ):
        logger.info("Loading consolidated safetensors checkpoint from %s into %s", consolidated_st_dir, attr_name)
        new_module = type(target).from_pretrained(str(consolidated_st_dir), torch_dtype=torch_dtype)
    elif sharded_dir.is_dir() and any(name.endswith(".distcp") for name in os.listdir(sharded_dir)):
        logger.info("Loading sharded FSDP checkpoint from %s into %s", sharded_dir, attr_name)
        new_module = _load_sharded_fsdp_checkpoint(target, str(sharded_dir), torch_dtype)
    elif sharded_dir.is_dir() and any(
        name.startswith("shard-") and name.endswith(".safetensors") for name in os.listdir(sharded_dir)
    ):
        # NeMo-AutoModel sharded HF safetensors: one ``shard-XXXXX-*.safetensors``
        # per FSDP rank from a run that set ``save_consolidated: false``. Use the
        # DCP HuggingFaceStorageReader to materialize the full state dict.
        logger.info("Loading sharded HF safetensors checkpoint from %s into %s", sharded_dir, attr_name)
        new_module = _load_sharded_hf_safetensors_checkpoint(target, str(sharded_dir), torch_dtype)
    else:
        logger.warning(
            "No recognized checkpoint format found in %s, leaving %s at base weights",
            checkpoint_dir,
            attr_name,
        )

    # The three paths above build the replacement on CPU. Release the existing
    # GPU-resident module before moving the replacement onto the device: holding
    # both at once needs twice the weights in VRAM, which OOMs an 80GB device for
    # a large transformer (LTX-2's is ~19B params, ~38GB in bf16).
    if new_module is not None:
        setattr(pipe, attr_name, None)
        target.to("cpu")
        del target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        new_module.to("cuda", dtype=torch_dtype)
        setattr(pipe, attr_name, new_module)


def load_lora_weights_into_pipeline(pipe, cfg):
    """Load LoRA adapter weights into the pipeline's transformer.

    Reads adapter_model.safetensors + adapter_config.json from the directory
    specified by model.lora_weights. Does nothing if lora_weights is null/unset.

    Args:
        pipe: The diffusion pipeline with a `.transformer` attribute.
        cfg: Config node with optional `model.lora_weights`, `model.lora_adapter_name`.
    """
    lora_weights = getattr(cfg.model, "lora_weights", None)
    if not lora_weights:
        return

    import json

    from safetensors.torch import load_file

    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules

    lora_path = Path(lora_weights)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights directory not found: {lora_path}")

    with open(lora_path / "adapter_config.json") as f:
        peft_config_dict = json.load(f)
    if "dim" not in peft_config_dict and "r" in peft_config_dict:
        peft_config_dict["dim"] = peft_config_dict["r"]
    if "alpha" not in peft_config_dict and "lora_alpha" in peft_config_dict:
        peft_config_dict["alpha"] = peft_config_dict["lora_alpha"]
    if "dropout" not in peft_config_dict and "lora_dropout" in peft_config_dict:
        peft_config_dict["dropout"] = peft_config_dict["lora_dropout"]

    peft_config = PeftConfig.from_dict(peft_config_dict)
    num_lora_modules = apply_lora_to_linear_modules(pipe.transformer, peft_config, skip_freeze=True)
    if num_lora_modules == 0:
        raise RuntimeError(f"No transformer modules matched LoRA config from {lora_path / 'adapter_config.json'}")

    state_dict = load_file(lora_path / "adapter_model.safetensors", device="cuda")
    state_dict = {key.removeprefix("base_model.model."): value for key, value in state_dict.items()}
    load_result = pipe.transformer.load_state_dict(state_dict, strict=False)
    missing_lora_keys = sorted(key for key in load_result.missing_keys if ".lora_" in key)
    unexpected_lora_keys = sorted(key for key in load_result.unexpected_keys if ".lora_" in key)
    if missing_lora_keys or unexpected_lora_keys:
        raise RuntimeError(
            "LoRA checkpoint did not load cleanly. "
            f"missing_lora_keys={missing_lora_keys[:10]}, unexpected_lora_keys={unexpected_lora_keys[:10]}"
        )
    logger.info("Loaded LoRA adapter from %s (%d modules, %d tensors)", lora_path, num_lora_modules, len(state_dict))


def _load_sharded_fsdp_checkpoint(transformer, sharded_dir, torch_dtype=torch.bfloat16):
    """Load sharded FSDP1 .distcp checkpoint into a transformer module.

    Creates a temporary gloo process group for single-GPU loading if
    torch.distributed is not already initialized.

    Args:
        transformer: The transformer nn.Module to load weights into.
        sharded_dir: Path to the directory containing .distcp shard files.
        torch_dtype: The dtype to cast the transformer to before loading.

    Returns:
        The unwrapped transformer module with loaded checkpoint weights.
    """
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint import load as dist_load
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp.api import ShardedStateDictConfig

    init_dist = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        init_dist = True

    try:
        transformer.to(device="cuda", dtype=torch_dtype)
        fsdp_transformer = FSDP(transformer, use_orig_params=True)
        FSDP.set_state_dict_type(
            fsdp_transformer,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        )
        model_state = fsdp_transformer.state_dict()
        dist_load(state_dict=model_state, storage_reader=FileSystemReader(sharded_dir))
        fsdp_transformer.load_state_dict(model_state)
        return fsdp_transformer.module
    finally:
        if init_dist:
            dist.destroy_process_group()


def _load_sharded_hf_safetensors_checkpoint(transformer, sharded_dir, torch_dtype=torch.bfloat16):
    """Load NeMo-AutoModel sharded HF safetensors checkpoint into a transformer.

    Handles directories containing ``shard-XXXXX-model-XXXXX-of-XXXXX.safetensors``
    files produced by training runs with ``save_consolidated: false``. Uses DCP's
    ``HuggingFaceStorageReader`` to gather all shards into the target state dict.

    Args:
        transformer: The transformer nn.Module to load weights into.
        sharded_dir: Path to the directory containing shard-*.safetensors files.
        torch_dtype: The dtype to cast the transformer to before loading.

    Returns:
        The transformer module with the merged state dict loaded.
    """
    from torch.distributed.checkpoint import load as dist_load

    # Prefer the upstream HF storage reader; fall back to NeMo's backport if
    # the torch version is too old to ship it.
    try:
        from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
    except ImportError:
        from nemo_automodel.components.checkpoint._backports.hf_storage import (
            _HuggingFaceStorageReader as HuggingFaceStorageReader,
        )

    init_dist = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        init_dist = True

    try:
        transformer.to(device="cuda", dtype=torch_dtype)
        state_dict = transformer.state_dict()
        dist_load(state_dict=state_dict, storage_reader=HuggingFaceStorageReader(path=sharded_dir))
        transformer.load_state_dict(state_dict, strict=True)
        return transformer
    finally:
        if init_dist:
            dist.destroy_process_group()


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


def _frames_to_uint8(frames) -> np.ndarray:
    """Convert one sample's video frames to a uint8 RGB array.

    Args:
        frames: One sample's frames, as returned by ``output.frames[0]``. Either a
            list of PIL images, or an array/tensor of shape
            [frames, height, width, 3] or [frames, 3, height, width] holding
            values in [0, 1].

    Returns:
        uint8 RGB array of shape [frames, height, width, 3].
    """
    if isinstance(frames, list) and frames and not isinstance(frames[0], (np.ndarray, torch.Tensor)):
        return np.stack([np.asarray(frame.convert("RGB")) for frame in frames])

    arr = frames.float().cpu().numpy() if torch.is_tensor(frames) else np.asarray(frames)
    if arr.shape[-1] != 3 and arr.shape[1] == 3:  # [frames, 3, H, W] -> [frames, H, W, 3]
        arr = arr.transpose(0, 2, 3, 1)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
    return arr


def _waveform_to_2d(audio) -> torch.Tensor:
    """Normalize a pipeline audio output to a 2-D waveform tensor.

    Args:
        audio: Pipeline audio output — an array/tensor of shape [channels, samples],
            [1, channels, samples], or [samples], or a list/tuple holding one such
            entry per generated sample. Values are expected in [-1, 1].

    Returns:
        float32 tensor of shape [channels, samples], clamped to [-1, 1].
    """
    wav = audio[0] if isinstance(audio, (list, tuple)) else audio
    if not torch.is_tensor(wav):
        wav = torch.from_numpy(np.asarray(wav))
    wav = wav.float().cpu()
    while wav.ndim > 2:
        wav = wav.squeeze(0)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return wav.clamp(-1.0, 1.0)


def _resolve_audio_sample_rate(pipe, cfg) -> int:
    """Resolve the sample rate for a pipeline's audio output.

    Prefers an explicit ``output.audio_sample_rate`` config value, falling back to
    the rate advertised by the pipeline's vocoder.

    Args:
        pipe: The diffusion pipeline.
        cfg: Config node with an `output` section.

    Returns:
        Sample rate in Hz.

    Raises:
        ValueError: If neither the config nor the pipeline provides a rate.
    """
    rate = getattr(cfg.output, "audio_sample_rate", None)
    if rate is not None:
        return int(rate)

    vocoder_config = getattr(getattr(pipe, "vocoder", None), "config", None)
    rate = getattr(vocoder_config, "output_sampling_rate", None)
    if rate is None:
        raise ValueError(
            "Pipeline returned audio but its sample rate could not be determined. "
            "Set output.audio_sample_rate in the config."
        )
    return int(rate)


def _write_video_with_audio(
    path: Path,
    frames: np.ndarray,
    fps: float,
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    """Mux video frames and an audio waveform into an mp4 (h264 + aac).

    Args:
        path: Output file path.
        frames: uint8 RGB video frames of shape [frames, height, width, 3].
        fps: Video frame rate.
        waveform: Audio of shape [channels, samples], float32 in [-1, 1].
        sample_rate: Audio sample rate in Hz.
    """
    import av

    pcm = (waveform.numpy().clip(-1.0, 1.0) * 32767.0).astype(np.int16)
    # The AAC encoder defaults to a stereo layout, which would silently upmix a
    # mono waveform, so declare the layout on the stream as well as the frames.
    layout = "stereo" if pcm.shape[0] == 2 else "mono"

    with av.open(str(path), mode="w") as container:
        vstream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        vstream.width, vstream.height = frames.shape[2], frames.shape[1]
        vstream.pix_fmt = "yuv420p"
        astream = container.add_stream("aac", rate=sample_rate, layout=layout)

        for frame in frames:
            for packet in vstream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)

        for start in range(0, pcm.shape[1], _AAC_FRAME_SAMPLES):
            chunk = np.ascontiguousarray(pcm[:, start : start + _AAC_FRAME_SAMPLES])
            aframe = av.AudioFrame.from_ndarray(chunk.reshape(1, -1, order="F"), format="s16", layout=layout)
            aframe.sample_rate = sample_rate
            aframe.pts = start
            for packet in astream.encode(aframe):
                container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)


def run_inference(pipe, cfg, is_rank0):
    """Run inference on all configured prompts and save outputs.

    Images are saved as png. Video is saved as mp4, with a generated audio track
    muxed in when the pipeline returns one.

    Args:
        pipe: The diffusion pipeline.
        cfg: Config node with `inference` and `output` sections. The optional
            `inference.input_images` list supplies one local path or URL per
            prompt for image-conditioned pipelines.
        is_rank0: Whether this is the main process (for saving outputs).
    """
    from diffusers.utils import export_to_video, load_image

    output_type = detect_output_type(pipe)
    prompts = cfg.inference.prompts
    input_images = getattr(cfg.inference, "input_images", None)
    if input_images is not None:
        if not isinstance(input_images, list):
            raise TypeError("inference.input_images must be a list with one local path or URL per prompt")
        if len(input_images) != len(prompts):
            raise ValueError(
                "inference.input_images must contain one entry per prompt, "
                f"got {len(input_images)} images for {len(prompts)} prompts"
            )
        if any(not isinstance(image, str) or not image for image in input_images):
            raise TypeError("Every inference.input_images entry must be a non-empty local path or URL string")

    max_samples = getattr(cfg.inference, "max_samples", len(prompts))
    prompts = prompts[:max_samples]
    if input_images is not None:
        input_images = input_images[:max_samples]

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

    if input_images is not None:
        if "image" in pipe_kwargs:
            raise ValueError("Set source images with inference.input_images, not inference.pipeline_kwargs.image")
        if "image" not in inspect.signature(pipe.__call__).parameters:
            raise ValueError(
                f"{type(pipe).__name__} does not accept image inputs, but inference.input_images is configured"
            )

    # LoRA scale: passed as attention_kwargs (newer diffusers) or
    # cross_attention_kwargs (older diffusers) so the transformer forward()
    # applies the correct contribution weight.
    lora_weights = getattr(cfg.model, "lora_weights", None)
    if lora_weights:
        lora_scale = getattr(cfg.model, "lora_scale", 1.0)
        call_sig = inspect.signature(pipe.__call__)
        if "attention_kwargs" in call_sig.parameters:
            pipe_kwargs["attention_kwargs"] = {"scale": lora_scale}
        elif "cross_attention_kwargs" in call_sig.parameters:
            pipe_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

    seed = getattr(cfg, "seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Generating %d samples (%s mode)", len(prompts), output_type)
    logger.info("Pipeline kwargs: %s", pipe_kwargs)

    for i, prompt_text in enumerate(prompts):
        logger.info("[%d/%d] Prompt: %s", i + 1, len(prompts), prompt_text[:80])

        generator = torch.Generator(device="cuda").manual_seed(seed + i)
        sample_kwargs = dict(pipe_kwargs)
        if input_images is not None:
            sample_kwargs["image"] = load_image(input_images[i])

        with torch.no_grad():
            output = pipe(prompt=prompt_text, generator=generator, **sample_kwargs)

        if not is_rank0:
            continue

        # Save output
        safe_name = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt_text)[:50].strip().replace(" ", "_")

        if output_type == "video":
            frames = output.frames[0]
            output_path = output_dir / f"sample_{i:03d}_{safe_name}.mp4"
            # Dual-stream pipelines (e.g. LTX-2) also return a waveform; muxing it
            # in keeps the generated audio track instead of silently dropping it.
            audio = getattr(output, "audio", None)
            if audio is None:
                export_to_video(frames, str(output_path), fps=fps)
            else:
                _write_video_with_audio(
                    output_path,
                    _frames_to_uint8(frames),
                    fps,
                    _waveform_to_2d(audio),
                    _resolve_audio_sample_rate(pipe, cfg),
                )
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
    """Run diffusion generation from a recipe configuration."""

    cfg = parse_args_and_load_config()
    setup_logging()

    # 1. Initialize distributed (if configured)
    dist_info = maybe_init_distributed(cfg)
    is_rank0 = dist_info is None or dist_info.is_main

    # 2. Load pipeline
    pipe = load_pipeline(cfg, dist_info)

    # 3. Load checkpoint (if configured)
    load_checkpoint_into_pipeline(pipe, cfg)

    # 3b. Load LoRA adapter weights (if configured)
    load_lora_weights_into_pipeline(pipe, cfg)

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
