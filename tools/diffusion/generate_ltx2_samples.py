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
Generate LTX-2 video+audio samples from training-set prompts for finetune evaluation.

Purpose: the memorization/overfit acceptance test. After finetuning on a small clip set,
generating with the SAME captions (and a fixed seed) should produce clips that visibly
resemble the training data — for both full finetunes and LoRA runs. Each generated clip
is written to an independent output folder next to a copy of its ground-truth training
clip for side-by-side comparison.

Checkpoint modes (mutually exclusive; omit both for the base-model baseline):
- --transformer_dir  : consolidated diffusers-layout full-finetune checkpoint
  (loads via ``LTX2VideoTransformer3DModel.from_pretrained``).
- --lora_ckpt        : LoRA checkpoint (.safetensors file/dir or torch .pt) containing
  ``lora_A.weight`` / ``lora_B.weight`` keys from an Automodel PEFT run. LoRA deltas are
  MERGED into the base weights for inference: ``W' = W + (alpha/dim) * B @ A``.

Example (full-FT eval):
    python tools/diffusion/generate_ltx2_samples.py \
        --transformer_dir ../ckpt_full/.../consolidated \
        --meta_json ../ghibili/meta.json --video_dir ../ghibili \
        --num_clips 4 --output_dir ../generated_full

Same command with --lora_ckpt for the LoRA eval, or neither for the base baseline.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from tools.diffusion.validate_ltx2_roundtrip import write_video_with_audio

logger = logging.getLogger("generate_ltx2_samples")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for LTX-2 sample generation."""
    p = argparse.ArgumentParser("LTX-2 finetune evaluation sampler")
    p.add_argument("--model_id", type=str, default="dg845/LTX-2.3-Diffusers")
    ckpt = p.add_mutually_exclusive_group()
    ckpt.add_argument("--transformer_dir", type=str, help="Consolidated full-finetune transformer checkpoint dir")
    ckpt.add_argument("--lora_ckpt", type=str, help="LoRA checkpoint (.safetensors file/dir or .pt) to merge")
    p.add_argument("--lora_dim", type=int, default=64, help="LoRA rank used in training (for merge scale)")
    p.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha used in training (for merge scale)")
    p.add_argument("--meta_json", type=str, required=True, help="Dataset meta.json with per-clip captions")
    p.add_argument("--video_dir", type=str, required=True, help="Directory holding the ground-truth clips")
    p.add_argument("--prompt_field", type=str, default="vila_caption")
    p.add_argument("--num_clips", type=int, default=4, help="Number of dataset entries to sample")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num_frames", type=int, default=89)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--num_inference_steps", type=int, default=40)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--audio_guidance_scale", type=float, default=None, help="Defaults to pipeline behavior")
    p.add_argument("--seed", type=int, default=42, help="Fixed seed so baseline/FT/LoRA runs are comparable")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, required=True, help="Independent folder for this run's samples")
    return p.parse_args()


def load_lora_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load a LoRA checkpoint into a flat state dict of ``lora_*`` tensors.

    Args:
        path: A .safetensors file, a directory containing .safetensors shards,
            or a torch-save .pt file.

    Returns:
        Mapping of parameter name (containing ``lora_A``/``lora_B``) to tensor.
    """
    p = Path(path)
    tensors: dict[str, torch.Tensor] = {}
    files = sorted(p.glob("**/*.safetensors")) if p.is_dir() else [p]
    for f in files:
        if f.suffix == ".safetensors":
            from safetensors.torch import load_file

            tensors.update(load_file(str(f)))
        else:
            loaded = torch.load(str(f), map_location="cpu", weights_only=True)
            state = loaded.get("model", loaded) if isinstance(loaded, dict) else loaded
            tensors.update({k: v for k, v in state.items() if torch.is_tensor(v)})
    lora = {k: v for k, v in tensors.items() if "lora_A" in k or "lora_B" in k}
    if not lora:
        raise ValueError(f"No lora_A/lora_B tensors found in {path} (found {len(tensors)} tensors)")
    return lora


def merge_lora_into_transformer(transformer: torch.nn.Module, lora_state: dict, scale: float) -> int:
    """Merge LoRA deltas into base Linear weights in place: ``W += scale * B @ A``.

    Matches Automodel's ``LinearLoRA`` forward (``res + lora_B(lora_A(x) * scale)``
    with ``scale = alpha / dim``), so the merged model reproduces the trained
    LoRA model exactly at inference.

    Args:
        transformer: Base diffusers transformer (weights modified in place).
        lora_state: Flat dict with ``<module>.lora_A.weight`` / ``<module>.lora_B.weight`` keys.
        scale: alpha / dim from the training config.

    Returns:
        Number of Linear modules merged.

    Raises:
        KeyError: If a LoRA pair references a module path absent from the transformer.
    """
    params = dict(transformer.named_parameters())
    merged = 0
    for a_key in sorted(k for k in lora_state if "lora_A.weight" in k):
        module_path = a_key[: a_key.index("lora_A.weight")].rstrip(".")
        target = f"{module_path}.weight"
        # Checkpoint keys may carry a wrapper prefix (e.g. "model.") relative to
        # the transformer, or vice versa. Match on exact name first, then on a
        # DOT-BOUNDARY suffix in either direction — a plain endswith would let
        # "audio_attn1.to_q" collide with "attn1.to_q".
        candidates = [k for k in params if k == target]
        if not candidates:
            candidates = [k for k in params if target.endswith(f".{k}") or k.endswith(f".{target}")]
        if not candidates:
            raise KeyError(f"LoRA key {a_key} has no matching base weight in the transformer")
        if len(candidates) > 1:
            raise KeyError(f"LoRA key {a_key} matches multiple base weights: {candidates}")
        weight = params[candidates[0]]
        b_key = a_key.replace("lora_A", "lora_B")
        delta = lora_state[b_key].float() @ lora_state[a_key].float()
        with torch.no_grad():
            weight.add_((scale * delta).to(weight.dtype).to(weight.device))
        merged += 1
    return merged


def frames_to_uint8(frames) -> np.ndarray:
    """Convert pipeline video output to uint8 frames [T, H, W, 3].

    Args:
        frames: Pipeline output — np.ndarray [B, T, H, W, C] in [0, 1], a torch
            tensor of the same layout, or a list of PIL images.

    Returns:
        uint8 RGB frames [T, H, W, 3].
    """
    if isinstance(frames, list) and frames and not isinstance(frames[0], (np.ndarray, torch.Tensor)):
        return np.stack([np.asarray(f.convert("RGB")) for f in frames])
    arr = frames[0] if isinstance(frames, (list, tuple)) else frames
    if torch.is_tensor(arr):
        arr = arr.float().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.shape[-1] != 3 and arr.shape[1] == 3:  # [T, C, H, W] -> [T, H, W, C]
        arr = arr.transpose(0, 2, 3, 1)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
    return arr


def waveform_to_2d(audio) -> torch.Tensor:
    """Normalize pipeline audio output to a [channels, samples] float32 tensor in [-1, 1]."""
    wav = audio[0] if isinstance(audio, (list, tuple)) else audio
    if not torch.is_tensor(wav):
        wav = torch.from_numpy(np.asarray(wav))
    wav = wav.float().cpu()
    while wav.ndim > 2:
        wav = wav.squeeze(0)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return wav.clamp(-1.0, 1.0)


def main() -> None:
    """Generate samples from training captions and save them with ground-truth copies."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import LTX2Pipeline
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    dtype = torch.bfloat16
    pipe_kwargs = {}
    if args.transformer_dir:
        logger.info("Loading finetuned transformer from %s", args.transformer_dir)
        pipe_kwargs["transformer"] = LTX2VideoTransformer3DModel.from_pretrained(
            args.transformer_dir, torch_dtype=dtype
        )
    pipe = LTX2Pipeline.from_pretrained(args.model_id, torch_dtype=dtype, **pipe_kwargs)

    if args.lora_ckpt:
        lora_state = load_lora_state_dict(args.lora_ckpt)
        scale = args.lora_alpha / args.lora_dim
        merged = merge_lora_into_transformer(pipe.transformer, lora_state, scale)
        logger.info("Merged LoRA from %s into %d Linear modules (scale=%.3f)", args.lora_ckpt, merged, scale)

    pipe.to(args.device)

    entries = json.loads(Path(args.meta_json).read_text())[: args.num_clips]
    logger.info(
        "Generating %d clips at %dx%d, %d frames, seed %d",
        len(entries),
        args.width,
        args.height,
        args.num_frames,
        args.seed,
    )

    manifest = []
    for entry in entries:
        clip_name = entry["file_name"]
        prompt = entry[args.prompt_field]
        stem = Path(clip_name).stem
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

        result = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.fps,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            audio_guidance_scale=args.audio_guidance_scale,
            generator=generator,
            output_type="np",
        )

        frames = frames_to_uint8(result.frames)
        waveform = waveform_to_2d(result.audio)
        sample_rate = int(getattr(pipe.vocoder.config, "output_sampling_rate", 16000))

        out_path = output_dir / f"{stem}_generated.mp4"
        write_video_with_audio(out_path, frames, args.fps, waveform, sample_rate)

        gt_src = Path(args.video_dir) / clip_name
        gt_path = output_dir / f"{stem}_groundtruth.mp4"
        if gt_src.exists() and not gt_path.exists():
            shutil.copy2(gt_src, gt_path)

        manifest.append(
            {
                "clip": clip_name,
                "prompt": prompt,
                "generated": out_path.name,
                "groundtruth": gt_path.name if gt_src.exists() else None,
                "seed": args.seed,
            }
        )
        logger.info(
            "Wrote %s (%d frames, audio %.2fs @ %d Hz)",
            out_path,
            frames.shape[0],
            waveform.shape[1] / sample_rate,
            sample_rate,
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Done. %d generated clips + ground truths in %s", len(manifest), output_dir)


if __name__ == "__main__":
    main()
