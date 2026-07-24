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
LTX-2.3 preprocessing round-trip validation: encode a clip to latents with the
real LTX2Processor path, decode the latents back, and write a watchable mp4
(video + audio muxed) next to a same-pipeline reference for A/B comparison.

Two input modes:
- --video CLIP.mp4  : runs the full preprocessing encode (video VAE, mel +
  audio VAE) exactly as tools/diffusion/preprocessing_multiprocess.py would,
  then decodes. This validates the whole encode path end-to-end.
- --cache_file X.pt : decodes an existing preprocessing cache file, validating
  what was actually written to disk.

Decode mirrors the LTX-2 reference pipeline:
- Video: optional decode-time noise + timestep conditioning (0.025 / 0.05),
  de-normalize (z * std / scaling_factor + mean), VAE decode -> [-1, 1] pixels.
- Audio: inverse per-(channel x mel-bin) normalization, audio VAE decode to
  log-mel, vocoder to waveform (LTX-2.3 BWE vocoder outputs 48 kHz; runs in
  fp32 - bf16 accumulation through its long conv stack degrades quality).

Outputs (in --output_dir):
- <stem>_roundtrip.mp4  : decoded video + decoded audio
- <stem>_reference.mp4  : the exact frames/waveform that entered the encoders
- printed video PSNR and audio RMS stats

Example:
    python tools/diffusion/validate_ltx2_roundtrip.py \
        --video /data/clips/dog.mp4 --height 512 --width 704 --num_frames 121 \
        --output_dir ./roundtrip_out
"""

import argparse
import logging
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from tools.diffusion.processors.ltx2 import LTX2Processor, _import_ltx2_classes

logger = logging.getLogger("validate_ltx2_roundtrip")

_AAC_FRAME_SAMPLES = 1024


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the LTX-2.3 round-trip validation."""
    p = argparse.ArgumentParser("LTX-2.3 preprocessing round-trip validation")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=str, help="Input clip; runs the full encode+decode round trip")
    source.add_argument("--cache_file", type=str, help="Existing preprocessing .pt cache file to decode")
    p.add_argument("--model_id", type=str, default="dg845/LTX-2.3-Diffusers")
    p.add_argument("--height", type=int, default=512, help="Target pixel height (multiple of 32; --video mode)")
    p.add_argument("--width", type=int, default=704, help="Target pixel width (multiple of 32; --video mode)")
    p.add_argument("--num_frames", type=int, default=121, help="Frames at 24 fps, 8n+1 (--video mode)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="./ltx2_roundtrip")
    return p.parse_args()


def load_roundtrip_models(model_id: str, device: str) -> dict:
    """Load only the components the round trip needs (video VAE, audio VAE, mel).

    Skips the Gemma-3 text encoder and connectors that
    ``LTX2Processor.load_models`` loads for full preprocessing - text
    conditioning plays no part in an encode/decode round trip.

    Returns:
        Models dict compatible with ``LTX2Processor.encode_video`` /
        ``encode_audio`` (keys: vae, audio_vae, mel_transform, dtype).
    """
    from tools.diffusion.processors._ltx2_audio import LTX2MelSpectrogram

    from nemo_automodel._diffusers._hf_cache import resolve_diffusion_model_dir

    AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio, _ = _import_ltx2_classes()

    # bf16 matches preprocessing on GPU; CPU conv kernels are far faster in fp32.
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32

    model_dir = resolve_diffusion_model_dir(model_id)
    logger.info("Loading video VAE...")
    vae = AutoencoderKLLTX2Video.from_pretrained(model_dir, subfolder="vae", torch_dtype=dtype)
    vae.to(device).eval().requires_grad_(False)
    logger.info("Loading audio VAE...")
    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(model_dir, subfolder="audio_vae", torch_dtype=dtype)
    audio_vae.to(device).eval().requires_grad_(False)

    return {"vae": vae, "audio_vae": audio_vae, "mel_transform": LTX2MelSpectrogram().to(device), "dtype": dtype}


def load_vocoder(model_id: str, device: str) -> torch.nn.Module:
    """Load the LTX-2 vocoder (BWE variant for 2.3, detected from its config)."""
    try:
        from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder, LTX2VocoderWithBWE
    except ImportError as e:
        raise ImportError("LTX-2 vocoder classes require a diffusers release with diffusers.pipelines.ltx2") from e

    from nemo_automodel._diffusers._hf_cache import resolve_diffusion_model_dir

    model_dir = resolve_diffusion_model_dir(model_id)
    config = LTX2Vocoder.load_config(model_dir, subfolder="vocoder")
    vocoder_cls = LTX2VocoderWithBWE if "bwe_in_channels" in config else LTX2Vocoder
    vocoder = vocoder_cls.from_pretrained(model_dir, subfolder="vocoder")
    vocoder.to(device).eval().requires_grad_(False)
    return vocoder


@torch.no_grad()
def decode_video_latents(video_latents: torch.Tensor, vae, device: str) -> torch.Tensor:
    """Decode normalized video latents [1, 128, F', H', W'] to pixels [1, 3, F, H, W] in [-1, 1]."""
    latents = video_latents.to(device=device, dtype=torch.float32)

    temb = None
    if getattr(vae.config, "timestep_conditioning", False):
        # Reference decoder applies decode-time noise before de-normalization.
        decode_noise_scale, decode_timestep = 0.025, 0.05
        latents = torch.randn_like(latents) * decode_noise_scale + (1.0 - decode_noise_scale) * latents
        temb = torch.full((latents.shape[0],), decode_timestep, device=device, dtype=vae.dtype)

    mean, std = LTX2Processor._latent_stats(vae)
    mean = mean.view(1, -1, 1, 1, 1).to(device, latents.dtype)
    std = std.view(1, -1, 1, 1, 1).to(device, latents.dtype)
    latents = latents * std / vae.config.scaling_factor + mean
    return vae.decode(latents.to(vae.dtype), temb, return_dict=False)[0].float().cpu()


@torch.no_grad()
def decode_audio_latents(audio_latents: torch.Tensor, audio_vae, vocoder, device: str) -> tuple[torch.Tensor, int]:
    """Decode normalized audio latents [1, 8, L, 16] to a waveform.

    Returns:
        Tuple of (waveform [channels, samples] float32 in [-1, 1],
        sample_rate from the vocoder config - 48 kHz for LTX-2.3 BWE).
    """
    latents = audio_latents.to(device=device, dtype=torch.float32)
    b, c, t, m = latents.shape

    mean, std = LTX2Processor._latent_stats(audio_vae)
    mean = mean.to(device, latents.dtype)
    std = std.to(device, latents.dtype)
    flat = latents.permute(0, 2, 1, 3).reshape(b, t, c * m)
    latents = (flat * std + mean).view(b, t, c, m).permute(0, 2, 1, 3)

    mel = audio_vae.decode(latents.to(audio_vae.dtype), return_dict=False)[0]
    # The vocoder is loaded in fp32 and fed fp32 mels: bf16 accumulation
    # through its 100+ sequential convs measurably degrades reconstruction.
    waveform = vocoder(mel.float())

    sample_rate = int(getattr(vocoder.config, "output_sampling_rate", 16000))
    waveform = waveform.float().cpu()
    while waveform.ndim > 2:
        waveform = waveform.squeeze(0)
    return waveform.clamp(-1.0, 1.0), sample_rate


def write_video_with_audio(
    path: Path,
    frames: np.ndarray,
    fps: float,
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    """Mux video frames and an audio waveform into an mp4 (h264 + aac).

    Args:
        path: Output file path.
        frames: Video frames [T, H, W, 3] uint8 RGB.
        fps: Video frame rate.
        waveform: Audio [channels, samples] float32 in [-1, 1].
        sample_rate: Audio sample rate in Hz.
    """
    import av

    with av.open(str(path), mode="w") as container:
        vstream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        vstream.width, vstream.height = frames.shape[2], frames.shape[1]
        vstream.pix_fmt = "yuv420p"
        astream = container.add_stream("aac", rate=sample_rate)

        for frame in frames:
            for packet in vstream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)

        pcm = (waveform.numpy().clip(-1.0, 1.0) * 32767.0).astype(np.int16)
        layout = "stereo" if pcm.shape[0] == 2 else "mono"
        for start in range(0, pcm.shape[1], _AAC_FRAME_SAMPLES):
            chunk = np.ascontiguousarray(pcm[:, start : start + _AAC_FRAME_SAMPLES])
            aframe = av.AudioFrame.from_ndarray(chunk.reshape(1, -1, order="F"), format="s16", layout=layout)
            aframe.sample_rate = sample_rate
            aframe.pts = start
            for packet in astream.encode(aframe):
                container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)


def pixels_to_frames(pixels: torch.Tensor) -> np.ndarray:
    """Convert pixels [1, 3, T, H, W] in [-1, 1] to uint8 frames [T, H, W, 3]."""
    frames = pixels.squeeze(0).permute(1, 2, 3, 0)
    return ((frames.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).numpy()


def main() -> None:
    """Run the encode/decode round trip and write comparison mp4 files."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = LTX2Processor()
    _import_ltx2_classes()  # fail fast with a version hint before any download
    models = load_roundtrip_models(args.model_id, args.device)
    vocoder = load_vocoder(args.model_id, args.device)

    reference_video = None
    reference_audio = None
    if args.video is not None:
        stem = Path(args.video).stem
        logger.info("Encoding %s with the LTX2Processor preprocessing path...", args.video)
        reference_video, _ = processor.load_video(
            args.video, target_size=(args.height, args.width), num_frames=args.num_frames
        )
        reference_audio = processor.load_audio(args.video, args.num_frames)
        video_latents = processor.encode_video(reference_video, models, args.device)
        audio_latents = processor.encode_audio(args.video, args.num_frames, models, args.device)["audio_latents"]
    else:
        stem = Path(args.cache_file).stem
        logger.info("Loading cache file %s...", args.cache_file)
        cache = torch.load(args.cache_file, weights_only=True)
        video_latents, audio_latents = cache["video_latents"], cache["audio_latents"]

    logger.info("Video latents: %s | audio latents: %s", tuple(video_latents.shape), tuple(audio_latents.shape))

    logger.info("Decoding video latents...")
    decoded_pixels = decode_video_latents(video_latents, models["vae"], args.device)
    logger.info("Decoding audio latents through the vocoder...")
    decoded_waveform, vocoder_rate = decode_audio_latents(audio_latents, models["audio_vae"], vocoder, args.device)

    roundtrip_path = output_dir / f"{stem}_roundtrip.mp4"
    write_video_with_audio(
        roundtrip_path, pixels_to_frames(decoded_pixels), processor.VIDEO_FPS, decoded_waveform, vocoder_rate
    )
    logger.info("Wrote %s", roundtrip_path)

    if reference_video is not None:
        reference_path = output_dir / f"{stem}_reference.mp4"
        write_video_with_audio(
            reference_path,
            pixels_to_frames(reference_video),
            processor.VIDEO_FPS,
            reference_audio,
            processor.AUDIO_SAMPLE_RATE,
        )
        logger.info("Wrote %s", reference_path)

        # Quantitative check: VAE round-trip PSNR (typically > ~28 dB when the
        # encode path is correct; single digits mean broken normalization).
        num_frames = min(reference_video.shape[2], decoded_pixels.shape[2])
        mse = torch.mean((reference_video[:, :, :num_frames].float() - decoded_pixels[:, :, :num_frames]) ** 2)
        psnr = 10.0 * torch.log10(4.0 / mse)  # signal range [-1, 1] -> peak-to-peak 2
        rms = decoded_waveform.pow(2).mean().sqrt()
        logger.info("Video round-trip PSNR: %.2f dB | decoded audio RMS: %.4f", psnr.item(), rms.item())
        if psnr.item() < 20.0:
            logger.warning("PSNR below 20 dB - check latent normalization in the encode path!")


if __name__ == "__main__":
    main()
