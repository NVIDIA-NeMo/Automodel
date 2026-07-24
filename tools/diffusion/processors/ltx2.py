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
LTX-2.3 video+audio processor for preprocessing.

LTX-2.3 is a dual-stream model: the transformer jointly denoises video and
audio latents. This processor therefore encodes three modalities per clip:

- Video: ``AutoencoderKLLTX2Video`` latents ``[1, 128, F', H', W']``.
- Audio: 16 kHz stereo waveform -> log-mel spectrogram ->
  ``AutoencoderKLLTX2Audio`` latents ``[1, 8, L, 16]``, where
  ``L = round(num_frames / 24 * 25)`` (audio runs at 25 latent frames/s).
- Text: Gemma-3 stacked hidden states passed through the frozen
  ``LTX2TextConnectors`` offline, caching the per-modality conditioning
  streams the transformer cross-attends to. Connectors stay at pretrained
  weights (they are NOT finetuned in this recipe).

A/V sync contract: video frames are sampled at 24 fps from t=0 and audio is
trimmed/padded to exactly ``num_frames / 24`` seconds from t=0, so the audio
latent length matches what the transformer derives from the video latent
frame count. Text embeddings are stored padded to ``MAX_SEQUENCE_LENGTH``
tokens (left padding) so cache tensors collate with ``torch.cat``.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ._ltx2_audio import LTX2MelSpectrogram
from .base_video import BaseVideoProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)

_DIFFUSERS_VERSION_HINT = (
    "LTX-2 classes not found in the installed diffusers package. "
    "LTX-2 support requires a diffusers release that provides "
    "diffusers.pipelines.ltx2 (>= 0.37.0); upgrade with: uv pip install -U diffusers"
)


def _import_ltx2_classes():
    """Import the diffusers LTX-2 model classes, raising a version hint on failure."""
    try:
        from diffusers.models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    except ImportError as e:
        raise ImportError(_DIFFUSERS_VERSION_HINT) from e
    return AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio, LTX2TextConnectors


@ProcessorRegistry.register("ltx2")
class LTX2Processor(BaseVideoProcessor):
    """
    Processor for LTX-2.3 T2V (video + audio) models.

    LTX-2.3 uses:
    - AutoencoderKLLTX2Video (32x spatial, 8x temporal compression, 128 latent channels)
    - AutoencoderKLLTX2Audio over log-mel spectrograms (8 channels, 16 latent mel bins)
    - Gemma-3 text encoder (all hidden-state layers stacked) + frozen LTX2TextConnectors
    """

    # Fixed by the LTX-2 reference pipeline.
    VIDEO_FPS = 24.0
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_LATENT_FPS = 25.0  # 16000 / hop 160 / audio-VAE temporal compression 4
    AUDIO_LATENT_CHANNELS = 8
    AUDIO_LATENT_MEL_BINS = 16
    MAX_SEQUENCE_LENGTH = 1024

    @property
    def model_type(self) -> str:
        return "ltx2"

    @property
    def model_version(self) -> str:
        return "ltx2.3"

    @property
    def default_model_name(self) -> str:
        # Diffusers-layout conversion of Lightricks/LTX-2.3 (the official repo is weights-only).
        return "dg845/LTX-2.3-Diffusers"

    @property
    def supported_modes(self) -> List[str]:
        return ["video"]

    @property
    def frame_constraint(self) -> Optional[str]:
        # Video VAE temporal compression is 8: pixel frame count must be 8n+1.
        return "8n+1"

    @property
    def quantization(self) -> int:
        # Video VAE downsamples spatially by 32; pixel dims must be divisible by 32.
        return 32

    def load_models(self, model_name: str, device: str) -> Dict[str, Any]:
        """
        Load LTX-2.3 encoding components (all frozen).

        Args:
            model_name: HuggingFace model path (e.g., 'dg845/LTX-2.3-Diffusers')
            device: Device to load models on

        Returns:
            Dict containing vae, audio_vae, text_encoder, tokenizer, connectors,
            mel_transform, and dtype.
        """
        from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

        from nemo_automodel._diffusers._hf_cache import resolve_diffusion_model_dir

        AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio, LTX2TextConnectors = _import_ltx2_classes()

        dtype = torch.bfloat16

        logger.info("[LTX2] Loading models from %s...", model_name)
        model_name = resolve_diffusion_model_dir(model_name)

        logger.info("  Loading AutoencoderKLLTX2Video...")
        vae = AutoencoderKLLTX2Video.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype)
        vae.to(device).eval().requires_grad_(False)

        logger.info("  Loading AutoencoderKLLTX2Audio...")
        audio_vae = AutoencoderKLLTX2Audio.from_pretrained(model_name, subfolder="audio_vae", torch_dtype=dtype)
        audio_vae.to(device).eval().requires_grad_(False)

        logger.info("  Loading Gemma-3 text encoder...")
        # bf16 cast is deliberate: the reference pipeline runs Gemma in bf16;
        # fp32 hidden states produce different conditioning.
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=dtype
        )
        text_encoder.to(device).eval().requires_grad_(False)

        logger.info("  Loading tokenizer...")
        tokenizer = GemmaTokenizerFast.from_pretrained(model_name, subfolder="tokenizer")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("  Loading LTX2TextConnectors (frozen, applied offline)...")
        connectors = LTX2TextConnectors.from_pretrained(model_name, subfolder="connectors", torch_dtype=dtype)
        connectors.to(device).eval().requires_grad_(False)

        mel_transform = LTX2MelSpectrogram().to(device)

        logger.info("[LTX2] Models loaded successfully!")
        return {
            "vae": vae,
            "audio_vae": audio_vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "connectors": connectors,
            "mel_transform": mel_transform,
            "dtype": dtype,
        }

    def load_video(
        self,
        video_path: str,
        target_size: Tuple[int, int],
        num_frames: Optional[int] = None,
        resize_mode: str = "bilinear",
        center_crop: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load video frames retimed to 24 fps from t=0.

        Unlike the base loader (which samples frames uniformly across the whole
        clip and therefore stretches time), frames here are picked at fixed
        24 fps wall-clock positions starting at t=0 so the audio track trimmed
        to ``num_frames / 24`` seconds stays in sync.

        Args:
            video_path: Path to video file
            target_size: Target (height, width)
            num_frames: Number of frames to extract at 24 fps (required)
            resize_mode: Interpolation mode for resizing
            center_crop: Whether to center crop

        Returns:
            Tuple of:
                - video_tensor: Tensor of shape (1, C, T, H, W), normalized to [-1, 1]
                - first_frame: First frame as numpy array (H, W, C) in uint8

        Raises:
            ValueError: If the clip is shorter than ``num_frames / 24`` seconds.
        """
        import cv2

        if num_frames is None:
            raise ValueError("LTX2Processor requires an explicit num_frames (e.g. 121)")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or self.VIDEO_FPS
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            duration_needed = num_frames / self.VIDEO_FPS
            if total_frames / src_fps < duration_needed:
                raise ValueError(
                    f"{video_path}: clip too short ({total_frames / src_fps:.2f}s) for "
                    f"{num_frames} frames at {self.VIDEO_FPS} fps ({duration_needed:.2f}s)"
                )

            # Nearest source frame for each 24 fps target timestamp from t=0.
            frame_indices = np.round(np.arange(num_frames) * (src_fps / self.VIDEO_FPS)).astype(int)
            frame_indices = np.clip(frame_indices, 0, total_frames - 1)

            target_height, target_width = target_size
            interp_map = {
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "nearest": cv2.INTER_NEAREST,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4,
            }
            interpolation = interp_map.get(resize_mode, cv2.INTER_LINEAR)

            frames = []
            current_idx = -1
            for target_idx in frame_indices:
                if target_idx != current_idx + 1 and target_idx != current_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
                if target_idx != current_idx:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        raise ValueError(f"{video_path}: failed to decode frame {target_idx}")
                    current_idx = int(target_idx)
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    if center_crop:
                        scale = max(target_width / orig_width, target_height / orig_height)
                        new_w, new_h = int(orig_width * scale), int(orig_height * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
                        x0 = (new_w - target_width) // 2
                        y0 = (new_h - target_height) // 2
                        frame = frame[y0 : y0 + target_height, x0 : x0 + target_width]
                    else:
                        frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
                frames.append(frame)
        finally:
            cap.release()

        frames = np.array(frames, dtype=np.uint8)
        first_frame = frames[0].copy()
        return self.frames_to_tensor(frames), first_frame

    def load_audio(self, video_path: str, num_frames: int) -> torch.Tensor:
        """
        Decode the clip's audio track: 16 kHz stereo, exactly ``num_frames / 24`` seconds.

        Args:
            video_path: Path to video file (audio read from the same container)
            num_frames: Number of 24 fps video frames the audio must align to

        Returns:
            Waveform tensor of shape [2, samples] where
            ``samples = round(num_frames / 24 * 16000)``. Zeros (with a warning)
            when the container has no audio stream.
        """
        try:
            import av
        except ImportError as e:
            raise ImportError(
                "LTX-2 preprocessing requires PyAV for audio decoding. "
                "Install with: uv pip install 'nemo_automodel[diffusion-media]'"
            ) from e

        target_samples = round(num_frames / self.VIDEO_FPS * self.AUDIO_SAMPLE_RATE)

        with av.open(video_path) as container:
            if not container.streams.audio:
                logger.warning("%s: no audio stream; using silence", video_path)
                return torch.zeros(2, target_samples)

            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=self.AUDIO_SAMPLE_RATE)
            chunks: List[np.ndarray] = []
            for frame in container.decode(container.streams.audio[0]):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray())  # [2, samples] planar float
            # Flush any samples buffered inside the resampler.
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray())

        if not chunks:
            logger.warning("%s: audio stream decoded to zero samples; using silence", video_path)
            return torch.zeros(2, target_samples)

        audio = torch.from_numpy(np.concatenate(chunks, axis=1)).float()  # [2, total_samples]
        if audio.shape[1] >= target_samples:
            audio = audio[:, :target_samples]
        else:
            audio = torch.nn.functional.pad(audio, (0, target_samples - audio.shape[1]))
        return audio.contiguous()

    def encode_video(
        self,
        video_tensor: torch.Tensor,
        models: Dict[str, Any],
        device: str,
        deterministic: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode video tensor to normalized LTX-2 latent space.

        Args:
            video_tensor: Video tensor (1, C, T, H, W), normalized to [-1, 1]
            models: Dict containing 'vae'
            device: Device to use
            deterministic: If True, use the distribution mode instead of sampling

        Returns:
            Normalized latent tensor (1, 128, T', H', W') in bfloat16 on CPU.
        """
        vae = models["vae"]
        dtype = models.get("dtype", torch.bfloat16)
        video_tensor = video_tensor.to(device=device, dtype=dtype)

        with torch.no_grad():
            dist = vae.encode(video_tensor, return_dict=False)[0]
            latents = dist.mode() if deterministic else dist.sample()

        mean, std = self._latent_stats(vae)
        mean = mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        std = std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - mean) * vae.config.scaling_factor / std
        return latents.detach().cpu().to(torch.bfloat16)

    def encode_audio(
        self,
        video_path: str,
        num_frames: int,
        models: Dict[str, Any],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode the clip's audio track to normalized LTX-2 audio latents.

        Pipeline: waveform [2, samples] -> magnitude mel [1, 2, 64, T_mel] ->
        log(clamp(min=1e-5)) -> permute to [1, 2, T_mel, 64] -> audio VAE mode()
        -> per-(channel x mel-bin) normalization on the flattened [1, L, 128]
        layout (matching the LTX-2 reference).

        Args:
            video_path: Path to the source video file
            num_frames: Number of 24 fps video frames (defines audio duration)
            models: Dict containing 'audio_vae' and 'mel_transform'
            device: Device to use

        Returns:
            Dict with 'audio_latents': [1, 8, L, 16] bfloat16 on CPU, where
            ``L = round(num_frames / 24 * 25)``.

        Raises:
            ValueError: If the encoded latent length deviates from the expected
                video-aligned length by more than 1 frame.
        """
        audio_vae = models["audio_vae"]
        mel_transform = models["mel_transform"]

        waveform = self.load_audio(video_path, num_frames)  # [2, samples]
        waveform = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)  # [1, 2, samples]

        with torch.no_grad():
            mel = mel_transform(waveform)  # [1, 2, 64, T_mel]
            mel = torch.log(torch.clamp(mel, min=1e-5))
            mel = mel.permute(0, 1, 3, 2).contiguous()  # [1, 2, T_mel, 64]
            dist = audio_vae.encode(mel.to(audio_vae.dtype), return_dict=False)[0]
            latents = dist.mode()  # [1, C, L, M]

        b, c, t, m = latents.shape
        mean, std = self._latent_stats(audio_vae)
        mean = mean.to(latents.device, latents.dtype)
        std = std.to(latents.device, latents.dtype)
        # Per-(C*M) stats broadcast over the flattened [B, L, C*M] layout.
        flat = latents.permute(0, 2, 1, 3).reshape(b, t, c * m)
        flat = (flat - mean) / std
        latents = flat.view(b, t, c, m).permute(0, 2, 1, 3).contiguous()

        expected_frames = round(num_frames / self.VIDEO_FPS * self.AUDIO_LATENT_FPS)
        if abs(t - expected_frames) > 1:
            raise ValueError(
                f"{video_path}: audio latent length {t} deviates from expected {expected_frames} "
                f"({num_frames} frames / {self.VIDEO_FPS} fps * {self.AUDIO_LATENT_FPS} latents/s) "
                "- audio/video alignment is broken"
            )

        return {"audio_latents": latents.detach().cpu().to(torch.bfloat16)}

    def encode_text(
        self,
        prompt: str,
        models: Dict[str, Any],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text through Gemma-3 and the frozen LTX2TextConnectors.

        All Gemma hidden-state layers are stacked and flattened to
        [1, T, hidden * num_layers], then split by the connectors into the
        video-stream and audio-stream conditioning tensors the transformer
        cross-attends to.

        Args:
            prompt: Text prompt
            models: Dict containing tokenizer, text_encoder, connectors
            device: Device to use

        Returns:
            Dict containing:
                - text_embeddings: video-stream connector output [1, T, D_v]
                - audio_text_embeddings: audio-stream connector output [1, T, D_a]
                - text_mask: post-connector attention mask [1, T]
        """
        tokenizer = models["tokenizer"]
        text_encoder = models["text_encoder"]
        connectors = models["connectors"]

        inputs = tokenizer(
            [prompt.strip()],
            padding="max_length",
            max_length=self.MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # Stack all hidden-state layers and flatten: [B, T, H, num_layers] -> [B, T, H*num_layers].
            hidden_states = torch.stack(outputs.hidden_states, dim=-1)
            prompt_embeds = self._pack_text_embeds(hidden_states, attention_mask, connectors)
            video_embeds, audio_embeds, connector_mask = connectors(prompt_embeds, attention_mask)

        return {
            "text_embeddings": video_embeds.detach().cpu().to(torch.bfloat16),
            "audio_text_embeddings": audio_embeds.detach().cpu().to(torch.bfloat16),
            "text_mask": connector_mask.detach().cpu(),
        }

    @staticmethod
    def _pack_text_embeds(hidden_states: torch.Tensor, attention_mask: torch.Tensor, connectors: Any) -> torch.Tensor:
        """
        Flatten stacked Gemma hidden states into the connectors' expected input.

        Newer diffusers normalizes inside ``LTX2TextConnectors.forward`` (its
        signature gains a ``padding_side`` argument) and takes the raw flattened
        states; older versions require ``LTX2Pipeline._pack_text_embeds`` first.

        Args:
            hidden_states: Stacked Gemma hidden states [B, T, H, num_layers].
            attention_mask: Tokenizer attention mask [B, T].
            connectors: The loaded LTX2TextConnectors module.

        Returns:
            Connector input embeddings [B, T, H * num_layers].
        """
        connectors_normalize = "padding_side" in inspect.signature(connectors.forward).parameters
        if connectors_normalize:
            return hidden_states.flatten(2, 3)

        from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline

        sequence_lengths = attention_mask.sum(dim=-1)
        return LTX2Pipeline._pack_text_embeds(hidden_states, sequence_lengths, device=hidden_states.device)

    @staticmethod
    def _latent_stats(vae: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (latents_mean, latents_std) from VAE buffers or config lists."""
        mean = getattr(vae, "latents_mean", None)
        std = getattr(vae, "latents_std", None)
        if mean is None or std is None:
            mean = torch.tensor(vae.config.latents_mean)
            std = torch.tensor(vae.config.latents_std)
        return mean, std

    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct the cache dictionary for LTX-2.3.

        Args:
            latent: Encoded video latent tensor (1, 128, T', H', W')
            text_encodings: Dict from encode_text()
            metadata: Additional metadata; must contain 'audio_latents' from
                encode_audio() (merged in by the preprocessing driver).

        Returns:
            Dict to save with torch.save(), containing both modality latents
            and both text-conditioning streams.

        Raises:
            ValueError: If 'audio_latents' is missing from metadata.
        """
        audio_latents = metadata.get("audio_latents")
        if audio_latents is None:
            raise ValueError(
                "LTX2Processor requires audio latents in metadata; run preprocessing through "
                "a driver that calls encode_audio() (tools/diffusion/preprocessing_multiprocess.py)"
            )
        return {
            # Modality latents
            "video_latents": latent,
            "audio_latents": audio_latents,
            # Text conditioning (post-connector, per stream)
            "text_embeddings": text_encodings["text_embeddings"],
            "audio_text_embeddings": text_encodings["audio_text_embeddings"],
            "text_mask": text_encodings["text_mask"],
            # Resolution and bucketing info
            "original_resolution": metadata.get("original_resolution"),
            "bucket_resolution": metadata.get("bucket_resolution"),
            "bucket_id": metadata.get("bucket_id"),
            "aspect_ratio": metadata.get("aspect_ratio"),
            # Video info
            "num_frames": metadata.get("num_frames"),
            "prompt": metadata.get("prompt"),
            "video_path": metadata.get("video_path"),
            # Processing settings
            "deterministic_latents": metadata.get("deterministic", True),
            "model_version": self.model_version,
            "processing_mode": metadata.get("mode", "video"),
            "model_type": self.model_type,
        }
