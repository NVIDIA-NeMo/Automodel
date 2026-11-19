# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Two-tower video backbone of the chunk-causal window.

The encoder tower reads ``[registers | anchor | text | noised chunks | clean chunks]`` and
returns the conditioning of the noised tokens plus the tap features for the action
expert. The decoder tower reads ``[registers | noised chunks]`` at the unpatched latent
resolution, modulated per token by the encoder output, and returns the video velocity.

Rotary positions: each latent frame owns a block of ``FRAME_STRIDE`` positions; inside a
frame a token sits at ``1 + row * ROW_STRIDE + column``. Registers are at position 0. In
the encoder the anchor takes frame block 0, the text frame block 1 and the noised frames
start at frame block 2; a clean frame shares the positions of its noised twin.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.models.praxis.config import BackboneConfig, TowerConfig
from nemo_automodel.components.models.praxis.layers import (
    IntegerRotaryEmbedding,
    TransformerStack,
    ZeroCenteredRMSNorm,
    build_moe_config,
    modulate,
)
from nemo_automodel.components.models.praxis.masks import (
    KIND_CLEAN,
    KIND_GLOBAL,
    KIND_NOISED,
    MaskCache,
    Slot,
    block_mask,
    chunk_causal_rule,
    dense_mask,
    window_tags,
)

ROW_STRIDE = 360
FRAME_STRIDE = 129_600


@dataclass
class EncoderFeatures:
    """Encoder outputs that the action expert reads.

    Attributes:
        taps: Block outputs at the tap blocks, each of shape [batch, tokens, hidden].
        embeddings: The encoder input sequence (block 0 input) of shape [batch, tokens, hidden].
        positions: Rotary positions of the sequence, shape [batch, tokens].
        global_spans: ``(start, length)`` of the ``register``, ``anchor`` and ``text`` runs.
        noised_span: ``(start, length)`` of the noised chunk tokens.
        clean_span: ``(start, length)`` of the clean chunk tokens, ``None`` for one chunk.
        frames_per_chunk: Latent frames per chunk.
        tokens_per_frame: Tokens per latent frame.
        n_chunks: Chunks in the window.
    """

    taps: list[Tensor]
    embeddings: Tensor
    positions: Tensor
    global_spans: dict[str, tuple[int, int]]
    noised_span: tuple[int, int]
    clean_span: tuple[int, int] | None
    frames_per_chunk: int
    tokens_per_frame: int
    n_chunks: int


@dataclass
class BackboneOutput:
    """``velocity`` of shape [batch, channels, frames, height, width] (``None`` without decoding) and the encoder features."""

    velocity: Tensor | None
    features: EncoderFeatures


class FourierTimestep(nn.Module):
    """Fixed random Fourier features of the flow time: ``0.5 * cos(2 pi (t f + phi))``."""

    freqs: Tensor
    phases: Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.zeros(1, dim))
        self.register_buffer("phases", torch.zeros(1, dim))

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps.

        Args:
            t: Tensor of shape [..., 1] or [...]; the last axis is broadcast against the features.

        Returns:
            Tensor of shape [..., dim] in the dtype of ``t``.
        """
        x = t[..., None].float() * self.freqs.float() + self.phases.float()
        x = x.to(t.dtype)
        return (x * 2 * torch.pi).cos() * 0.5


class TextInputProjection(nn.Module):
    """MLP from text-encoder states to trunk tokens, ending in a zero-centred RMSNorm."""

    def __init__(self, c_in: int, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(c_in, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.norm = ZeroCenteredRMSNorm(dim, cast_dtype=None)

    def forward(self, x: Tensor) -> Tensor:
        """Project text states.

        Args:
            x: Tensor of shape [batch, text_tokens, c_in].

        Returns:
            Tensor of shape [batch, text_tokens, hidden].
        """
        return self.norm(self.fc2(F.gelu(self.fc1(x))))


class DecoderOutputProjection(nn.Module):
    """Zero-centred RMSNorm followed by a linear map to latent channels."""

    def __init__(self, dim: int, c_out: int) -> None:
        super().__init__()
        self.norm = ZeroCenteredRMSNorm(dim, cast_dtype=None)
        self.linear = nn.Linear(dim, c_out)

    def forward(self, x: Tensor) -> Tensor:
        """Project decoder tokens.

        Args:
            x: Tensor of shape [batch, tokens, hidden].

        Returns:
            Tensor of shape [batch, tokens, c_out].
        """
        return self.linear(self.norm(x))


def video_positions(frames: int, height: int, width: int, start: int) -> Tensor:
    """Rotary positions of a video run in frame-major order.

    Returns:
        Int64 tensor of shape [frames * height * width].
    """
    frame = torch.arange(frames).view(-1, 1, 1) * FRAME_STRIDE
    row = torch.arange(height).view(1, -1, 1) * ROW_STRIDE
    col = torch.arange(width).view(1, 1, -1)
    return (start + 1 + frame + row + col).reshape(-1)


def encoder_positions(
    num_registers: int,
    anchor_hw: tuple[int, int],
    num_text: int,
    noised_thw: tuple[int, int, int],
    num_clean_frames: int,
    shifts: tuple[int, int, int],
) -> Tensor:
    """Rotary positions of the encoder sequence ``[registers | anchor | text | noised | clean]``.

    Args:
        num_registers: Register tokens (position 0).
        anchor_hw: Latent grid of the anchor frame.
        num_text: Text tokens.
        noised_thw: Frames and latent grid of the noised chunks.
        num_clean_frames: Frames of the clean copies.
        shifts: Random ``(anchor, text, video)`` offsets of training; zeros at inference.

    Returns:
        Int64 tensor of shape [tokens].
    """
    anchor_shift, text_shift, video_shift = shifts
    regs = torch.zeros(num_registers, dtype=torch.int64)
    anchor = video_positions(1, anchor_hw[0], anchor_hw[1], anchor_shift)
    text = torch.arange(1, num_text + 1) + FRAME_STRIDE + text_shift
    noised = video_positions(noised_thw[0], noised_thw[1], noised_thw[2], 2 * FRAME_STRIDE + video_shift)
    clean = noised[: num_clean_frames * noised_thw[1] * noised_thw[2]]
    return torch.cat([regs, anchor, text, noised, clean])


def unpatch(x: Tensor, patch: tuple[int, int]) -> Tensor:
    """Move the channel patch of ``[batch, channels * ph * pw, frames, h, w]`` into space.

    Returns:
        Tensor of shape [batch, channels, frames, h * ph, w * pw].
    """
    b, c, t, h, w = x.shape
    ph, pw = patch
    x = x.view(b, c // (ph * pw), ph, pw, t, h, w)
    return x.permute(0, 1, 4, 5, 2, 6, 3).reshape(b, -1, t, h * ph, w * pw)


def patchify(x: Tensor, patch: tuple[int, int]) -> Tensor:
    """Inverse of :func:`unpatch`: fold ``(ph, pw)`` spatial patches into the channels.

    Returns:
        Tensor of shape [batch, channels * ph * pw, frames, h // ph, w // pw].
    """
    b, c, t, hh, ww = x.shape
    ph, pw = patch
    x = x.view(b, c, t, hh // ph, ph, ww // pw, pw)
    return x.permute(0, 1, 4, 6, 2, 3, 5).reshape(b, c * ph * pw, t, hh // ph, ww // pw)


def _expand_rows(rows: Tensor, chunk_tokens: list[int]) -> Tensor:
    """Repeat row ``k`` of ``rows`` ([batch, chunks, hidden]) over ``chunk_tokens[k]`` tokens."""
    return torch.cat([rows[:, k : k + 1].expand(-1, n, -1) for k, n in enumerate(chunk_tokens)], dim=1)


@dataclass(frozen=True)
class MoeTrainingSettings:
    """Load-balancing terms of the routers during training (both zero at inference).

    Attributes:
        gate_bias_update_factor: Step of the per-expert selection bias toward balanced load,
            applied by :meth:`PraxisPolicy.update_moe_gate_bias` after each optimizer step.
        aux_loss_coeff: Weight of the sequence-wise balance loss of each router
            (``experts * sum_i f_i p_i``), injected through the routing weights.
    """

    gate_bias_update_factor: float = 0.0
    aux_loss_coeff: float = 0.0


def _make_stack(
    cfg: BackboneConfig,
    tower: TowerConfig,
    backend: BackendConfig,
    dtype: torch.dtype,
    moe_training: MoeTrainingSettings,
) -> TransformerStack:
    moe_config = build_moe_config(
        dim=cfg.dim,
        expert_hidden=cfg.moe.expert_hidden,
        num_routed_experts=cfg.moe.num_routed_experts,
        top_k_routed=cfg.moe.top_k_routed,
        routed_scaling_factor=cfg.moe.routed_scaling_factor,
        dtype=dtype,
        gate_bias_update_factor=moe_training.gate_bias_update_factor,
        aux_loss_coeff=moe_training.aux_loss_coeff,
    )
    rope = IntegerRotaryEmbedding(
        cfg.attention.head_dim, cfg.rope.effective_dim, cfg.rope.base, cfg.rope.efficient_cache_max
    )
    return TransformerStack(
        depth=tower.depth,
        dense_blocks=tower.dense_blocks,
        dim=cfg.dim,
        num_heads=cfg.attention.num_heads,
        num_kv_heads=cfg.attention.num_kv_heads,
        head_dim=cfg.attention.head_dim,
        qk_norm_eps=cfg.attention.qk_norm_eps,
        norm_eps=cfg.norm_eps,
        dense_hidden=cfg.dense_ffn_hidden,
        moe_config=moe_config,
        rope=rope,
        modulation_dim=tower.modulation_dim,
        modulation_dtype=cfg.modulation_dtype,
        backend=backend,
        dtype=dtype,
    )


class EncoderTower(nn.Module):
    """The encoder tower (see the module docstring)."""

    def __init__(
        self, cfg: BackboneConfig, backend: BackendConfig, dtype: torch.dtype, moe_training: MoeTrainingSettings
    ) -> None:
        super().__init__()
        tower = cfg.encoder
        if tower.text_in is None or tower.text_in.c_in is None or tower.video_in.c_in is None:
            raise ValueError("the encoder tower needs text and video input projections")
        self.registers = nn.Parameter(torch.zeros(1, cfg.num_registers, cfg.dim))
        self.t_embedding = FourierTimestep(cfg.timestep_dim)
        self.time_proj = nn.Linear(cfg.timestep_dim, 2 * cfg.dim)
        self.text_in = TextInputProjection(tower.text_in.c_in, cfg.dim)
        self.video_in = nn.Linear(tower.video_in.c_in, cfg.dim)
        self.video_out = nn.Linear(cfg.dim, cfg.dim)
        self.core = _make_stack(cfg, tower, backend, dtype, moe_training)
        self.tap_blocks = tower.tap_blocks
        self.modulation_dtype = cfg.modulation_dtype
        self.block_size = cfg.attention.flex_block_size
        self.use_flex = backend.attn == "flex"
        self._masks = MaskCache()

    def _video_tokens(self, latents: Tensor) -> Tensor:
        """``[batch, channels, frames, h, w]`` latents to ``[batch, frames * h * w, hidden]`` tokens."""
        return self.video_in(latents.movedim(1, -1)).flatten(1, -2)

    def _modulate_chunks(self, tokens: Tensor, timesteps: Tensor, chunk_tokens: list[int]) -> Tensor:
        shift, scale = self.time_proj(self.t_embedding(timesteps)).chunk(2, dim=-1)
        return modulate(
            tokens, _expand_rows(shift, chunk_tokens), _expand_rows(1 + scale, chunk_tokens), self.modulation_dtype
        )

    def _mask(self, slots: tuple[Slot, ...], chunk: int, device: torch.device) -> BlockMask | Tensor:
        key = (slots, chunk, str(device), self.use_flex)

        def build() -> BlockMask | Tensor:
            tags = window_tags(slots, chunk, device)
            rule = chunk_causal_rule(tags)
            if self.use_flex and device.type == "cuda":
                return block_mask(rule, tags.length, tags.length, device, self.block_size)
            return dense_mask(rule, tags.length, tags.length, device)

        return self._masks.get(key, build)

    def forward(
        self,
        text: Tensor,
        anchor: Tensor,
        clean: Tensor | None,
        noised: Tensor,
        timesteps: Tensor,
        rope_shifts: tuple[int, int, int] = (0, 0, 0),
    ) -> tuple[Tensor, EncoderFeatures]:
        """Encode a window of ``K`` chunks.

        Args:
            text: Text-encoder states of shape [batch, text_tokens, text_dim].
            anchor: Anchor latent of shape [batch, channels, 1, h, w].
            clean: Clean copies of chunks ``0 .. K-2`` of shape [batch, channels, (K-1) * F, h, w], or ``None`` for ``K == 1``.
            noised: Noised chunks of shape [batch, channels, K * F, h, w].
            timesteps: Flow time of each noised chunk, shape [batch, K].
            rope_shifts: ``(anchor, text, video)`` rotary offsets of training.

        Returns:
            The conditioning of the noised tokens, shape [batch, K * F, h, w, hidden], and the
            :class:`EncoderFeatures`.
        """
        kk = int(timesteps.shape[1])
        frames, height, width = (int(d) for d in noised.shape[2:])
        if frames % kk:
            raise ValueError(f"{frames} noised frames do not split into {kk} chunks")
        chunk = frames // kk
        n_clean_frames = 0 if clean is None else int(clean.shape[2])
        if n_clean_frames != (kk - 1) * chunk:
            raise ValueError(f"{kk} chunks need {(kk - 1) * chunk} clean frames, got {n_clean_frames}")
        tpf = height * width
        anchor_tok = self._video_tokens(anchor)
        text_tok = self.text_in(text)
        noised_tok = self._modulate_chunks(self._video_tokens(noised), timesteps, [chunk * tpf] * kk)
        parts = [self.registers.to(noised_tok.dtype), anchor_tok, text_tok, noised_tok]
        if clean is not None:
            clean_tok = self._modulate_chunks(
                self._video_tokens(clean), torch.zeros_like(timesteps), [chunk * tpf] * (kk - 1)
            )
            parts.append(clean_tok)
        x = torch.cat(parts, dim=1)

        n_reg, n_anchor, n_text = self.registers.shape[1], anchor_tok.shape[1], text_tok.shape[1]
        n_noised, n_clean = frames * tpf, n_clean_frames * tpf
        positions = encoder_positions(
            n_reg,
            (int(anchor.shape[3]), int(anchor.shape[4])),
            n_text,
            (frames, height, width),
            n_clean_frames,
            rope_shifts,
        )
        positions = positions.unsqueeze(0).to(x.device)
        slots = (
            Slot(KIND_GLOBAL, 1, n_reg),
            Slot(KIND_GLOBAL, 1, n_anchor),
            Slot(KIND_GLOBAL, 1, n_text),
            Slot(KIND_NOISED, frames, tpf),
        )
        if n_clean:
            slots = (*slots, Slot(KIND_CLEAN, n_clean_frames, tpf))
        mask = self._mask(slots, chunk, x.device)
        out, taps = self.core(x, positions, mask, None, self.tap_blocks)
        n_global = n_reg + n_anchor + n_text
        projected = self.video_out(out.narrow(1, n_global, n_noised + n_clean))
        cond = projected.narrow(1, 0, n_noised).view(noised.shape[0], frames, height, width, -1)
        features = EncoderFeatures(
            taps=taps,
            embeddings=x,
            positions=positions,
            global_spans={"register": (0, n_reg), "anchor": (n_reg, n_anchor), "text": (n_reg + n_anchor, n_text)},
            noised_span=(n_global, n_noised),
            clean_span=(n_global + n_noised, n_clean) if n_clean else None,
            frames_per_chunk=chunk,
            tokens_per_frame=tpf,
            n_chunks=kk,
        )
        return cond, features


class DecoderTower(nn.Module):
    """The decoder tower (see the module docstring)."""

    def __init__(
        self, cfg: BackboneConfig, backend: BackendConfig, dtype: torch.dtype, moe_training: MoeTrainingSettings
    ) -> None:
        super().__init__()
        tower = cfg.decoder
        if tower.video_in.c_in is None:
            raise ValueError("the decoder tower needs a video input projection")
        c_out = tower.video_in.c_in if tower.video_out.c_out is None else tower.video_out.c_out
        self.registers = nn.Parameter(torch.zeros(1, cfg.num_registers, cfg.dim))
        self.t_embedding = FourierTimestep(cfg.timestep_dim)
        self.time_proj = nn.Linear(cfg.timestep_dim, 2 * cfg.dim)
        self.video_in = nn.Linear(tower.video_in.c_in, cfg.dim)
        self.video_out = DecoderOutputProjection(cfg.dim, c_out)
        self.core = _make_stack(cfg, tower, backend, dtype, moe_training)
        self.patch_size = (tower.unpatch_factor[1], tower.unpatch_factor[2])
        self.modulation_dtype = cfg.modulation_dtype
        self.block_size = cfg.attention.flex_block_size
        self.use_flex = backend.attn == "flex"
        self._masks = MaskCache()

    def _mask(self, slots: tuple[Slot, ...], chunk: int, device: torch.device) -> BlockMask | Tensor:
        key = (slots, chunk, str(device), self.use_flex)

        def build() -> BlockMask | Tensor:
            tags = window_tags(slots, chunk, device)
            rule = chunk_causal_rule(tags)
            if self.use_flex and device.type == "cuda":
                return block_mask(rule, tags.length, tags.length, device, self.block_size)
            return dense_mask(rule, tags.length, tags.length, device)

        return self._masks.get(key, build)

    def forward(self, noised: Tensor, cond: Tensor, timesteps: Tensor, rope_shift: int = 0) -> Tensor:
        """Predict the velocity of the noised chunks.

        Args:
            noised: Noised latents of shape [batch, channels, K * F, h, w].
            cond: Encoder conditioning of shape [batch, K * F, h, w, hidden].
            timesteps: Flow time of each chunk, shape [batch, K].
            rope_shift: Rotary offset of the video positions in training.

        Returns:
            Velocity of shape [batch, channels, K * F, h, w].
        """
        ph, pw = self.patch_size
        kk = int(timesteps.shape[1])
        pixels = unpatch(noised, self.patch_size).movedim(1, -1)
        tok = self.video_in(pixels)
        b, frames, height, width, _ = tok.shape
        if frames % kk:
            raise ValueError(f"{frames} noised frames do not split into {kk} chunks")
        chunk = frames // kk
        tpf = height * width
        shift, scale = self.time_proj(self.t_embedding(timesteps)).chunk(2, dim=-1)
        rows = [chunk * tpf] * kk
        flat = modulate(
            tok.flatten(1, -2), _expand_rows(shift, rows), _expand_rows(1 + scale, rows), self.modulation_dtype
        )
        x = torch.cat([self.registers.to(flat.dtype), flat], dim=1)

        _, cf, h, w, c = cond.shape
        up = cond[:, :, :, None, :, None, :].expand(b, cf, h, ph, w, pw, c).reshape(b, cf * h * ph * w * pw, c)
        n_reg = self.registers.shape[1]
        x_mod = torch.cat([torch.zeros(b, n_reg, c, device=up.device, dtype=up.dtype), up], dim=1)

        positions = torch.cat(
            [torch.zeros(n_reg, dtype=torch.int64), video_positions(frames, height, width, rope_shift)]
        )
        positions = positions.unsqueeze(0).to(x.device)
        slots = (Slot(KIND_GLOBAL, 1, n_reg), Slot(KIND_NOISED, frames, tpf))
        mask = self._mask(slots, chunk, x.device)
        out, _ = self.core(x, positions, mask, x_mod, ())
        v = self.video_out(out.narrow(1, n_reg, flat.shape[1]))
        v = v.view(b, frames, height, width, -1).movedim(-1, 1)
        return patchify(v, self.patch_size)


class VideoBackbone(nn.Module):
    """Encoder tower and decoder tower of one denoising step."""

    def __init__(
        self,
        cfg: BackboneConfig,
        backend: BackendConfig,
        dtype: torch.dtype,
        moe_training: MoeTrainingSettings | None = None,
    ) -> None:
        super().__init__()
        if cfg.layout != "chunk_causal":
            raise ValueError(f"unsupported backbone layout {cfg.layout!r}")
        moe_training = moe_training or MoeTrainingSettings()
        self.encoder = EncoderTower(cfg, backend, dtype, moe_training)
        self.decoder = DecoderTower(cfg, backend, dtype, moe_training)

    def forward(
        self,
        text: Tensor,
        anchor: Tensor,
        clean: Tensor | None,
        noised: Tensor,
        timesteps: Tensor,
        decode: bool = True,
        rope_shifts: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> BackboneOutput:
        """Run one denoising step of a window.

        Args:
            text: Text-encoder states of shape [batch, text_tokens, text_dim].
            anchor: Anchor latent of shape [batch, channels, 1, h, w].
            clean: Clean copies of shape [batch, channels, (K-1) * F, h, w], or ``None``.
            noised: Noised chunks of shape [batch, channels, K * F, h, w].
            timesteps: Flow time of each chunk, shape [batch, K].
            decode: Run the decoder tower; otherwise ``velocity`` is ``None``.
            rope_shifts: ``(encoder anchor, encoder text, encoder video, decoder video)`` offsets.

        Returns:
            :class:`BackboneOutput` with the velocity of shape [batch, channels, K * F, h, w].
        """
        cond, features = self.encoder(text, anchor, clean, noised, timesteps, rope_shifts[:3])
        if not decode:
            return BackboneOutput(velocity=None, features=features)
        return BackboneOutput(velocity=self.decoder(noised, cond, timesteps, rope_shifts[3]), features=features)
