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

"""Flow-matching training objective of the Praxis policy.

A training window holds ``K`` clean chunk latents, the anchor latent, the text states, the
normalized action tokens and the boundary poses. One step noises the window and asks the
policy for the video and action velocities:

* Video: the whole window first gets one timestep ``t0`` and one noise ``eps0``; the target
  is ``eps0 - x0``. Then every chunk ``k`` is re-noised with its own timestep ``t_k`` from the
  bf16-rounded ``x_t``, so that the K noised chunks share one velocity target. The clean copies
  of chunks ``0 .. K-2`` are the recovered clean latents. Timesteps follow the shifted law
  ``t = s u / (1 + (s - 1) u)``.
* Actions: every chunk gets an independent uniform timestep; the target is ``eps - x0``.
* Text dropout swaps the text for the encoded blank caption (the unconditional branch of
  text guidance); proprio dropout zeroes the boundary poses.
* Training also moves the rotary positions of the anchor, the text and the video by random
  offsets inside one frame block so the model does not memorize absolute positions.

The loss is ``video_weight * mse(video) + action_weight * mse(action)``; a window whose action
loss is not finite or above ``action_guard`` contributes an action loss of exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nemo_automodel.components.models.praxis.backbone import FRAME_STRIDE, ROW_STRIDE
from nemo_automodel.components.models.praxis.model import PraxisPolicy


@dataclass(frozen=True)
class ObjectiveSettings:
    """Weights, timestep law and dropout rates of the objective."""

    video_weight: float = 1.0
    action_weight: float = 1.0
    video_timestep_shift: float = 5.0
    text_dropout: float = 0.1
    proprio_dropout: float = 0.1
    action_guard: float = 5.0
    # False trains only the action expert: the decoder tower is skipped, the backbone runs
    # without autograd and ``loss_video`` is reported as NaN.
    train_backbone: bool = True


@dataclass
class EncodedWindow:
    """One encoded training window (batch size 1).

    Attributes:
        text: Text-encoder states of shape [1, text_tokens, text_dim] (bf16).
        anchor: Anchor latent of shape [1, channels, 1, h, w].
        chunks: ``K`` clean chunk latents, each of shape [1, channels, F, h, w].
        actions: Normalized action tokens of shape [1, K * F, action_patch_dim] (float32), chunk-major.
        proprio: Normalized boundary poses of shape [1, K, proprio_patch_dim] (float32).
    """

    text: Tensor
    anchor: Tensor
    chunks: list[Tensor]
    actions: Tensor
    proprio: Tensor


@dataclass
class Draws:
    """The random values of one window, drawn before the forward pass.

    Attributes:
        text_u: Uniform value of shape [1] for the text dropout.
        video_u0: Uniform value of shape [1] for the window timestep.
        video_noise: Window noise of shape [1, channels, K * F, h, w] (bf16).
        video_u: Uniform values of shape [K] for the per-chunk timesteps.
        rope_shifts: ``(encoder anchor, encoder text, encoder video, decoder video)`` offsets.
        action_u: Uniform values of shape [K] for the action timesteps.
        action_noise: Action noise of shape [1, K * F, action_patch_dim] (float32).
        proprio_u: Uniform value of shape [1] for the proprio dropout.
    """

    text_u: Tensor
    video_u0: Tensor
    video_noise: Tensor
    video_u: Tensor
    rope_shifts: tuple[int, int, int, int]
    action_u: Tensor
    action_noise: Tensor
    proprio_u: Tensor


def shifted_timesteps(u: Tensor, shift: float) -> Tensor:
    """``shift * u / (1 + (shift - 1) * u)`` elementwise."""
    return shift * u / (1.0 + (shift - 1.0) * u)


@dataclass
class LossOutput:
    """Loss and detached metrics of one window.

    Attributes:
        loss: Scalar float32 loss with gradients.
        loss_video: Detached video loss.
        loss_action: Detached action loss before the guard.
        action_dropped: 1.0 when the guard zeroed the action loss, else 0.0.
    """

    loss: Tensor
    loss_video: Tensor
    loss_action: Tensor
    action_dropped: Tensor


class PraxisObjective(nn.Module):
    """The training objective around a :class:`PraxisPolicy`."""

    def __init__(self, policy: PraxisPolicy, settings: ObjectiveSettings, text_uncond: Tensor) -> None:
        """Wrap the policy.

        Args:
            policy: The model to train.
            settings: Objective settings.
            text_uncond: Encoded blank caption of shape [1, uncond_tokens, text_dim] used by the text dropout.
        """
        super().__init__()
        if text_uncond.dim() != 3 or text_uncond.shape[0] != 1:
            raise ValueError(f"text_uncond must be [1, tokens, dim], got {tuple(text_uncond.shape)}")
        self.policy = policy
        self.settings = settings
        self.register_buffer("text_uncond", text_uncond.detach().to(torch.bfloat16), persistent=False)

    def draw(self, batch: EncodedWindow, generator: torch.Generator) -> Draws:
        """Draw the random values of ``batch`` from a CPU ``generator`` in a fixed order."""
        s = self.settings
        text_u = torch.rand(1, generator=generator)
        video_u0 = torch.rand(1, generator=generator)
        shape = list(batch.chunks[0].shape)
        shape[2] = sum(int(c.shape[2]) for c in batch.chunks)
        video_noise = torch.randn(tuple(shape), generator=generator)
        kk = len(batch.chunks)
        video_u = torch.rand(kk, generator=generator)
        n_text = int(batch.text.shape[1])
        if self.training and bool(text_u < s.text_dropout):
            n_text = int(self.text_uncond.shape[1])
        h, w = int(batch.anchor.shape[-2]), int(batch.anchor.shape[-1])
        ph, pw = self.policy.backbone.decoder.patch_size
        spans = (w + h * ROW_STRIDE, n_text, w + h * ROW_STRIDE, w * pw + h * ph * ROW_STRIDE)
        shifts = tuple(int(torch.randint(0, FRAME_STRIDE - span, (1,), generator=generator)) for span in spans)
        action_u = torch.rand(kk, generator=generator)
        action_noise = torch.randn(tuple(batch.actions.shape), generator=generator)
        proprio_u = torch.rand(1, generator=generator)
        return Draws(
            text_u=text_u,
            video_u0=video_u0,
            video_noise=video_noise.to(torch.bfloat16),
            video_u=video_u,
            rope_shifts=(shifts[0], shifts[1], shifts[2], shifts[3]),
            action_u=action_u,
            action_noise=action_noise,
            proprio_u=proprio_u,
        )

    def video_inputs(
        self, batch: EncodedWindow, draws: Draws, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        """Noise the window.

        Returns:
            ``(noised [1, C, K*F, h, w], clean [1, C, (K-1)*F, h, w] or None, timesteps [1, K], target [1, C, K*F, h, w])``
            in ``dtype`` on ``device``.
        """
        s = self.settings
        kk = len(batch.chunks)
        lat = int(batch.chunks[0].shape[2])
        x0 = torch.cat([c.to(device, dtype) for c in batch.chunks], dim=2)
        eps0 = draws.video_noise.to(device, dtype)
        t0 = shifted_timesteps(draws.video_u0.to(device), s.video_timestep_shift)
        t0v = t0[0].view(1, 1, 1, 1, 1)
        xt = (x0 * (1 - t0v) + eps0 * t0v).to(dtype)
        target = (eps0 - x0).detach()
        z32, v32 = xt.float(), target.float()
        x0r = z32 - t0.to(dtype)[0].float() * v32
        epsr = x0r + v32
        t_k = shifted_timesteps(draws.video_u.to(device), s.video_timestep_shift).to(dtype)
        noised = torch.empty_like(x0r)
        for k in range(kk):
            tk = t_k[k].float()
            sl = slice(k * lat, (k + 1) * lat)
            noised[:, :, sl] = x0r[:, :, sl] * (1 - tk) + epsr[:, :, sl] * tk
        clean = x0r.to(dtype).detach()[:, :, : (kk - 1) * lat] if kk > 1 else None
        return noised.to(dtype), clean, t_k.view(1, kk), target

    def action_inputs(self, batch: EncodedWindow, draws: Draws) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Noise the actions on the CPU in float32.

        Returns:
            ``(noised [1, K*F, A], timesteps [1, K], target [1, K*F, A], proprio [1, K, P])``.
        """
        kk = len(batch.chunks)
        x0 = batch.actions.detach().to("cpu", torch.float32)
        t = draws.action_u.to(torch.float32)
        eps = draws.action_noise.to(torch.float32)
        t_tok = t.repeat_interleave(x0.shape[1] // kk)[None, :, None]
        noised = (1 - t_tok) * x0 + t_tok * eps
        target = eps - x0
        proprio = batch.proprio.detach().to("cpu", torch.float32)
        if self.training and bool(draws.proprio_u < self.settings.proprio_dropout):
            proprio = torch.zeros_like(proprio)
        return noised, t.view(1, kk), target, proprio

    def forward(self, batch: EncodedWindow, generator: torch.Generator) -> LossOutput:
        """Draw the random values and return the loss of ``batch``."""
        return self.loss(batch, self.draw(batch, generator))

    def loss(self, batch: EncodedWindow, draws: Draws) -> LossOutput:
        """Loss of one window for the given random values (no dropout, no rotary offset in eval mode)."""
        s = self.settings
        device, dtype = self.policy.device, self.policy.dtype_
        noised_v, clean, t_k, target_v = self.video_inputs(batch, draws, device, dtype)
        noised_a, t_a, target_a, proprio = self.action_inputs(batch, draws)
        text = batch.text
        if self.training and bool(draws.text_u < s.text_dropout):
            text = self.text_uncond
        shifts = draws.rope_shifts if self.training else (0, 0, 0, 0)
        out = self.policy(
            text.to(device, dtype),
            batch.anchor.to(device, dtype),
            clean,
            noised_v,
            t_k,
            noised_a.to(device),
            t_a.to(device),
            proprio.to(device),
            rope_shifts=shifts,
            decode=s.train_backbone,
            backbone_grad=s.train_backbone,
        )
        if out.velocity is not None:
            err_v = (out.velocity - target_v).pow(2)
            loss_v = err_v.sum() / err_v.numel()
        else:
            loss_v = torch.full((), float("nan"), device=device)
        pred_a = out.action_velocity[0].float()
        tgt_a = target_a[0].to(device, torch.float32)
        loss_a = (pred_a - tgt_a).square().mean(-1).mean()
        with torch.no_grad():
            keep = bool(torch.isfinite(loss_a)) and float(loss_a) <= s.action_guard
        loss_action = loss_a.detach().float()
        if not keep:
            loss_a = (torch.nan_to_num(pred_a, 0.0, 0.0, 0.0) * 0.0).sum()
        total = (s.action_weight * loss_a).float()
        if out.velocity is not None:
            total = total + (s.video_weight * loss_v).float()
        return LossOutput(
            loss=total,
            loss_video=loss_v.detach().float(),
            loss_action=loss_action,
            action_dropped=torch.tensor(0.0 if keep else 1.0, device=device),
        )
