# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Gated-LoRA multi-token prediction (Samragh et al., 2025; arXiv:2507.11851).

A frozen base model learns to predict ``D`` future tokens at once: ``D`` mask tokens are appended after a
position, and a LoRA adapter that is gated on for the mask tokens only turns their hidden states into
predictions of the second through ``D + 1``-th future tokens. All other tokens use the base weights, so
drafts can be verified against the clean-token path in the same forward. Randomizing reserved embedding
rows can change their output logits on models with tied input/output embeddings.

Training uses the paper's dense layout: every position is an anchor. One forward runs over the depth-major
``[clean (S) | depth 1 (S) | ... | depth D (S)]`` sequence under the attention mask of
``nemo_automodel.components.attention.gated_mtp_mask``, and the mask hidden states are returned as
``mtp_per_depth_h``, aligned with the native MTP heads: depth ``d`` at position ``t`` predicts token
``t + d + 1``. Training can use either Section 2.5's objective or the existing hard/soft MTP baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.speculative.gated_mtp.sampler import GatedMTPSampler


@dataclass
class GatedMTPConfig:
    """Typed config for gated-LoRA multi-token prediction.

    Attributes:
        mask_token_ids: One token id per MTP depth, appended after every anchor position. The ids must exist in
            the model vocabulary; reserved or otherwise unused special tokens work well, since only the gated
            LoRA adapter learns to use them.
        scaling_factor: Weight of the auxiliary loss for ``mtp``, or of the complete
            CE-plus-LCM objective for ``paper``.
        objective: ``paper`` trains base CE, sampler CE, and latent consistency from
            Section 2.5. ``mtp`` preserves the existing hard/soft MTP objectives.
    """

    mask_token_ids: list[int] = field(default_factory=list)
    scaling_factor: float = 1.0
    objective: Literal["mtp", "paper"] = "mtp"

    def build(self) -> GatedMTPLayout:
        """Build the layout helper for this config.

        Returns:
            The :class:`GatedMTPLayout` for ``mask_token_ids`` and ``scaling_factor``.

        Raises:
            ValueError: If ``mask_token_ids`` is empty, ``scaling_factor`` is not positive,
                or ``objective`` is unknown.
        """
        if not self.mask_token_ids:
            raise ValueError("gated_mtp.mask_token_ids needs one token id per MTP depth.")
        if self.scaling_factor <= 0:
            raise ValueError(f"gated_mtp.scaling_factor must be positive, got {self.scaling_factor}.")
        if self.objective not in ("mtp", "paper"):
            raise ValueError(f"gated_mtp.objective must be 'mtp' or 'paper', got {self.objective!r}.")
        return GatedMTPLayout(
            mask_token_ids=tuple(self.mask_token_ids), scaling_factor=self.scaling_factor, objective=self.objective
        )

    def build_sampler(self, *, hidden_size: int, device: torch.device, dtype: torch.dtype) -> GatedMTPSampler:
        """Build the shared sampler before the optimizer captures its parameters.

        Args:
            hidden_size: Width of the base model's hidden representation.
            device: Runtime device for the sampler parameters.
            dtype: Runtime storage dtype for the sampler parameters.

        Returns:
            A randomly initialized Section 2.3 sampler on the requested device.
        """
        return GatedMTPSampler(hidden_size, len(self.mask_token_ids)).to(device=device, dtype=dtype)


@dataclass
class GatedMTPInputs:
    """Model inputs of the dense gated-MTP layout.

    Attributes:
        input_ids: Tensor of shape [batch, sequence * (1 + depths)] in the depth-major
            ``[clean | depth 1 | ... | depth D]`` order.
        position_ids: Tensor of shape [batch, sequence * (1 + depths)]; the mask token of anchor ``t`` and depth
            ``d`` sits at the clean position of ``t`` plus ``d``.
        lora_gate: Bool Tensor of shape [batch, sequence * (1 + depths)], True on the mask tokens, for
            ``lora_token_gate``.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    lora_gate: torch.Tensor


@dataclass
class GatedMTPOutput:
    """Model output of the dense gated-MTP layout, split into the clean copy and the MTP depths.

    Mirrors the fields the MTP training path reads from a native MTP model output.

    Attributes:
        logits: Tensor of shape [batch, sequence, vocab] for the clean copy when the model returned logits for
            every position, else the model's logits unchanged (e.g. the last position only).
        hidden_states: Final hidden states of the clean copy, Tensor of shape [batch, sequence, hidden].
        mtp_per_depth_h: One Tensor of shape [batch, sequence, hidden] per depth, whose position ``t`` holds the
            final hidden state of the mask token of anchor ``t``.
        mtp_per_depth_logits: Always None; the MTP depths are projected through the shared LM head by the loss.
        mtp_loss_scaling_factor: Weight of the MTP loss.
    """

    logits: torch.Tensor | None
    hidden_states: torch.Tensor
    mtp_per_depth_h: list[torch.Tensor]
    mtp_per_depth_logits: None
    mtp_loss_scaling_factor: float


class GatedMTPLayout:
    """Build the dense gated-MTP inputs and split the model output into the clean copy and the MTP depths."""

    def __init__(
        self, *, mask_token_ids: tuple[int, ...], scaling_factor: float, objective: Literal["mtp", "paper"] = "mtp"
    ) -> None:
        """Store the per-depth mask token ids and the MTP loss weight.

        Args:
            mask_token_ids: One token id per MTP depth.
            scaling_factor: Weight of the MTP loss.
            objective: Training objective; ``paper`` enables Section 2.5's losses.
        """
        self.mask_token_ids = mask_token_ids
        self.scaling_factor = scaling_factor
        self.objective = objective

    @property
    def num_depths(self) -> int:
        """Number of MTP depths ``D``."""
        return len(self.mask_token_ids)

    def paper_loss(
        self,
        output: GatedMTPOutput,
        *,
        sampler_logits: list[torch.Tensor],
        lm_head: nn.Module,
        labels: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        """Compute Section 2.5's per-sample base CE, sampler CE, and LCM sum.

        CE is averaged over supervised NTP and MTP positions together. LCM first
        averages the matching mask states for each clean target, then averages
        over supervised clean positions. Clean LCM targets are detached. Mask
        positions whose anchor is unsupervised are excluded, as in Section 2.1.
        The dense layout still computes those positions, but they contribute no loss.

        Args:
            output: Clean logits, Tensor of shape [batch, sequence, vocab]; clean hidden
                states, Tensor of shape [batch, sequence, hidden]; and one mask-state
                Tensor of shape [batch, sequence, hidden] per depth. All tensors are local,
                unsharded tensors; this objective currently supports one GPU.
            sampler_logits: Clean sampler logits followed by one Tensor of shape
                [batch, sequence, vocab] per depth.
            lm_head: Frozen projection from hidden features to vocabulary logits.
            labels: Next-token labels, Tensor of shape [batch, sequence], with
                ``ignore_index`` at unsupervised and padded positions.
            ignore_index: Label value excluded from both objectives.

        Returns:
            Tensor of shape [batch], containing the scaled objective per sample.
            Empty-supervision samples contribute zero while retaining a backward graph.
        """
        if len(sampler_logits) != self.num_depths + 1:
            raise ValueError("paper_loss requires clean sampler logits and one tensor per MTP depth")
        if output.logits is None or len(output.mtp_per_depth_h) != self.num_depths:
            raise ValueError("paper_loss requires full clean logits and one hidden-state tensor per MTP depth")
        batch, sequence = labels.shape
        supervised = (labels != ignore_index) & (labels != -100)
        labels = labels.masked_fill(~supervised, ignore_index)

        def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """Return per-position CE with ignored labels contributing zero.

            Args:
                logits: Tensor of shape [batch, positions, vocab].
                targets: Tensor of shape [batch, positions].

            Returns:
                Tensor of shape [batch, positions], computed in float32.
            """
            return F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=ignore_index,
                reduction="none",
            ).reshape(batch, -1)

        ce = cross_entropy(output.logits, labels).sum(-1) + cross_entropy(sampler_logits[0], labels).sum(-1)
        ce_count = supervised.sum(-1)
        lcm = torch.zeros(batch, sequence, device=labels.device, dtype=torch.float32)
        lcm_count = torch.zeros_like(lcm)
        for depth, hidden in enumerate(output.mtp_per_depth_h, start=1):
            if depth >= sequence:
                # Keep every sampler parameter connected even with no valid targets.
                ce = ce + sampler_logits[depth].float().sum((1, 2)) * 0.0
                continue
            valid = supervised[:, :-depth] & supervised[:, depth:]
            targets = labels[:, depth:].masked_fill(~valid, ignore_index)
            ce = ce + cross_entropy(lm_head(hidden[:, :-depth]), targets).sum(-1)
            ce = ce + cross_entropy(sampler_logits[depth][:, :-depth], targets).sum(-1)
            ce_count = ce_count + valid.sum(-1)
            error = (hidden[:, :-depth].float() - output.hidden_states[:, depth:].detach().float()).square().mean(-1)
            lcm = lcm + F.pad(error * valid, (depth, 0))
            lcm_count = lcm_count + F.pad(valid.float(), (depth, 0))
        ce = ce / ce_count.clamp_min(1)
        lcm = ((lcm / lcm_count.clamp_min(1)) * supervised).sum(-1) / supervised.sum(-1).clamp_min(1)
        return self.scaling_factor * (ce + lcm)

    def prepare_inputs(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> GatedMTPInputs:
        """Append one mask token per depth after every position.

        Args:
            input_ids: Clean token ids, Tensor of shape [batch, sequence]; not packed.
            position_ids: Optional clean position ids, Tensor of shape [batch, sequence]; defaults to
                ``0..sequence - 1`` for every row.

        Returns:
            The :class:`GatedMTPInputs` of the depth-major layout.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).expand(batch, seq_len)
        depths = torch.arange(1, self.num_depths + 1, device=device)
        mask_ids = torch.tensor(self.mask_token_ids, dtype=input_ids.dtype, device=device)
        # [batch, depths, sequence] -> depth-major [batch, depths * sequence]
        mask_input_ids = mask_ids.view(1, -1, 1).expand(batch, -1, seq_len).reshape(batch, -1)
        mask_position_ids = (position_ids.unsqueeze(1) + depths.view(1, -1, 1)).reshape(batch, -1)
        lora_gate = torch.ones(batch, seq_len * (1 + self.num_depths), dtype=torch.bool, device=device)
        lora_gate[:, :seq_len] = False
        return GatedMTPInputs(
            input_ids=torch.cat([input_ids, mask_input_ids], dim=1),
            position_ids=torch.cat([position_ids.to(mask_position_ids.dtype), mask_position_ids], dim=1),
            lora_gate=lora_gate,
        )

    def split_output(self, logits: torch.Tensor | None, hidden_states: torch.Tensor) -> GatedMTPOutput:
        """Split the output of a forward over :meth:`prepare_inputs` into the clean copy and the MTP depths.

        Args:
            logits: The model's logits, Tensor of shape [batch, positions, vocab] where ``positions`` is either
                the full ``sequence * (1 + depths)`` or a trailing subset (``logits_to_keep``), or None.
            hidden_states: Final hidden states, Tensor of shape [batch, sequence * (1 + depths), hidden].

        Returns:
            The :class:`GatedMTPOutput`; its tensors are contiguous, since the losses flatten them with ``view``.
        """
        total = hidden_states.shape[1]
        if total % (1 + self.num_depths) != 0:
            raise ValueError(
                f"Hidden states of length {total} are not a gated-MTP layout with {self.num_depths} depths."
            )
        seq_len = total // (1 + self.num_depths)
        if logits is not None and logits.shape[1] == total:
            logits = logits[:, :seq_len].contiguous()
        return GatedMTPOutput(
            logits=logits,
            hidden_states=hidden_states[:, :seq_len].contiguous(),
            mtp_per_depth_h=[h.contiguous() for h in hidden_states[:, seq_len:].split(seq_len, dim=1)],
            mtp_per_depth_logits=None,
            mtp_loss_scaling_factor=self.scaling_factor,
        )
