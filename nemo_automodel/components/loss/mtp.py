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

from typing import Optional

import torch
import torch.nn as nn

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.utils import _get_final_hidden_states, _get_lm_head_module, calculate_loss
from nemo_automodel.components.models.common.mtp import get_mtp_loss_scaling_factor, roll_tensor


def calculate_mtp_loss(
    loss_fn,
    *,
    mtp_per_depth_h: list[torch.Tensor],
    labels: torch.Tensor,
    model: nn.Module,
    scaling_factor: float = 0.1,
    num_label_tokens: Optional[int] = None,
    ignore_index: int = -100,
    cu_seqlens: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the DeepSeek-V3 Multi-Token Prediction auxiliary loss.

    Each depth's CE is dispatched through :func:`calculate_loss` with the
    same loss class as the main path, so MTP inherits FusedLinearCrossEntropy
    / MaskedCrossEntropy memory and numerical characteristics.

    Args:
        loss_fn: Configured per-token loss class (same instance the main
            path uses).
        mtp_per_depth_h: Per-depth hidden states from the model's MTP head,
            one ``[B, S, H]`` tensor per depth.
        labels: Original (unshifted) labels.
        model: The wrapped model; used to fetch the shared LM head when the
            loss class needs materialized logits (non-FusedLinearCE path).
        scaling_factor: Coefficient applied to the summed per-depth CE.
        num_label_tokens: Total non-ignore label tokens (forwarded to the
            base loss for sum-reduction normalization).
        ignore_index: Label value masked out of the CE loss for the trailing
            ``k+1`` rolled positions at depth ``k``.
        cu_seqlens: Optional cumulative sequence lengths ``[num_seqs+1]``
            (THD-pack layout). When supplied and ``seq_idx`` is not, builds
            a per-token sub-sequence index via searchsorted. Without packing
            this can be omitted.
        seq_idx: Optional per-token sub-sequence index ``[B, S]`` (or ``[S]``).
            Equality classes are what matter; absolute values can be any
            ints. Takes precedence over ``cu_seqlens``. Used to mask label
            rolls whose source position lies in a different sub-sequence.

    Returns:
        Scalar MTP loss with autograd graph.
    """
    D = len(mtp_per_depth_h)

    # Reconcile per-depth hidden-state and label dims for the THD-packed
    # non-PP path: the model unsqueezes mtp_per_depth_h from ``[T, H]`` back
    # to ``[1, T, H]`` (model.py post-MTP-forward), while labels arrive as
    # 1D ``[T]`` from ``process_input_for_thd``. ``FusedLinearCrossEntropy``
    # / ``cut_cross_entropy`` asserts ``hidden_states.shape[:-1] == labels.shape``
    # so squeeze the synthetic batch axis when labels are flat.
    if labels.dim() == 1:
        mtp_per_depth_h = [h.squeeze(0) if (h.dim() == 3 and h.shape[0] == 1) else h for h in mtp_per_depth_h]

    cur_labels = labels
    total = mtp_per_depth_h[0].new_zeros(())

    if seq_idx is None and cu_seqlens is not None:
        cs = cu_seqlens
        if cs.dim() == 2 and cs.shape[0] == 1:
            cs = cs.squeeze(0)
        if cs.dim() == 1:
            # Span the full (padded) token axis; cu_seqlens[-1] excludes tail pad.
            # Matches the model's mamba seq_idx build (nemotron_v3/layers.py).
            total_len = labels.shape[-1]
            positions = torch.arange(total_len, device=labels.device)
            # ``right=True`` so a position equal to a boundary (the first token
            # of sub-seq k, position == cu_seqlens[k]) maps to k, not k-1.
            seq_idx = torch.searchsorted(cs[1:].contiguous(), positions, right=True)
            if labels.dim() == 2:
                seq_idx = seq_idx.unsqueeze(0).expand(labels.shape[0], -1)
    elif seq_idx is not None:
        if seq_idx.dim() == 1 and labels.dim() == 2:
            seq_idx = seq_idx.unsqueeze(0).expand(labels.shape[0], -1)
        elif seq_idx.dim() == 2 and labels.dim() == 1 and seq_idx.shape[0] == 1:
            seq_idx = seq_idx.squeeze(0)
        # Under PP the caller must chunk seq_idx to per-microbatch shape; a
        # mismatch is a wiring bug, not a runtime condition to swallow.
        if seq_idx.shape != labels.shape:
            raise ValueError(
                f"calculate_mtp_loss: seq_idx.shape={tuple(seq_idx.shape)} does not "
                f"match labels.shape={tuple(labels.shape)}; under PP, chunk seq_idx "
                f"into per-microbatch pieces before passing it in."
            )

    for k, h_k in enumerate(mtp_per_depth_h):
        cur_labels = roll_tensor(cur_labels, shifts=-1, dim=-1)
        masked = cur_labels.clone()
        n_invalid = min(k + 1, masked.shape[-1])
        masked[..., -n_invalid:] = ignore_index

        # Mask labels whose rolled source (position t+k+1) lives in a
        # different sub-seq than position t — predictions across sub-seq
        # boundaries are nonsensical.
        if seq_idx is not None:
            rolled_seq_idx = roll_tensor(seq_idx, shifts=-(k + 1), dim=-1)
            cross_seq = rolled_seq_idx != seq_idx
            masked = torch.where(cross_seq, torch.full_like(masked, ignore_index), masked)

        if isinstance(loss_fn, FusedLinearCrossEntropy):
            depth_loss = calculate_loss(
                loss_fn,
                hidden_states=h_k,
                labels=masked,
                model=model,
                num_label_tokens=num_label_tokens,
            )
        else:
            lm_head = _get_lm_head_module(model)
            if lm_head is None:
                raise ValueError("lm_head module not found in model")
            depth_loss = calculate_loss(
                loss_fn,
                logits=lm_head(h_k),
                labels=masked,
                model=model,
                num_label_tokens=num_label_tokens,
            )
        total = total + depth_loss

    return total * (scaling_factor / D)


class PipelineCausalLMLoss(nn.Module):
    """Pipeline schedule loss that can add MTP auxiliary CE on the last stage.

    Per-microbatch ``seq_idx`` is read from a trailing element of the
    last-stage output tuple — the model appends an ``[B, S] int32`` tail
    when MTP is enabled. This binds each microbatch's seq_idx to its loss
    call via the PP runtime's output→loss contract, so the wiring is
    schedule-agnostic. Legacy ``cu_seqlens`` (THD path) is a fallback for
    models that don't emit a seq_idx tail.
    """

    def __init__(self, loss_fn: nn.Module, model: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = model
        # Legacy THD-pack fallback used when the model has no seq_idx tail.
        self.cu_seqlens: Optional[torch.Tensor] = None

    @staticmethod
    def _extract_seq_idx_tail(output) -> tuple[Optional[torch.Tensor], object]:
        """Detect and strip a trailing per-microbatch seq_idx from output.

        Convention: with MTP enabled the last-stage output is
        ``(logits, *mtp_per_depth_h, seq_idx)`` with an ``[B, S] int32``
        tail — dtype alone discriminates.
        """
        if isinstance(output, tuple) and len(output) > 0:
            last = output[-1]
            if isinstance(last, torch.Tensor) and last.dtype == torch.int32 and last.dim() == 2:
                return last, output[:-1]
        return None, output

    def forward(self, output, labels: torch.Tensor) -> torch.Tensor:
        seq_idx_mb, output = self._extract_seq_idx_tail(output)

        if isinstance(output, tuple):
            logits = output[0]
            hidden_states = None
            mtp_per_depth_h = list(output[1:]) if len(output) > 1 else None
            scaling_factor = get_mtp_loss_scaling_factor(self.model)
        else:
            logits = getattr(output, "logits", output)
            hidden_states = _get_final_hidden_states(output)
            mtp_per_depth_h = getattr(output, "mtp_per_depth_h", None)
            scaling_factor = getattr(output, "mtp_loss_scaling_factor", get_mtp_loss_scaling_factor(self.model))

        loss = calculate_loss(
            self.loss_fn,
            logits=logits,
            labels=labels,
            model=self.model,
            hidden_states=hidden_states,
        )
        if mtp_per_depth_h is not None and self.model.training:
            loss = loss + calculate_mtp_loss(
                self.loss_fn,
                mtp_per_depth_h=mtp_per_depth_h,
                labels=labels,
                model=self.model,
                scaling_factor=scaling_factor,
                cu_seqlens=self.cu_seqlens,
                seq_idx=seq_idx_mb,
            )
        return loss
