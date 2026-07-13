# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ProjectorLike = Callable[[torch.Tensor], torch.Tensor] | Mapping[int, Callable[[torch.Tensor], torch.Tensor]]
HiddenStatesLike = Sequence[torch.Tensor] | Mapping[int, torch.Tensor]


class LayerCapture:
    """Forward-hook helper for capturing selected intermediate hidden states."""

    def __init__(self, detach: bool = False) -> None:
        self.detach = detach
        self._outputs: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, idx: int):
        def hook(module, args, output):  # noqa: ARG001
            hidden = output[0] if isinstance(output, tuple) else output
            if self.detach:
                hidden = hidden.detach()
            self._outputs[idx] = hidden

        return hook

    def attach(self, layers: nn.ModuleList, indices: Iterable[int]) -> None:
        self.detach_hooks()
        num_layers = len(layers)
        for idx in indices:
            if idx < 0 or idx >= num_layers:
                self.detach_hooks()
                raise IndexError(f"LayerCapture: index {idx} out of range for {num_layers} layers")
            self._handles.append(layers[idx].register_forward_hook(self._make_hook(idx)))

    def detach_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._outputs.clear()

    def reset(self) -> None:
        self._outputs.clear()

    @property
    def outputs(self) -> dict[int, torch.Tensor]:
        return self._outputs


def _pick_projector(projector: ProjectorLike | None, s_idx: int) -> Callable[[torch.Tensor], torch.Tensor] | None:
    if projector is None:
        return None
    if isinstance(projector, Mapping):
        if s_idx not in projector:
            raise KeyError(
                f"intermediate_loss_function: layer_pairs references student layer {s_idx} "
                f"but projector dict has keys {sorted(projector)}"
            )
        return projector[s_idx]
    return projector


def _masked_per_token_loss(
    s: torch.Tensor,
    t: torch.Tensor,
    attention_mask: torch.Tensor | None,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "cosine":
        s_n = F.normalize(s, dim=-1)
        t_n = F.normalize(t, dim=-1)
        per_token = 1.0 - (s_n * t_n).sum(dim=-1)
        if attention_mask is None:
            return per_token.mean()
        mask = attention_mask.to(per_token.dtype)
        denom = mask.sum().clamp(min=1.0)
        return (per_token * mask).sum() / denom

    if loss_type == "mse":
        diff = (s - t).pow(2)
    elif loss_type == "smooth_l1":
        diff = F.smooth_l1_loss(s, t, reduction="none")
    else:
        raise ValueError(
            f"intermediate_loss_function: unknown loss_type {loss_type!r}; expected one of 'mse', 'smooth_l1', 'cosine'."
        )

    if attention_mask is None:
        return diff.mean()

    mask = attention_mask.to(diff.dtype).unsqueeze(-1)
    denom = (mask.sum() * diff.shape[-1]).clamp(min=1.0)
    return (diff * mask).sum() / denom


def _lookup_layer(hidden_states: HiddenStatesLike, idx: int, side: str) -> torch.Tensor:
    try:
        return hidden_states[idx]
    except (IndexError, KeyError):
        available = sorted(hidden_states.keys()) if isinstance(hidden_states, Mapping) else list(range(len(hidden_states)))
        raise IndexError(
            f"intermediate_loss_function: {side} layer index {idx} not available (have layers {available})"
        ) from None


def intermediate_loss_function(
    student_hidden_states: HiddenStatesLike,
    teacher_hidden_states: HiddenStatesLike,
    layer_pairs: Sequence[tuple[int, int]],
    attention_mask: torch.Tensor | None = None,
    projector: ProjectorLike | None = None,
    loss_type: str = "mse",
    normalize: bool = False,
    layer_weights: Sequence[float] | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Per-token feature distillation between matched student/teacher layers."""
    if len(layer_pairs) == 0:
        ref_iter = iter(student_hidden_states.values()) if isinstance(student_hidden_states, Mapping) else iter(student_hidden_states)
        try:
            ref = next(ref_iter)
        except StopIteration:
            return torch.zeros((), dtype=torch.float32)
        return ref.new_zeros((), dtype=torch.float32)

    if layer_weights is not None and len(layer_weights) != len(layer_pairs):
        raise ValueError(
            f"intermediate_loss_function: layer_weights has length {len(layer_weights)} but layer_pairs has length {len(layer_pairs)}"
        )

    per_pair_losses: list[torch.Tensor] = []
    for i, (s_idx, t_idx) in enumerate(layer_pairs):
        s_h = _lookup_layer(student_hidden_states, s_idx, "student").float()
        t_h = _lookup_layer(teacher_hidden_states, t_idx, "teacher").detach().to(dtype=s_h.dtype)

        proj = _pick_projector(projector, s_idx)
        if proj is not None:
            s_h = proj(s_h)

        if s_h.shape != t_h.shape:
            raise ValueError(
                f"intermediate_loss_function: shape mismatch at pair (s={s_idx}, t={t_idx}): "
                f"student={tuple(s_h.shape)} vs teacher={tuple(t_h.shape)}"
            )

        if normalize:
            s_h = F.normalize(s_h, dim=-1)
            t_h = F.normalize(t_h, dim=-1)

        loss_i = _masked_per_token_loss(s_h, t_h, attention_mask, loss_type)
        if layer_weights is not None:
            loss_i = loss_i * float(layer_weights[i])
        per_pair_losses.append(loss_i)

    stacked = torch.stack(per_pair_losses)
    if reduction == "mean":
        return stacked.mean()
    if reduction == "sum":
        return stacked.sum()
    raise ValueError(
        f"intermediate_loss_function: unknown reduction {reduction!r}; expected 'mean' or 'sum'."
    )


def intermediate_loss_pair(
    s_q_hidden_states: HiddenStatesLike,
    t_q_hidden_states: HiddenStatesLike,
    s_d_hidden_states: HiddenStatesLike,
    t_d_hidden_states: HiddenStatesLike,
    layer_pairs: Sequence[tuple[int, int]],
    attn_q: torch.Tensor | None = None,
    attn_d: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Sum of intermediate loss on query and doc sides."""
    loss_q = intermediate_loss_function(
        s_q_hidden_states,
        t_q_hidden_states,
        layer_pairs=layer_pairs,
        attention_mask=attn_q,
        **kwargs,
    )
    loss_d = intermediate_loss_function(
        s_d_hidden_states,
        t_d_hidden_states,
        layer_pairs=layer_pairs,
        attention_mask=attn_d,
        **kwargs,
    )
    return loss_q + loss_d


class IntermediateDistillLoss(nn.Module):
    """Intermediate-layer feature distillation module."""

    def __init__(
        self,
        layer_pairs: Sequence[tuple[int, int]] | Sequence[list[int]],
        loss_type: str = "mse",
        normalize: bool = False,
        layer_weights: Sequence[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.layer_pairs = [tuple(int(x) for x in pair) for pair in layer_pairs]
        self.loss_type = loss_type
        self.normalize = normalize
        self.layer_weights = layer_weights
        self.reduction = reduction

    def forward(
        self,
        s_q_hidden_states: HiddenStatesLike,
        t_q_hidden_states: HiddenStatesLike,
        s_d_hidden_states: HiddenStatesLike,
        t_d_hidden_states: HiddenStatesLike,
        attn_q: torch.Tensor | None = None,
        attn_d: torch.Tensor | None = None,
        projector: ProjectorLike | None = None,
    ) -> torch.Tensor:
        return intermediate_loss_pair(
            s_q_hidden_states,
            t_q_hidden_states,
            s_d_hidden_states,
            t_d_hidden_states,
            layer_pairs=self.layer_pairs,
            attn_q=attn_q,
            attn_d=attn_d,
            projector=projector,
            loss_type=self.loss_type,
            normalize=self.normalize,
            layer_weights=self.layer_weights,
            reduction=self.reduction,
        )
