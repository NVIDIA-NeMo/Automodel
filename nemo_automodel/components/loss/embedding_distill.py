# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_distance(z_s_proj: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """Per-sample ``1 - cosine_similarity``."""
    z_s = F.normalize(z_s_proj, dim=-1)
    z_t_n = F.normalize(z_t, dim=-1)
    return 1.0 - (z_s * z_t_n).sum(dim=-1)


def distill_loss_pair(
    s_q_proj: torch.Tensor,
    t_q: torch.Tensor,
    s_d_proj: torch.Tensor,
    t_d: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Embedding-alignment cosine loss over query and doc sides."""
    per_sample = cosine_distance(s_q_proj, t_q) + cosine_distance(s_d_proj, t_d)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"Unknown reduction: {reduction!r}")


def mse_loss_pair(
    s_q_proj: torch.Tensor,
    t_q: torch.Tensor,
    s_d_proj: torch.Tensor,
    t_d: torch.Tensor,
    normalize: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """Per-element MSE alignment between projected student and teacher."""
    if normalize:
        s_q_proj = F.normalize(s_q_proj, dim=-1)
        t_q = F.normalize(t_q, dim=-1)
        s_d_proj = F.normalize(s_d_proj, dim=-1)
        t_d = F.normalize(t_d, dim=-1)

    if reduction in ("mean", "sum"):
        return F.mse_loss(s_q_proj, t_q, reduction=reduction) + F.mse_loss(s_d_proj, t_d, reduction=reduction)
    if reduction == "batchmean":
        per_sample_q = (s_q_proj - t_q).pow(2).sum(dim=-1).mean()
        per_sample_d = (s_d_proj - t_d).pow(2).sum(dim=-1).mean()
        return per_sample_q + per_sample_d
    raise ValueError(f"Unknown reduction: {reduction!r}")


def _pairwise_cosine_softmax(z: torch.Tensor, temperature: float) -> torch.Tensor:
    z_n = F.normalize(z, dim=-1)
    sim = z_n @ z_n.T
    return F.softmax(sim / temperature, dim=-1)


def _score_loss_side(s: torch.Tensor, t: torch.Tensor, temperature: float) -> torch.Tensor:
    p_s = _pairwise_cosine_softmax(s, temperature)
    p_t = _pairwise_cosine_softmax(t, temperature).detach()
    batch = s.shape[0]
    return F.mse_loss(p_s, p_t, reduction="sum") / batch


def score_distill_loss(
    s_q: torch.Tensor,
    t_q: torch.Tensor,
    s_d: torch.Tensor,
    t_d: torch.Tensor,
    temperature: float = 0.02,
) -> torch.Tensor:
    """Score-matching distillation (row-softmax pairwise cosine matrix MSE)."""
    return _score_loss_side(s_q, t_q, temperature) + _score_loss_side(s_d, t_d, temperature)


class EmbeddingDistillLoss(nn.Module):
    """Cosine embedding-distillation loss module."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        s_q_proj: torch.Tensor,
        t_q: torch.Tensor,
        s_d_proj: torch.Tensor,
        t_d: torch.Tensor,
    ) -> torch.Tensor:
        return distill_loss_pair(s_q_proj, t_q, s_d_proj, t_d, reduction=self.reduction)


class EmbeddingMSELoss(nn.Module):
    """MSE embedding-distillation loss module."""

    def __init__(self, normalize: bool = False, reduction: str = "mean") -> None:
        super().__init__()
        self.normalize = normalize
        self.reduction = reduction

    def forward(
        self,
        s_q_proj: torch.Tensor,
        t_q: torch.Tensor,
        s_d_proj: torch.Tensor,
        t_d: torch.Tensor,
    ) -> torch.Tensor:
        return mse_loss_pair(
            s_q_proj,
            t_q,
            s_d_proj,
            t_d,
            normalize=self.normalize,
            reduction=self.reduction,
        )


class ScoreDistillLoss(nn.Module):
    """Score-matching distillation loss module."""

    def __init__(self, temperature: float = 0.02) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        s_q: torch.Tensor,
        t_q: torch.Tensor,
        s_d: torch.Tensor,
        t_d: torch.Tensor,
    ) -> torch.Tensor:
        return score_distill_loss(s_q, t_q, s_d, t_d, temperature=self.temperature)
