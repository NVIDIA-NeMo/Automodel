# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.loss.dist_utils import all_gather_no_grad, all_gather_with_grad


def infonce_loss(
    queries: torch.Tensor,
    documents: torch.Tensor,
    hard_negatives: torch.Tensor | None = None,
    hard_negatives_mask: torch.Tensor | None = None,
    temperature: float | torch.Tensor = 0.05,
    direction: str = "q2d",
    use_in_batch_negatives: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """InfoNCE contrastive loss with optional hard negatives."""
    if queries.dim() != 2 or documents.dim() != 2:
        raise ValueError(
            f"infonce_loss: queries and documents must be 2-D [B, D]; got queries={tuple(queries.shape)}, "
            f"documents={tuple(documents.shape)}"
        )
    if queries.shape != documents.shape:
        raise ValueError(
            f"infonce_loss: queries.shape={tuple(queries.shape)} must equal documents.shape={tuple(documents.shape)}"
        )

    batch, hidden = queries.shape
    device = queries.device

    if direction not in ("q2d", "d2q", "symmetric"):
        raise ValueError(f"infonce_loss: unknown direction {direction!r}")

    has_hard_negs = hard_negatives is not None
    if has_hard_negs:
        if hard_negatives.dim() != 3 or hard_negatives.shape[0] != batch or hard_negatives.shape[-1] != hidden:
            raise ValueError(
                f"infonce_loss: hard_negatives must be [B, K, D] with B={batch}, D={hidden}; "
                f"got {tuple(hard_negatives.shape)}"
            )
        if hard_negatives_mask is not None and hard_negatives_mask.shape != (batch, hard_negatives.shape[1]):
            raise ValueError(
                f"infonce_loss: hard_negatives_mask must be [B, K]; got {tuple(hard_negatives_mask.shape)}"
            )

    if not use_in_batch_negatives and not has_hard_negs:
        raise ValueError("infonce_loss: no negatives provided")

    if normalize:
        q = F.normalize(queries, dim=-1)
        d = F.normalize(documents, dim=-1)
        n = F.normalize(hard_negatives, dim=-1) if has_hard_negs else None
    else:
        q, d, n = queries, documents, hard_negatives

    def _one_direction(query: torch.Tensor, doc: torch.Tensor, neg: torch.Tensor | None) -> torch.Tensor:
        if use_in_batch_negatives:
            logits = query @ doc.T
            target = torch.arange(batch, device=device)
        else:
            pos = (query * doc).sum(dim=-1, keepdim=True)
            logits = pos
            target = torch.zeros(batch, device=device, dtype=torch.long)

        if neg is not None:
            sim_qn = torch.einsum("bd,bkd->bk", query, neg)
            if hard_negatives_mask is not None:
                pad = hard_negatives_mask.to(sim_qn.dtype) == 0
                sim_qn = sim_qn.masked_fill(pad, float("-inf"))
            logits = torch.cat([logits, sim_qn], dim=-1)

        logits = logits / temperature
        return F.cross_entropy(logits, target)

    if direction == "q2d":
        return _one_direction(q, d, n)
    if direction == "d2q":
        return _one_direction(d, q, None)
    return 0.5 * (_one_direction(q, d, n) + _one_direction(d, q, None))


def infonce_distill_loss(
    student_queries: torch.Tensor,
    student_documents: torch.Tensor,
    teacher_queries: torch.Tensor,
    teacher_documents: torch.Tensor,
    student_hard_negatives: torch.Tensor | None = None,
    teacher_hard_negatives: torch.Tensor | None = None,
    hard_negatives_mask: torch.Tensor | None = None,
    temperature: float | torch.Tensor = 0.05,
    direction: str = "q2d",
    use_in_batch_negatives: bool = True,
    normalize: bool = True,
    divergence: str = "kl",
) -> torch.Tensor:
    """Soft listwise distillation on InfoNCE candidate sets."""
    if student_queries.dim() != 2 or student_documents.dim() != 2:
        raise ValueError(
            f"infonce_distill_loss: student embeddings must be 2-D [B, D_s]; got queries={tuple(student_queries.shape)}, "
            f"documents={tuple(student_documents.shape)}"
        )
    if teacher_queries.dim() != 2 or teacher_documents.dim() != 2:
        raise ValueError(
            f"infonce_distill_loss: teacher embeddings must be 2-D [B, D_t]; got queries={tuple(teacher_queries.shape)}, "
            f"documents={tuple(teacher_documents.shape)}"
        )
    if student_queries.shape != student_documents.shape:
        raise ValueError("infonce_distill_loss: student query/doc shapes must match")
    if teacher_queries.shape != teacher_documents.shape:
        raise ValueError("infonce_distill_loss: teacher query/doc shapes must match")
    if student_queries.shape[0] != teacher_queries.shape[0]:
        raise ValueError("infonce_distill_loss: student/teacher batch sizes must match")

    if direction not in ("q2d", "d2q", "symmetric"):
        raise ValueError(f"infonce_distill_loss: unknown direction {direction!r}")
    if divergence not in ("kl", "ce", "mse"):
        raise ValueError(f"infonce_distill_loss: unknown divergence {divergence!r}")

    batch = student_queries.shape[0]

    has_hard_negs = student_hard_negatives is not None
    if has_hard_negs:
        if teacher_hard_negatives is None:
            raise ValueError("infonce_distill_loss: teacher_hard_negatives required when student_hard_negatives is set")
        if student_hard_negatives.dim() != 3 or student_hard_negatives.shape[0] != batch:
            raise ValueError("infonce_distill_loss: student_hard_negatives must be [B, K, D_s]")
        if (
            teacher_hard_negatives.dim() != 3
            or teacher_hard_negatives.shape[0] != batch
            or teacher_hard_negatives.shape[1] != student_hard_negatives.shape[1]
        ):
            raise ValueError("infonce_distill_loss: teacher_hard_negatives must be [B, K, D_t] with matching B,K")
        if hard_negatives_mask is not None and hard_negatives_mask.shape != (batch, student_hard_negatives.shape[1]):
            raise ValueError("infonce_distill_loss: hard_negatives_mask must be [B, K]")

    if not use_in_batch_negatives and not has_hard_negs:
        raise ValueError("infonce_distill_loss: no negatives available")

    s_q = student_queries.float()
    s_d = student_documents.float()
    t_q = teacher_queries.float().detach()
    t_d = teacher_documents.float().detach()
    s_n = student_hard_negatives.float() if has_hard_negs else None
    t_n = teacher_hard_negatives.float().detach() if has_hard_negs else None

    if normalize:
        s_q = F.normalize(s_q, dim=-1)
        s_d = F.normalize(s_d, dim=-1)
        t_q = F.normalize(t_q, dim=-1)
        t_d = F.normalize(t_d, dim=-1)
        if has_hard_negs:
            s_n = F.normalize(s_n, dim=-1)
            t_n = F.normalize(t_n, dim=-1)

    def _build_logits(query: torch.Tensor, doc: torch.Tensor, neg: torch.Tensor | None) -> torch.Tensor:
        if use_in_batch_negatives:
            logits = query @ doc.T
        else:
            logits = (query * doc).sum(dim=-1, keepdim=True)
        if neg is not None:
            sim_qn = torch.einsum("bd,bkd->bk", query, neg)
            if hard_negatives_mask is not None:
                pad = hard_negatives_mask.to(sim_qn.dtype) == 0
                sim_qn = sim_qn.masked_fill(pad, float("-inf"))
            logits = torch.cat([logits, sim_qn], dim=-1)
        return logits

    def _one_direction(
        s_query: torch.Tensor,
        s_doc: torch.Tensor,
        s_neg: torch.Tensor | None,
        t_query: torch.Tensor,
        t_doc: torch.Tensor,
        t_neg: torch.Tensor | None,
    ) -> torch.Tensor:
        s_logits = _build_logits(s_query, s_doc, s_neg) / temperature
        with torch.no_grad():
            t_logits = _build_logits(t_query, t_doc, t_neg) / temperature

        if divergence == "kl":
            log_p_s = F.log_softmax(s_logits, dim=-1)
            p_t = F.softmax(t_logits, dim=-1)
            return F.kl_div(log_p_s, p_t, reduction="batchmean")
        if divergence == "ce":
            log_p_s = F.log_softmax(s_logits, dim=-1)
            p_t = F.softmax(t_logits, dim=-1)
            return -(p_t * log_p_s).sum(dim=-1).mean()
        p_s = F.softmax(s_logits, dim=-1)
        p_t = F.softmax(t_logits, dim=-1)
        return F.mse_loss(p_s, p_t, reduction="sum") / s_logits.shape[0]

    if direction == "q2d":
        return _one_direction(s_q, s_d, s_n, t_q, t_d, t_n)
    if direction == "d2q":
        return _one_direction(s_d, s_q, None, t_d, t_q, None)
    return 0.5 * (_one_direction(s_q, s_d, s_n, t_q, t_d, t_n) + _one_direction(s_d, s_q, None, t_d, t_q, None))


class InfoNCELoss(nn.Module):
    """InfoNCE loss module with optional learnable temperature."""

    def __init__(
        self,
        temperature: float = 0.05,
        learnable_temperature: bool = False,
        direction: str = "q2d",
        use_in_batch_negatives: bool = True,
        normalize: bool = True,
        cross_device_negatives: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.direction = direction
        self.use_in_batch_negatives = use_in_batch_negatives
        self.normalize = normalize
        self.cross_device_negatives = cross_device_negatives
        self.learnable_temperature = learnable_temperature

        if learnable_temperature:
            self.log_inv_tau = nn.Parameter(torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))
            self.register_buffer("tau", torch.tensor(temperature, dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("tau", torch.tensor(temperature, dtype=torch.float32), persistent=True)
            self.log_inv_tau = None

    def current_temperature(self) -> torch.Tensor:
        if self.learnable_temperature and self.log_inv_tau is not None:
            return torch.exp(-self.log_inv_tau)
        return self.tau

    def forward(
        self,
        queries: torch.Tensor,
        documents: torch.Tensor,
        hard_negatives: torch.Tensor | None = None,
        hard_negatives_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self.cross_device_negatives
            and self.use_in_batch_negatives
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            queries = all_gather_with_grad(queries)
            documents = all_gather_with_grad(documents)
            if hard_negatives is not None:
                hard_negatives = all_gather_with_grad(hard_negatives)
            if hard_negatives_mask is not None:
                hard_negatives_mask = all_gather_no_grad(hard_negatives_mask)

        return infonce_loss(
            queries,
            documents,
            hard_negatives=hard_negatives,
            hard_negatives_mask=hard_negatives_mask,
            temperature=self.current_temperature(),
            direction=self.direction,
            use_in_batch_negatives=self.use_in_batch_negatives,
            normalize=self.normalize,
        )


class InfoNCEDistillLoss(nn.Module):
    """InfoNCE soft listwise distillation loss module."""

    def __init__(
        self,
        temperature: float = 0.05,
        direction: str = "q2d",
        use_in_batch_negatives: bool = True,
        normalize: bool = True,
        divergence: str = "kl",
        cross_device_negatives: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.direction = direction
        self.use_in_batch_negatives = use_in_batch_negatives
        self.normalize = normalize
        self.divergence = divergence
        self.cross_device_negatives = cross_device_negatives

    def forward(
        self,
        student_queries: torch.Tensor,
        student_documents: torch.Tensor,
        teacher_queries: torch.Tensor,
        teacher_documents: torch.Tensor,
        student_hard_negatives: torch.Tensor | None = None,
        teacher_hard_negatives: torch.Tensor | None = None,
        hard_negatives_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self.cross_device_negatives
            and self.use_in_batch_negatives
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            student_queries = all_gather_with_grad(student_queries)
            student_documents = all_gather_with_grad(student_documents)
            teacher_queries = all_gather_no_grad(teacher_queries)
            teacher_documents = all_gather_no_grad(teacher_documents)
            if student_hard_negatives is not None:
                student_hard_negatives = all_gather_with_grad(student_hard_negatives)
            if teacher_hard_negatives is not None:
                teacher_hard_negatives = all_gather_no_grad(teacher_hard_negatives)
            if hard_negatives_mask is not None:
                hard_negatives_mask = all_gather_no_grad(hard_negatives_mask)

        return infonce_distill_loss(
            student_queries,
            student_documents,
            teacher_queries,
            teacher_documents,
            student_hard_negatives=student_hard_negatives,
            teacher_hard_negatives=teacher_hard_negatives,
            hard_negatives_mask=hard_negatives_mask,
            temperature=self.temperature,
            direction=self.direction,
            use_in_batch_negatives=self.use_in_batch_negatives,
            normalize=self.normalize,
            divergence=self.divergence,
        )
