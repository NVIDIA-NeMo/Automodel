# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import torch.distributed.nn.functional as dist_nn_func
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.distributed.tensor import DTensor, Shard

    _HAVE_DTENSOR = True
except Exception:
    DTensor = None  # type: ignore[assignment, misc]
    Shard = None  # type: ignore[assignment, misc]
    _HAVE_DTENSOR = False


def _infer_tp_group_from_dtensor(logits: torch.Tensor) -> Optional[torch.distributed.ProcessGroup]:
    """If *logits* is a DTensor sharded on the vocab (last) dimension, return its TP process group.

    Iterates over the DTensor placements to find the mesh dimension that holds a vocab-dim
    ``Shard`` and returns the corresponding process group.  Returns ``None`` for plain tensors
    or DTensors that are not vocab-sharded.
    """
    if not _HAVE_DTENSOR or not isinstance(logits, DTensor):
        return None
    vocab_dim = logits.ndim - 1
    for mesh_dim, placement in enumerate(logits.placements):
        if isinstance(placement, Shard) and (placement.dim == -1 or placement.dim == vocab_dim):
            return logits.device_mesh.get_group(mesh_dim)
    return None


def _kl_forward_tp(
    t_logits: torch.Tensor,
    s_logits: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Compute per-token negative cross-entropy ``sum(P * log Q)`` with tensor parallelism.

    Both ``t_logits`` and ``s_logits`` are **local** vocab-sharded tensors of shape
    ``[valid_tokens, local_vocab_size]``.  A numerically stable global softmax / log-softmax is
    computed via ``all_reduce`` over ``tp_group``, avoiding the need to gather the full vocab.

    Args:
        t_logits: Local teacher logit shard, shape ``[valid_tokens, local_vocab_size]``.
        s_logits: Local student logit shard, shape ``[valid_tokens, local_vocab_size]``.
        tp_group: Process group spanning the tensor-parallel ranks.

    Returns:
        Per-token sum(P * log Q), shape ``[valid_tokens]``.  This is the *negative* KL term;
        negate and average in the caller to obtain the final loss.
    """
    # --- Stable global softmax for teacher: P ---
    teacher_max, _ = torch.max(t_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(teacher_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    output_teacher = t_logits - teacher_max
    denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1, keepdim=True)
    torch.distributed.all_reduce(denom_teacher, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    teacher_prob = torch.exp(output_teacher) / denom_teacher.clamp(min=1e-12)

    # --- Stable global log-softmax for student: log Q ---
    student_max, _ = torch.max(s_logits, dim=-1, keepdim=True)
    student_max = student_max.detach()
    torch.distributed.all_reduce(student_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    output_student = s_logits - student_max
    denom_student = torch.sum(torch.exp(output_student), dim=-1, keepdim=True)
    denom_student = dist_nn_func.all_reduce(denom_student, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    student_log_prob = output_student - torch.log(denom_student.clamp(min=1e-12))

    # --- Per-token sum(P * log Q): local accumulate then global reduce ---
    # Mask -inf student logits so that 0 * -inf does not produce NaN.
    inf_mask = torch.isinf(s_logits)
    ce_local = torch.masked_fill(teacher_prob * student_log_prob, inf_mask, 0.0).sum(dim=-1)
    ce_local = dist_nn_func.all_reduce(ce_local, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    return ce_local  # shape: [valid_tokens]


def _kl_forward_chunked(
    t_logits: torch.Tensor,
    s_logits: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute per-token sum(P * log Q) in chunks to reduce peak memory.

    Processes ``chunk_size`` tokens at a time so that only one chunk's worth of the
    ``[chunk_size, vocab_size]`` fp32 probability matrix is live at any moment.

    Args:
        t_logits: Teacher logits, shape ``[num_valid_tokens, vocab_size]``.
        s_logits: Student logits, shape ``[num_valid_tokens, vocab_size]``.
        chunk_size: Number of tokens per chunk.

    Returns:
        Per-token sum(P * log Q), shape ``[num_valid_tokens]``.
    """
    num_tokens = t_logits.shape[0]
    kl_parts: list[torch.Tensor] = []
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        t_chunk = t_logits[start:end]
        s_chunk = s_logits[start:end]
        teacher_prob = F.softmax(t_chunk, dim=-1, dtype=torch.float32)
        student_logprob = F.log_softmax(s_chunk, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(s_chunk)
        kl_parts.append(torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0).sum(-1))
    return torch.cat(kl_parts, dim=0)


class KDLoss(nn.Module):
    """Forward KL divergence loss for knowledge distillation.

    Computes ``KL(P_teacher ‖ P_student)`` averaged over valid (non-padding) tokens.

    Supports tensor-parallel (TP) training: when logits are vocab-sharded ``DTensor``s, the TP
    group is inferred automatically and a distributed softmax is used to avoid gathering the full
    vocabulary on each rank.  A ``tp_group`` can also be supplied explicitly.

    Args:
        ignore_index: Label value marking padding tokens (default ``-100``).
        temperature: Softmax temperature *T*.  Both teacher and student logits are divided by *T*
            before computing probabilities.  The loss is then multiplied by *T²* so that gradient
            magnitudes remain independent of the chosen temperature (Hinton et al., 2015).
        fp32_upcast: Cast logits to float32 before computing softmax / log-softmax for numerical
            stability (default ``True``).
        tp_group: Explicit TP process group.  When ``None`` (default) the group is inferred from
            the DTensor placement of ``student_logits``, or the non-TP path is used for plain
            tensors.
        chunk_size: When positive, valid tokens are processed in chunks of this size to avoid
            materializing the full ``[num_valid_tokens, vocab_size]`` probability matrix in fp32.
            Reduces peak memory at the cost of slightly more kernel launches.  ``0`` (default)
            disables chunking.  Ignored when using the TP path.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        temperature: float = 1.0,
        fp32_upcast: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.fp32_upcast = fp32_upcast
        self.tp_group = tp_group
        self.chunk_size = chunk_size

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        num_batch_labels: int | None = None,
    ) -> torch.Tensor:
        """Compute the KD loss.

        Args:
            student_logits: Shape ``[*, vocab_size]`` or ``[*, local_vocab_size]`` for TP.
            teacher_logits: Same shape as ``student_logits``.
            labels: Shape ``[*]``.  Positions equal to ``ignore_index`` are excluded from the loss.
            num_batch_labels: Total number of valid tokens across all gradient-accumulation steps.
                When provided the loss is ``sum(kl_per_token) / num_batch_labels``; otherwise it
                is ``mean(kl_per_token)`` over the valid tokens in this micro-batch.

        Returns:
            Scalar KD loss.
        """
        # Exclude padding / ignored tokens from the loss.
        valid_mask = (labels != self.ignore_index).view(-1)
        if valid_mask.sum() == 0:
            # Entire batch contains only padding - return zero to keep gradients finite.
            return student_logits.new_tensor(0.0)

        if student_logits.ndim > 2:
            student_logits = student_logits.view(-1, student_logits.shape[-1])
        if teacher_logits.ndim > 2:
            teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])
        if labels.ndim > 1:
            labels = labels.view(-1)

        # Determine TP group: prefer explicit argument, then auto-detect from DTensor.
        tp_group = self.tp_group
        if tp_group is None and _HAVE_DTENSOR and isinstance(student_logits, DTensor):
            tp_group = _infer_tp_group_from_dtensor(student_logits)

        if tp_group is not None:
            # TP path: keep local shards to avoid gathering the full vocabulary.
            if _HAVE_DTENSOR and isinstance(student_logits, DTensor):
                student_logits = student_logits.to_local()
            if _HAVE_DTENSOR and isinstance(teacher_logits, DTensor):
                teacher_logits = teacher_logits.to_local()
        else:
            # Non-TP path: materialise full tensors.
            if _HAVE_DTENSOR and isinstance(student_logits, DTensor):
                student_logits = student_logits.full_tensor()
            if _HAVE_DTENSOR and isinstance(teacher_logits, DTensor):
                teacher_logits = teacher_logits.full_tensor()
            if _HAVE_DTENSOR and isinstance(labels, DTensor):
                labels = labels.full_tensor()

        t_logits = teacher_logits[valid_mask]
        s_logits = student_logits[valid_mask]

        # Up-cast to fp32 for numerical stability and apply temperature scaling.
        if self.fp32_upcast:
            t_logits = t_logits.float()
            s_logits = s_logits.float()

        if self.temperature != 1.0:
            t_logits = t_logits.mul(1.0 / self.temperature)
            s_logits = s_logits.mul(1.0 / self.temperature)

        # Compute per-token negative cross-entropy: sum(P * log Q).
        if tp_group is not None:
            kl_per_token = _kl_forward_tp(t_logits, s_logits, tp_group)
        elif self.chunk_size > 0:
            kl_per_token = _kl_forward_chunked(t_logits, s_logits, self.chunk_size)
        else:
            teacher_prob = F.softmax(t_logits, dim=-1, dtype=torch.float32)
            student_logprob = F.log_softmax(s_logits, dim=-1, dtype=torch.float32)
            inf_mask = torch.isinf(s_logits)
            kl_per_token = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0).sum(-1).view(-1)

        # T² scaling: dividing logits by T scales gradients by 1/T², so we multiply the loss by
        # T² to keep gradient magnitudes independent of temperature (Hinton et al., 2015).
        if self.temperature != 1.0:
            kl_per_token = kl_per_token * (self.temperature**2)

        # kl_per_token = sum(P * log Q) ≤ 0.  Negate to obtain a positive minimisation target.
        if num_batch_labels is not None:
            return -torch.sum(kl_per_token) / num_batch_labels
        else:
            return -torch.mean(kl_per_token)
