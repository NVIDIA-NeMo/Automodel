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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel.components.loss import vocab_parallel


def _forward_kl_from_log_probs(
    teacher_log_prob: torch.Tensor,
    student_log_prob: torch.Tensor,
) -> torch.Tensor:
    """Compute forward KL from full-vocabulary log probabilities.

    Args:
        teacher_log_prob: Tensor of shape ``[..., vocab]`` containing teacher
            log probabilities with arbitrary leading dimensions.
        student_log_prob: Tensor of shape ``[..., vocab]`` containing student
            log probabilities on the same dtype and device.

    Returns:
        Tensor of shape ``[...]`` containing per-token
        ``KL(P_teacher || P_student)``.
    """
    teacher_prob = teacher_log_prob.exp()
    log_ratio = teacher_log_prob - student_log_prob
    log_ratio = torch.where(teacher_prob > 0, log_ratio, 0.0)
    return (teacher_prob * log_ratio).sum(dim=-1)


def _vocab_layout(shard: vocab_parallel.VocabShard) -> tuple[tuple[int, ...] | None, int, int]:
    """Return the ranks, first id, and size of a vocabulary slice, comparable across meshes."""
    ranks = None if shard.group is None else tuple(torch.distributed.get_process_group_ranks(shard.group))
    return ranks, shard.start, shard.local.shape[-1]


def _kl_forward_tp(
    t_logits: torch.Tensor,
    s_logits: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Compute per-token forward KL with tensor parallelism.

    Args:
        t_logits: Tensor of shape ``[tokens, local_vocab]`` containing the local
            teacher vocabulary shard.
        s_logits: Tensor of shape ``[tokens, local_vocab]`` containing the local
            student vocabulary shard.
        tp_group: Process group spanning the tensor-parallel ranks.

    Returns:
        Tensor of shape ``[tokens]`` containing per-token forward KL, the same on
        every rank. The local vocabulary axis is a ``Shard(-1)`` across
        ``tp_group`` and is reduced without gathering the full vocabulary. Both
        inputs receive their exact gradients (``vocab_parallel`` convention).
    """
    teacher_log_prob = t_logits - vocab_parallel.logsumexp(t_logits, tp_group).unsqueeze(-1)
    student_log_prob = s_logits - vocab_parallel.logsumexp(s_logits, tp_group).unsqueeze(-1)
    kl_local = _forward_kl_from_log_probs(teacher_log_prob, student_log_prob)
    return vocab_parallel.replicated(vocab_parallel.sum_over_group(kl_local, tp_group), tp_group)


def _kl_forward_chunked(
    t_logits: torch.Tensor,
    s_logits: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute per-token forward KL in chunks to reduce peak memory.

    Processes ``chunk_size`` tokens at a time to bound forward temporaries.
    Autograd retains the intermediates needed for backward across chunks.

    Args:
        t_logits: Tensor of shape ``[tokens, vocab]`` containing teacher logits.
        s_logits: Tensor of shape ``[tokens, vocab]`` containing student logits.
        chunk_size: Number of tokens per chunk.

    Returns:
        Tensor of shape ``[tokens]`` containing per-token forward KL.
    """
    num_tokens = t_logits.shape[0]
    kl_parts: list[torch.Tensor] = []
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        t_chunk = t_logits[start:end]
        s_chunk = s_logits[start:end]
        teacher_logprob = F.log_softmax(t_chunk, dim=-1, dtype=torch.float32)
        student_logprob = F.log_softmax(s_chunk, dim=-1, dtype=torch.float32)
        kl_parts.append(_forward_kl_from_log_probs(teacher_logprob, student_logprob))
    return torch.cat(kl_parts, dim=0)


def _kl_forward_raw_chunk(
    t_logits: torch.Tensor,
    s_logits: torch.Tensor,
    *,
    temperature: float,
    fp32_upcast: bool,
) -> torch.Tensor:
    """Compute a raw-logit chunk's KL with chunk-local preprocessing.

    Args:
        t_logits: Tensor of shape ``[tokens, vocab]`` containing teacher logits.
        s_logits: Tensor of shape ``[tokens, vocab]`` containing student logits.
        temperature: Temperature captured for this forward call.
        fp32_upcast: Whether to cast logits before temperature scaling.

    Returns:
        Tensor of shape ``[tokens]`` containing per-token forward KL.
    """
    if fp32_upcast:
        t_logits = t_logits.float()
    if temperature != 1.0:
        t_logits = t_logits.mul(1.0 / temperature)
    teacher_logprob = F.log_softmax(t_logits, dim=-1, dtype=torch.float32)
    del t_logits
    if fp32_upcast:
        s_logits = s_logits.float()
    if temperature != 1.0:
        s_logits = s_logits.mul(1.0 / temperature)
    student_logprob = F.log_softmax(s_logits, dim=-1, dtype=torch.float32)
    del s_logits
    return _forward_kl_from_log_probs(teacher_logprob, student_logprob)


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
        tp_group: Explicit TP process group, in which case the logits must already be local
            vocabulary shards.  When ``None`` (default) the group is read from vocabulary-sharded
            ``DTensor`` logits; student and teacher logits must then share one vocabulary layout,
            and plain tensors use the non-TP path.  No path gathers the full vocabulary.
        chunk_size: When positive, valid tokens are processed in chunks of this size to avoid
            materializing the full ``[num_valid_tokens, vocab_size]`` probability matrix in fp32.
            With a Python scalar temperature, multi-chunk logits up to 32 bits use chunk-local
            preprocessing. Other inputs retain the existing preprocessing path.
            ``0`` (default) disables chunking.
            Ignored when using the TP path.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        temperature: float = 1.0,
        fp32_upcast: bool = True,
        tp_group: torch.distributed.ProcessGroup | None = None,
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
            student_logits: Tensor of shape ``[..., vocab]`` with arbitrary
                leading dimensions. Under TP, the local tensor has shape
                ``[..., local_vocab]`` with ``Shard(-1)`` placement.
            teacher_logits: Tensor with the same global and local shape contract
                as ``student_logits``.
            labels: Tensor of shape ``[...]`` matching the logits' leading
                dimensions. Positions equal to ``ignore_index`` are excluded.
            num_batch_labels: Total number of valid tokens across all gradient-accumulation steps.
                When provided the loss is ``sum(kl_per_token) / num_batch_labels``; otherwise it
                is ``mean(kl_per_token)`` over the valid tokens in this micro-batch.

        Returns:
            Scalar tensor containing zero-based forward KL.
        """
        if isinstance(labels, DTensor):
            labels = labels.full_tensor()

        # Keep each rank's vocabulary slice. An explicit tp_group means the inputs
        # already hold local shards; otherwise vocabulary-sharded DTensor logits
        # supply the group, and both logits must share one vocabulary layout.
        tp_group = self.tp_group
        if tp_group is None:
            student = vocab_parallel.vocab_shard(student_logits)
            teacher = vocab_parallel.vocab_shard(teacher_logits)
            if _vocab_layout(student) != _vocab_layout(teacher):
                raise ValueError(
                    "KDLoss needs student and teacher logits with one vocabulary layout: both sharded on the "
                    "vocabulary axis over the same group, or both holding the whole vocabulary. Got student ids "
                    f"[{student.start}, {student.start + student.local.shape[-1]}) and teacher ids "
                    f"[{teacher.start}, {teacher.start + teacher.local.shape[-1]}). Slice whole teacher logits to "
                    "the student's shard first (KDMeshBridge.match_student_vocab_shard)."
                )
            tp_group = student.group
            student_logits, teacher_logits = student.local, teacher.local
        else:
            if isinstance(student_logits, DTensor):
                student_logits = student_logits.to_local()
            if isinstance(teacher_logits, DTensor):
                teacher_logits = teacher_logits.to_local()

        # Exclude padding / ignored tokens from the loss.
        valid_mask = (labels != self.ignore_index).view(-1)
        if valid_mask.sum() == 0:
            # Keep a zero student gradient without reducing the ignored logits.
            return student_logits[..., :0].sum()

        # flatten, unlike reshape(-1, local_vocab), also handles an empty vocabulary shard.
        if student_logits.ndim > 2:
            student_logits = student_logits.flatten(0, -2)
        if teacher_logits.ndim > 2:
            teacher_logits = teacher_logits.flatten(0, -2)
        if labels.ndim > 1:
            labels = labels.view(-1)

        t_logits = teacher_logits[valid_mask]
        s_logits = student_logits[valid_mask]

        # Tensor temperatures retain the original gradient-reduction order.
        # Wider logits keep full-token preprocessing to release raw FP64 gathers before chunking.
        if (
            tp_group is None
            and 0 < self.chunk_size < t_logits.shape[0]
            and not isinstance(self.temperature, torch.Tensor)
            and t_logits.element_size() <= 4
            and s_logits.element_size() <= 4
        ):
            kl_parts: list[torch.Tensor] = []
            # Split shares one backward concatenation across all token chunks.
            for t_chunk, s_chunk in zip(t_logits.split(self.chunk_size), s_logits.split(self.chunk_size)):
                kl_parts.append(
                    _kl_forward_raw_chunk(t_chunk, s_chunk, temperature=self.temperature, fp32_upcast=self.fp32_upcast)
                )
            kl_per_token = torch.cat(kl_parts, dim=0)
        else:
            # Up-cast to fp32 for numerical stability and apply temperature scaling.
            if self.fp32_upcast:
                t_logits = t_logits.float()
                s_logits = s_logits.float()

            if self.temperature != 1.0:
                t_logits = t_logits.mul(1.0 / self.temperature)
                s_logits = s_logits.mul(1.0 / self.temperature)

            # Compute per-token forward KL: sum(P * (log P - log Q)).
            if tp_group is not None:
                kl_per_token = _kl_forward_tp(t_logits, s_logits, tp_group)
            elif self.chunk_size > 0:
                kl_per_token = _kl_forward_chunked(t_logits, s_logits, self.chunk_size)
            else:
                teacher_logprob = F.log_softmax(t_logits, dim=-1, dtype=torch.float32)
                student_logprob = F.log_softmax(s_logits, dim=-1, dtype=torch.float32)
                kl_per_token = _forward_kl_from_log_probs(teacher_logprob, student_logprob).view(-1)

        # T² scaling: dividing logits by T scales gradients by 1/T², so we multiply the loss by
        # T² to keep gradient magnitudes independent of temperature (Hinton et al., 2015).
        if self.temperature != 1.0:
            kl_per_token = kl_per_token * (self.temperature**2)

        if num_batch_labels is not None:
            return torch.sum(kl_per_token) / num_batch_labels
        else:
            return torch.mean(kl_per_token)
