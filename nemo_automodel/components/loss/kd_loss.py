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
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.distributed.tensor import DTensor

    _HAVE_DTENSOR = True
except Exception:
    DTensor = None  # type: ignore[assignment, misc]
    _HAVE_DTENSOR = False


def _infer_tp_group_from_dtensor(logits: "torch.Tensor") -> Optional[torch.distributed.ProcessGroup]:
    """If logits are a DTensor sharded on the vocab dim, return its device_mesh process group."""
    if not _HAVE_DTENSOR or not isinstance(logits, DTensor):
        return None
    is_vocab_sharded = any(
        isinstance(p, Shard) and (p.dim == -1 or p.dim == logits.ndim - 1)
        for p in logits.placements
    )
    if not is_vocab_sharded:
        return None
    return logits.device_mesh.get_group()


def _kl_forward_tp(
    t_logits: torch.Tensor,
    s_logits: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute forward Cross Entropy (Softmax) with tensor-parallelism.
    Returns:
        kl_per_token: The negative cross entropy (sum(P * log Q)) per token.
    """
    # 1. Stable Global Softmax for Teacher
    teacher_max, _ = torch.max(t_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(teacher_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    output_teacher = t_logits - teacher_max
    
    denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1, keepdim=True)
    torch.distributed.all_reduce(denom_teacher, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    
    # Calculate Teacher Prob (P)
    # We don't need full log_probs for teacher, just P for the weighting
    teacher_prob = torch.exp(output_teacher - torch.log(denom_teacher.clamp(min=1e-12)))

    # 2. Stable Global Log-Softmax for Student
    student_max, _ = torch.max(s_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(student_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    output_student = s_logits - student_max.detach() # Detach max for safety

    denom_student = torch.sum(torch.exp(output_student), dim=-1, keepdim=True)
    torch.distributed.all_reduce(denom_student, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    
    # Calculate Student Log-Prob (log Q)
    student_log_prob = output_student - torch.log(denom_student.clamp(min=1e-12))

    # 3. Calculate Negative Cross Entropy: Sum(P * log Q)
    # Mask infs in student logits locally before summing
    inf_mask = torch.isinf(s_logits) 
    
    # We calculate the local contribution: P_local * log_Q_local
    # If student logit is -inf, log_prob is -inf. P * -inf is bad. Mask it to 0.
    term = teacher_prob * student_log_prob
    term = torch.masked_fill(term, inf_mask, 0.0)
    
    # Sum over local vocab
    ce_local = term.sum(dim=-1)
    
    # All-reduce to get global sum over vocab
    torch.distributed.all_reduce(ce_local, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    return ce_local.view(-1) # This is sum(P * log Q), which is Negative Cross Entropy


class KDLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        temperature: float = 1.0,
        fp32_upcast: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.fp32_upcast = fp32_upcast
        self.tp_group = tp_group

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        num_batch_labels: int | None = None,
    ) -> torch.Tensor:
        """
        Calculates KL(P_teacher‖P_student) averaged over valid tokens.

        Logits are (optionally) cast to fp32 for numerical stability, probabilities
        are obtained with softmax / log_softmax after temperature scaling, and
        padding tokens (== ignore_index) are ignored in the average.

        Args:
            student_logits (torch.Tensor): The logits of the student model.
            teacher_logits (torch.Tensor): The logits of the teacher model.
            labels (torch.Tensor): The labels of the batch.
            num_batch_labels (int | None): The number of valid labels in the batch.

        Important note on num_batch_labels:
            - if `num_batch_labels` is None, it will return the mean over kl_per_token.
            - if `num_batch_labels` is not None, it will return the sum(kl_per_token) / num_batch_labels.
            Please do note that usually, num_batch_labels > #valid labels in labels tensor, for example,
            when doing gradient accumulation.

            We prefer the num_batch_labels variable over counting the number of valid labels in the batch,
            to allow for easier handling when doing gradient accumulation and per-token loss computation.

        Returns:
            The KL loss.
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

        # 1. Determine TP Group
        tp_group = self.tp_group
        
        # Check if inputs are DTensors and infer group if not provided
        if _HAVE_DTENSOR and isinstance(student_logits, DTensor):
            if tp_group is None:
                tp_group = _infer_tp_group_from_dtensor(student_logits)
        
        # 2. Prepare Logits (Keep local if TP, convert to full otherwise)
        if tp_group is not None:
            # TP Path: Convert to local shards to avoid OOM
            if isinstance(student_logits, DTensor):
                student_logits = student_logits.to_local()
            if isinstance(teacher_logits, DTensor):
                teacher_logits = teacher_logits.to_local()
        else:
            # Non-TP Path: Gather full tensors
            if _HAVE_DTENSOR and isinstance(student_logits, DTensor):
                student_logits = student_logits.full_tensor()
            if _HAVE_DTENSOR and isinstance(teacher_logits, DTensor):
                teacher_logits = teacher_logits.full_tensor()

        # 3. Apply Validity Mask
        # Note: If TP, we are masking the (Valid_Tokens, Local_Vocab) tensor.
        t_logits = teacher_logits[valid_mask]
        s_logits = student_logits[valid_mask]

        # 4. Upcast and Temperature (Standard)
        if self.fp32_upcast:
            t_logits = t_logits.float()
            s_logits = s_logits.float()
            
        if self.temperature != 1.0:
            t_logits = t_logits.mul(1 / self.temperature)
            s_logits = s_logits.mul(1 / self.temperature)

        # 5. Calculate Loss
        if tp_group is not None:
            # TP Path: Returns Negative Cross Entropy
            kl_per_token = _kl_forward_tp(t_logits, s_logits, tp_group)
        else:
            # Non-TP Path: Returns Negative Cross Entropy
            teacher_prob = F.softmax(t_logits, dim=-1, dtype=torch.float32)
            student_logprob = F.log_softmax(s_logits, dim=-1, dtype=torch.float32)
            inf_mask = torch.isinf(s_logits)
            kl_per_token = torch.masked_fill(
                teacher_prob * student_logprob, inf_mask, 0.0
            ).sum(-1).view(-1)

        # 6. Apply T^2 Scaling (Best Practice)
        # Gradient magnitudes scale with 1/T^2, so we multiply by T^2 to normalize
        if self.temperature != 1.0:
            kl_per_token = kl_per_token * (self.temperature ** 2)

        # 7. Return Positive Loss
        # kl_per_token is Negative Cross Entropy.
        # We want to MINIMIZE Cross Entropy.
        # So we return -1 * (Sum / N)
        if num_batch_labels is not None:
            return -torch.sum(kl_per_token) / num_batch_labels
        else:
            return -torch.mean(kl_per_token)
