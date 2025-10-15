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


class KDLoss(nn.Module):
    def __init__(self, ignore_index: int = -100, temperature: float = 1.0, fp32_upcast: bool = True):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.fp32_upcast = fp32_upcast

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
        t_logits = teacher_logits[valid_mask]
        s_logits = student_logits[valid_mask]
        labels = labels[valid_mask]

        # Up-cast logits to fp32 for numerical stability
        if self.fp32_upcast:
            t_logits = t_logits.float()
            s_logits = s_logits.float()
        #  and apply temperature scaling.
        if self.temperature != 1.0:
            t_logits.mul_(1 / self.temperature)
            s_logits.mul_(1 / self.temperature)

        # Probabilities / log-probabilities
        teacher_prob = F.softmax(t_logits, dim=-1, dtype=torch.float32)
        student_logprob = F.log_softmax(s_logits, dim=-1, dtype=torch.float32)

        # mask out infinities originating *only* from student logits
        # (teacher logits infs are extremely rare and do not
        # affect gradients w.r.t. student parameters).
        inf_mask = torch.isinf(s_logits)

        # Compute per-token forward KL contribution and flatten.
        kl_per_token = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0).sum(-1).view(-1)

        # Average over valid tokens.
        if num_batch_labels is not None:
            return -torch.sum(kl_per_token) / num_batch_labels
        else:
            return -torch.mean(kl_per_token)
