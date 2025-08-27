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
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


class MaskedCrossEntropy:
    def __init__(self, fp32_upcast: bool = True, ignore_index: int = -100, reduction: str = "sum"):
        """
        Masked cross-entropy loss.

        Args:
            fp32_upcast (bool): if True it will cast logits to float32 before computing
                cross entropy. Default: True.
            ignore_index (int): label to ignore in CE calculation. Defaults to -100.
            reduction (str): type of reduction. Defaults to "sum".
        """
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the masked cross-entropy loss between logits and targets.

        If a mask is provided, the loss is computed per element, multiplied by the mask,
        and then averaged. If no mask is provided, the standard cross-entropy loss is used.

        Args:
            logits (torch.Tensor): The predicted logits with shape [batch_size, seq_len, vocab_size] where C is the number of classes.
            labels (torch.Tensor): The ground truth class indices with shape [batch_size, seq_len].
            mask (torch.Tensor, optional): A tensor that masks the loss computation. Items marked with
                1 will be used to calculate loss, otherwise ignored. Must be broadcastable to the shape
                of the loss. Defaults to None.

        Returns:
            torch.Tensor: The computed loss as a scalar tensor.
        """
        # this may happen with CPUOffloadPolicy
        if labels.device != logits.device:
            labels = labels.to(logits.device)
        # reshape to (N, C) and (N,) respectively
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        if mask is not None:
            with torch.no_grad():
                if mask.device != labels.device:
                    mask = mask.to(labels.device)
                labels.masked_fill_(mask.view(-1) == 0, self.ignore_index)
                del mask
        if self.fp32_upcast:
            logits = logits.float()

        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        if isinstance(labels, DTensor):
            labels = labels.full_tensor()

        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        if num_label_tokens is not None:
            assert self.reduction == "sum", "num_label_tokens is only supported when reduction is 'sum'"
            loss = loss / num_label_tokens
        return loss
