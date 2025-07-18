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

from nemo_automodel.components.loss.loss_interface import LossFunction, LossType

class MaskedCrossEntropy(LossFunction):
    loss_type = LossType.TOKEN_LEVEL

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        fp32_upcast: bool = True,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute the masked cross-entropy loss between logits and targets.

        If a mask is provided, the loss is computed per element, multiplied by the mask,
        and then averaged. If no mask is provided, the standard cross-entropy loss is used.

        Args:
            next_token_logits (torch.Tensor): The predicted logits with shape [batch_size, seq_len, vocab_size] where C is the number of classes.
            labels (torch.Tensor): The ground truth class indices with shape [batch_size, seq_len].
            mask (torch.Tensor, optional): A tensor that masks the loss computation. Items marked with
                1 will be used to calculate loss, otherwise ignored. Must be broadcastable to the shape
                of the loss. Defaults to None.
            fp32_upcast (bool, optional): if True it will cast logits to float32 before computing
            cross entropy. Default: True.
            ignore_index (int): label to ignore in CE calculation. Defaults to -100.
            reduction (str): type of reduction. Defaults to "mean".

        Returns:
            torch.Tensor: The computed loss as a scalar tensor.
        """
        # this may happen with CPUOffloadPolicy
        if labels.device != next_token_logits.device:
            labels = labels.to(next_token_logits.device)
        # reshape to (N, C) and (N,) respectively
        next_token_logits = next_token_logits.view(-1, next_token_logits.size(-1))
        labels = labels.view(-1)
        if mask is not None:
            with torch.no_grad():
                if mask.device != labels.device:
                    mask = mask.to(labels.device)
                labels.masked_fill_(mask.view(-1) == 0, ignore_index)
                del mask
        if fp32_upcast:
            next_token_logits = next_token_logits.float()
        return F.cross_entropy(next_token_logits, labels, reduction=reduction)
