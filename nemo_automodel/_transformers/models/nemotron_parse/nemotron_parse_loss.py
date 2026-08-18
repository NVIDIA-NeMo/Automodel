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
from torch.nn import CrossEntropyLoss


class NemotronParseLoss(nn.Module):
    """
    Cross-entropy loss with coordinate token weighting for NemotronParse.

    This loss function computes cross-entropy across prediction heads with configurable
    weighting for coordinate tokens (tokens >= class_token_start_idx). When num_heads > 1,
    it implements per-head label shifting for multi-task output predictions.

    Args:
        coordinate_weight (float): Weight multiplier for coordinate tokens. Tokens with
            label IDs >= class_token_start_idx will have their loss multiplied by this factor.
            Default: 10.0
        class_token_start_idx (int): Token index threshold for coordinate tokens. Tokens
            with label IDs >= this value are considered coordinate/class tokens and receive
            higher loss weight. Default: 50000
        num_heads (int): Number of prediction heads (main + extra). Must match the model's
            num_extra_heads + 1. Default: 1
        ignore_index (int): Label value to ignore in loss computation. Default: -100
        reduction (str): Loss reduction strategy ("sum" or "mean"). Default: "sum"
        fp32_upcast (bool): Cast logits to fp32 for numerical stability. Default: True

    Example:
        >>> loss_fn = NemotronParseLoss(
        ...     coordinate_weight=10.0,
        ...     class_token_start_idx=50000,
        ...     num_heads=1,
        ... )
        >>> # logits shape: [batch, seq_len, vocab_size]
        >>> # labels shape: [batch, seq_len]
        >>> loss = loss_fn(logits=logits, labels=labels)
    """

    def __init__(
        self,
        coordinate_weight: float = 10.0,
        class_token_start_idx: int = 50000,
        num_heads: int = 1,
        ignore_index: int = -100,
        reduction: str = "sum",
        fp32_upcast: bool = True,
    ):
        super().__init__()
        self.coordinate_weight = coordinate_weight
        self.class_token_start_idx = class_token_start_idx
        self.num_heads = num_heads
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.fp32_upcast = fp32_upcast

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute loss with coordinate token weighting.

        Args:
            logits (torch.Tensor): Model logits with shape [batch_size, seq_len, vocab_size]
            labels (torch.Tensor): Ground truth labels with shape [batch_size, seq_len]
            decoder_inputs_embeds (torch.Tensor, optional): Decoder input embeddings.
                Currently unused but kept for API compatibility. Default: None
            num_label_tokens (int, optional): Total number of valid tokens for normalization
                across gradient accumulation steps. If provided, loss is normalized by this
                value instead of the actual token count. Only supported with reduction="sum".
                Default: None

        Returns:
            torch.Tensor: Computed loss value as a scalar tensor.
        """
        if logits.ndim != 3:
            raise ValueError(f"Expected logits shape [batch, seq_len, vocab_size], got {logits.shape}")

        if labels.device != logits.device:
            labels = labels.to(logits.device)

        if self.fp32_upcast:
            logits = logits.float()

        loss_fct = CrossEntropyLoss(reduction="none")
        loss_full = loss_fct(logits.permute(0, 2, 1), labels)

        coordinate_mask = labels >= self.class_token_start_idx
        loss_full[coordinate_mask] *= self.coordinate_weight

        valid_tokens = (labels != self.ignore_index).sum()
        if valid_tokens == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        if num_label_tokens is not None:
            assert self.reduction == "sum", (
                f"num_label_tokens is only supported when reduction='sum', got reduction='{self.reduction}'"
            )
            return loss_full.sum() / (num_label_tokens + 1e-6)

        return loss_full.sum() / (valid_tokens + 1e-6)
