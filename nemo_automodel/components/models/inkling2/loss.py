# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Fused softcapped cross entropy over the model's usable vocabulary."""

from dataclasses import replace

import torch
import torch.distributed as dist

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.linear_ce_base import ShardedLMHeadWeight


class Inkling2FusedLinearCrossEntropy(FusedLinearCrossEntropy):
    """Fused projection and cross entropy over the first ``unpadded_vocab_size`` head rows.

    The kernel projects BF16 hidden states against the BF16 head rows with FP32
    accumulation and applies the logit softcap and the cross entropy in FP32,
    which is the model's output contract. Rows past the usable vocabulary never
    enter the softmax. An FP32 head is rounded to BF16, the cast FSDP applies to
    the model's own projection.
    """

    def __init__(
        self,
        unpadded_vocab_size: int,
        *,
        logit_softcapping: float = 30.0,
        ignore_index: int = -100,
        reduction: str = "sum",
    ) -> None:
        if unpadded_vocab_size <= 0:
            raise ValueError("unpadded_vocab_size must be positive")
        if logit_softcapping <= 0:
            raise ValueError("logit_softcapping must be positive")
        super().__init__(ignore_index=ignore_index, logit_softcapping=logit_softcapping, reduction=reduction)
        self.unpadded_vocab_size = unpadded_vocab_size

    def resolve_lm_weight(
        self,
        lm_weight: torch.Tensor | ShardedLMHeadWeight,
        *,
        grad_reduce_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor | ShardedLMHeadWeight:
        """Resolve usable BF16 head rows without gathering a sharded head.

        Args:
            lm_weight: Head weight with global shape [padded_vocab, hidden], a regular
                tensor, an FSDP-sharded DTensor, or an already resolved operand.
            grad_reduce_group: Ranks contributing independent token losses, as in the
                shared resolution contract.

        Returns:
            BF16 tensor or streamed operand of effective global shape
            [unpadded_vocab_size, hidden]. Gradients reach the original weight;
            excluded rows receive zero. A streamed operand retains the physical
            padded shard boundaries and casts only each gathered chunk to BF16.
        """
        weight = super().resolve_lm_weight(lm_weight, grad_reduce_group=grad_reduce_group)
        if weight.shape[0] < self.unpadded_vocab_size:
            raise ValueError(
                f"lm_head has {weight.shape[0]} rows, fewer than unpadded_vocab_size={self.unpadded_vocab_size}"
            )
        if isinstance(weight, ShardedLMHeadWeight):
            return replace(weight, dtype=torch.bfloat16, vocab_size=self.unpadded_vocab_size)
        return weight[: self.unpadded_vocab_size].to(torch.bfloat16)
